"""The screening queue learns from the reviewers' decisions.

A relevance score measures how close an article is to the question. It knows nothing of
the review's own criteria: a European setting, a kind of study, a topic the question
does not name. On the ISS team's HPAI review, LiteRev's ranking needed 66% of their
1,518 records screened to meet 90% of the 73 papers they included; a classifier trained
on their own decisions needed 35%, and 28% to meet 80%.

So every decision teaches the queue. Once a scenario holds MIN_PER_CLASS included and
MIN_PER_CLASS excluded decisions, a model is trained on them and every article of the
scenario receives a `learned_score`, the probability that the reviewers would include
it. The double-blind queue then follows that score (api/double_blind.py), the panel says
how many relevant articles the model still expects among those left, and the articles
the relevance score left below the threshold are listed when the model would include
them: that is how a paper the question's wording missed gets back in front of a reviewer.

What the model reads: the stored title and abstract embedding of each article
(document_chunk, `title_abstract`) when the decided articles all have one and nearly
every candidate does, else TF-IDF of the title and abstract. The model is a logistic
regression with balanced classes (scikit-learn). Both are cheap: a scenario of 17,000
articles retrains in seconds, in a background thread, after each decision. What decides
(the label of a row, the training, the choice of representation) is pure and tested
without a database; nothing here samples, and every article of the scenario is scored.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             has_abstract_sql, relevant_gate_sql, screening_status_sql)

#: Decisions of each kind before the first model is trained. Below that, the queue keeps
#: the relevance order and the panel says what it waits for.
MIN_PER_CLASS = 3
#: Share of the candidates that must carry an embedding for the model to read embeddings
#: rather than TF-IDF: an article without one would receive no score and sink to the end.
EMBED_COVERAGE_MIN = 0.95
#: TF-IDF vocabulary cap: unigrams and bigrams of titles and abstracts.
TFIDF_MAX_FEATURES = 60_000
#: Below-threshold articles the model would include: listed to the reviewers, this many.
SUGGESTED_LIMIT = 10
#: Cross-validated AUC is reported from this many decisions of each kind: fewer folds
#: would say little.
CV_MIN_PER_CLASS = 10

DECIDED = ("included", "excluded")


def _ensure_learning_tables() -> None:
    """The learned score lives on the link (article_scenarios), next to the two scores the
    queue already reads; the model's state in a small table of its own."""
    from .schema_boot import _exec_ddl_isolated
    _exec_ddl_isolated([
        "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS learned_score DOUBLE PRECISION",
        """CREATE TABLE IF NOT EXISTS scenario_learning (
               scenario_id VARCHAR(80) PRIMARY KEY,
               status      JSONB NOT NULL,
               updated_at  TIMESTAMP DEFAULT NOW()
           )""",
    ], "_ensure_learning_tables")


try:
    _ensure_learning_tables()
except Exception as _e:  # pragma: no cover - a database without the base tables
    logger.warning(f"_ensure_learning_tables: {_e}")


# ── Pure: what a row teaches, which representation, the training ─────────────
def decision_label(screening_status: str | None, reviewer_1_status: str | None,
                   reviewer_2_status: str | None, kappa_final_status: str | None) -> int | None:
    """1 for an article the review includes, 0 for one it excludes, None when nothing has
    been decided. The review's own status wins (a decision by hand, an agreement, an
    arbitration); failing that a single reviewer's vote counts, since a vote is a
    decision in progress; two votes that disagree teach nothing until arbitrated."""
    for s in (screening_status, kappa_final_status):
        if s in DECIDED:
            return 1 if s == "included" else 0
    votes = [v for v in (reviewer_1_status, reviewer_2_status) if v in DECIDED]
    if not votes or len(set(votes)) > 1:
        return None
    return 1 if votes[0] == "included" else 0


def choose_representation(n_labelled: int, n_labelled_embedded: int,
                          n_candidates: int, n_candidates_embedded: int) -> str:
    """"embeddings" when every decided article has one and at least EMBED_COVERAGE_MIN of
    the candidates do, else "tfidf"."""
    if n_labelled and n_labelled_embedded == n_labelled and n_candidates \
            and n_candidates_embedded >= EMBED_COVERAGE_MIN * n_candidates:
        return "embeddings"
    return "tfidf"


def enough_to_train(n_included: int, n_excluded: int) -> bool:
    return n_included >= MIN_PER_CLASS and n_excluded >= MIN_PER_CLASS


def train_scores(x_labelled, y, x_all, *, sparse: bool) -> tuple[list[float], float | None]:
    """The probability of inclusion of every row of `x_all`, from a logistic regression
    fitted on the labelled rows, and the cross-validated AUC when there are enough
    decisions of each kind to measure it (else None). Regularisation differs by
    representation: a sparse TF-IDF needs less of it than a dense embedding."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    model = LogisticRegression(class_weight="balanced", C=4.0 if sparse else 1.0, max_iter=3000)
    auc = None
    n_min = int(min((y == 1).sum(), (y == 0).sum()))
    if n_min >= CV_MIN_PER_CLASS:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        folds = StratifiedKFold(n_splits=min(5, n_min), shuffle=True, random_state=0)
        try:
            auc = float(np.mean(cross_val_score(model, x_labelled, y, cv=folds, scoring="roc_auc")))
        except Exception as e:  # pragma: no cover - a degenerate fold
            logger.warning(f"learned ranking: cross-validation skipped ({e})")
    model.fit(x_labelled, y)
    return [float(p) for p in model.predict_proba(x_all)[:, 1]], auc


def parse_vector(s: str) -> list[float]:
    """A pgvector literal, "[0.1,0.2,...]", as floats."""
    return [float(v) for v in s.strip()[1:-1].split(",") if v]


# ── Reading the scenario ─────────────────────────────────────────────────────
def _candidates(conn, scenario_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(text("""
        SELECT ars.document_id AS id, d.title, d.abstract,
               ars.screening_status, ars.reviewer_1_status, ars.reviewer_2_status,
               ars.kappa_final_status
        FROM article_scenarios ars
        JOIN literature_document d ON d.id = ars.document_id
        WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
        ORDER BY ars.document_id
    """), {"sid": scenario_id}).mappings().all()]


def _embeddings(conn, scenario_id: str) -> dict[int, list[float]]:
    rows = conn.execute(text("""
        SELECT DISTINCT ON (c.document_id) c.document_id, CAST(c.embedding AS text) AS emb
        FROM document_chunk c
        JOIN article_scenarios ars ON ars.document_id = c.document_id AND ars.scenario_id = :sid
        WHERE c.chunk_type = 'title_abstract' AND c.embedding IS NOT NULL
        ORDER BY c.document_id, c.id
    """), {"sid": scenario_id}).all()
    return {int(d): parse_vector(e) for d, e in rows}


def _write_scores(conn, scenario_id: str, ids: list[int], scores: list[float]) -> None:
    conn.execute(text("UPDATE article_scenarios SET learned_score = NULL WHERE scenario_id = :sid"),
                 {"sid": scenario_id})
    if ids:
        conn.execute(text("""
            UPDATE article_scenarios a
            SET learned_score = s.score
            FROM unnest(CAST(:ids AS bigint[]), CAST(:scores AS double precision[])) AS s(document_id, score)
            WHERE a.scenario_id = :sid AND a.document_id = s.document_id
        """), {"sid": scenario_id, "ids": list(ids), "scores": list(scores)})


def _write_status(conn, scenario_id: str, status: dict) -> None:
    conn.execute(text("""
        INSERT INTO scenario_learning (scenario_id, status, updated_at)
        VALUES (:sid, CAST(:status AS jsonb), NOW())
        ON CONFLICT (scenario_id) DO UPDATE SET status = EXCLUDED.status, updated_at = NOW()
    """), {"sid": scenario_id, "status": json.dumps(status)})


def status_row(scenario_id: str, conn=None) -> dict | None:
    """The model's state as last written, or None when nothing was ever trained."""
    sql = text("SELECT status FROM scenario_learning WHERE scenario_id = :sid")
    if conn is not None:
        row = conn.execute(sql, {"sid": scenario_id}).first()
    else:
        with engine.connect() as c:
            row = c.execute(sql, {"sid": scenario_id}).first()
    return dict(row[0]) if row and row[0] else None


# ── Training ─────────────────────────────────────────────────────────────────
def retrain(scenario_id: str) -> dict[str, Any]:
    """Train on the scenario's decisions and score every article of it. Returns the
    state written, which the panel reads. Too few decisions: the scores are cleared, so
    the queue falls back to the relevance order, and the state says what it waits for."""
    t0 = time.perf_counter()
    with engine.connect() as conn:
        cands = _candidates(conn, scenario_id)
        labels = {c["id"]: decision_label(c["screening_status"], c["reviewer_1_status"],
                                          c["reviewer_2_status"], c["kappa_final_status"]) for c in cands}
        labelled = [c for c in cands if labels[c["id"]] is not None]
        n_inc = sum(1 for c in labelled if labels[c["id"]] == 1)
        n_exc = len(labelled) - n_inc
        embs = _embeddings(conn, scenario_id) if enough_to_train(n_inc, n_exc) else {}
    status: dict[str, Any] = {"trained": False, "n_included": n_inc, "n_excluded": n_exc,
                              "n_candidates": len(cands), "min_per_class": MIN_PER_CLASS,
                              "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not enough_to_train(n_inc, n_exc):
        status["reason"] = "too_few_decisions"
        with engine.begin() as conn:
            _write_scores(conn, scenario_id, [], [])
            _write_status(conn, scenario_id, status)
        return status

    rep = choose_representation(len(labelled), sum(1 for c in labelled if c["id"] in embs),
                                len(cands), sum(1 for c in cands if c["id"] in embs))
    if rep == "embeddings":
        import numpy as np
        scored = [c for c in cands if c["id"] in embs]
        x_all = np.asarray([embs[c["id"]] for c in scored], dtype=float)
        x_lab = np.asarray([embs[c["id"]] for c in labelled], dtype=float)
        sparse = False
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        scored = [c for c in cands if ((c["title"] or "") + (c["abstract"] or "")).strip()]
        texts = [f"{c['title'] or ''}. {c['abstract'] or ''}" for c in scored]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english",
                              max_features=TFIDF_MAX_FEATURES)
        x_all = vec.fit_transform(texts)
        pos = {c["id"]: i for i, c in enumerate(scored)}
        labelled = [c for c in labelled if c["id"] in pos]
        x_lab = x_all[[pos[c["id"]] for c in labelled]]
        sparse = True
    y = [labels[c["id"]] for c in labelled]
    scores, auc = train_scores(x_lab, y, x_all, sparse=sparse)
    status.update({"trained": True, "representation": rep, "n_scored": len(scored),
                   "cv_auc": None if auc is None else round(auc, 3),
                   "fit_seconds": round(time.perf_counter() - t0, 2)})
    with engine.begin() as conn:
        _write_scores(conn, scenario_id, [c["id"] for c in scored], scores)
        _write_status(conn, scenario_id, status)
    logger.info(f"learned ranking {scenario_id}: {n_inc} included, {n_exc} excluded, {len(scored)} scored "
                f"({rep}) in {status['fit_seconds']} s")
    return status


# One retraining in flight per scenario; a decision that lands while one runs makes it
# run once more at the end, never N times.
_lock = threading.Lock()
_running: set[str] = set()
_again: set[str] = set()


def _worker(scenario_id: str) -> None:
    while True:
        try:
            retrain(scenario_id)
        except Exception as e:
            logger.warning(f"learned ranking {scenario_id}: {e}")
        with _lock:
            if scenario_id in _again:
                _again.discard(scenario_id)
                continue
            _running.discard(scenario_id)
            return


def schedule_retrain(scenario_id: str) -> bool:
    """Retrain in the background once the caller's transaction is committed. Returns
    whether a thread was started (False: one is running and will run again)."""
    with _lock:
        if scenario_id in _running:
            _again.add(scenario_id)
            return False
        _running.add(scenario_id)
    threading.Thread(target=_worker, args=(scenario_id,), daemon=True).start()
    return True


# ── What the panel reads ─────────────────────────────────────────────────────
def learning_summary(scenario_id: str, conn=None) -> dict[str, Any]:
    """The model's state plus what it says of the work left: the undecided articles of
    the relevant set, the number of them the model expects to be included (the sum of
    their probabilities), and the undecided articles below the threshold the model would
    include, listed first."""
    own = conn is None
    conn = conn or engine.connect()
    try:
        status = status_row(scenario_id, conn) or {"trained": False, "n_included": 0, "n_excluded": 0,
                                                     "min_per_class": MIN_PER_CLASS}
        thr = _get_scenario_threshold(scenario_id)
        undecided = (f"({screening_status_sql('d', 'ars')} IS DISTINCT FROM 'included'"
                     f" AND {screening_status_sql('d', 'ars')} IS DISTINCT FROM 'excluded'"
                     " AND ars.reviewer_1_status IS NULL AND ars.reviewer_2_status IS NULL"
                     " AND ars.kappa_final_status IS NULL)")
        gate = relevant_gate_sql("d", "ars", ":thr")
        # `IS NOT TRUE`, never `NOT (...)`: the gate reads the screening status, NULL for
        # an article nobody judged, and NOT of a NULL counts nothing.
        row = conn.execute(text(f"""
            SELECT COUNT(*) FILTER (WHERE {undecided} AND {gate}) AS remaining_relevant,
                   SUM(ars.learned_score) FILTER (WHERE {undecided} AND {gate}) AS expected_left,
                   COUNT(*) FILTER (WHERE {undecided} AND ({gate}) IS NOT TRUE AND {has_abstract_sql('d')}
                                      AND ars.learned_score >= 0.5) AS below_suggested
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id, "thr": thr}).mappings().first() or {}
        suggested = []
        if status.get("trained"):
            suggested = [dict(r) for r in conn.execute(text(f"""
                SELECT ars.document_id AS id, d.title, d.year, ars.learned_score, ars.similarity_score
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
                  AND {undecided} AND ({gate}) IS NOT TRUE AND {has_abstract_sql('d')}
                  AND ars.learned_score >= 0.5
                ORDER BY ars.learned_score DESC, ars.document_id
                LIMIT :lim
            """), {"sid": scenario_id, "thr": thr, "lim": SUGGESTED_LIMIT}).mappings().all()]
    finally:
        if own:
            conn.close()
    expected = row.get("expected_left")
    return {
        **status,
        "scenario_id": scenario_id,
        "remaining_relevant": int(row.get("remaining_relevant") or 0),
        "expected_relevant_left": None if expected is None else round(float(expected), 1),
        "below_threshold_suggested": int(row.get("below_suggested") or 0),
        "suggested": suggested,
    }


@app.get("/user-scenarios/{scenario_id}/learned-ranking")
def get_learned_ranking(scenario_id: str) -> dict[str, Any]:
    """The state of the model trained on this scenario's decisions, and what it expects."""
    _get_user_scenario_or_404(scenario_id)
    return learning_summary(scenario_id)


@app.post("/user-scenarios/{scenario_id}/learned-ranking/retrain")
def retrain_learned_ranking(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Train now, in the request, and return the state: for a review whose decisions
    were made before this feature existed, or imported."""
    _get_user_scenario_or_404(scenario_id)
    try:
        retrain(scenario_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"learned ranking: {e}") from e
    return learning_summary(scenario_id)
