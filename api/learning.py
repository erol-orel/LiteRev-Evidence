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

from fastapi import Depends, HTTPException, Response
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
#: The card on screen shows five of them and the count said ninety-seven, so there was no
#: way to see the rest: a caller asks for more with `limit`, and the export takes them all.
SUGGESTED_LIMIT = 10
SUGGESTED_LIMIT_MAX = 2000
#: A below-threshold article is a suggestion from this score up. One number, used by the
#: list that counts them and by the screening queue that now carries them.
SUGGEST_MIN_SCORE = 0.5
#: How many suggestions the screening queue admits beside the relevant subset. A margin on
#: the review, not a second review: the model offers 97 suggestions on one HPAI corpus and
#: 1,442 on another, and a queue of 1,442 articles the search did not retain is a different
#: piece of work, which a reviewer must choose rather than inherit. The rest stay in the
#: export, which takes them all.
SUGGESTED_IN_QUEUE = 200
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
def learning_summary(scenario_id: str, conn=None, suggested_limit: int | None = None) -> dict[str, Any]:
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
                   -- Les deux côtés du seuil comptés de la même manière, ce qui manquait :
                   -- on savait combien d'articles SOUS le seuil le modèle garderait, et
                   -- rien de ce qu'il pense des articles AU-DESSUS, où se fait le travail.
                   COUNT(*) FILTER (WHERE {undecided} AND {gate}
                                      AND ars.learned_score >= :smin) AS above_suggested,
                   COUNT(*) FILTER (WHERE {undecided} AND {gate}
                                      AND ars.learned_score < :smin) AS above_rejected,
                   COUNT(*) FILTER (WHERE {undecided} AND ({gate}) IS NOT TRUE AND {has_abstract_sql('d')}
                                      AND ars.learned_score >= :smin) AS below_suggested
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id, "thr": thr, "smin": SUGGEST_MIN_SCORE}).mappings().first() or {}
        suggested = []
        if status.get("trained"):
            suggested = [dict(r) for r in conn.execute(text(f"""
                SELECT ars.document_id AS id, d.title, d.year, ars.learned_score, ars.similarity_score
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
                  AND {undecided} AND ({gate}) IS NOT TRUE AND {has_abstract_sql('d')}
                  AND ars.learned_score >= :smin
                ORDER BY ars.learned_score DESC, ars.document_id
                LIMIT :lim
            """), {"sid": scenario_id, "thr": thr, "smin": SUGGEST_MIN_SCORE,
                   "lim": max(1, min(int(suggested_limit or SUGGESTED_LIMIT), SUGGESTED_LIMIT_MAX))}
            ).mappings().all()]
    finally:
        if own:
            conn.close()
    expected = row.get("expected_left")
    return {
        **status,
        "scenario_id": scenario_id,
        "remaining_relevant": int(row.get("remaining_relevant") or 0),
        "expected_relevant_left": None if expected is None else round(float(expected), 1),
        # Au-dessus du seuil, ce que le modèle garderait et ce qu'il écarterait : un
        # relecteur qui a déjà jugé quelques centaines d'articles veut savoir ce qui
        # l'attend dans la file, pas seulement ce que le seuil a laissé dehors.
        "above_threshold_suggested": int(row.get("above_suggested") or 0),
        "above_threshold_rejected": int(row.get("above_rejected") or 0),
        "below_threshold_suggested": int(row.get("below_suggested") or 0),
        "suggested": suggested,
    }


@app.get("/user-scenarios/{scenario_id}/learned-ranking")
def get_learned_ranking(scenario_id: str, limit: int | None = None) -> dict[str, Any]:
    """The state of the model trained on this scenario's decisions, and what it expects.

    `limit` asks for more of the below-threshold suggestions than the card shows. The
    count has always been there; the list stopped at ten, so the other suggestions could
    be counted and never read."""
    _get_user_scenario_or_404(scenario_id)
    return learning_summary(scenario_id, suggested_limit=limit)


def learned_ranking_rows(scenario_id: str, conn: Any = None) -> list[dict[str, Any]]:
    """Every article left to screen, in the model's order, both sides of the threshold.

    The export used to carry the below-threshold suggestions alone, which is the smaller
    half of the question: a reviewer works through the articles above the threshold too,
    and the model has an opinion on each of them. One ranking, with the side of the
    threshold written on every row, so the file answers "what should I read next" without
    the reader having to join two lists."""
    own = conn is None
    conn = conn or engine.connect()
    try:
        thr = _get_scenario_threshold(scenario_id)
        undecided = (f"({screening_status_sql('d', 'ars')} IS DISTINCT FROM 'included'"
                     f" AND {screening_status_sql('d', 'ars')} IS DISTINCT FROM 'excluded'"
                     " AND ars.reviewer_1_status IS NULL AND ars.reviewer_2_status IS NULL"
                     " AND ars.kappa_final_status IS NULL)")
        gate = relevant_gate_sql("d", "ars", ":thr")
        return [dict(r) for r in conn.execute(text(f"""
            SELECT ars.document_id AS id, d.title, d.year, d.journal, d.doi,
                   ars.learned_score, ars.similarity_score, ars.rerank_score,
                   ({gate}) AS above_threshold
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND {undecided} AND ars.learned_score IS NOT NULL
            ORDER BY ars.learned_score DESC, ars.document_id
        """), {"sid": scenario_id, "thr": thr}).mappings().all()]
    finally:
        if own:
            conn.close()


@app.get("/user-scenarios/{scenario_id}/learned-ranking/export")
def export_learned_suggestions(scenario_id: str) -> Response:
    """Everything left to screen, in the model's order, as a CSV.

    Ordered as the model ranks them, with the two scores beside each other so a reviewer
    sees what the similarity order missed, and the side of the threshold on every row.
    Opened in Excel, hence the BOM, as the other exports of this project."""
    import csv
    import io
    row = _get_user_scenario_or_404(scenario_id)
    state = learning_summary(scenario_id, suggested_limit=1)
    rows = learned_ranking_rows(scenario_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"# Scenario: {row['name']} ({scenario_id})"])
    w.writerow([f"# The {len(rows)} articles left to screen, in the order of a model trained on"
                f" {state.get('n_included', 0)} included and {state.get('n_excluded', 0)}"
                f" excluded decisions. Cross-validated AUC {state.get('cv_auc')}."
                f" learned_score is the probability that this review would include the article;"
                f" above_threshold says which side of the relevance threshold it sits on."])
    w.writerow(["rank", "id", "title", "year", "journal", "doi", "learned_score",
                "similarity_score", "rerank_score", "above_threshold", "url"])
    for i, s in enumerate(rows, start=1):
        w.writerow([i, s.get("id"), s.get("title"), s.get("year"), s.get("journal"), s.get("doi"),
                    f"{float(s['learned_score']):.4f}" if s.get("learned_score") is not None else "",
                    f"{float(s['similarity_score']):.4f}" if s.get("similarity_score") is not None else "",
                    f"{float(s['rerank_score']):.4f}" if s.get("rerank_score") is not None else "",
                    "yes" if s.get("above_threshold") else "no",
                    f"https://literev-scenario.com/scenarios/{scenario_id}?article={s.get('id')}"])
    return Response(content=buf.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="learned_ranking_{scenario_id}.csv"'})


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
