"""Reviewing the extraction: accept, edit or reject each observation, per reviewer.

The extraction is a model's reading, and the protocol wants two reviewers to check it. A
decision is stored on its own, in `extraction_review`, one row per (article, observation,
reviewer), and NEVER written into the extraction itself. That keeps three things true:

  - the model's output stays as it was made (nothing a reviewer did can be mistaken for it);
  - a re-extraction does not erase the reviewers' work (a decision follows its observation by
    a key, and one whose observation changed is reported as stale, not applied);
  - two reviewers can disagree, and the disagreement is a result: it is counted, and an
    agreement score (Cohen's kappa) is computed between each pair.

An observation's status over all its reviewers is `unreviewed`, `accepted`, `edited`,
`rejected`, or `conflict` when one reviewer keeps it and another rejects it. A conflict is
never resolved silently: the row is neither used nor dropped until a person decides.

A reviewer is a name typed in the interface (the application has one write key, not user
accounts), so the name is a label for attribution and agreement, not an identity check.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .codebook import CODEBOOK_SHEETS, annotate, clean_label, get_index
from .core import app, engine, logger, require_api_key
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)
from .schema_boot import _exec_ddl_isolated

STATUSES = ("accepted", "edited", "rejected")
_NUMBER_FIELDS = ("value", "n_cases", "pop_risk")
_TEXT_FIELDS = {"group": 120, "covariate": 200, "descr": 400, "notes": 400, "disease": 120,
                "transmission_mode": 80}
_REVIEWER_MIN, _REVIEWER_MAX = 2, 40

_REVIEW_DDL = [
    """CREATE TABLE IF NOT EXISTS extraction_review (
        id BIGSERIAL PRIMARY KEY,
        document_id BIGINT NOT NULL,
        obs_key TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        status TEXT NOT NULL,
        edits JSONB,
        note TEXT,
        updated_at TIMESTAMP NOT NULL DEFAULT now(),
        UNIQUE (document_id, obs_key, reviewer)
    )""",
    "CREATE INDEX IF NOT EXISTS extraction_review_doc_idx ON extraction_review (document_id)",
]
try:
    _exec_ddl_isolated(_REVIEW_DDL, "_ensure_extraction_review")
except Exception as _e:                                       # noqa: BLE001 - never blocks startup
    logger.warning(f"_ensure_extraction_review: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pure: keys, combining decisions, edits, agreement
# ─────────────────────────────────────────────────────────────────────────────
def _quote_norm(q: Any) -> str:
    return re.sub(r"\s+", " ", str(q or "").lower().strip())[:200]


def observation_keys(observations: list[dict]) -> list[str]:
    """A stable key per observation: sheet, covariate and quote. The same observation keeps
    its key across a codebook change and across a re-extraction that reads it the same way;
    two identical rows are told apart by a counter."""
    keys, seen = [], {}
    for o in observations:
        base = hashlib.sha1("|".join((str(o.get("sheet") or ""), clean_label(o.get("covariate")),
                                      _quote_norm(o.get("quote")))).encode("utf-8")).hexdigest()[:12]
        seen[base] = seen.get(base, 0) + 1
        keys.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return keys


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(str(v).strip().replace(",", ".")) if not isinstance(v, (int, float)) else float(v)
    except ValueError:
        raise ValueError(f"{v!r} is not a number") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{v!r} is not a number")
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


def clean_edits(raw: Any) -> dict[str, Any]:
    """The fields a reviewer may correct, validated. Numbers are numbers (or null to clear
    one); text is trimmed and bounded; the sheet must be a real one."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("an edit needs at least one field")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in _NUMBER_FIELDS:
            out[k] = _num(v) if v not in ("", None) else None
        elif k in _TEXT_FIELDS:
            out[k] = (re.sub(r"\s+", " ", str(v)).strip()[:_TEXT_FIELDS[k]] or None) if v is not None else None
        elif k == "sheet":
            if v not in CODEBOOK_SHEETS:
                raise ValueError(f"unknown sheet {v!r}")
            out[k] = v
        else:
            raise ValueError(f"field {k!r} cannot be edited")
    if not out:
        raise ValueError("an edit needs at least one field")
    return out


def clean_reviewer(raw: Any) -> str:
    name = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not (_REVIEWER_MIN <= len(name) <= _REVIEWER_MAX):
        raise ValueError(f"the reviewer name must be {_REVIEWER_MIN} to {_REVIEWER_MAX} characters")
    return name


def combine(reviews: list[dict]) -> dict[str, Any]:
    """The status of an observation over all its reviewers, and the edits to apply.

    A conflict is one reviewer keeping it (accepted or edited) and another rejecting it. The
    edits applied are the most recent `edited` decision's."""
    if not reviews:
        return {"status": "unreviewed", "conflict": False, "reviewers": [], "edits": None}
    keep = [r for r in reviews if r["status"] in ("accepted", "edited")]
    rej = [r for r in reviews if r["status"] == "rejected"]
    edited = sorted((r for r in reviews if r["status"] == "edited"), key=lambda r: str(r.get("updated_at")))
    if keep and rej:
        status = "conflict"
    elif rej:
        status = "rejected"
    elif edited:
        status = "edited"
    else:
        status = "accepted"
    return {"status": status, "conflict": status == "conflict",
            "reviewers": sorted({r["reviewer"] for r in reviews}),
            "edits": (edited[-1].get("edits") if edited else None)}


def apply_edits(o: dict[str, Any], edits: dict[str, Any] | None) -> dict[str, Any]:
    """The observation with a reviewer's corrections applied. The original is not changed."""
    return {**o, **(edits or {})}


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Agreement between two reviewers on keep (True) or reject (False), over the
    observations both reviewed. `kappa` is None when it is undefined (no pair, or both
    reviewers gave one answer throughout, so there is no disagreement to measure)."""
    n = len(pairs)
    if n == 0:
        return {"n_common": 0, "observed": None, "kappa": None}
    a = sum(1 for x, y in pairs if x == y) / n
    p1 = sum(1 for x, _ in pairs if x) / n
    p2 = sum(1 for _, y in pairs if y) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = None if pe >= 1 else round((a - pe) / (1 - pe), 3)
    return {"n_common": n, "observed": round(a, 3), "kappa": kappa}


def agreement(by_obs: dict[Any, list[dict]]) -> list[dict[str, Any]]:
    """Cohen's kappa for every pair of reviewers, from {observation: [their decisions]}."""
    per: dict[str, dict[Any, bool]] = {}
    for obs, rs in by_obs.items():
        for r in rs:
            per.setdefault(r["reviewer"], {})[obs] = r["status"] in ("accepted", "edited")
    out = []
    for a, b in combinations(sorted(per), 2):
        common = per[a].keys() & per[b].keys()
        out.append({"reviewers": [a, b], **cohen_kappa([(per[a][o], per[b][o]) for o in common])})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reading and the overlay
# ─────────────────────────────────────────────────────────────────────────────
def load_reviews(doc_ids: list[int]) -> dict[tuple[int, str], list[dict]]:
    if not doc_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT document_id, obs_key, reviewer, status, edits, note, updated_at
            FROM extraction_review WHERE document_id = ANY(:ids)
        """), {"ids": list(doc_ids)}).mappings().all()
    out: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        d = dict(r)
        d["updated_at"] = d["updated_at"].isoformat(timespec="seconds") if d["updated_at"] else None
        out.setdefault((int(r["document_id"]), r["obs_key"]), []).append(d)
    return out


def overlay(doc_id: int, observations: list[dict], reviews: dict | None = None) -> tuple[list[dict], list[dict]]:
    """The observations with their review state, and the stale decisions (a reviewer's
    decision whose observation no longer exists in the extraction).

    Each observation gains `obs_key`, `review_status`, `reviews` and `effective`, the fields
    after the reviewers' edits. The stored fields are left as the model wrote them."""
    if reviews is None:
        reviews = load_reviews([doc_id])
    keys = observation_keys(observations)
    out = []
    for o, k in zip(observations, keys):
        rs = reviews.get((doc_id, k), [])
        c = combine(rs)
        out.append({**o, "obs_key": k, "review_status": c["status"],
                    "reviews": [{"reviewer": r["reviewer"], "status": r["status"], "edits": r.get("edits"),
                                 "note": r.get("note"), "updated_at": r.get("updated_at")} for r in rs],
                    "effective": apply_edits(o, c["edits"])})
    live = set(keys)
    stale = [{"obs_key": k, **{kk: r.get(kk) for kk in ("reviewer", "status", "edits", "note")}}
             for (d, k), rs in reviews.items() if d == doc_id and k not in live for r in rs]
    return out, stale


def overlay_annotated(doc_id: int, stored: list[dict], index, reviews: dict | None = None) -> tuple[list[dict], list[dict]]:
    """`overlay`, then each observation and its corrected (`effective`) values read through the
    codebook. The key comes from the STORED fields; the codebook position of the corrected values
    is recomputed, since a reviewer's edit may move a label."""
    obs, stale = overlay(doc_id, stored, reviews)
    for o, raw in zip(obs, stored):
        o.update({k: v for k, v in annotate(index, raw).items() if k not in o})
        o["effective"] = annotate(index, o["effective"])
    return obs, stale


def _observations_of(conn, scenario_id: str, article_id: int) -> list[dict] | None:
    row = conn.execute(text("""
        SELECT d.extraction_json FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid AND d.id = :id
    """), {"sid": scenario_id, "id": article_id}).mappings().first()
    if row is None:
        return None
    ex = row["extraction_json"]
    obs = ex.get("observations") if isinstance(ex, dict) else None
    return [o for o in obs if isinstance(o, dict)] if isinstance(obs, list) else []


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/extraction/review")
def review_observation(scenario_id: str, article_id: int, payload: dict[str, Any],
                       _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Accept, edit or reject one observation as `reviewer`, or clear that reviewer's
    decision with status `clear`. An edit carries the corrected fields."""
    _get_user_scenario_or_404(scenario_id)
    try:
        reviewer = clean_reviewer(payload.get("reviewer"))
        status = str(payload.get("status") or "")
        if status not in STATUSES + ("clear",):
            raise ValueError(f"status must be one of {', '.join(STATUSES)} or clear")
        edits = clean_edits(payload.get("edits")) if status == "edited" else None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    note = (str(payload.get("note") or "").strip()[:500] or None)
    key = str(payload.get("obs_key") or "")
    with engine.connect() as conn:
        obs = _observations_of(conn, scenario_id, article_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Article not in this scenario.")
    if key not in observation_keys(obs):
        raise HTTPException(status_code=404, detail="No such observation in this extraction.")
    with engine.begin() as conn:
        if status == "clear":
            conn.execute(text("DELETE FROM extraction_review WHERE document_id = :d AND obs_key = :k AND reviewer = :r"),
                         {"d": article_id, "k": key, "r": reviewer})
        else:
            conn.execute(text("""
                INSERT INTO extraction_review (document_id, obs_key, reviewer, status, edits, note, updated_at)
                VALUES (:d, :k, :r, :s, CAST(:e AS jsonb), :n, now())
                ON CONFLICT (document_id, obs_key, reviewer) DO UPDATE
                SET status = :s, edits = CAST(:e AS jsonb), note = :n, updated_at = now()
            """), {"d": article_id, "k": key, "r": reviewer, "s": status,
                   "e": json.dumps(edits, ensure_ascii=False) if edits else None, "n": note})
    ov, _stale = overlay_annotated(article_id, obs, get_index(scenario_id))
    return {"observation": next(o for o in ov if o["obs_key"] == key)}


@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/extraction/review/bulk")
def review_bulk(scenario_id: str, article_id: int, payload: dict[str, Any],
                _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Accept (or reject) many observations of one article at once: the listed `obs_keys`, or
    every observation whose quote was found in the text (`verified_only`). Observations this
    reviewer has already decided are left alone."""
    _get_user_scenario_or_404(scenario_id)
    try:
        reviewer = clean_reviewer(payload.get("reviewer"))
        status = str(payload.get("status") or "")
        if status not in ("accepted", "rejected"):
            raise ValueError("bulk status must be accepted or rejected")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    with engine.connect() as conn:
        obs = _observations_of(conn, scenario_id, article_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Article not in this scenario.")
    keys = observation_keys(obs)
    wanted = payload.get("obs_keys")
    if isinstance(wanted, list):
        chosen = [k for k in keys if k in set(map(str, wanted))]
    elif payload.get("verified_only"):
        chosen = [k for o, k in zip(obs, keys) if o.get("quote_verified") is True]
    else:
        raise HTTPException(status_code=422, detail="give obs_keys or verified_only")
    reviews = load_reviews([article_id])
    done = 0
    with engine.begin() as conn:
        for k in chosen:
            if any(r["reviewer"] == reviewer for r in reviews.get((article_id, k), [])):
                continue
            conn.execute(text("""
                INSERT INTO extraction_review (document_id, obs_key, reviewer, status, updated_at)
                VALUES (:d, :k, :r, :s, now())
                ON CONFLICT (document_id, obs_key, reviewer) DO NOTHING
            """), {"d": article_id, "k": k, "r": reviewer, "s": status})
            done += 1
    return {"article_id": article_id, "reviewer": reviewer, "status": status, "n_recorded": done,
            "n_skipped": len(chosen) - done}


def review_summary(scenario_id: str) -> dict[str, Any]:
    """The review state over ALL the relevant articles' observations: how many are in each
    status, who reviewed what, and the agreement between each pair of reviewers."""
    thr = _get_scenario_threshold(scenario_id)
    gate = relevant_gate_sql("d", "ars", ":thr")
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT d.id, d.extraction_json->'observations' AS obs
            FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate} AND jsonb_typeof(d.extraction_json->'observations') = 'array'
        """), {"sid": scenario_id, "thr": thr}).mappings().all()
    docs = {int(r["id"]): [o for o in (r["obs"] or []) if isinstance(o, dict)] for r in rows}
    reviews = load_reviews(list(docs))
    counts = {"unreviewed": 0, "accepted": 0, "edited": 0, "rejected": 0, "conflict": 0}
    by_reviewer: dict[str, int] = {}
    by_obs: dict[Any, list[dict]] = {}
    n_obs = n_stale = 0
    for did, obs in docs.items():
        keys = observation_keys(obs)
        n_obs += len(keys)
        for k in keys:
            rs = reviews.get((did, k), [])
            counts[combine(rs)["status"]] += 1
            for r in rs:
                by_reviewer[r["reviewer"]] = by_reviewer.get(r["reviewer"], 0) + 1
            if rs:
                by_obs[(did, k)] = rs
        live = set(keys)
        n_stale += sum(1 for (d, k) in reviews if d == did and k not in live)
    reviewed = n_obs - counts["unreviewed"]
    return {"scenario_id": scenario_id, "n_observations": n_obs, "counts": counts,
            "n_reviewed": reviewed, "share_reviewed": round(reviewed / n_obs, 3) if n_obs else 0.0,
            "reviewers": [{"reviewer": r, "n_decisions": n} for r, n in sorted(by_reviewer.items(), key=lambda kv: -kv[1])],
            "agreement": agreement(by_obs), "n_stale_decisions": n_stale,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/user-scenarios/{scenario_id}/extraction/review/summary")
def get_review_summary(scenario_id: str) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    return review_summary(scenario_id)
