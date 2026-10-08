"""Structured extraction in the shape of the T2.4 / T4.5 data extraction template.

One row per observation (a covariate of a study population), the way the reviewers' Excel
template wants it: a REF row per paper, then one long-format sheet per covariate group
(human susceptibility, human exposure, environment, animal or reservoir, vector). Every
observation carries the exact quote it comes from, the page or section, whether it came
from a table, a figure or the text, and whether the quote was found in the text that the
model was given. A paper that does not report something says so ("coverage" flags) instead
of staying silent.

House rule (CLAUDE.md): every extraction draws on ALL the relevant articles. This is the
MAP half: per-article facts are extracted once and cached on the article row
(`extraction_json`), so the cost is one-time and incremental; a paper is re-read only when
the full text arrives after an abstract-only pass, or when the extraction version moves.
The REDUCE half (counts and coverage over the whole relevant subset, in SQL) is the next
step. `EXTRACTION_MAX_ARTICLES` defaults to zero, meaning no limit; a positive value is an
operational fallback for a day when the LLM budget must be held.

The per-article TEXT is capped (`EXTRACTION_MAX_CHARS`): that is the size of one prompt,
not a sample of the corpus, and the extraction records when it had to cut.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import text

from .core import _env_int, app, engine, logger, require_api_key
from .digest import _rows
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)
from .schema_boot import _exec_ddl_isolated
from llm_usage import json_content as _json_content
from llm_usage import model_for as _model

#: Bump to make every cached extraction eligible for a new pass.
EXTRACTION_VERSION = 1
EXTRACTION_MAX_ARTICLES = _env_int("EXTRACTION_MAX_ARTICLES", 0, 0)
EXTRACTION_MAX_CHARS = _env_int("EXTRACTION_MAX_CHARS", 60000, 4000)
_EXTRACTION_WORKERS = _env_int("EXTRACTION_WORKERS", 4, 1)
#: Failed attempts after which an article is left alone (a deterministic failure would
#: otherwise be paid for again at every run). A success resets the counter.
_MAX_ATTEMPTS = 3
_MAX_OBSERVATIONS = 120
_MIN_ABSTRACT_CHARS = 80

SHEETS = ("human_susc", "human_exp", "env", "animal", "vector")
SOURCE_KINDS = ("table", "figure", "text")
#: What a paper can report on, true only when it gives data on it. A False is information:
#: "this paper has no sex-disaggregated data" is a result for the gap map, not a blank.
COVERAGE_KEYS = ("sex_gender", "age", "occupation", "kap_risk_perception", "ppe",
                 "vaccination", "human_testing", "animal_host", "environment", "vector")

# ── The template, verbatim ───────────────────────────────────────────────────
# Column titles exactly as the reviewers' workbook spells them (trailing and doubled
# spaces included), so that what we export pastes into their file without remapping.
REF_COLUMNS = (
    ("ID", "id"), ("first author name", "first_author"), ("reference", "reference"),
    ("year of publication", "year"), ("brief study description", "description"),
    ("Article type", "article_type"),
    ("Starting date of the study (DD/MM/YYYY)", "study_start"),
    ("Ending date of the study (DD/MM/YYYY)", "study_end"),
    ("geographic location and area", "location"),
    ("study area NUTS level 1 ", "nuts1"), ("study area NUTS level 2 ", "nuts2"),
    ("study area NUTS level 3", "nuts3"),
    ("Notes (space and time additional explanation)", "notes_geo"),
    ("Number of exposed (population at risk)", "risk_pop"),
    ("Number  of positive (cases)", "positive"),
    ("Proportion % of positive (cases)", "percent_positive"),
    ("Mathematical model (Y/N)", "math_model"), ("Model type", "model_type"),
    ("Exclusion", "exclusion"),
)
_HUM = (("ID", "id"), ("TRANSMISSION MODE", "transmission_mode"), ("DISEASE", "disease"),
        ("COV_HUM_GROUP", "group"), ("COVARIATE_hum", "covariate"), ("Value_hum", "value"),
        ("descr_hum", "descr"), ("notes_hum", "notes"), ("N_cases_hum", "n_cases"),
        ("Pop_risk_Hum", "pop_risk"), ("Variable original name", "original_name"),
        ("Page/section", "page_section"), ("Extraction from Table/Figure/text", "source_kind"))
SHEET_COLUMNS = {
    "human_susc": ("HUMAN_COV_SUSC", _HUM),
    "human_exp": ("HUMAN_COV_EXP", _HUM),
    "env": ("ENV_COV", (
        ("ID", "id"), ("TRANSMISSION MODE", "transmission_mode"), ("DISEASE", "disease"),
        ("COVARIATE_env", "covariate"), ("Value_env", "value"), ("descr_env", "descr"),
        ("notes_env", "notes"), ("N_cases_env", "n_cases"), ("Pop_risk_env", "pop_risk"),
        ("Extraction from Table/Figure/text", "source_kind"))),
    "animal": ("ANIMALorRESERVOIR_COV", (
        ("ID", "id"), ("TRANSMISSION MODE", "transmission_mode"), ("DISEASE", "disease"),
        ("COV_ANIM_GROUP", "group"), ("COVARIATE_res", "covariate"), ("Value_res", "value"),
        ("descr_res", "descr"), ("notes_res", "notes"), ("N_cases_res", "n_cases"),
        ("Pop_risk_res", "pop_risk"), ("Extraction from Table/Figure/text", "source_kind"))),
    "vector": ("VECTOR_COV", (
        ("ID", "id"), ("TRANSMISSION MODE", "transmission_mode"), ("DISEASE", "disease"),
        ("COV_VEC_GROUP", "group"), ("COVARIATE_vec", "covariate"), ("Value_vec", "value"),
        ("descr_vec", "descr"), ("notes_vec", "notes"), ("N_cases_vec", "n_cases"),
        ("Pop_risk_vec", "pop_risk"), ("Extraction from Table/Figure/text", "source_kind"))),
}
#: Added to the right of the template columns, so the template stays intact and a reviewer
#: can still check each value against the paper.
REVIEW_COLUMNS = (("Page/section", "page_section"), ("Quote", "quote"),
                  ("Quote found in text", "quote_verified"), ("Extracted from", "source"))


def _ensure_extraction_columns() -> None:
    stmts = [
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS extraction_json JSONB",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS extraction_at TIMESTAMP",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS extraction_attempts INTEGER DEFAULT 0",
    ]
    _exec_ddl_isolated(stmts, "_ensure_extraction_columns")


try:
    _ensure_extraction_columns()
except Exception as _e:                                       # noqa: BLE001 - never blocks startup
    logger.warning(f"_ensure_extraction_columns: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# The prompt
# ─────────────────────────────────────────────────────────────────────────────
_EXTRACTION_SYSTEM = (
    "You extract structured data from ONE scientific paper for a systematic review on "
    "emerging infectious diseases (One Health: humans, animals, environment, vectors). "
    "Report ONLY what the text states: never infer, never estimate, never carry a value over "
    "from another paper or another disease. If the paper does not report something, leave it "
    "out. Return ONLY JSON of this shape:\n"
    '{"ref": {"description": <one or two sentences: study type, population, setting>, '
    '"article_type": <research | short communication | outbreak report | review | other>, '
    '"study_start": <string as written or null>, "study_end": <string as written or null>, '
    '"location": <place name as written, or null>, "notes_geo": <string or null>, '
    '"risk_pop": <number or null>, "positive": <number or null>, '
    '"percent_positive": <number or null>, "math_model": <true|false>, "model_type": '
    '<e.g. SEIR, or null>},\n'
    ' "coverage": {"sex_gender": <bool>, "age": <bool>, "occupation": <bool>, '
    '"kap_risk_perception": <bool>, "ppe": <bool>, "vaccination": <bool>, "human_testing": '
    '<bool>, "animal_host": <bool>, "environment": <bool>, "vector": <bool>},\n'
    ' "observations": [{"sheet": <one of the five below>, "transmission_mode": <string or '
    'null>, "disease": <string>, "group": <string>, "covariate": <string>, "value": '
    '<number or null>, "descr": <string>, "notes": <string or null>, "n_cases": <number or '
    'null>, "pop_risk": <number or null>, "original_name": <the label used in the paper>, '
    '"page_section": <section name, table or figure number, or page>, "source_kind": '
    '<table | figure | text>, "quote": <verbatim excerpt of the text>}]}\n\n'
    "SHEETS. human_susc: host factors that change susceptibility or outcome (sex, gender, "
    "age, comorbidity, immune or vaccination status as a host factor). human_exp: factors "
    "that change exposure (occupation, type of contact, behaviour, knowledge, attitudes and "
    "practices, risk perception, protective equipment use). env: environmental covariates "
    "(temperature, humidity, biosecurity level, setting, wild-bird contact, persistence of "
    "the pathogen). animal: animal hosts and reservoirs (species, number tested, positive, "
    "serology, clinical status). vector: vectors (species, density, infection rate).\n\n"
    "RULES. One row per covariate per group: a sex-disaggregated result gives one row for "
    "males and one for females, with the group 'sex' and the covariate 'male' or 'female'. "
    "'value' is a single number as reported (proportion, odds ratio, mean, count); put the "
    "unit, the confidence interval and the statistic name in 'descr'. 'n_cases' is the number "
    "of cases or positives in that group and 'pop_risk' the number of people or animals in it; "
    "give them only when the paper does. 'quote' must be copied exactly from the text, at most "
    "300 characters, and must contain the figures you report. Scientific names in Latin for "
    "animals and vectors. At most " + str(_MAX_OBSERVATIONS) + " observations: keep the most "
    "informative, and always keep every sex, gender, age, occupation, knowledge-attitude-"
    "practice and protective-equipment result. 'coverage' flags are true only when the paper "
    "gives data on that item, false otherwise. Never write the em dash character; use a comma "
    "or a hyphen. Return ONLY the JSON."
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing and checking
# ─────────────────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def _to_number(v: Any) -> int | float | None:
    """A number from what the model returned, or None. Integral values stay integers."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str) and _NUM_RE.match(v.strip().replace(" ", "")):
        f = float(v.strip().replace(" ", "").replace(",", "."))
    else:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


def _clean_str(v: Any, limit: int) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s[:limit] if s else None


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", (s or "").lower()).strip()


def quote_in_text(quote: str | None, source_text: str) -> bool:
    """Whether the model's quote is really in the text it was given.

    Whitespace, case and punctuation are ignored (PDF text breaks lines inside a sentence),
    but the words and figures must be there in the same order. This is the cheap check
    that catches an invented excerpt; it does not prove the value is the right reading of it."""
    q, t = _norm_text(quote or ""), _norm_text(source_text)
    return bool(q) and q in t


def parse_extraction(raw: Any, source_text: str) -> dict[str, Any]:
    """The model's JSON as a clean, bounded record. Pure.

    Rows with an unknown sheet or no covariate are dropped. A value that is not a number
    is kept as text in `descr` rather than lost. `quote_verified` says whether the quote
    was found in `source_text`; an unverified row is KEPT and flagged, so the reviewer sees
    it instead of the model's hallucination being hidden by a silent filter."""
    if not isinstance(raw, dict):
        raise ValueError("extraction is not a JSON object")
    ref_in = raw.get("ref") if isinstance(raw.get("ref"), dict) else {}
    ref = {
        "description": _clean_str(ref_in.get("description"), 500),
        "article_type": _clean_str(ref_in.get("article_type"), 60),
        "study_start": _clean_str(ref_in.get("study_start"), 40),
        "study_end": _clean_str(ref_in.get("study_end"), 40),
        "location": _clean_str(ref_in.get("location"), 200),
        "notes_geo": _clean_str(ref_in.get("notes_geo"), 400),
        "risk_pop": _to_number(ref_in.get("risk_pop")),
        "positive": _to_number(ref_in.get("positive")),
        "percent_positive": _to_number(ref_in.get("percent_positive")),
        "math_model": bool(ref_in.get("math_model")) if isinstance(ref_in.get("math_model"), bool) else False,
        "model_type": _clean_str(ref_in.get("model_type"), 80),
    }
    cov_in = raw.get("coverage") if isinstance(raw.get("coverage"), dict) else {}
    coverage = {k: cov_in.get(k) is True for k in COVERAGE_KEYS}

    observations: list[dict[str, Any]] = []
    for o in (raw.get("observations") or []) if isinstance(raw.get("observations"), list) else []:
        if not isinstance(o, dict):
            continue
        sheet = _clean_str(o.get("sheet"), 20)
        covariate = _clean_str(o.get("covariate"), 200)
        if sheet not in SHEETS or not covariate:
            continue
        descr = _clean_str(o.get("descr"), 400)
        value = _to_number(o.get("value"))
        if value is None and o.get("value") not in (None, ""):
            as_text = _clean_str(o.get("value"), 120)
            if as_text:
                descr = f"{as_text}. {descr}" if descr else as_text
        kind = (_clean_str(o.get("source_kind"), 12) or "").lower()
        quote = _clean_str(o.get("quote"), 300)
        observations.append({
            "sheet": sheet,
            "transmission_mode": _clean_str(o.get("transmission_mode"), 80),
            "disease": _clean_str(o.get("disease"), 120),
            "group": _clean_str(o.get("group"), 120),
            "covariate": covariate,
            "value": value,
            "descr": descr,
            "notes": _clean_str(o.get("notes"), 400),
            "n_cases": _to_number(o.get("n_cases")),
            "pop_risk": _to_number(o.get("pop_risk")),
            "original_name": _clean_str(o.get("original_name"), 200),
            "page_section": _clean_str(o.get("page_section"), 120),
            "source_kind": kind if kind in SOURCE_KINDS else None,
            "quote": quote,
            "quote_verified": quote_in_text(quote, source_text),
        })
        if len(observations) >= _MAX_OBSERVATIONS:
            break
    return {"ref": ref, "coverage": coverage, "observations": observations}


# ─────────────────────────────────────────────────────────────────────────────
# Reading one article
# ─────────────────────────────────────────────────────────────────────────────
def _article_text(conn, row: dict) -> tuple[str, str, bool]:
    """(text, source, truncated): the full text when the article has it, else the abstract.

    The title always leads. `source` is what the reviewer must know: values extracted from
    an abstract are a fraction of what the paper reports (tables are not in abstracts)."""
    title = (row.get("title") or "").strip()
    full = conn.execute(text("""
        SELECT string_agg(content, E'\\n\\n' ORDER BY chunk_index, id)
        FROM document_chunk
        WHERE document_id = :id AND chunk_type IN ('fulltext_section', 'full_text')
    """), {"id": row["id"]}).scalar()
    if full and len(full) > 500:
        truncated = len(full) > EXTRACTION_MAX_CHARS
        return f"{title}\n\n{full[:EXTRACTION_MAX_CHARS]}", "fulltext", truncated
    return f"{title}\n\n{(row.get('abstract') or '').strip()}", "abstract", False


def extract_article(client, row: dict, source_text: str, source: str, truncated: bool,
                    disease_hint: str | None = None) -> dict[str, Any]:
    """One LLM call for one article, parsed and checked. Raises on any failure, so the
    caller counts an attempt instead of caching an empty result as if it were an answer."""
    payload = {"disease_of_interest": disease_hint or None, "text_is": source, "text": source_text}
    resp = client.chat.completions.create(
        model=_model("bulk"),
        messages=[{"role": "system", "content": _EXTRACTION_SYSTEM},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0, seed=42, max_tokens=8000,
        response_format={"type": "json_object"},
    )
    parsed = parse_extraction(json.loads(_json_content(resp, f"extraction {row.get('id')}")),
                              source_text)
    parsed.update({"v": EXTRACTION_VERSION, "source": source, "truncated": truncated,
                   "n_chars": len(source_text)})
    return parsed


def _needs_extraction(row: dict) -> bool:
    """An article is (re)extracted when it has none, when the extraction version moved, or
    when the full text arrived after an abstract-only pass. It is skipped once it has
    failed `_MAX_ATTEMPTS` times, and when it has nothing to read."""
    if int(row.get("extraction_attempts") or 0) >= _MAX_ATTEMPTS:
        return False
    has_text = bool(row.get("has_fulltext")) or len((row.get("abstract") or "").strip()) >= _MIN_ABSTRACT_CHARS
    if not has_text:
        return False
    ex = row.get("extraction_json")
    if not isinstance(ex, dict):
        return True
    if int(ex.get("v") or 0) < EXTRACTION_VERSION:
        return True
    return ex.get("source") == "abstract" and bool(row.get("has_fulltext"))


_ARTICLE_ROWS_SQL = """
    SELECT d.id, d.title, d.abstract, d.doi, d.year, d.authors, d.has_fulltext,
           d.extraction_json, COALESCE(d.extraction_attempts, 0) AS extraction_attempts,
           COALESCE(ars.screening_status, d.screening_status) AS screening_status
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id
    WHERE ars.scenario_id = :sid AND {gate}
    ORDER BY d.id
"""


def _relevant_rows(scenario_id: str) -> list[dict]:
    """EVERY relevant article of the scenario (above the threshold, or included by hand,
    never the excluded): the same gate as the rest of the application."""
    thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            text(_ARTICLE_ROWS_SQL.format(gate=relevant_gate_sql("d", "ars", ":thr"))),
            {"sid": scenario_id, "thr": thr}).mappings().all()]


# ─────────────────────────────────────────────────────────────────────────────
# The background job
# ─────────────────────────────────────────────────────────────────────────────
_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _llm_client():
    """The LLM client, in one place that the tests replace."""
    from llm_usage import MeteredOpenAI as _OAI
    return _OAI(timeout=180.0)


def _extract_scenario(scenario_id: str, disease_hint: str | None, max_articles: int = 0) -> dict[str, int]:
    rows = [r for r in _relevant_rows(scenario_id) if _needs_extraction(r)]
    cap = int(max_articles or EXTRACTION_MAX_ARTICLES or 0)
    if cap > 0:
        rows = rows[:cap]
    with _jobs_lock:
        _jobs[scenario_id].update({"total": len(rows), "done": 0, "failed": 0})
    if not rows:
        return {"total": 0, "done": 0, "failed": 0}
    client = _llm_client()

    def _work(row: dict) -> bool:
        try:
            with engine.connect() as conn:
                src_text, source, truncated = _article_text(conn, row)
            result = extract_article(client, row, src_text, source, truncated, disease_hint)
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE literature_document
                    SET extraction_json = CAST(:j AS jsonb), extraction_at = now(),
                        extraction_attempts = 0
                    WHERE id = :id
                """), {"j": json.dumps(result, ensure_ascii=False), "id": row["id"]})
            return True
        except Exception as e:                               # noqa: BLE001 - one article never stops the run
            logger.warning(f"extraction article {row.get('id')}: {e}")
            try:
                with engine.begin() as conn:
                    conn.execute(text("UPDATE literature_document SET extraction_attempts = "
                                      "COALESCE(extraction_attempts, 0) + 1 WHERE id = :id"),
                                 {"id": row["id"]})
            except Exception as e2:                          # noqa: BLE001
                logger.warning(f"extraction attempts {row.get('id')}: {e2}")
            return False

    with ThreadPoolExecutor(max_workers=_EXTRACTION_WORKERS) as ex:
        for ok in ex.map(_work, rows):
            with _jobs_lock:
                _jobs[scenario_id]["done" if ok else "failed"] += 1
    with _jobs_lock:
        return {k: _jobs[scenario_id][k] for k in ("total", "done", "failed")}


def start_extraction(scenario_id: str, disease_hint: str | None = None, max_articles: int = 0) -> dict[str, Any]:
    """Start the extraction in the background, one run per scenario at a time."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "no_llm"}
    with _jobs_lock:
        if _jobs.get(scenario_id, {}).get("running"):
            return {"status": "running", **{k: _jobs[scenario_id].get(k, 0) for k in ("total", "done", "failed")}}
        _jobs[scenario_id] = {"running": True, "started_at": time.time(), "total": 0, "done": 0, "failed": 0}

    def _run():
        try:
            _extract_scenario(scenario_id, disease_hint, max_articles)
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"extraction {scenario_id}: {e}")
            with _jobs_lock:
                _jobs[scenario_id]["error"] = str(e)[:300]
        finally:
            with _jobs_lock:
                _jobs[scenario_id]["running"] = False

    threading.Thread(target=_run, daemon=True, name=f"extraction-{scenario_id}").start()
    return {"status": "started"}


def extraction_status(scenario_id: str) -> dict[str, Any]:
    """Where the scenario stands, counted over ALL its relevant articles."""
    rows = _relevant_rows(scenario_id)
    done = [r for r in rows if isinstance(r.get("extraction_json"), dict)]
    pending = [r for r in rows if _needs_extraction(r)]
    with _jobs_lock:
        job = dict(_jobs.get(scenario_id) or {})
    return {
        "scenario_id": scenario_id,
        "n_relevant": len(rows),
        "n_extracted": len(done),
        "n_from_fulltext": sum(1 for r in done if r["extraction_json"].get("source") == "fulltext"),
        "n_from_abstract": sum(1 for r in done if r["extraction_json"].get("source") == "abstract"),
        "n_pending": len(pending),
        "n_observations": sum(len(r["extraction_json"].get("observations") or []) for r in done),
        "n_given_up": sum(1 for r in rows if int(r.get("extraction_attempts") or 0) >= _MAX_ATTEMPTS
                          and not isinstance(r.get("extraction_json"), dict)),
        "running": bool(job.get("running")),
        "job": {k: job.get(k, 0) for k in ("total", "done", "failed")} if job else None,
        "version": EXTRACTION_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The REDUCE half: counts over EVERY relevant article, in SQL, no LLM
# ─────────────────────────────────────────────────────────────────────────────
# What the assistant and the reviewers need is not a paragraph written from 24 excerpts
# but "how many of the relevant papers report sex-disaggregated data", counted over all of
# them. The map step above wrote the per-article facts; this aggregates them, the way
# api/digest.py does for the concepts.
#
# Two things the figures must never hide. The DENOMINATOR: only articles already extracted
# can be counted, so every count says out of how many, and how many relevant articles are
# still unread. And the SOURCE: a paper extracted from its abstract alone cannot show a
# table, so its "false" is weaker than a "false" read from the full text. Both are reported
# next to the count, not behind it.

_COVERAGE_LABELS_FR = {
    "sex_gender": "sexe ou genre", "age": "age", "occupation": "profession",
    "kap_risk_perception": "connaissances, attitudes, pratiques ou perception du risque",
    "ppe": "equipements de protection", "vaccination": "vaccination",
    "human_testing": "tests chez l'humain", "animal_host": "hotes animaux",
    "environment": "environnement", "vector": "vecteurs",
}
_TOP_GROUPS = 12
_TOP_CRUDE = 12

_OBS_FROM = """
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id
    CROSS JOIN LATERAL jsonb_array_elements(d.extraction_json->'observations') AS o
    WHERE ars.scenario_id = :sid AND {gate}
      AND jsonb_typeof(d.extraction_json->'observations') = 'array'
"""


def extraction_digest(scenario_id: str, threshold: float | None = None) -> dict[str, Any]:
    """Counts over ALL the relevant articles of what the extraction found.

    `n_extracted` is the denominator of every proportion below it; `n_relevant` -
    `n_extracted` articles have not been read yet and say nothing. `coverage[key]` counts
    the extracted articles that report the item, and `coverage_fulltext[key]` the ones read
    from the full text. `crude_counts` sums cases and population over rows that report both
    and whose quote was found in the text: it is a tally, NOT a pooled estimate (the
    populations differ, the labels are free text until the codebook, and one population can
    appear in two rows), and the digest says so."""
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    gate = relevant_gate_sql("d", "ars", ":thr")
    out: dict[str, Any] = {"scenario_id": scenario_id, "threshold": thr, "version": EXTRACTION_VERSION}
    try:
        cov_sql = ",\n".join(
            f"COUNT(*) FILTER (WHERE d.extraction_json->'coverage'->>'{k}' = 'true') AS c_{k}, "
            f"COUNT(*) FILTER (WHERE d.extraction_json->'coverage'->>'{k}' = 'true' "
            f"AND d.extraction_json->>'source' = 'fulltext') AS f_{k}" for k in COVERAGE_KEYS)
        with engine.connect() as conn:
            head = _rows(conn, f"""
                SELECT COUNT(*) AS n_relevant,
                       COUNT(*) FILTER (WHERE jsonb_typeof(d.extraction_json) = 'object') AS n_extracted,
                       COUNT(*) FILTER (WHERE d.extraction_json->>'source' = 'fulltext') AS n_fulltext,
                       COUNT(*) FILTER (WHERE d.extraction_json->>'source' = 'abstract') AS n_abstract,
                       {cov_sql}
                FROM literature_document d
                JOIN article_scenarios ars ON ars.document_id = d.id
                WHERE ars.scenario_id = :sid AND {gate}
            """, scenario_id, thr)[0]
            out["n_relevant"] = int(head["n_relevant"] or 0)
            out["n_extracted"] = int(head["n_extracted"] or 0)
            out["n_fulltext"] = int(head["n_fulltext"] or 0)
            out["n_abstract"] = int(head["n_abstract"] or 0)
            out["n_unread"] = out["n_relevant"] - out["n_extracted"]
            out["coverage"] = {k: int(head[f"c_{k}"] or 0) for k in COVERAGE_KEYS}
            out["coverage_fulltext"] = {k: int(head[f"f_{k}"] or 0) for k in COVERAGE_KEYS}

            out["by_sheet"] = [
                {"sheet": r["sheet"], "n_rows": int(r["n_rows"]), "n_articles": int(r["n_articles"]),
                 "n_quote_found": int(r["n_verified"])}
                for r in _rows(conn, f"""
                    SELECT o->>'sheet' AS sheet, COUNT(*) AS n_rows, COUNT(DISTINCT d.id) AS n_articles,
                           COUNT(*) FILTER (WHERE o->>'quote_verified' = 'true') AS n_verified
                    {_OBS_FROM.format(gate=gate)}
                    GROUP BY 1 ORDER BY n_rows DESC
                """, scenario_id, thr)]

            out["top_groups"] = _rows(conn, f"""
                SELECT o->>'sheet' AS sheet, LOWER(o->>'group') AS value, COUNT(DISTINCT d.id) AS n
                {_OBS_FROM.format(gate=gate)} AND NULLIF(TRIM(o->>'group'), '') IS NOT NULL
                GROUP BY 1, 2 ORDER BY n DESC, 2 LIMIT :top
            """, scenario_id, thr, top=_TOP_GROUPS)

            crude = _rows(conn, f"""
                SELECT o->>'sheet' AS sheet, LOWER(TRIM(o->>'covariate')) AS covariate,
                       COUNT(DISTINCT d.id) AS n_studies,
                       SUM((o->>'n_cases')::numeric) AS n_cases,
                       SUM((o->>'pop_risk')::numeric) AS pop_risk
                {_OBS_FROM.format(gate=gate)}
                  AND jsonb_typeof(o->'n_cases') = 'number' AND jsonb_typeof(o->'pop_risk') = 'number'
                  AND o->>'quote_verified' = 'true'
                GROUP BY 1, 2 ORDER BY n_studies DESC, pop_risk DESC, 2 LIMIT :top
            """, scenario_id, thr, top=_TOP_CRUDE)
            out["crude_counts"] = [
                {"sheet": r["sheet"], "covariate": r["covariate"], "n_studies": int(r["n_studies"]),
                 "n_cases": float(r["n_cases"] or 0), "pop_risk": float(r["pop_risk"] or 0)}
                for r in crude]
        out["crude_counts_note"] = ("A tally of rows that report both cases and population with a quote "
                                    "found in the text. Not a pooled estimate: populations differ, labels "
                                    "are free text, and one population may appear in two rows.")
        out["complete"] = True
    except Exception as e:                                   # noqa: BLE001 - never blocks the caller
        logger.warning(f"extraction_digest {scenario_id}: {e}")
        out.update({"complete": False, "error": str(e)[:300]})
    return out


def extraction_to_prompt(d: dict | None, max_chars: int = 2200) -> str:
    """The extraction digest as a block for a prompt, or "" when there is nothing to say
    (nothing extracted, or the aggregation failed: an incomplete block must assert nothing).

    It states its own denominator and its source, and tells the model to quote them, so
    "34 papers report sex-disaggregated data" is never read as "34 of all the relevant"."""
    if not d or not d.get("complete") or not d.get("n_extracted"):
        return ""
    n, rel = d["n_extracted"], d["n_relevant"]
    lines = [
        f"EXTRACTION STRUCTUREE: {n} des {rel} articles pertinents ont ete lus ({d['n_fulltext']} "
        f"depuis le texte integral, {d['n_abstract']} depuis le resume seul, qui ne contient pas les "
        f"tableaux : pour eux un element absent est un minorant, pas une absence).",
    ]
    if d.get("n_unread"):
        lines.append(f"{d['n_unread']} articles pertinents ne sont pas encore extraits : ne rien conclure sur eux.")
    cov = [f"{_COVERAGE_LABELS_FR[k]} {d['coverage'][k]} (dont {d['coverage_fulltext'][k]} en texte integral)"
           for k in COVERAGE_KEYS if d["coverage"].get(k)]
    lines.append(f"Parmi les {n} articles extraits, rapportent des donnees sur : "
                 + (", ".join(cov) if cov else "aucun des elements suivis (sexe, age, profession, CAP, EPI, vaccination...)") + ".")
    if d.get("by_sheet"):
        lines.append("Lignes extraites : " + ", ".join(
            f"{r['sheet']} {r['n_rows']} ({r['n_articles']} articles, citation retrouvee {r['n_quote_found']})"
            for r in d["by_sheet"]) + ".")
    if d.get("top_groups"):
        lines.append("Groupes les plus frequents : " + ", ".join(
            f"{r['value']} [{r['sheet']}] ({r['n']})" for r in d["top_groups"]) + ".")
    if d.get("crude_counts"):
        lines.append("Totaux bruts (lignes avec cas et population, citation retrouvee ; pas une estimation poolee) : "
                     + "; ".join(f"{r['covariate']} [{r['sheet']}] {r['n_studies']} etudes, "
                                 f"{r['n_cases']:g} cas sur {r['pop_risk']:g}" for r in d["crude_counts"]) + ".")
    lines.append(f"Toute proportion tiree de ce bloc doit citer son denominateur ({n} articles extraits sur {rel}).")
    return "\n".join(lines)[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Export in the template's shape
# ─────────────────────────────────────────────────────────────────────────────
def _first_author(authors: str | None) -> str:
    """The first author's family name when the stored form makes it clear ("Dressler A",
    "Dressler, Aparna"); otherwise the name as stored, for the reviewer to fix."""
    first = re.split(r";|\n", (authors or "").strip())[0].strip()
    if "," in first:
        head = first.split(",")[0].strip()
        if head and len(head.split()) <= 2:
            return head
    toks = first.split()
    if len(toks) >= 2 and re.fullmatch(r"[A-Z]{1,3}", toks[-1]):
        return " ".join(toks[:-1])
    return first


def _cell(v: Any) -> Any:
    if isinstance(v, bool):
        return "Y" if v else "N"
    return v


def template_rows(articles: list[dict], id_prefix: str = "LR") -> dict[str, list[dict]]:
    """The extractions as template rows: `{"ref": [...], "human_susc": [...], ...}`.
    `articles` are relevant-article rows carrying an `extraction_json`. Pure."""
    out: dict[str, list[dict]] = {"ref": [], **{s: [] for s in SHEETS}}
    for a in articles:
        ex = a.get("extraction_json")
        if not isinstance(ex, dict):
            continue
        aid = f"{id_prefix}{a['id']}"
        ref = ex.get("ref") or {}
        doi = (a.get("doi") or "").strip()
        out["ref"].append({
            "id": aid, "first_author": _first_author(a.get("authors")),
            "reference": f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
            "year": a.get("year"), "description": ref.get("description"),
            "article_type": ref.get("article_type"), "study_start": ref.get("study_start"),
            "study_end": ref.get("study_end"), "location": ref.get("location"),
            "nuts1": None, "nuts2": None, "nuts3": None, "notes_geo": ref.get("notes_geo"),
            "risk_pop": ref.get("risk_pop"), "positive": ref.get("positive"),
            "percent_positive": ref.get("percent_positive"),
            "math_model": ref.get("math_model"), "model_type": ref.get("model_type"),
            "exclusion": (a.get("screening_reason") if a.get("screening_status") == "excluded" else None),
        })
        for o in ex.get("observations") or []:
            if o.get("sheet") in out:
                out[o["sheet"]].append({**o, "id": aid, "source": ex.get("source")})
    return out


def build_workbook(articles: list[dict], coverage_note: str, id_prefix: str = "LR") -> bytes:
    """An .xlsx with the template's sheets and column titles, plus review columns on the
    right and a README sheet that says what the numbers cover."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    rows = template_rows(articles, id_prefix)
    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    for line in (
        "LiteRev structured extraction, shaped like the T2.4 / T4.5 data extraction template.",
        coverage_note,
        "Every row was produced by a model and must be checked against the paper by a reviewer.",
        "'Quote found in text' = N means the model's quote was not found in the text it read: "
        "check that row first.",
        "'Extracted from' = abstract means the paper's full text was not available: tables "
        "are not in abstracts, so most values are missing, not absent from the paper.",
        "NUTS columns are empty: geography is resolved in a later step.",
        "ID is LR + the LiteRev article id; rename it to your initials and a number if needed.",
    ):
        readme.append([line])
    readme.column_dimensions["A"].width = 120
    bold = Font(bold=True)

    ws = wb.create_sheet("REF")
    ws.append([c[0] for c in REF_COLUMNS])
    for r in rows["ref"]:
        ws.append([_cell(r.get(f)) for _, f in REF_COLUMNS])
    for key in SHEETS:
        title, cols = SHEET_COLUMNS[key]
        have = {f for _, f in cols}
        extra = [(h, f) for h, f in REVIEW_COLUMNS if f not in have]
        ws = wb.create_sheet(title)
        ws.append([c[0] for c in cols] + [c[0] for c in extra])
        for r in rows[key]:
            ws.append([_cell(r.get(f)) for _, f in cols] + [_cell(r.get(f)) for _, f in extra])
    for ws in wb.worksheets[1:]:
        for c in ws[1]:
            c.font = bold
        ws.freeze_panes = "B2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_long_csv(articles: list[dict], id_prefix: str = "LR") -> str:
    """One flat table, one row per observation, with its sheet: for scripts and datasets."""
    rows = template_rows(articles, id_prefix)
    fields = ["id", "sheet", "transmission_mode", "disease", "group", "covariate", "value", "descr",
              "notes", "n_cases", "pop_risk", "original_name", "page_section", "source_kind",
              "quote", "quote_verified", "source"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(fields)
    for key in SHEETS:
        for r in rows[key]:
            w.writerow([key if f == "sheet" else r.get(f) for f in fields])
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/user-scenarios/{scenario_id}/extraction/run")
def run_scenario_extraction(scenario_id: str, max_articles: int = 0,
                            _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Extract, in the background, every relevant article that has no extraction yet (or
    whose full text arrived since an abstract-only pass). No cap unless `max_articles` or
    EXTRACTION_MAX_ARTICLES says so."""
    scenario = _get_user_scenario_or_404(scenario_id)
    out = start_extraction(scenario_id, scenario.get("name"), max(0, int(max_articles or 0)))
    return {**out, **{k: v for k, v in extraction_status(scenario_id).items()
                      if k in ("n_relevant", "n_pending")}}


@app.get("/user-scenarios/{scenario_id}/extraction/status")
def get_scenario_extraction_status(scenario_id: str) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    return extraction_status(scenario_id)


@app.get("/user-scenarios/{scenario_id}/extraction/coverage")
def get_scenario_extraction_coverage(scenario_id: str) -> dict[str, Any]:
    """What the extraction found, counted over ALL the relevant articles (see
    `extraction_digest`): who reports sex, age, KAP, PPE..., and out of how many."""
    _get_user_scenario_or_404(scenario_id)
    return extraction_digest(scenario_id)


#: Same bound as the PICO list: a page is at most this many articles.
EXTRACTION_PAGE_MAX = 5000


@app.get("/user-scenarios/{scenario_id}/extraction/articles")
def list_scenario_extraction_articles(scenario_id: str, limit: int = EXTRACTION_PAGE_MAX,
                                      offset: int = 0) -> dict[str, Any]:
    """EVERY relevant article of the scenario with what its extraction found, paged.

    The extracted ones come first. The rows carry counts, not the observations (read one
    article's with /articles/{id}/extraction), so the page stays small. The response says
    how many articles exist, how many are extracted and whether more pages remain, so the
    screen never reads a page as the whole corpus."""
    _get_user_scenario_or_404(scenario_id)
    limit = max(1, min(int(limit), EXTRACTION_PAGE_MAX))
    offset = max(0, int(offset))
    thr = _get_scenario_threshold(scenario_id)
    gate = relevant_gate_sql("d", "ars", ":thr")
    obs = ("CASE WHEN jsonb_typeof(d.extraction_json->'observations') = 'array' "
           "THEN d.extraction_json->'observations' ELSE '[]'::jsonb END")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(f"""
            SELECT d.id, d.title, d.year, d.doi, d.journal,
                   COALESCE(jsonb_typeof(d.extraction_json) = 'object', FALSE) AS has_extraction,
                   d.extraction_json->>'source' AS source,
                   COALESCE(d.extraction_json->>'truncated', 'false') = 'true' AS text_truncated,
                   d.extraction_json->'coverage' AS coverage,
                   jsonb_array_length({obs}) AS n_observations,
                   (SELECT COUNT(*) FROM jsonb_array_elements({obs}) o
                     WHERE o->>'quote_verified' = 'true') AS n_quote_found,
                   COALESCE(d.extraction_attempts, 0) AS attempts
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
            ORDER BY COALESCE(jsonb_typeof(d.extraction_json) = 'object', FALSE) DESC, d.year DESC NULLS LAST, d.id DESC
            LIMIT :limit OFFSET :offset
        """), {"sid": scenario_id, "thr": thr, "limit": limit, "offset": offset}).mappings().all()]
        head = conn.execute(text(f"""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE jsonb_typeof(d.extraction_json) = 'object') AS extracted
            FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
        """), {"sid": scenario_id, "thr": thr}).mappings().first()
    for r in rows:
        r["n_observations"] = int(r["n_observations"] or 0)
        r["n_quote_found"] = int(r["n_quote_found"] or 0)
        r["coverage"] = {k: bool((r["coverage"] or {}).get(k)) for k in COVERAGE_KEYS} if r["has_extraction"] else None
    total = int(head["total"] or 0)
    returned = len(rows)
    return {
        "scenario_id": scenario_id, "threshold": thr, "total": total,
        "extracted": int(head["extracted"] or 0), "offset": offset, "limit": limit,
        "returned": returned, "truncated": offset + returned < total,
        "next_offset": offset + returned if offset + returned < total else None,
        "page_max": EXTRACTION_PAGE_MAX, "articles": rows,
    }


@app.get("/user-scenarios/{scenario_id}/articles/{article_id}/extraction")
def get_article_extraction(scenario_id: str, article_id: int) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT d.id, d.title, d.extraction_json, d.extraction_at
            FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND d.id = :id
        """), {"sid": scenario_id, "id": article_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Article not in this scenario.")
    return {"id": row["id"], "title": row["title"], "extracted_at": row["extraction_at"],
            "extraction": row["extraction_json"]}


@app.get("/user-scenarios/{scenario_id}/extraction/export")
def export_scenario_extraction(scenario_id: str, format: str = Query("xlsx"),
                               id_prefix: str = Query("LR", max_length=12)) -> Response:
    """The extractions of ALL the relevant articles, as the template workbook (`xlsx`) or
    one flat table (`csv`). Articles not extracted yet are not in it; the README sheet and
    the `X-Coverage` header say how many that is."""
    _get_user_scenario_or_404(scenario_id)
    fmt = (format or "xlsx").lower()
    if fmt not in ("xlsx", "csv"):
        raise HTTPException(status_code=400, detail="format must be xlsx or csv")
    rows = _relevant_rows(scenario_id)
    done = [r for r in rows if isinstance(r.get("extraction_json"), dict)]
    from_abs = sum(1 for r in done if r["extraction_json"].get("source") == "abstract")
    note = (f"{len(done)} of the {len(rows)} relevant articles are extracted "
            f"({len(rows) - len(done)} not yet); {from_abs} of them from the abstract only.")
    prefix = re.sub(r"[^A-Za-z0-9_-]", "", id_prefix) or "LR"
    if fmt == "csv":
        body, media, ext = build_long_csv(done, prefix).encode("utf-8-sig"), "text/csv; charset=utf-8", "csv"
    else:
        body, media, ext = (build_workbook(done, note, prefix),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx")
    return Response(content=body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="extraction_{re.sub(r"[^A-Za-z0-9_-]", "_", scenario_id)}.{ext}"',
        "X-Coverage": note,
    })
