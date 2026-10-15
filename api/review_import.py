"""Import the decisions of a review already screened elsewhere.

A team arrives with a review they have already done: a Rayyan or Covidence export, an
EndNote library, a spreadsheet a reviewer filled in. Those decisions are work, and until
now re-doing them by hand was the only way to bring them into a scenario. This module
reads such a file and applies its decisions to the corpus: included, excluded, and the
reason each carries.

Two rules shape it.

A model never decides a paper's fate. The decisions in the file are the reviewers' own,
and they must stay traceable: they feed the PRISMA flow, the agreement between reviewers
and the model that orders the screening queue. So the file is read deterministically, by
column name; an LLM is asked ONE question only, and only when the headers are not
recognised: which column is which. The mapping it proposes is returned for a person to
confirm, and the import itself is then plain data handling. Nothing is inferred from
prose, and no decision is invented for a row that does not carry one.

Nothing is overwritten in silence. A paper this scenario has already decided differently
is reported as a conflict and left alone, unless the caller asks for the file to win. A
paper of theirs that the corpus does not hold is added by hand (api/additions.py) when it
is an inclusion, so an "included" decision never silently vanishes. Every import records
what each row's status was before, so it can be undone in one call.

Preview first (`/decisions/preview`), apply second (`/decisions/import`), undo if needed
(`/decisions/undo`). The preview writes nothing and says exactly what the import would do.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from .additions import (MAX_FILE_BYTES, MAX_RECORDS_PER_FILE, TABLE_COLUMNS, _clean, _fold_name,
                        add_records, detect_kind, make_record, parse_bibtex, parse_nbib, parse_ris,
                        year_of)
from .core import app, engine, logger, require_api_key
from .documents import _normalize_doi, _normalize_title
from .double_blind import _write_ars_screening
from .learning import schedule_retrain
from .scenario_store import _get_user_scenario_or_404

#: Column names that carry the decision and its reason, folded (lower case, no punctuation).
DECISION_COLUMNS = ("decision", "decisions", "included", "include", "inclusion", "screening",
                    "screeningdecision", "status", "label", "verdict", "finaldecision",
                    "titleabstractscreening", "fulltextscreening")
REASON_COLUMNS = ("reason", "reasons", "exclusionreason", "exclusionreasons", "reasonforexclusion",
                  "notes", "note", "comment", "comments", "justification")

#: What a decision cell may say. Anything else leaves the row undecided, and the report
#: says so rather than guessing.
INCLUDED_WORDS = {"included", "include", "inclus", "incluso", "yes", "oui", "si", "y", "1", "true",
                  "keep", "kept", "accept", "accepted", "eligible", "in"}
EXCLUDED_WORDS = {"excluded", "exclude", "exclu", "escluso", "no", "non", "n", "0", "false",
                  "reject", "rejected", "drop", "dropped", "ineligible", "out", "irrelevant"}
UNDECIDED_WORDS = {"maybe", "unsure", "undecided", "pending", "conflict", "unclear", "peut-etre",
                   "peutetre", "forse", "tbd", ""}

#: Rows read from one file. A review of more than this is split; the report says so.
MAX_ROWS = MAX_RECORDS_PER_FILE


def _ensure_import_tables() -> None:
    """One row per import, with what it changed: that is what makes it undoable."""
    from .schema_boot import _exec_ddl_isolated
    _exec_ddl_isolated([
        """CREATE TABLE IF NOT EXISTS scenario_decision_import (
               id          BIGSERIAL PRIMARY KEY,
               scenario_id TEXT NOT NULL,
               filename    TEXT,
               created_at  TIMESTAMP DEFAULT NOW(),
               undone_at   TIMESTAMP,
               report      JSONB NOT NULL,
               undo        JSONB NOT NULL
           )""",
        "CREATE INDEX IF NOT EXISTS ix_decision_import_scenario ON scenario_decision_import (scenario_id, id DESC)",
    ], "_ensure_import_tables")


try:
    _ensure_import_tables()
except Exception as _e:  # pragma: no cover - a database without the base tables
    logger.warning(f"_ensure_import_tables: {_e}")


# ── Pure: reading a decision ─────────────────────────────────────────────────
def decision_of(value: Any) -> str | None:
    """"included", "excluded", or None when the cell decides nothing.

    Reads the words a screening tool writes, in the languages this project meets, and a
    boolean or a 0/1. A cell it does not know leaves the row undecided: the report names
    the value, rather than letting a guess become a reviewer's decision."""
    import re
    raw = str(value if value is not None else "").strip().lower()
    if raw in UNDECIDED_WORDS:
        return None
    if raw in INCLUDED_WORDS:
        return "included"
    if raw in EXCLUDED_WORDS:
        return "excluded"
    # The first WORD of a cell that carries its reason with it: "Included (full text)",
    # "Exclude: wrong population", "excluded - no abstract", "Included/eligible".
    words = [w for w in re.split(r"[^a-z0-9]+", raw) if w]
    if words:
        if words[0] in UNDECIDED_WORDS:
            return None
        if words[0] in INCLUDED_WORDS:
            return "included"
        if words[0] in EXCLUDED_WORDS:
            return "excluded"
    return None


#: Rayyan's CSV export carries the decision INSIDE its notes field, as
#: `RAYYAN-INCLUSION: {"Claudia"=>"Excluded"} | RAYYAN-EXCLUSION-REASONS: wrong study design`.
#: It is the export a team most often arrives with (the ISS team's own file holds 1,524
#: records this way), and no column of it says "decision": read as a plain reason column,
#: every row came back undecided.
_RAYYAN_VOTES = re.compile(r"RAYYAN-INCLUSION:\s*\{(.*?)\}", re.S)
_RAYYAN_ONE_VOTE = re.compile(r'"[^"]*"\s*=>\s*"([^"]+)"')
_RAYYAN_REASONS = re.compile(r"RAYYAN-EXCLUSION-REASONS:\s*([^|]+)")


def rayyan_decision(note: Any) -> tuple[str | None, str | None]:
    """(decision, reason) read from a Rayyan note, or (None, None).

    Several reviewers may have voted. They are read exactly as this project reads its own
    double-blind votes: agreement decides, a disagreement decides nothing until someone
    arbitrates. The reviewers' NAMES are never kept: the import needs the decision, and a
    name here is personal data that has no reason to travel into the corpus."""
    text_ = str(note or "")
    m = _RAYYAN_VOTES.search(text_)
    if not m:
        return None, None
    votes = {decision_of(v) for v in _RAYYAN_ONE_VOTE.findall(m.group(1))}
    reason = None
    mr = _RAYYAN_REASONS.search(text_)
    if mr:
        reason = _clean(mr.group(1))
    if len(votes) != 1:
        return None, reason                                  # nobody voted, or they disagree
    return votes.pop(), reason


def map_decision_columns(header: list) -> dict[str, Any]:
    """Which column carries what: the bibliographic fields `additions` already knows, plus
    the decision and its reason. The first header that matches a known name wins."""
    folded = [(_fold_name(h), h) for h in header]
    out: dict[str, Any] = {}
    for field, names in {**TABLE_COLUMNS, "decision": DECISION_COLUMNS, "reason": REASON_COLUMNS}.items():
        for fname, h in folded:
            if fname in names and field not in out:
                out[field] = h
    return out


def rows_from_table(data: bytes, kind: str) -> tuple[list[dict], list]:
    """The rows of a table and its header, whatever the delimiter."""
    import csv
    import io
    if kind == "xlsx":
        import pandas as pd
        df = pd.read_excel(io.BytesIO(data), dtype=str)
        df = df.where(df.notna(), None)
        return df.to_dict(orient="records"), [str(c) for c in df.columns]
    from .additions import _decode
    text_ = _decode(data)
    first = text_.splitlines()[0] if text_.strip() else ""
    delim = max(",;\t|", key=first.count) if first else ","
    reader = csv.DictReader(io.StringIO(text_), delimiter=delim)
    rows = [{k: v for k, v in r.items() if k is not None} for r in reader]
    return rows, list(reader.fieldnames or [])


def records_with_decisions(data: bytes, kind: str, mapping: dict | None = None) -> tuple[list[dict], list, dict]:
    """(records, header, mapping) read from an uploaded file.

    A table carries its decisions in a column. A reference file (RIS, nbib, BibTeX) does
    not: those formats have no decision field, so such a file brings the papers and their
    reasons only where a note field holds them; its records come back undecided, which the
    report says plainly."""
    if kind in ("csv", "xlsx"):
        rows, header = rows_from_table(data, kind)
        if len(rows) > MAX_ROWS:
            raise ValueError(f"{len(rows)} rows: at most {MAX_ROWS} per file")
        cols = dict(mapping) if mapping else map_decision_columns(header)
        if "title" not in cols and "doi" not in cols and "pmid" not in cols:
            raise ValueError("no title, DOI or PMID column: nothing could be matched to the corpus")
        out = []
        for r in rows:
            rec = make_record(r.get(cols.get("title")), r.get(cols.get("abstract")), r.get(cols.get("doi")),
                              r.get(cols.get("pmid")), r.get(cols.get("year")), r.get(cols.get("authors")),
                              r.get(cols.get("journal")), r.get(cols.get("url")))
            if rec is None:
                # No usable title, but an identifier can still carry a decision.
                doi = _normalize_doi(_clean(r.get(cols.get("doi")))) if cols.get("doi") else None
                pmid = _clean(r.get(cols.get("pmid"))) if cols.get("pmid") else None
                if not doi and not pmid:
                    continue
                rec = {"title": _clean(r.get(cols.get("title"))) or "", "abstract": None, "doi": doi,
                       "pmid": pmid, "year": year_of(r.get(cols.get("year"))), "authors": None,
                       "journal": None, "url": None, "fulltext": None}
            raw = r.get(cols.get("decision")) if cols.get("decision") else None
            rec["decision"] = decision_of(raw)
            rec["decision_raw"] = _clean(raw)
            rec["reason"] = _clean(r.get(cols.get("reason"))) if cols.get("reason") else None
            # Rayyan writes the decision inside its notes field, under no column of its own.
            # Looked for in every cell, so it is found whichever column holds it.
            if rec["decision"] is None or "RAYYAN-INCLUSION" in str(rec["reason"] or ""):
                for value in r.values():
                    if "RAYYAN-INCLUSION" not in str(value or ""):
                        continue
                    found, why = rayyan_decision(value)
                    rec["decision"] = rec["decision"] or found
                    # The reason is the one Rayyan names, never the whole note: that string
                    # carries the reviewer's name, which has no reason to enter the corpus.
                    if "RAYYAN-INCLUSION" in str(rec["reason"] or "") or not rec["reason"]:
                        rec["reason"] = why
                    rec["decision_raw"] = rec["decision_raw"] or "RAYYAN-INCLUSION"
                    break
            out.append(rec)
        return out, header, cols
    parser = {"ris": parse_ris, "nbib": parse_nbib, "bibtex": parse_bibtex}.get(kind)
    if parser is None:
        raise ValueError(f"a decision file must be a table (CSV, Excel) or a reference file; got {kind}")
    from .additions import _decode
    recs = parser(_decode(data))
    for rec in recs:
        rec["decision"], rec["decision_raw"], rec["reason"] = None, None, None
    return recs, [], {}


def suggest_mapping(header: list, sample_rows: list[dict], lang: str | None = None) -> dict[str, Any]:
    """Ask a model WHICH COLUMN IS WHICH, and nothing else.

    Only reached when the header names are not recognised. The model never sees a decision
    to make: it reads the column names and a few rows and answers with a mapping, which the
    caller shows to a person before anything is applied. Returns {} when no model is
    configured or the answer is unusable, and the caller then asks for a mapping by hand."""
    from .pico_map import json_completion
    from llm_usage import MeteredOpenAI as _OAI, model_for as _model
    fields = "title, abstract, doi, pmid, year, authors, journal, url, decision, reason"
    system = ("You map the columns of a literature screening export onto known fields. "
              "You never judge the articles and never invent a decision. Answer with JSON only. "
              "No em dash.")
    user = ("Columns of the file:\n" + json.dumps([str(h) for h in header], ensure_ascii=False)
            + "\n\nFirst rows:\n" + json.dumps(sample_rows[:5], ensure_ascii=False, default=str)[:4000]
            + f"\n\nReturn {{\"mapping\": {{field: column}}}} where field is one of: {fields}. "
              "Use a column name exactly as given. Leave a field out when no column carries it. "
              "`decision` is the column that says whether the article was included or excluded; "
              "`reason` is the one that says why.")
    try:
        client = _OAI(timeout=60.0)

        def _ask(messages):
            return client.chat.completions.create(model=_model("write"), temperature=0,
                                                  max_tokens=400, response_format={"type": "json_object"},
                                                  messages=messages)

        data = json.loads(json_completion(_ask, [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}], "decision column mapping"))
    except Exception as e:                                   # noqa: BLE001 - a suggestion, never fatal
        logger.warning(f"suggest_mapping: {e}")
        return {}
    known = set(TABLE_COLUMNS) | {"decision", "reason"}
    heads = {str(h) for h in header}
    return {str(k): str(v) for k, v in (data.get("mapping") or {}).items()
            if str(k) in known and str(v) in heads}


# ── Matching the corpus ──────────────────────────────────────────────────────
def _corpus_index(scenario_id: str) -> dict[str, dict[str, int]]:
    """The scenario's articles, by DOI, by PMID and by normalised title. Read once: a
    review holds thousands of records, and one query per record would be thousands."""
    idx: dict[str, dict[str, int]] = {"doi": {}, "pmid": {}, "title": {}}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.doi, d.pmid, d.title_norm, d.title
            FROM literature_document d
            JOIN article_scenarios a ON a.document_id = d.id AND a.scenario_id = :sid
        """), {"sid": scenario_id}).all()
    for doc_id, doi, pmid, title_norm, title in rows:
        if doi:
            idx["doi"].setdefault(_normalize_doi(doi) or "", int(doc_id))
        if pmid:
            idx["pmid"].setdefault(str(pmid).strip(), int(doc_id))
        tn = title_norm or _normalize_title(title)
        if tn and len(tn) >= 20:
            idx["title"].setdefault(tn, int(doc_id))
    idx["doi"].pop("", None)
    return idx


def match_record(rec: dict, idx: dict[str, dict[str, int]]) -> tuple[int | None, str | None]:
    """The scenario's article this record is, by DOI, then PMID, then normalised title."""
    doi = _normalize_doi(rec.get("doi")) if rec.get("doi") else None
    if doi and doi in idx["doi"]:
        return idx["doi"][doi], "doi"
    pmid = str(rec.get("pmid") or "").strip()
    if pmid and pmid in idx["pmid"]:
        return idx["pmid"][pmid], "pmid"
    tn = _normalize_title(rec.get("title"))
    if tn and len(tn) >= 20 and tn in idx["title"]:
        return idx["title"][tn], "title"
    return None, None


def _current_decisions(scenario_id: str, doc_ids: list[int]) -> dict[int, tuple]:
    """What this scenario already says about those articles: status, reason, notes."""
    if not doc_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT document_id, screening_status, screening_reason, screening_notes
            FROM article_scenarios
            WHERE scenario_id = :sid AND document_id = ANY(CAST(:ids AS bigint[]))
        """), {"sid": scenario_id, "ids": [int(i) for i in doc_ids]}).all()
    return {int(r[0]): (r[1], r[2], r[3]) for r in rows}


def plan_import(scenario_id: str, records: list[dict], overwrite: bool, add_missing: bool) -> list[dict]:
    """What the import WOULD do, record by record. Reads only."""
    idx = _corpus_index(scenario_id)
    matched = [(rec, *match_record(rec, idx)) for rec in records]
    current = _current_decisions(scenario_id, [doc_id for _r, doc_id, _h in matched if doc_id])
    plan = []
    for rec, doc_id, how in matched:
        decision, reason = rec.get("decision"), rec.get("reason")
        row = {"title": rec.get("title") or "", "doi": rec.get("doi"), "pmid": rec.get("pmid"),
               "year": rec.get("year"), "decision": decision, "decision_raw": rec.get("decision_raw"),
               "reason": reason, "id": doc_id, "matched_by": how}
        if decision is None:
            row["action"] = "no_decision"
        elif doc_id is None:
            # Not in the corpus: an inclusion is added by hand, an exclusion is not (adding
            # an article in order to exclude it would only inflate the corpus).
            row["action"] = "add_and_include" if (add_missing and decision == "included") else "not_in_corpus"
        else:
            was = (current.get(doc_id) or (None, None, None))[0]
            if was == decision:
                row["action"] = "already"
            elif was in ("included", "excluded"):
                row["action"] = "overwrite_conflict" if overwrite else "conflict"
                row["current_status"] = was
            else:
                row["action"] = "apply"
        plan.append(row)
    return plan


def _counts(plan: list[dict]) -> dict[str, int]:
    out = {k: 0 for k in ("apply", "already", "conflict", "overwrite_conflict", "add_and_include",
                          "not_in_corpus", "no_decision")}
    for row in plan:
        out[row["action"]] = out.get(row["action"], 0) + 1
    out["records"] = len(plan)
    out["included"] = sum(1 for r in plan if r["decision"] == "included")
    out["excluded"] = sum(1 for r in plan if r["decision"] == "excluded")
    out["matched"] = sum(1 for r in plan if r.get("id"))
    return out


def apply_import(scenario_id: str, plan: list[dict], filename: str, note: str) -> dict[str, Any]:
    """Write the decisions the plan calls for, remembering what each row said before."""
    undo: list[list] = []
    applied = 0
    to_apply = [r for r in plan if r["action"] in ("apply", "overwrite_conflict")]
    if to_apply:
        before = _current_decisions(scenario_id, [r["id"] for r in to_apply])
        with engine.begin() as conn:
            for row in to_apply:
                was = before.get(row["id"]) or (None, None, None)
                undo.append([row["id"], was[0], was[1], was[2]])
                _write_ars_screening(conn, scenario_id, row["id"], row["decision"],
                                     row.get("reason"), note)
                applied += 1
    # An inclusion their file holds and this corpus does not: added by hand, included.
    added = []
    to_add = [r for r in plan if r["action"] == "add_and_include"]
    if to_add:
        recs = [{"title": r["title"], "abstract": None, "doi": r.get("doi"), "pmid": r.get("pmid"),
                 "year": r.get("year"), "authors": None, "journal": None, "url": None, "fulltext": None}
                for r in to_add if (r["title"] or r.get("doi") or r.get("pmid"))]
        added = add_records(scenario_id, recs, filename, include=True)
        for done in added:
            if done.get("id") and done.get("included"):
                undo.append([int(done["id"]), None, None, None])
    report = {"counts": _counts(plan), "applied": applied, "added": len(added),
              "filename": filename, "at": datetime.now(timezone.utc).isoformat()}
    import_id = None
    try:
        with engine.begin() as conn:
            import_id = conn.execute(text("""
                INSERT INTO scenario_decision_import (scenario_id, filename, report, undo)
                VALUES (:sid, :f, CAST(:r AS jsonb), CAST(:u AS jsonb))
                RETURNING id
            """), {"sid": scenario_id, "f": filename[:200], "r": json.dumps(report),
                   "u": json.dumps(undo)}).scalar()
    except Exception as e:                                   # noqa: BLE001 - the decisions are written
        logger.warning(f"apply_import {scenario_id}: the import could not be recorded ({e}); no undo for it")
    if applied or added:
        schedule_retrain(scenario_id)
    return {"import_id": int(import_id) if import_id else None, **report}


# ── Endpoints ────────────────────────────────────────────────────────────────
async def _read_file(file: UploadFile) -> tuple[str, bytes]:
    import os
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"{file.filename}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    return os.path.basename(file.filename or "decisions"), data


def _prepare(scenario_id: str, name: str, data: bytes, mapping: dict | None,
             overwrite: bool, add_missing: bool, suggest: bool, preview: bool = False) -> dict[str, Any]:
    """Read the file and work out what it would do. `preview` makes an unreadable header a
    RESULT rather than an error: a file whose columns are not recognised is exactly the case
    the mapping flow exists for, and the screen needs to see the header to offer one."""
    kind = detect_kind(name, data[:4096])
    if kind is None:
        raise HTTPException(status_code=422,
                            detail="Format not recognised (CSV, Excel .xlsx, RIS, BibTeX, PubMed nbib).")
    if kind == "pdf":
        raise HTTPException(status_code=422,
                            detail="A PDF carries no decision column. Upload the screening export "
                                   "(CSV, Excel) or add the paper itself with /articles/upload.")
    header: list = []
    unreadable: str | None = None
    try:
        records, header, used = records_with_decisions(data, kind, mapping)
    except ValueError as e:
        if not preview:
            raise HTTPException(status_code=422, detail=str(e)) from None
        records, used, unreadable = [], {}, str(e)
        if kind in ("csv", "xlsx"):
            try:
                _rows, header = rows_from_table(data, kind)
            except Exception as _e:                          # noqa: BLE001 - the error is the answer
                logger.warning(f"decision preview {scenario_id}: {_e}")
    suggested: dict[str, Any] = {}
    if suggest and kind in ("csv", "xlsx") and (unreadable or "decision" not in used):
        try:
            rows, head = rows_from_table(data, kind)
            header = header or head
            suggested = suggest_mapping(header, rows[:5])
        except Exception as _e:                              # noqa: BLE001 - a suggestion, never fatal
            logger.warning(f"decision preview {scenario_id}: {_e}")
    plan = plan_import(scenario_id, records, overwrite, add_missing) if records else []
    return {"scenario_id": scenario_id, "filename": name, "kind": kind, "header": [str(h) for h in header],
            "mapping": {k: str(v) for k, v in used.items()}, "suggested_mapping": suggested,
            "unreadable": unreadable, "counts": _counts(plan), "plan": plan}


@app.post("/user-scenarios/{scenario_id}/decisions/preview")
async def preview_decisions(
    scenario_id: str,
    file: UploadFile = File(...),
    mapping: str | None = Form(None),
    overwrite: bool = False,
    add_missing: bool = True,
    suggest: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """What importing this review's decisions would do. Writes nothing.

    `mapping` is an optional JSON object {field: column} that overrides the column names
    read from the header. When the header is not recognised and `suggest` is on, a model is
    asked which column is which and its proposal comes back as `suggested_mapping`, for a
    person to confirm and pass back as `mapping`: it maps columns, it never decides."""
    _get_user_scenario_or_404(scenario_id)
    name, data = await _read_file(file)
    try:
        asked = json.loads(mapping) if mapping else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="mapping must be a JSON object {field: column}") from None
    out = await run_in_threadpool(_prepare, scenario_id, name, data, asked, overwrite, add_missing,
                                  suggest, True)
    out["plan"] = out["plan"][:500]                          # the preview shows, the import applies
    return out


@app.post("/user-scenarios/{scenario_id}/decisions/import")
async def import_decisions(
    scenario_id: str,
    file: UploadFile = File(...),
    mapping: str | None = Form(None),
    overwrite: bool = False,
    add_missing: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Apply this review's decisions to the scenario.

    Each matched article receives the file's decision and its reason; an article this
    scenario already decided otherwise is left alone and reported as a conflict, unless
    `overwrite` is true. An inclusion the corpus does not hold is added by hand when
    `add_missing` is true, so no inclusion is lost. The import is recorded and can be
    undone. Matching is by DOI, then PMID, then the normalised title."""
    _get_user_scenario_or_404(scenario_id)
    name, data = await _read_file(file)
    try:
        asked = json.loads(mapping) if mapping else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="mapping must be a JSON object {field: column}") from None
    prepared = await run_in_threadpool(_prepare, scenario_id, name, data, asked, overwrite, add_missing, False)
    note = f"Imported from {name[:120]}"
    done = await run_in_threadpool(apply_import, scenario_id, prepared["plan"], name, note)
    return {**done, "mapping": prepared["mapping"], "kind": prepared["kind"],
            "conflicts": [r for r in prepared["plan"] if r["action"] == "conflict"][:200]}


@app.get("/user-scenarios/{scenario_id}/decisions/imports")
def list_decision_imports(scenario_id: str, limit: int = 20) -> dict[str, Any]:
    """The imports this scenario has had, newest first, and whether each was undone."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, filename, created_at, undone_at, report
            FROM scenario_decision_import WHERE scenario_id = :sid
            ORDER BY id DESC LIMIT :n
        """), {"sid": scenario_id, "n": max(1, min(int(limit), 200))}).mappings().all()
    return {"scenario_id": scenario_id, "imports": [
        {"id": r["id"], "filename": r["filename"],
         "created_at": r["created_at"].isoformat() if r["created_at"] else None,
         "undone_at": r["undone_at"].isoformat() if r["undone_at"] else None,
         "report": r["report"]} for r in rows]}


@app.post("/user-scenarios/{scenario_id}/decisions/undo")
def undo_decision_import(scenario_id: str, import_id: int | None = None,
                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Put back what an import changed: the last one, or the one named.

    Each article returns to the status, reason and notes it carried before. An article the
    import ADDED to the corpus keeps its place and loses only its inclusion: removing a
    paper from a corpus is not the reverse of deciding about it."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT id, undo, undone_at FROM scenario_decision_import
            WHERE scenario_id = :sid {"AND id = :iid" if import_id else "AND undone_at IS NULL"}
            ORDER BY id DESC LIMIT 1
        """), {"sid": scenario_id, **({"iid": int(import_id)} if import_id else {})}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="No import to undo for this scenario.")
    if row["undone_at"]:
        raise HTTPException(status_code=409, detail="That import has already been undone.")
    entries = row["undo"] if isinstance(row["undo"], list) else json.loads(row["undo"] or "[]")
    with engine.begin() as conn:
        for doc_id, status, reason, notes in entries:
            _write_ars_screening(conn, scenario_id, int(doc_id), status, reason, notes)
        conn.execute(text("UPDATE scenario_decision_import SET undone_at = NOW() WHERE id = :i"),
                     {"i": int(row["id"])})
    if entries:
        schedule_retrain(scenario_id)
    return {"scenario_id": scenario_id, "import_id": int(row["id"]), "restored": len(entries)}
