"""Exports of a scenario's relevant articles: CSV, Excel, RIS, BibTeX, JSON, Markdown.

The relevant articles are the same set every tab uses (`_get_above_threshold_articles`:
included by a reviewer, or scored above the scenario threshold, never the excluded).
The formatters are pure functions over plain dicts, tested offline; the endpoint adds
the bibliographic fields the relevance query does not carry (PMID, source, URL, keywords).
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from fastapi import HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import text

from .core import app, engine, logger
from .scenario_store import _get_user_scenario_or_404
from .relevance import _get_above_threshold_articles

EXPORT_FORMATS = ("csv", "xlsx", "ris", "bibtex", "json", "md")
_CONTENT_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ris": "application/x-research-info-systems; charset=utf-8",
    "bibtex": "application/x-bibtex; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}
_EXTENSIONS = {"csv": "csv", "xlsx": "xlsx", "ris": "ris", "bibtex": "bib", "json": "json", "md": "md"}

# One row per article, the same columns in every tabular format.
EXPORT_COLUMNS = (
    "rank", "id", "title", "authors", "year", "journal", "doi", "pmid", "url", "source",
    "study_design", "similarity_score", "quality_score", "citation_count", "screening_status",
    "keywords", "pico_population", "pico_intervention", "pico_comparator", "pico_outcome", "abstract",
)


def _pico(a: dict, key: str) -> str:
    pj = a.get("pico_json")
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except Exception:
            pj = None
    return str((pj or {}).get(key) or "") if isinstance(pj, dict) else ""


def _article_url(a: dict) -> str:
    if a.get("url"):
        return str(a["url"])
    if a.get("doi"):
        return f"https://doi.org/{a['doi']}"
    if a.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"
    return ""


def export_rows(articles: list[dict], include_abstract: bool = True) -> list[dict]:
    """Plain export rows (strings and numbers only), in relevance order. Pure."""
    rows = []
    for i, a in enumerate(articles, 1):
        rows.append({
            "rank": i,
            "id": a.get("id"),
            "title": (a.get("title") or "").strip(),
            "authors": (a.get("authors") or "").strip(),
            "year": a.get("year"),
            "journal": (a.get("journal") or "").strip(),
            "doi": (a.get("doi") or "").strip(),
            "pmid": str(a.get("pmid") or "").strip(),
            "url": _article_url(a),
            "source": (a.get("source") or "").strip(),
            "study_design": (a.get("study_design") or _pico(a, "study_design") or "").strip(),
            "similarity_score": (round(float(a["similarity_score"]), 4)
                                 if a.get("similarity_score") is not None else None),
            "quality_score": (round(float(a["quality_score"]), 3)
                              if a.get("quality_score") is not None else None),
            "citation_count": a.get("citation_count"),
            "screening_status": a.get("screening_status") or "",
            "keywords": (a.get("keywords") or "").strip() if isinstance(a.get("keywords"), str)
                        else "; ".join(str(k) for k in (a.get("keywords") or [])),
            "pico_population": _pico(a, "P"),
            "pico_intervention": _pico(a, "I"),
            "pico_comparator": _pico(a, "C"),
            "pico_outcome": _pico(a, "O"),
            "abstract": (a.get("abstract") or "").strip() if include_abstract else "",
        })
    return rows


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(EXPORT_COLUMNS), extrasaction="ignore", lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in EXPORT_COLUMNS})
    return "﻿" + buf.getvalue()          # BOM: Excel opens UTF-8 accents correctly


def to_json(rows: list[dict], meta: dict | None = None) -> str:
    return json.dumps({"meta": meta or {}, "articles": rows}, ensure_ascii=False, indent=2, default=str)


def _authors_list(authors: str) -> list[str]:
    return [p.strip() for p in re.split(r"[;\n]|,\s(?=[A-Z][a-z]+\s[A-Z])", authors or "") if p.strip()]


def to_ris(rows: list[dict]) -> str:
    """RIS for Zotero, EndNote, Mendeley: one record per article, JOUR type."""
    out = []
    for r in rows:
        lines = ["TY  - JOUR", f"TI  - {r['title']}"]
        for au in _authors_list(r["authors"]):
            lines.append(f"AU  - {au}")
        if r.get("year"):
            lines.append(f"PY  - {r['year']}")
        if r.get("journal"):
            lines.append(f"JO  - {r['journal']}")
        if r.get("doi"):
            lines.append(f"DO  - {r['doi']}")
        if r.get("pmid"):
            lines.append(f"AN  - {r['pmid']}")
        if r.get("url"):
            lines.append(f"UR  - {r['url']}")
        for kw in [k.strip() for k in (r.get("keywords") or "").split(";") if k.strip()][:20]:
            lines.append(f"KW  - {kw}")
        if r.get("abstract"):
            _abstract_one_line = re.sub(r"\s+", " ", r["abstract"])
            lines.append(f"AB  - {_abstract_one_line}")
        notes = []
        if r.get("similarity_score") is not None:
            notes.append(f"relevance {r['similarity_score']}")
        if r.get("study_design"):
            notes.append(f"design {r['study_design']}")
        if notes:
            lines.append(f"N1  - LiteRev: {', '.join(notes)}")
        lines.append("ER  - ")
        out.append("\r\n".join(lines))
    return "\r\n".join(out) + ("\r\n" if out else "")


def _bib_escape(s: str) -> str:
    return (s or "").replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")


def _bib_key(r: dict) -> str:
    first = (_authors_list(r.get("authors") or "") or ["anon"])[0].split()[0]
    first = re.sub(r"[^A-Za-z]", "", first) or "anon"
    return f"{first.lower()}{r.get('year') or 'nd'}_{r.get('id')}"


def to_bibtex(rows: list[dict]) -> str:
    out = []
    for r in rows:
        fields = [("title", r["title"]), ("author", " and ".join(_authors_list(r["authors"]))),
                  ("year", str(r.get("year") or "")), ("journal", r.get("journal") or ""),
                  ("doi", r.get("doi") or ""), ("url", r.get("url") or ""),
                  ("keywords", r.get("keywords") or ""), ("abstract", r.get("abstract") or ""),
                  ("note", f"LiteRev relevance {r['similarity_score']}" if r.get("similarity_score") is not None else "")]
        body = ",\n".join(f"  {k} = {{{_bib_escape(v)}}}" for k, v in fields if v)
        out.append(f"@article{{{_bib_key(r)},\n{body}\n}}")
    return "\n\n".join(out) + ("\n" if out else "")


def to_markdown(rows: list[dict], title: str = "") -> str:
    lines = [f"# {title}".rstrip(), ""] if title else []
    lines.append(f"{len(rows)} relevant articles, most relevant first.")
    lines.append("")
    for r in rows:
        head = f"{r['rank']}. **{r['title']}**"
        meta = ", ".join(x for x in (r.get("authors"), str(r.get("year") or ""), r.get("journal")) if x)
        link = f" [{r['doi'] or 'link'}]({r['url']})" if r.get("url") else ""
        lines.append(f"{head}{link}")
        if meta:
            lines.append(f"   {meta}")
        extras = []
        if r.get("similarity_score") is not None:
            extras.append(f"relevance {r['similarity_score']}")
        if r.get("study_design"):
            extras.append(r["study_design"])
        if extras:
            lines.append(f"   _{' · '.join(extras)}_")
        lines.append("")
    return "\n".join(lines)


def to_xlsx(rows: list[dict], sheet_title: str = "Relevant articles") -> bytes:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_title[:31] or "Articles"
    ws.append(list(EXPORT_COLUMNS))
    for r in rows:
        ws.append([r.get(k) for k in EXPORT_COLUMNS])
    widths = {"title": 60, "authors": 40, "journal": 28, "abstract": 80, "url": 36, "doi": 24,
              "pico_population": 30, "pico_intervention": 30, "pico_comparator": 24, "pico_outcome": 30}
    for i, col in enumerate(EXPORT_COLUMNS, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 12)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_export(fmt: str, rows: list[dict], title: str, meta: dict | None = None) -> bytes:
    """Bytes of the export in `fmt`. Pure (openpyxl for xlsx)."""
    if fmt == "csv":
        return to_csv(rows).encode("utf-8")
    if fmt == "json":
        return to_json(rows, meta).encode("utf-8")
    if fmt == "ris":
        return to_ris(rows).encode("utf-8")
    if fmt == "bibtex":
        return to_bibtex(rows).encode("utf-8")
    if fmt == "md":
        return to_markdown(rows, title).encode("utf-8")
    if fmt == "xlsx":
        return to_xlsx(rows, title)
    raise ValueError(fmt)


def _export_extra_fields(ids: list[int]) -> dict[int, dict]:
    """PMID, source, URL and keywords of the articles: the relevance query does not carry them."""
    if not ids:
        return {}
    out: dict[int, dict] = {}
    try:
        with engine.connect() as conn:
            for r in conn.execute(text(
                "SELECT id, pmid, source, url, keywords FROM literature_document WHERE id = ANY(:ids)"
            ), {"ids": [int(i) for i in ids]}).mappings():
                out[int(r["id"])] = dict(r)
    except Exception as _e:                                  # a database without `url`, say
        logger.warning(f"export extra fields: {_e}")
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (s or "").strip()).strip("-").lower()
    return s[:60] or "scenario"


def relevant_articles_export(scenario_id: str, fmt: str, threshold: float | None,
                             include_abstract: bool) -> Response:
    fmt = (fmt or "csv").lower()
    if fmt == "bib":
        fmt = "bibtex"
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {', '.join(EXPORT_FORMATS)}")
    row = _get_user_scenario_or_404(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold=threshold)
    extra = _export_extra_fields([a["id"] for a in articles])
    for a in articles:
        a.update({k: v for k, v in (extra.get(int(a["id"])) or {}).items() if k != "id"})
    rows = export_rows(articles, include_abstract=include_abstract)
    title = str(row.get("name") or scenario_id)
    meta = {"scenario_id": scenario_id, "scenario": title, "query": row.get("query"),
            "n_articles": len(rows), "format": fmt}
    body = render_export(fmt, rows, title, meta)
    filename = f"{_slug(title)}_relevant-articles_{len(rows)}.{_EXTENSIONS[fmt]}"
    return Response(content=body, media_type=_CONTENT_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "X-Article-Count": str(len(rows))})


@app.get("/user-scenarios/{scenario_id}/relevant/export")
def export_user_scenario_relevant(
    scenario_id: str,
    format: str = Query("csv"),
    threshold: float | None = None,
    include_abstract: bool = True,
) -> Response:
    """The relevant articles of a scenario as a file: csv, xlsx, ris (Zotero, EndNote,
    Mendeley), bibtex, json or md. Same set and same order as the Corpus tab."""
    return relevant_articles_export(scenario_id, format, threshold, include_abstract)


@app.get("/gesica/scenarios/{scenario_id}/relevant/export")
def export_gesica_scenario_relevant(
    scenario_id: str,
    format: str = Query("csv"),
    threshold: float | None = None,
    include_abstract: bool = True,
) -> Response:
    """Same export for the built-in scenarios."""
    return relevant_articles_export(scenario_id, format, threshold, include_abstract)
