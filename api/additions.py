"""Papers added by hand to a scenario: files uploaded by a reviewer.

A search reaches the databases LiteRev can query. The papers it cannot reach still exist:
the ISS team's HPAI review holds records that only Embase indexes, with no PubMed
identifier, which no query of ours will ever return. A reviewer also simply has papers
on their desk. This module lets them put those papers into a scenario's corpus.

What a file can be:

- a PDF: the text is read (pdftotext, else pypdf), the title, abstract, DOI, PMID and
  year are guessed from the first pages, and when a DOI is found the record is completed
  from Crossref; the whole text is kept as full-text sections for the assistant and the
  extraction;
- a reference file: RIS, BibTeX, PubMed's nbib, or a CSV or Excel table with title,
  abstract, DOI, PMID, year, authors and journal columns.

What happens to a record: it goes through the same ingestion as a search result
(`_ingest_doc_direct`: one row per work, matched on PMID, DOI or title), is linked to the
scenario, and is remembered in `scenario_additions`. That memory is what makes the
addition last: a later search deletes the links it no longer matches, except those a
reviewer decided on and, now, those added by hand (`_set_scenario_corpus`), and the PRISMA
card counts them on their own line. The new links wait for a relevance score like any
other article and the cosine scoring is started at once; `include=true` marks them as
included by hand, which puts them in the relevant set whatever their score.

Pure and tested without a database: the parsers of each format, the metadata guessed
from a PDF's text, the kind of a file. With a database: the whole endpoint.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .documents import _normalize_doi, sanitize_db_text
from .double_blind import _write_ars_screening
from .learning import schedule_retrain
from .pipeline import _chunk_text_ft, _insert_fulltext_chunks_ft
from .relevance import score_new_links
from .scenario_store import _get_user_scenario_or_404
from .sources import _ingest_doc_direct

#: Files in one request, and bytes per file. A review's reference export is one file of
#: a few megabytes; a batch of PDFs is a few dozen files of a few megabytes each.
MAX_FILES = 50
MAX_FILE_BYTES = 30 * 1024 * 1024
#: Records read from one reference file: the request is synchronous, and ingesting a record
#: is a few round trips to the database.
MAX_RECORDS_PER_FILE = 5000
#: pdftotext on one file.
PDF_TIMEOUT_S = 90
#: Pages read to guess a PDF's metadata (the whole file is read for the full text).
PDF_METADATA_PAGES = 2
#: Crossref, when a PDF carries a DOI: one short call, never a reason to fail the upload.
CROSSREF_TIMEOUT_S = 8
USER_AGENT = "LiteRev/1.0 (mailto:api@literev.app)"

KINDS = ("pdf", "ris", "bibtex", "nbib", "csv", "xlsx")
#: What the screen offers; `.txt` is sniffed (an nbib, a RIS or a BibTeX file saved as text).
ACCEPTED_EXTENSIONS = (".pdf", ".ris", ".bib", ".bibtex", ".nbib", ".txt", ".csv", ".tsv", ".xlsx")


def _ensure_additions_table() -> None:
    """One row per (scenario, article) added by hand, with the file it came from: what a
    later search must keep, and what the PRISMA card counts."""
    from .schema_boot import _exec_ddl_isolated
    _exec_ddl_isolated([
        """CREATE TABLE IF NOT EXISTS scenario_additions (
               scenario_id TEXT   NOT NULL,
               document_id BIGINT NOT NULL,
               filename    TEXT,
               added_at    TIMESTAMP DEFAULT NOW(),
               PRIMARY KEY (scenario_id, document_id)
           )""",
    ], "_ensure_additions_table")


try:
    _ensure_additions_table()
except Exception as _e:  # pragma: no cover - a database without the base tables
    logger.warning(f"_ensure_additions_table: {_e}")


# ── Pure: the kind of a file ───────────────────────────────────────────────────
def detect_kind(filename: str, head: bytes) -> str | None:
    """The format of an uploaded file, from its content first (a PDF is a PDF whatever
    its name), then its extension, then a sniff of the text for the formats that share
    the `.txt` extension. None when nothing recognises it."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "xlsx"
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".ris":
        return "ris"
    if ext in (".bib", ".bibtex"):
        return "bibtex"
    if ext == ".nbib":
        return "nbib"
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext == ".xlsx":
        return "xlsx"
    sample = head[:4000].decode("utf-8", "ignore").lstrip("﻿")
    if re.search(r"^PMID- ", sample, re.M):
        return "nbib"
    if re.search(r"^TY  - ", sample, re.M):
        return "ris"
    if re.search(r"^\s*@\w+\s*\{", sample, re.M):
        return "bibtex"
    first = sample.splitlines()[0].lower() if sample.strip() else ""
    if "title" in first and any(sep in first for sep in (",", ";", "\t")):
        return "csv"
    return None


# ── Pure: one record ──────────────────────────────────────────────────────────
def _clean(s: Any) -> str | None:
    if s is None:
        return None
    s = " ".join(str(s).split())
    return s or None


def year_of(value: Any) -> int | None:
    """The year in a date field ("2024", "2024/03/01", "2024 Jan 5", "Y1  - 2019///")."""
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", str(value or ""))
    return int(m.group(1)) if m else None


def pmid_of(value: Any) -> str | None:
    m = re.search(r"\b(\d{1,9})\b", str(value or ""))
    return m.group(1) if m else None


def make_record(title: Any, abstract: Any = None, doi: Any = None, pmid: Any = None, year: Any = None,
                authors: Any = None, journal: Any = None, url: Any = None, fulltext: str | None = None) -> dict | None:
    """A record the ingestion accepts, or None when it has no usable title."""
    t = _clean(title)
    if not t or len(t) < 5:
        return None
    y = year if isinstance(year, int) else year_of(year)
    if y is not None and not (1800 <= y <= datetime.now(timezone.utc).year + 1):
        y = None
    d = _normalize_doi(_clean(doi)) if doi else None
    if d and not re.match(r"^10\.\d{4,9}/\S+$", d):
        d = None
    p = pmid_of(pmid) if pmid else None
    u = _clean(url)
    if not u:
        u = f"https://doi.org/{d}" if d else (f"https://pubmed.ncbi.nlm.nih.gov/{p}/" if p else None)
    a = _clean(abstract)
    return {"title": t[:500], "abstract": a, "doi": d, "pmid": p, "year": y,
            "authors": (_clean(authors) or None), "journal": _clean(journal), "url": u,
            "fulltext": fulltext}


# ── Pure: the parsers ─────────────────────────────────────────────────────────
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "ignore")


def parse_ris(text_: str) -> list[dict]:
    """RIS: `TAG  - value` lines, records closed by `ER  -`; a line without a tag
    continues the previous value. Title TI/T1, abstract AB/N2, DOI DO, year PY/Y1/DA,
    authors AU/A1, journal JO/JF/T2."""
    recs: list[dict[str, list[str]]] = []
    cur: dict[str, list[str]] = {}
    last: str | None = None
    for line in text_.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  -\s?(.*)$", line)
        if not m:
            if cur and last and line.strip():
                cur[last][-1] = (cur[last][-1] + " " + line.strip()).strip()
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            if cur:
                recs.append(cur)
            cur, last = {}, None
            continue
        cur.setdefault(tag, []).append(val)
        last = tag
    if cur:
        recs.append(cur)
    out = []
    for r in recs:
        def first(*tags: str) -> str | None:
            for tg in tags:
                for v in r.get(tg) or []:
                    if v:
                        return v
            return None
        authors = ", ".join(v for v in (r.get("AU") or r.get("A1") or []) if v)
        rec = make_record(first("TI", "T1", "CT"), first("AB", "N2"), first("DO"), first("PM"),
                          first("PY", "Y1", "DA"), authors, first("JO", "JF", "T2", "J2", "JA"), first("UR"))
        if rec:
            out.append(rec)
    return out


def _bib_fields(body: str) -> dict[str, str]:
    """The `name = value` pairs of one BibTeX entry body (after the key), values in
    braces (nested), in quotes, or bare."""
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*").match(body, i)
        if not m:
            i += 1
            continue
        name, i = m.group(1).lower(), m.end()
        if i >= n:
            break
        if body[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += 1 if body[j] == "{" else (-1 if body[j] == "}" else 0)
                j += 1
            raw, i = body[i + 1:j - 1], j
        elif body[i] == '"':
            j = i + 1
            while j < n and body[j] != '"':
                j += 1
            raw, i = body[i + 1:j], j + 1
        else:
            m2 = re.compile(r"[^,\n}]*").match(body, i)
            raw, i = (m2.group(0) if m2 else ""), (m2.end() if m2 else i + 1)
        val = re.sub(r"[{}]", "", raw)
        val = val.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
        fields[name] = " ".join(val.split())
    return fields


def parse_bibtex(text_: str) -> list[dict]:
    """BibTeX entries: title, abstract, doi, year, author ("and" separated), journal or
    booktitle, pmid, url."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{", text_):
        kind = m.group(1).lower()
        depth, i = 1, m.end()
        while i < len(text_) and depth:
            depth += 1 if text_[i] == "{" else (-1 if text_[i] == "}" else 0)
            i += 1
        if kind in ("comment", "preamble", "string"):
            continue
        body = text_[m.end():i - 1]
        body = body.split(",", 1)[1] if "," in body else ""
        f = _bib_fields(body)
        authors = ", ".join(a.strip() for a in re.split(r"\s+and\s+", f.get("author", "")) if a.strip())
        rec = make_record(f.get("title"), f.get("abstract"), f.get("doi"), f.get("pmid") or f.get("pubmed"),
                          f.get("year") or f.get("date"), authors,
                          f.get("journal") or f.get("journaltitle") or f.get("booktitle"), f.get("url"))
        if rec:
            out.append(rec)
    return out


def parse_nbib(text_: str) -> list[dict]:
    """PubMed's MEDLINE/nbib format: `TAG - value` lines, continuation lines indented by
    six spaces, records separated by a blank line. PMID, TI, AB, DP (year), AU, JT, and
    the DOI in an AID or LID line ending with [doi]."""
    recs: list[dict[str, list[str]]] = []
    cur: dict[str, list[str]] = {}
    last: str | None = None
    for line in text_.splitlines():
        if not line.strip():
            if cur:
                recs.append(cur)
            cur, last = {}, None
            continue
        m = re.match(r"^([A-Z]{1,4})\s*- (.*)$", line)
        if m:
            tag, val = m.group(1), m.group(2).strip()
            cur.setdefault(tag, []).append(val)
            last = tag
        elif line.startswith("      ") and last and cur.get(last):
            cur[last][-1] = (cur[last][-1] + " " + line.strip()).strip()
    if cur:
        recs.append(cur)
    out = []
    for r in recs:
        doi = next((v[:-6].strip() for v in (r.get("AID") or []) + (r.get("LID") or [])
                    if v.lower().endswith("[doi]")), None)
        rec = make_record(" ".join(r.get("TI") or []), " ".join(r.get("AB") or []), doi,
                          (r.get("PMID") or [None])[0], (r.get("DP") or [None])[0],
                          ", ".join(r.get("AU") or r.get("FAU") or []), (r.get("JT") or r.get("TA") or [None])[0])
        if rec:
            out.append(rec)
    return out


#: Column names a table may use for each field, lower case, punctuation and spaces folded.
TABLE_COLUMNS = {
    "title": ("title", "articletitle", "documenttitle", "ti", "primarytitle"),
    "abstract": ("abstract", "abstractnote", "ab", "summary"),
    "doi": ("doi", "digitalobjectidentifier"),
    "pmid": ("pmid", "pubmedid", "pubmed", "pmcid"),
    "year": ("year", "publicationyear", "pubyear", "py", "date", "publicationdate"),
    "authors": ("authors", "author", "authornames", "au", "creator"),
    "journal": ("journal", "sourcetitle", "publicationtitle", "journalbook", "journaltitle", "venue"),
    "url": ("url", "link", "urls"),
}


def _fold_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def map_columns(header: list) -> dict[str, Any]:
    """Which column of the table carries each field: the first header that matches a
    known name, by folded comparison."""
    folded = [(_fold_name(h), h) for h in header]
    out: dict[str, Any] = {}
    for field, names in TABLE_COLUMNS.items():
        for fname, h in folded:
            if fname in names and field not in out:
                out[field] = h
    return out


def parse_table_rows(rows: list[dict], header: list) -> list[dict]:
    cols = map_columns(header)
    if "title" not in cols:
        raise ValueError("no title column (expected a header such as Title, Abstract, DOI, PMID, Year)")
    out = []
    for r in rows:
        rec = make_record(r.get(cols["title"]), r.get(cols.get("abstract")), r.get(cols.get("doi")),
                          r.get(cols.get("pmid")), r.get(cols.get("year")), r.get(cols.get("authors")),
                          r.get(cols.get("journal")), r.get(cols.get("url")))
        if rec:
            out.append(rec)
    return out


def parse_csv(data: bytes) -> list[dict]:
    """A CSV or TSV table, the delimiter read from the header line."""
    text_ = _decode(data)
    first = text_.splitlines()[0] if text_.strip() else ""
    delim = max(",;\t|", key=first.count) if first else ","
    reader = csv.DictReader(io.StringIO(text_), delimiter=delim)
    rows = [{k: v for k, v in r.items() if k is not None} for r in reader]
    return parse_table_rows(rows, list(reader.fieldnames or []))


def parse_xlsx(data: bytes) -> list[dict]:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(data), dtype=str)
    df = df.where(df.notna(), None)
    return parse_table_rows(df.to_dict(orient="records"), [str(c) for c in df.columns])


# ── Pure: what a PDF's first pages say ────────────────────────────────────────
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>()\[\]]+)", re.I)
_PMID_RE = re.compile(r"\bPMID:?\s*(\d{6,9})\b", re.I)
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
#: A line of a first page that is not a title: a running head, a licence, an address.
_NOT_A_TITLE = re.compile(
    r"(doi|©|\(c\)|copyright|received|accepted|published|available online|www\.|https?:|issn|"
    r"e-?mail|corresponding author|citation:|cite this|open access|creative commons|licen[cs]e|"
    r"volume|\bvol\.|journal of|all rights reserved|downloaded from|elsevier|springer|wiley|"
    r"preprint|peer.?review|research article|original article|review article|short communication|"
    r"^\d+$|@)", re.I)
_ABSTRACT_END = re.compile(r"\n\s*(key ?words?|introduction|background|1\.?\s+introduction|1\.?\s+background|"
                           r"citation|©|copyright)\b", re.I)
#: A digit stuck to the end of a name, as an affiliation marker: "Rossi1, Bianchi2". It
#: never matches a subtype inside a title ("H5N8", "H7N9"), whose digits follow a capital.
_AFFILIATION_MARK = re.compile(r"[a-z]\d(?=[,;\s]|$)")


def _looks_like_title_line(line: str) -> bool:
    """A line that can belong to a title: no running-head words, mostly letters."""
    if not line or _NOT_A_TITLE.search(line):
        return False
    ok = sum(c.isalpha() or c in " -,:;'()" for c in line)
    return ok / max(1, len(line)) >= 0.75


def _looks_like_authors(line: str) -> bool:
    """An author line: affiliation digits on the names, or three or more short pieces
    separated by commas ("Maria Rossi, Luca Bianchi, Anna Verdi")."""
    if _AFFILIATION_MARK.search(line):
        return True
    parts = [p.strip() for p in line.split(",") if p.strip()]
    return len(parts) >= 3 and all(len(p.split()) <= 4 for p in parts)


def guess_pdf_metadata(text_: str) -> dict[str, Any]:
    """Title, abstract, DOI, PMID and year read from the text of a PDF's first pages.

    Heuristics, and said as such: the title is the first group of 4 to 60 words that
    looks like one, where a group ends at a blank line, a running head, an author line or
    a finished sentence (pdftotext does not always leave a blank line between the blocks
    of a page, so grouping cannot rely on them); the abstract is what follows the word
    "Abstract" up to the keywords or the introduction, else the first long paragraph; the
    year is the most frequent plausible year of the first page, the latest on a tie. A
    DOI, when the page carries one, lets Crossref replace the guesses."""
    lines = [ln.strip() for ln in (text_ or "").splitlines()]
    blocks: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    start = 0
    for i, ln in enumerate(lines):
        keep = bool(ln) and _looks_like_title_line(ln) and not (cur and _looks_like_authors(ln))
        if not keep:
            if cur:
                blocks.append((start, cur))
                cur = []
            continue
        if not cur:
            start = i
        cur.append(ln)
        if ln.endswith((".", ":", "?", "!")):            # a finished sentence ends the group
            blocks.append((start, cur))
            cur = []
    if cur:
        blocks.append((start, cur))
    title, title_end = None, None
    for first, b in blocks[:25]:
        joined = " ".join(b)
        words = joined.split()
        if not 4 <= len(words) <= 60 or len(joined) > 400:
            continue
        if _NOT_A_TITLE.search(joined):
            continue
        if sum(c.isalpha() or c in " -,:;'()" for c in joined) / max(1, len(joined)) < 0.75:
            continue
        title, title_end = joined.rstrip(" .;,"), first + len(b)
        break
    abstract = None
    m = re.search(r"\babstract\b[:.\s]*", text_ or "", re.I)
    if m:
        rest = (text_ or "")[m.end():]
        end = _ABSTRACT_END.search(rest)
        cand = " ".join((rest[:end.start()] if end else rest[:3000]).split())
        if len(cand) >= 150:
            abstract = cand[:4000]
    if abstract is None:
        after = title_end or 0
        for first, b in [bl for bl in blocks if bl[0] >= after][:12]:
            joined = " ".join(b)
            if len(joined) >= 400 and not _NOT_A_TITLE.search(joined[:80]):
                abstract = joined[:4000]
                break
    first_page = (text_ or "")[:6000]
    doi = None
    md = _DOI_RE.search(first_page) or _DOI_RE.search(text_ or "")
    if md:
        doi = _normalize_doi(md.group(1).rstrip(".,;:"))
    mp = _PMID_RE.search(text_ or "")
    years = Counter(int(y) for y in _YEAR_RE.findall(first_page))
    year = None
    if years:
        top = max(years.values())
        year = max(y for y, n in years.items() if n == top)
    return {"title": title, "abstract": abstract, "doi": doi, "pmid": mp.group(1) if mp else None, "year": year}


# ── Reading a PDF, asking Crossref ────────────────────────────────────────────
class PdfReaderMissing(RuntimeError):
    """Neither pdftotext nor pypdf is available on this server."""


def pdf_text(data: bytes, first_pages: int | None = None) -> str:
    """The text of a PDF: pdftotext (poppler), else pypdf when installed. Raises
    PdfReaderMissing when neither can run, ValueError when the file cannot be read."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        cmd = ["pdftotext"] + (["-l", str(first_pages)] if first_pages else []) + [path, "-"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=PDF_TIMEOUT_S)
        except FileNotFoundError:
            res = None
        if res is not None:
            if res.returncode != 0:
                raise ValueError((res.stderr or "pdftotext failed").strip()[:200])
            return sanitize_db_text(res.stdout) or ""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise PdfReaderMissing("pdftotext (poppler-utils) and pypdf are both missing") from e
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:first_pages] if first_pages else reader.pages
        return sanitize_db_text("\n\n".join((p.extract_text() or "") for p in pages)) or ""
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"pdftotext took more than {PDF_TIMEOUT_S} s") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def crossref_by_doi(doi: str) -> dict[str, Any] | None:
    """The record Crossref holds for a DOI: title, abstract (JATS tags removed), year,
    authors, journal. None when Crossref does not answer or does not know it."""
    import requests
    from urllib.parse import quote
    try:
        r = requests.get(f"https://api.crossref.org/works/{quote(doi, safe='')}",
                         headers={"User-Agent": USER_AGENT}, timeout=CROSSREF_TIMEOUT_S)
        if r.status_code != 200:
            return None
        m = r.json().get("message") or {}
    except Exception as e:                                   # noqa: BLE001 - never fatal
        logger.info(f"crossref_by_doi {doi}: {e}")
        return None
    year = None
    for key in ("issued", "published-print", "published-online", "created"):
        parts = ((m.get(key) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = int(parts[0])
            break
    abstract = re.sub(r"<[^>]+>", " ", m.get("abstract") or "")
    return {
        "title": _clean((m.get("title") or [None])[0]),
        "abstract": _clean(abstract),
        "year": year,
        "authors": ", ".join(_clean(f"{a.get('family', '')} {a.get('given', '')}") or ""
                             for a in (m.get("author") or [])[:30]) or None,
        "journal": _clean((m.get("container-title") or [None])[0]),
    }


def records_from_pdf(data: bytes) -> list[dict]:
    """The one record of a PDF: its metadata guessed from the first pages, completed from
    Crossref when a DOI was found, and its whole text as full text."""
    full = pdf_text(data)
    head = pdf_text(data, first_pages=PDF_METADATA_PAGES) if len(full) > 20000 else full
    guess = guess_pdf_metadata(head)
    meta = dict(guess)
    meta["crossref"] = False
    if guess.get("doi"):
        found = crossref_by_doi(guess["doi"])
        if found and found.get("title"):
            meta.update({k: v for k, v in found.items() if v})
            meta["crossref"] = True
    if not meta.get("title"):
        return []
    if not meta.get("abstract") and full:
        meta["abstract"] = " ".join(full.split())[:1500]
    rec = make_record(meta["title"], meta.get("abstract"), meta.get("doi"), meta.get("pmid"), meta.get("year"),
                      meta.get("authors"), meta.get("journal"), None, fulltext=full if len(full) > 500 else None)
    if rec:
        rec["metadata_from"] = "crossref" if meta["crossref"] else "text"
    return [rec] if rec else []


def records_from_file(kind: str, data: bytes) -> list[dict]:
    if kind == "pdf":
        return records_from_pdf(data)
    if kind == "ris":
        return parse_ris(_decode(data))
    if kind == "bibtex":
        return parse_bibtex(_decode(data))
    if kind == "nbib":
        return parse_nbib(_decode(data))
    if kind == "csv":
        return parse_csv(data)
    if kind == "xlsx":
        return parse_xlsx(data)
    raise ValueError(f"unsupported kind {kind}")


# ── Into the scenario ─────────────────────────────────────────────────────────
def add_records(scenario_id: str, records: list[dict], filename: str, include: bool) -> list[dict]:
    """Ingest each record (one row per work), link it to the scenario, remember the
    addition, include it by hand when asked, keep a PDF's text. Returns one report per
    record."""
    digest = hashlib.sha1(f"{scenario_id}|{filename}".encode()).hexdigest()[:12]
    out = []
    for i, rec in enumerate(records):
        pmid, doi = rec.get("pmid"), rec.get("doi")
        ext_id = f"pmid:{pmid}" if pmid else (doi if doi else f"upload:{digest}:{i}")
        try:
            doc_id, is_new = _ingest_doc_direct(
                source="upload", title=rec["title"], abstract=rec.get("abstract"), year=rec.get("year"),
                url=rec.get("url"), external_id=ext_id, doi=doi, authors=rec.get("authors"),
                journal=rec.get("journal"))
        except Exception as e:                               # noqa: BLE001 - one record, said
            logger.warning(f"add_records {scenario_id}: {e}")
            out.append({"title": rec["title"], "status": "failed", "error": str(e)[:200], "file": filename})
            continue
        if doc_id is None:
            out.append({"title": rec["title"], "status": "failed", "error": "not ingested", "file": filename})
            continue
        included, kept_decision, fulltext_stored = False, False, False
        with engine.begin() as c:
            linked = c.execute(text("""
                INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
                VALUES (:d, :s, NULL)
                ON CONFLICT (document_id, scenario_id) DO NOTHING
                RETURNING document_id
            """), {"d": doc_id, "s": scenario_id}).scalar()
            c.execute(text("""
                INSERT INTO scenario_additions (scenario_id, document_id, filename)
                VALUES (:s, :d, :f)
                ON CONFLICT (scenario_id, document_id) DO UPDATE
                    SET filename = EXCLUDED.filename, added_at = NOW()
            """), {"s": scenario_id, "d": doc_id, "f": filename[:200]})
            if include:
                status = c.execute(text("SELECT screening_status FROM article_scenarios "
                                        "WHERE scenario_id = :s AND document_id = :d"),
                                   {"s": scenario_id, "d": doc_id}).scalar()
                if status in ("included", "excluded"):
                    kept_decision = True
                else:
                    _write_ars_screening(c, scenario_id, doc_id, "included", None,
                                         f"Added by hand from {filename[:120]}")
                    included = True
        if rec.get("fulltext"):
            try:
                with engine.connect() as c:
                    has_ft = c.execute(text("SELECT has_fulltext FROM literature_document WHERE id = :d"),
                                       {"d": doc_id}).scalar()
                if is_new or not has_ft:
                    chunks = _chunk_text_ft(rec["fulltext"])
                    if chunks:
                        _insert_fulltext_chunks_ft(doc_id, chunks, "upload")
                        fulltext_stored = True
            except Exception as e:                           # noqa: BLE001 - the record is in
                logger.warning(f"add_records {scenario_id} full text of {doc_id}: {e}")
        out.append({"id": int(doc_id), "title": rec["title"], "year": rec.get("year"), "doi": doi, "pmid": pmid,
                    "status": "added" if linked is not None else "already_in_corpus",
                    "new_document": bool(is_new), "included": included, "kept_decision": kept_decision,
                    "fulltext_stored": fulltext_stored, "metadata_from": rec.get("metadata_from"),
                    "file": filename})
    return out


def _process_upload(scenario_id: str, files: list[tuple[str, bytes]], include: bool) -> dict[str, Any]:
    reports, articles = [], []
    for name, data in files:
        kind = detect_kind(name, data[:4096])
        rep: dict[str, Any] = {"name": name, "kind": kind, "records": 0, "added": 0, "already_in_corpus": 0,
                               "skipped": 0, "error": None}
        if kind is None:
            rep["error"] = "format not recognised (PDF, RIS, BibTeX, nbib, CSV, Excel .xlsx)"
            reports.append(rep)
            continue
        try:
            records = records_from_file(kind, data)
        except PdfReaderMissing as e:
            raise HTTPException(status_code=503, detail=f"PDF reading is not available on this server: {e}") from e
        except Exception as e:                               # noqa: BLE001 - one file, said
            rep["error"] = f"could not be read: {str(e)[:200]}"
            reports.append(rep)
            continue
        if len(records) > MAX_RECORDS_PER_FILE:
            rep["error"] = f"{len(records)} records: at most {MAX_RECORDS_PER_FILE} per file"
            reports.append(rep)
            continue
        rep["records"] = len(records)
        if not records:
            rep["skipped"] = 1 if kind == "pdf" else 0
            rep["error"] = "no record with a title found" if kind != "pdf" else "no title could be read from the text"
            reports.append(rep)
            continue
        done = add_records(scenario_id, records, name, include)
        rep["added"] = sum(1 for d in done if d["status"] == "added")
        rep["already_in_corpus"] = sum(1 for d in done if d["status"] == "already_in_corpus")
        rep["skipped"] = sum(1 for d in done if d["status"] == "failed")
        articles.extend(done)
        reports.append(rep)
    totals = {
        "records": sum(r["records"] for r in reports),
        "added": sum(r["added"] for r in reports),
        "new_documents": sum(1 for a in articles if a.get("new_document")),
        "already_in_corpus": sum(r["already_in_corpus"] for r in reports),
        "skipped": sum(r["skipped"] for r in reports),
        "included": sum(1 for a in articles if a.get("included")),
        "kept_decision": sum(1 for a in articles if a.get("kept_decision")),
        "fulltext_stored": sum(1 for a in articles if a.get("fulltext_stored")),
        "files_failed": sum(1 for r in reports if r["error"] and r["records"] == 0),
    }
    scoring_started = False
    if totals["added"] or totals["already_in_corpus"]:
        try:
            scoring_started = bool(score_new_links(scenario_id))
        except Exception as e:                               # noqa: BLE001 - the articles are in
            logger.warning(f"upload {scenario_id}: scoring not started ({e})")
    if totals["included"]:
        schedule_retrain(scenario_id)
    return {"scenario_id": scenario_id, "include": include, "files": reports, "totals": totals,
            "scoring_started": scoring_started, "articles": articles}


@app.post("/user-scenarios/{scenario_id}/articles/upload")
async def upload_scenario_articles(
    scenario_id: str,
    files: list[UploadFile] = File(...),
    include: bool = False,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Add outside papers to a scenario's corpus from uploaded files: PDFs, or reference
    files (RIS, BibTeX, PubMed nbib, CSV or Excel with title, abstract, DOI, PMID and year
    columns). Each record joins the library (one row per work), the corpus, and is
    remembered as added by hand so that a later search keeps it; `include=true` also marks
    it as included by hand. The cosine scoring of the new links starts at once. The
    scenario's own filters (years, preprints) are not applied: a paper added by hand is
    wanted as it is."""
    _get_user_scenario_or_404(scenario_id)
    if not files:
        raise HTTPException(status_code=422, detail="No file")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=422, detail=f"At most {MAX_FILES} files per upload")
    loaded: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"{f.filename}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        loaded.append((os.path.basename(f.filename or "upload"), data))
    return await run_in_threadpool(_process_upload, scenario_id, loaded, include)


@app.get("/user-scenarios/{scenario_id}/articles/added")
def list_scenario_additions(scenario_id: str, limit: int = 500) -> dict[str, Any]:
    """The papers added by hand to this scenario, newest first, with the file each came
    from and its screening status."""
    _get_user_scenario_or_404(scenario_id)
    limit = max(1, min(int(limit), 5000))
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT sa.document_id AS id, d.title, d.year, d.doi, d.pmid, sa.filename, sa.added_at,
                   ars.screening_status, ars.similarity_score,
                   (ars.document_id IS NOT NULL) AS in_corpus
            FROM scenario_additions sa
            LEFT JOIN literature_document d ON d.id = sa.document_id
            LEFT JOIN article_scenarios ars ON ars.document_id = sa.document_id AND ars.scenario_id = sa.scenario_id
            WHERE sa.scenario_id = :s
            ORDER BY sa.added_at DESC, sa.document_id DESC
            LIMIT :n
        """), {"s": scenario_id, "n": limit}).mappings().all()
        total = int(c.execute(text("SELECT COUNT(*) FROM scenario_additions WHERE scenario_id = :s"),
                              {"s": scenario_id}).scalar() or 0)
    return {"scenario_id": scenario_id, "total": total,
            "articles": [{**dict(r), "added_at": r["added_at"].isoformat() if r["added_at"] else None} for r in rows]}
