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
from .scenario_store import (_get_scenario_rerank_threshold, _get_scenario_threshold,
                             _get_user_scenario_or_404, relevance_order_sql,
                             relevant_gate_sql, screening_status_sql, threshold_provenance)
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

# One row per article, the same columns in every tabular format. `rerank_score` sits
# next to the cosine: the rows are ORDERED by it first (relevance_order_sql), and a file
# that sorts on a score it does not carry cannot be checked against the screen.
EXPORT_COLUMNS = (
    "rank", "id", "title", "authors", "year", "journal", "doi", "pmid", "url", "source",
    "study_design", "similarity_score", "rerank_score", "quality_score", "citation_count",
    "screening_status", "keywords", "pico_population", "pico_intervention", "pico_comparator",
    "pico_outcome", "abstract",
)


def _pico(a: dict, key: str) -> str:
    pj = a.get("pico_json")
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except Exception:
            pj = None
    return str((pj or {}).get(key) or "") if isinstance(pj, dict) else ""


def _ris_value(value: Any) -> str:
    """Une étiquette RIS tient sur UNE ligne : un retour à la ligne dans un titre ou un
    résumé coupait l'enregistrement en deux et l'import s'arrêtait là."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ris_doi(doi: str) -> str:
    """Le DOI nu : Zotero attend « 10.xxxx/yyy », pas une URL ni un préfixe « doi: »."""
    d = _ris_value(doi)
    d = re.sub(r"(?i)^\s*(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", d)
    return d


def _article_url(a: dict) -> str:
    # `url` n'est pas toujours une URL : un cinquième à un tiers des notices y portent un
    # DOI nu (« 10.1016/… »), hérité d'une source qui n'a pas d'autre adresse. La colonne
    # `url` de l'export, le tag UR du RIS et les liens Markdown étaient alors des chaînes
    # sur lesquelles rien ne clique. Ce qui ne commence pas par http est traité pour ce
    # que c'est.
    _u = str(a.get("url") or "").strip()
    if _u.lower().startswith(("http://", "https://")):
        return _u
    if _u:
        _d = _ris_doi(_u)
        if _d.startswith("10."):
            return f"https://doi.org/{_d}"
        if _u.isdigit():                      # un PMID écrit dans la colonne url
            return f"https://pubmed.ncbi.nlm.nih.gov/{_u}/"
    if a.get("doi"):
        # Le DOI est parfois stocké déjà sous forme d'URL : le préfixer sans le
        # normaliser donnait « https://doi.org/https://doi.org/10.… », un lien mort.
        return f"https://doi.org/{_ris_doi(a['doi'])}"
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
            "rerank_score": (round(float(a["rerank_score"]), 4)
                             if a.get("rerank_score") is not None else None),
            "language": (a.get("language") or "").strip(),
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


# Le type RIS d'un enregistrement. Zotero crée un objet par type et n'affiche pas les
# mêmes champs pour chacun : tout exporter en JOUR rangeait les préprints et les
# recommandations parmi les articles de revue, avec un « journal » qui n'existe pas.
_RIS_PREPRINT = ("biorxiv", "medrxiv", "arxiv", "preprint", "ssrn", "research square", "osf")
_RIS_TYPE_BY_DESIGN = (
    ("guideline", "RPRT"), ("recommandation", "RPRT"), ("report", "RPRT"),
    ("rapport", "RPRT"), ("thesis", "THES"), ("thèse", "THES"),
    ("clinical trial registration", "DATA"), ("registre", "DATA"),
)


def _ris_type(r: dict) -> str:
    """JOUR par défaut ; UNPB pour un préprint, RPRT pour une recommandation ou un
    rapport. Lu sur la source puis sur le type d'étude, tous deux en minuscules."""
    src = (r.get("source") or "").strip().lower()
    if any(p in src for p in _RIS_PREPRINT):
        return "UNPB"
    design = (r.get("study_design") or "").strip().lower()
    if any(p in design for p in _RIS_PREPRINT):
        return "UNPB"
    for needle, ty in _RIS_TYPE_BY_DESIGN:
        if needle in design:
            return ty
    return "JOUR"


def to_ris(rows: list[dict]) -> str:
    """RIS importable tel quel par Zotero, EndNote et Mendeley.

    Les écarts qui empêchaient Zotero de lire ces fichiers correctement :

    - le journal était écrit en `JO`, que Zotero range dans « Abrév. de revue » ; les
      références arrivaient donc SANS publication, le champ qui sert à les citer. Le
      titre de publication est `T2` (et `JF` pour les logiciels qui ne lisent que lui) ;
    - les enregistrements se suivaient sans ligne vide. `ER  - ` suivi immédiatement du
      `TY` du suivant n'est pas conforme et EndNote comme Mendeley s'y arrêtent ;
    - tout sortait en `TY  - JOUR`, y compris les préprints et les recommandations ;
    - un retour à la ligne resté dans un titre ou un résumé coupait l'enregistrement ;
    - le DOI pouvait partir sous forme d'URL, que Zotero ne reconnaît pas comme un DOI.

    Le PMID suit la convention de l'export RIS de PubMed lui-même (`DB  - PubMed`,
    `AN  - <pmid>`), que Zotero range dans Archive et Loc. dans l'archive.
    """
    out = []
    for r in rows:
        ty = _ris_type(r)
        lines = [f"TY  - {ty}"]
        title = _ris_value(r.get("title"))
        if title:
            lines.append(f"TI  - {title}")
        for au in _authors_list(r.get("authors") or ""):
            lines.append(f"AU  - {_ris_value(au)}")
        if r.get("year"):
            lines.append(f"PY  - {r['year']}")
        journal = _ris_value(r.get("journal"))
        if journal and ty == "JOUR":
            # T2 pour Zotero (publicationTitle), JF pour les importeurs qui ne lisent
            # que lui. JO est réservé à l'ABRÉVIATION et n'est donc plus écrit.
            lines.append(f"T2  - {journal}")
            lines.append(f"JF  - {journal}")
        elif journal:
            # Préprint ou rapport : le « journal » est le dépôt ou l'organisme.
            lines.append(f"PB  - {journal}")
        doi = _ris_doi(r.get("doi") or "")
        if doi:
            lines.append(f"DO  - {doi}")
        if r.get("pmid"):
            lines.append("DB  - PubMed")
            lines.append(f"AN  - {_ris_value(r['pmid'])}")
        if r.get("url"):
            lines.append(f"UR  - {_ris_value(r['url'])}")
        if r.get("language"):
            lines.append(f"LA  - {_ris_value(r['language'])}")
        # TOUS les mots-clés : le CSV, l'XLSX et le BibTeX les gardaient tous, le RIS
        # s'arrêtait au vingtième sans le dire.
        for kw in [k.strip() for k in (r.get("keywords") or "").split(";") if k.strip()]:
            lines.append(f"KW  - {_ris_value(kw)}")
        if r.get("abstract"):
            lines.append(f"AB  - {_ris_value(r['abstract'])}")
        notes = []
        # Le cosinus est une similarité, pas « la pertinence » : l'ordre du fichier est
        # celui du reclassement quand il existe, qui part avec lui.
        if _score_note(r):
            notes.append(_score_note(r))
        if r.get("study_design"):
            notes.append(f"design {_ris_value(r['study_design'])}")
        if notes:
            lines.append(f"N1  - LiteRev: {', '.join(notes)}")
        lines.append("ER  - ")
        out.append("\r\n".join(lines))
    # Ligne vide entre deux enregistrements, et fin de fichier après le dernier.
    return "\r\n\r\n".join(out) + ("\r\n" if out else "")


def _bib_escape(s: str) -> str:
    return (s or "").replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")


def _bib_key(r: dict) -> str:
    first = (_authors_list(r.get("authors") or "") or ["anon"])[0].split()[0]
    first = re.sub(r"[^A-Za-z]", "", first) or "anon"
    return f"{first.lower()}{r.get('year') or 'nd'}_{r.get('id')}"


def _score_note(r: dict) -> str:
    """Les deux scores, chacun sous son nom : « relevance » désignait le cosinus."""
    bits = []
    if r.get("similarity_score") is not None:
        bits.append(f"cosine similarity {r['similarity_score']}")
    if r.get("rerank_score") is not None:
        bits.append(f"rerank score {r['rerank_score']}")
    return ", ".join(bits)


def to_bibtex(rows: list[dict]) -> str:
    out = []
    for r in rows:
        fields = [("title", r["title"]), ("author", " and ".join(_authors_list(r["authors"]))),
                  ("year", str(r.get("year") or "")), ("journal", r.get("journal") or ""),
                  ("doi", r.get("doi") or ""), ("url", r.get("url") or ""),
                  ("keywords", r.get("keywords") or ""), ("abstract", r.get("abstract") or ""),
                  ("language", r.get("language") or ""),
                  ("note", f"LiteRev {_score_note(r)}" if _score_note(r) else "")]
        body = ",\n".join(f"  {k} = {{{_bib_escape(v)}}}" for k, v in fields if v)
        out.append(f"@article{{{_bib_key(r)},\n{body}\n}}")
    return "\n\n".join(out) + ("\n" if out else "")


def to_markdown(rows: list[dict], title: str = "", provenance: list[str] | None = None) -> str:
    lines = [f"# {title}".rstrip(), ""] if title else []
    # « relevant articles » quel que soit le sous-ensemble : un export fait à seuil zéro
    # s'annonçait ainsi, et un export par cluster aussi.
    lines.append(f"{len(rows)} articles, most relevant first.")
    lines.append("")
    if provenance:
        lines += [f"> {ln}" for ln in provenance] + [""]
    for r in rows:
        head = f"{r['rank']}. **{r['title']}**"
        meta = ", ".join(x for x in (r.get("authors"), str(r.get("year") or ""), r.get("journal")) if x)
        link = f" [{r['doi'] or 'link'}]({r['url']})" if r.get("url") else ""
        lines.append(f"{head}{link}")
        if meta:
            lines.append(f"   {meta}")
        extras = []
        if _score_note(r):
            extras.append(_score_note(r))
        if r.get("study_design"):
            extras.append(r["study_design"])
        if extras:
            lines.append(f"   _{' · '.join(extras)}_")
        lines.append("")
    return "\n".join(lines)


#: Les caractères de contrôle qu'un classeur XML ne peut pas porter. openpyxl lève
#: IllegalCharacterError au premier d'entre eux : UN titre ou UN résumé importé avec un
#: \x0b ou un \x1f faisait échouer l'export entier. Même classe que celle d'openpyxl.
_XLSX_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _xlsx_safe(value: Any) -> Any:
    """La valeur telle qu'une cellule peut la porter : les chaînes perdent leurs
    caractères de contrôle interdits, le reste passe inchangé."""
    return _XLSX_ILLEGAL.sub("", value) if isinstance(value, str) else value


def to_xlsx(rows: list[dict], sheet_title: str = "Relevant articles",
            provenance: list[str] | None = None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    # La provenance sur sa propre FEUILLE : en l'écrivant au-dessus du tableau, on
    # casserait le filtre automatique et la première ligne d'en-têtes.
    if provenance:
        _ps = wb.create_sheet("Provenance")
        _ps.append(["LiteRev export"])
        for _ln in provenance:
            _ps.append([_xlsx_safe(_ln)])
        _ps.column_dimensions["A"].width = 120
    ws = wb.active
    # Un titre de feuille refuse aussi []:*?/\ ; le nom du scénario peut en porter.
    ws.title = re.sub(r"[\[\]:*?/\\]", " ", _xlsx_safe(sheet_title))[:31].strip() or "Articles"
    ws.append(list(EXPORT_COLUMNS))
    for r in rows:
        ws.append([_xlsx_safe(r.get(k)) for k in EXPORT_COLUMNS])
    widths = {"title": 60, "authors": 40, "journal": 28, "abstract": 80, "url": 36, "doi": 24,
              "pico_population": 30, "pico_intervention": 30, "pico_comparator": 24, "pico_outcome": 30}
    for i, col in enumerate(EXPORT_COLUMNS, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 12)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def provenance_lines(meta: dict | None) -> list[str]:
    """Ce qu'un fichier doit porter pour se lire SEUL : quel scénario, quelle requête,
    quel sous-ensemble, quels seuils, et la phrase de couverture.

    Seul le JSON recevait `meta`. Les cinq autres formats partaient sans une ligne de
    provenance, si bien qu'un CSV de 190 lignes et un CSV de 640 lignes du même scénario
    étaient indistinguables, et que le Markdown appelait « relevant articles » n'importe
    quel sous-ensemble, y compris un export fait à seuil zéro."""
    m = meta or {}
    out: list[str] = []
    if m.get("scenario"):
        out.append(f"Scenario: {m['scenario']}" + (f" ({m['scenario_id']})" if m.get("scenario_id") else ""))
    if m.get("query"):
        out.append(f"Query: {m['query']}")
    if m.get("subset_label"):
        out.append(f"Subset: {m['subset_label']} ({m.get('n_articles', 0)} articles)")
    _thr, _rthr = m.get("similarity_threshold"), m.get("rerank_threshold")
    if _thr is not None:
        # Un seuil que personne n'a réglé est le point de départ de l'outil, pas une
        # décision de la revue : le fichier le dit.
        out.append(f"Similarity threshold: {_thr}"
                   + (" (default, not chosen)" if m.get("threshold_chosen") is False else "")
                   + (f" | rerank threshold: {_rthr}" if _rthr else "")
                   + (" | WARNING: threshold 0 means the WHOLE corpus, not a relevant subset"
                      if (_thr == 0 and not _rthr) else ""))
    if m.get("unscored") is not None:
        out.append(f"Unscored: {m['unscored']} articles without a similarity score "
                   f"(outside the relevant set until scored, unless included by hand)")
    if m.get("include_abstract") is False:
        out.append("Abstracts: not included in this file")
    if m.get("coverage"):
        out.append(f"Coverage: {m['coverage']}")
    if m.get("generated_at"):
        out.append(f"Generated: {m['generated_at']}")
    return out


def render_export(fmt: str, rows: list[dict], title: str, meta: dict | None = None) -> bytes:
    """Bytes of the export in `fmt`. Pure (openpyxl for xlsx).

    Every format carries the provenance block, each in its own comment syntax: a file
    that travels on its own has to say what it is."""
    prov = provenance_lines(meta)
    if fmt == "csv":
        # Le BOM reste au PREMIER octet : Excel ne détecte l'UTF-8 que là. Préfixer les
        # lignes de provenance devant lui faisait ouvrir « Rossi MÃ¼ller » au lieu de
        # « Rossi Müller ». On le retire de `to_csv` et on le remet en tête.
        body = to_csv(rows)
        bom = "\ufeff"
        if body.startswith(bom):
            body = body[len(bom):]
        head = "".join(f"# {ln}\n" for ln in prov)
        return (bom + head + body).encode("utf-8")
    if fmt == "json":
        return to_json(rows, meta).encode("utf-8")
    if fmt == "ris":
        # RIS n'a pas de commentaire : on utilise une notice GEN (« generic ») en tête,
        # que Zotero importe comme une note et qu'aucun gestionnaire ne confond avec un
        # article.
        head = ""
        if prov:
            head = ("TY  - GEN\nTI  - LiteRev export provenance\n"
                    + "".join(f"N1  - {ln}\n" for ln in prov) + "ER  - \n\n")
        return (head + to_ris(rows)).encode("utf-8")
    if fmt == "bibtex":
        head = "".join(f"% {ln}\n" for ln in prov) + ("\n" if prov else "")
        return (head + to_bibtex(rows)).encode("utf-8")
    if fmt == "md":
        return to_markdown(rows, title, prov).encode("utf-8")
    if fmt == "xlsx":
        return to_xlsx(rows, title, prov)
    raise ValueError(fmt)


def _export_extra_fields(ids: list[int]) -> dict[int, dict]:
    """PMID, source, URL, keywords and language of the articles: the relevance query does
    not carry them (without the language, the RIS `LA` tag was never written)."""
    if not ids:
        return {}
    out: dict[int, dict] = {}
    try:
        with engine.connect() as conn:
            for r in conn.execute(text(
                "SELECT id, pmid, source, url, keywords, language FROM literature_document "
                "WHERE id = ANY(:ids)"
            ), {"ids": [int(i) for i in ids]}).mappings():
                out[int(r["id"])] = dict(r)
    except Exception as _e:                                  # a database without `url`, say
        logger.warning(f"export extra fields: {_e}")
    return out


def _export_link_fields(scenario_id: str, ids: list[int]) -> dict[int, dict]:
    """The scenario's own scores of the articles that a row does not already carry: the
    relevance query reads the cosine but not the rerank score that orders it."""
    if not ids:
        return {}
    out: dict[int, dict] = {}
    try:
        with engine.connect() as conn:
            for r in conn.execute(text(
                "SELECT document_id AS id, rerank_score FROM article_scenarios "
                "WHERE scenario_id = :sid AND document_id = ANY(:ids)"
            ), {"sid": scenario_id, "ids": [int(i) for i in ids]}).mappings():
                out[int(r["id"])] = {"rerank_score": r["rerank_score"]}
    except Exception as _e:                                  # noqa: BLE001 - a detail, never the export
        logger.warning(f"export link fields: {_e}")
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (s or "").strip()).strip("-").lower()
    return s[:60] or "scenario"


def _normalize_format(fmt: str) -> str:
    """Le format demandé, ou 400. `bib` est un alias courant de `bibtex`."""
    fmt = (fmt or "csv").lower()
    if fmt == "bib":
        fmt = "bibtex"
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400,
                            detail=f"format must be one of {', '.join(EXPORT_FORMATS)}")
    return fmt


# Colonnes lues pour un export par identifiants. Mêmes noms que
# `_get_above_threshold_articles`, pour que `export_rows` ne voie aucune différence
# entre un sous-ensemble et le corpus pertinent.
_BY_IDS_SQL = f"""
    SELECT d.id, d.title, d.year, d.journal, d.authors, d.doi, d.study_design,
           d.citation_count, d.quality_score, d.abstract, d.pico_json,
           {screening_status_sql('d', 'ars')} AS screening_status,
           ars.similarity_score, ars.rerank_score
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
    WHERE d.id = ANY(:ids) AND d.is_duplicate IS NOT TRUE
    ORDER BY {relevance_order_sql('d', 'ars')}
"""


def articles_by_ids(scenario_id: str, ids: list[int]) -> list[dict]:
    """Les articles de CE scénario parmi `ids`, dans l'ordre de pertinence de l'app.

    Bornée au scénario par le JOIN : un identifiant qui ne lui appartient pas est
    simplement absent du résultat, jamais exporté. C'est ce qui permet d'ouvrir un
    export générique par identifiants sans en faire une fuite du corpus entier."""
    ids = [int(i) for i in ids if str(i).strip()]
    if not ids:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text(_BY_IDS_SQL), {"sid": scenario_id, "ids": ids}).mappings().all()
    return [dict(r) for r in rows]


def articles_export_response(scenario_id: str, fmt: str, articles: list[dict],
                             include_abstract: bool, *, subset: str, subset_label: str,
                             coverage: str = "", extra_meta: dict | None = None,
                             threshold: float | None = None) -> Response:
    """Rend N'IMPORTE QUEL sous-ensemble d'articles dans les six formats.

    Les formateurs étaient déjà purs ; seule la SÉLECTION des lignes était câblée sur le
    corpus pertinent. Ce noyau la prend en paramètre, de sorte qu'un cluster, un concept,
    une réponse du RAG ou une sélection à la main produisent exactement le même fichier,
    avec les mêmes colonnes.

    `coverage` : ce que le sous-ensemble couvre ET ce qu'il ne couvre pas (un cluster est
    tiré d'une projection plafonnée, pas du corpus entier). Il part dans les métadonnées
    du fichier, pas seulement dans l'interface : un export qui circule seul doit porter
    ses propres limites."""
    fmt = _normalize_format(fmt)
    row = _get_user_scenario_or_404(scenario_id)
    extra = _export_extra_fields([a["id"] for a in articles])
    for a in articles:
        a.update({k: v for k, v in (extra.get(int(a["id"])) or {}).items() if k != "id"})
    _without_rerank = [a["id"] for a in articles if "rerank_score" not in a]
    if _without_rerank:
        links = _export_link_fields(scenario_id, _without_rerank)
        for a in articles:
            a.setdefault("rerank_score", (links.get(int(a["id"])) or {}).get("rerank_score"))
    rows = export_rows(articles, include_abstract=include_abstract)
    title = str(row.get("name") or scenario_id)
    from datetime import datetime as _dt, timezone as _tz
    # Les SEUILS qui ont défini ce sous-ensemble. Sans eux, `?threshold=0` rendait le
    # corpus entier sous l'étiquette « tous les articles pertinents du scénario », et
    # rien dans le fichier ne permettait de s'en apercevoir.
    _thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    _rthr = _get_scenario_rerank_threshold(scenario_id)
    # Choisi : posé par une personne, ou donné explicitement à cet export. Et combien
    # d'articles restent hors du sous-ensemble faute de score.
    _prov = threshold_provenance(scenario_id)
    if threshold is not None:
        _prov["threshold_chosen"] = True
    meta = {"scenario_id": scenario_id, "scenario": title, "query": row.get("query"),
            "n_articles": len(rows), "format": fmt,
            "subset": subset, "subset_label": subset_label,
            "similarity_threshold": _thr, "rerank_threshold": _rthr,
            "threshold_chosen": _prov["threshold_chosen"], "unscored": _prov["unscored"],
            "include_abstract": bool(include_abstract),
            "generated_at": _dt.now(_tz.utc).isoformat(timespec="seconds")}
    if coverage:
        meta["coverage"] = coverage
    meta.update(extra_meta or {})
    doc_title = title if subset == "relevant" else f"{title} - {subset_label}"
    body = render_export(fmt, rows, doc_title, meta)
    filename = f"{_slug(title)}_{_slug(subset_label)}_{len(rows)}.{_EXTENSIONS[fmt]}"
    return Response(content=body, media_type=_CONTENT_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "X-Article-Count": str(len(rows)),
                             "X-Export-Subset": subset})


def relevant_articles_export(scenario_id: str, fmt: str, threshold: float | None,
                             include_abstract: bool) -> Response:
    articles = _get_above_threshold_articles(scenario_id, threshold=threshold)
    # Le second seuil, dans la phrase de couverture aussi : elle définissait « pertinent »
    # par la seule similarité alors qu'un seuil de rerank posé retire des articles.
    _rthr = float(_get_scenario_rerank_threshold(scenario_id) or 0.0)
    _rerank = (f" et, quand il a été calculé, au-dessus du seuil de reclassement "
               f"(cross-encoder) {_rthr:.2f}" if _rthr > 0 else "")
    return articles_export_response(
        scenario_id, fmt, articles, include_abstract, threshold=threshold,
        subset="relevant", subset_label="relevant-articles",
        coverage=(f"Tous les articles pertinents du scénario : au-dessus du seuil de "
                  f"similarité{_rerank}, ou inclus par un relecteur, jamais les exclus."))


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


# ── Le corpus ENTIER d'une recherche ─────────────────────────────────────────
# La page de recherche exportait les 200 lignes de son aperçu, avec des colonnes
# héritées que le corpus construit en direct ne porte pas (toutes vides). Elle passe
# désormais par le serveur : les pertinents (ci-dessus), ou tout ce que la recherche a
# trouvé, pertinent ou non (ici). `?threshold=0` sur l'export des pertinents n'en tient
# pas lieu : il garde la porte, donc écarte les exclus et ce que le seuil de rerank retire.

def corpus_articles_export(scenario_id: str, fmt: str, include_abstract: bool) -> Response:
    _get_user_scenario_or_404(scenario_id)
    thr = float(_get_scenario_threshold(scenario_id) or 0.0)
    with engine.connect() as conn:
        ids = [int(i) for i in conn.execute(text(
            "SELECT ars.document_id FROM article_scenarios ars "
            "JOIN literature_document d ON d.id = ars.document_id "
            "WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE"
        ), {"sid": scenario_id}).scalars()]
        n_relevant = int(conn.execute(text(
            "SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars "
            "JOIN literature_document d ON d.id = ars.document_id "
            f"WHERE ars.scenario_id = :sid AND {relevant_gate_sql('d', 'ars')}"
        ), {"sid": scenario_id, "thr": thr}).scalar() or 0)
    articles = articles_by_ids(scenario_id, ids)
    return articles_export_response(
        scenario_id, fmt, articles, include_abstract,
        subset="corpus", subset_label="whole-corpus",
        coverage=(f"Le corpus ENTIER de la recherche ({len(articles)} articles, doublons "
                  f"retirés) : au-dessus du seuil ou non, exclus au screening compris "
                  f"(colonne screening_status). {n_relevant} sont pertinents (seuil de "
                  f"similarité {thr:.2f}, ou inclus à la main) : ce sont eux que lisent "
                  f"les analyses."),
        extra_meta={"n_relevant": n_relevant})


@app.get("/user-scenarios/{scenario_id}/corpus/export")
def export_user_scenario_corpus(scenario_id: str, format: str = Query("csv"),
                                include_abstract: bool = True) -> Response:
    """Every article a search found, relevant or not, in the six formats, in the
    relevance order of the app. The relevant subset is `/relevant/export`."""
    return corpus_articles_export(scenario_id, format, include_abstract)


@app.get("/gesica/scenarios/{scenario_id}/corpus/export")
def export_gesica_scenario_corpus(scenario_id: str, format: str = Query("csv"),
                                  include_abstract: bool = True) -> Response:
    """Same export for the built-in scenarios."""
    return corpus_articles_export(scenario_id, format, include_abstract)


# ─── Exports de SOUS-ENSEMBLES du corpus ─────────────────────────────────────
# Partout où l'application découpe le corpus (clusters, carte des concepts, réponse du
# RAG, sélection à la main), on doit pouvoir sortir la liste des articles correspondante,
# dans les mêmes six formats et avec les mêmes colonnes que l'export des pertinents. Les
# formateurs ne changent pas : seule la sélection des lignes diffère.

@app.get("/user-scenarios/{scenario_id}/clusters/{cluster_id}/export")
def export_scenario_cluster(
    scenario_id: str,
    cluster_id: int,
    format: str = Query("csv"),
    lang: str | None = Query(None),
    include_abstract: bool = True,
) -> Response:
    """Les articles d'UN cluster, dans le format demandé.

    Le cache de clustering conserve TOUS les points avec leur identifiant de document
    (seule la charge utile servie est sous-échantillonnée pour l'affichage), donc
    l'export d'un cluster est complet POUR CE CLUSTER.

    En revanche le clustering lui-même ne porte que sur les `CLUSTER_MAX_DOCS` articles
    les plus pertinents : c'est une projection bornée par la mémoire, pas une extraction.
    Le fichier le dit dans ses métadonnées, pour qu'un export qui circule seul ne se lise
    pas comme « tous les articles de ce thème »."""
    from .clustering import CLUSTER_MAX_DOCS, _load_viz_cache

    _get_user_scenario_or_404(scenario_id)
    # `_load_viz_cache` prend la CLÉ du cache ("clustering"), pas le nom de colonne :
    # lui passer "clustering_json" levait un KeyError et l'export échouait toujours.
    cache = _load_viz_cache(scenario_id, "clustering")
    if not cache or not cache.get("clusters"):
        raise HTTPException(status_code=404,
                            detail="Aucun clustering en cache pour ce scénario : ouvrez "
                                   "l'onglet Clusters pour le calculer, puis réessayez.")
    match = next((c for c in cache["clusters"]
                  if isinstance(c, dict) and int(c.get("cluster_id", -99)) == int(cluster_id)), None)
    if match is None:
        _known = sorted(int(c.get("cluster_id")) for c in cache["clusters"]
                        if isinstance(c, dict) and c.get("cluster_id") is not None)
        raise HTTPException(status_code=404,
                            detail=f"Cluster {cluster_id} inconnu (disponibles : {_known}).")
    ids = [int(p["id"]) for p in (match.get("points") or [])
           if isinstance(p, dict) and p.get("id") is not None]
    articles = articles_by_ids(scenario_id, ids)
    _name = str(match.get("cluster_name") or f"cluster-{cluster_id}")
    _n_clustered = int(cache.get("n_docs") or 0)
    _n_eligible = int(cache.get("n_docs_total") or _n_clustered)
    # Le total du cache est le sous-ensemble pertinent ENTIER, articles sans résumé
    # compris : ceux-là ne peuvent pas être placés sur la carte, et le fichier le dit.
    _no_abs = int(cache.get("n_docs_no_abstract") or 0)
    coverage = (
        f"Articles du cluster « {_name} » ({len(articles)}). Le clustering porte sur "
        f"{_n_clustered} articles sur {_n_eligible} pertinents : c'est une projection "
        f"plafonnée à CLUSTER_MAX_DOCS={CLUSTER_MAX_DOCS} (les plus pertinents), pas le "
        f"corpus entier"
        + (f", et {_no_abs} article(s) sans résumé n'y figurent pas" if _no_abs else "")
        + ". Pour la liste complète des articles pertinents, utilisez l'export du corpus."
    )
    return articles_export_response(
        scenario_id, format, articles, include_abstract,
        subset="cluster", subset_label=_slug(_name) or f"cluster-{cluster_id}",
        coverage=coverage,
        extra_meta={"cluster_id": int(cluster_id), "cluster_name": _name,
                    "clustering_lang": cache.get("lang") or lang,
                    "n_docs_clustered": _n_clustered, "n_docs_eligible": _n_eligible,
                    "top_words": match.get("top_words") or [],
                    "cluster_summary": match.get("summary") or ""})


def _csv_set(value: str | None) -> set[str]:
    return {p.strip().lower() for p in (value or "").split(",") if p.strip()}


@app.get("/user-scenarios/{scenario_id}/concepts/export")
def export_scenario_concept_subset(
    scenario_id: str,
    concepts: str | None = Query(None, description="type:label, séparés par « | ». Ex: pathogen:dengue virus|vector:Aedes albopictus"),
    mode: str = Query("any", pattern="^(any|all)$"),
    format: str = Query("csv"),
    include_abstract: bool = True,
    composed: bool = Query(False, description="La carte telle que composée : tous les concepts de l'index, moins les filtres ci-dessous."),
    exclude_types: str | None = Query(None, description="Types masqués, séparés par des virgules (avec composed)."),
    min_count: int = Query(1, ge=1, description="Nombre minimal d'articles d'un concept (avec composed)."),
    exclude: str | None = Query(None, description="Concepts retirés à la main, type:label séparés par « | » (avec composed)."),
) -> Response:
    """Les articles derrière un concept, une SÉLECTION de concepts, ou la carte COMPOSÉE.

    `concepts` + `mode=any` : les articles citant AU MOINS UN des concepts (l'union).
    `mode=all` : ceux qui les citent TOUS (l'intersection, ce qu'on veut pour « les
    articles où ce pathogène ET ce vecteur apparaissent ensemble »).
    `composed=true` (sans `concepts`) : la carte telle que l'utilisateur l'a composée,
    c'est-à-dire tous les concepts moins les types masqués, ceux de moins de `min_count`
    articles et ceux retirés à la main. Toujours l'union.

    Tout se résout sur l'index COMPLET des concepts des articles pertinents, jamais sur
    la carte servie : ses soixante noeuds, ses quatorze concepts par type à l'écran et
    ses quarante articles par noeud sont des coupes d'affichage, et en hériter ici
    produisait un fichier amputé sans le dire (ou un 404 pour un concept bien présent
    dans le corpus). Les concepts sont désignés par `type:label` (le libellé canonique
    anglais, ou toute graphie qui s'y ramène) et non par leur identifiant de noeud, qui
    n'est qu'une position dans un calcul donné."""
    from .knowledge_graph import _concept_rows, parse_concept_refs, select_concepts

    _get_user_scenario_or_404(scenario_id)
    try:
        wanted = parse_concept_refs((concepts or "").split("|"))
        removed = parse_concept_refs((exclude or "").split("|"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not wanted and not composed:
        raise HTTPException(status_code=400, detail="Aucun concept demandé.")

    rows, n_total = _concept_rows(scenario_id)
    sel = select_concepts(rows, concepts=wanted or None, mode=mode,
                          exclude_types=_csv_set(exclude_types), min_count=min_count,
                          exclude=removed)
    if sel["missing"]:
        raise HTTPException(
            status_code=404,
            detail=f"Concept(s) absent(s) des articles pertinents : {', '.join(sel['missing'])}. "
                   f"Les libellés sont ceux de la carte (anglais canonique).")
    articles = articles_by_ids(scenario_id, sorted(sel["ids"]))
    _basis = (f"Calculé sur l'index COMPLET des concepts des {sel['n_with_concepts']} articles "
              f"pertinents porteurs de concepts (sur {n_total}), sans les coupes d'affichage "
              f"de la carte (noeuds dessinés, concepts par type, articles par noeud).")
    if wanted:
        labels = sel["labels"]
        coverage = f"Articles citant {sel['description']} ({len(articles)}). {_basis}"
        subset_label = _slug("-".join(labels)) or "concepts"
    else:
        labels = []
        coverage = (f"Articles de la carte des concepts telle que composée ({len(articles)}) : "
                    f"{sel['description']}, soit {sel['n_concepts']} concepts. {_basis}")
        subset_label = "concept-map"
    return articles_export_response(
        scenario_id, format, articles, include_abstract,
        subset="concepts", subset_label=subset_label,
        coverage=coverage,
        extra_meta={"concepts": labels, "mode": mode if wanted else "any",
                    "composed": not wanted,
                    "concept_filter": (None if wanted else
                                       {"exclude_types": sorted(_csv_set(exclude_types)),
                                        "min_count": int(min_count),
                                        "exclude": [f"{t}:{lab}" for t, lab in removed]}),
                    "n_concepts": sel["n_concepts"],
                    "n_with_concepts": sel["n_with_concepts"],
                    "n_total": n_total})


@app.get("/user-scenarios/{scenario_id}/articles/export")
def export_scenario_article_ids(
    scenario_id: str,
    ids: str = Query(..., description="Identifiants d'articles séparés par des virgules."),
    format: str = Query("csv"),
    label: str = Query("selection", description="Nom du sous-ensemble, pour le fichier."),
    include_abstract: bool = True,
) -> Response:
    """Export d'une LISTE EXPLICITE d'articles de ce scénario.

    C'est l'export générique : il sert les sources d'une réponse du RAG (qui portent leur
    `document_id`), une carte des concepts filtrée à l'écran, une sélection faite à la
    main, et tout découpage à venir, sans qu'il faille un endpoint de plus à chaque fois.

    Borné au scénario : un identifiant qui ne lui appartient pas n'est pas exporté, il est
    signalé dans `X-Missing-Ids`. La réponse ne peut donc pas servir à extraire le corpus
    d'un autre scénario en devinant des identifiants.

    Au-delà d'un millier d'identifiants, la ligne de requête dépasse ce que nginx accepte
    (414) : la variante POST ci-dessous prend la même liste dans le corps."""
    try:
        wanted = [int(x) for x in (ids or "").replace(" ", "").split(",") if x]
    except ValueError:
        raise HTTPException(status_code=400, detail="`ids` doit être une liste d'entiers séparés par des virgules.")
    return _article_ids_export(scenario_id, wanted, format, label, include_abstract)


@app.post("/user-scenarios/{scenario_id}/articles/export")
def export_scenario_article_ids_post(scenario_id: str, payload: dict[str, Any]) -> Response:
    """La même sélection explicite, la liste dans le CORPS : `{"ids": [...], "format":
    "csv", "label": "selection", "include_abstract": true}`. Une lecture comme le GET,
    donc sans clé ; elle existe parce qu'une carte filtrée ou une grosse sélection
    dépasse la longueur d'URL qu'accepte le proxy."""
    raw = payload.get("ids")
    if isinstance(raw, str):
        raw = raw.replace(" ", "").split(",")
    try:
        wanted = [int(x) for x in (raw or []) if str(x).strip()]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="`ids` doit être une liste d'entiers.")
    return _article_ids_export(scenario_id, wanted, str(payload.get("format") or "csv"),
                               str(payload.get("label") or "selection"),
                               payload.get("include_abstract") is not False)


def _article_ids_export(scenario_id: str, wanted: list[int], format: str, label: str,
                        include_abstract: bool) -> Response:
    """Le coeur partagé des deux variantes de l'export par identifiants."""
    if not wanted:
        raise HTTPException(status_code=400, detail="Aucun identifiant fourni.")
    if len(wanted) > 20000:
        raise HTTPException(status_code=400, detail="Trop d'identifiants (maximum 20000).")
    _get_user_scenario_or_404(scenario_id)
    articles = articles_by_ids(scenario_id, wanted)
    _found = {int(a["id"]) for a in articles}
    _missing = [i for i in wanted if i not in _found]
    coverage = (
        f"Sélection explicite de {len(articles)} article(s) sur {len(wanted)} demandé(s)."
        + (f" {len(_missing)} identifiant(s) n'appartiennent pas à ce scénario et ont été "
           f"écartés." if _missing else "")
    )
    resp = articles_export_response(
        scenario_id, format, articles, include_abstract,
        subset="selection", subset_label=_slug(label) or "selection",
        coverage=coverage,
        extra_meta={"requested": len(wanted), "returned": len(articles),
                    "missing_ids": _missing[:100]})
    if _missing:
        resp.headers["X-Missing-Ids"] = str(len(_missing))
    return resp


# Les scénarios PRÉRÉGLÉS passent par /gesica/scenarios (cf. `scenarioBase` côté front) :
# sans ces trois alias, leurs boutons d'export pointaient sur des routes inexistantes,
# comme l'export des pertinents l'avait déjà prévu de son côté.
@app.get("/gesica/scenarios/{scenario_id}/clusters/{cluster_id}/export")
def export_gesica_cluster(scenario_id: str, cluster_id: int, format: str = Query("csv"),
                          lang: str | None = Query(None),
                          include_abstract: bool = True) -> Response:
    """Idem pour les scénarios préréglés."""
    return export_scenario_cluster(scenario_id, cluster_id, format, lang, include_abstract)


@app.get("/gesica/scenarios/{scenario_id}/concepts/export")
def export_gesica_concept_subset(scenario_id: str, concepts: str | None = Query(None),
                                 mode: str = Query("any", pattern="^(any|all)$"),
                                 format: str = Query("csv"),
                                 include_abstract: bool = True,
                                 composed: bool = Query(False),
                                 exclude_types: str | None = Query(None),
                                 min_count: int = Query(1, ge=1),
                                 exclude: str | None = Query(None)) -> Response:
    """Idem pour les scénarios préréglés."""
    return export_scenario_concept_subset(scenario_id, concepts, mode, format, include_abstract,
                                          composed, exclude_types, min_count, exclude)


@app.get("/gesica/scenarios/{scenario_id}/articles/export")
def export_gesica_article_ids(scenario_id: str, ids: str = Query(...),
                              format: str = Query("csv"), label: str = Query("selection"),
                              include_abstract: bool = True) -> Response:
    """Idem pour les scénarios préréglés."""
    return export_scenario_article_ids(scenario_id, ids, format, label, include_abstract)


@app.post("/gesica/scenarios/{scenario_id}/articles/export")
def export_gesica_article_ids_post(scenario_id: str, payload: dict[str, Any]) -> Response:
    """Idem pour les scénarios préréglés."""
    return export_scenario_article_ids_post(scenario_id, payload)
