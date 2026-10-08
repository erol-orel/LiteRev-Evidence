"""A report of the structured extraction, as Markdown, Word or PDF.

One document model (headings, paragraphs, bullets, tables) is built once from the same
functions the screen uses, then written three ways, so the three say the same thing. The report
states how the numbers were made (the subset, the model and prompt, the codebook, the review, the
statistics) and what they cannot say (papers read from the abstract only, unreviewed rows,
disagreement between studies), because a table of pooled values handed to someone who was not
there is only as honest as the method it carries with it.

It carries a snapshot identifier: a short hash of what the figures rest on (the counts, the
models and prompts, the review state), so a report can be cited and a later one told apart.

The Word file is written without a library (a zip of XML, tables included). The PDF uses reportlab,
already a dependency, and also draws a forest plot for the largest pooled estimates. Word carries the
same studies as a table instead of a figure.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from fastapi import HTTPException, Query
from fastapi.responses import Response

from .codebook import get_codebook, unmapped_labels
from .core import app
from .extraction import extraction_digest, extraction_status
from .extraction_review import review_summary
from .pooling import pooled_estimates
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404

_FIGURES = 6                                  # forest plots drawn in the PDF

_S = {
    "en": {
        "title": "Structured extraction report", "generated": "Generated", "snapshot": "Snapshot",
        "method": "How this was made", "search": "Search", "subset": "Relevant articles",
        "subset_t": "{n} articles: above the similarity threshold {thr}, or included by hand. Excluded articles and duplicates are never counted.",
        "extraction": "Extraction",
        "extraction_t": "{n} articles extracted ({ft} from the full text, {ab} from the abstract only). {unread} relevant articles are not extracted and are not counted anywhere below.",
        "models": "Made with", "codebook": "Codebook",
        "codebook_default": "the default codebook", "codebook_custom": "the project's codebook ({n} nodes)",
        "codebook_t": "Labels were matched to {cb}; {pct}% of the extracted rows carry a codebook label.",
        "review": "Review", "review_t": "{n} of {total} extracted rows were reviewed by {who}.",
        "review_none": "No row has been reviewed yet: every value below is the model's reading, unchecked.",
        "stats": "Statistics",
        "stats_t": "Pooled proportions and odds ratios are random-effects estimates (DerSimonian-Laird between-study variance) on the logit and log scales, with Hartung-Knapp-Sidik-Jonkman 95% intervals, a 95% prediction interval from three studies, and Wilson intervals for single studies. Studies are combined only within one codebook label and disease; a label needs at least {k} studies.",
        "reports": "What the papers report", "reports_t": "Counts are out of the {n} extracted articles.",
        "col_item": "Item", "col_papers": "Papers", "col_share": "Share", "col_ft": "From full text",
        "items": {"sex_gender": "Sex or gender", "age": "Age", "occupation": "Occupation",
                  "kap_risk_perception": "Knowledge, attitudes, practices, risk perception", "ppe": "Protective equipment",
                  "vaccination": "Vaccination", "human_testing": "Human testing", "animal_host": "Animal hosts",
                  "environment": "Environment", "vector": "Vectors"},
        "review_h": "Review status", "col_status": "Status", "col_rows": "Rows",
        "status": {"accepted": "Accepted", "edited": "Edited", "rejected": "Rejected", "conflict": "Conflict", "unreviewed": "Not reviewed"},
        "agreement": "Agreement between reviewers", "agreement_row": "{a} and {b}: {n} rows in common, {pct}% agree, kappa {k}",
        "pooled_h": "Pooled proportions", "col_label": "Label", "col_disease": "Disease", "col_studies": "Studies",
        "col_people": "Cases / people", "col_pooled": "Pooled (95% interval)", "col_pred": "Prediction interval", "col_het": "Disagreement (I2)",
        "none_pooled": "No label has {k} or more studies with a case count and a group size.",
        "few": "Labels with fewer than {k} studies, not pooled:", "cmp_h": "Comparisons between groups",
        "cmp_t": "Odds ratio of the first label against the second, over the papers that report both. 1 means no difference.",
        "col_cmp": "Comparison", "col_papers_n": "Papers", "col_or": "Odds ratio (95% interval)",
        "left_h": "Rows left out of the pooled estimates", "left": {"rejected": "Rejected by a reviewer", "conflict": "Reviewers disagree",
            "not_reviewed": "Not reviewed (only when reviewed rows alone were asked for)", "quote_not_found": "Quote not found in the text",
            "missing_counts": "No count of cases or of people", "invalid_counts": "Counts that cannot be used"},
        "limits": "Limits to keep in mind",
        "lim_abstract": "{n} articles were read from the abstract only. Tables are not in abstracts, so for them an absent item is a minimum, not proof of absence.",
        "lim_review": "{pct}% of the extracted rows have not been reviewed. Their values are the model's reading.",
        "lim_unmapped": "{n} rows carry a label that is not in the codebook; they are pooled only with an identical spelling.",
        "lim_het": "{n} pooled estimates show substantial or considerable disagreement between studies (I2 of 50% or more). Read their prediction interval, not only the pooled value.",
        "lim_combine": "Studies that are combined may differ in population, setting and definition, which the pooled value cannot show.",
        "appendix": "Studies behind each pooled estimate", "band": {"low": "low", "moderate": "moderate", "substantial": "substantial", "considerable": "considerable"},
        "all_diseases": "all diseases", "versus": "against", "not_stated": "not stated",
    },
    "fr": {
        "title": "Rapport d'extraction structurée", "generated": "Généré le", "snapshot": "Instantané",
        "method": "Comment cela a été fait", "search": "Recherche", "subset": "Articles pertinents",
        "subset_t": "{n} articles : au-dessus du seuil de similarité {thr}, ou inclus à la main. Les articles exclus et les doublons ne sont jamais comptés.",
        "extraction": "Extraction",
        "extraction_t": "{n} articles extraits ({ft} depuis le texte intégral, {ab} depuis le résumé seul). {unread} articles pertinents ne sont pas extraits et ne sont comptés nulle part ci-dessous.",
        "models": "Réalisé avec", "codebook": "Codebook",
        "codebook_default": "le codebook par défaut", "codebook_custom": "le codebook du projet ({n} noeuds)",
        "codebook_t": "Les libellés ont été rapprochés de {cb} ; {pct} % des lignes extraites portent un libellé du codebook.",
        "review": "Relecture", "review_t": "{n} lignes extraites sur {total} ont été relues par {who}.",
        "review_none": "Aucune ligne n'a encore été relue : chaque valeur ci-dessous est la lecture du modèle, non vérifiée.",
        "stats": "Statistiques",
        "stats_t": "Les proportions et odds ratios combinés sont des estimations à effets aléatoires (variance inter-études de DerSimonian-Laird) sur les échelles logit et log, avec des intervalles à 95 % de Hartung-Knapp-Sidik-Jonkman, un intervalle de prédiction à 95 % à partir de trois études, et des intervalles de Wilson pour une étude seule. Les études ne sont combinées que dans un même libellé du codebook et une même maladie ; un libellé demande au moins {k} études.",
        "reports": "Ce que rapportent les articles", "reports_t": "Les comptes portent sur les {n} articles extraits.",
        "col_item": "Élément", "col_papers": "Articles", "col_share": "Part", "col_ft": "En texte intégral",
        "items": {"sex_gender": "Sexe ou genre", "age": "Âge", "occupation": "Profession",
                  "kap_risk_perception": "Connaissances, attitudes, pratiques, perception du risque", "ppe": "Équipements de protection",
                  "vaccination": "Vaccination", "human_testing": "Tests chez l'humain", "animal_host": "Hôtes animaux",
                  "environment": "Environnement", "vector": "Vecteurs"},
        "review_h": "État de la relecture", "col_status": "Statut", "col_rows": "Lignes",
        "status": {"accepted": "Acceptées", "edited": "Corrigées", "rejected": "Rejetées", "conflict": "En conflit", "unreviewed": "Non relues"},
        "agreement": "Accord entre relecteurs", "agreement_row": "{a} et {b} : {n} lignes en commun, {pct} % d'accord, kappa {k}",
        "pooled_h": "Proportions combinées", "col_label": "Libellé", "col_disease": "Maladie", "col_studies": "Études",
        "col_people": "Cas / personnes", "col_pooled": "Combiné (intervalle à 95 %)", "col_pred": "Intervalle de prédiction", "col_het": "Désaccord (I2)",
        "none_pooled": "Aucun libellé n'a {k} études ou plus avec un nombre de cas et une taille de groupe.",
        "few": "Libellés avec moins de {k} études, non combinés :", "cmp_h": "Comparaisons entre groupes",
        "cmp_t": "Odds ratio du premier libellé contre le second, sur les articles qui rapportent les deux. 1 signifie aucune différence.",
        "col_cmp": "Comparaison", "col_papers_n": "Articles", "col_or": "Odds ratio (intervalle à 95 %)",
        "left_h": "Lignes écartées des estimations combinées", "left": {"rejected": "Rejetées par un relecteur", "conflict": "Les relecteurs divergent",
            "not_reviewed": "Non relues (seulement si les lignes relues étaient seules demandées)", "quote_not_found": "Citation non retrouvée dans le texte",
            "missing_counts": "Pas de nombre de cas ou de personnes", "invalid_counts": "Effectifs inutilisables"},
        "limits": "Limites à garder en tête",
        "lim_abstract": "{n} articles ont été lus dans le résumé seul. Les tableaux ne sont pas dans les résumés : pour eux, un élément absent est un minimum, pas une preuve d'absence.",
        "lim_review": "{pct} % des lignes extraites n'ont pas été relues. Leurs valeurs sont la lecture du modèle.",
        "lim_unmapped": "{n} lignes portent un libellé absent du codebook ; elles ne sont combinées qu'à orthographe identique.",
        "lim_het": "{n} estimations combinées montrent un désaccord important ou considérable entre études (I2 d'au moins 50 %). Lisez leur intervalle de prédiction, pas seulement la valeur combinée.",
        "lim_combine": "Les études combinées peuvent différer par la population, le contexte et la définition, ce que la valeur combinée ne montre pas.",
        "appendix": "Les études derrière chaque estimation combinée", "band": {"low": "faible", "moderate": "modéré", "substantial": "important", "considerable": "considérable"},
        "all_diseases": "toutes maladies", "versus": "contre", "not_stated": "non précisée",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def pct(p: float | None) -> str:
    return "-" if p is None else f"{p * 100:.1f}".rstrip("0").rstrip(".") + "%"


def fmt_or(v: float | None) -> str:
    if v is None:
        return "-"
    return str(round(v)) if v >= 100 else f"{float(f'{v:.3g}'):g}"


def _interval(lo: float | None, hi: float | None, f) -> str:
    return "-" if lo is None or hi is None else f"{f(lo)} to {f(hi)}"


def _label(g: dict) -> str:
    return (g.get("label_path") or g["label"]).replace("_", " ")


def _author(s: dict) -> str:
    return f"{s.get('first_author') or '?'} {s.get('year') or ''}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# The document model
# ─────────────────────────────────────────────────────────────────────────────
def build_report(scenario_id: str, lang: str = "en") -> dict[str, Any]:
    """Everything the report says, as {title, subtitle, snapshot, blocks}. Blocks are
    ("h1"|"h2"|"p"|"note", text), ("bullets", [text]), ("table", header, rows) and, for
    the PDF only, ("figure", kind, data)."""
    S = _S["fr" if str(lang).lower().startswith("fr") else "en"]
    scenario = _get_user_scenario_or_404(scenario_id)
    thr = _get_scenario_threshold(scenario_id)
    status = extraction_status(scenario_id)
    digest = extraction_digest(scenario_id, thr)
    if not digest.get("complete") or not digest.get("n_extracted"):
        raise HTTPException(status_code=409, detail="Nothing is extracted yet: there is nothing to report.")
    rev = review_summary(scenario_id)
    pooled = pooled_estimates(scenario_id)
    cb = get_codebook(scenario_id)
    un = unmapped_labels(scenario_id, top=1)
    n_ex, n_rel = digest["n_extracted"], digest["n_relevant"]
    k = pooled["filters"]["min_studies"]
    now = datetime.now(timezone.utc)

    models = "; ".join(f"{m['model']}, prompt {m['prompt_sha']} ({m['n']})" for m in digest.get("by_model", [])) or "-"
    mapped_share = round(100 * un["rows_mapped"] / max(1, un["rows_mapped"] + un["rows_unmapped"]))
    cb_text = S["codebook_default"] if cb["source"] == "default" else S["codebook_custom"].format(n=len(cb["nodes"]))
    who = ", ".join(r["reviewer"] for r in rev["reviewers"])
    snapshot = hashlib.sha1(json.dumps({
        "s": scenario_id, "thr": thr, "ex": [n_ex, n_rel, digest["n_fulltext"]], "m": digest.get("by_model"),
        "rv": rev["counts"], "rows": pooled["n_rows_used"], "cb": [cb["source"], len(cb["nodes"])],
    }, sort_keys=True, default=str).encode()).hexdigest()[:10]

    blocks: list[tuple] = []
    add = blocks.append
    add(("h2", S["method"]))
    add(("bullets", [
        f"{S['search']}: {scenario.get('query') or '-'}",
        f"{S['subset']}: " + S["subset_t"].format(n=n_rel, thr=thr),
        f"{S['extraction']}: " + S["extraction_t"].format(n=n_ex, ft=digest["n_fulltext"], ab=digest["n_abstract"], unread=digest["n_unread"]),
        f"{S['models']}: {models}.",
        f"{S['codebook']}: " + S["codebook_t"].format(cb=cb_text, pct=mapped_share),
        f"{S['review']}: " + (S["review_t"].format(n=rev["n_reviewed"], total=rev["n_observations"], who=who or "-") if rev["n_reviewed"] else S["review_none"]),
        f"{S['stats']}: " + S["stats_t"].format(k=k),
    ]))

    add(("h2", S["reports"]))
    add(("p", S["reports_t"].format(n=n_ex)))
    add(("table", [S["col_item"], S["col_papers"], S["col_share"], S["col_ft"]],
         [[S["items"][key], str(digest["coverage"].get(key, 0)), f"{round(100 * digest['coverage'].get(key, 0) / n_ex)}%",
           str(digest["coverage_fulltext"].get(key, 0))] for key in S["items"]]))

    add(("h2", S["review_h"]))
    add(("table", [S["col_status"], S["col_rows"]], [[S["status"][k_], str(rev["counts"].get(k_, 0))] for k_ in S["status"]]))
    if rev["agreement"]:
        add(("p", S["agreement"] + ":"))
        add(("bullets", [S["agreement_row"].format(a=a["reviewers"][0], b=a["reviewers"][1], n=a["n_common"],
                                                    pct=round(100 * (a["observed"] or 0)),
                                                    k="-" if a["kappa"] is None else f"{a['kappa']:.2f}") for a in rev["agreement"]]))

    groups = [g for g in pooled["pooled"] if g["pooled"]]
    small = [g for g in pooled["pooled"] if not g["pooled"]]
    add(("h2", S["pooled_h"]))
    if not groups:
        add(("p", S["none_pooled"].format(k=k)))
    else:
        add(("table", [S["col_label"], S["col_disease"], S["col_studies"], S["col_people"], S["col_pooled"], S["col_pred"], S["col_het"]],
             [[_label(g), g["disease"] or S["all_diseases"], str(g["k"]), f"{g['events_total']}/{g['n_total']}",
               f"{pct(g['pooled']['p'])} ({_interval(g['pooled']['ci_low'], g['pooled']['ci_high'], pct)})",
               _interval(g["pooled"]["pi_low"], g["pooled"]["pi_high"], pct),
               f"{round(g['heterogeneity']['I2'])}% ({S['band'][g['heterogeneity']['band']]})"] for g in groups]))
        for g in groups[:_FIGURES]:
            add(("figure", "proportion", g))
    if small:
        add(("p", S["few"].format(k=k)))
        add(("bullets", [f"{_label(g)} ({g['disease'] or S['all_diseases']}): " + "; ".join(f"{_author(s)} {s['x']}/{s['n']}" for s in g["studies"])
                         for g in small[:30]]))

    if pooled["comparisons"]:
        add(("h2", S["cmp_h"]))
        add(("p", S["cmp_t"]))
        add(("table", [S["col_cmp"], S["col_disease"], S["col_papers_n"], S["col_or"], S["col_het"]],
             [[f"{c['a'].replace('_', ' ')} {S['versus']} {c['b'].replace('_', ' ')}", c["disease"] or S["all_diseases"], str(c["k"]),
               f"{fmt_or(c['pooled']['or'])} ({_interval(c['pooled']['ci_low'], c['pooled']['ci_high'], fmt_or)})",
               f"{round(c['heterogeneity']['I2'])}% ({S['band'][c['heterogeneity']['band']]})"] for c in pooled["comparisons"]]))
        for c in pooled["comparisons"][:_FIGURES]:
            add(("figure", "odds", c))

    left = [(S["left"][key], str(n)) for key, n in pooled["excluded"].items() if n and key in S["left"]]
    if left:
        add(("h2", S["left_h"]))
        add(("table", [S["col_status"], S["col_rows"]], [[a, b] for a, b in left]))

    unreviewed_pct = round(100 * rev["counts"]["unreviewed"] / max(1, rev["n_observations"]))
    het_n = sum(1 for g in groups if g["heterogeneity"]["I2"] >= 50) + sum(1 for c in pooled["comparisons"] if c["heterogeneity"]["I2"] >= 50)
    limits = []
    if digest["n_abstract"]:
        limits.append(S["lim_abstract"].format(n=digest["n_abstract"]))
    if unreviewed_pct:
        limits.append(S["lim_review"].format(pct=unreviewed_pct))
    if un["rows_unmapped"]:
        limits.append(S["lim_unmapped"].format(n=un["rows_unmapped"]))
    if het_n:
        limits.append(S["lim_het"].format(n=het_n))
    limits.append(S["lim_combine"])
    add(("h2", S["limits"]))
    add(("bullets", limits))

    if groups:
        add(("h2", S["appendix"]))
        for g in groups:
            add(("p", f"{_label(g)} ({g['disease'] or S['all_diseases']})"))
            add(("table", ["", S["col_people"], S["col_share"]],
                 [[f"{_author(s)}: {(s['title'] or '')[:90]}", f"{s['x']}/{s['n']}", pct(s.get("p"))] for s in g["studies"]]))

    return {"title": S["title"], "subtitle": scenario.get("name") or scenario_id,
            "meta": f"{S['generated']} {now.strftime('%Y-%m-%d %H:%M')} UTC. {S['snapshot']} {snapshot}.",
            "snapshot": snapshot, "blocks": blocks}


# ─────────────────────────────────────────────────────────────────────────────
# Markdown
# ─────────────────────────────────────────────────────────────────────────────
def to_markdown(doc: dict[str, Any]) -> str:
    cell = lambda v: str(v).replace("|", "/").replace("\n", " ")
    out = [f"# {doc['title']}", "", f"**{doc['subtitle']}**", "", f"*{doc['meta']}*", ""]
    for b in doc["blocks"]:
        kind = b[0]
        if kind == "h2":
            out += [f"## {b[1]}", ""]
        elif kind in ("p", "note"):
            out += [b[1], ""]
        elif kind == "bullets":
            out += [f"- {t}" for t in b[1]] + [""]
        elif kind == "table":
            header, rows = b[1], b[2]
            out += ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "---|" * len(header)]
            out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows] + [""]
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Word, without a library
# ─────────────────────────────────────────────────────────────────────────────
def to_docx(doc: dict[str, Any]) -> bytes:
    def run(txt: str, bold: bool = False, size: int | None = None, color: str | None = None, italic: bool = False) -> str:
        pr = ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + (f'<w:color w:val="{color}"/>' if color else "") + (f'<w:sz w:val="{size}"/>' if size else "")
        return f'<w:r>{f"<w:rPr>{pr}</w:rPr>" if pr else ""}<w:t xml:space="preserve">{escape(str(txt))}</w:t></w:r>'

    def para(txt: str, **kw) -> str:
        sp = kw.pop("space", None)
        ppr = f'<w:pPr><w:spacing w:before="{sp[0]}" w:after="{sp[1]}"/></w:pPr>' if sp else ""
        return f"<w:p>{ppr}{run(txt, **kw)}</w:p>"

    def table(header: list[str], rows: list[list[str]]) -> str:
        n = max(1, len(header))
        w = 9638 // n
        border = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))

        def tc(txt: str, head: bool = False) -> str:
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF5"/>' if head else ""
            return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>{run(txt, bold=head, size=18)}</w:p></w:tc>')
        grid = "".join('<w:gridCol w:w="%d"/>' % w for _ in range(n))
        trs = ['<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(tc(h, True) for h in header) + "</w:tr>"]
        trs += ["<w:tr>" + "".join(tc(c) for c in r) + "</w:tr>" for r in rows]
        return ('<w:tbl><w:tblPr><w:tblW w:w="9638" w:type="dxa"/><w:tblBorders>' + border + "</w:tblBorders>"
                '<w:tblLayout w:type="autofit"/></w:tblPr><w:tblGrid>' + grid + "</w:tblGrid>"
                + "".join(trs) + "</w:tbl><w:p/>")

    body = [para(doc["title"], bold=True, size=40, space=(0, 80)), para(doc["subtitle"], bold=True, size=26, space=(0, 60)),
            para(doc["meta"], italic=True, size=18, color="666666", space=(0, 200))]
    for b in doc["blocks"]:
        kind = b[0]
        if kind == "h2":
            body.append(para(b[1], bold=True, size=28, space=(280, 100)))
        elif kind in ("p", "note"):
            body.append(para(b[1], size=21, space=(40, 80)))
        elif kind == "bullets":
            body += [para("• " + t, size=21, space=(0, 40)) for t in b[1]]
        elif kind == "table":
            body.append(table(b[1], b[2]))
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                + "".join(body) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>')
    parts = {
        "[Content_Types].xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                                '<Default Extension="xml" ContentType="application/xml"/>'
                                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>'),
        "_rels/.rels": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                        '</Relationships>'),
        "docProps/core.xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                              '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                              'xmlns:dc="http://purl.org/dc/elements/1.1/">'
                              f'<dc:title>{escape(doc["title"])}: {escape(doc["subtitle"])}</dc:title></cp:coreProperties>'),
        "word/document.xml": document,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PDF, with forest plots
# ─────────────────────────────────────────────────────────────────────────────
def _forest(kind: str, data: dict[str, Any], S: dict[str, Any]):
    """A vector forest plot as a reportlab Drawing."""
    import math

    from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
    from reportlab.lib import colors

    studies = data["studies"]
    log = kind == "odds"
    est = (lambda s: s["or"]) if log else (lambda s: s["p"])
    n_rows = len(studies) + 3
    row_h, left, plot_w, right = 13, 120, 190, 150
    h = n_rows * row_h + 26
    d = Drawing(left + plot_w + right, h)
    lows = [s["ci_low"] for s in studies] + [data["pooled"].get("pi_low") or data["pooled"]["ci_low"]]
    highs = [s["ci_high"] for s in studies] + [data["pooled"].get("pi_high") or data["pooled"]["ci_high"]]
    if log:
        lo, hi = max(min(lows + [1]) / 1.15, 1e-3), min(max(highs + [1]) * 1.15, 1e3)
        scale = lambda v: left + plot_w * (math.log(max(v, 1e-9)) - math.log(lo)) / ((math.log(hi) - math.log(lo)) or 1)
        ticks = [t for t in (0.25, 0.5, 1, 2, 4) if lo < t < hi]
        f = fmt_or
    else:
        lo, hi = 0.0, min(1.0, math.ceil(max(highs + [0.05]) * 10) / 10)
        scale = lambda v: left + plot_w * max(0.0, min(1.0, (v - lo) / ((hi - lo) or 1)))
        ticks = [t for t in (0, 0.25, 0.5, 0.75, 1) if t <= hi + 1e-9]
        f = pct
    grey = colors.Color(0.55, 0.55, 0.55)
    top = h - 8
    for t in ticks:
        d.add(Line(scale(t), 14, scale(t), top, strokeColor=colors.Color(0.85, 0.85, 0.85), strokeWidth=0.4))
        d.add(String(scale(t), 4, f(t), fontSize=6, fillColor=grey, textAnchor="middle"))
    if log:
        d.add(Line(scale(1), 14, scale(1), top, strokeColor=colors.black, strokeWidth=0.6, strokeDashArray=[2, 2]))
    wmax = max((s.get("weight_pct") or 1) for s in studies)
    for i, s in enumerate(studies):
        y = top - (i + 0.5) * row_h
        d.add(String(left - 5, y - 2, _author(s)[:24], fontSize=6.5, textAnchor="end"))
        d.add(Line(scale(s["ci_low"]), y, scale(s["ci_high"]), y, strokeColor=grey, strokeWidth=0.8))
        side = 1.6 + 2.2 * (s.get("weight_pct") or 1) / wmax
        d.add(Rect(scale(est(s)) - side, y - side, 2 * side, 2 * side, fillColor=colors.Color(0.18, 0.55, 0.42), strokeColor=None))
        d.add(String(left + plot_w + 6, y - 2, f"{f(est(s))} ({f(s['ci_low'])} to {f(s['ci_high'])})", fontSize=6, fillColor=grey))
    y = top - (len(studies) + 0.6) * row_h
    p = data["pooled"]
    mid = p["or"] if log else p["p"]
    d.add(String(left - 5, y - 2, S["col_pooled"].split(" (")[0], fontSize=7, fontName="Helvetica-Bold", textAnchor="end"))
    d.add(Polygon([scale(p["ci_low"]), y, scale(mid), y + 4.5, scale(p["ci_high"]), y, scale(mid), y - 4.5],
                  fillColor=colors.Color(0.85, 0.62, 0.1), strokeColor=None))
    d.add(String(left + plot_w + 6, y - 2, f"{f(mid)} ({f(p['ci_low'])} to {f(p['ci_high'])})", fontSize=6.5, fontName="Helvetica-Bold"))
    if p.get("pi_low") is not None:
        y2 = y - row_h
        d.add(Line(scale(p["pi_low"]), y2, scale(p["pi_high"]), y2, strokeColor=colors.Color(0.85, 0.62, 0.1), strokeWidth=1.6))
        d.add(String(left - 5, y2 - 2, S["col_pred"].split(" ")[0], fontSize=6.5, fillColor=grey, textAnchor="end"))
        d.add(String(left + plot_w + 6, y2 - 2, f"{f(p['pi_low'])} to {f(p['pi_high'])}", fontSize=6, fillColor=grey))
    return d


def to_pdf(doc: dict[str, Any], lang: str = "en") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    S = _S["fr" if str(lang).lower().startswith("fr") else "en"]
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
    cellp = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)
    head = ParagraphStyle("head", parent=cellp, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=19, leading=23, spaceAfter=4)
    h1b = ParagraphStyle("h1b", parent=styles["Heading2"], fontSize=12, leading=15, textColor="#444444", spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, keepWithNext=1)
    meta = ParagraphStyle("meta", parent=body, fontSize=8, textColor="#777777", spaceAfter=8)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0)
    esc = lambda t: escape(str(t))
    flow: list[Any] = [Paragraph(esc(doc["title"]), h1), Paragraph(esc(doc["subtitle"]), h1b), Paragraph(esc(doc["meta"]), meta)]
    for b in doc["blocks"]:
        kind = b[0]
        if kind == "h2":
            flow.append(Paragraph(esc(b[1]), h2))
        elif kind in ("p", "note"):
            flow.append(Paragraph(esc(b[1]), body))
        elif kind == "bullets":
            flow += [Paragraph(esc(t), bullet, bulletText="•") for t in b[1]]
        elif kind == "table":
            header, rows = b[1], b[2]
            data = [[Paragraph(esc(h), head) for h in header]] + [[Paragraph(esc(c), cellp) for c in r] for r in rows]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.91, 0.94, 0.96)),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.Color(0.75, 0.75, 0.75)),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            flow += [t, Spacer(1, 6)]
        elif kind == "figure":
            g = b[2]
            title = (_label(g) if b[1] == "proportion" else f"{g['a'].replace('_', ' ')} {S['versus']} {g['b'].replace('_', ' ')}")
            flow.append(KeepTogether([Paragraph(esc(title), cellp), _forest(b[1], g, S), Spacer(1, 6)]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=f"{doc['title']}: {doc['subtitle']}", leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm).build(flow)
    return buf.getvalue()


_TYPES = {"md": "text/markdown; charset=utf-8",
          "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
          "pdf": "application/pdf"}


@app.get("/user-scenarios/{scenario_id}/extraction/report")
def export_extraction_report(scenario_id: str, format: str = Query("pdf"), lang: str = Query("en")) -> Response:
    """The extraction report as Markdown, Word or PDF. The three come from one document model."""
    fmt = (format or "pdf").lower()
    if fmt not in _TYPES:
        raise HTTPException(status_code=422, detail=f"format must be one of {', '.join(_TYPES)}")
    doc = build_report(scenario_id, lang)
    data = {"md": lambda: to_markdown(doc).encode("utf-8"), "docx": lambda: to_docx(doc), "pdf": lambda: to_pdf(doc, lang)}[fmt]()
    name = f"extraction_report_{re.sub(r'[^A-Za-z0-9_-]', '_', scenario_id)}_{doc['snapshot']}.{fmt}"
    return Response(content=data, media_type=_TYPES[fmt], headers={
        "Content-Disposition": f'attachment; filename="{name}"', "X-Snapshot": doc["snapshot"]})
