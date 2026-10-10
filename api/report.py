"""The evidence brief as a citable document: inline citations, numbered references, figures.

What this is for. The commercial report we compared against is a polished ten-page PDF
with numbered figures and a full reference list, and it reads like a paper. Our brief held
the same substance in seventeen JSON fields and a `key_references` list the model wrote
from memory, which reads like a tool's output. The difference is not cosmetic: a claim you
cannot trace to a row is a claim a reviewer cannot check.

So this assembles, from material that already exists and with no LLM call of its own:

  methods      the query, the threshold, the PRISMA counts, the models and the reasoning
               effort that produced the text, and the number of relevant articles the
               synthesis speaks for;
  narrative    the brief's prose, with the `[id]` markers the model was asked to place
               renumbered to the reference list, and any marker that does not resolve
               reported rather than deleted;
  claims       the graded claim table, strengths and their basis (api/evidence.py);
  gaps         the concept matrix, counted over the whole relevant subset (api/digest.py);
  references   built from the DATABASE, not from the model. This is the part a generated
               reference list cannot be trusted with, and the part that makes every
               number in the document resolvable to a row a reviewer can open.

Markdown, deliberately. It needs no new dependency in a deployment whose API boot is
dependency-light, it diffs, and pandoc turns it into whatever a journal asks for. The
interface keeps its own print-to-PDF view for reading on screen.

Everything here is pure except `evidence_report`, which only reads.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, Query
from sqlalchemy import text

from .core import _norm_lang, app, engine, logger
from .scenario_store import (_get_scenario_rerank_threshold, _get_scenario_threshold,
                             _get_user_scenario_or_404)
from .study_design import level_label

#: `[12]` or `[12, 34]`: what the brief's prompt asks the model to place in its prose. A
#: bare number in brackets is also how a numbered reference appears in a paper, which is
#: the point: the marker the model writes and the marker the reader sees are the same
#: shape, so renumbering is a substitution rather than a rewrite.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

#: Which brief fields carry prose a citation can appear in. Lists are flattened; the
#: structured fields (claims, key_references) are rendered separately.
_NARRATIVE_FIELDS = (
    ("executive_summary", "Résumé exécutif"),
    ("clinical_context", "Contexte"),
    ("evidence_synthesis", "Synthèse des évidences"),
    ("population_summary", "Populations étudiées"),
    ("intervention_summary", "Interventions et expositions"),
    ("outcome_summary", "Critères de jugement"),
    ("methodological_quality", "Qualité méthodologique"),
    ("clinical_implications", "Implications pratiques"),
    ("future_research", "Recherche future"),
)
_LIST_FIELDS = (
    ("key_findings", "Principaux constats"),
    ("recommended_actions", "Actions recommandées"),
    ("implementation_recommendations", "Recommandations de mise en oeuvre"),
    ("limitations", "Limites"),
    ("research_gaps", "Lacunes identifiées"),
)

#: The same titles in English. The report was French whatever the brief's language: an
#: English brief for a foreign team came out under "Niveau de preuve global" and "Qualité
#: méthodologique", with claim strengths reading "Faible".
_TITLES_EN = {
    "executive_summary": "Executive summary",
    "clinical_context": "Context",
    "evidence_synthesis": "Evidence synthesis",
    "population_summary": "Populations studied",
    "intervention_summary": "Interventions and exposures",
    "outcome_summary": "Outcomes",
    "methodological_quality": "Methodological quality",
    "clinical_implications": "Practical implications",
    "future_research": "Future research",
    "key_findings": "Key findings",
    "recommended_actions": "Recommended actions",
    "implementation_recommendations": "Implementation recommendations",
    "limitations": "Limitations",
    "research_gaps": "Research gaps",
}

#: Every other sentence of the report, in both languages, with the same placeholders.
_REPORT_STRINGS = {
    "fr": {
        "generated": "*Généré le {date}*",
        "methods": "## 1. Méthodes",
        "query": "Corpus interrogé par la requête booléenne suivante : `{query}`.",
        "query_missing": "non renseignée",
        "relevant": ("Un article est *pertinent* lorsqu'il dépasse le seuil de similarité "
                     "({threshold}){rerank} **ou** qu'un relecteur l'a inclus à la main ; les "
                     "articles écartés au screening ne le sont jamais. La synthèse ci-dessous "
                     "porte sur la **totalité des {n} articles pertinents** : les agrégats "
                     "(années, devis, pays, revues, concepts, matrice de lacunes) sont calculés "
                     "en SQL sur l'ensemble complet, sans échantillonnage. Les articles "
                     "reproduits intégralement dans le prompt du générateur ne servent qu'à "
                     "citer."),
        "rerank": (" et, quand il a été calculé, dont le score de reclassement (cross-encoder) "
                   "atteint {r}"),
        "warning": ("> **Avertissement** : le calcul des agrégats a échoué, les totaux "
                    "ci-dessus ne décrivent pas le corpus complet."),
        "years": "Années : {a} à {b}",
        "included": "Inclus par un relecteur : {n}",
        "with_pico": "Avec PICO extrait : {n}",
        "with_fulltext": "Avec texte intégral : {n}",
        "funnel": {"total": "Notices du corpus", "duplicates": "Doublons",
                   "excluded": "Écartés au screening", "pending": "En attente de screening"},
        "model": "modèle de rédaction `{m}`",
        "effort": "effort de raisonnement `{e}`",
        "ceiling": "plafond de certitude du corpus `{c}`",
        "not_recorded": "non renseigné",
        "produced": ("Texte produit avec : {bits}. Les forces de preuve des affirmations ne "
                     "sont pas écrites par le modèle : elles sont calculées d'après les devis "
                     "des articles cités."),
        "results": "## 2. Résultats",
        "single": "une seule étude, abaissé d'un niveau",
        "capped": "plafonné par le corpus (devis cités : {d})",
        "claims_title": "### Figure {n}. Affirmations et force des preuves",
        "claims_headers": ["Affirmation", "Force", "Base du calcul", "Références"],
        "claims_note": ("La force est calculée, non déclarée : le devis le plus solide parmi "
                        "les articles cités fixe le niveau, une affirmation reposant sur une "
                        "seule étude perd un niveau, et le plafond du corpus s'applique en "
                        "dernier. Ce barème n'évalue ni le risque de biais, ni la cohérence, ni "
                        "la précision : ce n'est pas une évaluation GRADE complète."),
        "unverified": ("> **Citations non vérifiables** ({n}) : {ids}. Ces identifiants ne "
                       "correspondent à aucun article du corpus de ce scénario."),
        "matrix_title": "### Figure {n}. Matrice de lacunes ({r} x {c})",
        "matrix_note": ("Nombre d'articles pertinents associant chaque concept en ligne à "
                        "chaque concept en colonne, compté sur l'ensemble du sous-ensemble "
                        "pertinent. Une case vide signifie qu'**aucun article de ce corpus** ne "
                        "traite les deux ensemble ; ce n'est pas une preuve d'absence dans la "
                        "littérature. Lisible sur {a} des {b} articles pertinents : les autres "
                        "n'ont pas de concepts extraits et n'apparaissent pas dans cette "
                        "matrice."),
        "matrix_truncated": ("> Axes tronqués pour la lecture : {r}/{rt} lignes, {c}/{ct} "
                             "colonnes. Les cases hors de cette fenêtre ne sont pas rapportées "
                             "comme lacunes."),
        "level_title": "## 3. Niveau de preuve global",
        "level": "- Niveau de preuve : {v}",
        "grade": "- Grade de recommandation : {v}",
        "refs_title": "## 4. Références",
        "no_refs": "*Aucune citation résolue dans ce brief.*",
        "refs_note": ("Les {n} références ci-dessus sont construites à partir des lignes du "
                      "corpus, et non rédigées par le modèle. `[#id]` est l'identifiant interne "
                      "de l'article, par lequel un relecteur retrouve la notice."),
        "unresolved": ("> **Appels de citation non résolus** : {ids}. Ils apparaissent dans le "
                       "texte suivis d'un `?` et ne correspondent à aucun article du corpus."),
        "no_authors": "[auteurs non renseignés]",
        "no_title": "[titre non renseigné]",
    },
    "en": {
        "generated": "*Generated on {date}*",
        "methods": "## 1. Methods",
        "query": "Corpus searched with the following Boolean query: `{query}`.",
        "query_missing": "not provided",
        "relevant": ("An article is *relevant* when it passes the similarity threshold "
                     "({threshold}){rerank} **or** a reviewer included it by hand; articles "
                     "excluded at screening never are. The synthesis below covers **all {n} "
                     "relevant articles**: the aggregates (years, designs, countries, journals, "
                     "concepts, gap matrix) are computed in SQL over the complete set, without "
                     "sampling. The articles reproduced in full in the generator's prompt serve "
                     "only for quotation."),
        "rerank": " and, where it was computed, whose reranking (cross-encoder) score reaches {r}",
        "warning": ("> **Warning**: the aggregation failed; the totals above do not describe "
                    "the complete corpus."),
        "years": "Years: {a} to {b}",
        "included": "Included by a reviewer: {n}",
        "with_pico": "With extracted PICO: {n}",
        "with_fulltext": "With full text: {n}",
        "funnel": {"total": "Records in the corpus", "duplicates": "Duplicates",
                   "excluded": "Excluded at screening", "pending": "Awaiting screening"},
        "model": "writing model `{m}`",
        "effort": "reasoning effort `{e}`",
        "ceiling": "corpus certainty ceiling `{c}`",
        "not_recorded": "not recorded",
        "produced": ("Text produced with: {bits}. The strength of each claim is not written by "
                     "the model: it is computed from the study designs of the articles cited."),
        "results": "## 2. Results",
        "single": "a single study, downgraded one level",
        "capped": "capped by the corpus (designs cited: {d})",
        "claims_title": "### Figure {n}. Claims and strength of evidence",
        "claims_headers": ["Claim", "Strength", "Basis", "References"],
        "claims_note": ("The strength is computed, not declared: the strongest design among the "
                        "articles cited sets the level, a claim resting on a single study loses "
                        "one level, and the corpus ceiling applies last. This scale does not "
                        "assess risk of bias, consistency or precision: it is not a full GRADE "
                        "assessment."),
        "unverified": ("> **Unverifiable citations** ({n}): {ids}. These identifiers match no "
                       "article in this scenario's corpus."),
        "matrix_title": "### Figure {n}. Gap matrix ({r} x {c})",
        "matrix_note": ("Number of relevant articles pairing each row concept with each column "
                        "concept, counted over the whole relevant subset. An empty cell means "
                        "that **no article in this corpus** addresses both together; it is not "
                        "evidence of absence from the literature. Readable on {a} of the {b} "
                        "relevant articles: the others have no extracted concepts and do not "
                        "appear in this matrix."),
        "matrix_truncated": ("> Axes truncated for reading: {r}/{rt} rows, {c}/{ct} columns. "
                             "Cells outside this window are not reported as gaps."),
        "level_title": "## 3. Overall level of evidence",
        "level": "- Level of evidence: {v}",
        "grade": "- Grade of recommendation: {v}",
        "refs_title": "## 4. References",
        "no_refs": "*No citation resolved in this brief.*",
        "refs_note": ("The {n} references above are built from the corpus rows, not written by "
                      "the model. `[#id]` is the article's internal identifier, by which a "
                      "reviewer finds the record."),
        "unresolved": ("> **Unresolved citation markers**: {ids}. They appear in the text "
                       "followed by `?` and match no article in the corpus."),
        "no_authors": "[authors not recorded]",
        "no_title": "[title not recorded]",
    },
}


def _lang_key(lang: str | None) -> str:
    return "en" if (lang or "fr").lower().startswith("en") else "fr"


def _level_text(value: Any, key: str) -> Any:
    """A level as the report's reader should read it. The values stay French on the server
    (they are compared elsewhere); in an English report they are shown in English, from
    the GRADE scale (`level_label`) or, for the older brief labels ("Modéré", "Fort"),
    from the brief's own vocabulary."""
    if key != "en" or not isinstance(value, str):
        return value
    shown = level_label(value, "en")
    if shown != value:
        return shown
    from .evidence import brief_level                       # lazy: evidence imports widely
    return brief_level(value, "en")


def cited_ids(brief: dict) -> list[int]:
    """Every article id the brief refers to, in order of first appearance.

    Both sources count: the `[id]` markers in the prose and the verified `article_ids` of
    the graded claims. A reference list assembled from only one of them would leave
    numbers in the text pointing at nothing."""
    seen: list[int] = []

    def _add(value):
        try:
            n = int(value)
        except (TypeError, ValueError):
            return
        if n not in seen:
            seen.append(n)

    for field, _ in (*_NARRATIVE_FIELDS, *_LIST_FIELDS):
        value = brief.get(field)
        for chunk in (value if isinstance(value, list) else [value]):
            if not isinstance(chunk, str):
                continue
            for match in _CITATION.finditer(chunk):
                for part in match.group(1).split(","):
                    _add(part.strip())
    for claim in (brief.get("claims") or []):
        if isinstance(claim, dict):
            for value in (claim.get("article_ids") or []):
                _add(value)
    return seen


def renumber(prose: str, numbers: dict[int, int]) -> tuple[str, list[int]]:
    """Rewrite `[id]` markers as `[n]` reference numbers. Returns (text, unresolved ids).

    An id with no reference is LEFT IN PLACE and reported, not silently removed: a
    citation the corpus cannot resolve is a fact about the brief, and deleting it would
    make the text look better than it is while destroying the evidence of why."""
    unresolved: list[int] = []

    def _sub(match):
        out = []
        for part in match.group(1).split(","):
            try:
                article_id = int(part.strip())
            except ValueError:
                continue
            if article_id in numbers:
                out.append(str(numbers[article_id]))
            else:
                if article_id not in unresolved:
                    unresolved.append(article_id)
                out.append(f"{article_id}?")
        return f"[{', '.join(out)}]" if out else match.group(0)

    return _CITATION.sub(_sub, prose or ""), unresolved


def format_reference(n: int, article: dict, lang: str | None = "fr") -> str:
    """One numbered reference, Vancouver-ish, from the DATABASE row.

    Not from the model: a generated bibliography is the single easiest thing in an
    evidence product to get confidently wrong, and the row is right there."""
    from .exports import _authors_list

    s = _REPORT_STRINGS[_lang_key(lang)]
    authors = _authors_list(article.get("authors") or "")
    if len(authors) > 6:
        who = ", ".join(authors[:6]) + ", et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = s["no_authors"]
    # Le point après les auteurs n'est pas cosmétique : sans lui le titre s'enchaîne au
    # dernier auteur ("Roe B A trial of vaccination."), ce qui casse la lecture et tout
    # analyseur de bibliographie. `rstrip` évite le double point après « et al. ».
    bits = [f"{n}. {who.rstrip('.')}.",
            (article.get("title") or s["no_title"]).rstrip(".") + "."]
    if article.get("journal"):
        bits.append(f"{article['journal']}.")
    if article.get("year"):
        bits.append(f"{article['year']}.")
    if article.get("doi"):
        bits.append(f"doi:{article['doi']}")
    elif article.get("pmid"):
        bits.append(f"PMID:{article['pmid']}")
    # L'identifiant interne reste visible : c'est par lui qu'un relecteur retrouve la
    # ligne dans le corpus, et c'est ce qu'un rapport commercial ne peut pas offrir.
    bits.append(f"[#{article.get('id')}]")
    return " ".join(bits)


def _cached_brief(scenario_id: str) -> dict | None:
    """The stored Evidence Brief, or None. Never generates one.

    `GET /scenarios/{id}/evidence-brief/llm` serves the cache but FALLS BACK to launching
    a generation when the fingerprint has moved, which is right for that endpoint and
    wrong for this one: a report is a read, and a GET that quietly spends a minute of
    model time and a few thousand tokens is not. A stale brief is still reported, and says
    when it was generated, so the reader can judge."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT evidence_brief_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"_cached_brief {scenario_id}: {e}")
        return None
    if not row or not row["evidence_brief_json"]:
        return None
    brief = dict(row["evidence_brief_json"])
    return brief if brief.get("error") is None else None


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table, with pipes in cell text escaped so a title cannot break it."""
    def _cell(value):
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return out


def _rerank_clause(rerank_threshold: float | None, lang: str | None = "fr") -> str:
    """La phrase qui dit le SECOND seuil, quand il est posé. La définition de « pertinent »
    de la section Méthodes ne nommait que la similarité : un seuil de rerank retirait des
    articles de l'ensemble analysé sans que le rapport le dise. Vide sans second seuil."""
    try:
        r = float(rerank_threshold or 0.0)
    except (TypeError, ValueError):
        return ""
    if r <= 0:
        return ""
    return _REPORT_STRINGS[_lang_key(lang)]["rerank"].format(r=f"{r:.2f}")


def build_report(brief: dict, digest: dict, matrix: dict | None, funnel: dict | None,
                 articles_by_id: dict, scenario_name: str, query: str | None,
                 threshold: float, rerank_threshold: float | None = None,
                 lang: str | None = "fr") -> dict[str, Any]:
    """The whole document as markdown, plus what could not be resolved. Pure.

    Figures are numbered in the order they appear, as a paper numbers them, so the text
    can refer to "figure 2" and mean it. `lang` is the language of the document around
    the brief's own prose (French by default, as before)."""
    from .evidence import brief_level                       # lazy: evidence imports widely

    key = _lang_key(lang)
    s = _REPORT_STRINGS[key]
    numbers = {}
    references = []
    for article_id in cited_ids(brief):
        article = articles_by_id.get(article_id)
        if article is None:
            continue                                  # compté comme non résolu plus bas
        numbers[article_id] = len(references) + 1
        references.append(format_reference(len(references) + 1, article, key))

    unresolved: list[int] = []
    lines: list[str] = []
    meta = brief.get("_meta") or {}

    lines += [f"# {scenario_name}", ""]
    if meta.get("generated_at"):
        lines += [s["generated"].format(date=str(meta["generated_at"])[:19]), ""]

    # ── Méthodes ─────────────────────────────────────────────────────────────
    lines += [s["methods"], ""]
    n_relevant = digest.get("n_articles") or 0
    lines += [
        s["query"].format(query=query or s["query_missing"]),
        "",
        s["relevant"].format(threshold=f"{threshold:.2f}",
                             rerank=_rerank_clause(rerank_threshold, key), n=n_relevant),
        "",
    ]
    if digest.get("complete") is False:
        lines += [s["warning"], ""]
    facts = []
    if digest.get("year_min") and digest.get("year_max"):
        facts.append(s["years"].format(a=digest["year_min"], b=digest["year_max"]))
    if digest.get("n_included"):
        facts.append(s["included"].format(n=digest["n_included"]))
    if digest.get("n_with_pico") is not None:
        facts.append(s["with_pico"].format(n=digest["n_with_pico"]))
    if digest.get("n_with_fulltext") is not None:
        facts.append(s["with_fulltext"].format(n=digest["n_with_fulltext"]))
    if funnel:
        # Clés de `corpus_stats` (api/evidence.py), pas inventées ici.
        for fkey in ("total", "duplicates", "excluded", "pending"):
            if isinstance(funnel.get(fkey), int):
                sep = " : " if key == "fr" else ": "
                facts.append(f"{s['funnel'][fkey]}{sep}{funnel[fkey]}")
    if facts:
        lines += ["- " + "\n- ".join(facts), ""]
    model_bits = [s["model"].format(m=meta.get("model", s["not_recorded"]))]
    if meta.get("reasoning_effort"):
        model_bits.append(s["effort"].format(e=meta["reasoning_effort"]))
    if meta.get("grade_ceiling"):
        model_bits.append(s["ceiling"].format(c=_level_text(meta["grade_ceiling"], key)))
    lines += [s["produced"].format(bits=", ".join(model_bits)), ""]

    # ── Résultats ────────────────────────────────────────────────────────────
    lines += [s["results"], ""]
    for field, title in _NARRATIVE_FIELDS:
        value = brief.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        rewritten, missing = renumber(value, numbers)
        unresolved += [m for m in missing if m not in unresolved]
        title = _TITLES_EN.get(field, title) if key == "en" else title
        lines += [f"### {title}", "", rewritten.strip(), ""]
    for field, title in _LIST_FIELDS:
        values = brief.get(field)
        if not isinstance(values, list) or not values:
            continue
        title = _TITLES_EN.get(field, title) if key == "en" else title
        lines += [f"### {title}", ""]
        for item in values:
            rewritten, missing = renumber(str(item), numbers)
            unresolved += [m for m in missing if m not in unresolved]
            lines.append(f"- {rewritten.strip()}")
        lines.append("")

    # ── Affirmations ─────────────────────────────────────────────────────────
    figure = 1
    claims = [c for c in (brief.get("claims") or []) if isinstance(c, dict)]
    if claims:
        rows = []
        for claim in claims:
            refs = ", ".join(f"[{numbers[i]}]" for i in (claim.get("article_ids") or [])
                             if i in numbers) or "-"
            basis = claim.get("basis") or {}
            why = [f"{basis.get('n_articles', 0)} article(s)"]
            if basis.get("designs"):
                why.append(", ".join(f"{d} ({n})" for d, n in basis["designs"].items()))
            if basis.get("downgraded_single_study"):
                why.append(s["single"])
            if basis.get("capped_by_corpus"):
                why.append(s["capped"].format(d=_level_text(basis.get("from_designs"), key)))
            # La force est une VALEUR de l'échelle (française, comparée ailleurs) : on en
            # affiche le libellé dans la langue du rapport.
            rows.append([claim.get("claim", ""), _level_text(claim.get("strength", ""), key),
                         " ; ".join(why) if key == "fr" else "; ".join(why), refs])
        lines += [s["claims_title"].format(n=figure), ""]
        lines += _table(s["claims_headers"], rows)
        lines += ["", s["claims_note"], ""]
        unverified = [str(x) for c in claims for x in (c.get("unverified_ids") or [])]
        if unverified:
            lines += [s["unverified"].format(n=len(unverified), ids=", ".join(unverified)), ""]
        figure += 1

    # ── Matrice de lacunes ───────────────────────────────────────────────────
    if matrix and matrix.get("rows") and matrix.get("cols"):
        cells = {(c["row"], c["col"]): c["n"] for c in matrix["cells"]}
        headers = [f"{matrix.get('row_type', '')} \\ {matrix.get('col_type', '')}"]
        headers += [c["label"] for c in matrix["cols"]]
        rows = [[r["label"]] + [str(cells.get((r["label"], c["label"]), 0) or "-")
                                for c in matrix["cols"]]
                for r in matrix["rows"]]
        lines += [s["matrix_title"].format(n=figure, r=matrix.get("row_type"),
                                           c=matrix.get("col_type")), ""]
        lines += _table(headers, rows)
        cov = matrix.get("coverage") or {}
        lines += ["", s["matrix_note"].format(a=cov.get("with_concepts", 0),
                                              b=cov.get("relevant", 0)), ""]
        if matrix.get("rows_total", 0) > len(matrix["rows"]) or \
                matrix.get("cols_total", 0) > len(matrix["cols"]):
            lines += [s["matrix_truncated"].format(r=len(matrix["rows"]), rt=matrix["rows_total"],
                                                   c=len(matrix["cols"]), ct=matrix["cols_total"]), ""]
        figure += 1

    # ── Conclusion et références ─────────────────────────────────────────────
    if brief.get("evidence_level") or brief.get("grade_recommendation"):
        lines += [s["level_title"], "",
                  s["level"].format(v=brief_level(brief.get("evidence_level"), key) or s["not_recorded"]),
                  s["grade"].format(v=brief.get("grade_recommendation") or s["not_recorded"]),
                  ""]
    lines += [s["refs_title"], ""]
    if references:
        lines += references
    else:
        lines += [s["no_refs"]]
    lines += ["", s["refs_note"].format(n=len(references)), ""]
    if unresolved:
        lines += [s["unresolved"].format(ids=", ".join(str(u) for u in unresolved)), ""]

    return {"markdown": "\n".join(lines).rstrip() + "\n",
            "references": len(references),
            "citations_unresolved": unresolved,
            "figures": figure - 1}


@app.get("/user-scenarios/{scenario_id}/evidence-report")
def evidence_report(scenario_id: str, download: bool = Query(False),
                    gaps_rows: str | None = Query(None),
                    gaps_cols: str | None = Query(None),
                    lang: str | None = Query(None)) -> Any:
    """The brief, its claim table, its gap matrix and a database-built reference list, as
    one markdown document.

    Reads only: it assembles what is already cached and computed, and makes no LLM call of
    its own, so asking for the report twice costs nothing and gives the same document."""
    _get_user_scenario_or_404(scenario_id)
    from fastapi.responses import PlainTextResponse

    from .digest import concept_matrix, corpus_digest
    from .relevance import _get_above_threshold_articles

    threshold = _get_scenario_threshold(scenario_id)
    # Le brief LLM EN CACHE, jamais une génération. Le rapport doit pouvoir être demandé
    # deux fois sans coûter deux briefs, et une génération ici prendrait une minute et
    # dépenserait des jetons derrière un simple GET. Pas de brief en cache : on le dit.
    llm = _cached_brief(scenario_id)
    if llm is None:
        _detail = ("Aucun Evidence Brief généré pour ce scénario. Générez-le "
                   "d'abord : le rapport assemble un brief existant, il n'en "
                   "produit pas.")
        if download:
            # Le navigateur suit un <a download> : renvoyer 200 avec un corps JSON
            # d'erreur lui faisait enregistrer un fichier de 166 octets PORTANT LE NOM
            # DU RAPPORT. Un refus doit être un refus.
            raise HTTPException(status_code=409, detail=_detail)
        return {"status": "no_brief", "message": _detail}
    digest = corpus_digest(scenario_id, threshold)

    matrix = None
    try:
        types = [t["value"] for t in
                 (concept_matrix(scenario_id, "", "", threshold).get("available_types") or [])]
        rows = gaps_rows or (types[0] if types else "")
        cols = gaps_cols or next((t for t in types if t != rows), rows)
        if rows:
            matrix = concept_matrix(scenario_id, rows, cols, threshold)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"evidence_report matrix {scenario_id}: {e}")

    # L'entonnoir de screening (notices, doublons, écartés) vient des agrégats SQL du
    # brief structuré, qui les calcule déjà. Pas de second passage lourd, et pas de LLM.
    funnel = None
    try:
        from .evidence import _build_evidence_brief
        funnel = (_build_evidence_brief(scenario_id) or {}).get("corpus_stats")
    except Exception as e:                                   # noqa: BLE001
        logger.info(f"evidence_report funnel {scenario_id}: {e}")

    articles = _get_above_threshold_articles(scenario_id, full_rows=0) or []
    by_id = {a["id"]: a for a in articles if a.get("id") is not None}

    name = scenario_id
    query = None
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT name, query FROM user_scenarios WHERE id = :sid"),
                               {"sid": scenario_id}).mappings().first()
        if row:
            name, query = row["name"] or scenario_id, row["query"]
    except Exception as e:                                   # noqa: BLE001
        logger.info(f"evidence_report name {scenario_id}: {e}")

    # La langue du document : celle demandée, sinon celle du brief, sinon le français.
    _lang = _norm_lang(lang) or str((llm.get("_meta") or {}).get("lang") or "") or "fr"
    out = build_report(llm, digest, matrix, funnel, by_id, name, query, threshold,
                       rerank_threshold=_get_scenario_rerank_threshold(scenario_id),
                       lang=_lang)
    if download:
        return PlainTextResponse(
            out["markdown"], media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="evidence-report-{scenario_id}.md"'})
    return out
