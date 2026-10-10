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

from .core import app, engine, logger
from .scenario_store import (_get_scenario_rerank_threshold, _get_scenario_threshold,
                             _get_user_scenario_or_404)

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


def format_reference(n: int, article: dict) -> str:
    """One numbered reference, Vancouver-ish, from the DATABASE row.

    Not from the model: a generated bibliography is the single easiest thing in an
    evidence product to get confidently wrong, and the row is right there."""
    from .exports import _authors_list

    authors = _authors_list(article.get("authors") or "")
    if len(authors) > 6:
        who = ", ".join(authors[:6]) + ", et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = "[auteurs non renseignés]"
    # Le point après les auteurs n'est pas cosmétique : sans lui le titre s'enchaîne au
    # dernier auteur ("Roe B A trial of vaccination."), ce qui casse la lecture et tout
    # analyseur de bibliographie. `rstrip` évite le double point après « et al. ».
    bits = [f"{n}. {who.rstrip('.')}.",
            (article.get("title") or "[titre non renseigné]").rstrip(".") + "."]
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


def _rerank_clause(rerank_threshold: float | None) -> str:
    """La phrase qui dit le SECOND seuil, quand il est posé. La définition de « pertinent »
    de la section Méthodes ne nommait que la similarité : un seuil de rerank retirait des
    articles de l'ensemble analysé sans que le rapport le dise. Vide sans second seuil."""
    try:
        r = float(rerank_threshold or 0.0)
    except (TypeError, ValueError):
        return ""
    if r <= 0:
        return ""
    return (f" et, quand il a été calculé, dont le score de reclassement (cross-encoder) "
            f"atteint {r:.2f}")


def build_report(brief: dict, digest: dict, matrix: dict | None, funnel: dict | None,
                 articles_by_id: dict, scenario_name: str, query: str | None,
                 threshold: float, rerank_threshold: float | None = None) -> dict[str, Any]:
    """The whole document as markdown, plus what could not be resolved. Pure.

    Figures are numbered in the order they appear, as a paper numbers them, so the text
    can refer to "figure 2" and mean it."""
    numbers = {}
    references = []
    for article_id in cited_ids(brief):
        article = articles_by_id.get(article_id)
        if article is None:
            continue                                  # compté comme non résolu plus bas
        numbers[article_id] = len(references) + 1
        references.append(format_reference(len(references) + 1, article))

    unresolved: list[int] = []
    lines: list[str] = []
    meta = brief.get("_meta") or {}

    lines += [f"# {scenario_name}", ""]
    if meta.get("generated_at"):
        lines += [f"*Généré le {str(meta['generated_at'])[:19]}*", ""]

    # ── Méthodes ─────────────────────────────────────────────────────────────
    lines += ["## 1. Méthodes", ""]
    n_relevant = digest.get("n_articles") or 0
    lines += [
        f"Corpus interrogé par la requête booléenne suivante : `{query or 'non renseignée'}`.",
        "",
        f"Un article est *pertinent* lorsqu'il dépasse le seuil de similarité "
        f"({threshold:.2f}){_rerank_clause(rerank_threshold)} **ou** qu'un relecteur l'a "
        f"inclus à la main ; les articles "
        f"écartés au screening ne le sont jamais. La synthèse ci-dessous porte sur la "
        f"**totalité des {n_relevant} articles pertinents** : les agrégats (années, devis, "
        f"pays, revues, concepts, matrice de lacunes) sont calculés en SQL sur l'ensemble "
        f"complet, sans échantillonnage. Les articles reproduits intégralement dans le "
        f"prompt du générateur ne servent qu'à citer.",
        "",
    ]
    if digest.get("complete") is False:
        lines += ["> **Avertissement** : le calcul des agrégats a échoué, les totaux "
                  "ci-dessus ne décrivent pas le corpus complet.", ""]
    facts = []
    if digest.get("year_min") and digest.get("year_max"):
        facts.append(f"Années : {digest['year_min']} à {digest['year_max']}")
    if digest.get("n_included"):
        facts.append(f"Inclus par un relecteur : {digest['n_included']}")
    if digest.get("n_with_pico") is not None:
        facts.append(f"Avec PICO extrait : {digest['n_with_pico']}")
    if digest.get("n_with_fulltext") is not None:
        facts.append(f"Avec texte intégral : {digest['n_with_fulltext']}")
    if funnel:
        # Clés de `corpus_stats` (api/evidence.py), pas inventées ici.
        for label, key in (("Notices du corpus", "total"), ("Doublons", "duplicates"),
                           ("Écartés au screening", "excluded"),
                           ("En attente de screening", "pending")):
            if isinstance(funnel.get(key), int):
                facts.append(f"{label} : {funnel[key]}")
    if facts:
        lines += ["- " + "\n- ".join(facts), ""]
    model_bits = [f"modèle de rédaction `{meta.get('model', 'non renseigné')}`"]
    if meta.get("reasoning_effort"):
        model_bits.append(f"effort de raisonnement `{meta['reasoning_effort']}`")
    if meta.get("grade_ceiling"):
        model_bits.append(f"plafond de certitude du corpus `{meta['grade_ceiling']}`")
    lines += [f"Texte produit avec : {', '.join(model_bits)}. "
              "Les forces de preuve des affirmations ne sont pas écrites par le modèle : "
              "elles sont calculées d'après les devis des articles cités.", ""]

    # ── Résultats ────────────────────────────────────────────────────────────
    lines += ["## 2. Résultats", ""]
    for field, title in _NARRATIVE_FIELDS:
        value = brief.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        rewritten, missing = renumber(value, numbers)
        unresolved += [m for m in missing if m not in unresolved]
        lines += [f"### {title}", "", rewritten.strip(), ""]
    for field, title in _LIST_FIELDS:
        values = brief.get(field)
        if not isinstance(values, list) or not values:
            continue
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
                why.append("une seule étude, abaissé d'un niveau")
            if basis.get("capped_by_corpus"):
                why.append(f"plafonné par le corpus (devis cités : {basis.get('from_designs')})")
            rows.append([claim.get("claim", ""), claim.get("strength", ""),
                         " ; ".join(why), refs])
        lines += [f"### Figure {figure}. Affirmations et force des preuves", ""]
        lines += _table(["Affirmation", "Force", "Base du calcul", "Références"], rows)
        lines += ["", "La force est calculée, non déclarée : le devis le plus solide parmi "
                  "les articles cités fixe le niveau, une affirmation reposant sur une "
                  "seule étude perd un niveau, et le plafond du corpus s'applique en "
                  "dernier. Ce barème n'évalue ni le risque de biais, ni la cohérence, ni "
                  "la précision : ce n'est pas une évaluation GRADE complète.", ""]
        unverified = [str(x) for c in claims for x in (c.get("unverified_ids") or [])]
        if unverified:
            lines += [f"> **Citations non vérifiables** ({len(unverified)}) : "
                      f"{', '.join(unverified)}. Ces identifiants ne correspondent à aucun "
                      "article du corpus de ce scénario.", ""]
        figure += 1

    # ── Matrice de lacunes ───────────────────────────────────────────────────
    if matrix and matrix.get("rows") and matrix.get("cols"):
        cells = {(c["row"], c["col"]): c["n"] for c in matrix["cells"]}
        headers = [f"{matrix.get('row_type', '')} \\ {matrix.get('col_type', '')}"]
        headers += [c["label"] for c in matrix["cols"]]
        rows = [[r["label"]] + [str(cells.get((r["label"], c["label"]), 0) or "-")
                                for c in matrix["cols"]]
                for r in matrix["rows"]]
        lines += [f"### Figure {figure}. Matrice de lacunes "
                  f"({matrix.get('row_type')} x {matrix.get('col_type')})", ""]
        lines += _table(headers, rows)
        cov = matrix.get("coverage") or {}
        lines += ["", f"Nombre d'articles pertinents associant chaque concept en ligne à "
                  f"chaque concept en colonne, compté sur l'ensemble du sous-ensemble "
                  f"pertinent. Une case vide signifie qu'**aucun article de ce corpus** ne "
                  f"traite les deux ensemble ; ce n'est pas une preuve d'absence dans la "
                  f"littérature. Lisible sur {cov.get('with_concepts', 0)} des "
                  f"{cov.get('relevant', 0)} articles pertinents : les autres n'ont pas de "
                  f"concepts extraits et n'apparaissent pas dans cette matrice.", ""]
        if matrix.get("rows_total", 0) > len(matrix["rows"]) or \
                matrix.get("cols_total", 0) > len(matrix["cols"]):
            lines += [f"> Axes tronqués pour la lecture : "
                      f"{len(matrix['rows'])}/{matrix['rows_total']} lignes, "
                      f"{len(matrix['cols'])}/{matrix['cols_total']} colonnes. Les cases "
                      "hors de cette fenêtre ne sont pas rapportées comme lacunes.", ""]
        figure += 1

    # ── Conclusion et références ─────────────────────────────────────────────
    if brief.get("evidence_level") or brief.get("grade_recommendation"):
        lines += ["## 3. Niveau de preuve global", "",
                  f"- Niveau de preuve : {brief.get('evidence_level', 'non renseigné')}",
                  f"- Grade de recommandation : {brief.get('grade_recommendation', 'non renseigné')}",
                  ""]
    lines += ["## 4. Références", ""]
    if references:
        lines += references
    else:
        lines += ["*Aucune citation résolue dans ce brief.*"]
    lines += ["", f"Les {len(references)} références ci-dessus sont construites à partir "
              "des lignes du corpus, et non rédigées par le modèle. `[#id]` est "
              "l'identifiant interne de l'article, par lequel un relecteur retrouve la "
              "notice.", ""]
    if unresolved:
        lines += [f"> **Appels de citation non résolus** : {', '.join(str(u) for u in unresolved)}. "
                  "Ils apparaissent dans le texte suivis d'un `?` et ne correspondent à "
                  "aucun article du corpus.", ""]

    return {"markdown": "\n".join(lines).rstrip() + "\n",
            "references": len(references),
            "citations_unresolved": unresolved,
            "figures": figure - 1}


@app.get("/user-scenarios/{scenario_id}/evidence-report")
def evidence_report(scenario_id: str, download: bool = Query(False),
                    gaps_rows: str | None = Query(None),
                    gaps_cols: str | None = Query(None)) -> Any:
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

    out = build_report(llm, digest, matrix, funnel, by_id, name, query, threshold,
                       rerank_threshold=_get_scenario_rerank_threshold(scenario_id))
    if download:
        return PlainTextResponse(
            out["markdown"], media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="evidence-report-{scenario_id}.md"'})
    return out
