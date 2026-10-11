"""Variables and model spec generated from PICO, with localisation.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import _env_int, _job_is_active, _msg, _norm_lang, app, engine, logger, require_api_key
from .documents import _llm_lang_directive
from .scenario_store import _get_scenario_threshold, relevant_gate_sql, relevant_gate_tail_sql
from .gesica import _get_scenario_name
from .pico_map import CutAnswer, json_completion
from .relevance import _evidence_fingerprint, _get_above_threshold_articles
from llm_usage import LLMTruncated, json_content
from llm_usage import model_for as _model

# ─── VARIABLES & MODÈLE AUTO-REMPLI DEPUIS PICO ──────────────────────────────

_VARIABLES_GENERATION_JOBS: dict[str, dict] = {}


# ─── EXTRACTION CIBLÉE DES PARAMÈTRES ÉPIDÉMIOLOGIQUES (famille SEIR) ─────────
#
# Le spec est généré à partir des 25 articles LES PLUS PERTINENTS du corpus. Sur un
# corpus de plusieurs milliers d'articles, ceux qui RAPPORTENT un R0 ou une période
# d'incubation n'y figurent quasiment jamais : le bloc epidemic_parameters ressortait
# vide et l'onglet Modèle annonçait « aucun paramètre extrait de la littérature » pour
# un corpus chikungunya de 2 732 articles, où ces valeurs sont pourtant publiées.
#
# On CHERCHE donc, dans tout le sous-ensemble pertinent, les articles qui MESURENT un
# paramètre (termes de mesure dans le titre ou le résumé), et on n'extrait que sur
# ceux-là, une observation par étude. Le regroupement pondéré par la qualité
# (seir_model.pool_weighted, déjà en place) fait le reste : la valeur servie au modèle
# reste une moyenne d'études réelles, chacune cliquable.

# Termes de MESURE par paramètre (EN + FR). Une même source pour le pré-filtre SQL et
# le filtre Python : pas de divergence possible entre les deux.
_PARAM_PHRASES: dict[str, tuple[str, ...]] = {
    "r0": ("basic reproduction number", "reproduction number", "reproductive number",
           "reproduction ratio", "nombre de reproduction", "taux de reproduction"),
    "serial_interval_days": ("serial interval", "generation interval", "generation time",
                             "intervalle sériel", "intervalle de génération"),
    "incubation_period_days": ("incubation period", "période d'incubation", "periode d'incubation",
                               "latent period", "période de latence"),
    "infectious_period_days": ("infectious period", "infectiousness duration", "duration of infectiousness",
                               "shedding duration", "duration of viremia", "période infectieuse",
                               "durée d'infectiosité"),
    "cfr": ("case fatality", "case-fatality", "fatality rate", "fatality ratio",
            "létalité", "letalite", "taux de décès", "death rate among cases"),
    "immunity_duration_days": ("duration of immunity", "immunity duration", "waning immunity",
                               "duration of protection", "durée de l'immunité", "durée de protection",
                               "seroprotection duration"),
}
# Sigles courts : exigent des frontières de mot, sinon « R0 » matche « macro0 ».
_PARAM_TOKENS: dict[str, tuple[str, ...]] = {
    "r0": ("R0", "R-0", "R_0", "Rzero"),
}
# Paramètres qui PILOTENT la dynamique : un seul d'entre eux, mesuré dans le corpus,
# suffit à démontrer que le scénario est transmissible (cf. _merge_epidemic_observations).
_TRANSMISSION_PARAMS = ("r0", "serial_interval_days", "incubation_period_days")

# Aucun plafond par défaut : l'extraction lit TOUS les articles pertinents qui mesurent
# un paramètre, jamais un échantillon. EPI_PARAM_MAX_ARTICLES > 0 en pose un (secours
# d'exploitation si le budget LLM doit être tenu un jour donné).
EPI_PARAM_MAX_ARTICLES = _env_int("EPI_PARAM_MAX_ARTICLES", 0, 0)
_EPI_PARAM_BATCH = 10
_EPI_PARAM_WORKERS = 4
# Les passages du texte intégral envoyés avec le titre et le résumé : une fenêtre autour de
# chaque terme de mesure (la valeur suit d'ordinaire le terme), quelques milliers de
# caractères au plus par article.
_EPI_WINDOW_BEFORE = 300
_EPI_WINDOW_AFTER = 600
_EPI_PASSAGES_CHARS = 3000


def _param_regex(params=None, boundary: str = r"\b") -> str:
    """Alternation regex des termes de mesure (paramètres demandés, ou tous). `boundary`
    vaut r'\\b' pour Python et r'\\y' pour Postgres. Pur."""
    names = tuple(params or _PARAM_PHRASES)
    parts: list[str] = []
    for name in names:
        parts.extend(re.escape(p) for p in _PARAM_PHRASES.get(name, ()))
        parts.extend(f"{boundary}{re.escape(t)}{boundary}" for t in _PARAM_TOKENS.get(name, ()))
    return "|".join(parts)


_PARAM_SCAN_RE = re.compile(_param_regex(), re.IGNORECASE)


def params_mentioned(text: str) -> list[str]:
    """Paramètres épidémiologiques dont un terme de MESURE apparaît dans ce texte
    (titre + résumé + passages du texte intégral), dans l'ordre de _PARAM_PHRASES. Pur,
    testé hors ligne."""
    t = text or ""
    found = []
    for name in _PARAM_PHRASES:
        if re.search(_param_regex([name]), t, re.IGNORECASE):
            found.append(name)
    return found


def _parameter_candidate_articles(scenario_id: str, threshold: float | None = None,
                                  limit: int = 0) -> list[dict]:
    """TOUS les articles du sous-ensemble PERTINENT qui rapportent un paramètre (terme de
    mesure dans le titre, le résumé OU le texte intégral), les meilleurs d'abord : synthèses
    en tête, puis qualité, citations et année. Même porte de screening que partout ailleurs.
    `limit <= 0` : aucun plafond (le cas par défaut).

    Il fallait un résumé, et le terme dans le titre ou le résumé : un article qui ne donne
    son R0 ou sa période d'incubation que dans ses résultats n'était jamais lu, et un
    article sans résumé non plus. Ceux dont le texte intégral porte un terme de mesure
    reçoivent les passages qui l'entourent (`fulltext_passages`)."""
    if threshold is None:
        threshold = _get_scenario_threshold(scenario_id)
    sql = text(f"""
        SELECT d.id, d.title, d.abstract, d.year, d.doi, d.study_design,
               d.quality_score, d.citation_count, d.source,
               -- Le résultat DÉJÀ extrait pour cet article, s'il existe : la règle de la
               -- maison veut que le travail par article soit mis en cache sur sa ligne,
               -- et que seule la réduction soit refaite. Cette extraction ne cachait
               -- rien : chaque régénération des variables renvoyait au modèle les mêmes
               -- résumés, 21 appels de dix résumés de 2 000 caractères sur un scénario
               -- de 9 389 articles, en entier, à chaque fois.
               d.epi_params_json,
               COALESCE(ars.similarity_score, 0) AS similarity,
               ft.in_fulltext
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        CROSS JOIN LATERAL (
            SELECT EXISTS (SELECT 1 FROM document_chunk c
                           WHERE c.document_id = d.id
                             AND c.chunk_type IN ('fulltext_section', 'full_text')
                             AND c.content ~* :rx) AS in_fulltext
        ) ft
        WHERE ars.scenario_id = :sid
          AND d.is_duplicate IS NOT TRUE
          AND {relevant_gate_tail_sql('d', 'ars', ':thr')}
          AND ((COALESCE(d.title, '') || ' ' || COALESCE(d.abstract, '')) ~* :rx OR ft.in_fulltext)
        ORDER BY
          CASE WHEN COALESCE(d.study_design, '') ~* 'systematic|meta-analy|méta-analy' THEN 0 ELSE 1 END,
          d.quality_score DESC NULLS LAST,
          d.citation_count DESC NULLS LAST,
          d.year DESC NULLS LAST,
          d.id
        LIMIT :cap
    """)
    # NULL en LIMIT = pas de limite en SQL : le défaut lit tout le corpus pertinent.
    _cap = int(limit) if int(limit or 0) > 0 else None
    rx = _param_regex(boundary=r"\y")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            sql, {"sid": scenario_id, "thr": threshold, "rx": rx, "cap": _cap}).mappings().all()]
        passages = _fulltext_passages(conn, [r["id"] for r in rows if r.get("in_fulltext")], rx)
    for r in rows:
        r["fulltext_passages"] = passages.get(r["id"], "")
        # Ce que le modèle lira de cet article : la clé de son cache (`_epi_cache_hit`).
        r["epi_source"] = "fulltext" if r["fulltext_passages"] else "abstract"
        r["params_mentioned"] = params_mentioned(
            f"{r.get('title') or ''} {r.get('abstract') or ''} {r['fulltext_passages']}")
    return [r for r in rows if r["params_mentioned"]]


def _fulltext_passages(conn, ids: list[int], rx: str) -> dict[int, str]:
    """{id: passages} : pour chaque article, le texte intégral autour de chaque terme de
    mesure (une fenêtre par occurrence, les fenêtres qui se chevauchent fusionnées), au
    plus `_EPI_PASSAGES_CHARS` caractères. Le type de morceau actuel d'abord, l'ancien à
    défaut, pour ne pas lire deux fois le même texte."""
    if not ids:
        return {}
    rows = conn.execute(text("""
        SELECT document_id, chunk_type, content
        FROM document_chunk
        WHERE document_id = ANY(CAST(:ids AS bigint[]))
          AND chunk_type IN ('fulltext_section', 'full_text')
          AND content ~* :rx
        ORDER BY document_id, (chunk_type = 'fulltext_section') DESC, chunk_index, id
    """), {"ids": list(ids), "rx": rx}).all()
    by_doc: dict[int, list[tuple[str, str]]] = {}
    for doc_id, kind, content in rows:
        by_doc.setdefault(int(doc_id), []).append((kind, content or ""))
    out: dict[int, str] = {}
    for doc_id, chunks in by_doc.items():
        kind = chunks[0][0]
        kept: list[str] = []
        used = 0
        for k, content in chunks:
            if k != kind:
                continue
            body = " ".join(content.split())
            spans: list[list[int]] = []
            for m in _PARAM_SCAN_RE.finditer(body):
                lo, hi = max(0, m.start() - _EPI_WINDOW_BEFORE), min(len(body), m.end() + _EPI_WINDOW_AFTER)
                if spans and lo <= spans[-1][1]:
                    spans[-1][1] = max(spans[-1][1], hi)
                else:
                    spans.append([lo, hi])
            for lo, hi in spans:
                window = body[lo:hi].strip()
                # Les morceaux se recouvrent : une fenêtre déjà lue n'est pas reprise, et
                # celle qui la prolonge (coupée au bout du morceau précédent) la remplace.
                if not window or any(window in w for w in kept):
                    continue
                shorter = next((j for j, w in enumerate(kept) if w in window), None)
                if shorter is not None:
                    used -= len(kept.pop(shorter))
                room = _EPI_PASSAGES_CHARS - used
                if room < 200:
                    break
                if len(window) > room:
                    window = window[:room].rsplit(" ", 1)[0]
                kept.append(window)
                used += len(window)
            if _EPI_PASSAGES_CHARS - used < 200:
                break
        if kept:
            out[doc_id] = " [...] ".join(kept)
    return out


_EPI_EXTRACT_SYSTEM = (
    "You extract epidemiological parameters from study abstracts and, when an article "
    "carries them, from passages of its full text around the measurement terms, for a "
    "compartmental (SEIR family) model. Report ONLY values the text states for the disease "
    "under study: never infer, never carry a value over from another disease, never invent. "
    "Normalise units: r0 is a ratio; serial_interval_days, incubation_period_days, "
    "infectious_period_days and immunity_duration_days are in DAYS; cfr is a PROPORTION "
    "between 0 and 1 (a case fatality of 1.5 percent is 0.015). When a study gives a "
    "range or an interval, put its central value in value and the bounds in ci_low and "
    "ci_high. Return ONLY JSON: {\"articles\": [{\"id\": <id>, \"disease\": <name or null>, "
    "\"parameters\": [{\"name\": <one of r0, serial_interval_days, incubation_period_days, "
    "infectious_period_days, cfr, immunity_duration_days>, \"value\": <number>, "
    "\"ci_low\": <number or null>, \"ci_high\": <number or null>}]}]}. An article that "
    "reports no usable value gets an empty parameters list."
)


def _epi_llm_client():
    """Client LLM de l'extraction (une seule couture, que les tests remplacent)."""
    from llm_usage import MeteredOpenAI as _OAI
    return _OAI(timeout=120.0)


def _epi_extract_batch(client, batch: list[dict], disease_hint: str | None) -> list[dict]:
    """Un appel LLM pour un lot d'articles. Renvoie [{id, disease, parameters:[...]}].

    LÈVE en cas d'échec, et `LLMTruncated` quand la réponse est coupée au plafond : c'est
    `_epi_extract_resilient` qui décide (scinder le lot, le compter perdu). Le lot perdu
    renvoyait [], indistinguable d'un lot sans valeur, et un lot coupé l'était de nouveau,
    identique, à chaque régénération."""
    import json as _json
    items = []
    for a in batch:
        item = {
            "id": int(a["id"]),
            "title": (a.get("title") or "")[:300],
            "abstract": (a.get("abstract") or "")[:2000],
            "looks_like": a.get("params_mentioned") or [],
        }
        if a.get("fulltext_passages"):
            item["fulltext_passages"] = a["fulltext_passages"]
        items.append(item)
    payload = {"disease_of_interest": disease_hint or None, "articles": items}
    resp = client.chat.completions.create(
        model=_model("bulk"),
        messages=[{"role": "system", "content": _EPI_EXTRACT_SYSTEM},
                  {"role": "user", "content": _json.dumps(payload, ensure_ascii=False)}],
        temperature=0, seed=42, max_tokens=3000,
        response_format={"type": "json_object"},
    )
    data = _json.loads(json_content(resp, f"epidemic parameters ({len(batch)} articles)"))
    return [a for a in (data.get("articles") or []) if isinstance(a, dict)] if isinstance(data, dict) else []


def _epi_extract_resilient(client, batch: list[dict], disease_hint: str | None) -> dict[str, Any]:
    """Un lot, repris s'il est coupé : scindé en deux moitiés, chacune demandée une fois.

    Renvoie {articles, failed_ids, split}. `failed_ids` : les articles qu'aucune réponse
    n'a couverts (lot ou moitié en échec), comptés et rapportés au lieu de disparaître."""
    ids = [int(a["id"]) for a in batch]
    try:
        return {"articles": _epi_extract_batch(client, batch, disease_hint),
                "failed_ids": [], "split": False}
    except LLMTruncated as cut:
        if len(batch) < 2:
            logger.warning(f"epidemic parameter extraction: article {ids} {cut}")
            return {"articles": [], "failed_ids": ids, "split": False}
        logger.warning(f"epidemic parameter extraction: lot de {len(batch)} coupé au plafond, "
                       f"scindé en deux.")
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"epidemic parameter extraction batch: {e}")
        return {"articles": [], "failed_ids": ids, "split": False}
    out: dict[str, Any] = {"articles": [], "failed_ids": [], "split": True}
    half = (len(batch) + 1) // 2
    for part in (batch[:half], batch[half:]):
        try:
            out["articles"] += _epi_extract_batch(client, part, disease_hint)
        except Exception as e:                               # noqa: BLE001 - coupé encore, ou panne
            logger.warning(f"epidemic parameter extraction, demi-lot de {len(part)}: {e}")
            out["failed_ids"] += [int(a["id"]) for a in part]
    return out


#: Version du format mis en cache. L'incrémenter invalide les lignes déjà extraites.
EPI_PARAM_VERSION = 1


def _epi_cache_hit(row: dict, disease_hint: str | None) -> dict | None:
    """Le résultat en cache de cet article, s'il est utilisable.

    Un résultat VIDE est un fait, pas une absence : « cet article mentionne un paramètre
    et n'en donne aucune valeur exploitable » a coûté un appel au modèle et vaut d'être
    gardé. Le cache porte l'indice de maladie qui a servi, parce que la réponse en
    dépend : un autre indice est un défaut de cache, pas une erreur."""
    cached = row.get("epi_params_json")
    if not isinstance(cached, dict):
        return None
    if int(cached.get("v") or 0) != EPI_PARAM_VERSION:
        return None
    if (cached.get("hint") or None) != (disease_hint or None):
        return None
    # Ce que le modèle avait lu : un résultat tiré du seul résumé ne vaut plus quand des
    # passages du texte intégral sont arrivés depuis (la valeur est souvent là). Les
    # entrées écrites avant ce champ venaient toutes du résumé.
    if (cached.get("src") or "abstract") != (row.get("epi_source") or "abstract"):
        return None
    return {"id": int(row["id"]), "disease": cached.get("disease"),
            "parameters": cached.get("parameters") or []}


def _epi_cache_write(results: list[dict], sources: dict[int, str],
                     disease_hint: str | None) -> int:
    """Écrit sur la ligne de chaque article ce que le modèle en a dit. Une fois pour
    toutes : la réduction (agrégation sur tout le sous-ensemble pertinent) se refait sans
    rien repayer. `sources` : {id candidat: ce que le modèle a lu, abstract ou fulltext}."""
    import json as _json
    rows = [{"id": int(r["id"]),
             "j": _json.dumps({"v": EPI_PARAM_VERSION, "hint": disease_hint or None,
                               "src": sources.get(int(r["id"])) or "abstract",
                               "disease": r.get("disease"),
                               "parameters": r.get("parameters") or []}, ensure_ascii=False)}
            for r in results if _as_int(r.get("id")) in sources]
    if not rows:
        return 0
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE literature_document
                SET epi_params_json = CAST(:j AS jsonb), epi_params_at = NOW()
                WHERE id = :id
            """), rows)
    except Exception as _e:                                  # noqa: BLE001 - jamais bloquant
        logger.warning(f"cache des paramètres épidémiologiques: {_e}")
        return 0
    return len(rows)


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def extract_epidemic_observations(scenario_id: str, disease_hint: str | None = None,
                                  threshold: float | None = None,
                                  max_articles: int = 0) -> dict[str, Any]:
    """Cherche dans TOUT le corpus pertinent les articles qui rapportent un paramètre
    épidémiologique et en extrait une observation par étude. Aucun échantillonnage :
    `max_articles <= 0` (le défaut) lit tous les articles qui en mesurent un.

    Renvoie ``{params: {nom: {observations: [{article_id, value, ci_low, ci_high}]}},
    disease, n_candidates, n_articles_used, n_with_values}``. Sans clé OpenAI :
    les candidats sont comptés, aucune observation n'est produite."""
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    candidates = _parameter_candidate_articles(scenario_id, threshold, max_articles or 0)
    # `articles` : de quoi TRACER la provenance. Ces articles sont hors des 25 plus
    # pertinents qui servent au reste du spec ; sans eux dans le pool de provenance,
    # _attach_model_spec filtrerait justement les observations qu'on vient de mesurer.
    out: dict[str, Any] = {
        "params": {}, "disease": None, "n_candidates": len(candidates),
        "n_from_fulltext": sum(1 for a in candidates if a.get("fulltext_passages")),
        "n_articles_used": 0, "n_with_values": 0,
        # Ce que l'extraction a FAIT, pas seulement ce qu'elle a trouvé. Sans ces
        # compteurs, une panne complète du modèle se lisait comme un constat sur la
        # littérature.
        "n_batches": 0, "n_batches_failed": 0, "n_batches_split": 0, "n_from_cache": 0,
        "n_not_examined": 0, "failed_article_ids": [],
        "status": "no_key" if not _os.getenv("OPENAI_API_KEY") else "ok",
        "articles": [{"id": a["id"], "title": a.get("title"), "year": a.get("year"),
                      "doi": a.get("doi"), "quality_score": a.get("quality_score"),
                      "study_design": a.get("study_design"), "source": a.get("source"),
                      "params_mentioned": a.get("params_mentioned"),
                      "from_fulltext": bool(a.get("fulltext_passages"))} for a in candidates],
    }
    if not candidates or not _os.getenv("OPENAI_API_KEY"):
        return out
    try:
        client = _epi_llm_client()
    except Exception as _e:                                  # noqa: BLE001 - SDK absent ou clé illisible
        logger.warning(f"Extraction des paramètres {scenario_id}: client LLM indisponible ({_e})")
        return out                                           # les candidats restent comptés
    # ── map : ce qui est déjà extrait n'est pas redemandé ────────────────────
    cached, to_ask = [], []
    for a in candidates:
        hit = _epi_cache_hit(a, disease_hint)
        (cached.append(hit) if hit else to_ask.append(a))
    batches = [to_ask[i:i + _EPI_PARAM_BATCH] for i in range(0, len(to_ask), _EPI_PARAM_BATCH)]
    valid_ids = {int(a["id"]) for a in candidates}
    diseases: dict[str, int] = {}
    fresh: list[list[dict]] = []
    runs: list[dict] = []
    if batches:
        with ThreadPoolExecutor(max_workers=_EPI_PARAM_WORKERS) as ex:
            runs = list(ex.map(lambda b: _epi_extract_resilient(client, b, disease_hint), batches))
        fresh = [r["articles"] for r in runs]
    # ── Ce que les lots ont fait, compté ─────────────────────────────────────
    # Un lot en échec (quota, délai, clé invalide, réponse malformée) renvoyait [] et
    # devenait indistinguable d'un lot où aucun article ne donne de valeur. Sur une
    # panne complète, le relecteur lisait « aucune valeur exploitable dans les articles
    # qui mentionnent un paramètre » : une affirmation définitive sur 205 articles que
    # le modèle n'a jamais examinés. Un lot coupé au plafond est désormais scindé ; ce qui
    # échoue encore est compté, lot par lot et article par article.
    _failed = sum(1 for r in runs if r["failed_ids"])
    out["n_batches"] = len(batches)
    out["n_batches_failed"] = _failed
    out["n_batches_split"] = sum(1 for r in runs if r["split"])
    out["n_from_cache"] = len(cached)
    out["failed_article_ids"] = sorted({i for r in runs for i in r["failed_ids"]})
    _asked_ids = {int(a["id"]) for b in batches for a in b}
    _answered_ids = {i for r in fresh for a in r if (i := _as_int(a.get("id"))) is not None}
    out["n_not_examined"] = len(_asked_ids - _answered_ids)
    if _failed:
        logger.warning(f"Paramètres épidémiologiques {scenario_id}: {_failed}/{len(batches)} "
                       f"lots en échec ; {out['n_not_examined']} articles non examinés.")
    _written = _epi_cache_write([a for r in fresh for a in r],
                                {int(a["id"]): a.get("epi_source") or "abstract" for a in candidates},
                                disease_hint)
    if _written:
        logger.info(f"Paramètres épidémiologiques {scenario_id}: {_written} articles mis en "
                    f"cache, {len(cached)} relus depuis le cache.")
    import seir_model as _seir
    for res in [cached] + fresh:
        for art in res:
            try:
                aid = int(art.get("id"))
            except (TypeError, ValueError):
                continue
            if aid not in valid_ids:
                continue                                     # id inventé : ignoré
            got = False
            for p in (art.get("parameters") or []):
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                val = _seir._num_or_none(p.get("value"))
                if name not in _PARAM_PHRASES or val is None:
                    continue
                obs = {"article_id": aid, "value": val}
                lo, hi = _seir._num_or_none(p.get("ci_low")), _seir._num_or_none(p.get("ci_high"))
                if lo is not None and hi is not None and lo < hi:
                    obs["ci_low"], obs["ci_high"] = lo, hi
                out["params"].setdefault(name, {"observations": []})["observations"].append(obs)
                got = True
            if got:
                out["n_with_values"] += 1
                _d = str(art.get("disease") or "").strip()
                if _d:
                    diseases[_d] = diseases.get(_d, 0) + 1
    # Les articles réellement examinés : ceux relus du cache, plus ceux dont un lot a
    # répondu. Il annonçait `len(candidates)` quoi qu'il arrive, donc 205 articles
    # examinés sur une panne où aucun ne l'avait été.
    out["n_articles_used"] = len(cached) + len(_answered_ids)
    # Indisponible quand AUCUN article demandé n'a reçu de réponse ; partiel quand une
    # partie seulement (un demi-lot coupé deux fois ne rend pas le reste du lot muet).
    out["status"] = ("llm_unavailable" if _asked_ids and not (_answered_ids & _asked_ids)
                     else "llm_partial" if _failed or out["n_not_examined"]
                     else "ok")
    if diseases:
        out["disease"] = max(diseases.items(), key=lambda kv: kv[1])[0]
    logger.info(f"Paramètres épidémiologiques {scenario_id}: {len(candidates)} articles candidats, "
                f"{out['n_with_values']} avec une valeur, paramètres={sorted(out['params'])}")
    return out


def _epi_meta(targeted: dict) -> dict[str, Any]:
    """Les compteurs de l'extraction ciblée tels que le spec les garde dans `_meta` : ce
    qu'elle a trouvé, ET ce qu'elle n'a pas pu lire (lots perdus, articles non examinés)."""
    return {
        "epidemic_parameter_candidates": int(targeted.get("n_candidates", 0)),
        "epidemic_parameter_articles_with_values": int(targeted.get("n_with_values", 0)),
        "epidemic_parameter_from_fulltext": int(targeted.get("n_from_fulltext", 0)),
        "epidemic_parameter_batches": int(targeted.get("n_batches", 0)),
        "epidemic_parameter_batches_failed": int(targeted.get("n_batches_failed", 0)),
        "epidemic_parameter_batches_split": int(targeted.get("n_batches_split", 0)),
        "epidemic_parameter_not_examined": int(targeted.get("n_not_examined", 0)),
        "epidemic_parameter_failed_article_ids": list(targeted.get("failed_article_ids") or []),
    }


def merge_epidemic_observations(block, targeted) -> dict:
    """Fusionne les observations CIBLÉES dans le bloc `epidemic_parameters` du LLM
    narratif (PUR, testé hors ligne).

    Les observations ciblées viennent d'articles qui MESURENT le paramètre : elles
    s'ajoutent à celles du bloc narratif, dédupliquées par (paramètre, article). Un
    paramètre absent du bloc narratif est créé. `applicable` passe à true lorsqu'un
    paramètre de TRANSMISSION est mesuré par au moins un article : le corpus le
    démontre, quoi qu'ait répondu le premier appel sur ses 25 articles."""
    merged = dict(block) if isinstance(block, dict) else {}
    tparams = (targeted or {}).get("params") or {}
    for name, blk in tparams.items():
        cur = dict(merged.get(name)) if isinstance(merged.get(name), dict) else {}
        obs = [o for o in (cur.get("observations") or []) if isinstance(o, dict)]
        seen = {(o.get("article_id"), o.get("value")) for o in obs}
        for o in blk.get("observations") or []:
            if (o.get("article_id"), o.get("value")) not in seen:
                obs.append(o)
                seen.add((o.get("article_id"), o.get("value")))
        cur["observations"] = obs
        # Provenance = les articles réellement mesurés (l'UI les rend cliquables).
        prov = list(dict.fromkeys([o["article_id"] for o in obs if o.get("article_id") is not None]))
        if prov:
            cur["provenance"] = prov
            cur["n_studies"] = len(prov)
        cur.setdefault("unit", "ratio" if name == "r0" else ("proportion" if name == "cfr" else "days"))
        merged[name] = cur
    if any(tparams.get(p, {}).get("observations") for p in _TRANSMISSION_PARAMS):
        merged["applicable"] = True
    if (targeted or {}).get("disease") and not merged.get("population_disease"):
        merged["population_disease"] = targeted["disease"]
    return merged


# ─── MODEL SPEC (Phase 1) : schéma machine + provenance ──────────────────────
# La littérature définit la SPÉCIFICATION du modèle (outcome, variables
# explicatives, algorithme). Les données d'entraînement viendront ensuite de
# l'utilisateur (CSV/XLSX) et de flux publics. Ces helpers normalisent la sortie
# LLM en un spec déterministe, exploitable par la machine, et tracé (provenance)
# vers les articles sources. Tout est ADDITIF : les clés existantes de
# variables_json restent intactes pour ne pas casser le frontend.

MODEL_SPEC_SCHEMA = "model_spec/1.0"

_TASK_TYPES = {"classification", "regression", "count", "survival"}
_DTYPES = {"float", "int", "bool", "category", "datetime"}
_FEATURE_SOURCES = {"user", "public_api", "seir"}
_ALGO_FAMILIES = {
    "gradient_boosting", "lightgbm", "xgboost", "random_forest", "logistic_regression",
    "linear_regression", "elasticnet", "svm", "mlp", "cox_ph", "knn",
    "prophet", "sarimax", "extremal_rf",
}
# Familles de PRÉVISION de série temporelle (routées hors du flux tabulaire par
# model_trainer). Une cible numérique s'impose → task_type ramené à 'regression'.
_TS_ALGO_FAMILIES = {"prophet", "sarimax"}
_METRICS = {"roc_auc", "average_precision", "rmse", "mae", "r2", "c_index"}
_CV_STRATEGIES = {"stratified_kfold", "kfold", "timeseries"}


def _slug_identifier(name: str, used: set[str]) -> str:
    """snake_case, identifiant valide et unique (pour colonnes CSV/DataFrame)."""
    import re as _re
    import unicodedata as _ud
    # Replier les accents (é -> e) avant de slugifier, sinon ils deviennent des "_".
    folded = _ud.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    base = _re.sub(r"[^a-z0-9]+", "_", folded.strip().lower()).strip("_")
    if not base or not _re.match(r"^[a-z_]", base):
        base = ("var_" + base).strip("_") if base else "var"
    candidate, i = base, 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return v if v in allowed else default


def _infer_algo_family(text_blob: str) -> str:
    t = (text_blob or "").lower()
    table = [
        (("extremal", "quantile forest", "quantile regression forest", "forêt quantile", "foret quantile", "surge", "extreme quantile"), "extremal_rf"),
        (("xgboost", "lightgbm", "gradient boost", "gradient_boost", "boosting", "gbm"), "gradient_boosting"),
        (("random forest", "random_forest", "forêt aléatoire", "foret aleatoire"), "random_forest"),
        (("logistic", "logistique"), "logistic_regression"),
        (("cox", "proportional hazard", "survie", "survival"), "cox_ph"),
        (("ridge", "lasso", "elastic"), "elasticnet"),
        (("linear regression", "régression linéaire", "regression lineaire", "ols", "moindres carrés"), "linear_regression"),
        (("svm", "support vector"), "svm"),
        (("neural", "mlp", "deep", "réseau de neur", "reseau de neur"), "mlp"),
        (("knn", "nearest neighbor", "plus proches voisins"), "knn"),
    ]
    for keys, fam in table:
        if any(k in t for k in keys):
            return fam
    return "gradient_boosting"


def _dtype_for_var_type(var_type: Any) -> str:
    return {
        "continuous": "float", "binary": "bool", "categorical": "category",
        "time_series": "float", "count": "int", "integer": "int",
    }.get(str(var_type or "").strip().lower(), "float")


def _infer_feature_source(data_source: str, declared: str) -> tuple[str, str | None]:
    """Retourne (source, public_provider) - déclaré LLM sinon heuristique texte."""
    declared = (declared or "").strip().lower()
    if declared in _FEATURE_SOURCES:
        src = declared
    else:
        ds = (data_source or "").lower()
        public_hint = any(k in ds for k in (
            "météo", "meteo", "weather", "temperature", "température", "forecast",
            "prévision", "prevision", "open-meteo", "openmeteo", "insee", "open data",
            "opendata", "données publiques", "donnees publiques", "santé publique",
            "sentinel", "réseau sentinelles", "reseau sentinelles", "pollen", "air quality",
        ))
        src = "public_api" if public_hint else "user"
    provider = None
    if src == "public_api":
        ds = (data_source or "").lower()
        if any(k in ds for k in ("météo", "meteo", "weather", "temp", "forecast", "prévision", "prevision", "open-meteo")):
            provider = "open-meteo"
    return src, provider


def _filter_provenance(raw: Any, valid_ids: set) -> list:
    """Ne garde que les ids réellement présents dans le contexte fourni au LLM."""
    out: list = []
    for x in (raw or []):
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid_ids and xi not in out:
            out.append(xi)
    return out


_TARGET_DTYPE_FOR_TASK = {"classification": "category", "regression": "float", "count": "int", "survival": "float"}


def _derive_data_template(outcome: dict, features: list[dict]) -> dict:
    """Dérive le data_template (noms de colonnes EXACTS attendus à l'upload) depuis
    l'outcome + les features. Fonction PURE partagée par le générateur de spec et
    l'éditeur de spec, pour que le template ne DÉRIVE JAMAIS de la liste réelle des
    variables (sinon la validation d'upload attendrait des colonnes fantômes)."""
    task_type = (outcome.get("task_type") or "classification").strip().lower()
    outcome_mn = outcome.get("machine_name") or "outcome"
    target_dtype = _TARGET_DTYPE_FOR_TASK.get(task_type, "float")
    columns = [{
        "name": outcome_mn, "dtype": target_dtype, "role": "outcome",
        "required": True, "source": "user", "description": outcome.get("name", ""),
    }]
    for f in features:
        col = {
            "name": f["machine_name"], "dtype": f["dtype"], "role": "feature",
            "required": f.get("importance") == "high",
            "source": f.get("source", "user"), "public_provider": f.get("public_provider"),
            "description": f.get("name", ""),
        }
        if f.get("seir_column"):
            col["seir_column"] = f["seir_column"]  # dérivée du sous-modèle SEIR
        columns.append(col)
    return {
        "target_column": outcome_mn,
        "columns": columns,
        "formats": ["csv", "xlsx"],
        "user_columns": [c["name"] for c in columns if c["source"] == "user"],
        "public_columns": [c["name"] for c in columns if c["source"] == "public_api"],
        # Colonnes dérivées du sous-modèle SEIR (remplies automatiquement, pas d'upload).
        "seir_columns": [c["name"] for c in columns if c["source"] == "seir"],
        "notes": ("Les en-têtes du fichier doivent correspondre EXACTEMENT à ces noms. "
                  "Les colonnes 'public_api' pourront être récupérées automatiquement (Phase 2). "
                  "Les colonnes 'seir' sont DÉRIVÉES du sous-modèle épidémique (aucun upload)."),
    }


def _coerce_family_for_task(family: Any, task_type: str) -> str:
    """Ramène une famille d'algorithme choisie vers une famille COMPATIBLE avec la
    tâche (sans vérifier la présence du paquet - c'est le rôle du trainer). Ex.:
    logistic_regression sur une régression -> linear_regression, et inversement."""
    fam = _coerce_enum(family, _ALGO_FAMILIES, "gradient_boosting")
    tt = (task_type or "classification").strip().lower()
    if tt in ("regression", "count"):
        if fam in ("logistic_regression",):
            return "linear_regression"
    else:  # classification / survival
        if fam in ("linear_regression", "elasticnet"):
            return "logistic_regression"
    return fam


def _attach_model_spec(variables: dict, prov_articles: list[dict]) -> dict:
    """
    Construit un `model_spec` déterministe (outcome, features, algorithme,
    data_template) + un index de provenance, à partir de la sortie LLM.
    Robuste : si le LLM omet les champs machine, ils sont reconstruits depuis
    les champs humains existants. N'altère aucune clé existante (ajoute
    seulement `machine_name` en cross-link et les blocs `model_spec`/_provenance_index).
    """
    valid_ids = {a["id"] for a in prov_articles if a.get("id") is not None}
    prov_meta = {
        a["id"]: {"title": (a.get("title") or "")[:160], "year": a.get("year"), "doi": a.get("doi")}
        for a in prov_articles if a.get("id") is not None
    }
    # Poids qualité par article (quality_score déterministe du corpus) → regroupement
    # pondéré des paramètres épidémiologiques par étude (seir_model.pool_weighted).
    quality_by_id = {
        a["id"]: a.get("quality_score")
        for a in prov_articles if a.get("id") is not None
    }
    used: set[str] = set()
    cited: set = set()

    # ── Outcome ──
    po = variables.get("primary_outcome") or {}
    outcome_mn = _slug_identifier(po.get("machine_name") or po.get("name") or "outcome", used)
    task_type = _coerce_enum(po.get("task_type"), _TASK_TYPES, "classification")
    outcome_prov = _filter_provenance(po.get("provenance"), valid_ids)
    cited.update(outcome_prov)
    po["machine_name"] = outcome_mn  # cross-link additif
    po["task_type"] = task_type
    outcome = {
        "name": po.get("name", ""),
        "machine_name": outcome_mn,
        "task_type": task_type,
        "unit": po.get("unit") or po.get("measurement") or "",
        "positive_class": po.get("positive_class") if task_type == "classification" else None,
        "provenance": outcome_prov,
    }

    # ── Features ──
    features = []
    has_time_series = False
    for pv in (variables.get("predictor_variables") or []):
        mn = _slug_identifier(pv.get("machine_name") or pv.get("name") or "feature", used)
        dtype = _coerce_enum(pv.get("dtype"), _DTYPES, _dtype_for_var_type(pv.get("type")))
        source, provider = _infer_feature_source(pv.get("data_source", ""), pv.get("source", ""))
        prov = _filter_provenance(pv.get("provenance"), valid_ids)
        cited.update(prov)
        if str(pv.get("type", "")).strip().lower() == "time_series" or dtype == "datetime":
            has_time_series = True
        pv["machine_name"] = mn  # cross-link additif
        features.append({
            "name": pv.get("name", ""),
            "machine_name": mn,
            "dtype": dtype,
            "source": source,
            "public_provider": provider,
            "importance": _coerce_enum(pv.get("importance"), {"high", "medium", "low"}, "medium"),
            "provenance": prov,
        })

    # ── Algorithme ──
    ra = variables.get("recommended_algorithm") or {}
    family = _coerce_enum(
        ra.get("family"), _ALGO_FAMILIES,
        _infer_algo_family(f"{ra.get('primary', '')} {' '.join(ra.get('alternatives') or [])}"),
    )
    metric = _coerce_enum(
        ra.get("metric"), _METRICS,
        {"classification": "roc_auc", "regression": "rmse", "count": "rmse", "survival": "c_index"}[task_type],
    )
    default_cv = "timeseries" if has_time_series else ("stratified_kfold" if task_type == "classification" else "kfold")
    cv_strategy = _coerce_enum(ra.get("cv_strategy"), _CV_STRATEGIES, default_cv)
    try:
        cv_folds = int(ra.get("cv_folds") or 5)
    except (TypeError, ValueError):
        cv_folds = 5
    cv_folds = min(max(cv_folds, 3), 10)
    algo_prov = _filter_provenance(ra.get("provenance"), valid_ids)
    cited.update(algo_prov)
    candidates = [c for c in (_coerce_enum(x, _ALGO_FAMILIES, "") for x in (ra.get("alternatives") or [])) if c]
    algorithm = {
        "family": family,
        "candidates": candidates,
        "rationale": ra.get("rationale", ""),
        "cv": {"strategy": cv_strategy, "folds": cv_folds},
        "metric": metric,
        "provenance": algo_prov,
    }

    # ── Seuils d'alerte (modalités green/orange/red) : on filtre leur provenance
    # sur le POOL PERTINENT (mêmes valid_ids que le reste du spec) pour garantir
    # que chaque modalité est bien sourcée par un article réellement retenu, et on
    # réécrit la provenance nettoyée pour que l'UI puisse lier les articles. ──
    at = variables.get("alert_thresholds")
    if isinstance(at, dict):
        for _lvl in ("green", "orange", "red"):
            band = at.get(_lvl)
            if isinstance(band, dict):
                band_prov = _filter_provenance(band.get("provenance"), valid_ids)
                band["provenance"] = band_prov
                cited.update(band_prov)

    # ── Paramètres épidémiologiques (famille SEIR) ──────────────────────────────
    # Nettoyés (nombres coercés, provenance filtrée sur le pool pertinent) par le
    # module PUR seir_model (même logique testée hors-ligne). Bloc VIDE si le scénario
    # n'est pas une maladie transmissible ou si rien n'est rapporté → le modèle
    # compartimental (Phase 3) ne s'active tout simplement pas.
    import seir_model as _seir
    _epi = _seir.normalize_extracted_parameters(
        variables.get("epidemic_parameters"), valid_ids, quality_by_id
    )
    cited.update(_epi["cited"])

    # ── Le SEIR comme SOUS-MODÈLE : reclasser les variables qu'il DÉRIVE ─────────
    # Scénario épidémique (paramètres SEIR extraits) → une variable qui EST un paramètre
    # du modèle (R0, CFR, incubation…) n'est PAS une feature du prédicteur : on la retire
    # (elle vit dans epidemic_parameters). Une variable qui est une SORTIE du modèle
    # (incidence, prévalence, cumul, décès) RESTE une feature mais sa source devient
    # "seir" : remplie automatiquement par le connecteur SEIR (aucun upload / API externe).
    if _epi.get("applicable"):
        _pv_by_mn = {pv.get("machine_name"): pv
                     for pv in (variables.get("predictor_variables") or []) if isinstance(pv, dict)}
        _kept = []
        for f in features:
            _txt = f"{f.get('name', '')} {f.get('machine_name', '')}"
            _pv = _pv_by_mn.get(f["machine_name"])
            if _seir.is_seir_parameter(_txt):
                if _pv is not None:
                    _pv["_seir_role"] = "parameter"  # cross-link additif pour l'UI
                continue  # input de simulation → jamais une feature
            _col = _seir.seir_feature_column(_txt)
            if _col:
                f["source"], f["public_provider"], f["seir_column"] = "seir", None, _col
                if _pv is not None:
                    _pv["source"], _pv["_seir_role"], _pv["_seir_column"] = "seir", "derived", _col
            _kept.append(f)
        features = _kept

    # ── Data template (dérivé → garanti cohérent avec les machine_name ci-dessus) ──
    data_template = _derive_data_template(outcome, features)

    variables["model_spec"] = {
        "schema": MODEL_SPEC_SCHEMA,
        "version": 1,
        "outcome": outcome,
        "features": features,
        "algorithm": algorithm,
        "data_template": data_template,
        "epidemic_parameters": {
            "applicable": _epi["applicable"],
            "disease": _epi["disease"],
            "params": _epi["params"],
        },
    }
    variables["_provenance_index"] = {str(i): prov_meta[i] for i in sorted(cited) if i in prov_meta}
    return variables


def _loads_lenient(raw: str, report: dict | None = None) -> dict:
    """`json.loads` TOLÉRANT aux sorties LLM tronquées (max_tokens atteint) : retire un
    éventuel fence ```json, puis, si le JSON est incomplet, ferme les chaînes / tableaux /
    objets restés ouverts en rognant la fin jusqu'à obtenir un objet valide. Best-effort :
    renvoie le meilleur dict récupérable (au pire {}) - les champs manquants sont
    reconstruits en aval (_attach_model_spec dérive les champs machine des champs humains).
    Évite qu'une réponse coupée d'un caractère fasse échouer TOUTE la génération.

    `report`, s'il est donné, reçoit `repaired`, `kept_chars` et `total_chars` quand la fin
    a été rognée : le spec le dit dans son `_meta` au lieu de le laisser au journal."""
    import json as _j
    s = (raw or "").strip()
    if s.startswith("```"):                                   # fence markdown éventuel
        s = s.strip("`")
        s = s[4:] if s[:4].lower() == "json" else s
        s = s.strip()
    try:
        return _j.loads(s)
    except Exception:
        pass
    # Réparation : on tente de fermer les structures ouvertes à des points de coupe
    # proches de la fin (là où la troncature s'est produite). On borne les essais.
    cut_points = [i for i, ch in enumerate(s) if ch in ',}]"']
    for cut in reversed(cut_points[-3000:]):
        frag = s[:cut + 1].rstrip().rstrip(",")
        if frag.count('"') % 2 == 1:                          # chaîne ouverte → la fermer
            frag = frag[:-1] if frag.endswith('"') else frag + '"'
        opens = frag.count("{") - frag.count("}")
        obrk = frag.count("[") - frag.count("]")
        if opens < 0 or obrk < 0:
            continue
        cand = frag + ("]" * obrk) + ("}" * opens)
        try:
            out = _j.loads(cand)
            if isinstance(out, dict) and out:
                logger.warning("Variables/JSON LLM tronqué → réparé (best-effort, %d/%d caractères).",
                               cut + 1, len(s))
                if report is not None:
                    report.update(repaired=True, kept_chars=cut + 1, total_chars=len(s))
                return out
        except Exception:
            continue
    return {}


#: La règle du compte de preuves d'une variable, recopiée dans le `_meta` du spec.
EVIDENCE_COUNT_RULE = (
    "evidence_count = the number of relevant articles of the scenario (all of them, counted "
    "in SQL, never the articles shown to the model) whose title, abstract or PICO (P, I, C, "
    "O, key finding) mentions the variable's name or one of the keywords the spec gives it, "
    "case-insensitive, as whole words. evidence_level = evidence_count / relevant articles.")
_MAX_EVIDENCE_TERMS = 12


def _evidence_terms(pv: dict) -> list[str]:
    """Le nom d'une variable et ses mots-clés / synonymes tels que le spec les donne,
    dédupliqués sans tenir compte de la casse. PUR."""
    raw: list = [pv.get("name")]
    for key in ("keywords", "synonyms"):
        val = pv.get(key)
        raw += val if isinstance(val, list) else ([val] if isinstance(val, str) else [])
    out: list[str] = []
    seen: set[str] = set()
    for term in raw:
        t = " ".join(str(term).split()) if isinstance(term, str) else ""
        if len(t) >= 3 and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:_MAX_EVIDENCE_TERMS]


def variable_evidence_counts(scenario_id: str, threshold: float,
                             variables: dict) -> dict[str, Any]:
    """Pour chaque variable prédictive, combien d'articles PERTINENTS la mentionnent :
    compté en SQL sur TOUT le sous-ensemble pertinent, selon `EVIDENCE_COUNT_RULE`.

    « evidence_level : Nombre d'études qui la mentionnent » était demandé au modèle, qui
    comptait parmi les 25 articles qu'il voyait sur un corpus de plusieurs milliers. Le
    compte est écrit sur chaque variable (`evidence_count`, `evidence_level` « n/N » et les
    termes cherchés, `evidence_terms`) ; renvoie {denominator, counted}."""
    pvs = [pv for pv in (variables.get("predictor_variables") or []) if isinstance(pv, dict)]
    terms = [_evidence_terms(pv) for pv in pvs]
    rxs = {f"rx{i}": r"\y(" + "|".join(re.escape(t) for t in ts) + r")\y"
           for i, ts in enumerate(terms) if ts}
    cols = "".join(f", COUNT(*) FILTER (WHERE t.txt ~* :{k}) AS {k}" for k in rxs)
    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT COUNT(*) AS n_relevant{cols}
            FROM (
                SELECT concat_ws(' ', d.title, d.abstract,
                                 d.pico_json->>'P', d.pico_json->>'I', d.pico_json->>'C',
                                 d.pico_json->>'O', d.pico_json->>'population',
                                 d.pico_json->>'intervention', d.pico_json->>'outcome',
                                 d.pico_json->>'key_finding') AS txt
                FROM literature_document d
                JOIN article_scenarios ars ON ars.document_id = d.id
                WHERE ars.scenario_id = :sid AND {relevant_gate_sql('d', 'ars', ':thr')}
            ) t
        """), {"sid": scenario_id, "thr": threshold, **rxs}).mappings().first()
    denominator = int(row["n_relevant"] or 0) if row else 0
    for i, (pv, ts) in enumerate(zip(pvs, terms)):
        n = int(row[f"rx{i}"] or 0) if (row and ts) else None
        pv["evidence_count"] = n
        pv["evidence_level"] = f"{n}/{denominator}" if n is not None else None
        pv["evidence_terms"] = ts
    return {"denominator": denominator, "counted": sum(1 for ts in terms if ts)}


def _generate_variables_from_pico(scenario_id: str, persist: str = "active", lang: str | None = None) -> dict[str, Any]:
    """
    Génère automatiquement les variables du modèle et l'outcome à partir des articles
    pertinents (PICO structuré + titre/résumé + extrait de texte intégral quand
    disponible), et non des seuls PICO. Sauvegarde dans scenario_settings.variables_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from llm_usage import MeteredOpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    # Résumés/PICO/texte intégral pour les 25 articles du contexte seulement.
    articles = _get_above_threshold_articles(scenario_id, threshold, include_fulltext=True,
                                             fulltext_query=_get_scenario_name(scenario_id),
                                             fulltext_top_docs=25, fulltext_char_cap=2200,
                                             full_rows=25)

    # On n'EXIGE plus un PICO extrait : tout article pertinent contribue au choix des
    # variables/outcome/algorithme via son titre+abstract (+texte intégral si dispo).
    # Le PICO, quand présent, ajoute la structure P/I/C/O. Auparavant, un article
    # pertinent sans PICO était purement ignoré (perte de signal injustifiée).
    # `pico_articles` = désormais l'ensemble pertinent complet (nom conservé pour le
    # reste du flux ; le PICO reste optionnel par article).
    pico_articles = articles
    if not pico_articles:
        return {"error": _msg(lang, "Aucun article au-dessus du seuil pour générer les variables.",
                              "No article above the threshold to generate the variables.")}
    _n_with_pico = sum(1 for a in pico_articles if a.get("has_pico") or a.get("pico_json"))

    scenario_name = _get_scenario_name(scenario_id)

    # Contexte à trois niveaux (PICO structuré → abstract → extrait de TEXTE
    # INTÉGRAL quand disponible) : le LLM choisit les prédicteurs/métriques sur le
    # texte réel des articles, pas seulement sur la compression PICO.
    pico_context = []
    for a in pico_articles[:25]:
        pj = a.get("pico_json") or {}
        _entry = {
            "id": a.get("id"),
            "title": a.get("title", "")[:100],
            "year": a.get("year"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            # Rempli désormais par chaque rédacteur PICO (api/pico_map) ; il était lu ici et
            # toujours vide.
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
            "finding_direction": pj.get("finding_direction", ""),
            "abstract": (a.get("abstract") or "")[:1200],
        }
        _ft = (a.get("fulltext") or "").strip()
        if _ft:
            _entry["fulltext_excerpt"] = _ft
        pico_context.append(_entry)

    context_str = _json.dumps(pico_context, ensure_ascii=False, indent=2)

    # Le spec engage le corpus ENTIER : le digest (agrégats sur TOUS les articles
    # pertinents, sans échantillonnage) précède les 25 articles reproduits, qui servent
    # à choisir des variables concrètes et à citer.
    from .digest import corpus_digest, digest_coverage_note, digest_to_prompt
    _digest = corpus_digest(scenario_id, threshold)
    _digest_block = digest_to_prompt(_digest)
    _coverage = digest_coverage_note(_digest, len(pico_context))

    system_prompt = """Tu es un expert en modélisation prédictive appliquée à la santé.
A partir d'une revue systématique de la littérature, tu identifies les variables clés,
l'outcome principal, et le meilleur algorithme pour un modèle prédictif.
Tu génères un JSON structuré. Ne pas utiliser de tiret cadratin (em dash).""" + _llm_lang_directive(lang)

    user_prompt = f"""Scénario : "{scenario_name}"

{_digest_block or f"Basé sur {len(pico_articles)} articles pertinents ({_n_with_pico} avec PICO extrait)."}

{_coverage}

Articles reproduits ({len(pico_context)} les plus proches de la question du scénario, dans l'ordre de pertinence du corpus ; PICO, abstract et extrait de texte intégral quand disponibles) :

{context_str}

Génère un JSON avec EXACTEMENT ces champs :
{{
  "primary_outcome": {{
    "name": "Nom de l'outcome principal",
    "definition": "Définition clinique précise",
    "measurement": "Comment le mesurer",
    "timeframe": "Horizon temporel",
    "machine_name": "identifiant_snake_case_court",
    "task_type": "classification|regression|count|survival",
    "unit": "Unité de mesure (ex: bool, jours, /100k)",
    "positive_class": "Classe positive si classification, sinon null",
    "provenance": [ids d'articles de la liste ci-dessus soutenant cet outcome]
  }},
  "secondary_outcomes": [
    {{"name": "...", "definition": "..."}}
  ],
  "predictor_variables": [
    {{
      "name": "Nom de la variable",
      "type": "continuous|binary|categorical|time_series",
      "definition": "Définition clinique",
      "data_source": "Source de données recommandée",
      "importance": "high|medium|low",
      "keywords": ["3 à 8 mots-clés ou synonymes, en anglais ET en français, tels qu'un article les écrirait (ex: temperature, température, heat)"],
      "machine_name": "identifiant_snake_case_court (ex: temp_max_j1)",
      "dtype": "float|int|bool|category|datetime",
      "source": "user (fournie par l'utilisateur) | public_api (récupérable: météo, open data...) | seir (DÉRIVÉE du sous-modèle épidémique: incidence/prévalence/cumul/décès)",
      "public_provider": "open-meteo si météo, sinon null",
      "provenance": [ids d'articles de la liste ci-dessus mentionnant cette variable]
    }}
  ],
  "recommended_algorithm": {{
    "primary": "Algorithme principal recommandé",
    "alternatives": ["Alternative 1", "Alternative 2"],
    "rationale": "Justification basée sur la littérature",
    "validation_method": "Méthode de validation recommandée",
    "family": "gradient_boosting|random_forest|logistic_regression|linear_regression|elasticnet|svm|mlp|cox_ph|knn",
    "metric": "roc_auc|average_precision|rmse|mae|c_index",
    "cv_strategy": "stratified_kfold|kfold|timeseries",
    "cv_folds": 5,
    "provenance": [ids d'articles de la liste ci-dessus justifiant l'algorithme]
  }},
  "required_databases": ["Base 1", "Base 2"],
  "sample_size_recommendation": "Estimation de la taille d'échantillon nécessaire",
  "update_frequency": "Fréquence de mise à jour recommandée",
  "alert_thresholds": {{
    "green":  {{"label": "Normal",  "range": "Plage de valeurs de l'OUTCOME considérée normale, NUMÉRIQUE et cohérente avec son unit/task_type (ex: '< 0.10' pour une probabilité, '< 5 /100k' pour un taux, '0-2' pour un compte)", "rationale": "Justification fondée sur les évidences (cite les seuils rapportés dans la littérature si disponibles)", "provenance": [ids d'articles]}},
    "orange": {{"label": "Tension", "range": "Plage intermédiaire (ex: '0.10-0.30')", "rationale": "...", "provenance": [ids d'articles]}},
    "red":    {{"label": "Alerte",  "range": "Plage critique (ex: '> 0.30')", "rationale": "...", "provenance": [ids d'articles]}}
  }},
  "epidemic_parameters": {{
    "applicable": true si le scénario porte sur une MALADIE TRANSMISSIBLE (épidémie / infection) modélisable par un compartimental SEIR, false sinon,
    "population_disease": "maladie / agent pathogène concerné si applicable, sinon null",
    "r0": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "ratio", "n_studies": entier, "provenance": [ids d'articles rapportant cette valeur], "observations": [{{"article_id": id, "value": nombre}}]}},
    "infectious_period_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "incubation_period_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "cfr": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "proportion", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "immunity_duration_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "serial_interval_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}}
  }},
  "implementation_notes": "Notes d'implémentation pratiques",
  "validation_status": "pending"
}}

IMPORTANT pour les champs "provenance" : ce sont des listes d'identifiants (champ "id")
des articles figurant dans la liste ci-dessus. N'invente AUCUN id : n'utilise que des id
réellement présents. Les "machine_name" doivent être de courts identifiants snake_case
(minuscules, chiffres, underscores) utilisables comme noms de colonnes.

IMPORTANT pour "keywords" : le serveur compte, sur TOUS les articles pertinents, ceux dont
le titre, le résumé ou le PICO contient le nom de la variable ou l'un de ces mots-clés. Donne
les termes qu'un article emploierait (pas des phrases), en anglais et en français ; ne
compte pas toi-même les études.

IMPORTANT pour "alert_thresholds" : donne des plages de valeurs NUMÉRIQUES concrètes de
l'OUTCOME (pas des phrases vagues), cohérentes avec son "unit" et son "task_type", et
fondées sur les seuils rapportés dans les évidences quand ils existent (sinon des seuils
cliniquement plausibles). Les trois plages doivent être contiguës et couvrir tout le
domaine. Tu PEUX renommer les catégories si l'outcome s'y prête (ex. pour un compte :
"Faible"/"Modéré"/"Élevé"), mais garde 3 niveaux du plus sûr au plus critique.

IMPORTANT - le SEIR est un SOUS-MODÈLE, pas des features (scénarios de maladie
transmissible) : ne mets JAMAIS un PARAMÈTRE du modèle compartimental (R0, Rt, létalité/
CFR, période d'incubation, période infectieuse, durée d'immunité, intervalle sériel) dans
"predictor_variables" - ces valeurs vont dans "epidemic_parameters", pas dans les features.
En revanche, une SORTIE du sous-modèle (incidence, prévalence, cas actifs, infections
cumulées, décès) PEUT être une variable prédictive utile : liste-la normalement mais mets
son "source": "seir" - le serveur la remplira automatiquement depuis le sous-modèle
(aucune donnée externe requise). Les autres variables (météo, vaccination, mobilité, lits…)
gardent "source": "user" ou "public_api".

IMPORTANT pour "epidemic_parameters" : ne renseigne ces paramètres qu'à partir de
valeurs RÉELLEMENT rapportées dans les articles ci-dessus (avec leur provenance) ; mets
"value": null pour tout paramètre non rapporté - n'invente AUCUN chiffre. Donne une
estimation centrale + un intervalle (ci_low/ci_high, idéalement l'IC 95 %) reflétant la
dispersion entre études quand plusieurs la rapportent. Pour agréger, PRIVILÉGIE les
estimations des synthèses de meilleure qualité (revues systématiques / méta-analyses,
cf. "study_design") sur les études isolées. "cfr" en PROPORTION (0..1, pas en
pourcentage). Pour CHAQUE paramètre, renseigne aussi "observations" : la liste des
valeurs par étude RÉELLEMENT rapportées ({{"article_id": id, "value": nombre}}), une
entrée par article rapportant ce paramètre (mets [] si aucune). Ces observations
individuelles servent à un regroupement pondéré par la qualité côté serveur. Si le
scénario ne porte PAS sur une maladie transmissible, mets "applicable": false et tous
les "value" à null.

Retourne UNIQUEMENT le JSON valide."""

    try:
        # ── Déterminisme du recheck ──────────────────────────────────────────
        # Empreinte de l'évidence = ensemble des articles (au-dessus du seuil) qui
        # alimentent le modèle. Si elle est INCHANGÉE depuis le dernier spec validé,
        # on RÉUTILISE ce spec tel quel : « rechecker » sur la même évidence donne
        # exactement le même résultat (plus d'appel LLM non déterministe). On ne
        # régénère via le LLM QUE lorsque l'évidence change réellement.
        # Empreinte partagée avec l'Evidence Brief (_evidence_fingerprint) : ensemble
        # ORDONNÉ des IDs pertinents + seuil + langue + version de contexte. Le suffixe
        # de version invalide UNE FOIS les specs générés avec un contexte plus pauvre
        # (PICO seul, ou sans texte intégral) : régénérés puis réutilisés à évidence
        # constante. Seuil et langue inclus : un spec FR au seuil 0.45 ne doit pas être
        # resservi pour une requête EN ni pour un autre seuil.
        # v6 : les paramètres épidémiologiques viennent désormais d'une extraction CIBLÉE
        # sur les articles qui les mesurent, pas des seuls 25 plus pertinents. Le suffixe
        # invalide UNE FOIS les specs construits sans elle.
        # v8 : le digest ne présente plus un score jamais calculé ni des citations
        # inconnues comme des zéros, et les articles enrichis par le lot de métadonnées
        # portent enfin un score : les paramètres poolés avec les anciens poids (un article
        # non noté pesait 1.0, plus que tout article noté) sont recalculés une fois.
        # v9 : l'empreinte porte TOUS les articles pertinents, comme celle du brief. Elle
        # ne prenait que les 25 premiers : un article ajouté ou retiré plus bas réutilisait
        # l'ancien spec et sautait l'extraction des paramètres épidémiologiques, qui lit,
        # elle, tout le corpus. Et le contexte change : résultats comptés dans le digest,
        # compte de preuves par variable calculé en SQL.
        _CTX_VERSION = "ctx-v9-whole-corpus-fingerprint"
        evidence_fingerprint = _evidence_fingerprint(
            [a.get("id") for a in pico_articles if a.get("id") is not None],
            threshold, lang, _CTX_VERSION)[:16]

        _reused = None
        with engine.connect() as _rc:
            _row = _rc.execute(text(
                "SELECT variables_json, variables_proposal_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        # On réutilise le spec validé (variables_json) en priorité, sinon la dernière
        # proposition (variables_proposal_json), si construit sur la MÊME évidence.
        for _slot in ("variables_json", "variables_proposal_json"):
            _cand = _row.get(_slot) if _row else None
            if (isinstance(_cand, dict)
                    and _cand.get("model_spec")
                    and _cand.get("_meta", {}).get("evidence_fingerprint") == evidence_fingerprint):
                _reused = _cand
                break

        _carried: dict[str, Any] = {}
        _parse: dict[str, Any] = {}
        _cut = False
        if _reused is not None:
            variables = dict(_reused)
            # Ce que la génération d'origine a constaté reste vrai du spec réutilisé.
            _carried = {k: v for k, v in (variables.get("_meta") or {}).items()
                        if k.startswith(("epidemic_parameter_", "truncated", "evidence_count_"))}
            logger.info(f"Variables {scenario_id}: évidence inchangée (fingerprint {evidence_fingerprint}) "
                        f"→ spec réutilisé (déterministe, sans appel LLM).")
        else:
            # 90 s pour une réponse de 8 000 jetons, qui en demande 40 à 80 : au bord, et un
            # dépassement relançait la requête entière dans le SDK. La génération tourne en
            # arrière-plan, la latence ne bloque personne.
            client = _OAI(timeout=300.0)

            def _ask(messages):
                return client.chat.completions.create(
                    model=_model("write"),
                    messages=messages,
                    temperature=0,
                    seed=42,
                    max_tokens=8000,  # 3000 tronquait le spec (observations par étude + SEIR) → JSON invalide
                    response_format={"type": "json_object"},
                )

            # Coupé au plafond : redemandé une fois, plus court. Coupé encore, la réponse est
            # réparée au mieux (TOLÉRANT : sa fin est rognée) et le spec le DIT dans son
            # `_meta`, au lieu de le laisser au seul journal.
            try:
                _raw = json_completion(_ask, [{"role": "system", "content": system_prompt},
                                              {"role": "user", "content": user_prompt}],
                                       f"variables {scenario_id}")
            except CutAnswer as _c:
                _raw, _cut = _c.content, True
            variables = _loads_lenient(_raw, report=_parse)
            if not variables:
                raise ValueError("Réponse LLM (variables) vide ou illisible même après réparation.")

            # Paramètres épidémiologiques : extraction CIBLÉE sur les articles du corpus
            # qui les MESURENT (les 25 plus pertinents n'en rapportent presque jamais).
            # Sans candidat, aucun appel LLM : un scénario non transmissible ne coûte rien.
            _epi_targeted: dict[str, Any] = {}
            try:
                _epi_targeted = extract_epidemic_observations(
                    scenario_id, disease_hint=scenario_name, threshold=threshold)
                if _epi_targeted.get("params"):
                    variables["epidemic_parameters"] = merge_epidemic_observations(
                        variables.get("epidemic_parameters"), _epi_targeted)
            except Exception as _epi_err:                    # jamais bloquant
                logger.warning(f"Extraction ciblée des paramètres {scenario_id}: {_epi_err}")

            # Phase 1 : normaliser en model_spec déterministe (machine_name, dtype,
            # algorithme/CV/métrique, data_template) + provenance tracée vers les articles.
            # Pool de provenance = les 25 articles du contexte UNION les articles qui ont
            # fourni une mesure (sinon leur provenance serait filtrée et le pooling vide).
            _prov_pool = list(pico_articles[:25])
            _prov_seen = {a.get("id") for a in _prov_pool}
            for _a in (_epi_targeted.get("articles") or []):
                if _a.get("id") not in _prov_seen:
                    _prov_pool.append(_a)
                    _prov_seen.add(_a.get("id"))
            try:
                variables = _attach_model_spec(variables, _prov_pool)
            except Exception as spec_err:  # le spec machine ne doit jamais bloquer la génération
                logger.error(f"model_spec build {scenario_id}: {spec_err}", exc_info=True)

            # Le compte de preuves de chaque variable, sur TOUT le corpus pertinent, en SQL :
            # il était demandé au modèle, qui comptait parmi ses 25 articles.
            try:
                _counts = variable_evidence_counts(scenario_id, threshold, variables)
                _carried["evidence_count_rule"] = EVIDENCE_COUNT_RULE
                _carried["evidence_count_denominator"] = _counts["denominator"]
            except Exception as _ev_err:                     # jamais bloquant, jamais le compte du modèle
                logger.warning(f"Comptes de preuves des variables {scenario_id}: {_ev_err}")
                for _pv in (variables.get("predictor_variables") or []):
                    if isinstance(_pv, dict):
                        _pv["evidence_count"], _pv["evidence_level"] = None, None
                _carried["evidence_count_error"] = str(_ev_err)[:300]
            _carried.update(_epi_meta(_epi_targeted))
            _carried["truncated"] = bool(_cut or _parse.get("repaired"))
            if _parse.get("repaired"):
                _carried["truncated_repair"] = {"kept_chars": _parse["kept_chars"],
                                                "total_chars": _parse["total_chars"]}

        try:
            with engine.connect() as _cc:
                _corpus_total = _cc.execute(text("""
                    SELECT COUNT(*) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
                """), {"sid": scenario_id}).scalar() or 0
        except Exception:
            _corpus_total = len(articles)
        variables["_meta"] = {
            "scenario_id": scenario_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # Outcome/variables/algorithme dérivés du sous-ensemble PERTINENT.
            "corpus_total": int(_corpus_total),            # corpus complet
            "relevant_total": len(articles),               # au-dessus du seuil
            "pico_articles_used": len(pico_articles),      # articles pertinents (entrée du modèle ; PICO optionnel par article)
            "auto_generated": True,
            "validation_status": "pending",
            "evidence_fingerprint": evidence_fingerprint,
            "reused": _reused is not None,
            # Combien d'articles du corpus MESURENT un paramètre, combien en ont donné une
            # valeur, et ce que l'extraction n'a pas pu lire : l'onglet Modèle peut dire « 0
            # sur 2 732 » au lieu de « aucun paramètre extrait », qui laissait croire à une
            # panne. Puis la règle du compte de preuves des variables, et si la réponse a dû
            # être réparée (`truncated`).
            **_carried,
        }

        # Sauvegarder en DB. persist="proposal" (Phase 5) écrit dans un slot
        # de staging sans toucher le spec actif validé.
        with engine.begin() as conn:
            if persist == "proposal":
                conn.execute(text("""
                    INSERT INTO scenario_settings (scenario_id, variables_proposal_json, proposal_generated_at, updated_at)
                    VALUES (:sid, CAST(:vars AS jsonb), NOW(), NOW())
                    ON CONFLICT (scenario_id) DO UPDATE
                    SET variables_proposal_json = CAST(:vars AS jsonb),
                        proposal_generated_at = NOW(),
                        updated_at = NOW()
                """), {"sid": scenario_id, "vars": _json.dumps(variables)})
            else:
                # variables_lang = langue de génération (pour l'affichage localisé) ;
                # variables_i18n remis à NULL car les traductions en cache sont périmées.
                conn.execute(text("""
                    INSERT INTO scenario_settings (scenario_id, variables_json, variables_validated, variables_generated_at, variables_lang, variables_i18n, updated_at)
                    VALUES (:sid, CAST(:vars AS jsonb), FALSE, NOW(), :vlang, NULL, NOW())
                    ON CONFLICT (scenario_id) DO UPDATE
                    SET variables_json = CAST(:vars AS jsonb),
                        variables_validated = FALSE,
                        variables_generated_at = NOW(),
                        variables_lang = :vlang,
                        variables_i18n = NULL,
                        updated_at = NOW()
                """), {"sid": scenario_id, "vars": _json.dumps(variables), "vlang": (_norm_lang(lang) or "fr")})

        logger.info(f"Variables & Modèle générés ({persist}) pour {scenario_id}: {len(pico_articles)} articles PICO.")
        return variables

    except Exception as e:
        logger.error(f"Variables generation {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


def _detect_variables_lang(variables: dict) -> str:
    """Heuristique FR/EN pour les lignes LEGACY dépourvues de variables_lang."""
    po = (variables or {}).get("primary_outcome") or {}
    sample = " ".join(str(po.get(k, "")) for k in ("name", "definition", "measurement")).lower()
    for pv in (variables.get("predictor_variables") or [])[:3]:
        sample += " " + str((pv or {}).get("definition", "")).lower()
    if not sample.strip():
        return "fr"
    fr = sum(sample.count(m) for m in (" le ", " la ", " les ", " des ", " une ", " du ", " et ",
                                       " pour ", " être", " taux ", "é", "è", "à", "ç"))
    en = sum(sample.count(m) for m in (" the ", " of ", " and ", " for ", " rate ", " number ",
                                       " is ", " to ", " in ", " with "))
    return "en" if en > fr else "fr"


def _translate_batch(client, texts: list, lang_name: str, may_split: bool = True) -> list:
    """Un appel de traduction. Coupé au plafond, le lot est scindé en deux moitiés, chacune
    demandée une fois : une traduction coupée n'est pas une traduction plus courte. LÈVE
    sur toute autre panne et sur une réponse de mauvaise forme."""
    import json as _json
    sys = (f"You are a professional medical translator. Translate each string in the input "
           f"array to {lang_name}. Preserve medical/technical terms, units (e.g. /100k, mg, %), "
           "acronyms (ARI, RSV, SARIMAX, ED), and all numbers. Do NOT add explanations or notes. "
           "Do not use the em dash character. "
           'Return ONLY a JSON object {"items": [...]} whose "items" is a JSON array of the '
           "translated strings, in the SAME order and with EXACTLY the same length as the input.")
    usr = _json.dumps({"items": list(texts)}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=_model("write"),
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0, seed=42, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        out = _json.loads(json_content(resp, f"translate_strings ({len(texts)} strings)"))
    except LLMTruncated:
        if not may_split or len(texts) < 2:
            raise
        half = (len(texts) + 1) // 2
        return (_translate_batch(client, texts[:half], lang_name, may_split=False)
                + _translate_batch(client, texts[half:], lang_name, may_split=False))
    items = out.get("items") if isinstance(out, dict) else None
    if isinstance(items, list) and len(items) == len(texts) and all(isinstance(x, str) for x in items):
        return items
    raise ValueError(f"shape mismatch ({type(items).__name__}, "
                     f"{len(items) if isinstance(items, list) else 'n/a'} vs {len(texts)})")


def _llm_translate_strings(texts: list, target_lang: str) -> list | None:
    """Traduit une liste de chaînes vers 'en'/'fr' (ordre + longueur conservés).

    Renvoie None en cas d'échec : il renvoyait les chaînes d'ORIGINE, que l'appelant
    mettait en cache COMME la traduction, si bien qu'une panne passagère figeait l'anglais
    dans l'affichage français jusqu'à la prochaine régénération du spec."""
    from llm_usage import MeteredOpenAI as _OAI
    if not texts:
        return texts
    lang_name = "English" if target_lang == "en" else "French"
    try:
        return _translate_batch(_OAI(timeout=90.0), list(texts), lang_name)
    except Exception as e:
        logger.warning(f"translate_strings failed: {e}")
        return None


def _translate_variables_payload(variables: dict, target_lang: str) -> dict:
    """Copie TRADUITE de variables_json : seuls les champs d'AFFICHAGE sont traduits
    (noms, définitions, libellés, justifications, listes de bases/notes). machine_name,
    dtype, task_type, family, metric, cv, provenance, ranges numériques et structure du
    model_spec restent identiques. Les noms/justification traduits sont propagés dans le
    model_spec miroir (lu par /model/spec)."""
    return _translate_variables(variables, target_lang)[0]


def _translate_variables(variables: dict, target_lang: str) -> tuple[dict, bool]:
    """(copie traduite, traduite ou non). Une traduction en échec rend la copie NON
    traduite et False : servie telle quelle, elle ne doit pas être mise en cache."""
    import copy
    v = copy.deepcopy(variables or {})
    texts: list = []
    setters: list = []

    def _add(container, key):
        val = container.get(key)
        if isinstance(val, str) and val.strip():
            texts.append(val)
            setters.append((len(texts) - 1, lambda new, c=container, k=key: c.__setitem__(k, new)))

    def _add_list(lst):
        if isinstance(lst, list):
            for i, item in enumerate(lst):
                if isinstance(item, str) and item.strip():
                    texts.append(item)
                    setters.append((len(texts) - 1, lambda new, l=lst, idx=i: l.__setitem__(idx, new)))

    po = v.get("primary_outcome") or {}
    for k in ("name", "definition", "measurement", "timeframe"):
        _add(po, k)
    for so in (v.get("secondary_outcomes") or []):
        if isinstance(so, dict):
            for k in ("name", "definition"):
                _add(so, k)
    for pv in (v.get("predictor_variables") or []):
        if isinstance(pv, dict):
            for k in ("name", "definition", "data_source"):
                _add(pv, k)
    ra = v.get("recommended_algorithm") or {}
    for k in ("primary", "rationale", "validation_method"):
        _add(ra, k)
    _add_list(ra.get("alternatives"))
    at = v.get("alert_thresholds") or {}
    for lvl in ("green", "orange", "red"):
        band = at.get(lvl)
        if isinstance(band, dict):
            for k in ("label", "rationale", "description"):
                _add(band, k)
    _add_list(v.get("required_databases"))
    for k in ("sample_size_recommendation", "update_frequency", "implementation_notes"):
        _add(v, k)

    if not texts:
        return v, True
    translated = _llm_translate_strings(texts, target_lang)
    if not isinstance(translated, list) or len(translated) != len(texts):
        return v, False   # sécurité : ne jamais corrompre, et ne rien mettre en cache
    for idx, setter in setters:
        setter(translated[idx])

    # Propager les noms/justification traduits dans le model_spec miroir.
    ms = v.get("model_spec")
    if isinstance(ms, dict):
        if isinstance(ms.get("outcome"), dict) and po.get("name"):
            ms["outcome"]["name"] = po["name"]
        name_by_mn = {pv.get("machine_name"): pv.get("name")
                      for pv in (v.get("predictor_variables") or []) if isinstance(pv, dict)}
        for f in (ms.get("features") or []):
            if isinstance(f, dict) and name_by_mn.get(f.get("machine_name")):
                f["name"] = name_by_mn[f["machine_name"]]
        if isinstance(ms.get("algorithm"), dict) and ra.get("rationale"):
            ms["algorithm"]["rationale"] = ra["rationale"]
    return v, True


def _get_localized_variables(scenario_id: str, lang, base_variables: dict | None = None,
                             variables_lang=None, variables_i18n=None) -> dict | None:
    """variables_json rendu dans la langue demandée (traduction non destructive mise en
    cache par langue dans variables_i18n). Charge depuis la DB si base_variables absent."""
    import json as _json
    req = _norm_lang(lang)
    if base_variables is None:
        with engine.connect() as conn:
            r = conn.execute(text(
                "SELECT variables_json, variables_lang, variables_i18n "
                "FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        if not (r and r["variables_json"]):
            return None
        base_variables = dict(r["variables_json"])
        variables_lang, variables_i18n = r.get("variables_lang"), r.get("variables_i18n")
    if not req:
        return base_variables
    base = _norm_lang(variables_lang) or _detect_variables_lang(base_variables)
    if req == base:
        return base_variables
    cache = variables_i18n if isinstance(variables_i18n, dict) else {}
    if isinstance(cache.get(req), dict):
        return cache[req]
    translated, ok = _translate_variables(base_variables, req)
    if not ok:
        # Un échec n'est pas une traduction : servi tel quel, NON mis en cache, pour que le
        # prochain affichage réessaie au lieu de resservir l'original pour toujours.
        return translated
    try:  # cache best-effort ; l'affichage ne doit pas échouer si l'écriture échoue
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE scenario_settings SET variables_i18n = "
                "jsonb_set(COALESCE(variables_i18n, '{}'::jsonb), :path, CAST(:val AS jsonb), true) "
                "WHERE scenario_id = :sid"
            ), {"sid": scenario_id, "path": "{" + req + "}", "val": _json.dumps(translated)})
    except Exception as e:
        logger.warning(f"cache variables_i18n {scenario_id}/{req}: {e}")
    return translated


@app.post("/scenarios/{scenario_id}/variables/generate")
def generate_scenario_variables(scenario_id: str, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Déclenche la génération asynchrone des Variables & Modèle depuis les articles pertinents."""
    import threading, time

    # Une revue de littérature ne fabrique pas de variables prédictives : on le dit
    # dans la forme que l'API parle déjà, plutôt que par une erreur.
    from .scenario_store import CAP_MODEL, capability_refusal, scenario_can
    if not scenario_can(scenario_id, CAP_MODEL):
        return {"status": "not_applicable", **capability_refusal(scenario_id, CAP_MODEL)}

    if _job_is_active(_VARIABLES_GENERATION_JOBS.get(scenario_id)):
        return {"status": "already_running"}

    _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_variables_from_pico(scenario_id, lang=lang)
            if "error" in result:
                _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _VARIABLES_GENERATION_JOBS[scenario_id] = {
                    "status": "done",
                    "generated_at": result.get("_meta", {}).get("generated_at"),
                    "variables_count": len(result.get("predictor_variables", [])),
                }
        except Exception as e:
            logger.error(f"Variables job {scenario_id}: {e}", exc_info=True)
            _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/variables/generate/status")
def get_variables_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération des variables."""
    return _VARIABLES_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/epidemic-parameters/candidates")
def get_epidemic_parameter_candidates(scenario_id: str, limit: int = 0) -> dict[str, Any]:
    """Les articles du corpus pertinent qui RAPPORTENT un paramètre épidémiologique, avec
    le ou les paramètres que chacun mesure. Lecture seule, sans LLM : dit ce que la
    littérature du scénario contient avant toute extraction.

    `n_candidates` compte TOUT le corpus, comme le fait l'extraction : cet endroit
    répondait « 40 » pour n'importe quel corpus au-delà de 40, donc un chiffre différent
    de celui annoncé par l'extraction pour le même scénario. `limit > 0` ne tronque que la
    LISTE renvoyée (`articles`), jamais les compteurs."""
    arts = _parameter_candidate_articles(scenario_id, None, 0)      # jamais d'échantillon
    by_param: dict[str, int] = {}
    for a in arts:
        for p in a["params_mentioned"]:
            by_param[p] = by_param.get(p, 0) + 1
    _shown = arts if int(limit or 0) <= 0 else arts[:int(limit)]
    return {
        "scenario_id": scenario_id,
        "n_candidates": len(arts),                 # tout le corpus
        "by_parameter": by_param,                  # compté sur tout le corpus
        "articles_listed": len(_shown),
        "articles": [{"id": a["id"], "title": (a.get("title") or "")[:200], "year": a.get("year"),
                      "doi": a.get("doi"), "study_design": a.get("study_design"),
                      "quality_score": a.get("quality_score"),
                      "parameters": a["params_mentioned"]} for a in _shown],
    }


@app.post("/scenarios/{scenario_id}/epidemic-parameters/extract")
def extract_scenario_epidemic_parameters(scenario_id: str, max_articles: int = 0,
                                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Relance la seule extraction CIBLÉE des paramètres épidémiologiques et la fusionne
    dans le spec déjà stocké (le reste du spec est inchangé et n'est PAS régénéré).

    Utile quand le corpus a grandi, ou pour un scénario dont le spec a été construit
    avant cette extraction. Le pooling pondéré par la qualité est refait ensuite."""
    import json as _json

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    # Une revue de littérature n'aura JAMAIS de spécification : l'étape qui en écrit
    # une lui est retirée. Exiger une spécification ici revenait à lui demander de
    # faire d'abord une chose qu'elle ne peut pas faire, pour obtenir une mise en
    # commun qui est pourtant un produit de revue à part entière.
    variables = dict(row["variables_json"]) if (row and row["variables_json"]) else {}
    name = _get_scenario_name(scenario_id)
    targeted = extract_epidemic_observations(scenario_id, disease_hint=name, max_articles=max_articles)
    if not targeted.get("params"):
        # ── Dire ce qui s'est passé, pas ce que dirait la littérature ────────
        # « Aucune valeur exploitable dans les articles qui mentionnent un paramètre »
        # est une affirmation définitive sur les articles. Elle était servie telle quelle
        # quand TOUS les lots avaient échoué (quota, délai, clé invalide, réponse
        # malformée) : le modèle n'avait jamais examiné un seul des 205 articles.
        _st = str(targeted.get("status") or "ok")
        _failed = int(targeted.get("n_batches_failed") or 0)
        _nb = int(targeted.get("n_batches") or 0)
        _unexamined = int(targeted.get("n_not_examined") or 0)
        if _st == "no_key":
            _msg = ("Aucune clé de modèle n'est configurée : l'extraction n'a pas eu "
                    "lieu. Ce n'est pas un constat sur les articles.")
        elif _st == "llm_unavailable":
            _msg = (f"L'extraction n'a pas pu être faite : les {_nb} lots ont tous "
                    f"échoué et {_unexamined} articles n'ont pas été examinés. "
                    "Ce n'est pas un constat sur les articles.")
        elif _st == "llm_partial":
            _msg = (f"Extraction partielle : {_failed} lots sur {_nb} ont échoué et "
                    f"{_unexamined} articles n'ont pas été examinés. Aucune valeur "
                    "exploitable dans ceux qui l'ont été.")
        elif targeted.get("n_candidates"):
            _msg = "Aucune valeur exploitable dans les articles qui mentionnent un paramètre."
        else:
            _msg = "Aucun article du corpus ne rapporte de paramètre épidémiologique."
        return {"status": ("llm_unavailable" if _st in ("llm_unavailable", "no_key")
                           else "partial" if _st == "llm_partial" else "no_parameters"),
                "scenario_id": scenario_id,
                "n_candidates": targeted.get("n_candidates", 0),
                "n_articles_examined": targeted.get("n_articles_used", 0),
                "n_batches": _nb, "n_batches_failed": _failed,
                "n_not_examined": _unexamined,
                "message": _msg}

    variables["epidemic_parameters"] = merge_epidemic_observations(
        variables.get("epidemic_parameters"), targeted)
    # Pool de provenance : les articles du contexte UNION ceux qui ont fourni une mesure.
    thr = _get_scenario_threshold(scenario_id)
    pool = _get_above_threshold_articles(scenario_id, thr, full_rows=25)[:25]
    seen = {a.get("id") for a in pool}
    for a in (targeted.get("articles") or []):
        if a.get("id") not in seen:
            pool.append(a)
            seen.add(a.get("id"))
    from .scenario_store import CAP_MODEL, scenario_can
    if scenario_can(scenario_id, CAP_MODEL):
        variables = _attach_model_spec(variables, pool)
    else:
        # Le regroupement pondéré par la qualité vit dans un module PUR : on l'appelle
        # seul. Une revue obtient ses paramètres mis en commun, avec leurs intervalles
        # et la provenance de chaque étude, sans qu'on lui fabrique un outcome, des
        # variables explicatives et un algorithme dont elle n'a que faire.
        import seir_model as _seir
        _ids = {a["id"] for a in pool if a.get("id") is not None}
        _q = {a["id"]: a.get("quality_score") for a in pool if a.get("id") is not None}
        _pooled = _seir.normalize_extracted_parameters(
            variables.get("epidemic_parameters"), _ids, _q)
        variables["epidemic_parameters_pooled"] = {
            k: v for k, v in _pooled.items() if k != "cited"}
    meta = dict(variables.get("_meta") or {})
    meta.update(_epi_meta(targeted))
    variables["_meta"] = meta

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, variables_json, updated_at)
            VALUES (:sid, CAST(:v AS jsonb), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
            SET variables_json = CAST(:v AS jsonb), variables_i18n = NULL,
                seir_projection_json = NULL, seir_projection_generated_at = NULL,
                updated_at = NOW()
        """), {"v": _json.dumps(variables, default=str), "sid": scenario_id})

    _epi = ((variables.get("model_spec") or {}).get("epidemic_parameters")
            or variables.get("epidemic_parameters_pooled") or {})
    return {
        "status": "ok",
        "scenario_id": scenario_id,
        "n_candidates": targeted.get("n_candidates", 0),
        "n_articles_with_values": targeted.get("n_with_values", 0),
        # Ce qui n'a PAS pu être lu, même quand le reste a donné des valeurs : la réponse
        # de succès taisait les lots perdus.
        "n_articles_examined": targeted.get("n_articles_used", 0),
        "n_from_fulltext": targeted.get("n_from_fulltext", 0),
        "n_batches": targeted.get("n_batches", 0),
        "n_batches_failed": targeted.get("n_batches_failed", 0),
        "n_batches_split": targeted.get("n_batches_split", 0),
        "n_not_examined": targeted.get("n_not_examined", 0),
        "failed_article_ids": targeted.get("failed_article_ids") or [],
        "disease": _epi.get("disease") or targeted.get("disease"),
        "applicable": bool(_epi.get("applicable")),
        "parameters": {k: {"value": v.get("value"), "ci_low": v.get("ci_low"), "ci_high": v.get("ci_high"),
                           "n_studies": v.get("n_studies"), "unit": v.get("unit")}
                       for k, v in (_epi.get("params") or {}).items()},
    }


@app.get("/scenarios/{scenario_id}/variables")
def get_scenario_variables(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne les variables & modèle générés.
    Si absent, déclenche la génération.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_validated, variables_generated_at,
                   variables_lang, variables_i18n
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["variables_json"]:
        # Affichage localisé (traduction non destructive mise en cache par langue) :
        # le contenu fonctionnel (machine_name, model_spec) reste celui de variables_json.
        result = dict(_get_localized_variables(
            scenario_id, lang, base_variables=dict(row["variables_json"]),
            variables_lang=row.get("variables_lang"), variables_i18n=row.get("variables_i18n")))
        result["_validated"] = row["variables_validated"]
        result["_generated_at"] = row["variables_generated_at"].isoformat() if row["variables_generated_at"] else None
        return result

    # Vérifier qu'il y a des articles avant de déclencher la génération
    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold, full_rows=0)   # existence seulement
    if not articles:
        return {"status": "empty", "message": _msg(lang,
                "Aucun article au-dessus du seuil. Ajoutez des articles ou abaissez le seuil de similarité.",
                "No article above the threshold. Add articles or lower the similarity threshold.")}

    # Job en échec : renvoyer l'erreur au lieu de relancer à chaque poll (évite
    # une boucle de regénération quand la génération échoue de façon persistante).
    _vj = _VARIABLES_GENERATION_JOBS.get(scenario_id, {})
    if _vj.get("status") == "error":
        return {"status": "error", "message": _vj.get("error", "Échec de la génération des variables.")}

    # Déclencher la génération. On passe lang EXPLICITEMENT : appeler l'endpoint
    # directement sans cet argument passerait l'objet Query(None) par défaut (et non
    # None) jusqu'à _llm_lang_directive → crash « 'Query' object has no attribute 'strip' ».
    started = generate_scenario_variables(scenario_id, lang=lang)
    # Refusée parce que la question n'appelle pas de modèle : on le dit, au lieu
    # d'annoncer une génération qui n'a pas commencé et ne commencera pas.
    if isinstance(started, dict) and started.get("status") == "not_applicable":
        return started
    return {"status": "generating", "message": _msg(lang, "Génération en cours, réessayez dans 30 secondes.",
                                                    "Generating, try again in 30 seconds.")}


@app.post("/scenarios/{scenario_id}/variables/validate")
def validate_scenario_variables(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Valide (ou modifie) les variables & modèle générés par LLM.
    payload peut contenir les variables modifiées.
    """
    import json as _json
    from datetime import datetime, timezone

    variables_json = payload.get("variables_json")
    with engine.begin() as conn:
        if variables_json:
            # La projection SEIR en cache est calculée à partir de ce spec : un spec
            # remplacé par l'utilisateur la rend caduque. Sans cette invalidation, l'onglet
            # Modèle resservait, en la marquant « from_cache », une courbe issue des
            # paramètres que l'utilisateur venait justement de corriger.
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_json = CAST(:vars AS jsonb),
                    variables_validated = TRUE,
                    variables_lang = NULL,
                    variables_i18n = NULL,
                    seir_projection_json = NULL,
                    seir_projection_generated_at = NULL,
                    updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id, "vars": _json.dumps(variables_json)})
        else:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_validated = TRUE, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})

    return {"status": "validated", "scenario_id": scenario_id, "validated_at": datetime.now(timezone.utc).isoformat()}


def _norm_col(s: Any) -> str:
    """Normalise un nom de colonne pour l'appariement fichier↔data_template. MÊMES
    règles que model_trainer._norm (repli NFKD des accents, minuscules, non-alnum →
    « _ ») pour que la VALIDATION et l'ENTRAÎNEMENT s'accordent : un en-tête réel
    « Température max (J1) » s'apparie au machine_name « temperature_max_j1 »."""
    import unicodedata as _ud
    _s = _ud.normalize("NFKD", str(s if s is not None else "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", _s.lower()).strip("_")
