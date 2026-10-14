"""Live literature sources: fetchers, parsers, federated search, direct ingestion.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
import re
import threading as _threading_ncbi
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import (EXHAUSTIVE_FEDERATION_BUDGET, POPULATE_FEDERATION_BUDGET, RELIEFWEB_APPNAME, app,
                   engine, logger, redact_keys, require_api_key)
from .documents import _normalize_doi, _normalize_title, same_work_doi_match, sanitize_db_text
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             has_abstract_sql, relevant_gate_tail_sql)
from .search import (EXHAUSTIVE_MAX_PER_SOURCE, LIVE_MAX_PER_SOURCE, RANKED_PASS_MAX,
                     _plain_keywords, _strip_field_tags)
from llm_usage import model_for as _model


class SourceFetchError(RuntimeError):
    """Une source live a échoué.

    Les cinq fetchers de la recherche en direct avalaient leur propre exception et
    renvoyaient une liste vide. L'appelant, lui, a une branche `error` et une branche
    `empty` : la branche `error` ne pouvait donc jamais s'allumer, et une source
    injoignable s'affichait « aucun résultat », ce qui est une affirmation sur la
    littérature et non sur le réseau."""


class PartialSourceFetch(RuntimeError):
    """Une source a renvoyé une PARTIE de sa réponse, puis a échoué.

    PubMed en est le cas typique : l'esearch réussit et donne le vrai total, l'esummary
    échoue. Ni « ok » ni « error » : ce qui est revenu est utilisable, le compte ne l'est
    pas."""

    def __init__(self, items: list, total: int | None, reason: str):
        super().__init__(reason)
        self.items = items or []
        self.total = total
        self.reason = reason

_NCBI_LOCK = _threading_ncbi.Lock()
_NCBI_LAST = [0.0]
_NCBI_MIN_INTERVAL = 0.4  # ~2.5 req/s, sous la limite de 3/s

#: Semantic Scholar avec une clé : UNE requête par seconde, tous points d'accès et tous appelants
#: confondus. Les recherches et le téléchargement des textes partagent la clé : sans espacement
#: commun, trois tâches simultanées se la disputaient, et la recherche de l'exposition du
#: 13 octobre 2026 s'est arrêtée sur cinq 429 d'affilée après 299 des 10 257 notices.
_S2_LOCK = _threading_ncbi.Lock()
_S2_LAST = [0.0]
S2_MIN_INTERVAL_S = float(os.getenv("S2_MIN_INTERVAL_S", "1.1") or 1.1)


def s2_key() -> str:
    """La clé de Semantic Scholar du serveur, ou une chaîne vide."""
    return (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()


def s2_headers_paced() -> dict:
    """Les en-têtes d'un appel à Semantic Scholar, rendus quand l'appel peut partir : avec la
    clé du serveur, un départ toutes les `S2_MIN_INTERVAL_S` secondes pour tout le processus
    (l'API tourne en un seul processus), comme `_ncbi_get` pour NCBI. Sans clé, rien n'est
    espacé ici : le quota anonyme est partagé avec tous les clients du monde, et ses 429
    restent attendus par les appelants."""
    import time as _time
    key = s2_key()
    if not key:
        return {}
    with _S2_LOCK:
        wait = S2_MIN_INTERVAL_S - (_time.time() - _S2_LAST[0])
        if wait > 0:
            _time.sleep(wait)
        _S2_LAST[0] = _time.time()
    return {"x-api-key": key}


#: L'URL de base de la recherche Europe PMC, écrite une fois : quatre appels la répétaient.
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

#: Au-delà de cette longueur de requête ENCODÉE, Europe PMC est interrogée en POST.
#: Mesuré sur trois scénarios de production, la même stratégie envoyée en GET :
#:
#:     EXP_HPAI           4 604 octets encodés   200
#:     EXP_HPAI_criteria  5 598 octets encodés   503
#:     HPAI_team          6 435 octets encodés   503
#:
#: Ce n'est donc pas une panne passagère : toute stratégie de revue un peu longue dépasse
#: ce que leur serveur accepte dans une URL, et aucun nouvel essai ne peut aboutir, puisque
#: c'est la même URL. L'API accepte les mêmes paramètres dans le CORPS d'un POST, où la
#: longueur ne joue plus. Même remède que pour eutils (`EUTILS_POST_ABOVE`). La borne est
#: prise bien en dessous de la première taille observée en échec.
EPMC_POST_ABOVE_ENCODED = 3000


def epmc_send(requests_mod, params: dict, timeout: float = 20.0, headers: dict | None = None):
    """Un appel à la recherche Europe PMC : GET pour une requête courte, POST pour une
    longue (cf. EPMC_POST_ABOVE_ENCODED).

    Si leur serveur refusait le POST (405/501), l'appel est retenté en GET : on ne perd
    pas la source sur une méthode non supportée, on retombe sur le comportement d'avant.
    Rend la réponse telle quelle ; l'appelant lève pour un code d'erreur, comme avant."""
    import urllib.parse as _up
    encoded = len(_up.quote_plus(str(params.get("query") or "")))
    if encoded <= EPMC_POST_ABOVE_ENCODED:
        return requests_mod.get(EPMC_SEARCH_URL, params=params, timeout=timeout, headers=headers)
    resp = requests_mod.post(EPMC_SEARCH_URL, data=params, timeout=timeout,
                             headers={**(headers or {}), "Content-Type": "application/x-www-form-urlencoded"})
    if getattr(resp, "status_code", 0) in (405, 501):
        logger.warning("Europe PMC refuse le POST (%s) : nouvel essai en GET.", resp.status_code)
        return requests_mod.get(EPMC_SEARCH_URL, params=params, timeout=timeout, headers=headers)
    return resp


def epmc_query(boolean: str) -> str:
    """La requête telle qu'Europe PMC la comprend : SANS les tags de champ de PubMed.

    Europe PMC ne parle pas `[mh]` ni `[tiab]` : elle les apparie comme du TEXTE, que
    presque rien ne contient. Mesuré sur la sonde de production, pour la même question :

        avian influenza AND occupational exposure            2 398 notices
        "Influenza in Birds"[mh] AND "Occupational Exposure"[mh]      3 notices
        "avian influenza"[tiab] AND "occupational*"[tiab]             6 notices

    Les tags coûtaient donc 99,9 % du rappel de la deuxième source biomédicale de la
    fédération, et c'est l'explication du `europepmc: 0` au tableau d'identification du
    scénario HPAI, dont la requête porte des dizaines de ces tags.

    Une TRADUCTION vers la syntaxe d'Europe PMC (`MESH:`, `TITLE:`, `ABSTRACT:`) serait
    plus précise que ce simple retrait, qui élargit la recherche à tous les champs. Elle
    demande d'éprouver chaque nom de champ contre l'API, ce que l'environnement de
    développement ne peut pas faire : à écrire le jour où on peut la vérifier, plutôt
    qu'à deviner. Le retrait, lui, est sûr et va dans le sens du rappel.

    Une entrée faite UNIQUEMENT de tags rend la chaîne vide, et non l'entrée : un repli
    « si le retrait ne laisse rien, garde l'original » rendait précisément le tag que
    cette fonction existe pour enlever."""
    return " ".join(_strip_field_tags(boolean or "").split())


#: Au-delà de cette longueur de paramètres encodés, eutils est interrogé en POST. NCBI
#: documente le POST pour les requêtes longues, et une URL de GET au-delà de ~2 000
#: caractères est refusée ou tronquée par les proxys. La requête du scénario HPAI fait
#: 3 075 caractères, soit une URL de GET de 4 247 : son esearch partait donc dans une URL
#: que rien ne garantit. `efetch` postait déjà sa longue liste d'identifiants ; c'est
#: `esearch`, qui porte la REQUÊTE, qui ne le faisait pas.
_NCBI_POST_ABOVE = 1800


def _ncbi_get(url: str, params: dict, timeout: int = 12, use_key: bool = True):
    """Requête NCBI throttlée (verrou global), avec un petit retry sur 429/erreur.

    En GET par défaut, en POST dès que les paramètres encodés dépassent
    `_NCBI_POST_ABOVE` : eutils accepte les mêmes paramètres dans le corps, et c'est la
    forme que NCBI prescrit pour une requête longue.

    Le verrou espace le DÉPART des requêtes, ce que NCBI compte (3 par seconde sans clé,
    10 avec) ; il n'est plus tenu pendant la requête elle-même, comme l'efetch du populate
    le faisait déjà. Tenu, il sérialisait tout le trafic NCBI derrière la requête la plus
    lente : un texte intégral PMC de trente secondes bloquait l'esearch d'une recherche.
    `use_key=False` : un service NCBI hors eutils (le convertisseur d'identifiants PMC)
    reçoit les mêmes paramètres qu'avant, sans la clé d'eutils, mais le même espacement."""
    import requests as _req
    import time as _time
    import urllib.parse as _up
    key = os.getenv("NCBI_API_KEY")
    if key and use_key:
        params = {**params, "api_key": key}
    # Avec une clé API, NCBI autorise 10 req/s (vs 3 sans) : on resserre l'espacement
    # pour réduire la sérialisation du verrou global sur le trio PubMed/PROSPERO/Cochrane.
    min_interval = 0.11 if (key and use_key) else _NCBI_MIN_INTERVAL
    _long = len(_up.urlencode({k: v for k, v in params.items() if v is not None})) > _NCBI_POST_ABOVE
    r = None
    last_exc: Exception | None = None
    for attempt in range(3):
        with _NCBI_LOCK:
            wait = min_interval - (_time.time() - _NCBI_LAST[0])
            if wait > 0:
                _time.sleep(wait)
            _NCBI_LAST[0] = _time.time()
        try:
            r = (_req.post(url, data=params, timeout=timeout) if _long
                 else _req.get(url, params=params, timeout=timeout))
        except Exception as _e:  # timeout / ConnectionError : transitoire → retry (cf. docstring)
            last_exc = _e
            r = None
        if r is None:
            _time.sleep(0.6 * (attempt + 1))
            continue
        if r.status_code == 429:
            _time.sleep(0.6 * (attempt + 1))
            continue
        return r
    if r is None:
        raise last_exc if last_exc else RuntimeError("NCBI request a échoué (3 tentatives)")
    return r


# ── PubMed au-delà de 10 000 notices : la même requête, par tranches d'années ─────
#: eutils ne rend aucun identifiant au-delà du 10 000e d'une même recherche PubMed
#: (`retstart + retmax` ≤ 10 000 ; NCBI le documente et conseille de découper la requête).
#: Une requête qui en annonçait 23 000 n'en rapportait donc que les 10 000 premiers, et la
#: carte PRISMA les présentait comme la recherche entière, plafond compris.
NCBI_ESEARCH_CEILING = 10_000
#: Borne NOMINALE d'une tranche ouverte vers le passé : elle sert à calculer un milieu,
#: jamais à écrire la requête. La tranche la plus ancienne s'écrit « NOT (Y:3000[dp]) » et
#: prend tout ce qui précède Y, quelle que soit sa date : une borne basse écrite en dur
#: aurait laissé dehors les notices plus anciennes qu'elle, que PubMed indexe.
_PUBMED_SPLIT_FLOOR = 1800


def pubmed_year_bounds(filters) -> tuple[int | None, int | None]:
    """Les bornes d'années du scénario (`year_min`, `year_max`), en entiers, pour PubMed.

    Une borne illisible est ignorée, et des bornes inversées aussi : la correspondance
    locale (`_build_where`) applique alors sa propre règle, comme avant. PUR."""
    f = filters if isinstance(filters, dict) else {}

    def _year(v) -> int | None:
        if v is None or isinstance(v, bool):
            return None
        try:
            y = int(float(str(v).strip()))
        except (TypeError, ValueError):
            return None
        return y if 1000 <= y <= 9999 else None

    lo, hi = _year(f.get("year_min")), _year(f.get("year_max"))
    if lo is not None and hi is not None and lo > hi:
        return None, None
    return lo, hi


def pubmed_years_term(term: str, lo: int | None, hi: int | None) -> str:
    """`term` restreinte aux années de publication [lo, hi] ; None ouvre ce côté.

    « 3000 » est la borne « jusqu'à aujourd'hui » de PubMed. Une tranche ouverte vers le
    passé s'écrit par exclusion, « NOT (hi+1:3000[dp]) », pour ne dépendre d'aucune année
    plancher. [dp] lit la date imprimée ET la date électronique : une notice mise en ligne
    en décembre et imprimée en janvier répond aux deux années, d'où la déduplication des
    identifiants quand les tranches sont réunies. PUR."""
    if lo is None and hi is None:
        return term
    if hi is None:
        return f"({term}) AND ({lo}:3000[dp])"
    if lo is None:
        return f"({term}) NOT ({hi + 1}:3000[dp])"
    return f"({term}) AND ({lo}:{hi}[dp])"


def pubmed_years_label(lo: int | None, hi: int | None) -> str:
    """Le libellé d'une tranche pour la carte PRISMA : « 2021 », « 1995-2010 », « ≤1962 »,
    « ≥2023 ». PUR."""
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"≤{hi}"
    if hi is None:
        return f"≥{lo}"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _pubmed_year_halves(lo, hi, floor: int, top: int):
    """Les deux moitiés de la tranche [lo, hi], ou None quand elle ne se coupe plus (une
    seule année, ou une tranche ouverte au-delà de sa borne nominale). PUR."""
    a = floor if lo is None else lo
    b = top if hi is None else hi
    if a >= b:
        return None
    mid = (a + b) // 2
    return (lo, mid), (mid + 1, hi)


def pubmed_slice_quotas(counts: list[int], cap: int, total: int,
                        ceiling: int = NCBI_ESEARCH_CEILING) -> list[int]:
    """Combien d'identifiants demander à chaque tranche.

    Tout ce qu'elle peut rendre (au plus `ceiling`) quand la requête entière tient sous le
    plafond par source. Au-delà, le plafond est PARTAGÉ entre les tranches à proportion de
    leur taille, chacune gardant ses plus pertinentes : prendre les tranches dans l'ordre
    jusqu'au plafond aurait gardé une époque et perdu les autres, ce que le tri par date
    faisait déjà (la littérature H5N1 de 2004 à 2012 perdue). Le reste de l'arrondi va aux
    plus grandes parts fractionnaires, jamais au-delà de ce qu'une tranche peut rendre.
    Déterministe. PUR."""
    counts = [max(0, int(c or 0)) for c in counts]
    avail = [min(c, int(ceiling)) for c in counts]
    cap = max(0, int(cap))
    if int(total) <= cap or sum(avail) <= cap:
        return avail
    weight = sum(counts) or 1
    raw = [cap * c / weight for c in counts]
    quota = [min(a, int(r)) for a, r in zip(avail, raw)]
    left = cap - sum(quota)
    order = sorted(range(len(counts)), key=lambda i: (-(raw[i] - int(raw[i])), i))
    while left > 0:
        moved = False
        for i in order:
            if left <= 0:
                break
            if quota[i] < avail[i]:
                quota[i] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return quota


def _interleave_by_rank(lists: list[list[str]]) -> list[str]:
    """Les identifiants de plusieurs tranches, chacune dans son ordre de pertinence, réunis
    par rang RELATIF (le premier dixième de chaque tranche, puis le deuxième...), chacun une
    fois. Une coupure par le budget pendant le téléchargement des notices perd ainsi les
    moins pertinentes de chaque tranche, et non les tranches les plus récentes. PUR."""
    keyed = []
    for i, ids in enumerate(lists):
        n = len(ids) or 1
        keyed.extend((r / n, i, r, pmid) for r, pmid in enumerate(ids))
    keyed.sort(key=lambda k: (k[0], k[1], k[2]))
    out: list[str] = []
    seen: set[str] = set()
    for _f, _i, _r, pmid in keyed:
        if pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


class _SplitStopped(Exception):
    """Le budget de temps est écoulé pendant le découpage."""


def pubmed_esearch_by_years(esearch, render, *, total: int, cap: int,
                            year_min: int | None = None, year_max: int | None = None,
                            page: int | None = None, ceiling: int | None = None,
                            stop=None, this_year: int | None = None) -> dict:
    """Tous les PMID d'une requête PubMed qui en annonce plus que `ceiling` (10 000), par la
    MÊME requête restreinte à des tranches d'années de publication.

    Décision du propriétaire du produit. La tranche des années du scénario (toutes, sans
    bornes) est coupée en deux, récursivement, jusqu'à ce que chaque tranche en annonce au
    plus `ceiling` ; chaque tranche est lue, et les identifiants sont réunis, chacun une
    fois (une notice électronique en décembre et imprimée en janvier répond aux deux
    années). Une seule année encore au-delà de `ceiling` garde ses `ceiling` plus
    pertinentes, et le dit (`capped`). Quand la requête entière dépasse le plafond par
    source `cap`, une tranche dont la part du plafond tient en une recherche n'est plus
    coupée (cf. `pubmed_slice_quotas`).

    `esearch(term, retstart, retmax)` rend l'objet `esearchresult` d'eutils ou lève : c'est
    l'appelant qui porte ses reprises, son espacement, son tri et le passage en POST d'une
    requête longue (`_ncbi_get`). `render(lo, hi)` écrit la requête d'une tranche, avec ce
    que l'appelant y ajoute (l'exclusion des préprints). `total` est le compte de la
    requête ENTIÈRE, celui que la carte PRISMA garde : la somme des tranches compterait
    deux fois les notices à cheval sur deux années. `stop()` vrai arrête tout (budget).

    Rend {ids, total, slices: [{years, from, to, count, kept, capped}], complete, stopped,
    error}. Un découpage qui n'a pas pu être planifié (erreur, budget) ne rend aucun
    identifiant : l'appelant garde alors ce que la requête entière lui a déjà rendu."""
    ceiling = int(ceiling or NCBI_ESEARCH_CEILING)
    page = max(1, min(int(page or ceiling), ceiling))
    cap = max(0, int(cap))
    total = max(0, int(total or 0))
    if this_year is None:
        from datetime import datetime as _dt
        this_year = _dt.now().year
    floor, top = _PUBMED_SPLIT_FLOOR, int(this_year)
    out = {"ids": [], "total": total, "slices": [], "complete": False, "stopped": False,
           "error": None}

    def _result(term: str, retstart: int, retmax: int) -> dict:
        res = esearch(term, retstart, retmax) or {}
        if res.get("ERROR"):
            raise RuntimeError(f"esearch: {str(res['ERROR'])[:200]}")
        return res

    def _needs_split(c: int) -> bool:
        if c <= ceiling:
            return False
        # Au-delà du plafond par source, une tranche dont la part tient en une recherche
        # n'a pas à être coupée : on n'en lira que sa part.
        return not (total > cap and c * cap <= ceiling * total)

    leaves: list[tuple[tuple, int, bool]] = []

    def _visit(rng: tuple, c: int) -> None:
        if not _needs_split(c):
            leaves.append((rng, c, False))
            return
        halves = _pubmed_year_halves(rng[0], rng[1], floor, top)
        if halves is None:
            leaves.append((rng, c, True))           # une seule année, encore trop grande
            return
        for half in halves:
            if stop is not None and stop():
                raise _SplitStopped()
            _visit(half, int(_result(render(*half), 0, 0).get("count") or 0))

    try:
        _visit((year_min, year_max), total)
    except _SplitStopped:
        out["stopped"] = True
        return out
    except Exception as e:                                   # noqa: BLE001 - dit, pas avalé
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out

    quotas = pubmed_slice_quotas([c for _r, c, _k in leaves], cap, total, ceiling)
    lists: list[list[str]] = []
    for (rng, c, capped), want in zip(leaves, quotas):
        got: list[str] = []
        while len(got) < want and not out["stopped"]:
            if stop is not None and stop():
                out["stopped"] = True
                break
            try:
                res = _result(render(*rng), len(got), min(page, want - len(got)))
            except Exception as e:                           # noqa: BLE001 - dit, pas avalé
                out["error"] = f"{type(e).__name__}: {e}"[:300]
                break
            ids = [str(p).strip() for p in (res.get("idlist") or []) if str(p).strip()]
            if not ids:
                break
            got.extend(ids)
        got = got[:want]
        lists.append(got)
        out["slices"].append({"years": pubmed_years_label(*rng), "from": rng[0], "to": rng[1],
                              "count": int(c), "kept": len(got), "capped": bool(capped)})
        if out["stopped"]:
            break
    out["ids"] = _interleave_by_rank(lists)
    out["complete"] = not out["stopped"] and out["error"] is None
    return out


def _live_fetch_pubmed(query: str, max_results: int) -> tuple[list[dict], int]:
    """Fetch from PubMed eSearch+eSummary.

    Renvoie (résultats, hitcount_réel). `hitcount_réel` = esearchresult.count = le
    NOMBRE VRAI d'enregistrements PubMed correspondant à la requête - le même
    compteur que sur pubmed.ncbi.nlm.nih.gov - qui peut dépasser de loin les
    `max_results` réellement rapatriés dans le panneau. Surfacé pour que le badge
    de source affiche le vrai total trouvé, et non les seules lignes récupérées
    (cause du « 306 sur PubMed mais 35 ici »)."""
    results: list[dict] = []
    total = 0
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = _ncbi_get(f"{base}/esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": max_results,
            # Le même ordre que le populate : le panneau montrait les plus récents pendant
            # que le corpus gardait les plus pertinents, deux listes pour une requête.
            "sort": "relevance",
            "retmode": "json", "tool": "literev", "email": "api@literev.app"
        })
        _esr = r.json().get("esearchresult", {})
        total = int(_esr.get("count", 0) or 0)          # vrai hitcount (comme le site)
        ids = _esr.get("idlist", [])
        if not ids:
            return [], total
        r2 = _ncbi_get(f"{base}/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            "tool": "literev", "email": "api@literev.app"
        })
        res2 = r2.json().get("result", {})
        _uids = list(res2.get("uids", []))
        for uid in _uids:
            item = res2.get(uid, {})
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": next((a["value"] for a in item.get("articleids", []) if a.get("idtype") == "doi"), None),
                "year": int(item.get("pubdate", "")[:4]) if item.get("pubdate", "")[:4].isdigit() else None,
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "journal": item.get("source", None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "external_id": f"pmid:{uid}",
                "source_name": "PubMed",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_pubmed error: {_e}")
        # Un échec APRÈS l'esearch laisse un vrai total et aucune notice : c'est un
        # fetch partiel, à dire comme tel. Avant l'esearch, il n'y a rien du tout.
        raise PartialSourceFetch(results, total, str(_e)[:200]) from _e
    # L'esummary peut RÉUSSIR et rendre moins de notices qu'on lui a demandé d'identifiants
    # (sous charge, eutils renvoie `{"result": {"uids": []}}`). Rien ne lève alors, et une
    # recherche dont l'esearch avait trouvé 306 enregistrements s'affichait « aucun
    # résultat » : une affirmation sur la littérature pour un aller-retour manqué. C'est le
    # même fetch partiel que l'exception ci-dessus, et il se dit de la même façon.
    if len(results) < len(ids):
        raise PartialSourceFetch(
            results, total,
            f"esummary a rendu {len(results)} notice(s) pour {len(ids)} identifiant(s)")
    return results, total


#: L'adresse donnée à OpenAlex avec chaque appel, celle que la recherche donnait déjà.
OPENALEX_MAILTO = "literev@gesica.ch"


def openalex_params(params: dict | None = None) -> dict:
    """Les paramètres d'un appel à OpenAlex : `mailto`, et la clé quand le serveur en a une.

    OpenAlex facture désormais son API : un appel sans clé puise dans un petit budget
    quotidien gratuit, partagé par tous les clients derrière la même adresse IP, et une fois
    ce budget dépensé chaque appel répond 429 « Insufficient budget » jusqu'à minuit UTC.
    C'est ce qui a arrêté la recherche HPAI exposition du 12 octobre 2026 après 1 594
    notices. Une clé gratuite (openalex.org/settings/api, enregistrée sous OPENALEX_API_KEY)
    donne au serveur un budget à lui, plus large ; une page de recherche coûte, une notice
    lue par son identifiant non. La clé voyage dans l'URL : toute erreur qui peut la citer
    passe par `redact_keys` avant d'être journalisée ou enregistrée."""
    out = {"mailto": OPENALEX_MAILTO, **dict(params or {})}
    key = (os.getenv("OPENALEX_API_KEY") or "").strip()
    if key:
        out["api_key"] = key
    return out


def _live_fetch_openalex(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # OpenAlex `search` n'accepte pas la syntaxe booléenne (renvoie HTTP 400)
        # → on lui passe des mots-clés simples.
        r = _req.get("https://api.openalex.org/works", params=openalex_params({
            "search": _plain_keywords(query), "per-page": min(max_results, 50),
            "select": "id,title,abstract_inverted_index,doi,publication_year,authorships,primary_location,open_access"
        }), headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        r.raise_for_status()      # un 429 ou un 500 n'est pas « aucun résultat »
        for item in r.json().get("results", []):
            doi = item.get("doi", "")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            loc = item.get("primary_location") or {}
            source = loc.get("source") or {}
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": doi or None,
                "year": item.get("publication_year"),
                "authors": [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:5]],
                "journal": source.get("display_name"),
                "url": item.get("id"),
                "external_id": item.get("id"),
                "source_name": "OpenAlex",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_openalex error: {_e}")
        raise SourceFetchError(str(_e)[:200]) from _e
    return results


def _live_fetch_crossref(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        r = _req.get("https://api.crossref.org/works", params={
            "query": query, "rows": min(max_results, 50),
            "select": "DOI,title,abstract,published,author,container-title"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        r.raise_for_status()      # un 429 ou un 500 n'est pas « aucun résultat »
        for item in r.json().get("message", {}).get("items", []):
            pub = item.get("published", {}).get("date-parts", [[None]])[0]
            year = pub[0] if pub else None
            results.append({
                "title": (item.get("title") or [""])[0],
                "abstract": item.get("abstract"),
                "doi": item.get("DOI"),
                "year": year,
                "authors": [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in item.get("author", [])[:5]],
                "journal": (item.get("container-title") or [None])[0],
                "url": f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                "external_id": item.get("DOI"),
                "source_name": "Crossref",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_crossref error: {_e}")
        raise SourceFetchError(str(_e)[:200]) from _e
    return results


def _live_fetch_europepmc(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # NB : ne PAS passer sort=RELEVANCE - c'est une valeur invalide pour
        # EuropePMC qui renvoie alors une liste vide. Sans 'sort', l'API trie
        # par pertinence par défaut.
        r = epmc_send(_req, {
            # Sans les tags de champ de PubMed, qu'Europe PMC apparie comme du texte.
            "query": epmc_query(query), "resultType": "lite", "pageSize": min(max_results, 50),
            "format": "json"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        r.raise_for_status()      # un 429 ou un 500 n'est pas « aucun résultat »
        for item in r.json().get("resultList", {}).get("result", []):
            results.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstractText"),
                "doi": item.get("doi"),
                "year": int(item["pubYear"]) if item.get("pubYear", "").isdigit() else None,
                "authors": item.get("authorString", "").split(", ")[:5] if item.get("authorString") else [],
                "journal": item.get("journalTitle"),
                "url": f"https://europepmc.org/article/{item.get('source','')}/{item.get('id','')}",
                "external_id": item.get("id"),
                "source_name": "EuropePMC",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_europepmc error: {_e}")
        raise SourceFetchError(str(_e)[:200]) from _e
    return results


def _live_fetch_preprints(query: str, max_results: int) -> list[dict]:
    """Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (filtre SRC:PPR).
    Europe PMC indexe les préprints AVEC recherche plein-texte par mots-clés - au
    contraire de l'API biorxiv (dates/DOI uniquement) qu'utilisaient les anciens
    scanners medRxiv/bioRxiv (peu/pas de résultats)."""
    import requests as _req
    results = []
    try:
        r = epmc_send(_req, {
            "query": f"({epmc_query(query)}) AND (SRC:PPR)", "resultType": "lite",
            "pageSize": min(max_results, 50), "format": "json",
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        r.raise_for_status()      # un 429 ou un 500 n'est pas « aucun résultat »
        for item in r.json().get("resultList", {}).get("result", []):
            results.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstractText"),
                "doi": item.get("doi"),
                "year": int(item["pubYear"]) if item.get("pubYear", "").isdigit() else None,
                "authors": item.get("authorString", "").split(", ")[:5] if item.get("authorString") else [],
                "journal": item.get("journalTitle") or "Preprint",
                "url": f"https://europepmc.org/article/{item.get('source','PPR')}/{item.get('id','')}",
                "external_id": item.get("id"),
                "source_name": "Preprints",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_preprints error: {_e}")
        raise SourceFetchError(str(_e)[:200]) from _e
    return results


def _live_fetch_pubmed_term(term: str, source_name: str, id_prefix: str, max_results: int) -> list[dict]:
    """Helper PubMed générique (esearch+esummary) avec un terme/filtre arbitraire.
    Sert de proxy pour les sources sans API libre (PROSPERO, Cochrane)."""
    results = []
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = _ncbi_get(f"{base}/esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": max_results,
            "retmode": "json", "tool": "literev", "email": "api@literev.app"
        })
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        r2 = _ncbi_get(f"{base}/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            "tool": "literev", "email": "api@literev.app"
        })
        res = r2.json().get("result", {})
        for uid in res.get("uids", []):
            item = res.get(uid, {})
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": next((a["value"] for a in item.get("articleids", []) if a.get("idtype") == "doi"), None),
                "year": int(item.get("pubdate", "")[:4]) if item.get("pubdate", "")[:4].isdigit() else None,
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "journal": item.get("source", None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "external_id": f"{id_prefix}:{uid}",
                "source_name": source_name,
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_pubmed_term({source_name}) error: {_e}")
    return results


def _live_dedup_key(result: dict) -> str | None:
    """La clé de dédup du panneau de recherche en direct : le DOI, sinon le titre normalisé
    quand il fait au moins 20 caractères, comme la dédup de l'ingestion. None : la notice
    reste seule. Le titre exact, sans longueur minimale, fondait en une deux « Editorial »
    ou deux « Correction » sans DOI. PUR/testable."""
    doi = (result.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    title = re.sub(r"[^a-z0-9]+", " ", (result.get("title") or "").lower()).strip()
    return ("ttl:" + title) if len(title) >= 20 else None


def _federated_live_search(
    query: str,
    max_per_source: int = 50,
    pubmed_query: str | None = None,
    general_query: str | None = None,
) -> tuple[list[dict], list[str], dict[str, int], dict[str, dict]]:
    """Interroge les sources externes en parallèle, déduplique (par DOI puis
    titre normalisé), marque in_local_db, et score chaque résultat
    (sémantique cosinus + lexical + hybride). Réutilisé par /search (fédéré)
    et par la recherche live des scénarios.

    Retourne (results triés par hybrid_score desc, sources_queried,
    raw_counts par source avant déduplication)."""
    import concurrent.futures
    pubmed_query = pubmed_query or query
    general_query = general_query or query
    # PubMed RECALL : la requête MeSH générée (pubmed_query) est souvent BEAUCOUP plus
    # étroite que le booléen portable (general_query) - p.ex. 35 vs 306 pour le même
    # booléen collé sur le site PubMed. On interroge PubMed avec l'UNION des deux (même
    # correctif que le populate) pour retrouver le rappel du site, quelle que soit sa
    # longueur : `_ncbi_get` l'envoie en POST au-delà de la limite d'URL. Repli : booléen seul.
    if pubmed_query and general_query and pubmed_query != general_query:
        _pubmed_q = f"({pubmed_query}) OR ({general_query})"
    else:
        _pubmed_q = general_query or pubmed_query

    source_fns = [
        ("PubMed", _live_fetch_pubmed, _pubmed_q),
        ("OpenAlex", _live_fetch_openalex, general_query),
        ("Crossref", _live_fetch_crossref, general_query),
        ("EuropePMC", _live_fetch_europepmc, general_query),
        # Préprints (bioRxiv/medRxiv/…) via Europe PMC (SRC:PPR) : recherche par
        # mots-clés réelle. Remplace les scanners medRxiv/bioRxiv (dates seules) et
        # les proxys PubMed PROSPERO/Cochrane (trompeurs et redondants avec PubMed).
        ("Preprints", _live_fetch_preprints, general_query),
    ]

    import time as _t_fed
    _t0_fed = _t_fed.time()
    all_results: list[dict] = []
    sources_queried: list[str] = []
    raw_counts: dict[str, int] = {name: 0 for name, _, _ in source_fns}
    # Statut par source pour le diagnostic ("not working / slow") : ok / empty /
    # error / timeout, + latence. Les sources non complétées dans le délai
    # restent "timeout" (auparavant silencieusement absentes de la réponse).
    source_status: dict[str, dict[str, Any]] = {
        name: {"status": "timeout", "count": 0, "latency_ms": None} for name, _, _ in source_fns
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn, q, max_per_source): name for name, fn, q in source_fns}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=30):
                name = futures[future]
                _ms = round((_t_fed.time() - _t0_fed) * 1000)
                try:
                    _res = future.result()
                    # Un fetcher renvoie soit une liste, soit (liste, total_trouvé_réel).
                    # PubMed fournit son vrai hitcount (esearchresult.count), qui peut
                    # dépasser les lignes rapatriées → le badge affiche le vrai total.
                    if isinstance(_res, tuple):
                        items, _found = _res
                    else:
                        items, _found = _res, None
                    _found = _found if (_found is not None and _found >= len(items)) else len(items)
                    raw_counts[name] = _found
                    all_results.extend(items)
                    source_status[name] = {
                        "status": "ok" if items else "empty",
                        "count": _found, "fetched": len(items), "latency_ms": _ms,
                    }
                    sources_queried.append(name)      # a répondu, et sa réponse est là
                except PartialSourceFetch as _pe:
                    # La moitié d'une réponse : ce qui est revenu est gardé, et le
                    # panneau le dit « partiel » au lieu de l'afficher complet.
                    logger.warning(f"federated source {name} partial: {_pe.reason}")
                    raw_counts[name] = _pe.total or len(_pe.items)
                    all_results.extend(_pe.items)
                    source_status[name] = {
                        "status": "partial", "count": raw_counts[name],
                        "fetched": len(_pe.items), "latency_ms": _ms,
                        "error": _pe.reason,
                    }
                    sources_queried.append(name)
                except Exception as _fe:
                    # PAS dans sources_queried : une source en échec n'est pas une
                    # source interrogée, et le panneau la listait parmi celles qu'il
                    # annonçait avoir fouillées.
                    logger.warning(f"federated source {name} error: {_fe}")
                    source_status[name] = {
                        "status": "error", "count": 0, "latency_ms": _ms,
                        "error": str(_fe)[:200],
                    }
        except concurrent.futures.TimeoutError:
            logger.warning("federated search: certaines sources ont dépassé le délai")

    # Marquage in_local_db - literature_document n'a pas de colonne doi dédiée et
    # external_id est hétérogène selon la source (DOI brut pour Crossref/EuropePMC,
    # "pmid:<id>" pour PubMed/PROSPERO/Cochrane, URL pour OpenAlex). On compare donc
    # l'external_id stocké à la fois aux DOIs ET aux external_id des résultats
    # (auparavant : DOI seul → PubMed/PROSPERO/Cochrane/OpenAlex jamais reconnus).
    keys: set[str] = set()
    for r in all_results:
        if r.get("doi"):
            keys.add(r["doi"].lower())
        if r.get("external_id"):
            keys.add(str(r["external_id"]).lower())
    in_db_keys: set[str] = set()
    if keys:
        try:
            with engine.connect() as conn:
                # Compare les clés (DOI + external_id des résultats) à la fois à la
                # colonne external_id ET à la colonne doi stockées : un article déjà
                # ingéré depuis PubMed (external_id='pmid:…', doi renseigné) doit être
                # reconnu par le DOI d'un résultat Crossref/OpenAlex (sinon new_count
                # surestimé → ré-ingestion redondante).
                rows_db = conn.execute(text(
                    "SELECT LOWER(external_id), LOWER(doi) FROM literature_document "
                    "WHERE (LOWER(external_id) = ANY(:keys) OR LOWER(doi) = ANY(:keys)) "
                    "AND project_context = 'literev'"
                ), {"keys": list(keys)}).fetchall()
                for _r in rows_db:
                    if _r[0]:
                        in_db_keys.add(_r[0])
                    if _r[1]:
                        in_db_keys.add(_r[1])
        except Exception as _dbe:
            logger.warning(f"federated DB check error: {_dbe}")
    for r in all_results:
        _doi = (r.get("doi") or "").lower()
        _eid = str(r.get("external_id") or "").lower()
        r["in_local_db"] = bool((_doi and _doi in in_db_keys) or (_eid and _eid in in_db_keys))

    # Déduplication par DOI (sinon titre normalisé), suivi des sources
    import re as _re2
    deduped: dict[str, dict] = {}
    for r in all_results:
        key = _live_dedup_key(r) or ("id:" + str(id(r)))
        if key in deduped:
            existing = deduped[key]
            srcs = existing.setdefault("also_in_sources", [])
            if r.get("source_name") and r["source_name"] not in srcs and r["source_name"] != existing.get("source_name"):
                srcs.append(r["source_name"])
            for f in ("abstract", "year", "url", "doi"):
                if not existing.get(f) and r.get(f):
                    existing[f] = r[f]
            existing["in_local_db"] = existing.get("in_local_db") or r.get("in_local_db")
        else:
            r.setdefault("also_in_sources", [])
            deduped[key] = r
    deduped_list = list(deduped.values())

    # Scoring sémantique + lexical + hybride
    def _lexical_overlap(q_words: set[str], text_blob: str) -> float:
        if not q_words:
            return 0.0
        hay = set(_re2.findall(r"[a-z0-9]{3,}", (text_blob or "").lower()))
        if not hay:
            return 0.0
        return min(1.0, len(q_words & hay) / max(1, len(q_words)))

    q_words = set(_re2.findall(r"[a-z0-9]{3,}", (query or "").lower()))
    openai_key = os.getenv("OPENAI_API_KEY")
    q_emb = None
    res_embs: list[list[float] | None] = [None] * len(deduped_list)
    if openai_key and deduped_list:
        # Borne anti-latence : le scoring sémantique est sur le chemin de la
        # requête (l'utilisateur attend la réponse). On ne ré-embedde donc QUE les
        # N meilleurs résultats par recouvrement lexical (retrieve-then-rerank) ;
        # au-delà, semantic_score reste 0 (ces résultats sont déjà peu pertinents).
        # Évite d'embedder ~400 résultats par requête. Timeout court + garde par
        # lot pour qu'un appel lent/échoué ne fige pas ni n'annule tout le scoring.
        SEM_SCORE_CAP = 60
        cand_idx = sorted(
            range(len(deduped_list)),
            key=lambda i: _lexical_overlap(
                q_words,
                (deduped_list[i].get("title", "") or "") + " " + (deduped_list[i].get("abstract") or "")),
            reverse=True,
        )[:SEM_SCORE_CAP]
        try:
            from llm_usage import MeteredOpenAI as _OAI
            _client = _OAI(api_key=openai_key, timeout=8.0)
            q_emb = _client.embeddings.create(
                input=[(query or "").replace("\n", " ").strip()],
                model=_model("embedding"),
            ).data[0].embedding
            texts = [((deduped_list[i].get("title", "") or "") + ". " + (deduped_list[i].get("abstract") or "")).replace("\n", " ").strip()[:2000]
                     for i in cand_idx]
            for b in range(0, len(texts), 256):
                try:
                    emb_resp = _client.embeddings.create(input=texts[b:b + 256], model=_model("embedding"))
                    for j, d in enumerate(emb_resp.data):
                        res_embs[cand_idx[b + j]] = d.embedding
                except Exception as _be:
                    logger.warning(f"federated scoring batch error: {_be}")
        except Exception as _ee:
            logger.warning(f"federated scoring embed error: {_ee}")
            q_emb = None

    def _cosine(a, b) -> float:
        if not a or not b:
            return 0.0
        import math as _m
        dot = sum(x * y for x, y in zip(a, b))
        na = _m.sqrt(sum(x * x for x in a)); nb = _m.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    for i, r in enumerate(deduped_list):
        blob = (r.get("title", "") or "") + " " + (r.get("abstract") or "")
        r["lexical_score"] = round(_lexical_overlap(q_words, blob), 4)
        sem = max(0.0, _cosine(q_emb, res_embs[i])) if (q_emb and res_embs[i]) else 0.0
        r["semantic_score"] = round(sem, 4)

    # ④ Fusion HYBRIDE par Reciprocal Rank Fusion (RRF) au lieu d'une somme pondérée :
    # on classe INDÉPENDAMMENT par lexical puis par sémantique et on somme 1/(k+rang).
    # Sans paramètre à régler, et robuste au fait que seuls les SEM_SCORE_CAP premiers
    # résultats ont un score sémantique (les autres, sem=0, ne comptent que par le
    # lexical) - une somme pondérée mélangeait ces échelles hétérogènes. k=60 (usuel).
    # Un résultat n'obtient de crédit d'un signal que si ce signal est > 0 (sinon tous
    # les ex-æquo à 0 en bas du classement pollueraient le score).
    _RRF_K = 60
    _rrf = [0.0] * len(deduped_list)
    _order_lex = sorted(range(len(deduped_list)),
                        key=lambda i: deduped_list[i]["lexical_score"], reverse=True)
    for _rank, _i in enumerate(_order_lex):
        if deduped_list[_i]["lexical_score"] > 0:
            _rrf[_i] += 1.0 / (_RRF_K + _rank + 1)
    _order_sem = sorted(range(len(deduped_list)),
                        key=lambda i: deduped_list[i]["semantic_score"], reverse=True)
    for _rank, _i in enumerate(_order_sem):
        if deduped_list[_i]["semantic_score"] > 0:
            _rrf[_i] += 1.0 / (_RRF_K + _rank + 1)
    # Normalisation en [0,1] pour l'affichage (le front affiche hybrid_score.toFixed(2)
    # comme badge) : on divise par le meilleur score RRF → 1er résultat ≈ 1.00, l'ORDRE
    # reste identique au RRF brut.
    _rrf_max = (max(_rrf) if _rrf else 0.0) or 1.0
    for _i, r in enumerate(deduped_list):
        r["hybrid_score"] = round(_rrf[_i] / _rrf_max, 4)

    deduped_list.sort(key=lambda r: r.get("hybrid_score", 0.0), reverse=True)
    return deduped_list, sources_queried, raw_counts, source_status


@app.post("/user-scenarios/{scenario_id}/search/live")
def search_live(
    scenario_id: str,
    max_per_source: int = 50,
    lang: str | None = Query(None),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Live federated search across the external sources in parallel.

    `lang` : langue de l'interface qui lance la recherche. Cinquième point d'entrée qui
    construit un corpus, il l'ignorait : la suite (résumés de clusters, brief, variables,
    actions) était donc écrite en français quel que soit le toggle, et un redémarrage
    reprenait le scénario en français puisque `pipeline_lang` avait été fixé à 'fr'."""
    from .scenarios import _launch_populate_job  # lazy: scenarios is loaded after this module
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]
    strategy = row.get("search_strategy") or {}
    pubmed_query = strategy.get("pubmed", query) if isinstance(strategy, dict) else query
    general_query = strategy.get("general", query) if isinstance(strategy, dict) else query

    all_results, sources_queried, raw_counts, source_status = _federated_live_search(
        query, max_per_source, pubmed_query=pubmed_query, general_query=general_query
    )
    new_count = sum(1 for r in all_results if not r["in_local_db"])

    # Compteur RÉEL du corpus du scénario (identique à l'onglet Corpus) pour que
    # le panneau « recherche en direct » soit cohérent avec le corpus. Le bloc
    # fédéré ci-dessus n'interroge que les APIs externes (plafonné) ; il ne
    # reflète PAS la correspondance locale réelle.
    _thr = _get_scenario_threshold(scenario_id)
    corpus_total = 0
    corpus_above = 0
    try:
        with engine.connect() as _cc:
            _cr = _cc.execute(text(f"""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE {relevant_gate_tail_sql('d', 'ars', ':thr')}) AS above
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            """), {"sid": scenario_id, "thr": _thr}).mappings().first()
        corpus_total = int(_cr["total"] or 0)
        corpus_above = int(_cr["above"] or 0)
    except Exception as _ce:
        logger.warning(f"search_live corpus count {scenario_id}: {_ce}")

    # Background ingest of new papers - via le lanceur verrouillé pour ne jamais
    # démarrer un populate concurrent (sinon les nettoyages post-ingestion se
    # marchent dessus : compteurs corrompus, liens supprimés par l'autre job).
    ingesting_background = False
    if new_count > 0:
        try:
            # LE plafond par source, celui du reste de l'application. Un 200 écrit à la
            # main ici construisait, pour le même scénario et la même requête, un corpus
            # d'une autre taille que /populate (LIVE_MAX_PER_SOURCE) et que le pipeline
            # complet : trois chemins, trois corpus, aucun moyen de savoir lequel on lit.
            status = _launch_populate_job(scenario_id, query, row.get("filters") or {},
                                          LIVE_MAX_PER_SOURCE, lang=lang)
            ingesting_background = (status == "started")
        except Exception as _be:
            logger.warning(f"search_live background ingest error: {_be}")

    return {
        "results": all_results,
        "total": len(all_results),
        "new_count": new_count,
        "corpus_total": corpus_total,
        "corpus_above_threshold": corpus_above,
        "threshold": _thr,
        "sources_queried": sources_queried,
        "source_raw_counts": raw_counts,
        "source_status": source_status,
        "ingesting_background": ingesting_background,
    }


@app.get("/search/limits")
def search_limits() -> dict[str, Any]:
    """Les plafonds et le budget de temps qu'applique CE serveur, pour l'écran de recherche.

    L'écran les écrivait en dur (« les 2 000 plus pertinentes », « jusqu'à 20 000 ») alors
    qu'ils viennent de l'environnement : la production règle LIVE_MAX_PER_SOURCE à
    100 000, et l'écran annonçait 2 000. Mêmes formules que `_run_user_scenario_populate`
    pour une recherche lancée de l'écran (sans `max_results`, donc LIVE_MAX_PER_SOURCE) :
    ce que la carte PRISMA dira après la recherche. Il n'y a plus qu'une recherche, la
    recherche exhaustive. Lecture seule, aucune clé requise.
    """
    live = int(LIVE_MAX_PER_SOURCE)
    return {
        # Toutes les notices que chaque base apparie, jusqu'à ce garde-fou.
        "per_source": max(live, int(EXHAUSTIVE_MAX_PER_SOURCE)),
        # Crossref et le passage en texte intégral d'OpenAlex, classés par pertinence.
        "ranked_per_source": max(1, min(live, int(RANKED_PASS_MAX))),
        # Le temps laissé aux bases pour tout rendre, en secondes.
        "budget_s": int(max(POPULATE_FEDERATION_BUDGET, EXHAUSTIVE_FEDERATION_BUDGET)),
    }


@app.get("/sources/health")
def sources_health(query: str = "cardiac arrest", timeout: int = 12, raw: bool = False) -> dict[str, Any]:
    """Diagnostic des sources externes (live search).

    Interroge en parallèle les SEPT sondes (PubMed, OpenAlex, Crossref, Europe PMC,
    OpenAIRE, l'API bioRxiv/medRxiv, ReliefWeb) avec une requête minimale et renvoie, par
    sonde, le statut HTTP, la latence (ms), un compteur de résultats et l'erreur
    éventuelle. Permet de diagnostiquer « sources lentes / ne répondent plus »
    directement en production (où l'accès réseau sortant diffère du sandbox). L'API
    bioRxiv/medRxiv n'est plus une source de la recherche (elle ne cherche pas par
    mots-clés ; Europe PMC indexe ces deux serveurs) : elle reste sondée parce que l'étape
    du texte intégral l'interroge par DOI. ReliefWeb est un flux séparé.

    ATTENTION : les autres fetchers de la fédération (Semantic Scholar, DOAJ,
    ClinicalTrials.gov, CORE, arXiv) ne sont PAS sondés. « 7 joignables sur 7 » ne
    signifie donc pas que toute la fédération est saine ; la réponse porte `probed` et
    `not_probed` pour que ce soit explicite.

    `raw=true` : OpenAlex reçoit la requête TELLE QUELLE (et non ses mots-clés), et DOAJ
    est sondée de la même façon. C'est le mode qui permet de bissecter, depuis la
    production et sans créer de scénario, ce qu'une de ces deux API refuse d'un booléen :
    sur HPAI_last, toutes deux ont répondu 400 au booléen portable brut.
    Lecture seule, aucune écriture, aucune clé requise.
    """
    import concurrent.futures
    import time as _t
    from datetime import datetime as _dtm, timezone as _tz
    import requests as _req

    ua = {"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}
    ncbi_key = os.getenv("NCBI_API_KEY")
    eutils_params = {"db": "pubmed", "term": query, "retmax": 1, "retmode": "json",
                     "tool": "literev", "email": "api@literev.app"}
    if ncbi_key:
        eutils_params["api_key"] = ncbi_key

    # (nom, url, params, headers, extracteur de compteur depuis le JSON)
    probes = [
        # `esearchresult.count` : le VRAI nombre d'enregistrements, celui du site PubMed.
        # Le sondage lisait la longueur de `idlist`, qui vaut au plus `retmax` (1 ici) :
        # PubMed affichait donc « 1 » dans le diagnostic là où les cinq autres sondes
        # rendaient leur total réel, et la source la plus fournie passait pour la plus
        # maigre. Le compteur sert à comparer les sources entre elles : il doit compter
        # la même chose chez toutes.
        ("PubMed (eutils)", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
         eutils_params, ua,
         lambda j: int(j.get("esearchresult", {}).get("count", 0) or 0)),
        ("OpenAlex", "https://api.openalex.org/works",
         openalex_params({"search": (query if raw else _plain_keywords(query)), "per-page": 1,
                          "select": "id,title"}), ua,
         lambda j: j.get("meta", {}).get("count")),
        ("Crossref", "https://api.crossref.org/works",
         {"query": query, "rows": 1, "select": "DOI,title"}, ua,
         lambda j: j.get("message", {}).get("total-results")),
        ("EuropePMC", EPMC_SEARCH_URL,
         {"query": epmc_query(query), "resultType": "lite", "pageSize": 1, "format": "json"}, ua,
         lambda j: j.get("hitCount")),
        # OpenAIRE (Graph API v2) : sondée AUSSI, parce que la fédération l'a vue répondre
        # 400 au booléen entier de la requête HPAI de production sans que rien d'autre que
        # le journal du populate ne le dise. `search=` reçoit la requête telle quelle :
        # sonder avec le booléen portable d'un scénario dit si c'est la syntaxe ou la
        # longueur qu'elle refuse. `header.numFound` est son total.
        ("OpenAIRE", "https://api.openaire.eu/graph/v2/researchProducts",
         {"search": query, "pageSize": 1}, {**ua, "Accept": "application/json"},
         lambda j: int((j.get("header") or {}).get("numFound"))),
        # Plus une source de recherche : l'API du texte intégral des préprints, par DOI.
        ("bioRxiv/medRxiv (full text)", "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-07/0/json",
         None, ua,
         lambda j: (j.get("messages", [{}])[0] or {}).get("total")),
        # ReliefWeb (flux SÉPARÉ, littérature grise) : sondé ici pour qu'une panne, un
        # appname non approuvé ou un quota épuisé soit VISIBLE dans le diagnostic de
        # production, et pas seulement au moment d'une ingestion.
        ("ReliefWeb (situation reports)", "https://api.reliefweb.int/v2/reports",
         {"appname": RELIEFWEB_APPNAME or "unconfigured", "limit": 0,
          "filter[field]": "disaster_type.name", "filter[value]": "Epidemic"}, ua,
         lambda j: j.get("totalCount")),
    ]

    if raw:
        import urllib.parse as _ulib
        # Le même chemin que le fetcher du populate : la requête dans le CHEMIN de l'URL.
        probes.append(("DOAJ", f"https://doaj.org/api/search/articles/{_ulib.quote(query, safe='')}",
                       {"pageSize": 1}, ua, lambda j: j.get("total")))

    def _probe(name: str, url: str, params, headers, count_fn) -> dict[str, Any]:
        t0 = _t.time()
        try:
            r = _req.get(url, params=params, headers=headers, timeout=timeout)
            ms = round((_t.time() - t0) * 1000)
            count = None
            if r.status_code == 200:
                try:
                    count = count_fn(r.json())
                except Exception:
                    count = None
            return {"source": name, "ok": r.status_code == 200, "http": r.status_code,
                    "latency_ms": ms, "count": count,
                    "error": None if r.status_code == 200 else redact_keys(r.text or "")[:200]}
        except Exception as e:
            return {"source": name, "ok": False, "http": None,
                    "latency_ms": round((_t.time() - t0) * 1000), "count": None,
                    "error": redact_keys(f"{type(e).__name__}: {e}")[:200]}

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(probes)) as ex:
        futs = {ex.submit(_probe, *p): p[0] for p in probes}
        try:
            for f in concurrent.futures.as_completed(futs, timeout=timeout + 5):
                results.append(f.result())
        except concurrent.futures.TimeoutError:
            done = {r["source"] for r in results}
            for name in futs.values():
                if name not in done:
                    results.append({"source": name, "ok": False, "http": None,
                                    "latency_ms": None, "count": None,
                                    "error": "probe timed out"})
    results.sort(key=lambda r: r["source"])
    # Les fetchers de la fédération que ce diagnostic NE sonde PAS. Les nommer dans la
    # réponse évite de lire « 6 joignables sur 6 » comme « toute la fédération est saine ».
    not_probed = [s for s in ("Semantic Scholar", "DOAJ", "ClinicalTrials.gov", "CORE", "arXiv")
                  if not (raw and s == "DOAJ")]
    return {
        "query": query,
        "checked_at": _dtm.now(_tz.utc).isoformat(),
        "sources": results,
        "reachable": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "probed": [r["source"] for r in results],
        "not_probed": not_probed,
        # Sept sondes, dont deux ne sont pas des sources de la recherche (ReliefWeb, flux
        # séparé ; l'API bioRxiv, texte intégral) : « 7 sur 12 fetchers » additionnait
        # des sondes et des fetchers. On nomme ce qui n'est pas testé, sans compte.
        "coverage_note": (
            f"{len(results)} sondes ; sources de la recherche non testées par cet endpoint : "
            f"{', '.join(not_probed) or 'aucune'}."
        ),
        "config": {
            "ncbi_api_key": bool(ncbi_key),
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            # Présence seulement, jamais la valeur : « Semantic Scholar : 429 après 3
            # tentatives » ne dit pas si la clé est configurée, et c'est la première
            # question à se poser.
            "semantic_scholar_api_key": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
            "core_api_key": bool(os.getenv("CORE_API_KEY")),
            # Sans elle, OpenAlex puise dans un budget quotidien partagé par l'adresse IP du
            # serveur, et répond 429 « Insufficient budget » une fois ce budget dépensé.
            "openalex_api_key": bool(os.getenv("OPENALEX_API_KEY")),
        },
    }


# ── Parseurs de sources littéraires (PURS / testables) ───────────────────────
# Chaque parseur transforme la réponse brute d'une API en une liste de docs
# {title, abstract, year, url, external_id, doi, source_type}. Le fetcher (closure
# dans le populate) ne fait que le HTTP + la pagination puis délègue ici - de sorte
# que la forme de chaque réponse est vérifiée par des tests SANS réseau.
def _parse_semantic_scholar(payload: dict) -> list[dict]:
    """Semantic Scholar Graph API /paper/search → docs. external_id = s2:<paperId>."""
    out: list[dict] = []
    for it in (payload.get("data") or []) if isinstance(payload, dict) else []:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        pid = it.get("paperId")
        if not title or not pid:
            continue
        ext = it.get("externalIds") or {}
        doi = ext.get("DOI") or ext.get("doi")
        year = it.get("year")
        out.append({
            "title": title, "abstract": (it.get("abstract") or None),
            "year": int(year) if isinstance(year, int) else None,
            "url": it.get("url") or (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"s2:{pid}", "source_type": "article",
        })
    return out


def _parse_doaj(payload: dict) -> list[dict]:
    """DOAJ /api/search/articles → docs. external_id = doaj:<id>."""
    out: list[dict] = []
    for it in (payload.get("results") or []) if isinstance(payload, dict) else []:
        bj = (it.get("bibjson") or {}) if isinstance(it, dict) else {}
        title = (bj.get("title") or "").strip()
        did = it.get("id") if isinstance(it, dict) else None
        if not title or not did:
            continue
        doi = None
        for ident in (bj.get("identifier") or []):
            if isinstance(ident, dict) and str(ident.get("type", "")).lower() == "doi":
                doi = ident.get("id"); break
        url = None
        for lk in (bj.get("link") or []):
            if isinstance(lk, dict) and lk.get("url"):
                url = lk.get("url"); break
        try:
            year = int(bj.get("year")) if bj.get("year") else None
        except (TypeError, ValueError):
            year = None
        out.append({
            "title": title, "abstract": (bj.get("abstract") or None), "year": year,
            "url": url or (f"https://doi.org/{doi}" if doi else None), "doi": doi,
            "external_id": f"doaj:{did}", "source_type": "article",
        })
    return out


def _parse_clinicaltrials(payload: dict) -> list[dict]:
    """ClinicalTrials.gov API v2 /studies → docs. external_id = nct:<id>."""
    out: list[dict] = []
    for st in (payload.get("studies") or []) if isinstance(payload, dict) else []:
        ps = (st.get("protocolSection") or {}) if isinstance(st, dict) else {}
        idm = ps.get("identificationModule") or {}
        nct = idm.get("nctId")
        title = (idm.get("briefTitle") or idm.get("officialTitle") or "").strip()
        if not nct or not title:
            continue
        desc = ps.get("descriptionModule") or {}
        year = None
        date = ((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date") or ""
        if len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        out.append({
            "title": title,
            "abstract": (desc.get("briefSummary") or desc.get("detailedDescription") or None),
            "year": year, "url": f"https://clinicaltrials.gov/study/{nct}",
            "doi": None, "external_id": f"nct:{nct}", "source_type": "clinical_trial",
        })
    return out


def _parse_core(payload: dict) -> list[dict]:
    """CORE API v3 /search/works → docs. external_id = core:<id>."""
    out: list[dict] = []
    for it in (payload.get("results") or []) if isinstance(payload, dict) else []:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        cid = it.get("id")
        if not title or cid is None:
            continue
        doi = it.get("doi")
        try:
            year = int(it.get("yearPublished")) if it.get("yearPublished") else None
        except (TypeError, ValueError):
            year = None
        out.append({
            "title": title, "abstract": (it.get("abstract") or None), "year": year,
            "url": it.get("downloadUrl") or (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"core:{cid}", "source_type": "article",
        })
    return out


def _parse_arxiv(xml_text: str) -> list[dict]:
    """arXiv Atom feed (export.arxiv.org/api/query) → docs. external_id = arxiv:<id>."""
    import xml.etree.ElementTree as _ET
    _ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out: list[dict] = []
    try:
        root = _ET.fromstring(xml_text)
    except Exception:
        return out
    for e in root.findall("a:entry", _ns):
        title = (e.findtext("a:title", default="", namespaces=_ns) or "").strip()
        aid_url = (e.findtext("a:id", default="", namespaces=_ns) or "").strip()
        if not title or not aid_url:
            continue
        aid = aid_url.rsplit("/abs/", 1)[-1] if "/abs/" in aid_url else aid_url.rsplit("/", 1)[-1]
        summary = (e.findtext("a:summary", default="", namespaces=_ns) or "").strip() or None
        published = e.findtext("a:published", default="", namespaces=_ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        out.append({
            "title": " ".join(title.split()), "abstract": summary, "year": year,
            "url": aid_url, "doi": e.findtext("arxiv:doi", default=None, namespaces=_ns),
            "external_id": f"arxiv:{aid}", "source_type": "preprint",
        })
    return out


def _parse_arxiv_total(xml_text: str) -> int | None:
    """Le total annoncé par arXiv (`opensearch:totalResults`), ou None.

    arXiv ne rendait aucun total au tableau d'identification : « arxiv 2 000 » au plafond
    se lisait comme ce qu'arXiv contient. Sur le premier run de production après #326, le
    sac de huit mots, lu en OU, y avait 2 000 notices ; ce total aurait dit combien de
    dizaines de milliers il y en avait derrière, et donc que la requête reçue n'était pas
    la bonne. PUR/testable."""
    import re as _re
    m = _re.search(r"<opensearch:totalResults[^>]*>\s*(\d+)\s*<", xml_text or "")
    return int(m.group(1)) if m else None


def _openaire_text(v: Any) -> str | None:
    """Extrait une chaîne d'un champ OpenAIRE (str, dict {"$":…}, ou liste)."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("$")
    if isinstance(v, list):
        for x in v:
            t = _openaire_text(x)
            if t:
                return t
    return None


def _parse_openaire_graph(payload: dict) -> list[dict]:
    """OpenAIRE Graph API v2 (/graph/v2/researchProducts) → docs. DÉFENSIF : les champs
    (descriptions, pids, publicationDate) varient en forme. external_id = openaire:<id>.
    Remplace l'ancien /search/publications (retiré le 2026-05-31)."""
    out: list[dict] = []
    results = (payload.get("results") if isinstance(payload, dict) else None) or []
    for r in results:
        if not isinstance(r, dict):
            continue
        title = (r.get("mainTitle") or _openaire_text(r.get("title")) or "").strip()
        oid = r.get("id")
        if not title or not oid:
            continue
        _desc = r.get("descriptions") or r.get("description")
        if isinstance(_desc, list):
            _desc = " ".join((d.get("value") if isinstance(d, dict) else str(d)) or "" for d in _desc)
        elif isinstance(_desc, dict):
            _desc = _desc.get("value")
        abstract = (_desc or "").strip() or None
        _pd = str(r.get("publicationDate") or r.get("publicationYear") or "")
        year = int(_pd[:4]) if _pd[:4].isdigit() else None
        doi = None
        for p in (r.get("pids") or r.get("pid") or []):
            if isinstance(p, dict):
                _sch = str(p.get("scheme") or p.get("type") or p.get("@classid") or "").lower()
                _val = p.get("value") or p.get("id") or p.get("$")
                if _sch == "doi" and _val:
                    doi = _val
                    break
        doi = _normalize_doi(doi)
        out.append({
            "title": title, "abstract": abstract, "year": year,
            "url": (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"openaire:{oid}", "source_type": "article",
        })
    return out


def _parse_openaire(payload: dict) -> list[dict]:
    """OpenAIRE search/publications (format=json) → docs. BEST-EFFORT : structure
    profondément imbriquée et variable ($-wrapped, dict-ou-liste). external_id = doi:<…>
    ou openaire:<objIdentifier>."""
    out: list[dict] = []
    results = (((payload or {}).get("response") or {}).get("results") or {}).get("result") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        results = [results]
    for r in (results or []):
        if not isinstance(r, dict):
            continue
        res = ((r.get("metadata") or {}).get("oaf:entity") or {}).get("oaf:result")
        if not isinstance(res, dict):
            continue
        title = _openaire_text(res.get("title"))
        if not title:
            continue
        doi = None
        pids = res.get("pid")
        if isinstance(pids, dict):
            pids = [pids]
        for p in (pids or []):
            if isinstance(p, dict) and str(p.get("@classid", "")).lower() == "doi":
                doi = p.get("$"); break
        oid = _openaire_text((r.get("header") or {}).get("dri:objIdentifier")) or _openaire_text(res.get("objIdentifier"))
        eid = (f"doi:{doi}" if doi else (f"openaire:{oid}" if oid else None))
        if not eid:
            continue
        date = _openaire_text(res.get("dateofacceptance")) or ""
        out.append({
            "title": " ".join(title.split()), "abstract": _openaire_text(res.get("description")),
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "url": (f"https://doi.org/{doi}" if doi else None), "doi": doi,
            "external_id": eid, "source_type": "article",
        })
    return out


#: Le titre normalisé réunit une notice à une ligne existante quand un DOI au plus est
#: connu, ou que les deux sont égaux ; jamais quand les deux en portent un, différent.
#: Paramètres liés : has_tn, tn, has_doi, doi (cf. _ingest_doc_direct).
_TITLE_MATCH_SQL = ("(:has_tn AND title_norm = :tn AND (NOT :has_doi"
                    " OR NULLIF(btrim(doi), '') IS NULL"
                    " OR lower(btrim(doi)) = lower(CAST(:doi AS text))))")


def _refresh_title_abstract_chunks(conn, doc_ids) -> list[int]:
    """Reconstruit le chunk `title_abstract` des documents dont le résumé est arrivé APRÈS
    lui, dans la transaction de l'appelant.

    Une notice ingérée sans résumé reçoit un chunk fait de son seul titre, vectorisé et
    scoré tel quel. Quand le résumé arrive (une autre source, le backfill Europe PMC, la
    dédup qui le recopie d'un doublon), le chunk restait celui du titre : le document
    gardait un score de titre, que la porte de pertinence lit désormais comme un vrai
    score. Le chunk est réécrit (titre + résumé), son vecteur effacé pour que le worker
    le recalcule, et le score de ses liens remis à NULL : « pas encore scoré sur son
    résumé », que le scoring suivant comble, plutôt qu'un score de titre. Seuls les
    chunks qui ne contiennent pas le début du résumé sont touchés. Dans un point de
    sauvegarde : un échec (schéma sans colonne d'embedding) ne défait pas l'appelant.
    Renvoie les documents dont le chunk a été réécrit."""
    ids = sorted({int(i) for i in (doc_ids or []) if i is not None})
    if not ids:
        return []
    try:
        with conn.begin_nested():
            changed = conn.execute(text(f"""
                UPDATE document_chunk c
                   SET content = btrim(coalesce(d.title, '') || E'\\n\\n' || d.abstract),
                       embedding = NULL
                  FROM literature_document d
                 WHERE d.id = c.document_id
                   AND c.chunk_type = 'title_abstract'
                   AND d.id = ANY(CAST(:ids AS bigint[]))
                   AND {has_abstract_sql('d')}
                   AND strpos(c.content, left(btrim(d.abstract), 60)) = 0
                RETURNING c.document_id
            """), {"ids": ids}).scalars().all()
            changed = sorted({int(x) for x in changed})
            if changed:
                conn.execute(text("UPDATE article_scenarios SET similarity_score = NULL "
                                  "WHERE document_id = ANY(CAST(:ids AS bigint[]))"), {"ids": changed})
        return changed
    except Exception as _e:                                  # noqa: BLE001 - jamais bloquant
        logger.warning(f"_refresh_title_abstract_chunks: {_e}")
        return []


def _ingest_doc_direct(
    source: str,
    title: str,
    abstract: str | None,
    year: int | None,
    url: str | None,
    external_id: str | None,
    doi: str | None,
    authors: str | None = None,
    journal: str | None = None,
    source_type: str = "article",
    project_context: str = "literev",
) -> tuple[int | None, bool]:
    """INSERT SQL direct d'un document + chunk title_abstract.
    Évite les appels HTTP à l'API locale (POST /documents + POST /chunks).
    Retourne (id_document, is_new) : is_new=True si la ligne vient d'être INSÉRÉE,
    False si le document existait déjà (dédup pré-SELECT, ou course entre fetchers
    parallèles résolue par ON CONFLICT). Le compteur « ingested » ne s'incrémente
    que sur les vrais INSERT - sinon il surcompte les doublons inter-sources.
    """
    # Même classe de bug que le texte intégral, sur le chemin d'ingestion PRINCIPAL :
    # une API JSON peut renvoyer la séquence d'échappement u+0000, que json.loads décode
    # en NUL RÉEL. L'INSERT échoue alors et le document est perdu pour cette recherche.
    # On nettoie à l'entrée, avant `title_norm` (la dédup doit voir le titre déjà
    # normalisé) et avant la construction du chunk title_abstract.
    title = sanitize_db_text(title)
    abstract = sanitize_db_text(abstract)
    authors = sanitize_db_text(authors)
    journal = sanitize_db_text(journal)
    doi = _normalize_doi(doi)
    title_norm = _normalize_title(title)
    content_text = f"{title}\n\n{abstract or ''}".strip()

    # PMID dérivé de l'external_id (« pmid:123 » - format PubMed unifié, cf.
    # _live_fetch_pubmed et le populate) ou d'un external_id purement numérique
    # d'une source PubMed. Renseigne la colonne `pmid` (jusqu'ici NULL par ce
    # chemin) pour que la dédup PMID (maintenance corpus / _softdedup) opère -
    # sans dépendre du format d'external_id. Aucune signature d'appelant modifiée.
    _pmid: str | None = None
    if external_id:
        _eid_s = external_id.strip()
        _mp = re.match(r"^pmid:(\d+)$", _eid_s, re.I)
        if _mp:
            _pmid = _mp.group(1)
        elif source == "pubmed" and _eid_s.isdigit():
            _pmid = _eid_s

    # Dédup en UN SEUL aller-retour (au lieu de deux SELECT séparés) : par external_id,
    # sinon par DOI, sinon par TITRE normalisé - capte les doublons sans DOI (préprints,
    # essais). Titres < 20 car. normalisés ignorés (faux positifs). Le try/except couvre
    # le cas d'une colonne title_norm absente (migration).
    # Le titre ne réunit JAMAIS deux DOI de revue différents : dix éditions d'un « Avian
    # influenza situation update » ou deux « Correction » de même libellé sont des travaux
    # distincts, et la seconde arrivée était repliée sur la première sans laisser de trace.
    # Un DOI manquant d'un côté, ou le même DOI, réunit encore (cf. _TITLE_MATCH_SQL et
    # l'index uq_litdoc_title_norm_doi) ; deux DOI du même travail aussi (plus bas).
    _has_tn = bool(title_norm and len(title_norm) >= 20)
    _match_params = {"ctx": project_context, "eid": external_id,
                     "tn": title_norm, "has_tn": _has_tn,
                     "doi": doi, "has_doi": bool(doi)}
    try:
        with engine.connect() as _c:
            existing = _c.execute(text(f"""
                SELECT id FROM literature_document
                WHERE project_context = :ctx AND (
                    external_id = :eid
                    OR {_TITLE_MATCH_SQL}
                    OR (:has_doi AND doi = :doi))
                ORDER BY (external_id = :eid) DESC, (:has_doi AND doi = :doi) DESC
                LIMIT 1
            """), _match_params).scalar()
    except Exception:
        with engine.connect() as _c:   # repli : dédup external_id seul (colonne title_norm absente)
            existing = _c.execute(text(
                "SELECT id FROM literature_document "
                "WHERE external_id = :eid AND project_context = :ctx LIMIT 1"
            ), {"eid": external_id, "ctx": project_context}).scalar()
    if not existing and _has_tn and doi:
        # Même titre long, autre DOI : un autre travail, SAUF si les deux DOI désignent le
        # même (une version /v2, une figure figshare ou un préprint d'un article publié sous
        # ce titre ; cf. documents.same_work_doi_match). Sans cela, chaque source qui
        # renvoie une autre version ajoutait une notice, et l'étude comptait deux fois.
        try:
            with engine.connect() as _c:
                _same_title = _c.execute(text("""
                    SELECT id, doi FROM literature_document
                    WHERE project_context = :ctx AND title_norm = :tn
                      AND NULLIF(btrim(doi), '') IS NOT NULL
                    ORDER BY id LIMIT 200
                """), {"ctx": project_context, "tn": title_norm}).all()
            existing = same_work_doi_match(doi, [(r[0], r[1]) for r in _same_title])
        except Exception as _se:
            logger.warning(f"_ingest_doc_direct même travail {doi}: {_se}")
    if existing:
        # ── La notice retrouvée peut être MEILLEURE que celle en base ────────
        # Le document existait, donc on le renvoyait tel quel et on jetait le reste.
        # Or la ligne en base vient souvent de PubMed eSummary, qui ne donne PAS de
        # résumé : l'article restait sans résumé, et la règle « pas de résumé → hors
        # corpus » l'écartait de toutes les recherches à venir, alors qu'Europe PMC ou
        # Crossref venaient précisément de le fournir dans la même fédération.
        # On ne REMPLACE rien : on ne comble que les champs vides.
        if any((abstract, year, authors, journal, doi, _pmid)):
            try:
                with engine.begin() as _uc:
                    _uc.execute(text("""
                        UPDATE literature_document SET
                            abstract = COALESCE(NULLIF(TRIM(COALESCE(abstract, '')), ''), :abstract),
                            year     = COALESCE(year, :year),
                            authors  = COALESCE(NULLIF(TRIM(COALESCE(authors, '')), ''), :authors),
                            journal  = COALESCE(NULLIF(TRIM(COALESCE(journal, '')), ''), :journal),
                            doi      = COALESCE(NULLIF(TRIM(COALESCE(doi, '')), ''), :doi),
                            pmid     = COALESCE(NULLIF(TRIM(COALESCE(pmid, '')), ''), :pmid)
                        WHERE id = :id
                    """), {"id": existing, "abstract": abstract, "year": year,
                           "authors": authors, "journal": journal, "doi": doi,
                           "pmid": _pmid})
                    # Résumé enfin présent : le chunk title_abstract doit exister, sinon
                    # le document reste invisible au scoring sémantique.
                    if abstract:
                        _uc.execute(text("""
                            INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type)
                            SELECT :id, 0, :content, 'title_abstract'
                            WHERE NOT EXISTS (
                                SELECT 1 FROM document_chunk
                                WHERE document_id = :id AND chunk_type = 'title_abstract')
                        """), {"id": existing, "content": content_text})
                        # Et s'il existait déjà, construit sur le titre seul, il est
                        # reconstruit : sinon le score resterait celui du titre.
                        _refresh_title_abstract_chunks(_uc, [existing])
            except Exception as _ue:
                logger.warning(f"_ingest_doc_direct enrichissement {existing}: {_ue}")
        return (existing, False)   # dédup pré-SELECT (external_id / titre normalisé / DOI)

    # INSERT document + chunk dans UNE SEULE transaction : sinon un crash entre
    # les deux laisse un document sans chunk (jamais indexable/cherchable) - c'est
    # l'origine des documents orphelins observés en production.
    with engine.begin() as _c:
        # ON CONFLICT DO NOTHING SANS cible : capte un conflit sur N'IMPORTE quel
        # index unique - uq_literature_document_doi (DOI) ET uq_litdoc_title_norm_doi
        # (titre normalisé len≥20, et DOI). Ferme la course entre fetchers parallèles qui,
        # avant l'index unique sur le titre, pouvaient tous deux passer le pré-SELECT
        # puis INSÉRER deux lignes pour le même article sans DOI (préprint, essai).
        doc_id = _c.execute(text("""
            INSERT INTO literature_document (
                source, title, abstract, year, url, external_id,
                project_context, source_type, doi, authors, journal, title_norm, pmid
            ) VALUES (
                :source, :title, :abstract, :year, :url, :external_id,
                :project_context, :source_type, :doi, :authors, :journal, :title_norm, :pmid
            )
            ON CONFLICT DO NOTHING
            RETURNING id
        """), {
            "source": source, "title": title, "abstract": abstract,
            "year": year, "url": url, "external_id": external_id,
            "project_context": project_context, "source_type": source_type,
            "doi": doi, "authors": authors, "journal": journal,
            "title_norm": (title_norm or None), "pmid": _pmid,
        }).scalar()
        is_new = doc_id is not None
        if doc_id is None:
            # Un INSERT concurrent a gagné la course (conflit DOI ou titre normalisé).
            # On récupère la ligne canonique via les MÊMES clés que le pré-SELECT, au
            # lieu de supposer que le conflit portait sur le DOI (qui peut être NULL).
            # En dernier recours, le même titre quel que soit le DOI : tant que l'ancien
            # index unique (project_context, title_norm) n'est pas remplacé (migration au
            # démarrage), c'est lui qui a refusé la ligne, et la notice n'est pas perdue.
            doc_id = _c.execute(text(f"""
                SELECT id FROM literature_document
                WHERE project_context = :ctx AND (
                    external_id = :eid
                    OR (:has_tn AND title_norm = :tn)
                    OR (:has_doi AND doi = :doi))
                ORDER BY (external_id = :eid) DESC, (:has_doi AND doi = :doi) DESC,
                         ({_TITLE_MATCH_SQL}) DESC
                LIMIT 1
            """), _match_params).scalar()

        # INSERT du chunk title_abstract, idempotent : ne crée PAS de second chunk
        # si le document en a déjà un (cas d'un doc atteint via dédup DOI avec un
        # external_id différent - l'origine des chunks dupliqués observés).
        if doc_id is not None and len(content_text) >= 30:
            _c.execute(text("""
                INSERT INTO document_chunk (
                    document_id, chunk_index, content, chunk_type,
                    token_count, chunk_weight, metadata_json
                )
                SELECT :doc_id, 0, :content, 'title_abstract', :token_count, 1.0, '{}'
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_chunk
                    WHERE document_id = :doc_id AND chunk_type = 'title_abstract'
                )
            """), {
                "doc_id": doc_id,
                "content": content_text,
                "token_count": len(content_text.split()),
            })
    # title_norm est désormais inséré directement (colonne présente) → plus d'UPDATE séparé.
    return (doc_id, is_new)
