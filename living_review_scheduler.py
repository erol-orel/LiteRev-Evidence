#!/usr/bin/env python3
"""
living_review_scheduler.py - Scheduler de Living Review automatique par scénario GESICA

Fonctionnement :
  - Tourne en boucle (daemon) ou en mode one-shot
  - Pour chaque scénario, exécute les requêtes PubMed définies dans SCENARIO_QUERIES, puis
    les MÊMES requêtes sur les préprints (bioRxiv, medRxiv, Research Square…) par la
    recherche Europe PMC du populate
  - Lit TOUTES les notices de sa fenêtre, page par page, jusqu'à un garde-fou qu'il dit
    avoir atteint, et note le total annoncé par chaque base
  - Ingère les articles par l'ingestion standard de la bibliothèque (contexte 'literev',
    même déduplication qu'une recherche) et les LIE au scénario quand il existe
  - Les embeddings des nouveaux chunks sont calculés ensuite par le worker de l'API
  - Écrit son état (living_review_run.json) et son rapport (living_review_last_run.json),
    que lit /living-review/status ; les digests par email partent de /alerts/run-digests

Usage :
  python3 living_review_scheduler.py --mode daemon --interval-hours 24
  python3 living_review_scheduler.py --mode once --scenario epidemic-early-warning
  python3 living_review_scheduler.py --mode once --scenario usr-0123456789ab
  python3 living_review_scheduler.py --mode once --all-scenarios
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("living-review")

# ─── Configuration ────────────────────────────────────────────────────────────

def _load_secrets() -> str:
    """La clé OpenAI, via le chargeur unique du dépôt (env_files).

    Cherchait la ligne OPENAI_API_KEY= dans deux fichiers choisis ici, dont un
    `secrets.env` à la racine du dépôt qui n'a jamais existé, et ignorait celui que
    systemd passe au service. Voir l'en-tête de env_files.py."""
    if not os.environ.get("OPENAI_API_KEY"):
        load_env()
    return os.environ.get("OPENAI_API_KEY", "")

# Lu AVANT DB_URL : sinon la variable doit venir du shell alors que le fichier canonique
# la contient.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_files import load_env  # noqa: E402

load_env()

# La base est celle de l'API, jointe par SON moteur : DB_URL d'abord, comme l'API. La
# veille ouvrait sa propre connexion psycopg sur `DATABASE_URL or DB_URL`, et psycopg
# refuse l'URL SQLAlchemy que DB_URL porte en production (« postgresql+psycopg://… » :
# ProgrammingError, missing "=" in connection info string). Elle s'arrêtait donc sur
# « Connexion DB impossible » avant la première requête, avec un code de sortie 0.
DB_URL = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")

# Garde-fou PAR REQUÊTE, et non plus plafond des plus récentes. La veille demandait les 200
# notices les plus récentes de sa fenêtre, sans paginer et sans noter le total : au-delà de
# 200, les plus anciennes n'étaient jamais lues, et chaque passage rapportait les mêmes 200.
# Elle parcourt maintenant TOUTE sa fenêtre (pagination retstart, curseur Europe PMC) et ne
# s'arrête qu'ici, en disant qu'elle s'est arrêtée (`truncated`, total annoncé à côté).
# Le volume reste borné par la fenêtre elle-même, et l'ingestion standard ne crée pas de
# doublon : seul un article réellement nouveau devient un chunk à vectoriser.
# Réglable sans redéploiement via LIVING_REVIEW_MAX_RESULTS.
try:
    MAX_RESULTS_PER_QUERY = max(1, int(os.environ.get("LIVING_REVIEW_MAX_RESULTS", "5000")))
except (TypeError, ValueError):
    MAX_RESULTS_PER_QUERY = 5000

#: La fenêtre, en jours, quand ni l'appelant ni le préréglage n'en donnent une.
DEFAULT_LOOKBACK_DAYS = 30

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
#: Une page d'identifiants esearch, puis un lot efetch (le BATCH_SIZE du populate).
_ESEARCH_PAGE = 500
_EFETCH_BATCH = 200
#: eutils ne rend aucun identifiant au-delà du 10 000e d'une même recherche PubMed.
_NCBI_ESEARCH_CEILING = 10_000
#: Une page de la facette préprints d'Europe PMC, la taille qu'emploie le populate.
_EPMC_PAGE = 100

# ─── Queries PubMed par scénario ──────────────────────────────────────────────
# Les plages d'années sont OUVERTES : « 2020:3000[dp] », 3000 étant la borne « jusqu'à
# aujourd'hui » de PubMed. Elles finissaient en 2026, si bien qu'à partir du 1er janvier
# 2027 ces requêtes n'auraient plus rien trouvé de neuf, et qu'elles excluaient déjà les
# numéros 2027 sous presse. La récence, elle, est bornée par la fenêtre [EDAT].
# Les préprints sont cherchés avec ces MÊMES requêtes (cf. search_preprints) : les
# `biorxiv_terms` n'étaient appariés que par l'ancien balayage de l'API bioRxiv.

SCENARIO_QUERIES: dict[str, dict] = {
    "epidemic-early-warning": {
        "label": "Alerte précoce épidémique",
        "pubmed_queries": [
            '(influenza OR "respiratory syncytial virus" OR gastroenteritis) AND (surveillance OR "early warning" OR forecasting) AND (emergency medical services OR EMS OR prehospital)',
            '("epidemic threshold" OR "sentinel surveillance" OR "syndromic surveillance") AND (machine learning OR SARIMAX OR Prophet) AND (2022:3000[dp])',
            '("Réseau Sentinelles" OR "ILI surveillance" OR "influenza-like illness") AND (prediction OR forecast OR "time series")',
        ],
        "days_lookback": 30,
    },
    "demand-forecasting": {
        "label": "Prévision de la demande EMS",
        "pubmed_queries": [
            '("emergency medical services" OR EMS OR ambulance) AND (demand OR call OR volume) AND (forecasting OR prediction OR "time series") AND (2020:3000[dp])',
            '("ambulance demand" OR "EMS call volume") AND (machine learning OR "random forest" OR XGBoost OR LightGBM OR Prophet)',
            '(weather OR temperature OR season) AND ("EMS demand" OR "ambulance calls") AND (prediction OR forecast)',
        ],
        "days_lookback": 30,
    },
    "response-time-optimization": {
        "label": "Optimisation du temps de réponse",
        "pubmed_queries": [
            '("response time" OR "dispatch time") AND ("emergency medical services" OR ambulance) AND (optimization OR routing OR GIS) AND (2020:3000[dp])',
            '("cross-border" OR transfrontier OR "mutual aid") AND (EMS OR ambulance OR prehospital) AND (response OR coordination)',
            '("ambulance placement" OR "base location" OR "coverage optimization") AND (emergency OR EMS)',
        ],
        "days_lookback": 30,
    },
    "cardiac-arrest-prediction": {
        "label": "Prédiction arrêt cardiaque OHCA",
        "pubmed_queries": [
            '("out-of-hospital cardiac arrest" OR OHCA) AND (prediction OR forecasting OR "machine learning") AND (2020:3000[dp])',
            '(OHCA OR "cardiac arrest") AND (weather OR temperature OR season OR circadian) AND (incidence OR risk)',
            '("cardiac arrest" OR OHCA) AND ("survival" OR "return of spontaneous circulation" OR ROSC) AND (EMS OR prehospital) AND (2022:3000[dp])',
        ],
        "days_lookback": 30,
    },
    "heatwave-ems-impact": {
        "label": "Impact canicule sur les EMS",
        "pubmed_queries": [
            '(heatwave OR "heat wave" OR "extreme heat") AND ("emergency medical services" OR EMS OR ambulance) AND (impact OR demand OR calls) AND (2020:3000[dp])',
            '("heat-related illness" OR hyperthermia OR "heat stroke") AND (prehospital OR EMS OR ambulance)',
            '(UTCI OR "universal thermal climate index" OR "wet bulb globe temperature") AND (health OR mortality OR morbidity)',
        ],
        "days_lookback": 30,
    },
    "stroke-detection": {
        "label": "Détection et orientation AVC",
        "pubmed_queries": [
            '(stroke OR "cerebrovascular accident") AND ("door-to-needle" OR "door to needle" OR thrombolysis OR thrombectomy) AND (prehospital OR EMS OR ambulance) AND (2020:3000[dp])',
            '("FAST score" OR "NIHSS" OR "Cincinnati prehospital stroke scale") AND (EMS OR prehospital OR ambulance)',
            '("stroke unit" OR "comprehensive stroke center") AND ("transport time" OR "response time") AND (outcome OR survival)',
        ],
        "days_lookback": 30,
    },
    "triage-support": {
        "label": "Aide au triage EMS",
        "pubmed_queries": [
            '(triage OR "CCMU" OR "NEWS2" OR "Manchester triage") AND ("emergency medical services" OR prehospital OR ambulance) AND (2020:3000[dp])',
            '("artificial intelligence" OR "machine learning" OR "natural language processing") AND (triage OR "clinical decision support") AND (emergency OR prehospital)',
            '("undertriage" OR "overtriage") AND (EMS OR ambulance OR prehospital) AND (accuracy OR sensitivity OR specificity)',
        ],
        "days_lookback": 30,
    },
    "undertriage-risk": {
        "label": "Risque de sous-triage",
        "pubmed_queries": [
            '(undertriage OR "under-triage") AND (EMS OR ambulance OR prehospital OR emergency) AND (risk OR prediction OR detection) AND (2018:3000[dp])',
            '("triage accuracy" OR "triage error") AND (prehospital OR EMS) AND (outcome OR mortality OR morbidity)',
            '("missed diagnosis" OR "undertriage") AND (trauma OR cardiac OR stroke) AND (ambulance OR EMS)',
        ],
        "days_lookback": 60,
    },
    "trauma-care": {
        "label": "Prise en charge trauma",
        "pubmed_queries": [
            '("injury severity score" OR ISS OR TRISS OR RTS) AND (prehospital OR EMS OR ambulance) AND (outcome OR survival OR mortality) AND (2020:3000[dp])',
            '("damage control" OR "damage control resuscitation") AND (prehospital OR EMS OR trauma) AND (2020:3000[dp])',
            '("major trauma" OR "polytrauma") AND ("trauma center" OR "trauma system") AND (EMS OR ambulance OR prehospital) AND (outcome)',
        ],
        "days_lookback": 30,
    },
    "mass-casualty": {
        "label": "Événement à victimes multiples",
        "pubmed_queries": [
            '("mass casualty" OR "multiple casualty" OR MCI OR MCE) AND (EMS OR ambulance OR prehospital) AND (triage OR management OR response) AND (2018:3000[dp])',
            '("SALT triage" OR "START triage" OR "METHANE") AND (mass casualty OR disaster OR MCI)',
            '("disaster medicine" OR "mass casualty incident") AND (simulation OR planning OR preparedness) AND (EMS OR prehospital)',
        ],
        "days_lookback": 60,
    },
}

# ─── La base et les sources : celles de l'API ─────────────────────────────────

def _sqlalchemy_url(url: str) -> str:
    """Une URL libpq (« postgresql://… ») sous la forme qu'attend le moteur de l'API."""
    for plain in ("postgresql://", "postgres://"):
        if url.startswith(plain):
            return "postgresql+psycopg://" + url[len(plain):]
    return url


def _api():
    """Le moteur de l'API et ses sources, importés à la demande.

    La veille écrit par l'ingestion STANDARD de la bibliothèque (`_ingest_doc_direct`) :
    contexte 'literev', déduplication par identifiant, DOI et titre, chunk title_abstract.
    Elle écrivait ses propres lignes, en 'gesica' ou 'user' et sans lien à aucun scénario :
    l'enrichissement et les extractions, qui lisent les lignes 'literev' liées à un
    scénario, les sautaient, et le populate suivant réingérait le même article en seconde
    ligne. Importé ici et non en tête : `--list-scenarios`, et l'API qui lit
    SCENARIO_QUERIES, n'ont pas besoin de la base."""
    if not os.environ.get("DB_URL"):
        os.environ["DB_URL"] = _sqlalchemy_url(DB_URL)
    from api import core, sources
    return core, sources


def _window_days(days: Optional[int], lookback: Optional[int] = None) -> int:
    """La fenêtre d'une exécution : celle de l'APPELANT s'il en donne une, sinon celle du
    préréglage, sinon DEFAULT_LOOKBACK_DAYS.

    Le préréglage écrasait l'appelant : `?days=` de l'API et `--days` du cron n'avaient
    aucun effet, et un trou de plus de 30 ou 60 jours ne pouvait jamais être comblé."""
    if days is not None:
        return max(1, int(days))
    return max(1, int(lookback or DEFAULT_LOOKBACK_DAYS))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── État et rapport, lus par /living-review/status ───────────────────────────
RUN_STATE_FILE = "living_review_run.json"
REPORT_FILE = "living_review_last_run.json"


def report_dir() -> Path:
    """Où la veille écrit son état et son rapport : LIVING_REVIEW_REPORT_DIR, sinon le
    dossier de ce script. En production c'est /opt/literev-api, le chemin écrit en dur
    jusqu'ici ; sur un poste où ce chemin n'existe pas, le rapport n'était jamais écrit."""
    return Path(os.environ.get("LIVING_REVIEW_REPORT_DIR")
                or os.path.dirname(os.path.abspath(__file__)))


def write_run_state(state: dict) -> None:
    """L'état de l'exécution, écrit d'un bloc (fichier temporaire puis renommage) : l'API
    le lit pendant que la veille tourne, et ne doit jamais tomber sur un fichier à moitié
    écrit."""
    path = report_dir() / RUN_STATE_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"Impossible d'écrire l'état de la veille : {e}")


def read_run_state() -> Optional[dict]:
    try:
        data = json.loads((report_dir() / RUN_STATE_FILE).read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def scheduler_alive(pid) -> bool:
    """Le processus `pid` est-il une veille vivante ? Un état « running » dont le processus
    a disparu (redémarrage de l'API, mémoire) est une exécution INTERROMPUE, pas en cours.
    La ligne de commande est vérifiée quand /proc la donne : un numéro de processus est
    réattribué, et une autre commande ne doit pas passer pour la veille."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                    # vivant, mais à un autre utilisateur
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True                             # pas de /proc : on s'en tient au signal
    return b"living_review_scheduler" in cmdline


# ─── Sources d'ingestion ──────────────────────────────────────────────────────

def _eutils(endpoint: str, params: dict, timeout: int = 30):
    """Un appel eutils par le client de l'API : espacement des requêtes, clé NCBI_API_KEY,
    reprise sur 429, POST au-delà d'une certaine longueur. La seule couture réseau de
    PubMed, que les tests remplacent."""
    _, sources = _api()
    return sources._ncbi_get(f"{EUTILS_BASE}/{endpoint}", params, timeout=timeout)


def _epmc_search(params: dict, timeout: int = 30) -> dict:
    """Une page de la recherche Europe PMC. La seule couture réseau des préprints."""
    _, sources = _api()
    r = requests.get(sources.EPMC_SEARCH_URL, params=params, timeout=timeout,
                     headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"})
    r.raise_for_status()
    return r.json()


def search_pubmed(query: str, days: int = DEFAULT_LOOKBACK_DAYS,
                  max_results: int = MAX_RESULTS_PER_QUERY) -> dict:
    """Tous les PMID que la requête trouve dans la fenêtre, page par page (retstart).

    Renvoie {"ids", "total", "fetched", "truncated", "error"} : `total` est le compte que
    PubMed annonce, `truncated` dit que le garde-fou (ou une erreur) a arrêté la lecture
    avant ce total. Une page qui échoue arrête la lecture, et l'erreur est rendue."""
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
    date_to = datetime.now().strftime("%Y/%m/%d")
    # [EDAT] = date d'ENTRÉE dans PubMed (indexation), pas [PDAT] (date de publication).
    # PubMed indexe souvent un article des jours/mois après sa date de publication ;
    # une fenêtre [PDAT] ratait donc les articles récemment indexés mais publiés il y a
    # plus de `days`. Dédup à l'ingestion → élargir la fenêtre est sans risque.
    term = f"({query}) AND ({date_from}[EDAT]:{date_to}[EDAT])"
    cap = max(0, min(int(max_results), _NCBI_ESEARCH_CEILING))
    ids: list[str] = []
    seen: set[str] = set()
    total: Optional[int] = None
    error: Optional[str] = None
    retstart = 0
    while retstart < cap:
        try:
            r = _eutils("esearch.fcgi", {
                "db": "pubmed", "term": term, "retmode": "json",
                "retstart": retstart, "retmax": min(_ESEARCH_PAGE, cap - retstart),
            })
            r.raise_for_status()
            res = (r.json() or {}).get("esearchresult") or {}
            if res.get("ERROR"):
                raise RuntimeError(str(res["ERROR"])[:200])
        except Exception as e:
            error = f"esearch: {type(e).__name__}: {e}"[:300]
            logger.warning(f"PubMed esearch '{query[:60]}' (retstart={retstart}) : {e}")
            break
        total = int(res.get("count") or 0)
        page = [str(p).strip() for p in (res.get("idlist") or []) if str(p).strip()]
        for p in page:
            if p not in seen:
                seen.add(p)
                ids.append(p)
        retstart += len(page)
        if not page or retstart >= total:
            break
    return {"ids": ids, "total": total, "fetched": len(ids),
            "truncated": total is None or total > len(ids), "error": error}


def _text_of(el) -> str:
    """TOUT le texte d'un élément, balises en ligne comprises (itertext), comme le populate.

    `findtext` et `.text` s'arrêtent à la première balise : « Effect of <i>H5N1</i> on… »
    devenait « Effect of », et un résumé coupé à son premier exposant."""
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _pubmed_year(art) -> Optional[int]:
    """L'année : PubDate/Year, sinon les quatre premiers chiffres de PubDate/MedlineDate
    (« 1998 Dec-1999 Jan », « 2026 Spring »), sinon ArticleDate/Year. Lue sur Year
    seulement, l'année d'une notice datée par MedlineDate restait NULL, et tout scénario
    borné par des années l'écartait."""
    for path in (".//PubDate/Year", ".//PubDate/MedlineDate", ".//ArticleDate/Year"):
        m = re.search(r"\d{4}", art.findtext(path) or "")
        if m:
            return int(m.group(0))
    return None


def parse_pubmed_articles(xml) -> list[dict]:
    """Les notices d'une réponse efetch (XML). PUR."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    out: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext("MedlineCitation/PMID") or art.findtext(".//PMID") or "").strip()
        title = _text_of(art.find(".//ArticleTitle"))
        if not pmid or not title:
            continue
        abstract = " ".join(t for t in (_text_of(n) for n in art.findall(".//Abstract/AbstractText")) if t)
        if len(f"{title}\n\n{abstract}".strip()) < 30:      # la règle du populate
            continue
        # Le DOI de l'ARTICLE : PubmedData/ArticleIdList, et non `.//ArticleId`, qui
        # rencontre aussi les identifiants de sa liste de références.
        doi = (art.findtext("PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
               or art.findtext(".//Article/ELocationID[@EIdType='doi']") or "").strip()
        authors = []
        for a in art.findall(".//AuthorList/Author"):
            ln = (a.findtext("LastName") or "").strip()
            fn = (a.findtext("ForeName") or a.findtext("Initials") or "").strip()
            if ln:
                authors.append(f"{ln} {fn}".strip())
            elif a.find("CollectiveName") is not None:
                authors.append(_text_of(a.find("CollectiveName")))
        out.append({
            "pmid": pmid,
            "external_id": f"pmid:{pmid}",      # le format du populate et de la recherche live
            "title": title,
            "abstract": abstract,
            "year": _pubmed_year(art),
            "doi": doi or None,
            "authors": "; ".join(x for x in authors if x) or None,
            "journal": (art.findtext(".//Journal/Title")
                        or art.findtext(".//Journal/ISOAbbreviation") or None),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return out


def fetch_pubmed_records(pmids: list[str]) -> tuple[list[dict], Optional[str]]:
    """Les notices complètes de `pmids`, par lots efetch. Un lot qui échoue est signalé
    (l'erreur est rendue) et les suivants sont quand même lus."""
    out: list[dict] = []
    error: Optional[str] = None
    for i in range(0, len(pmids), _EFETCH_BATCH):
        batch = pmids[i:i + _EFETCH_BATCH]
        try:
            r = _eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                        "rettype": "xml", "retmode": "xml"}, timeout=90)
            r.raise_for_status()
            out.extend(parse_pubmed_articles(getattr(r, "content", None) or r.text))
        except Exception as e:
            error = f"efetch: {type(e).__name__}: {e}"[:300]
            logger.warning(f"PubMed efetch lot {i // _EFETCH_BATCH} : {e}")
    return out, error


def _preprint_record(res: dict) -> Optional[dict]:
    """Une notice de la facette préprints d'Europe PMC, lue comme le populate la lit."""
    _, sources = _api()
    doi = sources._normalize_doi(res.get("doi"))
    ext_id = res.get("id") or doi
    title = (res.get("title") or "").strip()
    if not ext_id or not title:
        return None
    abstract = res.get("abstractText")
    if abstract and abstract.startswith("<"):
        try:
            import xml.etree.ElementTree as ET
            abstract = "".join(ET.fromstring(f"<root>{abstract}</root>").itertext()).strip()
        except Exception:
            pass
    yt = res.get("pubYear")
    year = int(yt) if yt and str(yt).isdigit() else None
    if len(f"{title}\n\n{abstract or ''}".strip()) < 30:
        return None
    url = (f"https://europepmc.org/article/{res.get('source', 'PPR')}/{res.get('id')}"
           if res.get("id") else (f"https://doi.org/{doi}" if doi else None))
    return {"external_id": ext_id, "title": title, "abstract": abstract or None,
            "year": year, "doi": doi, "url": url,
            "authors": (res.get("authorString") or None), "journal": None}


def search_preprints(query: str, days: int = DEFAULT_LOOKBACK_DAYS,
                     max_results: int = MAX_RESULTS_PER_QUERY) -> dict:
    """Les préprints de la fenêtre que la MÊME requête apparie, par la facette préprints
    d'Europe PMC (SRC:PPR) : la recherche du populate (_fetch_preprints dans
    api/pipeline.py), bornée à la fenêtre par la date de première publication.

    Remplace un balayage de l'API bioRxiv qui ne lisait que la première page (100 notices
    sur 30 jours), appariait les mots comme des sous-chaînes (« ems » trouvait « systems »),
    datait 2024 toute notice sans date et coupait à 20 résultats. FIRST_PDATE et non une
    date d'indexation : Europe PMC moissonne les serveurs de préprints quotidiennement, et
    le recouvrement des fenêtres absorbe ces quelques jours."""
    _, sources = _api()
    core_q = sources.epmc_query(query)
    if not core_q:
        return {"query": None, "records": [], "total": 0, "fetched": 0,
                "truncated": False, "error": None}
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    q = f"({core_q}) AND (SRC:PPR) AND (FIRST_PDATE:[{date_from} TO {date_to}])"
    cap = max(0, int(max_results))
    records: list[dict] = []
    total: Optional[int] = None
    error: Optional[str] = None
    fetched, cursor = 0, "*"
    while fetched < cap:
        try:
            data = _epmc_search({"query": q, "format": "json", "resultType": "core",
                                 "pageSize": _EPMC_PAGE, "cursorMark": cursor})
        except Exception as e:
            error = f"europepmc: {type(e).__name__}: {e}"[:300]
            logger.warning(f"Préprints (Europe PMC) '{query[:60]}' : {e}")
            break
        if total is None:
            try:
                total = int(data.get("hitCount") or 0)
            except (TypeError, ValueError):
                total = None
        results = (data.get("resultList") or {}).get("result") or []
        if not results:
            break
        for res in results[:cap - fetched]:
            rec = _preprint_record(res) if isinstance(res, dict) else None
            if rec:
                records.append(rec)
        fetched += min(len(results), cap - fetched)
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor or len(results) < _EPMC_PAGE:
            break
        cursor = nxt
        time.sleep(0.3)
    return {"query": q, "records": records, "total": total, "fetched": fetched,
            "truncated": total is None or total > fetched, "error": error}


# ─── Bibliothèque et scénarios ────────────────────────────────────────────────

def _existing_scenario(scenario_id: str) -> Optional[dict]:
    """Le scénario qui porte cet identifiant, s'il existe : c'est celui auquel un préréglage
    appartient (les scénarios GESICA sont des lignes de user_scenarios)."""
    core, _ = _api()
    with core.engine.connect() as c:
        row = c.execute(text("SELECT id, filters FROM user_scenarios WHERE id = :s"),
                        {"s": scenario_id}).mappings().first()
    return dict(row) if row else None


def _known_pmids(pmids: list[str]) -> dict[str, int]:
    """Les PMID que la bibliothèque détient déjà (contexte 'literev') → id du document.

    Même lecture que le populate (colonne `pmid`, ou external_id « pmid:<n> ») : ceux-là
    sont liés sans être retéléchargés, seuls les autres passent par efetch. Une ligne d'un
    autre contexte ne compte pas : c'est justement celle que les extractions ne lisent pas.
    En cas d'échec, tout passe par efetch, et la dédup de l'ingestion fait le reste."""
    if not pmids:
        return {}
    core, _ = _api()
    want, known = set(pmids), {}
    try:
        with core.engine.connect() as c:
            for i in range(0, len(pmids), 1000):
                chunk = pmids[i:i + 1000]
                rows = c.execute(text("""
                    SELECT id, pmid::text AS pmid, external_id FROM literature_document
                     WHERE project_context = 'literev' AND pmid = ANY(:p)
                    UNION
                    SELECT id, pmid::text AS pmid, external_id FROM literature_document
                     WHERE project_context = 'literev' AND external_id = ANY(:e)
                """), {"p": chunk, "e": [f"pmid:{p}" for p in chunk]}).mappings().all()
                for r in rows:
                    pm = str(r["pmid"] or "").strip()
                    if not pm and str(r["external_id"] or "").startswith("pmid:"):
                        pm = str(r["external_id"])[5:]
                    if pm in want:
                        known.setdefault(pm, int(r["id"]))
    except Exception as e:
        logger.warning(f"PMID déjà connus : lecture impossible ({e}) ; efetch pour tous")
        return {}
    return known


def _in_library(rec: dict) -> bool:
    """La notice est-elle déjà dans la bibliothèque ? Les clés de `_ingest_doc_direct`
    (identifiant externe, DOI, titre normalisé de 20 signes ou plus), en lecture seule :
    un essai à blanc n'écrit rien, mais doit compter comme le ferait une vraie exécution."""
    core, sources = _api()
    doi = sources._normalize_doi(rec.get("doi"))
    tn = sources._normalize_title(rec.get("title"))
    try:
        with core.engine.connect() as c:
            return c.execute(text("""
                SELECT 1 FROM literature_document
                 WHERE project_context = 'literev' AND (
                       external_id = :eid
                       OR (:has_tn AND title_norm = :tn)
                       OR (:has_doi AND doi = :doi))
                 LIMIT 1
            """), {"eid": rec.get("external_id"), "tn": tn, "has_tn": len(tn) >= 20,
                   "doi": doi, "has_doi": bool(doi)}).scalar() is not None
    except Exception:
        return False


def _ingest_records(records: list[dict], *, source: str, source_type: str,
                    dry_run: bool, tally: dict) -> None:
    """Chaque notice par l'ingestion standard (contexte 'literev', même dédup que la
    recherche). Les ids retenus vont dans `tally["doc_ids"]`, pour le lien au scénario."""
    _, sources = _api()
    for rec in records:
        if dry_run:
            if _in_library(rec):
                tally["known"] += 1
            else:
                tally["new"] += 1
                tally["titles"].append(rec["title"][:80])
            continue
        try:
            doc_id, is_new = sources._ingest_doc_direct(
                source=source, title=rec["title"], abstract=rec.get("abstract") or None,
                year=rec.get("year"), url=rec.get("url"), external_id=rec["external_id"],
                doi=rec.get("doi"), authors=rec.get("authors"), journal=rec.get("journal"),
                source_type=source_type, project_context="literev",
            )
        except Exception as e:
            logger.warning(f"Ingestion '{rec.get('title', '')[:60]}' : {e}")
            tally["errors"] += 1
            continue
        if doc_id is None:
            tally["errors"] += 1
            continue
        tally["doc_ids"].add(int(doc_id))
        if is_new:
            tally["new"] += 1
            tally["titles"].append(rec["title"][:80])
        else:
            tally["known"] += 1       # « déjà connu » est le cas normal d'une veille


def _link_documents(scenario_id: str, doc_ids) -> int:
    """Lie des documents au scénario : LA même insertion que la recherche et le populate
    (score NULL, renseigné au prochain scoring ; ON CONFLICT DO NOTHING). Renvoie le nombre
    de liens CRÉÉS. `assigned_at` prend sa valeur par défaut, now() : la date à laquelle
    l'article est entré dans le corpus du scénario, celle que lisent les alertes."""
    ids = sorted({int(d) for d in doc_ids if d is not None})
    if not ids:
        return 0
    core, _ = _api()
    with core.engine.begin() as c:
        res = c.execute(text("""
            INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
            SELECT unnest(CAST(:ids AS bigint[])), :sid, NULL
            ON CONFLICT (document_id, scenario_id) DO NOTHING
        """), {"ids": ids, "sid": scenario_id})
    return int(res.rowcount or 0)


def _corpus_changed(scenario_id: str) -> None:
    """Le corpus du scénario a gagné des liens : la copie de son compte d'articles et ses
    vues dérivées (clustering, graphe, carte des concepts, actions, projection SEIR) sont
    périmées. Fait ICI, là où les liens se créent, pour qu'une veille lancée par cron les
    invalide aussi : seul le déclencheur de l'API le faisait, d'après le rapport."""
    from api.scenario_store import CORPUS_DERIVED_CACHE_RESET
    core, _ = _api()
    try:
        with core.engine.begin() as c:
            c.execute(text(f"UPDATE scenario_settings SET {CORPUS_DERIVED_CACHE_RESET} "
                           "WHERE scenario_id = :sid"), {"sid": scenario_id})
    except Exception as e:
        logger.warning(f"[{scenario_id}] invalidation des vues dérivées : {e}")
    try:
        with core.engine.begin() as c:
            c.execute(text("""
                UPDATE user_scenarios
                SET article_count = (
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                ),
                updated_at = NOW()
                WHERE id = :sid
            """), {"sid": scenario_id})
    except Exception as e:
        logger.warning(f"[{scenario_id}] compte d'articles : {e}")


# ─── Logique principale par scénario ─────────────────────────────────────────

def _review(scenario_id: str, *, label: str, queries: list[str], window: int,
            target: Optional[dict], dry_run: bool, with_preprints: bool,
            max_results: int = MAX_RESULTS_PER_QUERY) -> dict:
    """Une veille : chaque requête sur PubMed (et sur les préprints), toutes les notices de
    la fenêtre, l'ingestion standard, puis le lien au scénario `target` s'il existe."""
    from api.search import excludes_preprints
    # Un scénario qui écarte les préprints ne doit pas en recevoir par la veille : ni la
    # facette préprints, ni les préprints que PubMed indexe (même règle que le populate).
    filters = (target or {}).get("filters")
    if isinstance(filters, str):
        try:
            filters = json.loads(filters)
        except ValueError:
            filters = None
    no_preprints = bool(target) and excludes_preprints(filters if isinstance(filters, dict) else None)
    tally: dict[str, Any] = {"new": 0, "known": 0, "errors": 0, "titles": [], "doc_ids": set()}
    searches: list[dict] = []
    # Une notice que deux requêtes du même scénario renvoient n'est traitée qu'une fois :
    # sinon elle serait retéléchargée, et comptée « déjà en base » par la seconde.
    seen: set[str] = set()

    for q in queries:
        q_pm = f"({q}) NOT preprint[pt]" if no_preprints else q
        s = search_pubmed(q_pm, days=window, max_results=max_results)
        fresh = [p for p in s["ids"] if f"pmid:{p}" not in seen]
        seen.update(f"pmid:{p}" for p in fresh)
        known = _known_pmids(fresh)
        tally["known"] += len(known)
        tally["doc_ids"].update(known.values())
        unknown = [p for p in fresh if p not in known]
        records, err = fetch_pubmed_records(unknown) if unknown else ([], None)
        _ingest_records(records, source="pubmed", source_type="article",
                        dry_run=dry_run, tally=tally)
        searches.append({"source": "pubmed", "query": q_pm, "total": s["total"],
                         "fetched": s["fetched"], "truncated": s["truncated"],
                         "error": s["error"] or err})

    if with_preprints and not no_preprints:
        for q in queries:
            p = search_preprints(q, days=window, max_results=max_results)
            records = [r for r in p["records"] if r["external_id"] not in seen]
            seen.update(r["external_id"] for r in records)
            _ingest_records(records, source="preprint", source_type="preprint",
                            dry_run=dry_run, tally=tally)
            searches.append({"source": "preprints", "query": p["query"], "total": p["total"],
                             "fetched": p["fetched"], "truncated": p["truncated"],
                             "error": p["error"]})

    linked, link_error = 0, None
    if target and not dry_run and tally["doc_ids"]:
        try:
            linked = _link_documents(target["id"], tally["doc_ids"])
        except Exception as e:
            link_error = f"{type(e).__name__}: {e}"[:300]
            logger.error(f"[{scenario_id}] lien au scénario : {e}")
        # On n'invalide QUE ce qui a changé : un cycle qui ne ramène rien, le cas de loin
        # le plus fréquent, ne doit pas faire recalculer le clustering, le graphe, la carte
        # des concepts, les actions et la projection du scénario.
        if linked:
            _corpus_changed(target["id"])

    result = {
        "scenario_id": scenario_id,
        "label": label,
        "run_at": _now_iso(),
        "window_days": window,
        "new_documents": tally["new"],
        "already_in_library": tally["known"],
        "skipped_existing": tally["known"],        # ancien nom du même compte
        "errors": tally["errors"],
        "linked_scenario_id": target["id"] if target else None,
        "linked_to_scenario": linked,
        "scenario_missing": target is None,
        "preprints_excluded": no_preprints,
        "searches": searches,
        "truncated": any(s["truncated"] for s in searches),
        "dry_run": dry_run,
        "new_titles": tally["titles"][:5],
    }
    if link_error:
        result["link_error"] = link_error
    logger.info(
        f"[{scenario_id}] Terminé - {tally['new']} nouveaux, {tally['known']} déjà en base, "
        f"{linked} liés au scénario, {tally['errors']} erreurs")
    return result


def run_living_review_for_scenario(
    scenario_id: str,
    dry_run: bool = False,
    days: Optional[int] = None,
) -> dict:
    """
    Lance la living review pour un scénario préréglé.
    Retourne un résumé des nouvelles publications.
    """
    config = SCENARIO_QUERIES.get(scenario_id)
    if not config:
        return {"scenario_id": scenario_id, "error": f"Scénario '{scenario_id}' inconnu"}

    label = config["label"]
    # Fenêtre : celle de l'appelant s'il en donne une, sinon celle du préréglage
    # (days_lookback=60 pour undertriage-risk / mass-casualty), sinon 30 jours.
    window = _window_days(days, config.get("days_lookback"))
    logger.info(f"[{scenario_id}] Living review - {label} (fenêtre {window} j)")
    # Le scénario auquel le préréglage appartient : celui qui porte son identifiant.
    # Trois préréglages (undertriage-risk, trauma-care, mass-casualty) n'en ont aucun :
    # leurs articles entrent dans la bibliothèque sans lien, et le rapport le dit.
    target = _existing_scenario(scenario_id)
    if target is None:
        logger.warning(f"[{scenario_id}] aucun scénario ne porte cet identifiant : "
                       f"articles ingérés dans la bibliothèque, liés à aucun scénario")
    return _review(scenario_id, label=label, queries=config.get("pubmed_queries", []),
                   window=window, target=target, dry_run=dry_run, with_preprints=True)


def run_all_scenarios(dry_run: bool = False, days: Optional[int] = None) -> list[dict]:
    """Lance la living review pour tous les scénarios préréglés."""
    results = []
    for scenario_id in SCENARIO_QUERIES:
        result = run_living_review_for_scenario(scenario_id, dry_run=dry_run, days=days)
        results.append(result)
        time.sleep(1)  # politesse entre scénarios
    return results


def run_user_scenarios(dry_run: bool = False, days: Optional[int] = None,
                       max_results: int = MAX_RESULTS_PER_QUERY,
                       only: Optional[str] = None) -> list[dict]:
    """Living review pour les scénarios UTILISATEUR possédés (owner_email renseigné), ou
    pour le seul scénario `only` (possédé ou non) : exécute leur `query` sur PubMed,
    ingère les nouveautés dans la bibliothèque et les LIE au scénario.

    Le lien est créé ici, avec un score NULL renseigné au prochain scoring, comme le fait
    le populate pour ce que PubMed renvoie : sans lui, ces articles n'entraient dans le
    corpus du scénario qu'au populate suivant, et l'ingestion d'avant les rangeait dans
    un contexte 'user' que rien d'autre ne lit."""
    results: list[dict] = []
    core, _ = _api()
    sql = ("SELECT id, name, query, filters FROM user_scenarios "
           "WHERE query IS NOT NULL AND query <> ''")
    params: dict[str, Any] = {}
    if only:
        sql += " AND id = :id AND COALESCE(is_system, FALSE) = FALSE"
        params["id"] = only
    else:
        sql += " AND owner_email IS NOT NULL"
    try:
        with core.engine.connect() as c:
            scenarios = [dict(r) for r in c.execute(text(sql), params).mappings().all()]
    except Exception as e:
        logger.error(f"Chargement des scénarios utilisateur impossible : {e}")
        return results

    for row in scenarios:
        sid = row["id"]
        res = _review(sid, label=row.get("name") or sid, queries=[row["query"]],
                      window=_window_days(days), target=row, dry_run=dry_run,
                      with_preprints=False, max_results=max_results)
        res["user_scenario"] = True
        results.append(res)
        time.sleep(1)
    return results


def save_run_report(results: list[dict], output_dir: Optional[str] = None):
    """Sauvegarde un rapport JSON de la dernière exécution.

    Il dit aussi ce qui n'a PAS été fait : les préréglages qu'aucun scénario ne porte
    (leurs articles sont dans la bibliothèque, liés à rien) et les scénarios dont une
    recherche s'est arrêtée avant le total annoncé."""
    report = {
        "generated_at": _now_iso(),
        "total_new_documents": sum(r.get("new_documents", 0) for r in results),
        "total_linked_to_scenarios": sum(r.get("linked_to_scenario", 0) for r in results),
        "presets_without_scenario": sorted(
            r["scenario_id"] for r in results if r.get("scenario_missing")),
        "truncated_scenarios": sorted(r["scenario_id"] for r in results if r.get("truncated")),
        "scenarios": results,
    }
    path = Path(output_dir) if output_dir else report_dir()
    path = path / REPORT_FILE
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        logger.info(f"Rapport sauvegardé : {path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le rapport : {e}")
    return report


def _db_ready() -> None:
    """La base répond, ou l'exécution s'arrête en ERREUR : elle finissait en code 0."""
    core, _ = _api()
    with core.engine.connect() as c:
        c.execute(text("SELECT 1"))


def run_once(*, scenario: Optional[str] = None, all_scenarios: bool = False,
             user_scenarios: bool = False, days: Optional[int] = None,
             dry_run: bool = False) -> int:
    """Une exécution, avec son état écrit au début et à la fin. Renvoie le code de sortie.

    L'API la lance détachée et lit cet état : sans lui, une exécution longue n'avait pour
    trace que les journaux, et celle que l'API tuait à 10 minutes ne laissait rien."""
    base = {"pid": os.getpid(), "started_at": _now_iso(),
            "scenario": scenario or ("all" if all_scenarios else None),
            "user_scenarios": bool(user_scenarios), "days": days, "dry_run": bool(dry_run)}
    write_run_state({**base, "state": "running"})
    try:
        _db_ready()
        results: list[dict] = []
        if scenario:
            if scenario in SCENARIO_QUERIES:
                results.append(run_living_review_for_scenario(scenario, dry_run=dry_run, days=days))
            else:
                # Un scénario utilisateur : `--scenario` ne connaissait que les préréglages,
                # et l'API passait `--user-scenarios`, qui traitait TOUS les scénarios
                # possédés sauf, faute de propriétaire, souvent celui qu'on demandait.
                results += run_user_scenarios(dry_run=dry_run, days=days, only=scenario)
                if not results:
                    results.append({"scenario_id": scenario,
                                    "error": f"Scénario '{scenario}' inconnu"})
        else:
            if all_scenarios:
                results += run_all_scenarios(dry_run=dry_run, days=days)
            if user_scenarios:
                results += run_user_scenarios(dry_run=dry_run, days=days)
        report = save_run_report(results)
    except Exception as e:
        logger.error(f"Living review interrompue : {e}", exc_info=True)
        write_run_state({**base, "state": "error", "finished_at": _now_iso(),
                         "error": f"{type(e).__name__}: {e}"[:500]})
        return 1
    write_run_state({**base, "state": "done", "finished_at": _now_iso(),
                     "total_new_documents": report["total_new_documents"],
                     "total_linked_to_scenarios": report["total_linked_to_scenarios"],
                     "presets_without_scenario": report["presets_without_scenario"],
                     "truncated_scenarios": report["truncated_scenarios"]})
    print(f"\nLiving review terminée - {report['total_new_documents']} nouveaux documents, "
          f"{report['total_linked_to_scenarios']} liés à un scénario")
    for r in results:
        status = "DRY-RUN" if dry_run else ("ERREUR" if r.get("error") else "OK")
        tag = "user" if r.get("user_scenario") else "preset"
        print(f"  [{status}] ({tag}) {r['scenario_id']:35s} +{r.get('new_documents', 0)} nouveaux, "
              f"{r.get('skipped_existing', 0)} existants, {r.get('linked_to_scenario', 0)} liés")
    if report["presets_without_scenario"]:
        print("  Sans scénario (ingérés, liés à rien) : "
              + ", ".join(report["presets_without_scenario"]))
    return 0


# ─── Daemon mode ─────────────────────────────────────────────────────────────

def run_daemon(interval_hours: int = 24, days: Optional[int] = None):
    """Tourne en boucle, exécute la living review toutes les N heures."""
    logger.info(f"Démarrage du daemon living review - intervalle {interval_hours}h")
    while True:
        run_once(all_scenarios=True, days=days)
        logger.info(f"Prochain cycle dans {interval_hours}h")
        time.sleep(interval_hours * 3600)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Living Review Scheduler GESICA")
    parser.add_argument("--mode", choices=["once", "daemon"], default="once",
                        help="Mode d'exécution : once (une fois) ou daemon (boucle)")
    parser.add_argument("--scenario", default=None,
                        help="ID du scénario à traiter : un préréglage (ex: epidemic-early-warning) "
                             "ou un scénario utilisateur (usr-…)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Traiter tous les scénarios préréglés")
    parser.add_argument("--user-scenarios", action="store_true",
                        help="Traiter aussi (ou uniquement) les scénarios utilisateur possédés (owner_email)")
    parser.add_argument("--days", type=int, default=None,
                        help="Nombre de jours de lookback (défaut : celui du préréglage, sinon 30)")
    parser.add_argument("--interval-hours", type=int, default=24,
                        help="Intervalle en heures pour le mode daemon (défaut: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation sans écriture en base")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="Lister tous les scénarios disponibles")
    args = parser.parse_args()

    if args.list_scenarios:
        print("\nScénarios disponibles :")
        for sid, cfg in SCENARIO_QUERIES.items():
            print(f"  {sid:35s} - {cfg['label']}")
        return 0

    if args.mode == "daemon":
        run_daemon(interval_hours=args.interval_hours, days=args.days)
        return 0

    if not (args.scenario or args.all_scenarios or args.user_scenarios):
        parser.print_help()
        return 0
    return run_once(scenario=args.scenario, all_scenarios=args.all_scenarios,
                    user_scenarios=args.user_scenarios, days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
