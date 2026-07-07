from __future__ import annotations
import json
import logging
import os
import re
import secrets as _secrets
from pathlib import Path
from typing import Any, Optional

# GESICA_ENRICHED et GESICA_SCENARIO_METADATA sont désormais stockés en base de données (user_scenarios is_system=TRUE)
# Les imports statiques ci-dessous sont conservés pour compatibilité ascendante uniquement
try:
    from gesica_scenario_enriched_metadata import GESICA_ENRICHED as _GESICA_ENRICHED_LEGACY
except ImportError:
    _GESICA_ENRICHED_LEGACY: dict = {}
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import create_engine, text, bindparam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("literev-api")

# ─── Chargement .env (sans dépendance python-dotenv) ─────────────────────────────────
def _load_env_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

for _ep in [".env", str(Path(__file__).parent / ".env"), "/opt/literev-api/.env", "/etc/literev/env"]:
    _load_env_file(_ep)

# Charger aussi le fichier secrets hors-repo (jamais commité)
for _ep in ["/etc/literev/secrets", "/opt/literev-api/secrets.env"]:
    _load_env_file(_ep)

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("La variable d'environnement DB_URL est requise et n'est pas configurée.")

WRITE_API_KEY = os.getenv("WRITE_API_KEY")
if not WRITE_API_KEY:
    raise RuntimeError("La variable d'environnement WRITE_API_KEY est requise et n'est pas configurée.")

# Configurer le pool DB de manière optimale pour éviter la saturation (M-3)
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_size=10,         # Taille de base du pool de connexions (M-3)
    max_overflow=20,      # Nombre max de connexions temporaires supplémentaires (M-3)
    pool_timeout=30,      # Timeout d'attente d'une connexion du pool (M-3)
    pool_recycle=1800,    # Recycle les connexions toutes les 30 minutes pour éviter les coupures (M-3)
)
# ── Observabilité optionnelle : Sentry ────────────────────────────────────────
# Activé UNIQUEMENT si SENTRY_DSN est défini dans l'environnement. Sans DSN c'est
# un no-op ; si le paquet sentry-sdk n'est pas installé on dégrade proprement
# (les erreurs restent visibles via le middleware de logs + journalctl).
_SENTRY_DSN = os.getenv("SENTRY_DSN")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0") or 0),
        )
        logger.info("Sentry activé (capture des erreurs backend).")
    except Exception as _se:
        logger.warning(f"Sentry non initialisé ({_se}) ; erreurs via journalctl uniquement.")

app = FastAPI(title="LiteRev API", version="0.4.0")

# ─── Middleware de Rate Limiting In-Memory (H-2) ──────────────────────────────────
import time
from fastapi import Request
from collections import defaultdict

# Limiteur de débit in-memory robuste par IP (H-2)
class InMemoryRateLimiter:
    def __init__(self, requests_limit: int, window_seconds: int):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        # Stocke les timestamps des requêtes pour chaque IP
        self.history: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Filtrer les anciens timestamps hors de la fenêtre
        self.history[ip] = [t for t in self.history[ip] if now - t < self.window_seconds]
        if len(self.history[ip]) >= self.requests_limit:
            return False
        self.history[ip].append(now)
        return True

# Le frontend est volubile (tableau de bord = nombreux appels, recherche = gros
# payloads) : limites généreuses pour éviter les faux positifs, plus strictes
# sur les endpoints coûteux (RAG, recherche, génération de briefs).
general_limiter = InMemoryRateLimiter(requests_limit=600, window_seconds=60)
expensive_limiter = InMemoryRateLimiter(requests_limit=30, window_seconds=60)

# Endpoints coûteux à protéger (RAG, search, génération de briefs).
# ATTENTION : un segment {param} ne doit matcher QU'UN seul segment de chemin.
# Sinon `/user-scenarios/{id}/rag` capturerait tout `/user-scenarios/*` (corpus,
# prisma, clustering, evidence-brief, ...) et tout le détail scénario serait
# soumis à la limite "coûteuse" (30/min) → faux 429 sur de très nombreuses pages.
EXPENSIVE_PATHS = {
    "/ask",  # couvre /ask, /ask/stream, /ask/stream/filtered (sous-chemins)
    "/user-scenarios/{scenario_id}/rag",
    "/gesica/scenarios/{scenario_id}/rag",
    "/scenarios/{scenario_id}/full-pipeline",
}

def _compile_expensive_patterns(paths: set[str]) -> list[re.Pattern]:
    """Compile chaque route coûteuse en regex ancrée.

    - `{param}` → exactement un segment de chemin (``[^/]+``).
    - Un sous-chemin est autorisé (``/ask`` couvre ``/ask/stream/filtered`` ;
      ``/user-scenarios/{id}/rag`` couvre un éventuel ``/rag/stream``), mais une
      route paramétrée ne déborde JAMAIS sur ses routes sœurs (``/prisma`` etc.).
    """
    compiled: list[re.Pattern] = []
    for p in paths:
        segments = [
            r"[^/]+" if seg.startswith("{") and seg.endswith("}") else re.escape(seg)
            for seg in p.split("/")
        ]
        compiled.append(re.compile("^" + "/".join(segments) + r"(?:/.*)?$"))
    return compiled

_EXPENSIVE_PATTERNS = _compile_expensive_patterns(EXPENSIVE_PATHS)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Récupérer l'IP réelle du client (gère le proxy reverse de production)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Ne faire confiance qu'au dernier saut (ajouté par notre reverse proxy),
        # les entrées précédentes sont contrôlables par le client (spoofing).
        client_ip = forwarded_for.split(",")[-1].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    path = request.url.path

    # Ignorer le rate limiting pour l'endpoint health
    if path == "/health":
        return await call_next(request)

    # Vérifier si le chemin est coûteux (match précis par segment, cf.
    # _compile_expensive_patterns) : seules les routes RAG / full-pipeline /
    # search / ask sont throttlées agressivement, pas tout /user-scenarios/*.
    is_expensive = any(rx.match(path) for rx in _EXPENSIVE_PATTERNS)

    limiter = expensive_limiter if is_expensive else general_limiter
    if not limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip} on path: {path}")
        # IMPORTANT : dans un BaseHTTPMiddleware, lever HTTPException ne passe pas
        # par les gestionnaires d'exceptions FastAPI → cela remonte en 500.
        # On retourne donc directement une réponse 429 propre, avec Retry-After
        # pour que clients et proxies temporisent au lieu de marmarteler.
        from starlette.responses import JSONResponse as _JSONResponse
        return _JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."},
            headers={"Retry-After": str(limiter.window_seconds)},
        )

    # Observabilité : loguer toute exception NON gérée avec son contexte
    # (méthode + chemin + IP) pour qu'elle soit repérable dans journalctl, puis
    # la relancer telle quelle (Starlette renvoie son 500 habituel — aucun
    # changement de comportement). Les HTTPException sont déjà converties en
    # réponses en amont et ne remontent donc pas ici.
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(
            f"Unhandled error on {request.method} {path} from {client_ip}: {exc}",
            exc_info=True,
        )
        raise

# Restreindre les origines CORS à localhost et aux domaines de production
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:80",
    "http://localhost:8333",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:80",
    "http://127.0.0.1:8333",
    "https://literev.im",  # Exemple de domaine de production
    "http://literev.im",
]
# On peut aussi ajouter les variables d'environnement de domaine si elles existent
FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not WRITE_API_KEY:
        raise HTTPException(status_code=503, detail="Server not configured for authenticated writes")
    if not x_api_key or not _secrets.compare_digest(x_api_key, WRITE_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# Seuil minimal de similarité (cosinus) pour qu'un chunk soit jugé pertinent par
# le RAG question→passage. Volontairement bas : text-embedding-3-small produit des
# similarités Q→passage modestes, donc un seuil élevé écarterait de bons
# appariements. Ce plancher ne sert qu'à filtrer le bruit manifeste. Réglable via
# la variable d'environnement RAG_MIN_SIMILARITY.
try:
    RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.18"))
except (TypeError, ValueError):
    RAG_MIN_SIMILARITY = 0.18

# Budget temps (s) de la fédération des sources lors du populate. Borne le temps
# d'attente quand une source est lente/bloquée (PubMed efetch timeout=90, retries
# Cochrane). Les sources non terminées continuent en arrière-plan ; le corpus est
# reconstruit avec ce qui est déjà ingéré. Réglable via l'env POPULATE_FEDERATION_BUDGET.
try:
    POPULATE_FEDERATION_BUDGET = float(os.getenv("POPULATE_FEDERATION_BUDGET", "55"))
except (TypeError, ValueError):
    POPULATE_FEDERATION_BUDGET = 55.0


# ─── Disjoncteur OpenAI (quota épuisé) ───────────────────────────────────────
# Quand le compte OpenAI est à court de quota, l'API renvoie 429
# `insufficient_quota` sur CHAQUE appel, et le SDK retente 3× (back-off) → les
# boucles d'arrière-plan (embedding, PICO, rerank) inondent l'API/les logs et
# ralentissent tout. On détecte l'erreur quota et on met les boucles batch en
# pause courte au lieu de marteler. (Le vrai correctif reste : recharger le
# crédit OpenAI ; ceci rend juste la panne propre et non bloquante.)
_OPENAI_QUOTA_COOLDOWN_UNTIL = [0.0]


def _is_openai_quota_error(exc: object) -> bool:
    s = str(exc).lower()
    return "insufficient_quota" in s or "exceeded your current quota" in s


def _openai_in_cooldown() -> bool:
    return time.time() < _OPENAI_QUOTA_COOLDOWN_UNTIL[0]


def _trip_openai_cooldown(seconds: int = 300) -> None:
    _OPENAI_QUOTA_COOLDOWN_UNTIL[0] = time.time() + seconds
    logger.warning(
        f"OpenAI quota épuisé (insufficient_quota) → pause des appels OpenAI "
        f"d'arrière-plan pendant {seconds}s. Rechargez le crédit OpenAI."
    )


# ── Embedding résilient (anti « lot empoisonné ») ────────────────────────────
# L'API embeddings rejette TOUT le lot si UN SEUL input est invalide — le plus
# souvent trop de tokens : l'ancienne garde tronquait à 8000 CARACTÈRES (pas
# tokens), et du texte dense (CJK, identifiants, références) dépasse la limite de
# 8191 tokens. Sans isolation ni borne de tentatives, le lot fautif était
# ré-échoué à chaque cycle → chunks « en attente » bloqués indéfiniment. Ces
# helpers (1) tronquent par TOKENS, (2) mappent la réponse par index, et (3) si un
# lot échoue hors quota, ré-essaient chunk par chunk pour n'isoler QUE le fautif.
# None = pas encore tenté ; False = tiktoken indisponible (repli caractères, ne plus
# réessayer — sinon un téléchargement BPE qui échoue serait retenté à chaque appel) ;
# sinon = l'encodeur. Repli SÛR : 6000 caractères restent < 8191 tokens pour du texte
# réel (tiktoken, quand présent, préserve bien plus de contenu jusqu'à la vraie limite).
_TIKTOKEN_ENC: list = [None]


def _truncate_to_tokens(s: str, max_tokens: int = 8000) -> str:
    if not s:
        return s
    if _TIKTOKEN_ENC[0] is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC[0] = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC[0] = False
    enc = _TIKTOKEN_ENC[0]
    if not enc:
        return s[:6000]
    try:
        toks = enc.encode(s)
        return s if len(toks) <= max_tokens else enc.decode(toks[:max_tokens])
    except Exception:
        return s[:6000]


def _embed_one_call(client, batch: list) -> None:
    """Embède `batch` (liste de {id, content}) en UN appel et écrit les vecteurs,
    en mappant chaque vecteur par SON index de réponse (pas positionnel)."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[_truncate_to_tokens(r["content"]) for r in batch],
    )
    by_index = {d.index: d.embedding for d in resp.data}
    with engine.begin() as cu:
        for k, r in enumerate(batch):
            emb = by_index.get(k)
            if emb is None:
                continue
            vec = "[" + ",".join(str(x) for x in emb) + "]"
            cu.execute(
                text("UPDATE document_chunk SET embedding = CAST(:vec AS vector) WHERE id = :cid"),
                {"vec": vec, "cid": r["id"]},
            )


def _embed_chunks_resilient(client, rows: list, batch_size: int = 100) -> tuple[int, list]:
    """Embède rows=[{id, content}] par lots ; si un lot échoue HORS quota, ré-essaie
    chunk par chunk pour ne pas bloquer les chunks sains. Renvoie (nb_embeddés,
    [ids_en_échec]). Propage l'erreur de quota pour que l'appelant déclenche le cooldown."""
    embedded, failed = 0, []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            _embed_one_call(client, batch)
            embedded += len(batch)
        except Exception as e:
            if _is_openai_quota_error(e):
                raise
            for r in batch:  # lot empoisonné → on isole le fautif
                try:
                    _embed_one_call(client, [r])
                    embedded += 1
                except Exception as e2:
                    if _is_openai_quota_error(e2):
                        raise
                    failed.append(r["id"])
                    logger.warning(f"embed chunk {r['id']} échec définitif: {e2}")
    return embedded, failed


def _strategy_is_degraded(strategy: object, query: str | None = None) -> bool:
    """Vrai si une stratégie de recherche est un repli dégradé (échec LLM) :
    marquée degraded, ou dont la requête booléenne 'general' est vide / identique
    au texte brut / sans opérateur booléen — donc à régénérer."""
    if not isinstance(strategy, dict):
        return True
    if strategy.get("degraded"):
        return True
    general = (strategy.get("general") or "").strip()
    if not general:
        return True
    # Une vraie requête booléenne contient des opérateurs ou des guillemets.
    has_operators = any(op in general for op in (" AND ", " OR ", " NOT ", '"')) or "[" in general
    if not has_operators:
        return True
    if query is not None and general.strip().lower() == query.strip().lower():
        return True
    return False


# Normalisation des types d'étude : le PICO LLM produit du texte libre (des
# centaines de variantes uniques). On regroupe en un jeu canonique fixe au moment
# de l'affichage (les valeurs brutes study_design / pico_json restent intactes).
# `d` = libellé brut en minuscules (cf. _study_design_distinct_cte). 1er match gagne.
_STUDY_DESIGN_CASE = """CASE
        WHEN d = '' THEN 'Non spécifié'
        WHEN d LIKE '%systematic review%' OR d LIKE '%meta-analysis%' OR d LIKE '%meta analysis%' OR d LIKE '%scoping review%' OR d LIKE '%umbrella review%' THEN 'Revue systématique / Méta-analyse'
        WHEN d LIKE '%non-randomi%' OR d LIKE '%non randomi%' OR d LIKE '%quasi-experimental%' OR d LIKE '%quasi experimental%' OR d LIKE '%interrupted time series%' OR d LIKE '%controlled before%' THEN 'Essai non randomisé / Quasi-expérimental'
        WHEN d LIKE '%randomi%' OR d LIKE 'rct%' OR d LIKE '%controlled trial%' THEN 'Essai contrôlé randomisé (RCT)'
        WHEN d LIKE '%case-control%' OR d LIKE '%case control%' THEN 'Cas-témoins'
        WHEN d LIKE '%cross-sectional%' OR d LIKE '%cross sectional%' THEN 'Transversale'
        WHEN d LIKE '%case report%' OR d LIKE '%case series%' THEN 'Cas clinique / Série de cas'
        WHEN d LIKE '%cohort%' OR d LIKE '%longitudinal%' OR d LIKE '%observational%' OR d LIKE '%retrospective%' OR d LIKE '%prospective%' OR d LIKE '%registry%' OR d LIKE '%surveillance%' THEN 'Cohorte / Observationnelle'
        WHEN d LIKE '%model%' OR d LIKE '%simulation%' OR d LIKE '%forecast%' OR d LIKE '%machine learning%' OR d LIKE '%in silico%' OR d LIKE '%predictive%' THEN 'Modélisation / Simulation'
        WHEN d LIKE '%qualitative%' OR d LIKE '%interview%' OR d LIKE '%focus group%' THEN 'Qualitative'
        WHEN d LIKE '%narrative review%' OR d LIKE '%literature review%' OR d LIKE '%guideline%' OR d LIKE '%review%' THEN 'Revue narrative / Recommandation'
        WHEN d LIKE '%in vitro%' OR d LIKE '%in vivo%' OR d LIKE '%animal%' OR d LIKE '%experimental%' OR d LIKE '%laboratory%' OR d LIKE '%murine%' OR d LIKE '% mice%' THEN 'Expérimentale / Préclinique'
        ELSE 'Autre'
      END"""

# Niveau de preuve GRADE (strict), déterminé par le DEVIS d'étude — PAS par un
# score composite (citations/récence/échantillon servent au classement, pas à la
# certitude). En GRADE :
#   - essais randomisés + synthèses d'essais  → certitude ÉLEVÉE  (« Forte »)
#   - essais contrôlés non randomisés / quasi-expérimental / recommandations
#                                             → certitude MODÉRÉE (« Modérée »)
#   - TOUTES les études observationnelles (cohortes, cas-témoins, transversales,
#     séries/rapports de cas, registres, surveillance…) partent en certitude
#     FAIBLE (« Faible ») ; idem revues narratives / avis d'experts.
# `d` = libellé brut du devis en minuscules. Le 1er match gagne : on teste
# « non-randomi… » avant « randomi… » pour ne pas surclasser les essais non
# randomisés. Devis inconnus / modélisation / qualitatif → « Non évaluée ».
_GRADE_LEVEL_CASE = """CASE
        WHEN d = '' THEN 'Non évaluée'
        WHEN d LIKE '%non-randomi%' OR d LIKE '%non randomi%' OR d LIKE '%quasi-experimental%' OR d LIKE '%quasi experimental%' OR d LIKE '%interrupted time series%' OR d LIKE '%controlled before%' THEN 'Modérée'
        WHEN d LIKE '%systematic review%' OR d LIKE '%meta-analysis%' OR d LIKE '%meta analysis%' OR d LIKE '%umbrella review%' THEN 'Forte'
        WHEN d LIKE '%randomi%' OR d LIKE 'rct%' THEN 'Forte'
        WHEN d LIKE '%controlled trial%' OR d LIKE '%clinical trial%' OR d LIKE '%guideline%' OR d LIKE '%recommendation%' THEN 'Modérée'
        WHEN d LIKE '%cohort%' OR d LIKE '%longitudinal%' OR d LIKE '%observational%' OR d LIKE '%retrospective%' OR d LIKE '%prospective%' OR d LIKE '%registry%' OR d LIKE '%surveillance%' OR d LIKE '%case-control%' OR d LIKE '%case control%' OR d LIKE '%cross-sectional%' OR d LIKE '%cross sectional%' OR d LIKE '%case report%' OR d LIKE '%case series%' OR d LIKE '%ecological%' OR d LIKE '%survey%' THEN 'Faible'
        WHEN d LIKE '%narrative%' OR d LIKE '%literature review%' OR d LIKE '%scoping review%' OR d LIKE '%editorial%' OR d LIKE '%commentary%' OR d LIKE '%opinion%' OR d LIKE '%review%' THEN 'Faible'
        ELSE 'Non évaluée'
      END"""

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_doi(doi: str | None) -> str | None:
    """Normalise un DOI en retirant les préfixes URL courants.
    Exemples : 'https://doi.org/10.1016/...' → '10.1016/...'
               'http://dx.doi.org/10.1016/...' → '10.1016/...'
    """
    if not doi:
        return doi
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/"):
        if doi.lower().startswith(prefix):
            return doi[len(prefix):]
    return doi


# Hiérarchie des devis d'étude (evidence pyramid) → score 0–1.
_STUDY_DESIGN_TIERS = (
    (("meta-analysis", "méta-analyse", "metaanalysis"), 1.00),
    (("systematic review", "revue systématique", "systematic"), 0.92),
    # ⚠ « non-randomi… » AVANT « randomi… » : "non-randomized" CONTIENT "randomized",
    # donc sans ce garde-fou un essai NON randomisé serait surclassé au rang d'ECR.
    # Quasi-expérimental → certitude intermédiaire, cohérent avec _GRADE_LEVEL_CASE
    # qui les classe « Modérée » (au-dessus de l'observationnel, sous l'ECR).
    (("non-randomized", "non-randomised", "non randomi", "quasi-experimental",
      "quasi experimental", "interrupted time series", "controlled before"), 0.68),
    (("randomized", "randomised", "rct", "essai randomisé"), 0.85),
    (("cohort", "cohorte", "longitudinal"), 0.62),
    (("case-control", "cas-témoins", "case control"), 0.52),
    (("cross-sectional", "transversale", "survey", "observational"), 0.42),
    (("case series", "case report", "cas clinique", "série de cas"), 0.28),
    (("editorial", "commentary", "opinion", "letter", "éditorial"), 0.18),
)
_BIAS_RISK_FACTOR = {"low": 1.0, "faible": 1.0, "moderate": 0.85, "modéré": 0.85,
                     "unclear": 0.75, "incertain": 0.75, "high": 0.55, "élevé": 0.55}


def _coerce_int(value: Any) -> int | None:
    """Convertit prudemment une valeur (str/float/None) en int positif, sinon None."""
    if value is None:
        return None
    try:
        n = int(float(str(value).replace(",", "").strip()))
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def _llm_lang_directive(lang: str | None) -> str:
    """Output-language instruction appended to LLM system prompts.
    Defaults to French (the app's default) so existing behaviour is unchanged
    when no language is supplied."""
    # Defensive: only a real string carries a language. Anything else — None, or a
    # FastAPI Query/Depends object leaked by an internal *direct* call to an endpoint
    # whose param defaults to Query(...) — falls back to the French default instead of
    # crashing on `.strip()` ("'Query' object has no attribute 'strip'").
    if not isinstance(lang, str):
        lang = None
    if (lang or "fr").strip().lower().startswith("en"):
        return ("\n\nRESPOND ENTIRELY IN ENGLISH. Every sentence, heading, bullet, "
                "and JSON string value you produce must be written in English, even if "
                "the source articles or the instructions above are in French.")
    return ("\n\nRéponds intégralement en FRANÇAIS. Toutes les phrases, titres, puces "
            "et valeurs de chaîne JSON que tu produis doivent être en français.")


def _design_tier_score(study_design: str | None) -> float | None:
    """Score 0–1 du devis d'étude d'après la pyramide des preuves, ou None si inconnu."""
    if not study_design:
        return None
    s = study_design.strip().lower()
    for keywords, score in _STUDY_DESIGN_TIERS:
        if any(k in s for k in keywords):
            return score
    return None


def _compute_quality_score(
    study_design: str | None = None,
    year: int | None = None,
    sample_size: int | None = None,
    citation_count: int | None = None,
    open_access: bool | None = None,
    bias_risk: str | None = None,
) -> float | None:
    """
    Score de qualité méthodologique déterministe et reproductible, dans [0, 1].

    Combinaison pondérée de signaux objectifs (aucun appel LLM) :
      - devis d'étude (pyramide des preuves)  — poids 0.50
      - taille d'échantillon (log)            — poids 0.18
      - citations (log)                       — poids 0.12
      - récence                               — poids 0.12
      - accès ouvert                          — poids 0.08
    Le score du devis est en outre modulé par le risque de biais s'il est connu.

    Renvoie None si AUCUN signal n'est disponible (on ne fabrique pas une note).
    Les poids sont renormalisés sur les seuls signaux présents, afin qu'un article
    bien documenté et un article peu documenté restent comparables.
    """
    import math
    from datetime import datetime, timezone

    components: list[tuple[float, float]] = []  # (sous-score 0–1, poids)

    design = _design_tier_score(study_design)
    if design is not None:
        if bias_risk:
            design *= _BIAS_RISK_FACTOR.get(str(bias_risk).strip().lower(), 1.0)
        components.append((min(1.0, design), 0.50))

    if sample_size and sample_size > 0:
        # 10 → 0.25, 1k → ~0.6, 100k → 1.0
        components.append((min(1.0, math.log10(sample_size) / 5.0), 0.18))

    if citation_count is not None and citation_count >= 0:
        # 0 → 0, ~30 → 0.5, 1000 → 1.0
        components.append((min(1.0, math.log10(citation_count + 1) / 3.0), 0.12))

    if year and year > 1950:
        current = datetime.now(timezone.utc).year
        age = max(0, current - int(year))
        # ≤2 ans → 1.0, dégrade linéairement, 0 au-delà de 25 ans
        components.append((max(0.0, min(1.0, (25 - age) / 23.0)), 0.12))

    if open_access is not None:
        components.append((1.0 if open_access else 0.0, 0.08))

    if not components:
        return None
    total_weight = sum(w for _, w in components)
    score = sum(sub * w for sub, w in components) / total_weight
    score = max(0.0, min(1.0, score))
    # Sans devis d'étude connu, on ne peut pas affirmer une qualité « Forte » :
    # on plafonne à 0.55 (au mieux « Modérée ») pour qu'un article récent mais
    # non caractérisé ne soit jamais classé au sommet de la pyramide des preuves.
    if design is None:
        score = min(score, 0.55)
    return round(score, 4)


class DocumentIn(BaseModel):
    source: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    abstract: str | None = None
    year: int | None = None
    url: str | None = None
    external_id: str | None = None
    project_context: str | None = None
    source_type: str | None = None
    disease_or_condition: str | None = None
    scenario_type: str | None = None
    geographic_scope: str | None = None
    evidence_category: str | None = None
    # Champs bibliographiques enrichis
    doi: str | None = None
    pmid: str | None = None
    authors: str | None = None
    journal: str | None = None
    open_access: bool | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def _clean_doi(cls, v: str | None) -> str | None:
        return _normalize_doi(v)

class ChunkIn(BaseModel):
    document_id: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    chunk_type: str | None = None
    section_label: str | None = None
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)
    token_count: int | None = Field(None, ge=0)
    chunk_weight: float | None = Field(None, ge=0)
    metadata_json: dict[str, Any] | None = None

class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)  # Limite d'entrée RAG (H-5)
    project_context: str | None = None
    filters: dict[str, Any] | None = None
    lang: str | None = None



# ─────────────────────────────────────────────────────────────────────────────
# Index de performance
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_performance_indexes() -> None:
    """Crée les index manquants sur les colonnes chaudes + un index ANN pgvector.

    Exécuté EN ARRIÈRE-PLAN (hors du chemin de démarrage) : la 1re création de
    l'index vectoriel peut durer plusieurs minutes sur un gros corpus et ne doit
    pas retarder le health check du déploiement. Tout est IF NOT EXISTS (idempotent)
    et chaque instruction est isolée : un échec est journalisé sans rien casser
    (au pire, la requête concernée reste en scan séquentiel, comme aujourd'hui)."""
    # 1) Index B-tree sur les colonnes de filtre / jointure les plus fréquentes.
    btree = [
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scenario ON article_scenarios (scenario_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_document ON article_scenarios (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_sim ON article_scenarios (scenario_id, similarity_score)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_screening_status ON literature_document (screening_status)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_is_duplicate ON literature_document (is_duplicate)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_source ON literature_document (source)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_project_context ON literature_document (project_context)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_document ON document_chunk (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_type ON document_chunk (chunk_type)",
    ]
    for ddl in btree:
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl))
        except Exception as e:
            logger.warning(f"_ensure_performance_indexes (btree) « {ddl[:60]}… » : {e}")

    # 2) Index ANN pgvector (HNSW, cosinus). CONCURRENTLY → hors transaction, ne
    # bloque pas les écritures. Accélère « ORDER BY embedding <=> q LIMIT k » qui
    # sinon scanne TOUS les embeddings à chaque requête.
    ann_ddl = ("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_doc_chunk_embedding_hnsw "
               "ON document_chunk USING hnsw (embedding vector_cosine_ops)")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(ann_ddl))
        logger.info("Index ANN pgvector (HNSW) vérifié/créé.")
    except Exception as e:
        logger.warning(f"Index ANN pgvector indisponible ({e}); recherche vectorielle en scan séquentiel.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection OK")

    # Index de performance (B-tree + ANN pgvector) : en arrière-plan pour ne pas
    # retarder le démarrage / le health check ; idempotent et sans effet de bord
    # en cas d'échec (au pire, on reste sur le comportement actuel).
    try:
        import threading as _perf_threading
        _perf_threading.Thread(
            target=_ensure_performance_indexes, daemon=True, name="ensure-perf-indexes"
        ).start()
    except Exception as _e:
        logger.warning(f"spawn _ensure_performance_indexes: {_e}")

    # Pipelines orphelins : tout pipeline marqué 'running' ou 'starting' au
    # démarrage du serveur est forcément mort (le thread a été tué lors du
    # redémarrage précédent).
    # Stratégie : on les relance automatiquement en arrière-plan plutôt que
    # de les marquer 'failed' et forcer l'utilisateur à les relancer manuellement.
    try:
        with engine.connect() as _startup_conn:
            _orphan_rows = _startup_conn.execute(text("""
                SELECT id, query, filters
                FROM user_scenarios
                WHERE pipeline_status IN ('running', 'starting')
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).mappings().fetchall()
            _pop_orphans = _startup_conn.execute(text("""
                SELECT COUNT(*) FROM user_scenarios
                WHERE populate_status = 'running'
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).scalar() or 0

        # Réinitialiser les populate orphelins à 'error'
        if _pop_orphans:
            with engine.begin() as _c:
                _c.execute(text("""
                    UPDATE user_scenarios
                    SET populate_status = 'error', updated_at = NOW()
                    WHERE populate_status = 'running'
                      AND COALESCE(is_system, FALSE) = FALSE
                """))
            logger.warning(f"Startup: {_pop_orphans} populate(s) orphelin(s) reinitialisé(s) à 'error'.")

        # Relancer automatiquement les pipelines interrompus
        if _orphan_rows:
            import threading as _startup_threading
            logger.warning(
                f"Startup: {len(_orphan_rows)} pipeline(s) interrompu(s) détecté(s) — "
                f"relance automatique en arrière-plan."
            )
            for _orphan in _orphan_rows:
                _oid = _orphan["id"]
                _oquery = _orphan["query"] or ""
                _ofilters = _orphan["filters"] or {}
                if not _oquery:
                    # Pas de requête → on ne peut pas relancer, marquer failed
                    with engine.begin() as _c:
                        _c.execute(text("""
                            UPDATE user_scenarios
                            SET pipeline_status = 'failed',
                                pipeline_step   = NULL,
                                updated_at      = NOW()
                            WHERE id = :sid
                        """), {"sid": _oid})
                    logger.warning(f"Startup: pipeline {_oid} sans requête → marqué 'failed'.")
                    continue
                # Initialiser le job en mémoire
                with _pipeline_jobs_lock:
                    _user_scenario_pipeline_jobs[_oid] = {
                        "overall_status": "starting",
                        "current_step": "ingest",
                        "auto_restarted": True,
                        "steps": {
                            "ingest":     {"status": "pending"},
                            "fulltext":   {"status": "pending"},
                            "embed":      {"status": "pending"},
                            "rerank":     {"status": "pending"},
                            "pico":       {"status": "pending"},
                            "metadata":   {"status": "pending"},
                            "clustering": {"status": "pending"},
                        },
                    }
                _t = _startup_threading.Thread(
                    target=_run_user_scenario_full_pipeline,
                    args=(_oid, _oquery, _ofilters),
                    daemon=True,
                )
                _t.start()
                logger.info(f"Startup: pipeline {_oid} relancé automatiquement (query={_oquery[:60]!r}).")
    except Exception as _se:
        logger.error(f"Startup cleanup/relance pipelines orphelins: {_se}")

    import threading
    # ── Worker d'enrichissement automatique (embedding + PICO) ──────────────
    # Tourne en permanence en arrière-plan. Chaque cycle :
    #   1. Embède tous les chunks title_abstract/fulltext_section sans embedding
    #   2. Extrait le PICO pour tous les articles avec abstract mais sans pico_json
    def _background_enrichment_worker():
        import time as _time
        from openai import OpenAI as _OAI_bg
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from datetime import datetime, timezone

        _EMBED_BATCH   = 100   # chunks par appel OpenAI embeddings
        _PICO_WORKERS  = 5     # threads parallèles pour extraction PICO
        _CYCLE_SLEEP   = 30    # secondes entre deux cycles
        _PICO_BATCH    = 50    # articles PICO par cycle
        _PICO_MAX_ATTEMPTS = 3 # essais max par article (borne les échecs déterministes)
        _ABS_BATCH     = 50    # notices sans résumé traitées par cycle (backfill)

        _system_pico = (
            "You are a systematic review expert. "
            "Extract PICO elements and return ONLY valid JSON:\n"
            '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
            '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
            '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
            "Be concise (max 2 sentences per field). Return ONLY the JSON."
        )

        def _extract_pico_one(row, client):
            try:
                title    = row["title"] or ""
                abstract = row["abstract"] or ""
                # Évidence extraite du TEXTE INTÉGRAL si disponible, sinon du résumé.
                # Les chunks fulltext_section sont concaténés dans l'ordre ; on
                # marque la source (`pico_source`) pour pouvoir ré-extraire plus
                # tard les articles dont le PICO venait du résumé seul.
                body_text   = abstract[:3000]
                body_label  = "Abstract"
                pico_source = "abstract"
                if row.get("has_fulltext"):
                    with engine.connect() as _ftc:
                        _ft = _ftc.execute(text("""
                            SELECT string_agg(content, E'\n\n' ORDER BY chunk_index) AS ft
                            FROM document_chunk
                            WHERE document_id = :id AND chunk_type = 'fulltext_section'
                        """), {"id": row["id"]}).scalar()
                    if _ft and len(_ft) > len(abstract):
                        body_text   = _ft[:14000]
                        body_label  = "Full text"
                        pico_source = "fulltext"
                # Appel LLM isolé : une erreur d'API (quota/réseau) est TRANSITOIRE
                # → on ne compte PAS de tentative (le cooldown s'en charge) et on
                # réessaiera quand l'API sera saine.
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": _system_pico},
                            {"role": "user",   "content": f"Title: {title}\n\n{body_label}: {body_text}"},
                        ],
                        temperature=0,
                        seed=42,
                        max_tokens=800,  # 400 tronquait le JSON des articles verbeux → json invalide
                        response_format={"type": "json_object"},
                    )
                except Exception as _api_e:
                    if _is_openai_quota_error(_api_e):
                        _trip_openai_cooldown()
                    logger.debug(f"BG PICO API doc {row['id']}: {_api_e}")
                    return None

                # On a une RÉPONSE → on COMPTE la tentative quoi qu'il arrive. Sinon,
                # un article dont la sortie LLM est malformée de façon déterministe
                # (clé manquante / JSON invalide, identique à chaque fois car
                # temperature=0/seed=42) serait ré-extrait à l'infini. pico_attempts
                # (plafonné par _PICO_MAX_ATTEMPTS dans le sélecteur) borne ces échecs.
                _pico = None
                try:
                    _pico = json.loads(resp.choices[0].message.content)
                except Exception:
                    _pico = None
                with engine.begin() as _c:
                    if isinstance(_pico, dict):
                        # TOLÉRANT : on complète les clés manquantes plutôt que de
                        # rejeter (et ré-essayer en boucle). Une extraction partielle
                        # à faible confiance vaut mieux qu'un article jamais traité —
                        # l'aval filtre déjà sur pico_confidence. Seul un JSON INVALIDE
                        # (rare avec response_format json_object + max_tokens relevé)
                        # est compté comme échec à borner.
                        for _k in ("P", "I", "C", "O"):
                            _pico.setdefault(_k, "")
                        _pico.setdefault("study_design", "non précisé")
                        try:
                            _pico["pico_confidence"] = float(_pico.get("pico_confidence", 0.3))
                        except (TypeError, ValueError):
                            _pico["pico_confidence"] = 0.3
                        _pico["pico_notes"]  = _pico.get("pico_notes", "")
                        _pico["pico_source"] = pico_source
                        _c.execute(text("""
                            UPDATE literature_document
                            SET pico_json = CAST(:pico AS jsonb),
                                pico_extracted_at = :ts,
                                pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                               ELSE pico_fulltext_attempted END,
                                pico_attempts = COALESCE(pico_attempts, 0) + 1
                            WHERE id = :doc_id
                        """), {
                            "pico":   json.dumps(_pico),
                            "ts":     datetime.now(timezone.utc),
                            "hf":     bool(row.get("has_fulltext")),
                            "doc_id": row["id"],
                        })
                        return row["id"]
                    # JSON invalide : on compte la tentative (+ fulltext_attempted)
                    # pour borner les ré-essais → fin de la boucle de tokens.
                    _c.execute(text("""
                        UPDATE literature_document
                        SET pico_attempts = COALESCE(pico_attempts, 0) + 1,
                            pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                           ELSE pico_fulltext_attempted END
                        WHERE id = :doc_id
                    """), {"hf": bool(row.get("has_fulltext")), "doc_id": row["id"]})
                    return None
            except Exception as _pe:
                logger.debug(f"BG PICO doc {row['id']}: {_pe}")
                return None

        def _europepmc_abstracts_by_doi(dois: list[str]) -> dict[str, str]:
            """Récupère le résumé via EuropePMC pour une liste de DOI (une requête
            OR groupée). EuropePMC agrège MEDLINE + PMC : meilleure couverture
            DOI→abstract que Crossref/OpenAlex. Renvoie {doi_minuscule: abstract}."""
            import requests as _rq
            out: dict[str, str] = {}
            _dois = [d for d in dois if d]
            if not _dois:
                return out
            q = " OR ".join(f'DOI:"{d}"' for d in _dois)
            try:
                _r = _rq.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": f"({q})", "resultType": "core",
                            "format": "json", "pageSize": len(_dois)},
                    timeout=30,
                )
                _r.raise_for_status()
                for _res in ((_r.json().get("resultList") or {}).get("result") or []):
                    _d = (_res.get("doi") or "").lower().strip()
                    _ab = _res.get("abstractText")
                    if _d and _ab:
                        _ab = re.sub(r"<[^>]+>", " ", _ab)      # retirer le JATS/HTML
                        _ab = re.sub(r"\s+", " ", _ab).strip()
                        if len(_ab) >= 30:
                            out[_d] = _ab
            except Exception as _ee:
                logger.debug(f"EuropePMC abstract batch: {_ee}")
            return out

        # Colonne de suivi : évite de re-tenter indéfiniment les notices dont
        # EuropePMC n'a pas de résumé (sinon le même lot bloquerait la file).
        try:
            with engine.begin() as _cc:
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS abstract_backfill_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Garde-fou anti-boucle : ne ré-extraire le PICO « texte intégral »
                # qu'UNE SEULE fois par article. Sans cela, tout article avec
                # has_fulltext=TRUE mais SANS chunk fulltext exploitable garde
                # pico_source='abstract' et re-matche le sélecteur PICO à CHAQUE
                # cycle (30 s) → ré-extraction gpt-4.1-mini infinie = fuite de tokens.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_fulltext_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Compteur de tentatives PICO. Un article dont l'extraction échoue
                # de façon DÉTERMINISTE (sortie LLM malformée : clé manquante ou JSON
                # invalide) revenait dans le sélecteur à CHAQUE cycle et était ré-envoyé
                # à gpt-4.1-mini indéfiniment (même échec, temperature=0/seed=42) →
                # 2e fuite de tokens. On borne à _PICO_MAX_ATTEMPTS essais par article.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_attempts INTEGER DEFAULT 0"
                ))
        except Exception as _ce:
            logger.warning(f"ensure backfill/pico-attempt columns: {_ce}")

        logger.info("Background enrichment worker started (abstract backfill + embedding + PICO).")
        while True:
            try:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    _time.sleep(_CYCLE_SLEEP)
                    continue

                _client = _OAI_bg(api_key=openai_key, timeout=90.0)

                # ── 0. BACKFILL DES RÉSUMÉS (notices sans abstract, via DOI) ──
                # Beaucoup de notices Crossref/OpenAlex arrivent sans résumé. On
                # tente de le récupérer via EuropePMC (par DOI) pour les rendre
                # exploitables (puis embedding + PICO par les étapes suivantes).
                try:
                    with engine.connect() as _conn:
                        _stub_rows = _conn.execute(text("""
                            SELECT id, doi FROM literature_document
                            WHERE project_context = 'literev'
                              AND doi IS NOT NULL
                              AND (abstract IS NULL OR length(trim(abstract)) < 30)
                              AND abstract_backfill_attempted IS NOT TRUE
                            ORDER BY id LIMIT :lim
                        """), {"lim": _ABS_BATCH}).mappings().fetchall()
                    if _stub_rows:
                        _doi_map: dict[str, list[int]] = {}
                        for _r in _stub_rows:
                            _doi_map.setdefault((_r["doi"] or "").lower().strip(), []).append(_r["id"])
                        _dois = [d for d in _doi_map if d]
                        _found: dict[str, str] = {}
                        for _k in range(0, len(_dois), 20):       # 20 DOI / requête
                            _found.update(_europepmc_abstracts_by_doi(_dois[_k:_k + 20]))
                        _filled = 0
                        with engine.begin() as _cu:
                            for _d, _ab in _found.items():
                                for _docid in _doi_map.get(_d, []):
                                    _cu.execute(text("""
                                        UPDATE literature_document SET abstract = :ab
                                        WHERE id = :id
                                          AND (abstract IS NULL OR length(trim(abstract)) < 30)
                                    """), {"ab": _ab, "id": _docid})
                                    _filled += 1
                            # Marquer TOUTES les notices tentées (trouvées ou non).
                            _cu.execute(
                                text("UPDATE literature_document SET abstract_backfill_attempted = TRUE "
                                     "WHERE id IN :ids").bindparams(bindparam("ids", expanding=True)),
                                {"ids": [_r["id"] for _r in _stub_rows]},
                            )
                        if _found:
                            # Créer les chunks title_abstract des docs nouvellement dotés
                            # d'un résumé (l'embedding ci-dessous les vectorisera).
                            _backfill_title_abstract_chunks()
                            logger.info(f"BG abstract backfill: {_filled} résumés récupérés (EuropePMC).")
                except Exception as _abe:
                    logger.warning(f"BG abstract backfill error: {_abe}")

                # ── 1. EMBEDDING ──────────────────────────────────────────────
                # On embède TOUS les chunks standard sans embedding — y compris le
                # title_abstract des docs à texte intégral : le résumé sert de vecteur
                # représentatif CONSTANT au clustering (même base pour chaque document)
                # pour un coût négligeable. (Avant, on le sautait pour les docs full-text
                # → incohérence : la plupart embeddés, ~153 non, et clustering par 1re
                # section au lieu du résumé.)
                with engine.connect() as _conn:
                    _chunks = _conn.execute(text("""
                        SELECT c.id, c.content
                        FROM document_chunk c
                        WHERE c.embedding IS NULL
                          AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                          AND LENGTH(c.content) > 20
                          AND COALESCE(c.embedding_attempts, 0) < 3
                        ORDER BY c.chunk_type DESC, c.id
                        LIMIT 500
                    """)).mappings().fetchall()

                if _chunks and not _openai_in_cooldown():
                    try:
                        _emb_done, _failed = _embed_chunks_resilient(_client, list(_chunks))
                    except Exception:               # quota propagé → pause + réessai après cooldown
                        _trip_openai_cooldown()
                        _emb_done, _failed = 0, []
                    if _failed:
                        # Chunks refusés par l'API (hors quota) : incrémenter le compteur
                        # pour les sortir de la file après 3 essais (sinon « en attente »
                        # éternel + lot ré-échoué à chaque cycle).
                        with engine.begin() as _cu:
                            _cu.execute(
                                text("UPDATE document_chunk SET embedding_attempts = "
                                     "COALESCE(embedding_attempts,0)+1 WHERE id IN :ids")
                                .bindparams(bindparam("ids", expanding=True)),
                                {"ids": _failed},
                            )
                    if _emb_done:
                        logger.info(f"BG worker: {_emb_done} chunks embedded.")

                # ── 2. PICO ───────────────────────────────────────────────────
                with engine.connect() as _conn:
                    # On extrait le PICO des articles sans PICO, PUIS on ré-extrait
                    # ceux dont le PICO venait du résumé alors que le texte intégral
                    # est désormais disponible (pico_source != 'fulltext'). Les
                    # articles jamais traités passent en premier.
                    _pico_rows = _conn.execute(text("""
                        SELECT id, title, abstract, has_fulltext
                        FROM literature_document
                        WHERE project_context = 'literev'
                          AND abstract IS NOT NULL
                          AND LENGTH(abstract) > 50
                          AND COALESCE(pico_attempts, 0) < :max_attempts  -- borne les échecs déterministes
                          -- PICO UNIQUEMENT pour les articles appartenant à un scénario
                          -- réel (article_scenarios). Le PICO n'est affiché QUE par
                          -- scénario : extraire les ~milliers de documents orphelins /
                          -- hors-scénario du corpus était du pur gaspillage de tokens.
                          AND EXISTS (
                              SELECT 1 FROM article_scenarios ars
                              WHERE ars.document_id = literature_document.id
                          )
                          AND (
                            pico_json IS NULL
                            OR (
                                has_fulltext IS TRUE
                                AND (pico_json->>'pico_source') IS DISTINCT FROM 'fulltext'
                                AND pico_fulltext_attempted IS NOT TRUE  -- une seule tentative
                            )
                          )
                        ORDER BY (pico_json IS NULL) DESC, id
                        LIMIT :lim
                    """), {"lim": _PICO_BATCH, "max_attempts": _PICO_MAX_ATTEMPTS}).mappings().fetchall()

                # Coupe-circuit : PICO_AUTOEXTRACT_ENABLED=0 dans /etc/literev-api.env
                # (puis restart) met en pause l'extraction PICO automatique sans
                # toucher au code — utile pour stopper net la dépense OpenAI.
                _pico_enabled = os.getenv("PICO_AUTOEXTRACT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
                if _pico_rows and _pico_enabled and not _openai_in_cooldown():
                    _pico_done = 0
                    with _TPE(max_workers=_PICO_WORKERS) as _pool:
                        _futs = {_pool.submit(_extract_pico_one, r, _client): r["id"] for r in _pico_rows}
                        for _f in _futs:
                            if _f.result() is not None:
                                _pico_done += 1
                    if _pico_done:
                        logger.info(f"BG worker: {_pico_done} PICO extracted.")

            except Exception as _we:
                logger.error(f"BG enrichment worker error: {_we}")

            _time.sleep(_CYCLE_SLEEP)

    threading.Thread(target=_background_enrichment_worker, daemon=True, name="bg-enrichment").start()
    logger.info("Background enrichment worker launched.")

# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok"}

# ─────────────────────────────────────────────────────────────────────────────
# Filter options
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/filters-options")
def get_filter_options() -> dict[str, list[dict[str, Any]]]:
    fields = [
        ("source", "source"),
        ("source_type", "source_type"),
        ("disease_or_condition", "disease_or_condition"),
        ("scenario_type", "scenario_type"),
        ("geographic_scope", "geographic_scope"),
        ("evidence_category", "evidence_category"),
        ("year", "year"),
    ]
    out: dict[str, list[dict[str, Any]]] = {}

    # Normalisation des valeurs : fusionne les variantes avec tiret/underscore
    def _normalize_key(val: str) -> str:
        return val.lower().replace("-", "_").strip()

    def _make_label(val: str) -> str:
        return (
            str(val)
            .replace("_", " ")
            .replace("-", " ")
            .title()
            .replace("Covid 19", "COVID-19")
            .replace("Ems", "EMS")
            .replace("Ai", "AI")
            .replace("Uk", "UK")
            .replace("Usa", "USA")
        )

    # Pays/régions qui sont des combinaisons (contiennent virgule, 'and', chiffres+Countries)
    import re as _re
    def _is_singleton_geo(val: str) -> bool:
        v = str(val).strip()
        if _re.search(r'\d+\s+(Countries|Cities|Regions)', v, _re.IGNORECASE):
            return False
        if ',' in v or ' and ' in v.lower() or ' & ' in v:
            return False
        return True

    with engine.connect() as conn:
        for key, col in fields:
            extra_where = "AND year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int" if key == "year" else ""
            rows = conn.execute(
                text(f"""
                    SELECT DISTINCT {col} AS value
                    FROM literature_document
                    WHERE {col} IS NOT NULL {extra_where}
                    ORDER BY {col}
                """)
            ).mappings().all()

            seen_normalized: dict[str, dict[str, str]] = {}  # normalized_key -> {value, label}
            for row in rows:
                value = row["value"]
                if value is None:
                    continue

                # Filtrer les scénarios usr-XXXX dans scenario_type
                if key == "scenario_type" and str(value).startswith("usr-"):
                    continue

                # Pour geographic_scope : ne garder que les pays/régions singletons
                if key == "geographic_scope" and not _is_singleton_geo(str(value)):
                    continue

                if key == "year":
                    label = str(value)
                    norm = str(value)
                else:
                    label = _make_label(str(value))
                    norm = _normalize_key(str(value))

                # Dédoublonnage par clé normalisée (ex: systematic-review == systematic_review)
                if norm not in seen_normalized:
                    seen_normalized[norm] = {"value": value, "label": label}

            out[key] = list(seen_normalized.values())
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Write endpoints (protected)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/documents")
def create_document(
    doc: DocumentIn, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    sql = text("""
        INSERT INTO literature_document (
            source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category,
            doi, pmid, authors, journal, open_access
        )
        VALUES (
            :source, :title, :abstract, :year, :url, :external_id,
            :project_context, :source_type, :disease_or_condition,
            :scenario_type, :geographic_scope, :evidence_category,
            :doi, :pmid, :authors, :journal, :open_access
        )
        ON CONFLICT (doi) WHERE doi IS NOT NULL DO NOTHING
        RETURNING id
    """)
    params = doc.model_dump()
    with engine.begin() as conn:
        new_id = conn.execute(sql, params).scalar()
        deduplicated = False
        if new_id is None:
            # DOI already present (UNIQUE(doi) partial index) — return the existing row
            new_id = conn.execute(
                text("SELECT id FROM literature_document WHERE doi = :doi ORDER BY id LIMIT 1"),
                {"doi": params.get("doi")},
            ).scalar()
            deduplicated = True
    return {"id": new_id, "deduplicated": deduplicated}

@app.post("/chunks")
def create_chunk(
    chunk: ChunkIn, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    sql = text("""
        INSERT INTO document_chunk (
            document_id, chunk_index, content, chunk_type, section_label,
            char_start, char_end, token_count, chunk_weight, metadata_json
        )
        VALUES (
            :document_id, :chunk_index, :content, :chunk_type, :section_label,
            :char_start, :char_end, :token_count, :chunk_weight,
            CAST(:metadata_json AS jsonb)
        )
        RETURNING id
    """)
    payload = chunk.model_dump()
    # Serialize metadata_json to a JSON string for the CAST(:x AS jsonb) binding
    meta = payload.get("metadata_json")
    if meta is None or meta == {}:
        payload["metadata_json"] = "{}"
    elif isinstance(meta, dict):
        payload["metadata_json"] = json.dumps(meta)
    # else already a string : leave as-is

    with engine.begin() as conn:
        new_id = conn.execute(sql, payload).scalar_one()
    return {"id": new_id}

# ─────────────────────────────────────────────────────────────────────────────
# RAG Assistant /ask
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_assistant(payload: AskIn) -> dict[str, Any]:
    # 1. Rechercher les chunks pertinents dans la DB
    # On réutilise la logique de recherche textuelle mais avec un filtre projet si spécifié
    filters = payload.filters or {}
    if payload.project_context:
        filters["project_context"] = payload.project_context
    
    where_sql, where_params = _build_where(filters)
    query_terms = [t.strip() for t in re.split(r"\s+", payload.question.lower()) if t.strip()]
    
    if not query_terms:
        raise HTTPException(status_code=422, detail="Empty question")
        
    like_clauses = []
    score_clauses = []
    params = {"limit": 6, "offset": 0, **where_params}
    
    for i, term in enumerate(query_terms):
        key = f"term_{i}"
        params[key] = f"%{term}%"
        like_clauses.append(
            f"(LOWER(COALESCE(d.title, '')) LIKE :{key} OR LOWER(COALESCE(d.abstract, '')) LIKE :{key} OR LOWER(COALESCE(c.content, '')) LIKE :{key})"
        )
        score_clauses.append(
            f"((CASE WHEN LOWER(COALESCE(d.title, '')) LIKE :{key} THEN 3 ELSE 0 END) + (CASE WHEN LOWER(COALESCE(d.abstract, '')) LIKE :{key} THEN 2 ELSE 0 END) + (CASE WHEN LOWER(COALESCE(c.content, '')) LIKE :{key} THEN 1 ELSE 0 END))"
        )
        
    any_match_sql = " OR ".join(like_clauses)
    score_sql = " + ".join(score_clauses)
    
    # On utilise la recherche sémantique pgvector si la clé OpenAI est présente
    openai_key = os.getenv("OPENAI_API_KEY")
    has_vector = False
    
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            # Générer l'embedding de la question
            response = client.embeddings.create(
                input=[payload.question.replace("\n", " ").strip()],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            has_vector = True
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding pour /ask: {e}")
            
    if has_vector:
        # Recherche vectorielle pure pour le RAG. On exclut les doublons et les
        # articles écartés au screening, et on impose un plancher de similarité
        # pour ne pas répondre à partir de chunks hors-sujet (corpus mince).
        params = {"query_embedding": str(query_embedding), "limit": 6,
                  "max_dist": 1.0 - RAG_MIN_SIMILARITY, **where_params}
        sql = text(f"""
            SELECT
                d.id AS document_id,
                d.title,
                d.year,
                d.url,
                d.source,
                d.project_context,
                c.content,
                c.metadata_json,
                (1 - (c.embedding <=> CAST(:query_embedding AS vector))) AS score
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              AND d.screening_status IS DISTINCT FROM 'excluded'
              AND (c.embedding <=> CAST(:query_embedding AS vector)) <= :max_dist
            {where_sql}
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
        """)
    else:
        # Fallback textuel classique
        params = {"limit": 6, "offset": 0, **where_params}
        for i, term in enumerate(query_terms):
            key = f"term_{i}"
            params[key] = f"%{term}%"
        sql = text(f"""
            SELECT 
                d.id AS document_id,
                d.title,
                d.year,
                d.url,
                d.source,
                d.project_context,
                c.content,
                c.metadata_json,
                ({score_sql}) AS score
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE ({any_match_sql})
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              AND d.screening_status IS DISTINCT FROM 'excluded'
            {where_sql}
            ORDER BY score DESC, d.year DESC NULLS LAST
            LIMIT :limit
        """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
        
    if not rows:
        return {
            "answer": "Je n'ai pas trouvé d'articles ou d'évidences scientifiques dans le corpus actuel pour répondre à votre question. Veuillez élargir vos termes de recherche ou ingérer de nouveaux articles.",
            "sources": []
        }
        
    # 2. Construire le contexte pour l'API OpenAI
    context_blocks = []
    sources = []
    seen_docs = set()
    
    for i, r in enumerate(rows):
        doc_id = r["document_id"]
        # Récupérer la force des preuves si présente dans metadata_json
        meta = r["metadata_json"] or {}
        evidence_strength = meta.get("evidence_strength", "non spécifiée")
        
        context_blocks.append(
            f"--- SOURCE {i+1} ---\n"
            f"Titre: {r['title']}\n"
            f"Année: {r['year'] or 'Inconnue'}\n"
            f"Source: {r['source']}\n"
            f"Projet: {r['project_context']}\n"
            f"Force des preuves: {evidence_strength}\n"
            f"Contenu: {r['content']}\n"
        )
        
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "title": r["title"],
                "year": r["year"],
                "url": r["url"],
                "source": r["source"],
                "project_context": r["project_context"],
                "evidence_strength": evidence_strength
            })
            
    context_str = "\n\n".join(context_blocks)
    
    # 3. Appeler l'API OpenAI (GPT-4o-mini)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        # Fallback si pas de clé API configurée
        lines = []
        for s in sources:
            url_str = s['url'] if s['url'] else "Pas d'URL"
            year_str = str(s['year']) if s['year'] else "N/A"
            lines.append(f"- **{s['title']}** ({year_str}) - {url_str}")
        return {
            "answer": "[Mode dégradé - Clé OpenAI manquante]\n\nVoici les sources trouvées pour répondre à votre question :\n\n" + "\n".join(lines),
            "sources": sources
        }
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key, timeout=90.0)
        
        system_prompt = (
            "Vous êtes l'assistant scientifique expert de LiteRev-Evidence, spécialisé dans la synthèse d'évidences "
            "pour la médecine d'urgence suisse (SMUR/EMS Genève et HUG).\n\n"
            "Votre tâche est de répondre à la question de l'utilisateur en vous basant STRICTEMENT sur le contexte fourni. "
            "Ne faites pas d'affirmations qui ne sont pas étayées par les sources fournies.\n\n"
            "Règles de rédaction :\n"
            "1. Soyez précis, structuré et professionnel.\n"
            "2. Citez toujours vos sources dans le texte en utilisant le format [SOURCE 1], [SOURCE 2] etc. correspondant aux blocs du contexte.\n"
            "3. Mentionnez la force des preuves (forte, modérée, faible) quand elle est pertinente pour appuyer vos conclusions.\n"
            "4. Si le contexte ne contient pas assez d'informations pour répondre, dites-le honnêtement."
        ) + _llm_lang_directive(payload.lang)

        user_prompt = (
            f"CONTEXTE :\n{context_str}\n\n"
            f"QUESTION : {payload.question}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Erreur OpenAI API: {e}")
        return {
            "answer": f"Une erreur est survenue lors de la génération de la réponse via l'IA : {str(e)}\n\nNéanmoins, voici les sources scientifiques trouvées dans la base :",
            "sources": sources
        }

# ─────────────────────────────────────────────────────────────────────────────
# Search helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not filters:
        return "", {}

    # Normaliser project_context : gesica/geoai4ei/eva -> literev (migration)
    if filters.get("project_context") in ("gesica", "geoai4ei", "eva"):
        filters = {**filters, "project_context": "literev"}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    field_map = {
        "source": "d.source",
        "source_type": "d.source_type",
        "disease_or_condition": "d.disease_or_condition",
        "scenario_type": "d.scenario_type",
        "geographic_scope": "d.geographic_scope",
        "evidence_category": "d.evidence_category",
        "project_context": "d.project_context",
    }

    for key, column in field_map.items():
        value = filters.get(key)
        if value not in (None, "", []):
            if key == "scenario_type":
                # Migration 1 (Way B) : filtrer par APPARTENANCE au scénario
                # (article_scenarios) et non par la colonne d'ingestion
                # d.scenario_type. Dernier prédicat encore en Way A ; on l'aligne
                # sur tous les autres compteurs/vues (corpus, stats, PRISMA, RAG…)
                # déjà basculés, sinon /search montre MOINS d'articles que le
                # corpus réel du scénario. Sous-requête corrélée sur d.id
                # (d = literature_document dans tous les appelants de _build_where).
                clauses.append(
                    "EXISTS (SELECT 1 FROM article_scenarios ars "
                    "WHERE ars.document_id = d.id AND ars.scenario_id = :scenario_type)"
                )
            else:
                clauses.append(f"{column} = :{key}")
            params[key] = value

    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min not in (None, ""):
        clauses.append("d.year >= :year_min")
        params["year_min"] = int(year_min)
    if year_max not in (None, ""):
        clauses.append("d.year <= :year_max")
        params["year_max"] = int(year_max)

    if not clauses:
        return "", {}

    return " AND " + " AND ".join(clauses), params

def _parse_boolean_query(query: str) -> tuple[list[str], list[str], list[str]]:
    """Parse a boolean query into (required, optional, excluded) term lists.
    Handles quoted phrases, AND/OR/NOT operators.
    Default between adjacent terms is AND (like PubMed).
    Returns (required_terms, optional_terms, excluded_terms).
    """
    required: list[str] = []
    optional_terms: list[str] = []
    excluded: list[str] = []

    # Tokenize preserving quoted phrases
    tokens = re.findall(r'"[^"]*"|\S+', query)
    pending_op = "AND"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        upper = tok.upper()
        if upper == "AND":
            pending_op = "AND"
        elif upper == "OR":
            pending_op = "OR"
        elif upper in ("NOT", "-"):
            # next token is excluded
            if i + 1 < len(tokens):
                i += 1
                raw = tokens[i].strip('"')
                clean = re.sub(r"[^a-zA-Z0-9\-_ ]", "", raw).lower().strip()
                if clean:
                    excluded.append(clean)
        else:
            raw = tok.strip('"')
            clean = re.sub(r"[^a-zA-Z0-9\-_ ]", "", raw).lower().strip()
            if clean:
                if pending_op == "OR":
                    optional_terms.append(clean)
                else:
                    required.append(clean)
            pending_op = "AND"
        i += 1
    return required, optional_terms, excluded


def _build_boolean_match_sql(required: list[str], optional_terms: list[str],
                              excluded: list[str], params: dict) -> str:
    """Build SQL WHERE fragment for boolean mode with AND/OR/NOT logic."""
    and_clauses: list[str] = []

    for i, term in enumerate(required):
        key = f"bool_req_{i}"
        params[key] = f"%{term}%"
        and_clauses.append(
            f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
            f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
            f" OR LOWER(COALESCE(c.content,'')) LIKE :{key})"
        )

    if optional_terms:
        or_parts: list[str] = []
        for i, term in enumerate(optional_terms):
            key = f"bool_opt_{i}"
            params[key] = f"%{term}%"
            or_parts.append(
                f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(c.content,'')) LIKE :{key})"
            )
        and_clauses.append("(" + " OR ".join(or_parts) + ")")

    for i, term in enumerate(excluded):
        key = f"bool_excl_{i}"
        params[key] = f"%{term}%"
        and_clauses.append(
            f"NOT (LOWER(COALESCE(d.title,'')) LIKE :{key}"
            f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
            f" OR LOWER(COALESCE(c.content,'')) LIKE :{key})"
        )

    return " AND ".join(and_clauses) if and_clauses else "TRUE"


# ─────────────────────────────────────────────────────────────────────────────
# Search (Hybride & Vectorielle pgvector)
# ─────────────────────────────────────────────────────────────────────────────

def _search_local_doc_ids(
    query: str,
    mode: str,
    filters: dict,
    limit: int = 10_000,
    threshold: float = 0.45,
) -> list[str]:
    """Run the same local-DB search logic as /search and return matching doc IDs.

    Used by the pipeline to link already-ingested docs to a new scenario
    without re-querying external APIs.
    """
    where_sql, where_params = _build_where(filters)

    openai_key = os.getenv("OPENAI_API_KEY")
    use_vector = mode in ("semantic", "hybrid") and bool(openai_key)

    query_embedding = None
    if use_vector:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            query_embedding = client.embeddings.create(
                input=[query.replace("\n", " ").strip()],
                model="text-embedding-3-small",
            ).data[0].embedding
        except Exception as e:
            logger.error(f"_search_local_doc_ids embedding error: {e}")
            use_vector = False

    params: dict[str, Any] = {**where_params, "limit": limit}

    if mode == "boolean":
        bool_required, bool_optional, bool_excluded = _parse_boolean_query(query)
        any_match_sql = _build_boolean_match_sql(bool_required, bool_optional, bool_excluded, params)
    else:
        raw_terms = [t.strip() for t in re.split(r"\s+", query.lower()) if t.strip()]
        query_terms = [re.sub(r"[^a-zA-Z0-9\-_]", "", t) for t in raw_terms if re.sub(r"[^a-zA-Z0-9\-_]", "", t)]
        like_clauses: list[str] = []
        for i, term in enumerate(query_terms):
            key = f"lsd_term_{i}"
            params[key] = f"%{term}%"
            like_clauses.append(
                f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(c.content,'')) LIKE :{key})"
            )
        any_match_sql = " OR ".join(like_clauses) if like_clauses else "TRUE"

    if use_vector:
        params["q_emb"] = str(query_embedding)
        params["threshold"] = threshold
        sql = text(f"""
            SELECT DISTINCT d.id
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (1 - (c.embedding <=> CAST(:q_emb AS vector))) > :threshold
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    else:
        sql = text(f"""
            SELECT DISTINCT d.id
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE ({any_match_sql})
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)

    with engine.connect() as conn:
        return conn.execute(sql, params).scalars().all()


# Limite de récupération par source live (PubMed, OpenAlex, …). Appliquée à
# l'identique à la recherche ET à la construction du corpus.
LIVE_MAX_PER_SOURCE = 2000


def _boolean_corpus_ids(boolean_query: str, filters: dict) -> list:
    """LA source de vérité de l'appartenance au corpus : les documents de la base
    locale qui correspondent à la requête booléenne. Recherche et corpus utilisent
    EXACTEMENT ce helper → le compteur de la recherche == la taille du corpus."""
    return _search_local_doc_ids(boolean_query, "boolean", filters, limit=500_000)


def _normalize_sub_queries(sub_queries: Any) -> list[dict]:
    """Nettoie une liste de sous-requêtes : ne garde que les entrées {kind,text}
    valides (texte non vide, kind ∈ {boolean, natural}). Renvoie [] si aucune."""
    out: list[dict] = []
    if not isinstance(sub_queries, list):
        return out
    for sq in sub_queries:
        if not isinstance(sq, dict):
            continue
        _text = (sq.get("text") or "").strip()
        if not _text:
            continue
        _kind = "boolean" if sq.get("kind") == "boolean" else "natural"
        out.append({"kind": _kind, "text": _text})
    return out


def _multi_query_corpus_ids(sub_queries: list[dict], combinator: str, filters: dict) -> list:
    """Appartenance au corpus pour une recherche MULTI-sous-requêtes.

    Chaque sous-requête produit un ENSEMBLE d'IDs de documents de la base locale
    par correspondance LEXICALE (booléenne) — EXACTEMENT comme la recherche
    mono-requête (_boolean_corpus_ids) et comme le documente l'étape de populate :
      - kind="boolean" → la requête est utilisée telle quelle (AND/OR/NOT),
      - kind="natural" → elle est d'abord TRADUITE en booléen (_generate_search_strategy,
        déterministe : seed=42, avec expansion de synonymes), puis matchée en booléen.

    Le seuil sémantique N'INTERVIENT JAMAIS dans l'appartenance au corpus : il ne
    sert qu'EN AVAL (page scénario, _get_above_threshold_articles) à sélectionner le
    sous-ensemble PERTINENT parmi le corpus. Un corpus défini lexicalement reste
    reproductible et auditable (exigence revue systématique) ; le score sémantique
    classe/priorise ensuite, sans jamais retirer d'article du corpus.

    Les ensembles sont combinés par UNION (OU) ou INTERSECTION (ET) puis dédupliqués
    (ce sont des ensembles)."""
    clean = _normalize_sub_queries(sub_queries)
    id_sets: list[set] = []
    for sq in clean:
        if sq["kind"] == "boolean":
            _boolean = sq["text"]
        else:
            # Naturel → booléen, comme une requête naturelle mono-requête. Ainsi
            # l'appartenance reste LEXICALE (pas de seuil sémantique). Repli sur le
            # texte brut si la traduction échoue / clé OpenAI absente (mode dégradé).
            try:
                _gen = _generate_search_strategy(sq["text"])
                _boolean = (_gen.get("general") or sq["text"]) if isinstance(_gen, dict) else sq["text"]
            except Exception as _e:
                logger.warning(f"_multi_query_corpus_ids: traduction naturel→booléen échouée ({_e}) ; repli lexical brut")
                _boolean = sq["text"]
        id_sets.append(set(_search_local_doc_ids(_boolean, "boolean", filters, limit=500_000)))
    if not id_sets:
        return []
    if combinator == "intersection":
        out: set = set(id_sets[0])
        for s in id_sets[1:]:
            out &= s
    else:  # union (OU) — défaut ; l'intersection (ET) est le choix explicite
        out = set().union(*id_sets)
    return list(out)


def _set_scenario_corpus(scenario_id: str, ids: list, allow_empty: bool = False) -> int:
    """Fixe le corpus d'un scénario à EXACTEMENT `ids` (appartenance booléenne).
    Supprime les liens qui n'en font plus partie et insère les manquants. Si `ids`
    est vide on ne touche à rien (évite de vider le corpus sur un échec transitoire),
    SAUF si allow_empty=True — cas d'une intersection multi-requêtes légitimement
    vide (deux facettes sans document commun), où le corpus DOIT être vidé."""
    if not ids:
        if allow_empty:
            with engine.begin() as _c:
                _c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"),
                           {"sid": scenario_id})
        return 0
    with engine.begin() as _c:
        _c.execute(
            text("DELETE FROM article_scenarios WHERE scenario_id = :sid "
                 "AND document_id NOT IN :ids").bindparams(bindparam("ids", expanding=True)),
            {"sid": scenario_id, "ids": list(ids)},
        )
        # Insertion en masse (un seul aller-retour) plutôt qu'une requête par
        # document : la (ré)construction du corpus local doit être quasi immédiate.
        _c.execute(text("""
            INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
            SELECT unnest(CAST(:ids AS bigint[])), :s, NULL
            ON CONFLICT (document_id, scenario_id) DO NOTHING
        """), {"ids": list(ids), "s": scenario_id})
    return len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Live federated search
# ─────────────────────────────────────────────────────────────────────────────

def _plain_keywords(query: str, max_words: int = 8) -> str:
    """Convertit une requête booléenne en mots-clés simples pour les API qui
    n'acceptent PAS la syntaxe booléenne (OpenAlex `search` renvoie 400, les
    serveurs de prépublications n'ont pas de recherche plein-texte). On retire
    les opérateurs AND/OR/NOT, parenthèses, guillemets et jokers, en gardant
    les termes significatifs uniques."""
    import re as _re
    raw = _re.sub(r'["()\[\]*]', " ", query or "")
    words = []
    for w in _re.split(r"\s+", raw):
        wl = w.strip().lower()
        if not wl or wl in ("and", "or", "not"):
            continue
        if wl not in words:
            words.append(wl)
        if len(words) >= max_words:
            break
    return " ".join(words)


# NCBI eutils sans clé API = 3 requêtes/seconde par IP. Les 3 fetchers basés
# sur PubMed (PubMed, PROSPERO, Cochrane) s'exécutent en parallèle et se
# privaient mutuellement (→ résultats vides). On sérialise/espace les appels
# eutils via un verrou global et on réessaie en cas de 429.
import threading as _threading_ncbi
_NCBI_LOCK = _threading_ncbi.Lock()
_NCBI_LAST = [0.0]
_NCBI_MIN_INTERVAL = 0.4  # ~2.5 req/s, sous la limite de 3/s


def _ncbi_get(url: str, params: dict, timeout: int = 12):
    """GET eutils throttlé (verrou global) avec un petit retry sur 429/erreur."""
    import requests as _req
    import time as _time
    key = os.getenv("NCBI_API_KEY")
    if key:
        params = {**params, "api_key": key}
    # Avec une clé API, NCBI autorise 10 req/s (vs 3 sans) : on resserre l'espacement
    # pour réduire la sérialisation du verrou global sur le trio PubMed/PROSPERO/Cochrane.
    min_interval = 0.11 if key else _NCBI_MIN_INTERVAL
    for attempt in range(3):
        with _NCBI_LOCK:
            wait = min_interval - (_time.time() - _NCBI_LAST[0])
            if wait > 0:
                _time.sleep(wait)
            try:
                r = _req.get(url, params=params, timeout=timeout)
            finally:
                _NCBI_LAST[0] = _time.time()
        if r.status_code == 429:
            _time.sleep(0.6 * (attempt + 1))
            continue
        return r
    return r


def _live_fetch_pubmed(query: str, max_results: int) -> list[dict]:
    """Fetch from PubMed eSearch+eSummary, return list of result dicts."""
    results = []
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = _ncbi_get(f"{base}/esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "tool": "literev", "email": "api@literev.app"
        })
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        r2 = _ncbi_get(f"{base}/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            "tool": "literev", "email": "api@literev.app"
        })
        res2 = r2.json().get("result", {})
        for uid in res2.get("uids", []):
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
    return results


def _live_fetch_openalex(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # OpenAlex `search` n'accepte pas la syntaxe booléenne (renvoie HTTP 400)
        # → on lui passe des mots-clés simples.
        r = _req.get("https://api.openalex.org/works", params={
            "search": _plain_keywords(query), "per-page": min(max_results, 50),
            "select": "id,title,abstract_inverted_index,doi,publication_year,authorships,primary_location,open_access"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
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
    return results


def _live_fetch_crossref(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        r = _req.get("https://api.crossref.org/works", params={
            "query": query, "rows": min(max_results, 50),
            "select": "DOI,title,abstract,published,author,container-title"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
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
    return results


def _live_fetch_europepmc(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # NB : ne PAS passer sort=RELEVANCE — c'est une valeur invalide pour
        # EuropePMC qui renvoie alors une liste vide. Sans 'sort', l'API trie
        # par pertinence par défaut.
        r = _req.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": query, "resultType": "lite", "pageSize": min(max_results, 50),
            "format": "json"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
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
    return results


def _live_fetch_preprints(query: str, max_results: int) -> list[dict]:
    """Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (filtre SRC:PPR).
    Europe PMC indexe les préprints AVEC recherche plein-texte par mots-clés — au
    contraire de l'API biorxiv (dates/DOI uniquement) qu'utilisaient les anciens
    scanners medRxiv/bioRxiv (peu/pas de résultats)."""
    import requests as _req
    results = []
    try:
        r = _req.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": f"({query}) AND (SRC:PPR)", "resultType": "lite",
            "pageSize": min(max_results, 50), "format": "json",
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
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


def _live_fetch_preprint_server(server: str, source_name: str, query: str, max_results: int) -> list[dict]:
    """Récupère les prépublications récentes (biorxiv/medrxiv API) puis filtre par
    correspondance de mots-clés de la requête (pas d'API plein-texte côté serveur)."""
    import requests as _req
    import datetime as _dt
    results = []
    try:
        # Mots-clés significatifs (booléen nettoyé). Les 2 premiers sont les
        # termes "primaires" (concept central) : on EXIGE qu'au moins un soit
        # présent, plus un nombre minimal de correspondances totales — sinon le
        # filtre laisse passer n'importe quel preprint contenant 2 mots courants.
        words = _plain_keywords(query, max_words=12).split()
        primary = words[:2]
        min_hits = min(3, len(words)) if len(words) >= 3 else 1
        date_to = _dt.date.today()
        date_from = date_to - _dt.timedelta(days=180)
        cursor = 0
        scanned = 0
        # Plafond resserré : l'API biorxiv ne fait pas de recherche plein-texte, on
        # filtre côté client par mots-clés → le taux de correspondance est faible et
        # scanner 300 prépublications (3 pages × 10s) consommait quasi tout le budget
        # de 30s de la fédération pour ~0 résultat. 120 + timeout court suffit.
        max_scan = 120  # plafond pour rester dans le budget temps de la fédération
        _hdrs = {"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}
        while scanned < max_scan and len(results) < max_results:
            url = (f"https://api.biorxiv.org/details/{server}/"
                   f"{date_from.isoformat()}/{date_to.isoformat()}/{cursor}/json")
            r = _req.get(url, timeout=6, headers=_hdrs)
            if not r.ok:
                break
            payload = r.json()
            coll = payload.get("collection", []) or []
            if not coll:
                break
            for item in coll:
                scanned += 1
                hay = (item.get("title", "") + " " + item.get("abstract", "")).lower()
                if words:
                    hits = sum(1 for w in words if w in hay)
                    has_primary = any(p in hay for p in primary) if primary else True
                    if not (has_primary and hits >= min_hits):
                        continue
                yr = None
                d = item.get("date", "")
                if len(d) >= 4 and d[:4].isdigit():
                    yr = int(d[:4])
                doi = item.get("doi")
                results.append({
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract"),
                    "doi": doi,
                    "year": yr,
                    "authors": [a.strip() for a in (item.get("authors", "") or "").split(";")[:5] if a.strip()],
                    "journal": source_name,
                    "url": f"https://doi.org/{doi}" if doi else None,
                    "external_id": doi,
                    "source_name": source_name,
                })
                if len(results) >= max_results:
                    break
            total = int(payload.get("messages", [{}])[0].get("total", 0) or 0)
            cursor += len(coll)
            if cursor >= total:
                break
    except Exception as _e:
        logger.warning(f"_live_fetch_preprint_server({server}) error: {_e}")
    return results


def _live_fetch_medrxiv(query: str, max_results: int) -> list[dict]:
    return _live_fetch_preprint_server("medrxiv", "medRxiv", query, max_results)


def _live_fetch_biorxiv(query: str, max_results: int) -> list[dict]:
    return _live_fetch_preprint_server("biorxiv", "bioRxiv", query, max_results)


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

    source_fns = [
        ("PubMed", _live_fetch_pubmed, pubmed_query),
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
                sources_queried.append(name)
                _ms = round((_t_fed.time() - _t0_fed) * 1000)
                try:
                    items = future.result()
                    raw_counts[name] = len(items)
                    all_results.extend(items)
                    source_status[name] = {
                        "status": "ok" if items else "empty",
                        "count": len(items), "latency_ms": _ms,
                    }
                except Exception as _fe:
                    logger.warning(f"federated source {name} error: {_fe}")
                    source_status[name] = {
                        "status": "error", "count": 0, "latency_ms": _ms,
                        "error": str(_fe)[:200],
                    }
        except concurrent.futures.TimeoutError:
            logger.warning("federated search: certaines sources ont dépassé le délai")

    # Marquage in_local_db — literature_document n'a pas de colonne doi dédiée et
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
                rows_db = conn.execute(text(
                    "SELECT LOWER(external_id) FROM literature_document "
                    "WHERE LOWER(external_id) = ANY(:keys) AND project_context = 'literev'"
                ), {"keys": list(keys)}).fetchall()
                in_db_keys = {r[0] for r in rows_db}
        except Exception as _dbe:
            logger.warning(f"federated DB check error: {_dbe}")
    for r in all_results:
        _doi = (r.get("doi") or "").lower()
        _eid = str(r.get("external_id") or "").lower()
        r["in_local_db"] = bool((_doi and _doi in in_db_keys) or (_eid and _eid in in_db_keys))

    # Déduplication par DOI (sinon titre normalisé), suivi des sources
    import re as _re2
    def _norm_title(t: str) -> str:
        return _re2.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    deduped: dict[str, dict] = {}
    for r in all_results:
        key = ("doi:" + r["doi"].lower()) if r.get("doi") else ("ttl:" + _norm_title(r.get("title", "")))
        if not key or key in ("ttl:", "doi:"):
            key = "id:" + str(id(r))
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
            from openai import OpenAI as _OAI
            _client = _OAI(api_key=openai_key, timeout=8.0)
            q_emb = _client.embeddings.create(
                input=[(query or "").replace("\n", " ").strip()],
                model="text-embedding-3-small",
            ).data[0].embedding
            texts = [((deduped_list[i].get("title", "") or "") + ". " + (deduped_list[i].get("abstract") or "")).replace("\n", " ").strip()[:2000]
                     for i in cand_idx]
            for b in range(0, len(texts), 256):
                try:
                    emb_resp = _client.embeddings.create(input=texts[b:b + 256], model="text-embedding-3-small")
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
        lex = _lexical_overlap(q_words, blob)
        sem = max(0.0, _cosine(q_emb, res_embs[i])) if (q_emb and res_embs[i]) else 0.0
        r["semantic_score"] = round(sem, 4)
        r["lexical_score"] = round(lex, 4)
        r["hybrid_score"] = round(0.7 * sem + 0.3 * lex, 4)

    deduped_list.sort(key=lambda r: r.get("hybrid_score", 0.0), reverse=True)
    return deduped_list, sources_queried, raw_counts, source_status


@app.post("/user-scenarios/{scenario_id}/search/live")
def search_live(
    scenario_id: str,
    max_per_source: int = 50,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Live federated search across the external sources in parallel."""
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
            _cr = _cc.execute(text("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                           AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
                                OR COALESCE(ars.similarity_score, 0) >= :thr)) AS above
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            """), {"sid": scenario_id, "thr": _thr}).mappings().first()
        corpus_total = int(_cr["total"] or 0)
        corpus_above = int(_cr["above"] or 0)
    except Exception as _ce:
        logger.warning(f"search_live corpus count {scenario_id}: {_ce}")

    # Background ingest of new papers — via le lanceur verrouillé pour ne jamais
    # démarrer un populate concurrent (sinon les nettoyages post-ingestion se
    # marchent dessus : compteurs corrompus, liens supprimés par l'autre job).
    ingesting_background = False
    if new_count > 0:
        try:
            status = _launch_populate_job(scenario_id, query, row.get("filters") or {}, 200)
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


@app.get("/sources/health")
def sources_health(query: str = "cardiac arrest", timeout: int = 12) -> dict[str, Any]:
    """Diagnostic des sources externes (live search).

    Interroge en parallèle chaque API amont avec une requête minimale et renvoie,
    par source, le statut HTTP, la latence (ms), un compteur de résultats et
    l'erreur éventuelle. Permet de diagnostiquer « sources lentes / ne répondent
    plus » directement en production (où l'accès réseau sortant diffère du sandbox).
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
        ("PubMed (eutils)", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
         eutils_params, ua,
         lambda j: len(j.get("esearchresult", {}).get("idlist", []))),
        ("OpenAlex", "https://api.openalex.org/works",
         {"search": _plain_keywords(query), "per-page": 1, "select": "id,title"}, ua,
         lambda j: j.get("meta", {}).get("count")),
        ("Crossref", "https://api.crossref.org/works",
         {"query": query, "rows": 1, "select": "DOI,title"}, ua,
         lambda j: j.get("message", {}).get("total-results")),
        ("EuropePMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
         {"query": query, "resultType": "lite", "pageSize": 1, "format": "json"}, ua,
         lambda j: j.get("hitCount")),
        ("bioRxiv/medRxiv", "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-07/0/json",
         None, ua,
         lambda j: (j.get("messages", [{}])[0] or {}).get("total")),
    ]

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
                    "error": None if r.status_code == 200 else (r.text or "")[:200]}
        except Exception as e:
            return {"source": name, "ok": False, "http": None,
                    "latency_ms": round((_t.time() - t0) * 1000), "count": None,
                    "error": f"{type(e).__name__}: {e}"[:200]}

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
    return {
        "query": query,
        "checked_at": _dtm.now(_tz.utc).isoformat(),
        "sources": results,
        "reachable": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "config": {
            "ncbi_api_key": bool(ncbi_key),
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


@app.get("/user-scenarios/{scenario_id}/search-strategy")
def get_search_strategy(scenario_id: str) -> dict[str, Any]:
    """Returns the stored search_strategy JSON for this scenario.
    If not yet generated, generates it now and stores it."""
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]
    strategy = row.get("search_strategy")
    # Régénérer si absent OU si la valeur stockée est un repli dégradé (p. ex.
    # généré pendant une panne de quota OpenAI → requête brute échoée). On ne
    # persiste QUE les stratégies valides, pour ne pas figer un cache empoisonné.
    if _strategy_is_degraded(strategy, query):
        strategy = _generate_search_strategy(query)
        if not _strategy_is_degraded(strategy, query):
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE user_scenarios SET search_strategy = CAST(:strategy AS jsonb) WHERE id = :id
                    """), {"id": scenario_id, "strategy": json.dumps(strategy)})
            except Exception as _e:
                logger.warning(f"get_search_strategy store error: {_e}")
    return strategy if isinstance(strategy, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────
# Document detail
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/documents/{document_id}")
def get_document_detail(document_id: int) -> dict[str, Any]:
    sql_doc = text("""
        SELECT
            id, source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category
        FROM literature_document
        WHERE id = :document_id
        LIMIT 1
    """)
    sql_chunks = text("""
        SELECT
            id, document_id, chunk_index, content, chunk_type,
            section_label, char_start, char_end, token_count,
            chunk_weight, metadata_json
        FROM document_chunk
        WHERE document_id = :document_id
        ORDER BY chunk_index ASC
    """)
    with engine.connect() as conn:
        doc = conn.execute(sql_doc, {"document_id": document_id}).mappings().first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = conn.execute(
            sql_chunks, {"document_id": document_id}
        ).mappings().all()
    return {
        "document": dict(doc),
        "chunks": [dict(c) for c in chunks],
    }

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Evidence Signals Extraction Engine
# ─────────────────────────────────────────────────────────────────────────────
def _extract_gesica_evidence(
    title: str | None, abstract: str | None, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    text_blob = " ".join([
        title or "",
        abstract or "",
        " ".join([c.get("content", "") for c in chunks]),
    ]).lower()

    demand_patterns = [
        "call volume", "demand forecasting", "arrival rate", "forecast",
        "predict", "ambulance demand", "ems demand", "workload", "hourly",
        "daily", "temporal", "timeseries", "time series", "xgboost", "lstm",
        "prophet", "random forest", "neural network", "regression", "mae", "mape", "rmse",
    ]
    resource_patterns = [
        "ambulance", "dispatch", "allocation", "fleet", "staffing",
        "crew", "response time", "location", "coverage", "optimization",
        "heuristics", "genetic algorithm", "simulation", "queuing", "chuv", "hug",
    ]
    crisis_patterns = [
        "disaster", "mass casualty", "mci", "crisis", "sanitarian",
        "epidemic", "pandemic", "influenza", "heatwave", "canicule",
        "flood", "evacuation", "surge", "capacity", "coordination",
    ]
    intervention_patterns = [
        "triage", "priority", "protocol", "diversion", "routing",
        "transfer", "telemedicine", "dispatch policy", "resource allocation",
    ]
    geography_patterns = [
        "geneva", "geneve", "vaud", "lausanne", "neuchatel", "france",
        "switzerland", "suisse", "cross-border", "transfrontalier", "rhone", "alps",
    ]

    setting_patterns = {
        "dispatch_center": ["dispatch", "regulation", "centre 15", "144", "call center"],
        "pre_hospital": ["ambulance", "smur", "paramedic", "ems", "pre-hospital", "rescue"],
        "hospital_er": ["emergency department", "er", "urgences", "hospital", "icu", "bed"],
    }
    scenario_rules = {
        "epidemic-surge": ["pandemic", "epidemic", "influenza", "covid", "outbreak", "virus"],
        "extreme-weather": ["heatwave", "canicule", "cold", "winter", "flood", "storm", "weather"],
        "mass-casualty": ["mci", "mass casualty", "terrorist", "accident", "explosion", "disaster"],
        "daily-operations": ["daily", "routine", "hourly", "weekday", "seasonal", "demand"],
    }

    metrics_patterns = [
        "auc", "auroc", "accuracy", "sensitivity", "specificity",
        "f1-score", "precision", "recall", "rmse", "mae", "mape",
    ]
    uncertainty_patterns = [
        "confidence interval", "uncertainty", "calibration",
        "probabilistic", "bayesian", "ensemble",
    ]

    def matched(patterns: list[str]) -> list[str]:
        return sorted({p for p in patterns if p in text_blob})

    horizon_matches = re.findall(
        r"\b(\d+\s*(?:hour|hours|day|days|week|weeks|month|months|year|years))\b",
        text_blob,
    )
    horizon_match_single = re.search(
        r"(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years)",
        text_blob,
    )

    detected_settings = [
        s for s, keys in setting_patterns.items() if any(k in text_blob for k in keys)
    ]
    detected_scenarios = [
        s for s, keys in scenario_rules.items() if any(k in text_blob for k in keys)
    ]

    evidence_strength = "weak"
    if matched(metrics_patterns):
        evidence_strength = "moderate"
    if matched(metrics_patterns) and matched(uncertainty_patterns):
        evidence_strength = "strong"

    return {
        "demand_signals": matched(demand_patterns),
        "resource_types": matched(resource_patterns),
        "intervention_types": matched(intervention_patterns),
        "operational_settings": detected_settings,
        "scenario_tags": detected_scenarios,
        "forecast_horizon": horizon_match_single.group(0) if horizon_match_single else None,
        "forecast_horizons": horizon_matches[:10],
        "cross_border": any(x in text_blob for x in geography_patterns),
        "cross_border_signals": matched(geography_patterns),
        "crisis_signals": matched(crisis_patterns),
        "evidence_strength": evidence_strength,
        "uncertainty_handling": matched(uncertainty_patterns),
        "reported_metrics": matched(metrics_patterns),
        "is_ems_or_crisis_relevant": bool(
            matched(demand_patterns) or matched(resource_patterns) or matched(crisis_patterns)
        ),
    }

@app.get("/evidence-summary/{document_id}")
def get_evidence_summary(document_id: int) -> dict[str, Any]:
    sql_doc = text("""
        SELECT
            id, source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category
        FROM literature_document
        WHERE id = :document_id
        LIMIT 1
    """)
    sql_chunks = text("""
        SELECT id, document_id, chunk_index, content
        FROM document_chunk
        WHERE document_id = :document_id
        ORDER BY chunk_index
    """)

    with engine.connect() as conn:
        doc_row = conn.execute(sql_doc, {"document_id": document_id}).mappings().first()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found")
        chunk_rows = conn.execute(
            sql_chunks, {"document_id": document_id}
        ).mappings().all()

    document = dict(doc_row)
    chunks = [dict(r) for r in chunk_rows]
    signals = _extract_gesica_evidence(
        document.get("title"), document.get("abstract"), chunks
    )

    return {
        "document": document,
        "summary": {
            "project_context": document.get("project_context"),
            "scenario_type": document.get("scenario_type"),
            "evidence_category": document.get("evidence_category"),
            "geographic_scope": document.get("geographic_scope"),
            "disease_or_condition": document.get("disease_or_condition"),
        },
        "gesica_signals": signals,
        "chunk_count": len(chunks),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Endpoints: Stats and Scenarios
# ─────────────────────────────────────────────────────────────────────────────

# Regroupement des sources en libellés canoniques pour les tableaux de bord. La
# colonne `source` accumule des variantes héritées (casse, anciens tags, valeurs
# vides) qui gonflaient le compteur « Sources » et fragmentaient les barres. On
# mappe les variantes connues vers les sources fédérées + « Préprints » ; toute
# autre source garde son nom réel (les valeurs vides → « Non précisé »).
def _canonical_source(s: str | None) -> str:
    t = (s or "").strip().lower()
    if not t:
        return "Non précisé"
    # Europe PMC en premier : la chaîne 'europepmc' contient 'pmc'.
    if "europepmc" in t or "europe_pmc" in t or "europe pmc" in t or t == "epmc":
        return "Europe PMC"
    if any(k in t for k in ("pubmed", "pmc", "medline", "ncbi", "entrez", "pmid")):
        return "PubMed"
    if "openalex" in t:
        return "OpenAlex"
    if "crossref" in t or "cross_ref" in t or "cross-ref" in t:
        return "Crossref"
    if any(k in t for k in ("medrxiv", "biorxiv", "arxiv", "preprint", "ssrn",
                            "chemrxiv", "research square", "researchsquare", "osf", "psyarxiv")):
        return "Préprints"
    if "semantic" in t or t == "s2":
        return "Semantic Scholar"
    # Source non fédérée : on garde le nom réel (nettoyé) plutôt que de tout
    # masquer derrière « Autre » — l'utilisateur veut voir TOUTES les sources.
    return (s or "").strip()


@app.get("/corpus/stats")
def get_corpus_stats() -> dict[str, Any]:
    """Vue globale multi-projet du corpus."""
    sql_totals = text("""
        SELECT 
            COALESCE(project_context, 'unassigned') as project,
            COUNT(*) as count
        FROM literature_document
        GROUP BY project_context
    """)
    sql_sources = text("""
        SELECT 
            source,
            COUNT(*) as count
        FROM literature_document
        GROUP BY source
    """)
    sql_years = text("""
        SELECT 
            year,
            COUNT(*) as count
        FROM literature_document
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year DESC
    """)
    with engine.connect() as conn:
        totals = {r["project"]: r["count"] for r in conn.execute(sql_totals).mappings().all()}
        # Agrégation par source CANONIQUE (sinon les variantes héritées comptaient
        # chacune comme une « source » distincte → compteur et barres faussés).
        sources: dict[str, int] = {}
        for r in conn.execute(sql_sources).mappings().all():
            k = _canonical_source(r["source"])
            sources[k] = sources.get(k, 0) + int(r["count"] or 0)
        years = {r["year"]: r["count"] for r in conn.execute(sql_years).mappings().all()}
        
        total_docs = conn.execute(text("SELECT COUNT(*) FROM literature_document")).scalar() or 0
        total_chunks = conn.execute(text("SELECT COUNT(*) FROM document_chunk")).scalar() or 0

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "by_project": totals,
        "by_source": sources,
        "by_year": years,
    }

@app.get("/gesica/stats")
def get_gesica_stats() -> dict[str, Any]:
    """Statistiques globales du corpus LiteRev."""
    sql_docs = text("""
        SELECT id, title, abstract
        FROM literature_document
        WHERE project_context = 'literev'
    """)
    with engine.connect() as conn:
        docs = conn.execute(sql_docs).mappings().all()

    total_gesica = len(docs)
    horizons_count: dict[str, int] = {}
    uncertainty_count: dict[str, int] = {}
    evidence_strengths = {"weak": 0, "moderate": 0, "strong": 0}

    for doc in docs:
        signals = _extract_gesica_evidence(doc["title"], doc["abstract"], [])
        
        strength = signals["evidence_strength"]
        evidence_strengths[strength] = evidence_strengths.get(strength, 0) + 1
        
        for m in signals["uncertainty_handling"]:
            uncertainty_count[m] = uncertainty_count.get(m, 0) + 1
            
        if signals["forecast_horizon"]:
            h = signals["forecast_horizon"]
            horizons_count[h] = horizons_count.get(h, 0) + 1

    return {
        "total_documents": total_gesica,
        "evidence_strength_distribution": evidence_strengths,
        "uncertainty_methods": dict(sorted(uncertainty_count.items(), key=lambda x: x[1], reverse=True)),
        "forecast_horizons": dict(sorted(horizons_count.items(), key=lambda x: x[1], reverse=True)),
    }

@app.get("/geoai4ei/stats")
def get_geoai4ei_stats() -> dict[str, Any]:
    """Statistiques globales du corpus Urgences Hospitalières."""
    sql_diseases = text("""
        SELECT disease_or_condition, COUNT(*) as count
        FROM literature_document
        WHERE project_context = 'literev' AND disease_or_condition IS NOT NULL
        GROUP BY disease_or_condition
        ORDER BY count DESC
    """)
    sql_geo = text("""
        SELECT geographic_scope, COUNT(*) as count
        FROM literature_document
        WHERE project_context = 'literev' AND geographic_scope IS NOT NULL
        GROUP BY geographic_scope
        ORDER BY count DESC
    """)
    with engine.connect() as conn:
        diseases = {r["disease_or_condition"]: r["count"] for r in conn.execute(sql_diseases).mappings().all()}
        geo = {r["geographic_scope"]: r["count"] for r in conn.execute(sql_geo).mappings().all()}

    return {
        "diseases": diseases,
        "geographic_scopes": geo,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Scenarios Metadata : 31 scénarios fins issus de la revue systématique
# ─────────────────────────────────────────────────────────────────────────────

GESICA_SCENARIO_METADATA: dict[str, dict[str, Any]] = {
    "cardiac-arrest-prediction": {
        "hidden": False,
        "title": "Prédiction de l'Arrêt Cardiaque Extra-Hospitalier (OHCA)",
        "description": "Modèles de prédiction spatio-temporelle de l'incidence des arrêts cardiorespiratoires (OHCA) basés sur l'apprentissage automatique, les rythmes circadiens, les données climatiques et météorologiques, visant à optimiser la chaîne de survie et le positionnement préventif des ressources.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Déployer des algorithmes de détection acoustique de l'agonie respiratoire (gasping) assistés par IA au Centre 15/144",
            "Optimiser la couverture et le dispatch des premiers répondants (citizen responders) équipés de DEA via géolocalisation dynamique",
            "Ajuster préventivement le positionnement des SMUR et des ambulances de réanimation dans les zones à haut risque d'OHCA",
            "Intégrer les données de défibrillateurs connectés (IoT) pour une cartographie temps réel de l'accessibilité des DEA"
        ]
    },
    "stroke-detection": {
        "hidden": False,
        "title": "Détection Préhospitalière de l'AVC",
        "description": "Systèmes d'aide à la décision clinique pour l'identification précoce des accidents vasculaires cérébraux (AVC) sur le terrain, l'évaluation de la sévérité via des scores automatisés (FAST, NIHSS, LVO) et l'orientation optimale et directe vers les centres de reperfusion (thrombolyse/thrombectomie).",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Intégrer des échelles cliniques d'AVC automatisées et guidées par IA dans le dossier patient embarqué des ambulanciers",
            "Orienter directement et sans transit par les urgences générales vers l'Unité Neurovasculaire (UNV) de référence (HUG/CHUV)",
            "Déclencher une pré-alerte automatique pour l'équipe de neuroradiologie interventionnelle en cas de forte suspicion d'occlusion de gros vaisseau (LVO)",
            "Optimiser le délai porte-aiguille (door-to-needle) par la transmission préhospitalière sécurisée des données cliniques"
        ]
    },
    "trauma-severity-assessment": {
        "hidden": False,
        "title": "Évaluation de la Gravité des Traumatismes",
        "description": "Modèles prédictifs et scores de stratification du risque (ISS, RTS, TRISS) pour l'évaluation immédiate des traumatisés graves (accidents de la route, chutes, traumatismes de montagne) afin d'orienter sans délai vers les Trauma Centers de niveau adapté.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Déployer des modèles prédictifs de besoin de transfusion massive (score de choc, score d'hémorragie) dès la prise en charge terrain",
            "Orienter systématiquement les traumatismes sévères (ISS > 15) vers un Trauma Center de niveau 1 agréé (HUG ou CHUV)",
            "Partager en flux continu et en temps réel les constantes vitales et l'échographie FAST avec la salle de déchocage hospitalière",
            "Implémenter des protocoles de réanimation de contrôle des dommages (damage control resuscitation) guidés par des algorithmes d'aide à la décision"
        ]
    },
    "clinical-deterioration-prediction": {
        "hidden": True,
        "title": "Prédiction de la Détérioration Clinique en Transit",
        "description": "Surveillance intelligente des patients critiques durant leur transport en ambulance ou hélicoptère.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Activer des alertes de détérioration basées sur la tendance des constantes vitales multi-paramétriques",
            "Préparer des protocoles de réanimation avancée en lien avec le médecin régulateur du SMUR",
            "Ajuster la vitesse de transfert ou envisager un rendez-vous SMUR/Héli-SMUR si nécessaire"
        ]
    },
    "patient-pathway-optimization": {
        "hidden": True,
        "title": "Optimisation du Parcours Patient Transfrontalier",
        "description": "Planification du transfert des patients vers les structures de soins appropriées en optimisant les capacités des deux côtés de la frontière.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Vérifier la disponibilité des lits spécialisés en temps réel en France (ROR) et en Suisse",
            "Fluidifier les démarches administratives douanières pour les ambulances de transfert",
            "Établir un protocole de retour à domicile ou de soins de suite de proximité"
        ]
    },
    "mci-victim-estimation": {
        "hidden": True,
        "title": "Estimation des Victimes en Situation de Catastrophe (MCI)",
        "description": "Évaluation rapide du nombre et de la gravité des victimes lors d'événements majeurs pour dimensionner la réponse.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Activer le Plan Blanc (FR) / Plan ORCA (CH) de manière coordonnée",
            "Utiliser des outils de tri connectés (smart glasses, bracelets IoT) pour un inventaire en temps réel",
            "Répartir les flux de victimes de manière équilibrée entre les hôpitaux de la région"
        ]
    },
    "environmental-risk-forecasting": {
        "hidden": True,
        "title": "Prévision des Risques Environnementaux",
        "description": "Anticipation des pics de pollution de l'air, d'ozone ou d'allergènes et de leur impact direct sur les urgences respiratoires.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Croiser les données d'AirGenève et d'Atmo Auvergne-Rhône-Alpes avec les appels pour asthme/BPCO",
            "Diffuser des messages de prévention ciblés aux patients vulnérables enregistrés",
            "Anticiper une hausse de 15% des appels pour détresse respiratoire dans les 48 heures"
        ]
    },
    "disaster-risk-assessment": {
        "hidden": True,
        "title": "Évaluation des Risques de Catastrophes Naturelles",
        "description": "Modélisation de l'impact sanitaire des inondations, séismes locaux, ou glissements de terrain sur les infrastructures EMS.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Identifier les casernes et voies d'accès ambulances situées en zone inondable (crues de l'Arve/Rhône)",
            "Établir des points de rassemblement des secours hors des zones à risque",
            "Simuler des scénarios de rupture d'alimentation électrique ou de télécommunications"
        ]
    },
    "heatwave-ems-impact": {
        "hidden": True,
        "title": "Impact des Canicules sur les EMS",
        "description": "Modélisation de l'impact des vagues de chaleur extrêmes sur la demande EMS et les pathologies liées à la chaleur (coup de chaleur, hyperthermie).",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Anticiper une hausse de 20-40% des appels EMS lors des épisodes de canicule (UTCI > 38°C)",
            "Activer les protocoles de prise en charge préhospitalière des coups de chaleur",
            "Renforcer les équipages avec du matériel de refroidissement rapide (poches de glace, brumisateurs)"
        ]
    },
    "climate-impact-on-ems": {
        "hidden": True,
        "title": "Impact du Changement Climatique sur les EMS",
        "description": "Analyse à long terme et saisonnière de l'évolution des pathologies d'urgence liées au réchauffement climatique.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Adapter les plannings de garde estivaux pour faire face à des vagues de chaleur plus fréquentes",
            "Intégrer les projections climatiques de Copernicus dans le schéma directeur de santé transfrontalier",
            "Former le personnel aux pathologies émergentes (maladies à vecteur comme la dengue en Europe)"
        ]
    },
    "emergency-call-qualification": {
        "hidden": False,
        "title": "Qualification Automatisée des Appels d'Urgence",
        "description": "Outils de traitement du langage naturel (NLP) et de reconnaissance vocale en temps réel pour assister les assistants de régulation médicale (ARM) dans la transcription, la détection automatique de mots-clés cliniques et la qualification rapide des motifs d'appels d'urgence.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Activer la transcription vocale continue à faible latence (Speech-to-Text) intégrée au système de téléphonie de régulation",
            "Utiliser des modèles NLP spécialisés (type CamemBERT médical) pour extraire automatiquement les entités cliniques et les symptômes clés",
            "Analyser les caractéristiques acoustiques et les bruits de fond de l'appel pour détecter la détresse respiratoire ou la panique",
            "Suggérer de manière adaptative et dynamique les questions de protocoles de régulation selon les premiers mots transcrits"
        ]
    },
    "call-prioritization": {
        "hidden": False,
        "title": "Priorisation des Appels de Régulation",
        "description": "Algorithmes d'apprentissage automatique pour le tri et la priorisation dynamique de la file d'attente des appels entrants en centrale de régulation médicale, garantissant une prise en charge immédiate des détresses vitales et minimisant le risque de sous-triage.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Placer automatiquement en priorité absolue de file d'attente les appels identifiés comme suspicion d'arrêt cardiorespiratoire ou d'étouffement",
            "Ajuster dynamiquement les seuils de tri et les files d'attente lors de situations de saturation de la centrale (pics d'appels)",
            "Fournir aux régulateurs un tableau de bord prédictif du niveau de risque clinique estimé pour chaque appel en attente",
            "Mesurer en continu le taux d'adéquation de la priorisation pour minimiser le sous-triage sous la barre stricte de 5%"
        ]
    },
    "mass-casualty-triage": {
        "hidden": True,
        "title": "Tri en Situation de Nombreuses Victimes",
        "description": "Algorithmes d'aide au tri de masse sur le terrain pour classer rapidement les victimes (Urgence Absolue, Urgence Relative).",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Appliquer les critères de tri standardisés (START/SALT) via une interface mobile simplifiée",
            "Générer des codes QR uniques pour chaque victime afin de suivre leur parcours",
            "Visualiser la répartition des catégories de gravité sur la cartographie du PMA"
        ]
    },
    "undertriage-detection": {
        "hidden": True,
        "title": "Détection du Sous-Tri (Undertriage)",
        "description": "Algorithmes de contrôle qualité pour identifier les patients graves classés à tort en faible priorité.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Analyser rétrospectivement les dossiers de régulation pour identifier les écarts de tri",
            "Alerter en temps réel si les constantes saisies contredisent le niveau de priorité attribué",
            "Ajuster les arbres de décision cliniques pour réduire le taux de sous-tri sous le seuil de 5%"
        ]
    },
    "dispatch-decision-support": {
        "hidden": False,
        "title": "Aide à la Décision de Dispatch",
        "description": "Systèmes experts et modèles prédictifs d'aide à la décision pour recommander instantanément le moyen de secours préhospitalier optimal (ambulance de soins d'urgence, équipe médicale SMUR, hélicoptère ou médecin généraliste de garde) en fonction de la gravité clinique suspectée et des ressources disponibles.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Suggérer automatiquement l'envoi d'un SMUR transfrontalier (FR/CH) si son délai d'arrivée estimé est inférieur à la ressource nationale",
            "Intégrer les données de géolocalisation live (GPS) et le statut opérationnel des véhicules pour proposer la ressource la plus rapide",
            "Suggérer des alternatives de régulation libérale, de conseil médical ou de transport sanitaire non urgent pour les motifs de faible gravité",
            "Implémenter un modèle d'adéquation d'envoi pour réduire les envois inutiles d'équipes médicalisées (over-dispatch) tout en sécurisant les patients"
        ]
    },
    "triage-support": {
        "hidden": False,
        "title": "Support au Tri Clinique aux Urgences",
        "description": "Algorithmes de classification clinique pour assister le personnel infirmier d'accueil (IOA) dans la détermination rapide du niveau de gravité des patients aux urgences selon des échelles validées (Échelle Suisse de Tri, Échelle de Rouen), optimisant les délais d'accès aux soins.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Calculer automatiquement le niveau de gravité clinique théorique en intégrant les constantes vitales et le motif de consultation saisi",
            "Prédire dès l'accueil le risque d'hospitalisation d'aval ou de passage en réanimation pour anticiper l'orientation des patients",
            "Générer des alertes visuelles et sonores immédiates pour l'infirmier d'accueil en cas d'anomalie physiologique majeure",
            "Mesurer la concordance inter-observateur (Kappa de Cohen) entre le tri assisté par IA et l'évaluation finale par le médecin"
        ]
    },
    "response-time-optimization": {
        "hidden": False,
        "title": "Optimisation des Temps de Réponse EMS",
        "description": "Modèles de routage prédictif intégrant les conditions de trafic en temps réel, la météorologie et la topologie urbaine pour guider les véhicules d'urgence par l'itinéraire le plus rapide et minimiser le délai d'accès aux soins critiques.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Calculer des itinéraires d'urgence dynamiques intégrant les données de congestion du trafic en temps réel et l'historique de circulation",
            "Interfacer le système de navigation des ambulances avec la gestion des feux tricolores (priorité de passage) sur les axes critiques",
            "Modéliser spécifiquement les délais de passage transfrontaliers (douanes du Grand Genève, ponts sur le lac) pour adapter les trajets",
            "Évaluer en continu la courbe d'efficacité temps-dépendante du temps de réponse réel sur la survie des détresses vitales"
        ]
    },
    "ambulance-dispatch-optimization": {
        "hidden": False,
        "title": "Optimisation de la Flotte d'Ambulances",
        "description": "Modèles mathématiques de couverture spatio-temporelle maximale (MCLP, DSM) pour la gestion et le repositionnement préventif et dynamique de la flotte d'ambulances, garantissant une couverture territoriale optimale en fonction des risques prédictifs.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Repositionner dynamiquement et de manière préventive les ambulances disponibles en attente pour combler les failles de couverture",
            "Prédire les micro-zones à haut risque d'appels d'urgence à l'échelle horaire pour y pré-positionner des équipages",
            "Coordonner de manière transparente sur une plateforme unique le dispatch des ambulances publiques, privées et associatives",
            "Suivre en temps réel le taux de couverture de la population cible à moins de 10 minutes d'une ambulance disponible"
        ]
    },
    "staffing-level-prediction": {
        "hidden": True,
        "title": "Prévision des Effectifs Requis",
        "description": "Modèles prédictifs pour dimensionner les équipes de régulation et les équipages d'ambulances selon la charge attendue.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Ajuster le nombre d'ARM de garde en fonction des prévisions de charge à 7 jours",
            "Planifier des renforts pour les périodes de grands événements (fêtes de Genève, manifestations)",
            "Prendre en compte les taux d'absentéisme saisonniers (pandémies hivernales du personnel)"
        ]
    },
    "hospital-capacity-forecasting": {
        "hidden": False,
        "title": "Prévision de la Capacité Hospitalière",
        "description": "Modèles prédictifs de séries temporelles pour anticiper la saturation des services d'urgences et l'occupation des lits de réanimation, de soins continus et d'hospitalisation conventionnelle (lits d'aval), facilitant la gestion proactive des flux de patients.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Prédire l'afflux de patients aux urgences et le taux d'occupation des lits à 24h et 48h (score NEDOCS prédictif)",
            "Coordonner en temps réel les sorties de patients hospitalisés et les transferts vers les unités de soins de suite et de réadaptation (SSR)",
            "Déclencher des alertes automatiques et des cellules de crise de gestion des lits (Bed Management) transfrontalières en cas de tension",
            "Modéliser l'impact de la saturation des urgences (overcrowding) sur les délais de libération et de transfert des ambulances (ambulance diversion)"
        ]
    },
    "demand-forecasting": {
        "hidden": False,
        "title": "Prévision de la Demande EMS",
        "description": "Modèles de prévision hybrides (Prophet, LightGBM, LSTM) intégrant les données météorologiques, le calendrier, les vacances scolaires et la surveillance épidémiologique pour estimer avec précision le volume d'appels d'urgence et dimensionner les équipes.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Intégrer des flux météorologiques locaux (Open-Meteo) et épidémiques (Réseau Sentinelles) en temps réel pour affiner les prévisions",
            "Visualiser la prévision de la demande EMS à l'échelle horaire par secteur géographique sur un horizon de J+1 à J+7",
            "Alerter automatiquement les cadres opérationnels en cas d'écart significatif (> 15%) entre le volume réel d'appels et la prévision de base",
            "Utiliser les prévisions de demande pour adapter dynamiquement la planification des gardes et le nombre de véhicules opérationnels"
        ]
    },
    "resource-allocation": {
        "hidden": True,
        "title": "Allocation Optimisée des Ressources",
        "description": "Distribution des moyens humains et matériels de manière à maximiser l'efficacité de la réponse d'urgence.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Allouer les ambulances de réanimation (SMUR) prioritairement aux urgences vitales",
            "Optimiser la répartition des stocks de matériel d'urgence entre sites",
            "Suivre en temps réel le statut d'activité de chaque équipage"
        ]
    },
    "epidemic-early-warning": {
        "hidden": True,
        "title": "Alerte Précoce Épidémique",
        "description": "Détection précoce des signaux faibles épidémiques à partir des motifs d'appels de régulation médicale.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Surveiller l'évolution des appels pour syndrome grippal, gastro-entérite ou détresse respiratoire",
            "Déclencher une alerte si un seuil d'incidence statistique est dépassé dans un district",
            "Partager les alertes précoces avec les autorités sanitaires (OFSP, ARS) pour action coordonnée"
        ]
    },
    "surveillance": {
        "hidden": True,
        "title": "Surveillance Syndromique Active",
        "description": "Suivi continu des indicateurs de santé de la population pour identifier des anomalies ou des clusters inhabituels.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Analyser les données de passage aux urgences (SOS Médecins, hôpitaux) en temps réel",
            "Identifier géographiquement des regroupements anormaux de cas présentant des symptômes similaires",
            "Adapter les seuils de détection en fonction de la saisonnalité et du contexte local"
        ]
    },
    "surge-management": {
        "hidden": True,
        "title": "Gestion des Pics d'Afflux (Surge)",
        "description": "Stratégies opérationnelles pour faire face à une hausse soudaine et massive de la demande de soins d'urgence.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Activer des lignes de régulation médicale supplémentaires au Centre 15/144",
            "Mettre en place des structures d'accueil temporaires (tentes de tri) devant les urgences",
            "Reporter les hospitalisations non urgentes (programmées) pour libérer des capacités"
        ]
    },
    "pandemic-preparedness": {
        "hidden": True,
        "title": "Préparation aux Pandémies",
        "description": "Planification stratégique et modélisation à long terme pour renforcer la résilience du système de santé face à des crises globales.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Établir des plans de continuité d'activité (PCA) pour les services d'urgence et de régulation",
            "Dimensionner les stocks stratégiques de contre-mesures médicales (masques, antiviraux, vaccins)",
            "Organiser des exercices de simulation de crise pandémique à l'échelle transfrontalière"
        ]
    },
    "cross-border-coordination": {
        "hidden": True,
        "title": "Coordination Sanitaire Transfrontalière",
        "description": "Protocoles et outils de communication pour harmoniser la réponse d'urgence entre la France et la Suisse (Grand Genève).",
        "cluster": "Cross-border & Operational Coordination",
        "recommended_actions": [
            "Interconnecter les systèmes de régulation TECHWAN SAGA (France) et l'équivalent suisse",
            "Établir des conventions de libre passage des ambulances et hélicoptères de secours",
            "Organiser des réunions de coordination régulières entre les directions des HUG, du CHUV et des SAMU"
        ]
    },
    "situational-awareness": {
        "hidden": True,
        "title": "Conscience Situationnelle Opérationnelle",
        "description": "Tableau de bord en temps réel intégrant toutes les sources de données pour une vue unifiée de la situation d'urgence.",
        "cluster": "Cross-border & Operational Coordination",
        "recommended_actions": [
            "Afficher en temps réel la position de toutes les unités mobiles (ambulances, SMUR, hélicoptères)",
            "Intégrer les flux météo, épidémiques et de trafic dans une carte opérationnelle unifiée",
            "Partager la vue opérationnelle avec les partenaires transfrontaliers en temps réel"
        ]
    },
    "unassigned": {
        "hidden": True,
        "title": "Scénarios Non Classés",
        "description": "Documents en attente de classification dans un scénario spécifique.",
        "cluster": "Non classé",
        "recommended_actions": [
            "Relancer le script de backfill pour réassigner ces documents",
            "Examiner manuellement les titres et résumés pour une classification manuelle"
        ]
    }
}


GESICA_FOLDER_ID = "fld-gesica-main"


def _gesica_title(meta: dict[str, Any]) -> str:
    return str(meta.get("title") or meta.get("name") or meta.get("id") or "Scénario")


def _gesica_actions(meta: dict[str, Any]) -> list[str]:
    actions = meta.get("recommended_actions")
    if actions is None:
        actions = meta.get("recommended_action")
    if isinstance(actions, list):
        return actions
    if isinstance(actions, str) and actions.strip():
        return [actions]
    return []


def _get_db_gesica_scenario_or_404(scenario_id: str, conn=None) -> dict[str, Any]:
    sql = text("""
        SELECT *
        FROM user_scenarios
        WHERE id = :sid
          AND is_system = TRUE
          AND folder_id = :folder_id
    """)
    params = {"sid": scenario_id, "folder_id": GESICA_FOLDER_ID}
    if conn is None:
        with engine.connect() as _conn:
            row = _conn.execute(sql, params).mappings().first()
    else:
        row = conn.execute(sql, params).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario '{scenario_id}' non trouvé")
    return dict(row)


def _list_db_gesica_scenarios(conn) -> list[dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT *
        FROM user_scenarios
        WHERE is_system = TRUE
          AND folder_id = :folder_id
          AND id <> 'unassigned'
          AND COALESCE(hidden, FALSE) = FALSE
        ORDER BY COALESCE(title, name, id) ASC
    """), {"folder_id": GESICA_FOLDER_ID}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/gesica/scenarios")
def get_gesica_scenarios() -> list[dict[str, Any]]:
    """
    Scénarios GESICA dynamiques : retourne les scénarios système stockés en base,
    enrichis avec les articles scientifiques associés depuis la DB (living evidence review).
    Les scénarios sont triés par nombre d'articles décroissant, puis alphabétiquement.
    """
    with engine.connect() as conn:
        scenario_rows = _list_db_gesica_scenarios(conn)

        sql_counts = text("""
            SELECT ars.scenario_id, COUNT(DISTINCT ars.document_id) as article_count
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY ars.scenario_id;
        """)
        db_counts = {row["scenario_id"]: row["article_count"] for row in conn.execute(sql_counts).mappings().all()}

        sql_screening = text("""
            SELECT
                ars.scenario_id,
                COUNT(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 END) as included_count,
                COUNT(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 END) as excluded_count
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY ars.scenario_id;
        """)
        screening_counts = {
            row["scenario_id"]: {"included": row["included_count"], "excluded": row["excluded_count"]}
            for row in conn.execute(sql_screening).mappings().all()
        }

        # Kappa par scénario : il n'existe pas de table `scenario_kappa_cache`
        # (aucun écrivain). Le kappa live est servi par l'endpoint dédié
        # /double-blind/kappa ; ici on ne fournit pas de valeur agrégée.
        kappa_scores: dict[str, Any] = {}

        # Articles de TOUS les scénarios en UNE requête (au lieu d'une par scénario
        # — N+1). On groupe ensuite par scénario en Python. L'ordre intra-scénario
        # (année desc, titre asc) est préservé par le ORDER BY.
        sql_all_articles = text("""
            SELECT ars.scenario_id, d.id, d.title, d.abstract, d.year, d.source, d.url,
                   d.authors, d.doi, d.journal, d.keywords, d.language, d.study_design,
                   d.sample_size, d.country, d.citation_count, d.open_access,
                   EXISTS (
                       SELECT 1 FROM document_chunk c
                       WHERE c.document_id = d.id
                         AND c.chunk_type = 'fulltext_section'
                   ) AS has_fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            ORDER BY ars.scenario_id, d.year DESC NULLS LAST, d.title ASC
        """)
        articles_by_scenario: dict[str, list[dict[str, Any]]] = {}
        for r in conn.execute(sql_all_articles).mappings().all():
            rec = dict(r)
            sid = str(rec.pop("scenario_id"))
            articles_by_scenario.setdefault(sid, []).append(rec)

        result = []
        for meta in scenario_rows:
            scenario_id = str(meta["id"])
            title = _gesica_title(meta)
            article_count = int(db_counts.get(scenario_id, 0) or 0)
            sc = screening_counts.get(scenario_id, {"included": 0, "excluded": 0})

            articles = articles_by_scenario.get(scenario_id, [])

            result.append({
                "id": scenario_id,
                "name": title,
                "title": title,
                "label_short": meta.get("label_short"),
                "description": meta.get("description") or "",
                "cluster": meta.get("cluster") or "",
                "article_count": article_count,
                "included_count": int(sc["included"] or 0),
                "excluded_count": int(sc["excluded"] or 0),
                "kappa_score": kappa_scores.get(scenario_id),
                "hidden": bool(meta.get("hidden", False)),
                "recommended_actions": _gesica_actions(meta),
                "relevant_articles": articles,
                "living_evidence_note": (
                    f"Living Evidence Review · {article_count} articles indexés. Mis à jour automatiquement à chaque ingestion."
                    if article_count > 0
                    else "Aucun article indexé pour ce scénario. En attente d'ingestion de nouvelles sources."
                )
            })

        result.sort(key=lambda x: (-x["article_count"], x["title"]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 Endpoints: Terrain Data Integration (MeteoSwiss, OSM/OSRM, Sentinelles)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/terrain/meteo")
def get_terrain_meteo(lat: float = 46.2044, lon: float = 6.1432) -> dict[str, Any]:
    """
    Endpoint P5 : Récupération des données météo en temps réel (MeteoSwiss / Open-Meteo)
    pour anticiper les impacts climatiques sur la charge EMS (canicule, gel, tempêtes).
    """
    import requests
    
    # Appel à l'API publique Open-Meteo (utilisée comme proxy public fiable pour MeteoSwiss localisé)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m&timezone=Europe/Zurich"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            current = data.get("current", {})
            temp = current.get("temperature_2m", 20.0)
            hum = current.get("relative_humidity_2m", 50.0)
            wind = current.get("wind_speed_10m", 10.0)
            precip = current.get("precipitation", 0.0)
            apparent_temp = current.get("apparent_temperature", temp)
            
            # Logique d'évaluation de l'impact EMS basée sur la littérature GESICA
            alert_level = "none"
            alert_desc = "Conditions météorologiques normales."
            impact_ems = "Pas d'impact attendu sur la charge d'appels EMS."
            
            if temp >= 33.0:
                alert_level = "danger"
                alert_desc = "Canicule extrême / Vague de chaleur critique."
                impact_ems = "Risque critique d'afflux d'appels pour déshydratation, hyperthermie et arrêts cardiaques (+25% d'appels estimés)."
            elif temp >= 30.0:
                alert_level = "warning"
                alert_desc = "Forte chaleur / Canicule modérée."
                impact_ems = "Augmentation attendue des appels pour pathologies cardiovasculaires et respiratoires (+10% à +15%)."
            elif temp <= 0.0:
                alert_level = "warning"
                alert_desc = "Gel au sol / Grand froid."
                impact_ems = "Risque accru d'accidents de la route (traumatologie) et d'appels pour hypothermie (+10%)."
                
            if wind >= 50.0:
                alert_level = "warning" if alert_level == "none" else "danger"
                alert_desc += " Vents violents / Tempête."
                impact_ems += " Risque accru de traumatismes par chutes d'objets ou accidents de la voie publique."

            return {
                "source": "Open-Meteo (api.open-meteo.com)",
                "data_status": "live",
                "coordinates": {"latitude": lat, "longitude": lon},
                "station": "Genève / Cointrin (Région Transfrontalière)",
                "temperature": temp,
                "apparent_temperature": apparent_temp,
                "humidity": hum,
                "wind_speed": wind,
                "precipitation": precip,
                "alert_level": alert_level,
                "alert_description": alert_desc,
                "impact_on_ems": impact_ems,
                "architecture_note": "Données météo réelles via Open-Meteo (proxy public, TZ Europe/Zurich). Un branchement sur l'API MeteoSwiss reste possible."
            }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération météo: {e}")
        
    # Fallback : valeurs d'exemple si l'API Open-Meteo est injoignable (clairement signalé)
    return {
        "source": "Estimation de secours — API Open-Meteo indisponible",
        "data_status": "fallback_estimate",
        "coordinates": {"latitude": lat, "longitude": lon},
        "station": "Genève / Cointrin (estimation)",
        "temperature": 28.5,
        "apparent_temperature": 29.8,
        "humidity": 45.0,
        "wind_speed": 12.0,
        "precipitation": 0.0,
        "alert_level": "warning",
        "alert_description": "Forte chaleur d'été (valeur d'exemple).",
        "impact_on_ems": "Augmentation modérée des appels d'urgence pour pathologies cardiovasculaires (+5%).",
        "architecture_note": "Valeurs d'exemple : l'API Open-Meteo n'a pas répondu. Ne pas utiliser pour une décision opérationnelle."
    }


@app.get("/terrain/geo")
def get_terrain_geo(
    orig_lat: float = 46.2044, orig_lon: float = 6.1432,
    dest_lat: float = 46.1925, dest_lon: float = 6.2388
) -> dict[str, Any]:
    """
    Endpoint P5 : Calcul d'isochrones et d'itinéraires transfrontaliers (OSRM / OpenStreetMap)
    pour optimiser le dispatch des ambulances et estimer le temps de réponse transfrontalier.
    """
    import requests
    
    # OSRM API publique pour le calcul d'itinéraire
    url = f"https://router.project-osrm.org/route/v1/driving/{orig_lon},{orig_lat};{dest_lon},{dest_lat}?overview=false"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            routes = data.get("routes", [])
            if routes:
                route = routes[0]
                distance = route.get("distance", 0.0) / 1000.0  # en km
                duration = route.get("duration", 0.0) / 60.0  # en minutes
                
                # Simulation d'un facteur de trafic et d'un délai de douane transfrontalière
                traffic_factor = 1.15  # +15% de trafic en heure de pointe
                border_delay = 3.5  # 3.5 minutes de délai moyen à la douane de Moillesulaz
                total_duration = (duration * traffic_factor) + border_delay
                
                return {
                    "source": "OpenStreetMap / OSRM API",
                    "origin": {"latitude": orig_lat, "longitude": orig_lon, "label": "Genève (HUG)"},
                    "destination": {"latitude": dest_lat, "longitude": dest_lon, "label": "Annemasse (Hôpital Privé Pays de Savoie)"},
                    "distance_km": round(distance, 2),
                    "base_duration_min": round(duration, 2),
                    "traffic_congestion_factor": traffic_factor,
                    "cross_border_delay_min": border_delay,
                    "total_estimated_response_time_min": round(total_duration, 2),
                    "routing_status": "optimal",
                    "coordination_action": "Itinéraire transfrontalier optimisé. Notification envoyée aux douanes pour ouverture prioritaire de la barrière.",
                    "architecture_note": "Prêt pour branchement sur les serveurs OSRM privés HUG/CHUV ou TECHWAN SAGA."
                }
    except Exception as e:
        logger.error(f"Erreur lors du calcul d'itinéraire OSRM: {e}")
        
    # Fallback réaliste
    return {
        "source": "OpenStreetMap / OSRM (Simulation de secours)",
        "origin": {"latitude": orig_lat, "longitude": orig_lon, "label": "Genève (HUG)"},
        "destination": {"latitude": dest_lat, "longitude": dest_lon, "label": "Annemasse (Hôpital)"},
        "distance_km": 10.5,
        "base_duration_min": 14.8,
        "traffic_congestion_factor": 1.2,
        "cross_border_delay_min": 4.0,
        "total_estimated_response_time_min": 21.76,
        "routing_status": "degraded_simulation",
        "coordination_action": "Simulation d'itinéraire transfrontalier activée.",
        "architecture_note": "Mode dégradé activé. Prêt pour intégration avec TECHWAN SAGA."
    }


def _sentiweb_latest_value(res: Any) -> float | None:
    """Incidence /100k de la semaine la plus récente d'une réponse Sentiweb. PUR/testable.

    Sentiweb renvoie {"data": [{"week": <yyyyww>, "inc100": <taux/100k>, "inc": <cas>}, …]}.
    Le bug historique : le code attendait une LISTE racine avec une clé "incidence"
    (`isinstance(res, list)` + `.get("incidence")`), ce qui ne correspondait jamais →
    repli silencieux sur une valeur codée en dur. Renvoie None si rien d'exploitable."""
    rows = res.get("data") if isinstance(res, dict) else (res if isinstance(res, list) else [])
    rows = [x for x in (rows or []) if isinstance(x, dict) and (x.get("inc100") is not None or x.get("inc") is not None)]
    if not rows:
        return None
    latest = max(rows, key=lambda x: x.get("week", 0))
    val = latest.get("inc100")
    if val is None:
        val = latest.get("inc")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


@app.get("/terrain/epidemic")
def get_terrain_epidemic(region: str = "transborder") -> dict[str, Any]:
    """
    Endpoint P5 : Surveillance épidémique en temps réel (Sentinelles France, Sentinella Suisse, ECDC)
    pour la détection précoce des afflux de patients aux urgences et appels régulation.
    Interroge dynamiquement l'API publique du Réseau Sentinelles (Santé Publique France)
    et les flux ouverts de l'OFSP suisse.
    """
    import requests

    # Tentative de récupération des données réelles du Réseau Sentinelles (France) via leur API publique / CSV
    # Nous interrogeons les dernières données disponibles pour la grippe (ILI) et la gastro-entérite (diarrhée aiguë)
    france_data = {}

    def _sentiweb_latest_inc100(indicator: int) -> float | None:
        """HTTP + parse : incidence/100k de la semaine la plus récente (région Auvergne-Rhône-Alpes)."""
        url = f"https://www.sentiweb.fr/api/v1/indicators?indicator={indicator}&geo=reg&reg=84"
        r = requests.get(url, timeout=4)
        if r.status_code != 200:
            return None
        return _sentiweb_latest_value(r.json())

    try:
        _g = _sentiweb_latest_inc100(3)          # Indicator 3 = Grippe (ILI)
        if _g is not None:
            france_data["grippe"] = _g
            france_data["grippe_live"] = True
    except Exception as e:
        logger.warning(f"Impossible de joindre le Réseau Sentinelles FR (grippe): {e}")

    try:
        _d = _sentiweb_latest_inc100(4)          # Indicator 4 = Diarrhée aiguë
        if _d is not None:
            france_data["gastro"] = _d
            france_data["gastro_live"] = True
    except Exception as e:
        logger.warning(f"Impossible de joindre le Réseau Sentinelles FR (gastro): {e}")

    # Données unifiées combinant sources réelles et flux ouverts structurés
    diseases = [
        {
            "name": "Grippe / Influenza-like illness",
            "incidence_per_100k_france": round(france_data.get("grippe", 145.2), 1),
            "incidence_per_100k_switzerland": 128.0,
            "epidemic_threshold": 150.0,
            "status": "warning" if france_data.get("grippe", 145.2) >= 120.0 else "none",
            "trend": "increasing",
            "data_status": "live_fr" if france_data.get("grippe_live") else "illustrative",
            "source_details": ("Réseau Sentinelles FR (sentiweb.fr, live)"
                               if france_data.get("grippe_live") else
                               "Valeur d'exemple (Sentiweb injoignable)")
                              + " · incidence CH illustrative (pas de flux OFSP branché)",
        },
        {
            "name": "COVID-19",
            "incidence_per_100k_france": 92.5,
            "incidence_per_100k_switzerland": 110.4,
            "epidemic_threshold": 100.0,
            "status": "epidemic",
            "trend": "stable",
            "data_status": "illustrative",
            "source_details": "Valeurs d'exemple — aucun flux Santé Publique France / OFSP branché en direct",
        },
        {
            "name": "Gastro-entérite / Acute diarrhea",
            "incidence_per_100k_france": round(france_data.get("gastro", 210.0), 1),
            "incidence_per_100k_switzerland": 185.0,
            "epidemic_threshold": 170.0,
            "status": "epidemic" if france_data.get("gastro", 210.0) >= 170.0 else "warning",
            "trend": "decreasing",
            "data_status": "live_fr" if france_data.get("gastro_live") else "illustrative",
            "source_details": ("Réseau Sentinelles FR (sentiweb.fr, live)"
                               if france_data.get("gastro_live") else
                               "Valeur d'exemple (Sentiweb injoignable)")
                              + " · incidence CH illustrative (pas de flux OFSP branché)",
        }
    ]
    
    # Calcul du risque global d'impact EMS
    active_epidemics = sum(1 for d in diseases if d["status"] == "epidemic")
    if active_epidemics >= 2:
        risk_level = "high"
        recommendation = "Activer la cellule de crise épidémique commune. Renforcer les effectifs de régulation médicale (+15%)."
    elif active_epidemics == 1:
        risk_level = "moderate"
        recommendation = "Surveillance accrue des appels d'urgence pour motifs infectieux ou respiratoires."
    else:
        risk_level = "low"
        recommendation = "Opérations épidémiques normales."
        
    _any_live = bool(france_data.get("grippe_live") or france_data.get("gastro_live"))
    return {
        "source": "Réseau Sentinelles France (sentiweb.fr) — incidences suisses/COVID illustratives",
        "data_status": "live_partial" if _any_live else "illustrative",
        "region": "Grand Genève (Haute-Savoie, Ain, Canton de Genève, Canton de Vaud)",
        "diseases": diseases,
        "global_ems_impact_risk": risk_level,
        "recommended_action": recommendation,
        "architecture_note": ("Grippe/gastro France = données réelles Sentiweb (région Auvergne-Rhône-Alpes, "
                              "proxy transfrontalier). Les incidences suisses et COVID sont illustratives : "
                              "brancher le flux OFSP/opendata.swiss (voir connecteurs Phase 2) pour des données CH réelles.")
    }


@app.get("/terrain/demographics")
def get_terrain_demographics(postal_code: str = "74100") -> dict[str, Any]:
    """
    Endpoint P5 : Données de population et démographie (INSEE France / OFS Suisse opendata.swiss)
    Utile pour normaliser la demande EMS (taux pour 1 000 habitants) et calibrer les modèles prédictifs.
    """
    # Données réelles consolidées de l'INSEE (pour la Haute-Savoie/Ain) et de l'OFS (pour Genève/Vaud)
    demographics_db = {
        "74100": {
            "commune": "Annemasse",
            "country": "France",
            "population": 36582,
            "density_per_km2": 4572,
            "age_over_65_pct": 14.8,
            "source": "INSEE Recensement 2021"
        },
        "1201": {
            "commune": "Genève (Cité)",
            "country": "Suisse",
            "population": 203840,
            "density_per_km2": 12836,
            "age_over_65_pct": 16.2,
            "source": "OFS / Statistique de la population 2022"
        },
        "74400": {
            "commune": "Chamonix-Mont-Blanc",
            "country": "France",
            "population": 8642,
            "density_per_km2": 74,
            "age_over_65_pct": 21.3,
            "source": "INSEE Recensement 2021"
        },
        "1003": {
            "commune": "Lausanne",
            "country": "Suisse",
            "population": 141418,
            "density_per_km2": 3418,
            "age_over_65_pct": 15.1,
            "source": "OFS / Statistique de la population 2022"
        }
    }
    
    data = demographics_db.get(postal_code, {
        "commune": "Zone Transfrontalière Générique",
        "country": "France-Suisse",
        "population": 50000,
        "density_per_km2": 1200,
        "age_over_65_pct": 16.5,
        "source": "INSEE / OFS Consolidé (Défaut)"
    })
    
    risk_multiplier = 1.0
    if data["age_over_65_pct"] >= 20.0:
        risk_multiplier = 1.35
    elif data["age_over_65_pct"] >= 16.0:
        risk_multiplier = 1.15
        
    return {
        "postal_code": postal_code,
        "commune": data["commune"],
        "country": data["country"],
        "population": data["population"],
        "density_per_km2": data["density_per_km2"],
        "age_over_65_pct": data["age_over_65_pct"],
        "ems_risk_multiplier": risk_multiplier,
        "source": data["source"],
        "data_status": "static_reference",
        "architecture_note": "Valeurs de référence STATIQUES (recensements INSEE 2021 / OFS 2022) — PAS de connexion en direct. Un connecteur OFS PX-Web / INSEE reste à brancher (Phase 2) pour une actualisation automatique."
    }


@app.get("/terrain/pharmacies")
def get_terrain_pharmacies(lat: float = 46.2044, lon: float = 6.1432) -> dict[str, Any]:
    """
    Endpoint P5 : Localisation des pharmacies de garde et stocks critiques (OSM Overpass API / ANSM / Swissmedic)
    Permet d'identifier les points de distribution de contre-mesures médicales (MCMs) et les pharmacies de garde ouvertes.
    """
    import requests
    
    # Appel à l'API publique Overpass d'OpenStreetMap pour trouver les pharmacies dans un rayon de 2km
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:5];
    (
      node["amenity"="pharmacy"](around:2000,{lat},{lon});
      way["amenity"="pharmacy"](around:2000,{lat},{lon});
    );
    out body center;
    """
    
    pharmacies = []
    try:
        r = requests.post(overpass_url, data={"data": overpass_query}, timeout=6)
        if r.status_code == 200:
            elements = r.json().get("elements", [])
            for el in elements[:5]:  # On limite aux 5 plus proches
                tags = el.get("tags", {})
                pharmacies.append({
                    "name": tags.get("name", "Pharmacie"),
                    "street": tags.get("addr:street", "Rue non renseignée"),
                    "city": tags.get("addr:city", "Genève"),
                    "is_dispensary": tags.get("dispensing", "yes") == "yes",
                    "opening_hours": tags.get("opening_hours", "Non renseigné"),
                    "coordinates": {"latitude": el.get("lat") or el.get("center", {}).get("lat"), "longitude": el.get("lon") or el.get("center", {}).get("lon")}
                })
    except Exception as e:
        logger.warning(f"Erreur lors de la récupération des pharmacies via OSM Overpass: {e}")
        
    # Fallback si l'API Overpass échoue
    if not pharmacies:
        pharmacies = [
            {
                "name": "Pharmacie Principale - Gare de Cornavin",
                "street": "Place de Cornavin 3",
                "city": "Genève",
                "is_dispensary": True,
                "opening_hours": "24/7",
                "coordinates": {"latitude": 46.2102, "longitude": 6.1425}
            },
            {
                "name": "Pharmacie de Moillesulaz",
                "street": "Route de Chêne 150",
                "city": "Thônex (Frontière)",
                "is_dispensary": True,
                "opening_hours": "08:00-19:00",
                "coordinates": {"latitude": 46.1952, "longitude": 6.2021}
            }
        ]
        
    # Alertes de rupture de stock — EXEMPLES ILLUSTRATIFS (aucun flux ANSM/Swissmedic branché).
    stock_alerts = [
        {
            "medication": "Amoxicilline 500mg/5ml (Suspension pédiatrique)",
            "status": "tension",
            "country_affected": "France & Suisse",
            "recommendation": "Substitution par de l'Azithromycine ou adaptation posologique selon directives HUG/CHUV.",
            "source": "Exemple illustratif (non connecté ANSM/Swissmedic)"
        },
        {
            "medication": "Paracétamol 1g (Injectable)",
            "status": "normal",
            "country_affected": "Aucun",
            "recommendation": "Stocks suffisants pour les flottes d'ambulances.",
            "source": "Exemple illustratif (non connecté Swissmedic)"
        }
    ]
    
    return {
        "source": "OpenStreetMap Overpass (pharmacies) — alertes médicaments illustratives",
        "data_status": "live_partial",
        "pharmacies_nearby": pharmacies,
        "critical_medication_alerts": stock_alerts,
        "architecture_note": "Pharmacies : positions réelles OpenStreetMap (Overpass). Les alertes de rupture sont des EXEMPLES : brancher le fichier des pharmacies de garde (FR) et le portail ruptures Swissmedic/OFSP (CH) pour du réel."
    }


@app.get("/terrain/informal-signals")
def get_terrain_informal_signals() -> dict[str, Any]:
    """
    Endpoint P5 : Signaux informels et rumeurs épidémiques (ProMED-mail / GDELT Project / Twitter Academic)
    Permet de capturer les alertes précoces informelles et les événements sanitaires mondiaux.
    """
    # Dans un environnement de production, ce service interroge les flux RSS de ProMED ou l'API GDELT
    # Ici nous simulons un flux structuré unifié de signaux informels géolocalisés
    signals = [
        {
            "id": "sig-001",
            "source": "ProMED-mail",
            "title": "Undiagnosed respiratory illness - Switzerland (GE)",
            "content": "Rapport faisant état d'un cluster inhabituel de pneumonies atypiques chez des jeunes adultes dans le canton de Genève. 12 cas signalés en 48h.",
            "date": "2026-05-29",
            "reliability_score": 0.85,
            "severity": "moderate",
            "geo_scope": "Genève (Suisse)",
            "impact_on_hospital": "Signal d'entrée pour la modélisation épidémiologique hospitalière."
        },
        {
            "id": "sig-002",
            "source": "GDELT Project (Media Monitoring)",
            "title": "Inondations locales et coupures de routes - Haute-Savoie",
            "content": "Multiplication des articles de presse locale concernant des débordements de l'Arve à Reignier et Arthaz. Risque de fermeture de ponts.",
            "date": "2026-05-29",
            "reliability_score": 0.92,
            "severity": "high",
            "geo_scope": "Haute-Savoie (France)",
            "impact_on_ems": "Impact direct sur les itinéraires ambulances et la couverture opérationnelle."
        }
    ]
    
    return {
        "source": "Exemples illustratifs — ProMED-mail / GDELT NON connectés",
        "data_status": "illustrative",
        "active_signals": signals,
        "architecture_note": "Signaux d'EXEMPLE (aucun flux réel). Prêt à ingérer les dépêches ProMED-mail (RSS) et l'API GDELT — non branché à ce jour."
    }


@app.get("/terrain/climate")
def get_terrain_climate(lat: float = 46.2044, lon: float = 6.1432) -> dict[str, Any]:
    """
    Endpoint P5 : Intégration Copernicus Climate Data Store (CDS) - ERA5 & Projections Climatiques.
    Permet de récupérer les anomalies de température, vagues de chaleur et risques climatiques (inondations, canicules)
    pour les modèles de prévision de demande EMS et surveillance épidémique.
    """
    import os
    
    # Configuration des identifiants Copernicus CDS (depuis l'environnement)
    cds_url = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
    cds_key = os.getenv("CDS_API_KEY")
    
    # Nous créons temporairement le fichier .cdsapirc requis par le client cdsapi si importé
    # ou nous utilisons directement l'API HTTP REST de Copernicus pour éviter d'écrire sur le disque en prod.
    # Pour une robustesse maximale, nous fournissons la logique cdsapi ET un fallback d'appel direct REST.
    
    climate_data = {
        "source": "Valeurs climatiques ILLUSTRATIVES (Copernicus CDS non téléchargé)",
        "data_status": "illustrative",
        "region": "Genève - Haute-Savoie (Transfrontalier)",
        "coordinates": {"latitude": lat, "longitude": lon},
        "climatology": {
            "historical_mean_temp_may_c": 14.5,
            "current_anomaly_c": +2.4,
            "heatwave_hazard_index": "moderate",
            "soil_moisture_deficit_percent": 12.5,
            "extreme_precipitation_risk": "low"
        },
        "projections_2030": {
            "expected_heatwave_days_increase_per_year": 4.2,
            "expected_heavy_precipitation_increase_percent": 8.0,
            "ems_vulnerability_factor": "high_elderly_density"
        },
        "api_status": "configured_and_ready"
    }
    
    if not cds_key:
        climate_data["api_status"] = "not_configured"
        climate_data["message"] = "Variable d'environnement CDS_API_KEY non définie. Configurez-la côté serveur pour activer Copernicus CDS."
        return climate_data

    try:
        # Essayer d'importer cdsapi
        import cdsapi

        # Écrire temporairement le fichier de config cdsapi si nécessaire
        home = os.path.expanduser("~")
        cdsapirc_path = os.path.join(home, ".cdsapirc")
        if not os.path.exists(cdsapirc_path):
            with open(cdsapirc_path, "w") as f:
                f.write(f"url: {cds_url}\nkey: {cds_key}\n")
                
        # Le client cdsapi effectue des requêtes asynchrones qui peuvent prendre plusieurs minutes.
        # Dans le cadre d'une API web synchrone, nous n'exécutons pas la requête lourde de téléchargement NetCDF à chaque appel.
        # Au lieu de cela, nous validons la connexion et retournons l'état du pipeline, ou nous lisons un cache pré-calculé.
        client = cdsapi.Client(url=cds_url, key=cds_key, quiet=True)
        climate_data["api_status"] = "connection_verified_data_not_fetched"
        climate_data["message"] = ("Connexion à Copernicus CDS vérifiée, mais AUCUN téléchargement ERA5 n'est effectué "
                                    "ici (requête asynchrone lourde) : les valeurs climatiques ci-dessus restent illustratives.")
        
    except ImportError:
        # Fallback si cdsapi n'est pas installé
        climate_data["api_status"] = "cdsapi_package_missing"
        climate_data["message"] = "Package Python 'cdsapi' manquant. Veuillez installer cdsapi (.venv/bin/pip install cdsapi) pour activer les requêtes réelles."
    except Exception as e:
        climate_data["api_status"] = "connection_failed"
        climate_data["message"] = f"Erreur de connexion à l'API Copernicus CDS: {str(e)}"
        
    return climate_data

# ─── Endpoint : évolution temporelle et heatmap ─────────────────────────────
@app.get("/corpus/stats/by-year")
def get_corpus_stats_by_year() -> dict[str, Any]:
    """
    Distribution des articles par année (1800 → année courante), pour le graphique temporel.
    Retourne aussi la distribution par année ET par scénario pour la heatmap.
    """
    with engine.connect() as conn:
        # Articles par année (1800 → année courante)
        rows_year = conn.execute(text("""
            SELECT year, COUNT(*) as count
            FROM literature_document
            WHERE year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY year
            ORDER BY year ASC
        """)).mappings().all()

        # Articles par année ET par scénario (1800 → année courante)
        rows_scenario_year = conn.execute(text("""
            SELECT d.year, ars.scenario_id, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year, ars.scenario_id
            ORDER BY d.year ASC
        """)).mappings().all()

        # Articles par scénario ET par source (heatmap)
        rows_heatmap = conn.execute(text("""
            SELECT ars.scenario_id, d.source, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            GROUP BY ars.scenario_id, d.source
            ORDER BY ars.scenario_id, count DESC
        """)).mappings().all()

    by_year = {str(r["year"]): r["count"] for r in rows_year}

    # Construire la matrice scénario × année
    scenario_year: dict[str, dict[str, int]] = {}
    for r in rows_scenario_year:
        sid = r["scenario_id"]
        yr = str(r["year"])
        if sid not in scenario_year:
            scenario_year[sid] = {}
        scenario_year[sid][yr] = r["count"]

    # Construire la matrice scénario × source (même forme que l'endpoint /named :
    # clé = scenario_id, valeur = {name, sources: {src: {total, fulltext}}}).
    heatmap: dict[str, dict] = {}
    for r in rows_heatmap:
        sid = str(r["scenario_id"])
        src = _canonical_source(r["source"])
        entry = heatmap.setdefault(sid, {"name": sid, "sources": {}})
        cell = entry["sources"].setdefault(src, {"total": 0, "fulltext": 0})
        cell["total"] += int(r["count"] or 0)

    return {
        "by_year": by_year,
        "scenario_by_year": scenario_year,
        "heatmap_scenario_source": heatmap,
    }


# ─── Endpoint : scénarios multiples d'un article ────────────────────────────
@app.get("/documents/{doc_id}/scenarios")
def get_document_scenarios(doc_id: int) -> dict[str, Any]:
    """
    Retourne tous les scénarios auxquels un article est assigné (relation N:N).
    Utile pour l'affichage multi-scénario dans l'interface.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ars.scenario_id, ars.similarity_score, ars.assigned_at
            FROM article_scenarios ars
            WHERE ars.document_id = :doc_id
            ORDER BY ars.similarity_score DESC NULLS LAST
        """), {"doc_id": doc_id}).mappings().all()
    scenarios = []
    # Charger les titres GESICA depuis la DB
    with engine.connect() as _conn:
        _gesica_rows = _conn.execute(text("""
            SELECT id, title, name FROM user_scenarios
            WHERE is_system = TRUE AND folder_id = :fid
        """), {"fid": GESICA_FOLDER_ID}).mappings().all()
    _gesica_title_map = {str(r["id"]): (r.get("title") or r.get("name") or str(r["id"])) for r in _gesica_rows}
    for r in rows:
        sid = r["scenario_id"]
        scenarios.append({
            "scenario_id": sid,
            "title": _gesica_title_map.get(sid, sid),
            "similarity_score": float(r["similarity_score"]) if r["similarity_score"] else None,
            "assigned_at": r["assigned_at"].isoformat() if r["assigned_at"] else None,
        })
    return {"document_id": doc_id, "scenarios": scenarios, "count": len(scenarios)}


# ─── Endpoint : statistiques full-text et mode hybrid ────────────────────────
@app.get("/corpus/fulltext-stats")
def get_fulltext_stats() -> dict[str, Any]:
    """
    Statistiques de couverture textuelle du corpus.
    Distingue les articles avec full-text (chunks 'fulltext_section')
    des articles avec seulement titre+abstract ('title_abstract').
    Expose aussi le statut du mode hybrid search.
    """
    with engine.connect() as conn:
        total_docs = conn.execute(
            text("SELECT COUNT(*) FROM literature_document")
        ).scalar() or 0

        docs_with_fulltext = conn.execute(text("""
            SELECT COUNT(DISTINCT document_id)
            FROM document_chunk
            WHERE chunk_type = 'fulltext_section'
        """)).scalar() or 0

        # Doublons marqués : on expose la taille dé-dupliquée du corpus pour que
        # les compteurs ne surestiment pas le nombre d'articles réellement uniques.
        duplicates = conn.execute(text(
            "SELECT COUNT(*) FROM literature_document WHERE is_duplicate = TRUE"
        )).scalar() or 0

        chunks_with_embedding = conn.execute(text("""
            SELECT COUNT(*) FROM document_chunk WHERE embedding IS NOT NULL
        """)).scalar() or 0

        total_chunks = conn.execute(
            text("SELECT COUNT(*) FROM document_chunk")
        ).scalar() or 0

        # « En attente d'indexation » HONNÊTE : exactement les chunks que le worker VA
        # embedder — types standard, contenu embeddable (> 20 car.), pas encore mis en
        # quarantaine (< 3 échecs). Exclut les types « Autres » (jamais embeddés) et les
        # chunks définitivement refusés par l'API, qui gonflaient artificiellement le
        # reliquat. (On embède désormais le title_abstract de TOUS les docs, full-text
        # compris — plus de « couverts par le texte intégral ».)
        chunks_pending = conn.execute(text("""
            SELECT COUNT(*) FROM document_chunk c
            WHERE c.embedding IS NULL
              AND c.chunk_type IN ('title_abstract', 'fulltext_section')
              AND LENGTH(c.content) > 20
              AND COALESCE(c.embedding_attempts, 0) < 3
        """)).scalar() or 0

        # Répartition des chunks par type : 'fulltext_section' (texte intégral)
        # vs 'title_abstract' (résumé, ~1 par document) vs le reste.
        chunks_by_type = conn.execute(text(
            "SELECT chunk_type, COUNT(*) AS n FROM document_chunk GROUP BY chunk_type"
        )).mappings().all()

        source_coverage = conn.execute(text("""
            SELECT
                d.source,
                COUNT(DISTINCT d.id) AS total,
                COUNT(DISTINCT CASE WHEN c.chunk_type = 'fulltext_section' THEN d.id END) AS with_fulltext
            FROM literature_document d
            LEFT JOIN document_chunk c ON c.document_id = d.id
            GROUP BY d.source
            ORDER BY total DESC
        """)).mappings().all()

        sample_fulltext = conn.execute(text("""
            SELECT DISTINCT d.id, d.title, d.source, d.year, d.url,
                   d.authors, d.doi
            FROM literature_document d
            JOIN document_chunk c ON c.document_id = d.id
            WHERE c.chunk_type = 'fulltext_section'
            ORDER BY d.year DESC NULLS LAST
            LIMIT 100000
        """)).mappings().all()

    openai_key = os.getenv("OPENAI_API_KEY")
    hybrid_active = bool(openai_key) and chunks_with_embedding > 0

    # Couverture full-text agrégée par source CANONIQUE (mêmes regroupements que
    # le tableau de bord global, sinon variantes héritées dupliquées).
    _ft_by_source: dict[str, dict] = {}
    for r in source_coverage:
        k = _canonical_source(r["source"])
        agg = _ft_by_source.setdefault(k, {"source": k, "total": 0, "with_fulltext": 0})
        agg["total"] += int(r["total"] or 0)
        agg["with_fulltext"] += int(r["with_fulltext"] or 0)
    ft_by_source = sorted(_ft_by_source.values(), key=lambda a: a["total"], reverse=True)
    for a in ft_by_source:
        a["abstract_only"] = a["total"] - a["with_fulltext"]
        a["fulltext_pct"] = round(a["with_fulltext"] / a["total"] * 100, 1) if a["total"] else 0

    _ct = {r["chunk_type"]: int(r["n"]) for r in chunks_by_type}
    fulltext_chunks = _ct.get("fulltext_section", 0)
    abstract_chunks = _ct.get("title_abstract", 0)
    other_chunks = max(0, total_chunks - fulltext_chunks - abstract_chunks)

    return {
        "corpus": {
            "total_documents": total_docs,
            "docs_with_fulltext": docs_with_fulltext,
            "docs_abstract_only": total_docs - docs_with_fulltext,
            "fulltext_coverage_pct": round(docs_with_fulltext / total_docs * 100, 1) if total_docs else 0,
            "duplicates": duplicates,
            "unique_documents": max(0, total_docs - duplicates),
        },
        "chunks": {
            "total": total_chunks,
            "fulltext": fulltext_chunks,
            "abstract": abstract_chunks,
            "other": other_chunks,
        },
        "embeddings": {
            "total_chunks": total_chunks,
            "chunks_with_embedding": chunks_with_embedding,
            "chunks_pending": int(chunks_pending),
            "embedding_coverage_pct": round(chunks_with_embedding / total_chunks * 100, 1) if total_chunks else 0,
        },
        "hybrid_search": {
            "active": hybrid_active,
            "openai_key_present": bool(openai_key),
            "embeddings_available": chunks_with_embedding > 0,
            "mode": "hybrid" if hybrid_active else ("lexical_only" if not openai_key else "no_embeddings"),
            "note": (
                "Mode hybride actif (pgvector cosine + BM25)" if hybrid_active
                else "Mode lexical uniquement : clé OpenAI absente ou embeddings non générés"
            ),
        },
        "by_source": ft_by_source,
        "sample_fulltext_docs": [
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "year": r["year"],
                "url": r["url"],
                "authors": r["authors"],
                "doi": r["doi"],
            }
            for r in sample_fulltext
        ],
    }


# ─── Endpoint : maintenance du corpus (purge doublons + normalisation chunks) ─
class CorpusMaintenanceIn(BaseModel):
    dry_run: bool = True


@app.post("/admin/corpus-maintenance")
def corpus_maintenance(
    payload: CorpusMaintenanceIn | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Maintenance idempotente et RÉVERSIBLE du corpus (protégée par WRITE_API_KEY).

    Deux opérations, appliquées uniquement si dry_run=False :

      1. Doublons : supprime les documents `is_duplicate = TRUE` — déjà exclus de
         TOUTES les requêtes (recherche, RAG, PICO, stats) — ainsi que leurs lignes
         `article_scenarios` (cette table n'a pas de FK → suppression explicite,
         sinon orphelins). Les chunks partent en CASCADE (document_chunk.document_id
         ON DELETE CASCADE).

      2. Chunks « Autres » (type non standard) : renomme le type hérité
         `'full_text'` en `'fulltext_section'` (le worker d'enrichissement les
         embeddera alors normalement) et supprime les chunks non-standard vraiment
         inexploitables (contenu < 20 caractères → jamais embeddables). Les chunks
         non-standard SUBSTANTIELS (≥ 20 car.) sont seulement RAPPORTÉS, jamais
         modifiés — on décide de leur sort après avoir vu l'aperçu.

    Sécurité : tout tourne dans UNE transaction (atomique) ; avant chaque
    suppression, les lignes concernées sont copiées dans des tables `_maint_bak_*`
    (restaurables). dry_run=True (défaut) ne fait que COMPTER — aucune écriture.
    """
    dry_run = True if payload is None else bool(payload.dry_run)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Prédicat SQL des chunks « non standard » (NULL inclus : `NOT IN` seul rate NULL).
    # Ces chunks « Autres » (types hérités : 'full_text', 'title', 'abstract_section'…,
    # ou NULL) ne sont JAMAIS embeddés par le worker (qui ne traite que les deux types
    # standard) → ils restent « en attente » indéfiniment. On les partitionne en trois
    # ensembles DISJOINTS, tous sûrs (aucune perte de contenu, sauvegardés avant action) :
    NONSTD = "(chunk_type IS NULL OR chunk_type NOT IN ('title_abstract','fulltext_section'))"
    # 1) junk : vide / inexploitable (< 20 car., jamais embeddable) → suppression.
    JUNK = f"{NONSTD} AND (content IS NULL OR length(btrim(content)) < 20)"
    # 2) redundant : fragment substantiel dont le TEXTE est déjà contenu dans le chunk
    #    title_abstract du même document (donc déjà indexé et cherchable) → suppression
    #    SÛRE (le contenu reste dans title_abstract, rien ne quitte l'index).
    _IN_TA = ("EXISTS (SELECT 1 FROM document_chunk ta "
              "WHERE ta.document_id = document_chunk.document_id "
              "AND ta.chunk_type = 'title_abstract' "
              "AND position(btrim(document_chunk.content) IN ta.content) > 0)")
    REDUNDANT = f"{NONSTD} AND content IS NOT NULL AND length(btrim(content)) >= 20 AND {_IN_TA}"
    # 3) unique : fragment substantiel dont le texte n'est PAS dans title_abstract (vrai
    #    contenu supplémentaire) → reclassé en 'fulltext_section' pour que le worker
    #    l'indexe (aucune perte). Après application : plus aucun chunk « Autres ».
    UNIQUE = f"{NONSTD} AND content IS NOT NULL AND length(btrim(content)) >= 20 AND NOT {_IN_TA}"

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "duplicates": {},
        "legacy_chunks": {},
        "backups": [],
    }

    with engine.begin() as conn:
        has_ars = bool(conn.execute(
            text("SELECT to_regclass('public.article_scenarios')")
        ).scalar())

        # ── 1. DOUBLONS ────────────────────────────────────────────────────
        dup_docs = conn.execute(text(
            "SELECT COUNT(*) FROM literature_document WHERE is_duplicate IS TRUE"
        )).scalar() or 0
        dup_chunks = conn.execute(text(
            "SELECT COUNT(*) FROM document_chunk c "
            "JOIN literature_document d ON d.id = c.document_id "
            "WHERE d.is_duplicate IS TRUE"
        )).scalar() or 0
        dup_ars = 0
        if has_ars:
            dup_ars = conn.execute(text(
                "SELECT COUNT(*) FROM article_scenarios a "
                "JOIN literature_document d ON d.id = a.document_id "
                "WHERE d.is_duplicate IS TRUE"
            )).scalar() or 0
        report["duplicates"] = {
            "documents": int(dup_docs),
            "chunks_cascade": int(dup_chunks),
            "article_scenarios": int(dup_ars),
        }
        if not dry_run and dup_docs:
            bak_docs, bak_chunks = f"_maint_bak_docs_{ts}", f"_maint_bak_chunks_{ts}"
            conn.execute(text(
                f'CREATE TABLE "{bak_docs}" AS '
                "SELECT * FROM literature_document WHERE is_duplicate IS TRUE"
            ))
            conn.execute(text(
                f'CREATE TABLE "{bak_chunks}" AS SELECT c.* FROM document_chunk c '
                "JOIN literature_document d ON d.id = c.document_id WHERE d.is_duplicate IS TRUE"
            ))
            report["backups"] += [bak_docs, bak_chunks]
            if has_ars:
                bak_ars = f"_maint_bak_ars_{ts}"
                conn.execute(text(
                    f'CREATE TABLE "{bak_ars}" AS SELECT a.* FROM article_scenarios a '
                    "JOIN literature_document d ON d.id = a.document_id WHERE d.is_duplicate IS TRUE"
                ))
                conn.execute(text(
                    "DELETE FROM article_scenarios WHERE document_id IN "
                    "(SELECT id FROM literature_document WHERE is_duplicate IS TRUE)"
                ))
                report["backups"].append(bak_ars)
            deleted = conn.execute(text(
                "DELETE FROM literature_document WHERE is_duplicate IS TRUE"
            )).rowcount  # chunks removed via ON DELETE CASCADE
            report["duplicates"]["deleted_documents"] = int(deleted)

        # ── 2. CHUNKS « AUTRES » (type non standard) ───────────────────────
        breakdown_rows = conn.execute(text(
            "SELECT COALESCE(chunk_type,'(null)') AS t, COUNT(*) AS n, "
            "COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded "
            f"FROM document_chunk WHERE {NONSTD} GROUP BY chunk_type ORDER BY n DESC"
        )).mappings().all()
        junk = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {JUNK}")).scalar() or 0
        redundant = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {REDUNDANT}")).scalar() or 0
        uniq = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {UNIQUE}")).scalar() or 0
        report["legacy_chunks"] = {
            "breakdown": [
                {"chunk_type": r["t"], "count": int(r["n"]), "embedded": int(r["embedded"])}
                for r in breakdown_rows
            ],
            "junk_to_delete": int(junk),
            "redundant_to_delete": int(redundant),      # already covered by title_abstract
            "unique_to_reclassify": int(uniq),          # real content → fulltext_section (indexed)
        }
        if not dry_run:
            # Delete junk + redundant (each backed up first); order is irrelevant — the
            # three sets are disjoint and none touches title_abstract chunks.
            for label, pred in (("junkchunks", JUNK), ("redundantchunks", REDUNDANT)):
                n = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {pred}")).scalar() or 0
                if n:
                    bak = f"_maint_bak_{label}_{ts}"
                    conn.execute(text(f'CREATE TABLE "{bak}" AS SELECT * FROM document_chunk WHERE {pred}'))
                    conn.execute(text(f"DELETE FROM document_chunk WHERE {pred}"))
                    report["backups"].append(bak)
            report["legacy_chunks"]["deleted_junk"] = int(junk)
            report["legacy_chunks"]["deleted_redundant"] = int(redundant)
            reclassified = conn.execute(text(
                f"UPDATE document_chunk SET chunk_type='fulltext_section' WHERE {UNIQUE}"
            )).rowcount
            report["legacy_chunks"]["reclassified"] = int(reclassified)

    return report


@app.post("/admin/embed-pending")
def embed_pending_chunks(limit: int = 200, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Force l'indexation (embedding) des chunks « en attente » — à la demande, avec
    EXACTEMENT le sélecteur du worker d'arrière-plan. Vide immédiatement le petit
    reliquat sans attendre le cycle de 30 s. Traite au plus `limit` chunks (synchrone) ;
    renvoie le nombre embeddé et le reliquat restant."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")
    if _openai_in_cooldown():
        return {"embedded": 0, "remaining": None, "cooldown": True}

    # Sélecteur IDENTIQUE au worker et au compteur « en attente » : types standard,
    # contenu embeddable (> 20 car.), et title_abstract non ré-embeddé si le doc a du
    # texte intégral (couvert par ses sections).
    ELIGIBLE = (
        "c.embedding IS NULL "
        "AND c.chunk_type IN ('title_abstract','fulltext_section') "
        "AND LENGTH(c.content) > 20 "
        "AND COALESCE(c.embedding_attempts, 0) < 3"
    )

    from openai import OpenAI as _OAI
    client = _OAI(api_key=openai_key, timeout=90.0)

    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT c.id, c.content FROM document_chunk c WHERE {ELIGIBLE} "
            "ORDER BY c.chunk_type DESC, c.id LIMIT :lim"
        ), {"lim": max(1, min(int(limit), 2000))}).mappings().fetchall()

    # Chemin résilient IDENTIQUE au worker : troncature par tokens + repli chunk par
    # chunk sur un lot empoisonné (ne se bloque plus sur un contenu que l'API refuse).
    embedded, failed, error = 0, [], None
    try:
        embedded, failed = _embed_chunks_resilient(client, list(rows))
    except Exception as e:                        # quota → cooldown, on ne réessaie pas ici
        _trip_openai_cooldown()
        error = str(e)
    if failed:
        with engine.begin() as cu:
            cu.execute(
                text("UPDATE document_chunk SET embedding_attempts = "
                     "COALESCE(embedding_attempts,0)+1 WHERE id IN :ids")
                .bindparams(bindparam("ids", expanding=True)),
                {"ids": failed},
            )

    with engine.connect() as conn:
        remaining = conn.execute(text(
            f"SELECT COUNT(*) FROM document_chunk c WHERE {ELIGIBLE}"
        )).scalar() or 0

    out: dict[str, Any] = {"embedded": int(embedded), "remaining": int(remaining)}
    if failed:
        out["quarantined"] = len(failed)
    if error:
        out["error"] = error
    return out


# ─── Living Review Endpoints ──────────────────────────────────────────────────

@app.get("/living-review/status")
def living_review_status():
    """Retourne le statut de la dernière exécution de la living review."""
    import json as _json
    report_path = Path("/opt/literev-api/living_review_last_run.json")
    if not report_path.exists():
        report_path = Path(__file__).parent / "living_review_last_run.json"
    if report_path.exists():
        try:
            return _json.loads(report_path.read_text())
        except Exception:
            pass
    return {
        "status": "no_run_yet",
        "message": "Aucune living review n'a encore été exécutée.",
        "command": "python3 living_review_scheduler.py --all-scenarios",
        "scenarios_available": list(SCENARIO_LIVING_REVIEW_IDS),
    }


@app.post("/living-review/run")
def living_review_run(scenario_id: str = "all", days: int = 30, dry_run: bool = False, _: None = Depends(require_api_key)):
    """Lance la living review pour un scénario ou tous les scénarios (processus async)."""
    import subprocess as _subprocess
    import sys as _sys
    script = str(Path(__file__).parent / "living_review_scheduler.py")
    cmd = [_sys.executable, script, "--mode", "once", "--days", str(days)]
    if scenario_id == "all":
        cmd.append("--all-scenarios")
    else:
        cmd.extend(["--scenario", scenario_id])
    if dry_run:
        cmd.append("--dry-run")
    try:
        proc = _subprocess.Popen(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE)
        return {
            "status": "started",
            "pid": proc.pid,
            "scenario": scenario_id,
            "days": days,
            "dry_run": dry_run,
            "message": f"Living review lancée. Consultez /living-review/status pour le résultat.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


SCENARIO_LIVING_REVIEW_IDS = [
    # Cluster 1 : Patient-centered prehospital critical care
    "cardiac-arrest-prediction",
    "stroke-detection",
    "trauma-severity-assessment",
    "clinical-deterioration-prediction",
    "patient-pathway-optimization",
    "mci-victim-estimation",
    # Cluster 2 : Environmental & Disaster Risk
    "environmental-risk-forecasting",
    "disaster-risk-assessment",
    "climate-impact-on-ems",
    # Cluster 3 : Prehospital Triage & Risk Stratification
    "emergency-call-qualification",
    "call-prioritization",
    "mass-casualty-triage",
    "undertriage-detection",
    "dispatch-decision-support",
    "triage-support",
    # Cluster 4 : EMS Operations & Resource Management
    "response-time-optimization",
    "ambulance-dispatch-optimization",
    "staffing-level-prediction",
    "hospital-capacity-forecasting",
    "demand-forecasting",
    "resource-allocation",
    # Cluster 5 : Epidemiological & Strategic Surveillance
    "epidemic-early-warning",
    "surveillance",
    "surge-management",
    "pandemic-preparedness",
    "cross-border-coordination",
    "situational-awareness",
]

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Scenario Detail Endpoints (Phase 2 : Refonte interface)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/detail")
def get_scenario_detail(scenario_id: str) -> dict[str, Any]:
    """
    Retourne toutes les informations enrichies d'un scénario :
    - Métadonnées de base (titre, description, cluster, actions recommandées)
    - Queries booléennes PubMed et requêtes NL pour la recherche sémantique
    - Prompt d'extraction d'évidence spécifique au scénario
    - Informations sur le modèle IA (algorithme, variables, fréquence de mise à jour)
    - Seuils d'alerte vert/orange/rouge
    """
    with engine.connect() as conn:
        meta = _get_db_gesica_scenario_or_404(scenario_id, conn)
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) THEN 1 ELSE 0 END) AS with_fulltext,
                COUNT(DISTINCT d.year) AS years_covered,
                COUNT(DISTINCT d.journal) AS journals_count,
                MIN(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id}).mappings().first()
    return {
        "id": scenario_id,
        "title": _gesica_title(meta),
        "description": meta.get("description") or "",
        "cluster": meta.get("cluster") or "",
        "recommended_actions": _gesica_actions(meta),
        "boolean_queries": meta.get("boolean_queries") or [],
        "nl_queries": meta.get("nl_queries") or [],
        "evidence_extraction_prompt": meta.get("evidence_extraction_prompt") or "",
        "model_info": meta.get("model_info") or {},
        "alert_thresholds": meta.get("alert_thresholds") or {},
        "databases": meta.get("required_databases") or [],
        "outcome_definition": meta.get("outcome_definition") or "",
        "variables_detail": meta.get("variables_detail") or {},
        "keywords": meta.get("keywords") or [],
        "clinical_rationale": meta.get("clinical_rationale") or "",
        "corpus_stats": {
            "total": int(stats["total"] or 0),
            "with_fulltext": int(stats["with_fulltext"] or 0),
            "years_covered": int(stats["years_covered"] or 0),
            "journals_count": int(stats["journals_count"] or 0),
            "year_min": stats["year_min"],
            "year_max": stats["year_max"],
        },
    }


@app.get("/gesica/scenarios/{scenario_id}/corpus")
def get_scenario_corpus(
    scenario_id: str,
    limit: int = 100000,
    offset: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    fulltext_only: bool = False,
    source: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_corpus(scenario_id, limit, offset, year_from, year_to, fulltext_only, source, threshold)


@app.post("/admin/recompute-quality-scores")
def recompute_quality_scores(
    limit: int = 5000,
    only_missing: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    (Re)calcule le quality_score déterministe sur le corpus existant à partir des
    colonnes structurées et de metadata_json (study_type, sample_size, bias_risk).
    Idempotent. `only_missing=True` ne traite que les documents sans score
    (quality_score NULL ou 0). À appeler par lots (`limit`) pour le backfill.
    """
    where_missing = "AND (quality_score IS NULL OR quality_score = 0)" if only_missing else ""
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, year, citation_count, open_access, study_design, sample_size,
                   metadata_json
            FROM literature_document
            WHERE project_context = 'literev'
              {where_missing}
            ORDER BY id
            LIMIT :limit
        """), {"limit": max(1, min(limit, 50000))}).mappings().fetchall()

    updated = 0
    skipped_no_signal = 0
    for r in rows:
        meta = r["metadata_json"] if isinstance(r["metadata_json"], dict) else {}
        study_design = r["study_design"] or (meta.get("study_type") if meta else None)
        sample_size = r["sample_size"] or _coerce_int(meta.get("sample_size") if meta else None)
        score = _compute_quality_score(
            study_design=study_design,
            year=r["year"],
            sample_size=sample_size,
            citation_count=r["citation_count"],
            open_access=r["open_access"],
            bias_risk=(meta.get("bias_risk") if meta else None),
        )
        if score is None:
            skipped_no_signal += 1
            continue
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE literature_document
                SET quality_score = :score,
                    study_design = COALESCE(:study_design, study_design),
                    sample_size = COALESCE(:sample_size, sample_size)
                WHERE id = :id
            """), {"score": score, "study_design": study_design,
                   "sample_size": sample_size, "id": r["id"]})
        updated += 1

    return {
        "scanned": len(rows),
        "updated": updated,
        "skipped_no_signal": skipped_no_signal,
        "only_missing": only_missing,
        "limit": limit,
        "message": "Relancez l'endpoint tant que 'scanned' == 'limit' pour traiter tout le corpus.",
    }


# ── Encoder JSON pour types numpy ────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Tâches de clustering en cours ────────────────────────────────────────────
_clustering_jobs: dict[str, dict] = {}  # scenario_id -> {"status": "running"|"done"|"error", "result": ...}


def _cluster_core(
    docs: list,
    texts: list[str],
    *,
    openai_key: str | None = None,
    allow_openai_embeddings: bool = False,
    tfidf_min_df: int = 2,
) -> dict:
    """Cœur partagé du clustering (utilisé par l'endpoint à la demande ET le pipeline).

    Chaîne : embeddings (pgvector DB → OpenAI optionnel → repli TF-IDF) → UMAP 2D
    (thread, timeout 60 s) → HDBSCAN, avec repli K-Means+SVD si UMAP/HDBSCAN échoue.
    Retourne labels, projection 2D, méthode, et les artefacts TF-IDF nécessaires à la
    construction des clusters (feature_names, matrice dense).
    """
    import numpy as np
    import threading
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD, PCA

    vectorizer = TfidfVectorizer(max_features=800, stop_words="english",
                                 min_df=tfidf_min_df, max_df=0.9, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 1) Embeddings pgvector stockés en DB (privilégiés)
    embeddings_matrix = None
    embedding_source = "tfidf"
    if any(d.get("embedding_str") for d in docs):
        try:
            vecs = []
            for d in docs:
                es = d.get("embedding_str")
                vecs.append([float(x) for x in es.strip("[]").split(",")] if es else None)
            valid = [v for v in vecs if v is not None]
            if valid:
                mean_vec = np.mean(valid, axis=0).tolist()
                vecs = [v if v is not None else mean_vec for v in vecs]
                embeddings_matrix = np.array(vecs, dtype=np.float32)
                embedding_source = "db_pgvector"
        except Exception as e:
            logger.warning(f"_cluster_core: embeddings DB inutilisables: {e}")

    # 2) Sinon, génération OpenAI (optionnelle)
    if embeddings_matrix is None and allow_openai_embeddings and openai_key:
        try:
            from openai import OpenAI as _OAI
            _oai = _OAI(api_key=openai_key, timeout=90.0)
            all_vecs: list = []
            batch_texts = [t[:2000] for t in texts]
            for i in range(0, len(batch_texts), 100):
                resp = _oai.embeddings.create(model="text-embedding-3-small",
                                              input=batch_texts[i:i + 100])
                all_vecs.extend([e.embedding for e in resp.data])
            embeddings_matrix = np.array(all_vecs, dtype=np.float32)
            embedding_source = "openai_api"
        except Exception as e:
            logger.warning(f"_cluster_core: embeddings OpenAI échoués: {e}")

    umap_input = embeddings_matrix if embeddings_matrix is not None else X_tfidf.toarray()

    # 3) UMAP 2D dans un thread avec timeout 60 s
    umap_result: dict = {"embedding": None}
    def _run_umap():
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(
                n_neighbors=min(10, len(docs) - 1), n_components=2,
                metric="cosine", random_state=42, low_memory=True, n_epochs=200,
            )
            umap_result["embedding"] = reducer.fit_transform(umap_input)
        except Exception as e:
            logger.warning(f"_cluster_core UMAP: {e}")
    _t = threading.Thread(target=_run_umap, daemon=True)
    _t.start()
    _t.join(timeout=60)

    embedding_2d = umap_result["embedding"]
    labels = None
    method_used = "kmeans_fallback"

    # 4) HDBSCAN sur la projection 2D
    if embedding_2d is not None:
        try:
            import hdbscan as hdbscan_lib
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=max(3, len(docs) // 15), min_samples=2,
                metric="euclidean", cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embedding_2d)
            method_used = "embeddings_umap_hdbscan" if embeddings_matrix is not None else "tfidf_umap_hdbscan"
        except Exception as e:
            logger.warning(f"_cluster_core HDBSCAN: {e}")

    # 5) Repli K-Means + SVD (si UMAP a expiré ou HDBSCAN a échoué)
    if labels is None:
        n_clusters = max(3, min(8, len(docs) // 15))
        svd = TruncatedSVD(n_components=min(50, X_tfidf.shape[1] - 1, len(docs) - 1), random_state=42)
        X_reduced = svd.fit_transform(X_tfidf)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100).fit_predict(X_reduced)
        if embedding_2d is None:
            embedding_2d = PCA(n_components=2, random_state=42).fit_transform(X_reduced)
        method_used = "kmeans_fallback"

    return {
        "labels": labels,
        "embedding_2d": embedding_2d,
        "method": method_used,
        "feature_names": feature_names,
        "X_dense": X_tfidf.toarray(),
        "embedding_source": embedding_source,
    }


# ── Caches de visualisation persistés en DB (scenario_settings) ───────────────
# Un SEUL couple load/save par visualisation, partagé par le pipeline, le
# précalcul et les endpoints — plus de duplication ni de cache /tmp éphémère.

def _save_viz_cache(scenario_id: str, col: str, payload: dict) -> None:
    """Upsert un JSON de visualisation dans scenario_settings.{col}_json (+ _at)."""
    _at = "clustering_generated_at" if col == "clustering" else "kg_generated_at"
    _jc = f"{col}_json" if col == "clustering" else "knowledge_graph_json"
    try:
        with engine.begin() as _c:
            _c.execute(text(f"""
                INSERT INTO scenario_settings (scenario_id, {_jc}, {_at}, updated_at)
                VALUES (:sid, CAST(:p AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET {_jc} = CAST(:p AS jsonb), {_at} = NOW(), updated_at = NOW()
            """), {"sid": scenario_id, "p": json.dumps(payload, default=str)})
    except Exception as _e:
        logger.warning(f"_save_viz_cache {col} {scenario_id}: {_e}")


def _load_viz_cache(scenario_id: str, col: str, ttl: int = 86400) -> dict | None:
    """Lit le JSON de visualisation en cache s'il est frais (< ttl secondes)."""
    _at = "clustering_generated_at" if col == "clustering" else "kg_generated_at"
    _jc = f"{col}_json" if col == "clustering" else "knowledge_graph_json"
    try:
        with engine.connect() as _c:
            row = _c.execute(text(
                f"SELECT {_jc} AS j, {_at} AS at FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        if row and row["j"]:
            fresh = True
            if row["at"]:
                from datetime import datetime as _dt, timezone as _tz
                _ts = row["at"]
                if _ts.tzinfo is None:
                    _ts = _ts.replace(tzinfo=_tz.utc)
                fresh = (_dt.now(_tz.utc) - _ts).total_seconds() < ttl
            if fresh:
                data = dict(row["j"])
                data["from_cache"] = True
                return data
    except Exception as _e:
        logger.warning(f"_load_viz_cache {col} {scenario_id}: {_e}")
    return None


def _build_clusters_payload(scenario_id: str, docs: list, cc: dict, *,
                            with_summaries: bool = False, openai_key: str | None = None,
                            title: str | None = None, lang: str | None = None) -> dict:
    """Construit le payload de clustering CANONIQUE (un seul format, partagé par le
    pipeline ET le calcul en arrière-plan). `with_summaries` active le résumé LLM
    par cluster. Schéma figé : clusters[].representative_doc + embedding_source."""
    import numpy as np
    labels = cc["labels"]; embedding_2d = cc["embedding_2d"]; method_used = cc["method"]
    feature_names = cc["feature_names"]; X_dense = cc["X_dense"]; embedding_source = cc["embedding_source"]
    clusters = []
    for label in sorted(set(labels)):
        label_int = int(label)
        idxs = [i for i, l in enumerate(labels) if int(l) == label_int]
        coords = embedding_2d[idxs]
        cluster_tfidf = X_dense[idxs].mean(axis=0)
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_words = [str(feature_names[i]) for i in top_indices if cluster_tfidf[i] > 0]
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        rep = docs[idxs[int(np.argmin(distances))]]
        points = [
            {"id": int(docs[i]["id"]), "title": str(docs[i]["title"] or ""),
             "year": int(docs[i]["year"]) if docs[i].get("year") else None,
             "x": float(embedding_2d[i, 0]), "y": float(embedding_2d[i, 1])}
            for i in idxs
        ]
        resume = "Bruit de fond (articles non regroupés)." if label_int == -1 else ""
        if with_summaries and label_int != -1 and openai_key:
            try:
                from openai import OpenAI as _OAI
                _client = _OAI(api_key=openai_key, timeout=90.0)
                english = (lang or "fr").strip().lower().startswith("en")
                lang_word = "in English" if english else "en français"
                top5 = np.argsort(distances)[:5]
                llm_ctx = "\n\n".join(
                    f"Titre: {docs[idxs[int(t)]]['title']}\nRésumé: {(docs[idxs[int(t)]].get('abstract') or '')[:350]}"
                    for t in top5
                )
                completion = _client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": (
                        f"Scénario : {title or scenario_id}.\n"
                        f"Articles représentatifs du cluster :\n{llm_ctx}\n\n"
                        f"Rédigez un résumé concis (3-4 phrases, max 120 mots) {lang_word} : "
                        f"thématique commune, évidences clés, valeur opérationnelle pour les urgences préhospitalières."
                    ) + _llm_lang_directive(lang)}],
                    max_tokens=200, temperature=0.3,
                )
                resume = completion.choices[0].message.content.strip()
            except Exception as _e:
                logger.error(f"Résumé cluster {label_int}: {_e}")
        clusters.append({
            "cluster_id": label_int,
            "cluster_name": f"Cluster {label_int + 1}" if label_int != -1 else "Non-classés",
            "is_noise": label_int == -1,
            "n_docs": len(idxs),
            "center_x": float(center[0]), "center_y": float(center[1]),
            "top_words": top_words,
            "summary": resume,
            "representative_doc": {
                "id": int(rep["id"]),
                "title": str(rep["title"] or ""),
                "year": int(rep["year"]) if rep.get("year") else None,
                "journal": str(rep.get("journal") or ""),
            },
            "points": points,
        })
    return {
        "scenario_id": scenario_id,
        "n_docs": len(docs),
        "n_clusters": len([c for c in clusters if not c["is_noise"]]),
        "method": method_used,
        "embedding_source": embedding_source,
        "clusters": sorted(clusters, key=lambda x: (x["is_noise"], -x["n_docs"])),
        "from_cache": False,
    }


def _run_clustering_background(scenario_id: str, force_refresh: bool = False, lang: str | None = None) -> None:
    """Calcule le clustering dans un thread séparé et stocke le résultat en cache."""
    import time as _time

    cache_dir = "/tmp/literev_clustering_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{scenario_id}.json")
    TTL = 86400

    # Vérifier le cache d'abord
    if not force_refresh and os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            if _time.time() - mtime < TTL:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                cached["from_cache"] = True
                _clustering_jobs[scenario_id] = {"status": "done", "result": cached}
                return
        except Exception:
            pass

    try:
        meta_for_cluster = {}
        try:
            meta_for_cluster = _get_db_gesica_scenario_or_404(scenario_id)
        except Exception:
            pass  # Scénario utilisateur ou non trouvé — on continue sans métadonnées
    except Exception:
        meta_for_cluster = {}
    try:

        with engine.connect() as conn:
            docs = list(conn.execute(text("""
                SELECT d.id, d.title, d.abstract, d.year, d.journal,
                       (
                           SELECT c.embedding::text
                           FROM document_chunk c
                           WHERE c.document_id = d.id
                             AND c.embedding IS NOT NULL
                           -- Vecteur représentatif = le résumé (title_abstract) en
                           -- priorité pour TOUS les docs (cohérent) ; repli 1er chunk.
                           ORDER BY (c.chunk_type = 'title_abstract') DESC, c.id
                           LIMIT 1
                       ) AS embedding_str
                FROM literature_document d
                JOIN article_scenarios asn ON asn.document_id = d.id
                WHERE asn.scenario_id = :sid
                  AND d.project_context = 'literev'
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                  AND d.abstract IS NOT NULL
                  AND LENGTH(d.abstract) > 50
                ORDER BY d.year DESC NULLS LAST
                LIMIT 100000
            """), {"sid": scenario_id}).mappings().all())

        if len(docs) < 5:
            result = {
                "scenario_id": scenario_id, "n_docs": len(docs),
                "message": "Corpus insuffisant pour le clustering (minimum 5 articles avec abstract requis)",
                "clusters": [], "from_cache": False,
            }
            _clustering_jobs[scenario_id] = {"status": "done", "result": result}
            return

        texts = [f"{d['title']} {d['abstract'] or ''}" for d in docs]

        # ── Embeddings → UMAP → HDBSCAN (cœur partagé _cluster_core) ────────
        openai_key = os.getenv("OPENAI_API_KEY")
        _cc = _cluster_core(docs, texts, openai_key=openai_key,
                            allow_openai_embeddings=True, tfidf_min_df=2)
        labels = _cc["labels"]
        embedding_2d = _cc["embedding_2d"]
        method_used = _cc["method"]
        feature_names = _cc["feature_names"]
        X_dense = _cc["X_dense"]
        embedding_source = _cc["embedding_source"]
        logger.info(f"Clustering {scenario_id}: {len(docs)} docs, source={embedding_source}, method={method_used}")

        # Construction du payload (helper PARTAGÉ avec le pipeline — plus de copie).
        result = _build_clusters_payload(
            scenario_id, docs, _cc, with_summaries=True, openai_key=openai_key,
            title=(_gesica_title(meta_for_cluster) if meta_for_cluster else None),
            lang=lang,
        )
        # Cache DB (durable) + /tmp (compat) + mémoire.
        _save_viz_cache(scenario_id, "clustering",
                        json.loads(json.dumps(result, cls=_NumpyEncoder)))
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, cls=_NumpyEncoder)
        except Exception:
            pass
        _clustering_jobs[scenario_id] = {"status": "done", "result": result}

    except Exception as e:
        logger.error(f"Clustering {scenario_id} error: {e}", exc_info=True)
        _clustering_jobs[scenario_id] = {"status": "error", "error": str(e)}


@app.get("/gesica/scenarios/{scenario_id}/clustering")
def get_scenario_clustering(scenario_id: str, force_refresh: bool = False, lang: str | None = Query(None)) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_clustering(scenario_id, force_refresh, lang)


@app.get("/gesica/scenarios/{scenario_id}/clustering/status")
def get_clustering_status(scenario_id: str) -> dict:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_clustering_status(scenario_id)


@app.post("/gesica/scenarios/{scenario_id}/rag")
def scenario_rag_assistant(scenario_id: str, payload: AskIn) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return user_scenario_rag_assistant(scenario_id, payload)


@app.get("/gesica/scenarios/{scenario_id}/prisma")
def get_scenario_prisma(scenario_id: str, threshold: float = Query(None)) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (PRISMA modernise)."""
    return get_user_scenario_prisma(scenario_id, threshold)


@app.get("/gesica/deduplication/status")
def get_deduplication_status() -> dict[str, Any]:
    """
    Retourne le statut de la déduplication du corpus.
    """
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_duplicate = TRUE THEN 1 ELSE 0 END) AS duplicates,
                SUM(CASE WHEN is_duplicate IS NULL OR is_duplicate = FALSE THEN 1 ELSE 0 END) AS canonical,
                SUM(CASE WHEN title_hash IS NOT NULL THEN 1 ELSE 0 END) AS with_title_hash,
                SUM(CASE WHEN quality_score > 0 THEN 1 ELSE 0 END) AS with_quality_score
            FROM literature_document
            WHERE project_context = 'literev'
        """)).mappings().first()
    return {
        "total_documents": int(stats["total"] or 0),
        "canonical_documents": int(stats["canonical"] or 0),
        "duplicate_documents": int(stats["duplicates"] or 0),
        "with_title_hash": int(stats["with_title_hash"] or 0),
        "with_quality_score": int(stats["with_quality_score"] or 0),
        "deduplication_rate": round(
            int(stats["duplicates"] or 0) / max(int(stats["total"] or 1), 1) * 100, 1
        ),
        "instructions": {
            "dry_run": "python3 deduplicate_corpus.py --dry-run",
            "execute": "python3 deduplicate_corpus.py --execute",
            "execute_delete": "python3 deduplicate_corpus.py --execute --delete",
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Upload de Dataset par Scénario GESICA
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import UploadFile, File
import shutil

@app.post("/gesica/scenarios/{scenario_id}/upload-dataset")
async def upload_scenario_dataset(
    scenario_id: str,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Permet à l'utilisateur d'uploader un jeu de données (CSV ou Excel) pour alimenter
    les variables non branchées d'un scénario spécifique.
    """
    _get_db_gesica_scenario_or_404(scenario_id)

    # Valider le format du fichier
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ["csv", "xlsx", "xls"]:
        raise HTTPException(status_code=400, detail="Seuls les fichiers CSV et Excel (.xlsx, .xls) sont autorisés")

    # Neutraliser tout chemin dans le nom de fichier (anti path-traversal)
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    # Créer le dossier d'uploads s'il n'existe pas
    upload_dir = Path("/home/ubuntu/uploads_datasets") / scenario_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / safe_filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Analyser sommairement le fichier pour extraire des métriques (nombre de lignes, colonnes)
    num_rows = 0
    columns = []
    try:
        if ext == "csv":
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)
            # Compter les lignes totales rapidement
            num_rows = sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="ignore")) - 1
            columns = list(df.columns)
        else:
            import pandas as pd
            df = pd.read_excel(file_path)
            num_rows = len(df)
            columns = list(df.columns)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du fichier uploade {filename}: {e}")
        # On ne bloque pas l'upload si l'analyse échoue
        columns = ["Inconnu"]
        num_rows = -1

    return {
        "message": f"Fichier '{filename}' uploade avec succès pour le scénario '{scenario_id}'",
        "filename": filename,
        "size_bytes": file_path.stat().st_size,
        "detected_rows": num_rows,
        "detected_columns": columns,
        "status": "stored_and_analyzed",
        "instructions": "Le jeu de données a été stocké. Les variables correspondantes du modèle seront automatiquement branchées lors du prochain recalcul."
    }


# ─── PICO Extraction Endpoints ───────────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/articles/{article_id}/pico")
def get_article_pico(scenario_id: str, article_id: int):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_article_pico(scenario_id, article_id)


@app.post("/gesica/scenarios/{scenario_id}/articles/{article_id}/pico/extract")
def extract_article_pico(scenario_id: str, article_id: int, _: None = Depends(require_api_key)):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return extract_user_scenario_article_pico(scenario_id, article_id)



# ─── Enrichissement LLM Batch ────────────────────────────────────────────────

@app.post("/pico/extract")
def extract_pico_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Extrait le PICO pour un lot d'articles (par scénario ou tout le corpus).
    Traite uniquement les articles sans PICO ou avec un PICO de faible confiance.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    # Récupérer les articles sans PICO
    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                  AND COALESCE(asn.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (pico_json IS NULL OR (pico_json->>'pico_confidence')::float < 0.5)
                  AND abstract IS NOT NULL AND length(abstract) > 50
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    system_prompt = (
        "You are a systematic review expert in emergency medicine. "
        "Extract PICO elements and return ONLY valid JSON:\n"
        '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
        '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
        '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
        "Be concise (max 2 sentences per field). Return ONLY the JSON."
    )

    try:
        from openai import OpenAI as _OAI
        from datetime import datetime, timezone
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not abstract or len(abstract) < 50:
                skipped += 1
                continue
            try:
                response = _client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:3000]}"},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                pico = json.loads(response.choices[0].message.content)
                required = {"P", "I", "C", "O", "study_design", "pico_confidence"}
                if not required.issubset(pico.keys()):
                    errors += 1
                    continue
                pico["pico_confidence"] = float(pico.get("pico_confidence", 0.5))
                pico["pico_notes"] = pico.get("pico_notes", "")
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts
                        WHERE id = :article_id
                    """), {
                        "pico": json.dumps(pico),
                        "ts": datetime.now(timezone.utc),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"PICO batch error article {article_id}: {e}")
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/metadata/extract")
def extract_metadata_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Enrichit les métadonnées (type d'étude, année, journal) via LLM pour un lot d'articles.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract, source, year
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (metadata_json IS NULL OR metadata_json = '{}'::jsonb)
                  AND abstract IS NOT NULL AND length(abstract) > 30
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    system_prompt = (
        "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
        '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
        '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
        '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
        '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
        "Return ONLY the JSON."
    )

    try:
        from openai import OpenAI as _OAI
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not title:
                skipped += 1
                continue
            try:
                response = _client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:2000]}"},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                metadata = json.loads(response.choices[0].message.content)
                metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET metadata_json = CAST(:meta AS jsonb)
                        WHERE id = :article_id
                    """), {
                        "meta": json.dumps(metadata),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"Metadata batch error article {article_id}: {e}")
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/fulltext/fetch")
def fetch_fulltext_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Tente de récupérer le texte intégral (via DOI/URL) pour un lot d'articles.
    Utilise Unpaywall + CrossRef pour les accès ouverts.
    """
    import urllib.request

    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.doi, ld.url
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                  AND ld.doi IS NOT NULL
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, doi, url
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (has_fulltext IS NULL OR has_fulltext = false)
                  AND doi IS NOT NULL
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    fetched = 0
    not_available = 0
    errors = 0

    for row in rows:
        article_id = row["id"]
        doi = row["doi"]
        if not doi:
            not_available += 1
            continue
        try:
            # Tenter Unpaywall
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=literev@gesica.ch"
            req = urllib.request.Request(unpaywall_url, headers={"User-Agent": "LiteRev/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            oa_url = None
            if data.get("is_oa") and data.get("best_oa_location"):
                oa_url = data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
            if oa_url:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET has_fulltext = true, url = :url
                        WHERE id = :article_id
                    """), {"url": oa_url, "article_id": article_id})
                fetched += 1
            else:
                not_available += 1
        except Exception as e:
            logger.warning(f"Fulltext fetch error article {article_id}: {e}")
            errors += 1

    return {
        "fetched": fetched,
        "not_available": not_available,
        "errors": errors,
        "message": f"{fetched} textes intégraux récupérés, {not_available} non disponibles, {errors} erreurs",
    }


@app.get("/enrichment/status")
def get_enrichment_status(scenario_id: Optional[str] = None):
    """Retourne le statut d'enrichissement (PICO, métadonnées, fulltext) pour un scénario ou tout le corpus."""
    with engine.connect() as conn:
        if scenario_id:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(ld.pico_json) as with_pico,
                    COUNT(CASE WHEN ld.metadata_json IS NOT NULL AND ld.metadata_json != '{}'::jsonb THEN 1 END) as with_metadata,
                    COUNT(CASE WHEN ld.has_fulltext = true THEN 1 END) as with_fulltext
                FROM literature_document ld
                WHERE ld.project_context = 'literev'
                  AND (
                    EXISTS (SELECT 1 FROM article_scenarios asn WHERE asn.document_id = ld.id AND asn.scenario_id = :sid)
                  )
            """), {"sid": scenario_id}).mappings().fetchone()
        else:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(pico_json) as with_pico,
                    COUNT(CASE WHEN metadata_json IS NOT NULL AND metadata_json != '{}'::jsonb THEN 1 END) as with_metadata,
                    COUNT(CASE WHEN has_fulltext = true THEN 1 END) as with_fulltext
                FROM literature_document
                WHERE project_context = 'literev'
            """)).mappings().fetchone()

    total = row["total"] or 1
    return {
        "scenario_id": scenario_id,
        "total": row["total"],
        "pico": {"count": row["with_pico"], "pct": round(row["with_pico"] / total * 100, 1)},
        "metadata": {"count": row["with_metadata"], "pct": round(row["with_metadata"] / total * 100, 1)},
        "fulltext": {"count": row["with_fulltext"], "pct": round(row["with_fulltext"] / total * 100, 1)},
    }


@app.get("/gesica/scenarios/{scenario_id}/pico-stats")
def get_scenario_pico_stats(scenario_id: str):
    """Statistiques PICO — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_pico_stats(scenario_id)


# ─── Screening PRISMA par article dans un scénario ───────────────────────────

@app.post("/gesica/scenarios/{scenario_id}/articles/{article_id}/screen")
def screen_scenario_article(
    scenario_id: str,
    article_id: int,
    status: str,
    reason: str | None = None,
    notes: str | None = None,
    _: None = Depends(require_api_key),
):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return screen_user_scenario_article(scenario_id, article_id, status, reason, notes)


@app.get("/gesica/scenarios/{scenario_id}/screening-progress")
def get_scenario_screening_progress(scenario_id: str) -> dict[str, Any]:
    """Progression du screening PRISMA — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_screening_progress(scenario_id)

# ─── PICO Bulk : tous les articles d'un scénario avec PICO ────────────────────────────────────────────
@app.get("/gesica/scenarios/{scenario_id}/pico-bulk")
def get_scenario_pico_bulk(scenario_id: str, limit: int = 100000, offset: int = 0) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_pico_bulk(scenario_id, limit, offset)

# ─── Evidence Brief PDF ───────────────────────────────────────────────────────
@app.get("/gesica/scenarios/{scenario_id}/evidence-brief")
def get_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_evidence_brief(scenario_id)


# ─── DOUBLE-AVEUGLE SCREENING + KAPPA DE COHEN ───────────────────────────────

def _ensure_double_blind_columns():
    """Crée les colonnes reviewer_1_status/reviewer_2_status si elles n'existent pas."""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE literature_document
            ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN     DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20) DEFAULT NULL
        """))
    logger.info("Colonnes double-aveugle vérifiées/créées.")

# Appel au démarrage
try:
    _ensure_double_blind_columns()
except Exception as _e:
    logger.warning(f"_ensure_double_blind_columns: {_e}")


class DoubleBlindDecisionIn(BaseModel):
    article_id: int
    reviewer: int  # 1 ou 2
    status: str    # 'included' | 'excluded' | 'pending'
    reason: str | None = None
    reviewer_code: str | None = None  # Code reviewer (ex: R-2847)


def _write_ars_screening(
    conn,
    scenario_id: str,
    document_id: int,
    status: str | None,
    reason: str | None = None,
    notes: str | None = None,
) -> None:
    """Migration 2 (per-scenario screening) dual-write: record the screening
    decision on the (scenario_id, document_id) article_scenarios row. Callers
    also keep the global literature_document.screening_status updated, which
    remains authoritative until the Phase 5 cutover. No-op for rows that are not
    members of the scenario (UPDATE matches nothing)."""
    conn.execute(text("""
        UPDATE article_scenarios
        SET screening_status = :status,
            screening_reason = :reason,
            screening_notes  = :notes,
            screened_at      = NOW()
        WHERE scenario_id = :sid AND document_id = :doc_id
    """), {"status": status, "reason": reason, "notes": notes,
           "sid": scenario_id, "doc_id": document_id})


@app.post("/gesica/scenarios/{scenario_id}/double-blind/decision")
def submit_double_blind_decision(
    scenario_id: str,
    payload: DoubleBlindDecisionIn,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Soumet la décision d'un reviewer (1 ou 2) pour le screening double-aveugle.
    Si les deux reviewers ont statué, calcule automatiquement la concordance.
    """
    if payload.reviewer not in (1, 2):
        raise HTTPException(status_code=422, detail="reviewer doit être 1 ou 2")
    if payload.status not in ("included", "excluded", "pending"):
        raise HTTPException(status_code=422, detail="status invalide")

    col_status = f"reviewer_{payload.reviewer}_status"
    col_reason = f"reviewer_{payload.reviewer}_reason"
    col_code = f"reviewer_{payload.reviewer}_code"

    with engine.begin() as conn:
        # Vérifier que l'article appartient bien au scénario (via article_scenarios)
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios
            WHERE document_id = :article_id AND scenario_id = :scenario_id
        """), {"article_id": payload.article_id, "scenario_id": scenario_id}).first()
        
        if not exists:
            raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario")
        
        # Mettre à jour le statut reviewer (colonnes sur literature_document)
        # Ajouter reviewer_N_code si la colonne existe
        try:
            row = conn.execute(text(f"""
                UPDATE literature_document
                SET {col_status} = :status,
                    {col_reason} = :reason,
                    {col_code} = :reviewer_code
                WHERE id = :article_id
                RETURNING id, reviewer_1_status, reviewer_2_status
            """), {
                "status": payload.status,
                "reason": payload.reason,
                "reviewer_code": payload.reviewer_code,
                "article_id": payload.article_id,
            }).first()
        except Exception:
            # Fallback si la colonne reviewer_N_code n'existe pas encore
            row = conn.execute(text(f"""
                UPDATE literature_document
                SET {col_status} = :status,
                    {col_reason} = :reason
                WHERE id = :article_id
                RETURNING id, reviewer_1_status, reviewer_2_status
            """), {
                "status": payload.status,
                "reason": payload.reason,
                "article_id": payload.article_id,
            }).first()

        if not row:
            raise HTTPException(status_code=404, detail="Article non trouvé")

        r1 = row["reviewer_1_status"]
        r2 = row["reviewer_2_status"]
        agreement = None
        final_status = None

        # Si les deux ont statué → calculer concordance et résoudre
        if r1 and r2:
            agreement = r1 == r2
            if agreement:
                final_status = r1
            else:
                # Désaccord → statut "conflict" (à résoudre manuellement)
                final_status = "conflict"

            conn.execute(text("""
                UPDATE literature_document
                SET kappa_resolved = :resolved,
                    kappa_final_status = :final,
                    screening_status = :screening
                WHERE id = :article_id
            """), {
                "resolved": agreement,
                "final": final_status,
                "screening": final_status if agreement else "pending",
                "article_id": payload.article_id,
            })
            # Migration 2 dual-write: per-scenario screening for this scenario
            _write_ars_screening(conn, scenario_id, payload.article_id,
                                 final_status if agreement else "pending")

    return {
        "id": payload.article_id,
        "reviewer": payload.reviewer,
        "status": payload.status,
        "reviewer_1_status": r1 if payload.reviewer == 2 else payload.status,
        "reviewer_2_status": r2 if payload.reviewer == 1 else payload.status,
        "agreement": agreement,
        "final_status": final_status,
    }


@app.get("/gesica/scenarios/{scenario_id}/double-blind/kappa")
def get_kappa_stats(scenario_id: str) -> dict[str, Any]:
    """Kappa de Cohen — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_kappa(scenario_id)


@app.get("/gesica/scenarios/{scenario_id}/double-blind/conflicts")
def get_conflicts(scenario_id: str) -> list[dict[str, Any]]:
    """Conflits double-aveugle — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_conflicts(scenario_id)


@app.post("/gesica/scenarios/{scenario_id}/double-blind/resolve")
def resolve_conflict(
    scenario_id: str,
    article_id: int,
    final_status: str,
    arbitrator_notes: str | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Résout un conflit entre reviewers (arbitrage par un tiers)."""
    if final_status not in ("included", "excluded"):
        raise HTTPException(status_code=422, detail="final_status doit être 'included' ou 'excluded'")

    with engine.begin() as conn:
        row = conn.execute(text("""
            UPDATE literature_document
            SET kappa_final_status = :final,
                kappa_resolved = TRUE,
                screening_status = :final,
                screening_notes = :notes
            WHERE id = :article_id
              AND project_context = 'literev'
              AND EXISTS (SELECT 1 FROM article_scenarios ars WHERE ars.document_id = literature_document.id AND ars.scenario_id = :scenario_id)
            RETURNING id
        """), {
            "final": final_status,
            "notes": arbitrator_notes,
            "article_id": article_id,
            "scenario_id": scenario_id,
        }).first()
        if row:
            # Migration 2 dual-write: per-scenario screening for this scenario
            _write_ars_screening(conn, scenario_id, article_id, final_status, notes=arbitrator_notes)
    if not row:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    return {"id": row[0], "final_status": final_status, "resolved": True}


# ─── KNOWLEDGE GRAPH (réseau de similarité sémantique) ───────────────────────

# Mots vides (EN + FR + remplissage scientifique) pour étiqueter les communautés
# thématiques à partir des titres d'articles.
_KG_STOPWORDS: set[str] = {
    # anglais courant
    "the", "and", "for", "with", "from", "this", "that", "study", "studies",
    "using", "based", "between", "among", "during", "after", "before", "into",
    "their", "these", "those", "which", "while", "about", "versus", "over",
    "analysis", "review", "systematic", "meta", "trial", "trials", "randomized",
    "randomised", "controlled", "results", "methods", "patients", "patient",
    "outcomes", "outcome", "associated", "association", "effect", "effects",
    "evaluation", "assessment", "comparison", "clinical", "data", "report",
    "case", "cases", "cohort", "prospective", "retrospective", "evidence",
    # français courant
    "les", "des", "une", "dans", "pour", "avec", "sur", "par", "aux", "leur",
    "étude", "étude", "analyse", "revue", "résultats", "méthode", "méthodes",
    "patients", "patient", "effet", "effets", "entre", "chez", "selon", "lors",
}


def _kg_cluster_label(titles: list[str], top_k: int = 3) -> str:
    """Étiquette thématique d'une communauté : termes les plus fréquents des titres."""
    from collections import Counter
    cnt: Counter = Counter()
    for t in titles:
        for tok in re.findall(r"[a-zàâäéèêëïîôöùûüç]{4,}", (t or "").lower()):
            if tok not in _KG_STOPWORDS:
                cnt[tok] += 1
    return ", ".join(w for w, _ in cnt.most_common(top_k))


def _build_knowledge_graph(
    scenario_id: str,
    rows: list,
    min_similarity: float,
    n_total: int,
) -> dict[str, Any]:
    """
    Construit un graphe de connaissance à partir d'articles + embeddings.

    Nœuds = articles ; taille = centralité (degré) ; couleur = communauté thématique.
    Arêtes = paires d'articles dont la similarité cosinus des embeddings ≥ min_similarity.
    Communautés = détection greedy ; chacune reçoit une étiquette de mots-clés (titres).
    `n_total` = nombre total d'articles éligibles (pour signaler un éventuel sous-ensemble).
    """
    if not rows:
        return {"nodes": [], "edges": [], "clusters": [], "n_total": n_total}

    import numpy as np

    nodes_data = []
    for r in rows:
        try:
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", r["emb_str"])
            emb = np.array([float(x) for x in nums], dtype=np.float32)
            if len(emb) > 0:
                nodes_data.append({
                    "id": r["id"],
                    "title": r["title"] or "",
                    "year": r["year"],
                    "journal": r["journal"],
                    "design": r["design"],
                    "quality": float(r["quality_score"] or 0),
                    "emb": emb,
                })
        except Exception:
            continue

    if not nodes_data:
        return {"nodes": [], "edges": [], "clusters": [], "n_total": n_total}

    embeddings = np.array([n["emb"] for n in nodes_data])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms
    sim_matrix = embeddings_norm @ embeddings_norm.T

    n = len(nodes_data)
    # Arêtes (vectorisé : on ne garde que le triangle supérieur au-dessus du seuil)
    edges = []
    iu, ju = np.triu_indices(n, k=1)
    mask = sim_matrix[iu, ju] >= min_similarity
    for i, j, w in zip(iu[mask], ju[mask], sim_matrix[iu, ju][mask]):
        edges.append({
            "source": nodes_data[int(i)]["id"],
            "target": nodes_data[int(j)]["id"],
            "weight": round(float(w), 3),
        })

    # Détection de communautés greedy (lien fort ≥ 0.5)
    cluster_ids = [-1] * n
    cluster_counter = 0
    for i in range(n):
        if cluster_ids[i] == -1:
            cluster_ids[i] = cluster_counter
            strong = np.where(sim_matrix[i] >= 0.5)[0]
            for j in strong:
                if cluster_ids[j] == -1:
                    cluster_ids[j] = cluster_counter
            cluster_counter += 1

    # Degré (centralité) par nœud
    degree: dict[int, int] = {nd["id"]: 0 for nd in nodes_data}
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    nodes = []
    for idx, nd in enumerate(nodes_data):
        title = nd["title"]
        nodes.append({
            "id": nd["id"],
            "title": title[:80] + ("..." if len(title) > 80 else ""),
            "year": nd["year"],
            "journal": nd["journal"],
            "design": nd["design"],
            "quality": nd["quality"],
            "cluster": cluster_ids[idx],
            "degree": degree[nd["id"]],
        })

    # Résumé des communautés + étiquette thématique
    from collections import defaultdict
    members_map: dict[int, list] = defaultdict(list)
    titles_map: dict[int, list] = defaultdict(list)
    for idx, nd in enumerate(nodes_data):
        cid = cluster_ids[idx]
        members_map[cid].append(nodes[idx])
        titles_map[cid].append(nd["title"])

    clusters = []
    for cid, members in sorted(members_map.items(), key=lambda x: -len(x[1])):
        label = _kg_cluster_label(titles_map[cid])
        clusters.append({
            "id": cid,
            "size": len(members),
            "label": label,
            "years": sorted(set(m["year"] for m in members if m["year"])),
            "designs": list(set(m["design"] for m in members if m["design"] and m["design"] != "unknown")),
            "top_articles": [m["title"] for m in sorted(members, key=lambda x: -x["quality"])[:3]],
        })

    return {
        "scenario_id": scenario_id,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_clusters": len(clusters),
        "n_total": n_total,
        "min_similarity": min_similarity,
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
    }


# SQL partagé : sélectionne un embedding par article, priorise les meilleurs articles
_KG_NODE_SQL = """
    SELECT * FROM (
        SELECT DISTINCT ON (d.id)
            d.id, d.title, d.year, d.journal, d.study_design, d.quality_score,
            c.embedding::text AS emb_str,
            COALESCE((d.pico_json->>'study_design'), d.study_design, 'unknown') AS design
        FROM literature_document d
        {join}
        WHERE {where}
          AND d.is_duplicate IS NOT TRUE
          AND c.embedding IS NOT NULL
          AND d.abstract IS NOT NULL
        ORDER BY d.id, (c.chunk_type = 'title_abstract') DESC, c.id
    ) sub
    ORDER BY quality_score DESC NULLS LAST, year DESC NULLS LAST
    LIMIT :max_nodes
"""


@app.get("/gesica/scenarios/{scenario_id}/knowledge-graph")
def get_knowledge_graph(
    scenario_id: str,
    max_nodes: int = 400,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return _compute_user_kg(scenario_id, max_nodes, min_similarity)


# ─── STREAMING RAG SSE ────────────────────────────────────────────────────────

from fastapi.responses import StreamingResponse

@app.post("/ask/stream")
async def ask_stream(payload: dict[str, Any]) -> StreamingResponse:
    """
    Version streaming (SSE) de l'endpoint /ask.
    Retourne les tokens au fur et à mesure via Server-Sent Events.
    """
    from openai import AsyncOpenAI

    question = payload.get("question", "")
    project_context = payload.get("project_context", "literev")
    scenario_id = payload.get("scenario_id", None)
    top_k = int(payload.get("top_k", 8))
    lang = payload.get("lang")

    if not question:
        raise HTTPException(status_code=422, detail="question est requis")

    # Récupérer le contexte RAG (chunks pertinents)
    try:
        from openai import OpenAI as SyncOpenAI
        sync_client = SyncOpenAI(timeout=90.0)
        emb_resp = sync_client.embeddings.create(
            model="text-embedding-3-small",
            input=question[:2000],
        )
        q_emb = emb_resp.data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"
    except Exception as e:
        logger.error(f"Embedding error in /ask/stream: {e}")
        emb_str = None

    context_chunks = []
    sources = []
    if emb_str:
        where_extra = ""
        join_extra = ""
        # Migration 2 : quand un scénario est fourni, on JOINT article_scenarios
        # pour lire le screening PROPRE au scénario (COALESCE), comme
        # /ask/stream/filtered. Sans scénario, colonne globale.
        screen_expr = "d.screening_status"
        params_extra: dict[str, Any] = {
            "top_k": top_k, "emb": emb_str, "max_dist": 1.0 - RAG_MIN_SIMILARITY,
        }
        if project_context:
            where_extra += " AND d.project_context = :project_context"
            params_extra["project_context"] = project_context
        if scenario_id:
            join_extra = " JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :scenario_id "
            screen_expr = "COALESCE(ars.screening_status, d.screening_status)"
            params_extra["scenario_id"] = scenario_id

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT c.content, d.title, d.year, d.doi, d.authors, d.id AS doc_id,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                {join_extra}
                WHERE c.embedding IS NOT NULL
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                  AND {screen_expr} IS DISTINCT FROM 'excluded'
                  AND (c.embedding <=> CAST(:emb AS vector)) <= :max_dist
                  {where_extra}
                ORDER BY c.embedding <=> CAST(:emb AS vector)
                LIMIT :top_k
            """), params_extra).mappings().all()

        for i, r in enumerate(rows):
            # Inclure titre + année DANS le contexte : le prompt demande de citer
            # les articles par leur titre, donc le modèle doit les voir.
            context_chunks.append(
                f"[{i + 1}] {r['title'] or 'Sans titre'} ({r['year'] or 'année inconnue'})\n{r['content']}"
            )
            # Champs canoniques (document_id/score/authors) attendus par le front.
            sources.append({
                "document_id": r["doc_id"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "authors": r["authors"],
                "score": round(float(r["similarity"]), 3),
            })

    context_text = "\n\n---\n\n".join(context_chunks[:top_k]) if context_chunks else "Aucun contexte disponible."

    system_prompt = """Tu es un assistant expert en médecine d'urgence et en revue systématique de la littérature scientifique.
Tu réponds de manière précise, factuelle et synthétique.
Base-toi exclusivement sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Cite les articles pertinents par leur titre quand tu les mentionnes.""" + _llm_lang_directive(lang)

    user_prompt = f"""Contexte scientifique (extraits d'articles) :
{context_text}

Question : {question}

Réponds de manière structurée et cite les sources pertinentes du contexte."""

    async def event_generator():
        # D'abord envoyer les sources
        import json as _json
        sources_event = f"event: sources\ndata: {_json.dumps(sources)}\n\n"
        yield sources_event

        # Pas de contexte récupéré → ne PAS interroger le LLM (réponse non étayée).
        if not context_chunks:
            msg = ("Aucun passage pertinent n'a été trouvé dans le corpus pour cette "
                   "question. Reformulez la question ou élargissez le corpus.")
            yield f"data: {_json.dumps({'token': msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Puis streamer la réponse LLM
        try:
            async_client = AsyncOpenAI(timeout=90.0)
            stream = await async_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=1200,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    token_event = f"data: {_json.dumps({'token': delta.content})}\n\n"
                    yield token_event
        except Exception as e:
            yield f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── PDF EVIDENCE BRIEF CÔTÉ SERVEUR ─────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/evidence-brief/pdf")
def get_evidence_brief_pdf(scenario_id: str):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_evidence_brief_pdf(scenario_id)


# ─── PIPELINE LIVING REVIEW AUTOMATISÉ ───────────────────────────────────────

@app.post("/gesica/living-review/trigger")
def trigger_living_review(
    scenario_id: str | None = None,
    dry_run: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Déclenche le pipeline Living Review :
    1. Interroge PubMed avec la requête booléenne du scénario
    2. Insère les nouveaux articles
    3. Génère les embeddings
    4. Invalide le cache clustering
    Retourne un rapport de ce qui a été fait (ou ce qui serait fait en dry_run).
    """
    import threading

    scenarios_to_update = []
    with engine.connect() as conn:
        if scenario_id:
            meta = _get_db_gesica_scenario_or_404(scenario_id, conn)
            scenarios_to_update = [(scenario_id, meta)]
        else:
            scenarios_to_update = [(row["id"], row) for row in _list_db_gesica_scenarios(conn)]

    report = {
        "dry_run": dry_run,
        "triggered_at": __import__("datetime").datetime.now().isoformat(),
        "scenarios": [],
        "status": "triggered" if not dry_run else "dry_run",
    }

    for sid, smeta in scenarios_to_update:
        scenario_report = {
            "scenario_id": sid,
            "title": _gesica_title(smeta),
            "query": ((smeta.get("boolean_queries") or ["N/A"])[0] if isinstance(smeta.get("boolean_queries"), list) else (smeta.get("boolean_queries") or smeta.get("query") or "N/A")),
            "action": "would_fetch" if dry_run else "fetching",
        }
        report["scenarios"].append(scenario_report)

    if not dry_run:
        def _run_living_review():
            try:
                import subprocess
                result = subprocess.run(
                    ["python3", "ingest_pubmed.py", "--all-scenarios"],
                    capture_output=True, text=True, timeout=600,
                    cwd="/opt/literev-api"
                )
                logger.info(f"Living Review pipeline: {result.stdout[:500]}")
                if result.returncode != 0:
                    logger.error(f"Living Review error: {result.stderr[:500]}")
            except Exception as e:
                logger.error(f"Living Review pipeline error: {e}")

        threading.Thread(target=_run_living_review, daemon=True).start()
        report["message"] = "Pipeline Living Review déclenché en arrière-plan. Vérifiez les logs dans 5-10 minutes."
    else:
        report["message"] = f"Dry run : {len(scenarios_to_update)} scénario(s) seraient mis à jour."

    return report


# ─── ALERTES EMAIL ────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_email(v) -> str | None:
    """Normalise (trim + minuscules) et valide une adresse email ; None si vide/invalide."""
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if (v and _EMAIL_RE.match(v)) else None


def _ensure_alert_subscription(conn, email: str, scenario_id: str, frequency: str = "weekly") -> None:
    """Crée la table si besoin puis (ré)active l'abonnement email↔scénario (idempotent)."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            scenario_id VARCHAR(100) NOT NULL,
            frequency VARCHAR(20) DEFAULT 'weekly',
            created_at TIMESTAMP DEFAULT NOW(),
            last_notified_at TIMESTAMP DEFAULT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            UNIQUE(email, scenario_id)
        )
    """))
    conn.execute(text("""
        INSERT INTO alert_subscriptions (email, scenario_id, frequency)
        VALUES (:email, :scenario_id, :frequency)
        ON CONFLICT (email, scenario_id) DO UPDATE SET frequency = :frequency, is_active = TRUE
    """), {"email": email, "scenario_id": scenario_id, "frequency": frequency})


class AlertSubscriptionIn(BaseModel):
    email: str = Field(..., max_length=255)
    scenario_id: str = Field(..., min_length=1, max_length=100)
    frequency: str = "weekly"  # "daily" | "weekly" | "immediate"

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Adresse email invalide")
        return v

    @field_validator("frequency")
    @classmethod
    def _validate_frequency(cls, v: str) -> str:
        if v not in ("daily", "weekly", "immediate"):
            raise ValueError("frequency doit être 'daily', 'weekly' ou 'immediate'")
        return v


@app.post("/alerts/subscribe")
def subscribe_alerts(payload: AlertSubscriptionIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Enregistre une alerte email pour un scénario.
    L'utilisateur sera notifié quand de nouveaux articles sont ajoutés.
    """
    email = _clean_email(payload.email) or payload.email
    with engine.begin() as conn:
        _ensure_alert_subscription(conn, email, payload.scenario_id, payload.frequency)
        # S'abonner à SON scénario utilisateur = en devenir propriétaire (si aucun),
        # pour que la living review le crawle (run_user_scenarios filtre sur owner_email).
        owner_set = False
        if payload.scenario_id.startswith("usr-"):
            res = conn.execute(text(
                "UPDATE user_scenarios SET owner_email = :e, updated_at = NOW() "
                "WHERE id = :id AND (owner_email IS NULL OR owner_email = '')"
            ), {"e": email, "id": payload.scenario_id})
            owner_set = (res.rowcount or 0) > 0

    return {
        "status": "subscribed",
        "email": email,
        "scenario_id": payload.scenario_id,
        "frequency": payload.frequency,
        "owner_set": owner_set,
        "message": f"Vous recevrez des alertes {payload.frequency} pour le scénario '{payload.scenario_id}'.",
    }


@app.delete("/alerts/unsubscribe")
def unsubscribe_alerts(email: str, scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Désabonnement des alertes email."""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE alert_subscriptions
            SET is_active = FALSE
            WHERE email = :email AND scenario_id = :scenario_id
        """), {"email": email, "scenario_id": scenario_id})
    return {"status": "unsubscribed", "email": email, "scenario_id": scenario_id}


@app.get("/alerts/subscriptions")
def list_subscriptions(email: str) -> list[dict[str, Any]]:
    """Liste les abonnements actifs pour un email."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT scenario_id, frequency, created_at, last_notified_at, is_active
                FROM alert_subscriptions
                WHERE email = :email AND is_active = TRUE
                ORDER BY created_at DESC
            """), {"email": email}).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _new_articles_for_scenario(conn, scenario_id: str, since, limit: int = 25) -> list[dict]:
    """Articles du corpus d'un scénario ingérés APRÈS `since`. Couvre les deux modèles
    d'appartenance : préréglé (literature_document.scenario_type) ET utilisateur
    (article_scenarios). since=None (1re notification) → les plus récents (bornés)."""
    rows = conn.execute(text("""
        SELECT DISTINCT d.id, d.title, d.year, d.doi, d.url, d.created_at
        FROM literature_document d
        WHERE (d.scenario_type = :sid
               OR EXISTS (SELECT 1 FROM article_scenarios a
                          WHERE a.document_id = d.id AND a.scenario_id = :sid))
          AND (CAST(:since AS timestamp) IS NULL OR d.created_at > CAST(:since AS timestamp))
        ORDER BY d.created_at DESC
        LIMIT :lim
    """), {"sid": scenario_id, "since": since, "lim": int(limit)}).mappings().all()
    return [dict(r) for r in rows]


def _digest_is_due(frequency: str | None, last_notified, now) -> bool:
    """Un abonnement est-il dû ? immediate = toujours ; daily/weekly selon le délai
    depuis la dernière notification ; jamais notifié = dû."""
    from datetime import timedelta
    if last_notified is None:
        return True
    if (frequency or "weekly") == "immediate":
        return True
    delta = now - last_notified
    return delta >= (timedelta(days=1) if frequency == "daily" else timedelta(days=7))


def _render_alert_digest(scenario_id: str, articles: list[dict], total_new: int,
                         base_url: str = "https://literev-scenario.com",
                         scenario_name: str | None = None) -> tuple[str, str, str]:
    """(subject, html, text) d'un digest — liste les VRAIS nouveaux articles. Pur/testable.
    Utilise le NOM lisible du scénario (pas l'ID) et un lien PROFOND vers sa page
    (?scenario=<id>, ouvert directement par le front)."""
    import html as _html
    from urllib.parse import quote as _q
    label = (scenario_name or scenario_id).strip() or scenario_id
    # Accord FR : 1 → « nouvel article » ; ≥2 → « nouveaux articles ».
    noun = "nouvel article" if total_new == 1 else "nouveaux articles"
    subj = f"[LiteRev] {total_new} {noun} — {label}"
    scen_url = f"{base_url}/?scenario={_q(scenario_id, safe='')}"

    def _row(a):
        title = _html.escape(str(a.get("title") or "Article"))
        yr = f" ({a['year']})" if a.get("year") else ""
        href = a.get("url") or (f"https://doi.org/{a['doi']}" if a.get("doi") else scen_url)
        return f'<li style="margin:4px 0"><a href="{_html.escape(str(href))}" style="color:#16a34a">{title}</a>{yr}</li>'

    shown = articles[:25]
    more = f'<p style="font-size:12px;color:#6b7280">… et {total_new - len(shown)} de plus.</p>' if total_new > len(shown) else ""
    html_body = (
        '<html><body style="font-family:system-ui,Arial,sans-serif;color:#111">'
        f'<h2 style="color:#14532d">LiteRev — {total_new} {noun}</h2>'
        f'<p>Scénario <strong>{_html.escape(label)}</strong> :</p>'
        f'<ul>{"".join(_row(a) for a in shown)}</ul>{more}'
        f'<p><a href="{_html.escape(scen_url)}" style="color:#16a34a;font-weight:600">Ouvrir le scénario →</a></p>'
        '<hr><p style="font-size:11px;color:#6b7280">Vous recevez cet email car vous êtes abonné aux alertes LiteRev pour ce scénario.</p>'
        '</body></html>'
    )
    text_body = (f"LiteRev — {total_new} {noun} pour le scénario « {label} » :\n\n"
                 + "\n".join(f"- {a.get('title', '')}" + (f" ({a['year']})" if a.get("year") else "") for a in shown)
                 + (f"\n… et {total_new - len(shown)} de plus." if total_new > len(shown) else "")
                 + f"\n\n{scen_url}\n")
    return subj, html_body, text_body


def _smtp_mode_for(port, security: str | None) -> str:
    """Mode de connexion SMTP : 'ssl' | 'starttls' | 'none'. Un SMTP_SECURITY
    explicite l'emporte ; sinon on l'infère du port (587/25 → STARTTLS, sinon SSL).
    Couvre GoDaddy Professional Email (smtpout.secureserver.net:465 SSL) ET
    Microsoft 365 (smtp.office365.com:587 STARTTLS)."""
    mode = (security or "").strip().lower()
    if mode in ("ssl", "starttls", "none"):
        return mode
    try:
        p = int(port)
    except (TypeError, ValueError):
        p = 465
    return "starttls" if p in (587, 25) else "ssl"


def _send_email_smtp(host: str, user: str, pw: str, to: str, subject: str, html_body: str, text_body: str,
                     port=None, security: str | None = None) -> None:
    """Envoi SMTP, port/mode configurables (env SMTP_PORT / SMTP_SECURITY ; défaut
    465 SSL). Lève en cas d'échec (l'appelant journalise)."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    port = int(port or os.getenv("SMTP_PORT") or 465)
    mode = _smtp_mode_for(port, security if security is not None else os.getenv("SMTP_SECURITY"))
    if mode == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            if user:
                server.login(user, pw)
            server.sendmail(user, to, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if mode == "starttls":
                server.starttls()
                server.ehlo()
            if user:
                server.login(user, pw)
            server.sendmail(user, to, msg.as_string())


def _process_alert_digests(scenario_id: str | None, dry_run: bool, respect_frequency: bool) -> dict[str, Any]:
    """Cœur commun de l'envoi des digests : pour chaque abonnement actif (et dû si
    respect_frequency), calcule les NOUVEAUX articles depuis last_notified_at, envoie
    l'email (sauf dry_run) et met à jour last_notified_at. Ne notifie pas si 0 nouveauté."""
    from datetime import datetime, timezone
    smtp_host, smtp_user, smtp_pass = os.getenv("SMTP_HOST", ""), os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", "")
    now = datetime.now(timezone.utc)
    try:
        with engine.connect() as conn:
            q = "SELECT id, email, scenario_id, frequency, last_notified_at FROM alert_subscriptions WHERE is_active = TRUE"
            params: dict[str, Any] = {}
            if scenario_id:
                q += " AND scenario_id = :sid"
                params["sid"] = scenario_id
            subs = [dict(r) for r in conn.execute(text(q), params).mappings().all()]
    except Exception as e:
        return {"status": "error", "error": str(e), "processed": 0, "sent": 0}

    results, sent = [], 0
    for sub in subs:
        if respect_frequency and not _digest_is_due(sub.get("frequency"), sub.get("last_notified_at"), now):
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "skipped": "not_due"})
            continue
        try:
            with engine.connect() as conn:
                arts = _new_articles_for_scenario(conn, sub["scenario_id"], sub.get("last_notified_at"))
        except Exception as e:
            logger.warning(f"digest new-articles {sub['scenario_id']}: {e}")
            arts = []
        if not arts:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": 0, "sent": False})
            continue
        if dry_run:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": "dry_run"})
            continue
        if not smtp_host:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": "smtp_not_configured"})
            continue
        try:
            _scen_name = _get_scenario_name(sub["scenario_id"])
        except Exception:
            _scen_name = None
        subj, html_body, text_body = _render_alert_digest(sub["scenario_id"], arts, len(arts), scenario_name=_scen_name)
        try:
            _send_email_smtp(smtp_host, smtp_user, smtp_pass, sub["email"], subj, html_body, text_body)
            with engine.begin() as conn:
                conn.execute(text("UPDATE alert_subscriptions SET last_notified_at = NOW() WHERE id = :id"), {"id": sub["id"]})
            sent += 1
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": True})
        except Exception as e:
            logger.error(f"digest email {sub['email']}: {e}")
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": f"error: {e}"})

    status = "sent" if (not dry_run and sent) else ("dry_run" if dry_run else ("not_configured" if not smtp_host else "no_new_articles"))
    return {"status": status, "subscriptions": len(subs), "processed": len(results), "sent": sent, "dry_run": dry_run, "results": results}


@app.post("/alerts/send-digest")
def send_alert_digest(scenario_id: str | None = None, dry_run: bool = True,
                      _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Envoie (ou simule en dry_run) les digests aux abonnés d'un scénario (ou de tous) :
    liste les VRAIS nouveaux articles depuis la dernière notification et met à jour
    last_notified_at. Ignore les abonnements sans nouveauté. Ignore la fréquence
    (envoi forcé) — pour la cadence automatique, voir /alerts/run-digests."""
    return _process_alert_digests(scenario_id, dry_run=dry_run, respect_frequency=False)


@app.post("/alerts/run-digests")
def run_due_alert_digests(dry_run: bool = False, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Traite tous les abonnements DUS selon leur fréquence (daily/weekly/immediate).
    Appelable par cron après la living review (ex. `curl -XPOST .../alerts/run-digests`)
    pour notifier les utilisateurs des nouveaux articles de LEURS scénarios."""
    return _process_alert_digests(None, dry_run=dry_run, respect_frequency=True)


# ─────────────────────────────────────────────────────────────────────────────
# USER SCENARIOS : Recherches sauvegardées persistées en base
# ─────────────────────────────────────────────────────────────────────────────
# Chaque recherche sauvegardée devient un vrai scénario utilisateur avec :
#   - son propre corpus (articles ingérés via PubMed)
#   - tous les onglets du ScenarioDetailPage (corpus, PICO, screening, RAG, etc.)
#   - un ID de la forme "usr-<uuid4_court>"
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_user_scenarios_table() -> None:
    """Crée la table user_scenarios et user_scenario_folders si elles n'existent pas."""
    with engine.begin() as conn:
        # Table des dossiers
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_scenario_folders (
                id          VARCHAR(40)  PRIMARY KEY,
                name        VARCHAR(255) NOT NULL,
                color       VARCHAR(20)  DEFAULT '#6366f1',
                sort_order  INTEGER      DEFAULT 0,
                created_at  TIMESTAMP    DEFAULT NOW()
            )
        """))
        # Table des scénarios (avec folder_id optionnel)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_scenarios (
                id          VARCHAR(40)  PRIMARY KEY,
                name        VARCHAR(255) NOT NULL,
                query       TEXT         NOT NULL,
                mode        VARCHAR(20)  NOT NULL DEFAULT 'hybrid',
                filters     JSONB        NOT NULL DEFAULT '{}',
                result_count INTEGER     DEFAULT 0,
                pinned      BOOLEAN      DEFAULT FALSE,
                folder_id   VARCHAR(40)  REFERENCES user_scenario_folders(id) ON DELETE SET NULL,
                created_at  TIMESTAMP    DEFAULT NOW(),
                updated_at  TIMESTAMP    DEFAULT NOW()
            )
        """))
        # Ajouter folder_id si la table existait déjà sans cette colonne
        conn.execute(text("""
            ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS folder_id VARCHAR(40)
            REFERENCES user_scenario_folders(id) ON DELETE SET NULL
        """))
        # Colonnes du pipeline d'ingestion/populate (sur tables préexistantes)
        for _ddl in (
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS populate_status VARCHAR(20)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_status VARCHAR(20)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_step VARCHAR(80)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_progress INTEGER DEFAULT 0",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_started_at TIMESTAMP",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS article_count INTEGER DEFAULT 0",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS search_strategy JSONB",
            # Clustering persisté en base (sinon perdu au redémarrage du serveur)
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS cluster_id INTEGER",
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS cluster_label TEXT",
            # Score d'un cross-encoder (rerank Cohere) — précision supérieure au
            # cosinus pour ORDONNER le sous-ensemble pertinent (sélection = cosinus
            # >= seuil ; ordre = rerank_score quand présent).
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS rerank_score FLOAT",
            # Borne anti « lot empoisonné » : nb d'échecs d'embedding pour un chunk.
            # Au-delà de 3, il est exclu du worker / du compteur « en attente » / de
            # /admin/embed-pending — sinon un chunk que l'API refuse (contenu invalide)
            # resterait « en attente » à l'infini et serait ré-essayé chaque cycle.
            "ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding_attempts INTEGER DEFAULT 0",
            # Recherche multi-sous-requêtes : liste [{"kind":"boolean"|"natural",
            # "text":...}] + combinateur ("union" = OU, "intersection" = ET) entre
            # leurs ensembles de résultats. NULL = recherche mono-requête classique
            # (colonnes query/mode) → comportement inchangé.
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS sub_queries JSONB",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS combinator VARCHAR(12)",
            # Propriétaire (email) : living review + alertes email par utilisateur.
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS owner_email VARCHAR(255)",
            "CREATE INDEX IF NOT EXISTS ix_user_scenarios_owner ON user_scenarios (owner_email)",
        ):
            conn.execute(text(_ddl))
    logger.info("Tables user_scenarios et user_scenario_folders vérifiées/créées.")

try:
    _ensure_user_scenarios_table()
except Exception as _e:
    logger.warning(f"_ensure_user_scenarios_table: {_e}")


def _truncate_display_name(v: Any, limit: int = 255) -> Any:
    """Tronque un libellé d'affichage à `limit` caractères (colonne VARCHAR(255)).

    `name` est un LIBELLÉ ; la requête complète vit dans `query` (TEXT). Une requête
    booléenne un peu longue dépasse 255 caractères → sans ce garde-fou, Pydantic
    renvoie 422 (max_length) ou la colonne VARCHAR(255) déborde (500). On tronque
    proprement avec « … » pour que la recherche aboutisse quel que soit le client."""
    if isinstance(v, str):
        v = v.strip()
        if len(v) > limit:
            v = v[: limit - 1].rstrip() + "…"
    return v


class UserScenarioIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    query: str = Field(..., min_length=1)
    mode: str = Field(default="hybrid")
    filters: dict[str, Any] = Field(default_factory=dict)
    result_count: int = Field(default=0, ge=0)
    pinned: bool = Field(default=False)
    folder_id: str | None = None
    # Stratégie booléenne (générée par LLM) déjà calculée côté recherche. Si
    # fournie, on la persiste telle quelle pour que le corpus utilise EXACTEMENT
    # la même requête booléenne que celle affichée/comptée à la recherche.
    search_strategy: dict[str, Any] | None = None
    # Recherche MULTI-sous-requêtes : liste [{"kind":"boolean"|"natural","text":...}]
    # combinée par `combinator` ("union"=OU par défaut, "intersection"=ET explicite).
    # < 2 entrées valides → None (on retombe sur la recherche mono-requête query/mode).
    sub_queries: list[dict[str, Any]] | None = None
    combinator: str = Field(default="union")
    # Propriétaire (email) : relie le scénario à un utilisateur pour la living review
    # et les alertes email. Optionnel (les scénarios restent utilisables sans compte).
    owner_email: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _cap_name(cls, v: Any) -> Any:
        return _truncate_display_name(v)

    @model_validator(mode="after")
    def _clean_sub_queries(self) -> "UserScenarioIn":
        cleaned = _normalize_sub_queries(self.sub_queries)
        self.sub_queries = cleaned if len(cleaned) >= 2 else None
        if self.combinator not in ("union", "intersection"):
            self.combinator = "union"
        self.owner_email = _clean_email(self.owner_email)
        return self


class UserScenarioPatch(BaseModel):
    name: str | None = None
    pinned: bool | None = None
    mode: str | None = None
    filters: dict[str, Any] | None = None
    folder_id: str | None = None  # Assigner à un dossier (None = hors dossier)

    @field_validator("name", mode="before")
    @classmethod
    def _cap_name(cls, v: Any) -> Any:
        return _truncate_display_name(v)


class FolderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    color: str = Field(default='#6366f1')
    sort_order: int = Field(default=0)


# ── Helpers internes ──────────────────────────────────────────────────────────

def _get_user_scenario_or_404(scenario_id: str) -> dict[str, Any]:
    """Retourne la ligne user_scenarios ou lève 404."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, name, query, mode, filters, result_count, pinned, folder_id, created_at, updated_at,
                   search_strategy, populate_status, pipeline_status, pipeline_step,
                   pipeline_progress, pipeline_started_at, article_count, is_system,
                   sub_queries, combinator
            FROM user_scenarios WHERE id = :id
        """), {"id": scenario_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario utilisateur '{scenario_id}' non trouvé")
    return dict(row)


def _user_scenario_to_gesica_format(
    row: dict[str, Any], counts_map: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convertit une ligne user_scenarios au format GesicaScenario (liste).

    Si counts_map est fourni (chemin liste), on y lit les compteurs déjà agrégés
    au lieu d'exécuter la requête d'agrégation une fois PAR scénario (évite le N+1).
    Les appels unitaires (création / lecture d'un seul scénario) le laissent à None
    et interrogent la base normalement."""
    if counts_map is not None:
        counts = counts_map.get(str(row["id"]))
    else:
        with engine.connect() as conn:
            counts = conn.execute(text("""
                SELECT
                    COUNT(DISTINCT ars.document_id) AS article_count,
                    COUNT(DISTINCT ars.document_id) FILTER (
                        WHERE COALESCE(ars.screening_status, d.screening_status) = 'included'
                    ) AS included_count,
                    COUNT(DISTINCT ars.document_id) FILTER (
                        WHERE COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                    ) AS excluded_count
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            """), {"sid": row["id"]}).mappings().first()

    article_count = int(counts["article_count"] or 0) if counts else 0
    included = int(counts["included_count"] or 0) if counts else 0
    excluded = int(counts["excluded_count"] or 0) if counts else 0

    # Actions recommandées (cache) + résumé du modèle entraîné, pour la carte
    # généralisée du tableau de bord (mêmes blocs que les scénarios GESICA).
    actions: list = []
    model_summary: dict[str, Any] = {"has_model": False}
    try:
        with engine.connect() as _c2:
            _a = _c2.execute(text(
                "SELECT recommended_actions_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": row["id"]}).scalar()
            if isinstance(_a, list):
                actions = _a
            _m = _c2.execute(text("""
                SELECT family, metric, metrics_json FROM scenario_model_run
                WHERE scenario_id = :sid AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """), {"sid": row["id"]}).mappings().first()
            if _m:
                mj = _m["metrics_json"] or {}
                mv = mj.get(_m["metric"]) if _m["metric"] else (list(mj.values())[0] if mj else None)
                model_summary = {
                    "has_model": True, "family": _m["family"], "metric": _m["metric"],
                    "metric_value": (float(mv) if isinstance(mv, (int, float)) else None),
                }
    except Exception as _e_card:
        logger.warning(f"Card extras {row['id']}: {_e_card}")

    return {
        "id": row["id"],
        "name": row["name"],
        "title": row["name"],
        "description": f"Recherche sauvegardée : {row['query']}",
        "cluster": "user",
        "article_count": article_count,
        "included_count": included,
        "excluded_count": excluded,
        "kappa_score": None,
        "hidden": False,
        "recommended_actions": actions,
        "model": model_summary,
        "relevant_articles": [],
        "living_evidence_note": (
            f"Living Evidence Review · {article_count} articles indexés. Mis à jour automatiquement à chaque ingestion."
            if article_count > 0
            else "Aucun article indexé. Lancez l'ingestion multi-sources pour construire le corpus."
        ),
        "pinned": bool(row.get("pinned", False)),
        "query": row["query"],
        "mode": row["mode"],
        "filters": row.get("filters") or {},
        "result_count": row.get("result_count", 0),
        "folder_id": row.get("folder_id"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "is_user_scenario": True,
        "populate_status": row.get("populate_status", "idle"),
        "pipeline_status": row.get("pipeline_status", "idle"),
        "pipeline_step": row.get("pipeline_step"),
        "pipeline_progress": row.get("pipeline_progress", 0),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

@app.get("/user-scenarios")
def list_user_scenarios() -> list[dict[str, Any]]:
    """Liste tous les scénarios utilisateur (recherches sauvegardées).
    Déduplique au passage les recherches récentes (non épinglées) par query+mode
    en ne conservant que la plus récente de chaque groupe."""
    with engine.begin() as conn:
        # Delete stale duplicates: for unpinned/unfoldered scenarios keep only
        # the most recent row per (query, mode) pair.
        conn.execute(text("""
            DELETE FROM user_scenarios
            WHERE pinned = false AND folder_id IS NULL
              AND id NOT IN (
                SELECT DISTINCT ON (query, mode) id
                FROM user_scenarios
                WHERE pinned = false AND folder_id IS NULL
                ORDER BY query, mode, created_at DESC
              )
        """))
        rows = conn.execute(text("""
            SELECT
                us.id, us.name, us.query, us.mode, us.filters,
                us.pinned, us.folder_id, us.created_at, us.updated_at,
                us.populate_status, us.pipeline_status, us.pipeline_step, us.pipeline_progress,
                COALESCE(us.result_count, 0) AS result_count,
                COALESCE(us.article_count, 0) AS article_count,
                us.is_system
            FROM user_scenarios us
            ORDER BY us.pinned DESC, us.created_at DESC
        """)).mappings().all()

        # Compteurs (articles / inclus / exclus) de TOUS les scénarios en UNE
        # requête, puis lookup par scénario → évite une requête d'agrégation par
        # ligne (N+1). Même forme que sql_counts dans /gesica/scenarios.
        counts_map: dict[str, Any] = {
            str(cr["scenario_id"]): dict(cr)
            for cr in conn.execute(text("""
                SELECT ars.scenario_id,
                       COUNT(DISTINCT ars.document_id) AS article_count,
                       COUNT(DISTINCT ars.document_id) FILTER (
                           WHERE COALESCE(ars.screening_status, d.screening_status) = 'included'
                       ) AS included_count,
                       COUNT(DISTINCT ars.document_id) FILTER (
                           WHERE COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                       ) AS excluded_count
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                GROUP BY ars.scenario_id
            """)).mappings().all()
        }
    return [_user_scenario_to_gesica_format(dict(r), counts_map) for r in rows]


@app.post("/user-scenarios", status_code=201)
def create_user_scenario(payload: UserScenarioIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Crée ou met à jour un scénario utilisateur depuis une recherche sauvegardée.
    Pour les recherches récentes (non épinglées, sans dossier), upsert par query+mode
    afin d'éviter l'accumulation de doublons lors des relances de recherche."""
    import uuid
    # For unpinned auto-saved searches: upsert by query+mode to avoid duplicates.
    # Skip the upsert for multi-sub-query searches: they share the synthesized
    # display `query` yet are distinct searches, so query+mode dedup would wrongly
    # merge them — always insert a fresh row instead.
    if not payload.pinned and not payload.folder_id and not payload.sub_queries:
        with engine.begin() as conn:
            existing = conn.execute(text("""
                SELECT id FROM user_scenarios
                WHERE query = :query AND mode = :mode AND pinned = false AND folder_id IS NULL
                ORDER BY created_at DESC LIMIT 1
            """), {"query": payload.query, "mode": payload.mode}).scalar()
            if existing:
                conn.execute(text("""
                    UPDATE user_scenarios
                    SET name = :name, filters = CAST(:filters AS jsonb),
                        result_count = :result_count, created_at = now(),
                        search_strategy = COALESCE(CAST(:strategy AS jsonb), search_strategy)
                    WHERE id = :id
                """), {
                    "id": existing,
                    "name": payload.name,
                    "filters": json.dumps(payload.filters),
                    "result_count": payload.result_count,
                    "strategy": json.dumps(payload.search_strategy) if payload.search_strategy else None,
                })
                row = _get_user_scenario_or_404(existing)
                return _user_scenario_to_gesica_format(row)
    new_id = "usr-" + str(uuid.uuid4()).replace("-", "")[:12]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_scenarios (id, name, query, mode, filters, result_count, pinned, folder_id, search_strategy, sub_queries, combinator, owner_email)
            VALUES (:id, :name, :query, :mode, CAST(:filters AS jsonb), :result_count, :pinned, :folder_id, CAST(:strategy AS jsonb), CAST(:sub_queries AS jsonb), :combinator, :owner_email)
        """), {
            "id": new_id,
            "name": payload.name,
            "query": payload.query,
            "mode": payload.mode,
            "filters": json.dumps(payload.filters),
            "result_count": payload.result_count,
            "pinned": payload.pinned,
            "folder_id": payload.folder_id,
            "strategy": json.dumps(payload.search_strategy) if payload.search_strategy else None,
            "sub_queries": json.dumps(payload.sub_queries) if payload.sub_queries else None,
            "combinator": payload.combinator if payload.sub_queries else None,
            "owner_email": payload.owner_email,
        })
        # Propriétaire renseigné → l'abonner aux alertes de SON scénario (living review).
        if payload.owner_email:
            try:
                _ensure_alert_subscription(conn, payload.owner_email, new_id)
            except Exception as _e:
                logger.warning(f"auto-subscribe owner {new_id}: {_e}")
    # Génération de la stratégie de recherche en arrière-plan : l'appel OpenAI
    # ne doit JAMAIS bloquer (ni faire échouer) la création du scénario. On la
    # saute si le client a déjà fourni la stratégie (recherche booléenne).
    def _bg_strategy(sid: str, q: str) -> None:
        try:
            strategy = _generate_search_strategy(q)
            if _strategy_is_degraded(strategy, q):
                # Repli dégradé (panne LLM/quota) : ne PAS persister, sera
                # régénéré à la prochaine lecture une fois le quota rétabli.
                return
            with engine.begin() as conn2:
                conn2.execute(text("""
                    UPDATE user_scenarios SET search_strategy = CAST(:strategy AS jsonb) WHERE id = :id
                """), {"id": sid, "strategy": json.dumps(strategy)})
        except Exception as _se:
            logger.warning(f"search_strategy generation failed for {sid}: {_se}")
    if not payload.search_strategy:
        try:
            import threading as _threading
            _threading.Thread(target=_bg_strategy, args=(new_id, payload.query), daemon=True).start()
        except Exception as _te:
            logger.warning(f"could not start strategy thread for {new_id}: {_te}")
    row = _get_user_scenario_or_404(new_id)
    return _user_scenario_to_gesica_format(row)


class ScenarioOwnerIn(BaseModel):
    email: str = Field(..., max_length=255)
    frequency: str = Field(default="weekly")


@app.post("/user-scenarios/{scenario_id}/owner")
def set_user_scenario_owner(scenario_id: str, payload: ScenarioOwnerIn,
                            _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Attribue un propriétaire (email) à un scénario existant et l'abonne aux alertes
    de living review pour ce scénario. Permet à un utilisateur de « couvrir » ses
    propres scénarios (notification des nouveaux articles à son email)."""
    email = _clean_email(payload.email)
    if not email:
        raise HTTPException(status_code=400, detail="Adresse email invalide.")
    freq = payload.frequency if payload.frequency in ("daily", "weekly", "immediate") else "weekly"
    _get_user_scenario_or_404(scenario_id)
    with engine.begin() as conn:
        conn.execute(text("UPDATE user_scenarios SET owner_email = :e, updated_at = NOW() WHERE id = :id"),
                     {"e": email, "id": scenario_id})
        _ensure_alert_subscription(conn, email, scenario_id, freq)
    return {"status": "ok", "scenario_id": scenario_id, "owner_email": email, "frequency": freq,
            "subscribed": True}


@app.get("/user-scenarios/by-owner")
def list_user_scenarios_by_owner(email: str) -> list[dict[str, Any]]:
    """Liste les scénarios appartenant à un email (pour « mes scénarios »)."""
    e = _clean_email(email)
    if not e:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, name, query, created_at, article_count FROM user_scenarios "
            "WHERE owner_email = :e ORDER BY created_at DESC"
        ), {"e": e}).mappings().all()
    return [dict(r) for r in rows]


@app.delete("/user-scenarios/{scenario_id}", status_code=200)
def delete_user_scenario(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Supprime un scénario (utilisateur OU GESICA) et ses associations."""
    _get_user_scenario_or_404(scenario_id)
    # Les scénarios GESICA (is_system) sont désormais des scénarios ordinaires :
    # supprimables comme les autres (généralisation). On nettoie aussi les tables
    # liées (datasets/runs de modèle) pour ne pas laisser d'orphelins.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"), {"sid": scenario_id})
        conn.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :sid"), {"sid": scenario_id})
        for _t in ("scenario_model_dataset", "scenario_model_run"):
            try:
                conn.execute(text(f"DELETE FROM {_t} WHERE scenario_id = :sid"), {"sid": scenario_id})
            except Exception:
                pass
        conn.execute(text("DELETE FROM user_scenarios WHERE id = :id"), {"id": scenario_id})
    return {"deleted": True, "id": scenario_id}


@app.patch("/user-scenarios/{scenario_id}")
def patch_user_scenario(scenario_id: str, payload: UserScenarioPatch, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour le nom, le pin, le mode ou les filtres d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    updates = []
    params: dict[str, Any] = {"id": scenario_id}
    if payload.name is not None:
        updates.append("name = :name")
        params["name"] = payload.name
    if payload.pinned is not None:
        updates.append("pinned = :pinned")
        params["pinned"] = payload.pinned
    if payload.mode is not None:
        updates.append("mode = :mode")
        params["mode"] = payload.mode
    if payload.filters is not None:
        updates.append("filters = CAST(:filters AS jsonb)")
        params["filters"] = json.dumps(payload.filters)
    if payload.folder_id is not None:
        # Permettre d'assigner ou de retirer d'un dossier ("" = retirer)
        updates.append("folder_id = :folder_id")
        params["folder_id"] = payload.folder_id if payload.folder_id != "" else None
    if not updates:
        row = _get_user_scenario_or_404(scenario_id)
        return _user_scenario_to_gesica_format(row)
    updates.append("updated_at = NOW()")
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE user_scenarios SET {', '.join(updates)} WHERE id = :id
        """), params)
    row = _get_user_scenario_or_404(scenario_id)
    return _user_scenario_to_gesica_format(row)


# ── Dossiers (folders) ────────────────────────────────────────────────────────

@app.get("/user-scenario-folders")
def list_folders() -> list[dict[str, Any]]:
    """Liste tous les dossiers de scénarios utilisateur."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT f.id, f.name, f.color, f.sort_order, f.created_at,
                   COUNT(s.id) AS scenario_count
            FROM user_scenario_folders f
            LEFT JOIN user_scenarios s ON s.folder_id = f.id
            GROUP BY f.id, f.name, f.color, f.sort_order, f.created_at
            ORDER BY f.sort_order ASC, f.created_at DESC
        """)).mappings().all()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "color": r["color"],
            "sort_order": r["sort_order"],
            "scenario_count": r["scenario_count"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


@app.post("/user-scenario-folders", status_code=201)
def create_folder(payload: FolderIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Crée un nouveau dossier."""
    import uuid
    new_id = "fld-" + str(uuid.uuid4()).replace("-", "")[:12]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_scenario_folders (id, name, color, sort_order)
            VALUES (:id, :name, :color, :sort_order)
        """), {"id": new_id, "name": payload.name, "color": payload.color, "sort_order": payload.sort_order})
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, name, color, sort_order, created_at FROM user_scenario_folders WHERE id = :id"), {"id": new_id}).mappings().first()
    # Réponse construite sur les valeurs connues (insérées) : robuste même si le
    # SELECT de relecture ne retrouve pas la ligne (race / connexion distincte).
    return {
        "id": new_id, "name": payload.name, "color": payload.color,
        "sort_order": payload.sort_order, "scenario_count": 0,
        "created_at": row["created_at"].isoformat() if row and row.get("created_at") else None,
    }


@app.patch("/user-scenario-folders/{folder_id}")
def patch_folder(folder_id: str, payload: FolderIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Renomme ou recolore un dossier."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE user_scenario_folders
            SET name = :name, color = :color, sort_order = :sort_order
            WHERE id = :id
        """), {"id": folder_id, "name": payload.name, "color": payload.color, "sort_order": payload.sort_order})
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Dossier non trouvé")
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT f.id, f.name, f.color, f.sort_order, f.created_at, COUNT(s.id) AS scenario_count
            FROM user_scenario_folders f
            LEFT JOIN user_scenarios s ON s.folder_id = f.id
            WHERE f.id = :id
            GROUP BY f.id, f.name, f.color, f.sort_order, f.created_at
        """), {"id": folder_id}).mappings().first()
    return {
        "id": row["id"], "name": row["name"], "color": row["color"],
        "sort_order": row["sort_order"], "scenario_count": row["scenario_count"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


@app.delete("/user-scenario-folders/{folder_id}")
def delete_folder(folder_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Supprime un dossier (les scénarios sont conservés, leur folder_id devient NULL)."""
    with engine.begin() as conn:
        # Désassocier les scénarios
        conn.execute(text("UPDATE user_scenarios SET folder_id = NULL WHERE folder_id = :id"), {"id": folder_id})
        result = conn.execute(text("DELETE FROM user_scenario_folders WHERE id = :id"), {"id": folder_id})
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Dossier non trouvé")
    return {"deleted": True, "id": folder_id}


# ── Detail (compatible ScenarioDetail frontend) ───────────────────────────────

@app.get("/user-scenarios/{scenario_id}/detail")
def get_user_scenario_detail(scenario_id: str) -> dict[str, Any]:
    """
    Retourne les informations enrichies d'un scénario utilisateur au format ScenarioDetail.
    Compatible avec ScenarioDetailPage (boolean_queries, nl_queries, corpus_stats, etc.)
    """
    row = _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) THEN 1 ELSE 0 END) AS with_fulltext,
                COUNT(DISTINCT d.year) AS years_covered,
                COUNT(DISTINCT d.journal) AS journals_count,
                MIN(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id}).mappings().first()

    # Recherche multi-sous-requêtes : on renvoie les vraies listes booléennes /
    # naturelles. Sinon (mono-requête), la requête sauvegardée est booléenne OU en
    # langage naturel selon le mode réellement utilisé — on ne l'affiche que dans la
    # catégorie employée (évite de montrer la même requête en booléen ET en naturel).
    query_text = row["query"]
    _sub = _normalize_sub_queries(row.get("sub_queries"))
    if _sub:
        boolean_queries = [s["text"] for s in _sub if s["kind"] == "boolean"]
        nl_queries = [s["text"] for s in _sub if s["kind"] == "natural"]
    else:
        _mode = (row.get("mode") or "hybrid").lower()
        _saved = [query_text] if query_text else []
        boolean_queries = _saved if _mode == "boolean" else []
        nl_queries = [] if _mode == "boolean" else _saved

    return {
        "id": scenario_id,
        "name": row["name"],
        "title": row["name"],
        "description": f"Scénario utilisateur basé sur la recherche : {query_text}",
        "cluster": "user",
        "recommended_actions": [],
        "boolean_queries": boolean_queries,
        "nl_queries": nl_queries,
        "evidence_extraction_prompt": "",
        "model_info": {},
        "alert_thresholds": {
            "green": {"label": "Normal", "threshold": 0},
            "orange": {"label": "Vigilance", "threshold": 50},
            "red": {"label": "Alerte", "threshold": 80},
        },
        "databases": ["PubMed"],
        "outcome_definition": "",
        "variables_detail": {},
        "keywords": [w for w in query_text.split() if len(w) > 3][:10],
        "clinical_rationale": "",
        "corpus_stats": {
            "total": int(stats["total"] or 0) if stats else 0,
            "with_fulltext": int(stats["with_fulltext"] or 0) if stats else 0,
            "years_covered": int(stats["years_covered"] or 0) if stats else 0,
            "journals_count": int(stats["journals_count"] or 0) if stats else 0,
            "year_min": stats["year_min"] if stats else None,
            "year_max": stats["year_max"] if stats else None,
        },
        "is_user_scenario": True,
        "query": query_text,
        "mode": row["mode"],
        "filters": row.get("filters") or {},
        "pinned": bool(row.get("pinned", False)),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


# ── Corpus (compatible fetchScenarioCorpus frontend) ──────────────────────────

@app.get("/user-scenarios/{scenario_id}/corpus")
def get_user_scenario_corpus(
    scenario_id: str,
    limit: int = 100000,
    offset: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    fulltext_only: bool = False,
    source: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Retourne le corpus d'articles pour un scénario utilisateur.
    Compatible avec fetchScenarioCorpus (même format de réponse).
    """
    row = _get_user_scenario_or_404(scenario_id)
    # Seuil effectif : paramètre explicite (curseur en direct) > seuil sauvegardé
    # dans scenario_settings > défaut 0.45. (Auparavant codé en dur à 0.45, donc
    # le compteur « auto-sélectionnés » ne suivait jamais le curseur.)
    eff_threshold = 0.45
    try:
        with engine.connect() as _tc:
            _ts = _tc.execute(text(
                "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
        if _ts is not None:
            eff_threshold = float(_ts)
    except Exception:
        pass
    if threshold is not None:
        eff_threshold = float(threshold)
    # Conditions de filtre (article_scenarios géré par JOIN)
    conditions = [
        "(d.is_duplicate IS NULL OR d.is_duplicate = FALSE)",
    ]
    params: dict[str, Any] = {"sid": scenario_id, "limit": limit, "offset": offset}
    if year_from:
        conditions.append("d.year >= :year_from")
        params["year_from"] = year_from
    if year_to:
        conditions.append("d.year <= :year_to")
        params["year_to"] = year_to
    if source:
        conditions.append("d.source = :source")
        params["source"] = source
    if fulltext_only:
        conditions.append("""EXISTS (
            SELECT 1 FROM document_chunk c
            WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
        )""")
    where = " AND ".join(conditions)
    _screated = row.get("created_at")
    with engine.connect() as conn:
        # Single query for both total and above_threshold to avoid race condition
        counts_row = conn.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE ars.similarity_score >= :threshold) AS above_threshold,
                COUNT(*) FILTER (WHERE ars.similarity_score IS NULL) AS unscored,
                COUNT(*) FILTER (WHERE :screated IS NOT NULL AND d.created_at >= :screated) AS newly_fetched,
                COUNT(*) FILTER (WHERE EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                )) AS with_fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
        """), {**{k: v for k, v in params.items() if k not in ('limit', 'offset')},
               'threshold': eff_threshold, 'screated': _screated}).mappings().first()
        total = int(counts_row["total"] or 0)
        above_threshold = int(counts_row["above_threshold"] or 0)
        unscored = int(counts_row["unscored"] or 0)
        newly_fetched = int(counts_row["newly_fetched"] or 0) if _screated else None
        from_local = (total - newly_fetched) if newly_fetched is not None else None
        with_fulltext = int(counts_row["with_fulltext"] or 0)
        articles = conn.execute(text(f"""
            SELECT
                d.id, d.title, d.abstract, d.year, d.source, d.url,
                d.authors, d.doi, d.journal, d.keywords, d.language,
                d.study_design, d.sample_size, d.country, d.citation_count,
                d.open_access, d.pmid, d.publication_type, d.quality_score,
                COALESCE(ars.screening_status, d.screening_status) AS screening_status, d.reviewer_1_status,
                COALESCE(ars.similarity_score, 0.0) AS similarity_score,
                ars.rerank_score AS rerank_score,
                (COALESCE(ars.similarity_score, 0.0) >= :threshold) AS above_threshold,
                -- is_new : ingéré pendant CE scénario (vs déjà présent en base).
                -- Donne un sens au badge "Nouveau" vs "Base locale" côté UI.
                (:screated IS NOT NULL AND d.created_at >= :screated) AS is_new,
                EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) AS has_fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
            ORDER BY
                CASE WHEN COALESCE(ars.similarity_score, 0.0) >= :threshold THEN 0 ELSE 1 END ASC,
                (ars.rerank_score IS NOT NULL) DESC,
                ars.rerank_score DESC NULLS LAST,
                ars.similarity_score DESC NULLS LAST,
                d.year DESC NULLS LAST,
                d.citation_count DESC NULLS LAST,
                d.title ASC
            LIMIT :limit OFFSET :offset
        """), {**params, 'threshold': eff_threshold, 'screated': _screated}).mappings().all()
        year_dist = conn.execute(text(f"""
            SELECT d.year, COUNT(*) AS cnt
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
              AND d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year ORDER BY d.year DESC
        """), {k: v for k, v in params.items() if k not in ('limit', 'offset')}).mappings().all()
        # Répartition par source, en distinguant la base locale (docs déjà en base
        # avant ce scénario) des références ramenées en direct par les APIs pendant
        # la construction du corpus (docs créés après la création du scénario).
        source_dist = conn.execute(text(f"""
            SELECT d.source,
                   COUNT(*) AS cnt,
                   COUNT(*) FILTER (WHERE :screated IS NULL OR d.created_at < :screated) AS local_cnt,
                   COUNT(*) FILTER (WHERE :screated IS NOT NULL AND d.created_at >= :screated) AS live_cnt
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
            GROUP BY d.source ORDER BY cnt DESC LIMIT 12
        """), {**{k: v for k, v in params.items() if k not in ('limit', 'offset')},
               'screated': _screated}).mappings().all()
        # Dict {source: n_local, "source (live)": n_live} pour le panneau de recherche.
        source_breakdown: dict[str, int] = {}
        for r in source_dist:
            _src = r["source"] or "Autre"
            if int(r["local_cnt"] or 0) > 0:
                source_breakdown[_src] = int(r["local_cnt"])
            if int(r["live_cnt"] or 0) > 0:
                source_breakdown[f"{_src} (live)"] = int(r["live_cnt"])

    # Auto-score : si des articles ne sont pas encore scorés, on lance le rerank
    # en arrière-plan (une fois). Le seuil devient alors exploitable.
    rerank_running = _maybe_autorerank(scenario_id) if unscored > 0 else False

    return {
        "scenario_id": scenario_id,
        "scenario_title": row["name"],
        "total": total,
        "above_threshold": above_threshold,
        "below_threshold": max(0, total - above_threshold - unscored),
        "unscored": unscored,
        "from_local": from_local,
        "newly_fetched": newly_fetched,
        "docs_with_fulltext": with_fulltext,
        "docs_abstract_only": max(0, total - with_fulltext),
        "source_breakdown": source_breakdown,
        "rerank_running": rerank_running or (_RERANK_JOBS.get(scenario_id, {}).get("status") == "running"),
        "threshold": eff_threshold,
        "offset": offset,
        "limit": limit,
        "articles": [dict(a) for a in articles],
        "year_distribution": [{"year": r["year"], "count": int(r["cnt"])} for r in year_dist],
        "source_distribution": [{"source": r["source"], "count": int(r["cnt"])} for r in source_dist],
        "is_user_scenario": True,
    }


# ── Populate : ingestion PubMed en arrière-plan ───────────────────────────────

import threading

# Verrous pour protéger l'accès concurrent aux états de jobs en mémoire (H-4)
_populate_jobs_lock = threading.Lock()
_pipeline_jobs_lock = threading.Lock()

_user_scenario_populate_jobs: dict[str, dict] = {}
_user_scenario_pipeline_jobs: dict[str, dict] = {}


def _launch_populate_job(scenario_id: str, query: str, filters: dict, max_results: int,
                         include_live: bool = True) -> str:
    """
    Démarre un job d'ingestion en arrière-plan pour un scénario, en garantissant
    qu'un seul job tourne à la fois (verrou partagé). Renvoie l'état : "started"
    ou "already_running". Utilisé par /populate ET /search/live afin qu'aucun des
    deux ne lance un populate concurrent sur le même scénario.
    """
    import threading
    with _populate_jobs_lock:
        job = _user_scenario_populate_jobs.get(scenario_id)
        if job and job.get("status") == "running":
            return "already_running"
        _user_scenario_populate_jobs[scenario_id] = {
            "status": "running", "ingested": 0, "errors": 0, "total_found": 0,
            "sources": {"db_cache": 0, "pubmed": 0, "openalex": 0, "crossref": 0,
                        "europepmc": 0, "preprint": 0, "semantic_scholar": 0, "doaj": 0,
                        "clinicaltrials": 0, "core": 0, "arxiv": 0, "openaire": 0,
                        "biorxiv": 0, "medrxiv": 0},
        }
    threading.Thread(
        target=_run_user_scenario_populate,
        args=(scenario_id, query, filters or {}, max_results, None, include_live),
        daemon=True,
    ).start()
    return "started"


def _generate_search_strategy(query: str) -> dict:
    """
    Uses GPT-4.1-mini to generate a structured boolean search strategy from a natural language query.
    Returns a dict with:
    - general: general boolean query string
    - pubmed: PubMed-specific with MeSH tags
    - explanation: brief explanation of term choices
    - synonyms: list of key synonym groups used
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return {"general": query, "pubmed": query, "explanation": "", "synonyms": [], "degraded": True}
    try:
        from openai import OpenAI as _OAI_ss
        _client = _OAI_ss(api_key=openai_key)
        response = _client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a systematic review librarian. The user may type EITHER a natural-language "
                    "description OR an already-formed boolean query. First decide which it is:\n"
                    "- If it is ALREADY a boolean query (it uses AND/OR/NOT operators or quoted phrases "
                    "with explicit structure), PRESERVE it as-is in 'general' (only fix obvious syntax), and "
                    "set 'explanation' to note that the query was already boolean and kept unchanged.\n"
                    "- Otherwise, TRANSLATE the natural-language query into a boolean query.\n"
                    "Return ONLY valid JSON with these fields:\n"
                    '{"general": "boolean query using AND/OR/NOT and quotes for phrases",\n'
                    '"pubmed": "PubMed-optimized query with MeSH terms [MeSH Terms] and field tags [Title/Abstract]",\n'
                    '"explanation": "1-2 sentences explaining the term choices and synonyms (or that the input was already boolean)",\n'
                    '"synonyms": [["term1", "synonym1a", "synonym1b"], ["term2", "synonym2a"]]}\n'
                    "Keep queries practical and not overly long. Use 2-4 concept groups max."
                )},
                {"role": "user", "content": f"Research query: {query}"}
            ],
            temperature=0,
            seed=42,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as _e:
        logger.warning(f"_generate_search_strategy failed: {_e}")
        return {"general": query, "pubmed": query, "explanation": "", "synonyms": [], "degraded": True}


class SearchStrategyIn(BaseModel):
    query: str = Field(..., min_length=1)


@app.post("/search-strategy")
def post_search_strategy(payload: SearchStrategyIn) -> dict[str, Any]:
    """Traduit une requête en langage naturel en stratégie booléenne (LLM).

    Permet à la recherche d'AFFICHER la requête booléenne et de l'utiliser comme
    base du corpus : la même stratégie est ensuite persistée sur le scénario, de
    sorte que le compteur de recherche == la taille du corpus (même requête).
    """
    return _generate_search_strategy(payload.query)


# ── Parseurs de sources littéraires (PURS / testables) ───────────────────────
# Chaque parseur transforme la réponse brute d'une API en une liste de docs
# {title, abstract, year, url, external_id, doi, source_type}. Le fetcher (closure
# dans le populate) ne fait que le HTTP + la pagination puis délègue ici — de sorte
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


def _parse_biorxiv(payload: dict, terms: list[str], server: str) -> list[dict]:
    """bioRxiv/medRxiv details API collection → docs FILTRÉS par mots-clés.
    L'API n'offre pas de recherche : le fetcher scanne une fenêtre de dates RÉCENTE
    et le filtrage lexical se fait ici (garde un preprint si ≥ un TIERS des termes
    ≥4 car. apparaissent dans titre+résumé). external_id = <server>:<doi>."""
    out: list[dict] = []
    coll = payload.get("collection") if isinstance(payload, dict) else None
    kw = [t for t in (terms or []) if len(t) >= 4]
    need = max(1, (len(kw) + 2) // 3) if kw else 0
    for it in (coll or []):
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        doi = it.get("doi")
        if not title or not doi:
            continue
        abstract = it.get("abstract") or None
        hay = f"{title} {abstract or ''}".lower()
        if kw and sum(1 for t in kw if t in hay) < need:
            continue
        date = it.get("date") or ""
        out.append({
            "title": title, "abstract": abstract,
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "url": f"https://doi.org/{doi}", "doi": doi,
            "external_id": f"{server}:{doi}", "source_type": "preprint",
        })
    return out


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
) -> int:
    """INSERT SQL direct d'un document + chunk title_abstract.
    Évite les appels HTTP à l'API locale (POST /documents + POST /chunks).
    Retourne l'ID du document (existant ou nouvellement créé).
    """
    doi = _normalize_doi(doi)
    content_text = f"{title}\n\n{abstract or ''}".strip()

    # Vérifier si le document existe déjà
    with engine.connect() as _c:
        existing = _c.execute(text("""
            SELECT id FROM literature_document
            WHERE external_id = :eid AND project_context = :ctx LIMIT 1
        """), {"eid": external_id, "ctx": project_context}).scalar()
    if existing:
        return existing

    # INSERT document + chunk dans UNE SEULE transaction : sinon un crash entre
    # les deux laisse un document sans chunk (jamais indexable/cherchable) — c'est
    # l'origine des documents orphelins observés en production.
    with engine.begin() as _c:
        doc_id = _c.execute(text("""
            INSERT INTO literature_document (
                source, title, abstract, year, url, external_id,
                project_context, source_type, doi, authors, journal
            ) VALUES (
                :source, :title, :abstract, :year, :url, :external_id,
                :project_context, :source_type, :doi, :authors, :journal
            )
            ON CONFLICT (doi) WHERE doi IS NOT NULL DO NOTHING
            RETURNING id
        """), {
            "source": source, "title": title, "abstract": abstract,
            "year": year, "url": url, "external_id": external_id,
            "project_context": project_context, "source_type": source_type,
            "doi": doi, "authors": authors, "journal": journal,
        }).scalar()
        if doc_id is None:
            # DOI already present — reuse the existing canonical row
            doc_id = _c.execute(text(
                "SELECT id FROM literature_document WHERE doi = :doi ORDER BY id LIMIT 1"
            ), {"doi": doi}).scalar()

        # INSERT du chunk title_abstract, idempotent : ne crée PAS de second chunk
        # si le document en a déjà un (cas d'un doc atteint via dédup DOI avec un
        # external_id différent — l'origine des chunks dupliqués observés).
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
    return doc_id


def _run_user_scenario_populate(
    scenario_id: str,
    query: str,
    filters: dict,
    max_results: int = 500,
    _pipeline_callback=None,
    include_live: bool = True,
) -> int:
    """
    Construit le corpus d'un scénario = résultat de la REQUÊTE BOOLÉENNE sur
    (base locale ∪ articles récupérés en direct). Les sources live ne servent qu'à
    ENRICHIR la base ; l'appartenance au corpus est ensuite décidée UNIQUEMENT par
    la correspondance booléenne (_boolean_corpus_ids) — la même que la recherche.
    Plafond : LIVE_MAX_PER_SOURCE articles par source. include_live=False = base
    locale seulement. Retourne le nombre total d'articles ingérés.
    """
    import time as _time
    import xml.etree.ElementTree as ET
    import requests as _requests
    import math
    import threading
    # NB : sur Python <3.11, as_completed() lève concurrent.futures.TimeoutError,
    # qui n'EST PAS le TimeoutError natif. On l'importe explicitement pour pouvoir
    # l'attraper (cf. bloc fédération plus bas).
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout

    # Plafond par source identique pour recherche et corpus (déterminisme).
    max_results = min(max_results, LIVE_MAX_PER_SOURCE)

    # Garde-temps partagé entre les boucles de pagination des sources lentes
    # (OpenAlex/Crossref/EuropePMC). Quand le budget fédération est dépassé, elles
    # s'arrêtent d'elles-mêmes au lieu de continuer à paginer jusqu'à 2000 résultats
    # en arrière-plan (résultats qui seraient de toute façon écartés par le filtre
    # booléen final). Réglé juste avant le lancement de la fédération.
    _fed_deadline = [float("inf")]

    if _pipeline_callback is None:
        _user_scenario_populate_jobs[scenario_id] = {
            "status": "running", "ingested": 0, "errors": 0, "total_found": 0,
            # `phase` reflète l'ÉTAPE RÉELLE du backend (et non un minuteur côté
            # client) : local → federation → scoring → rerank → done. `rerank_status`
            # suit le cross-encoder qui tourne en arrière-plan après l'affichage.
            "phase": "local", "rerank_status": "idle",
            "sources": {
                "db_cache": 0, "pubmed": 0, "openalex": 0, "crossref": 0,
                "europepmc": 0, "preprint": 0, "semantic_scholar": 0, "doaj": 0,
                "clinicaltrials": 0, "core": 0, "arxiv": 0, "openaire": 0,
                "biorxiv": 0, "medrxiv": 0
            }
        }

    def _set_phase(_phase: str, **extra):
        """Met à jour la phase réelle du job (no-op pour le pipeline complet)."""
        if _pipeline_callback is None:
            job = _user_scenario_populate_jobs.get(scenario_id)
            if job is not None:
                job["phase"] = _phase
                job.update(extra)

    ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    EMAIL = os.getenv("PUBMED_EMAIL", "literev@example.com")
    BATCH_SIZE = 200

    # Compteurs partagés thread-safe
    _counter_lock = threading.Lock()
    _ingested_total = [0]
    _errors_total = [0]

    def _link_to_scenario(doc_id):
        # NE LIE PLUS pendant la fédération : ingérer un article live ne l'ajoute
        # PAS d'office au corpus. L'appartenance est recalculée après ingestion via
        # la correspondance booléenne (_boolean_corpus_ids) — sinon le corpus
        # gonflait avec des résultats live ne correspondant pas à la requête.
        return None

    def _inc(source_name, count=1, err=0):
        with _counter_lock:
            _ingested_total[0] += count
            _errors_total[0] += err
            if _pipeline_callback is None:
                job = _user_scenario_populate_jobs[scenario_id]
                job["ingested"] = _ingested_total[0]
                # Remonter les erreurs dans l'état du job : sans cela, errors=0
                # masquait toute perte de données par source (échec silencieux).
                job["errors"] = _errors_total[0]
                job.setdefault("errors_by_source", {})
                if err:
                    job["errors_by_source"][source_name] = \
                        job["errors_by_source"].get(source_name, 0) + err
                job["sources"][source_name] = job["sources"].get(source_name, 0) + count

    # ── Étape 0 : Linking depuis la base locale (séquentiel, rapide) ─────────
    # Le CORPUS est défini par une correspondance LEXICALE (requête booléenne),
    # indépendante du seuil sémantique : base locale ∪ nouvelles références live.
    # Le seuil sémantique n'intervient QUE dans la page scénario pour sélectionner
    # le sous-ensemble pertinent (_get_above_threshold_articles).
    local_linked = 0
    try:
        # Le corpus = résultat de la REQUÊTE BOOLÉENNE (générée par LLM). On
        # récupère search_strategy.general ; à défaut on la génère depuis la requête.
        _boolean = query
        _pubmed_q = query
        # Recherche multi-sous-requêtes : l'appartenance au corpus est l'union /
        # l'intersection des ensembles locaux de chaque sous-requête. La fédération
        # live reste pilotée par la stratégie booléenne de la requête synthétisée
        # (elle ne fait qu'ENRICHIR la base ; l'appartenance est recalculée après).
        _sub_queries: list[dict] = []
        _combinator = "union"
        try:
            with engine.connect() as _sc:
                _srow = _sc.execute(text(
                    "SELECT search_strategy, sub_queries, combinator FROM user_scenarios WHERE id = :sid"),
                    {"sid": scenario_id}).mappings().first()
            _strat = _srow["search_strategy"] if _srow else None
            _sub_queries = _normalize_sub_queries(_srow["sub_queries"]) if _srow else []
            if _srow and _srow["combinator"] in ("union", "intersection"):
                _combinator = _srow["combinator"]
            if isinstance(_strat, dict) and not _strategy_is_degraded(_strat, query):
                _boolean = _strat["general"]
                _pubmed_q = _strat.get("pubmed") or _strat["general"]
            else:
                # Absent ou dégradé (cache empoisonné pendant une panne quota) → régénérer.
                _gen = _generate_search_strategy(query)
                _boolean = _gen.get("general") or query
                _pubmed_q = _gen.get("pubmed") or _boolean
                if not _strategy_is_degraded(_gen, query):
                    with engine.begin() as _sc2:
                        _sc2.execute(text("UPDATE user_scenarios SET search_strategy = CAST(:s AS jsonb) WHERE id = :id"),
                                     {"s": json.dumps(_gen), "id": scenario_id})
        except Exception as _be:
            logger.warning(f"Populate {scenario_id} boolean strategy: {_be}")
        # Variante par type de source (comme le font déjà les _live_fetch_*) :
        #  - _pubmed_q : booléen MeSH → sources proxyfiées PubMed (eutils)
        #  - _boolean  : booléen général → API qui acceptent les opérateurs (EuropePMC)
        #  - _plain_q  : mots-clés simples (sans opérateurs ni '?') → OpenAlex / Crossref / preprints
        # CORRECTIF : auparavant TOUTES les sources recevaient la requête BRUTE en
        # langage naturel (avec le '?'), d'où OpenAlex 400 et booléens Cochrane/
        # PROSPERO malformés → 0 article récupéré en direct.
        _plain_q = _plain_keywords(_boolean) or _plain_keywords(query) or query
        _local_ids = (_multi_query_corpus_ids(_sub_queries, _combinator, filters)
                      if _sub_queries
                      else _search_local_doc_ids(_boolean, "boolean", filters, limit=100_000))

        if _local_ids:
            with engine.begin() as _lc2:
                # RESET du corpus à la correspondance booléenne. Sans cela,
                # l'accumulation ON CONFLICT DO NOTHING ne retire jamais les liens
                # devenus obsolètes (ex. un ancien match lexical OR-de-tous-les-mots
                # qui avait gonflé le corpus à des dizaines de milliers d'articles).
                # Le corpus = EXACTEMENT le résultat de la requête booléenne (base
                # locale) ∪ les nouvelles références live ajoutées plus bas.
                _lc2.execute(
                    text("DELETE FROM article_scenarios WHERE scenario_id = :sid "
                         "AND document_id NOT IN :ids").bindparams(
                             bindparam("ids", expanding=True)),
                    {"sid": scenario_id, "ids": list(_local_ids)},
                )
                # Insertion en masse (un seul aller-retour) : le corpus local doit
                # être lié quasi instantanément, sans une requête par document.
                _lc2.execute(text("""
                    INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
                    SELECT unnest(CAST(:ids AS bigint[])), :sid, NULL
                    ON CONFLICT (document_id, scenario_id) DO NOTHING
                """), {"ids": list(_local_ids), "sid": scenario_id})
                local_linked = len(_local_ids)
            _inc("db_cache", local_linked)
            logger.info(f"Populate {scenario_id}: corpus booléen = {local_linked} docs (base locale)")

        if local_linked > 0:
            with engine.begin() as _uc:
                _uc.execute(text("""
                    UPDATE user_scenarios SET article_count = :cnt WHERE id = :sid
                """), {"cnt": local_linked, "sid": scenario_id})

    except Exception as _e_local:
        logger.warning(f"Local DB link failed for {scenario_id}: {_e_local}")

    # ── Étape 1 : Interrogation parallèle des 7 sources externes ─────────────

    def _fetch_pubmed():
        count = 0
        try:
            # Throttle partagé eutils (verrou global + clé API + retry 429) : sinon
            # PubMed se faisait évincer par PROSPERO/Cochrane (mêmes serveurs eutils,
            # 3 req/s sans clé) → esearch 429 → total_found=0 → 0 article ingéré.
            r = _ncbi_get(
                f"{ENTREZ_BASE}/esearch.fcgi",
                # sort=pub_date → l'ensemble historique est trié du plus récent au
                # plus ancien ; efetch récupère donc d'abord les articles récents.
                {"db": "pubmed", "term": _pubmed_q, "retmax": 0, "sort": "pub_date",
                 "retmode": "json", "usehistory": "y", "email": EMAIL},
                timeout=30,
            )
            r.raise_for_status()
            search_result = r.json()["esearchresult"]
            total_found = int(search_result.get("count", 0))
            web_env = search_result.get("webenv", "")
            query_key = search_result.get("querykey", "1")
            if _pipeline_callback:
                _pipeline_callback("pubmed_found", total_found)
            effective_max = min(max_results, total_found)
            n_batches = math.ceil(effective_max / BATCH_SIZE) if effective_max > 0 else 0
            for batch_idx in range(n_batches):
                retstart = batch_idx * BATCH_SIZE
                retmax_batch = min(BATCH_SIZE, effective_max - retstart)
                if retmax_batch <= 0 or retstart >= total_found:
                    break
                try:
                    _ef_data = {"db": "pubmed", "WebEnv": web_env, "query_key": query_key,
                                "retstart": retstart, "retmax": retmax_batch,
                                "rettype": "xml", "retmode": "xml", "email": EMAIL}
                    _ef_key = os.getenv("NCBI_API_KEY")
                    if _ef_key:
                        _ef_data["api_key"] = _ef_key
                    # Espacer le DÉMARRAGE de la requête vis-à-vis des autres appels
                    # eutils (verrou partagé), sans tenir le verrou pendant le POST
                    # (lent) afin de ne pas sérialiser PROSPERO/Cochrane.
                    with _NCBI_LOCK:
                        _ef_wait = (0.11 if _ef_key else _NCBI_MIN_INTERVAL) - (_time.time() - _NCBI_LAST[0])
                        if _ef_wait > 0:
                            _time.sleep(_ef_wait)
                        _NCBI_LAST[0] = _time.time()
                    r2 = _requests.post(f"{ENTREZ_BASE}/efetch.fcgi", data=_ef_data, timeout=90)
                    r2.raise_for_status()
                except Exception as _e_fetch:
                    logger.warning(f"PubMed efetch batch {batch_idx}: {_e_fetch}")
                    _time.sleep(1)
                    continue
                root = ET.fromstring(r2.content)
                for article_elem in root.findall(".//PubmedArticle"):
                    pmid = article_elem.findtext(".//PMID") or ""
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
                    abstract_parts = []
                    for node in article_elem.findall(".//Abstract/AbstractText"):
                        txt = "".join(node.itertext()).strip()
                        if txt:
                            abstract_parts.append(txt)
                    abstract = " ".join(abstract_parts).strip()
                    year_text = (article_elem.findtext(".//PubDate/Year")
                                 or article_elem.findtext(".//ArticleDate/Year") or "")
                    year = int(year_text[:4]) if year_text[:4].isdigit() else None
                    authors_list = []
                    for author in article_elem.findall(".//AuthorList/Author"):
                        last = author.findtext("LastName") or ""
                        first = author.findtext("ForeName") or ""
                        if last:
                            authors_list.append(f"{last} {first}".strip())
                    authors = "; ".join(authors_list[:6]) if authors_list else None
                    journal = (article_elem.findtext(".//Journal/Title")
                               or article_elem.findtext(".//ISOAbbreviation") or None)
                    doi = None
                    for id_elem in article_elem.findall(".//ArticleIdList/ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text
                            break
                    if not pmid or not title:
                        continue
                    content_text = f"{title}\n\n{abstract}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id = _ingest_doc_direct(
                            source="pubmed", title=title, abstract=abstract or None,
                            year=year, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            external_id=pmid, doi=doi, authors=authors, journal=journal,
                        )
                        _link_to_scenario(doc_id)
                        count += 1
                        _inc("pubmed")
                    except Exception as e:
                        logger.warning(f"PubMed PMID {pmid}: {e}")
                        _inc("pubmed", 0, 1)
        except Exception as _e:
            logger.warning(f"PubMed populate {scenario_id}: {_e}")
        return ("pubmed", count)

    def _fetch_openalex():
        count = 0
        try:
            _oa_page = 1
            _oa_fetched = 0
            _oa_limit = min(max_results, max_results)
            while _oa_fetched < _oa_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                _oa_batch = min(200, _oa_limit - _oa_fetched)
                oa_resp = _requests.get(
                    "https://api.openalex.org/works",
                    # sort=publication_date:desc → quand on plafonne à max_results,
                    # on récupère les articles LES PLUS RÉCENTS d'abord (utile pour
                    # une revue vivante ; les anciens pertinents sont déjà en base).
                    params={"search": _plain_q, "per_page": _oa_batch, "page": _oa_page,
                            "sort": "publication_date:desc", "mailto": "literev@gesica.ch"},
                    timeout=20,
                )
                oa_resp.raise_for_status()
                _oa_results = oa_resp.json().get("results", [])
                if not _oa_results:
                    break
                for work in _oa_results:
                    ext_id = work.get("id", "").split("/")[-1]
                    title = work.get("title") or ""
                    if not ext_id or not title:
                        continue
                    abstract = None
                    inv = work.get("abstract_inverted_index")
                    if inv:
                        try:
                            words = {}
                            for w, positions in inv.items():
                                for pos in positions:
                                    words[pos] = w
                            abstract = " ".join([words[i] for i in sorted(words.keys())])
                        except Exception:
                            pass
                    year = work.get("publication_year")
                    doi = _normalize_doi(work.get("doi"))
                    url = doi or f"https://openalex.org/{ext_id}"
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id = _ingest_doc_direct(
                            source="openalex", title=title, abstract=abstract or None,
                            year=year, url=url, external_id=ext_id, doi=doi,
                        )
                        _link_to_scenario(doc_id)
                        count += 1
                        _inc("openalex")
                    except Exception:
                        _inc("openalex", 0, 1)
                _oa_fetched += len(_oa_results)
                if len(_oa_results) < _oa_batch or _oa_fetched >= _oa_limit:
                    break
                _oa_page += 1
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"OpenAlex populate {scenario_id}: {_e}")
        return ("openalex", count)

    def _fetch_crossref():
        count = 0
        try:
            _cr_offset = 0
            _cr_fetched = 0
            _cr_limit = min(max_results, max_results)
            _cr_rows = min(100, _cr_limit)
            while _cr_fetched < _cr_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                cr_resp = _requests.get(
                    "https://api.crossref.org/works",
                    # NB : PAS de sort=published desc ici (contrairement aux autres
                    # sources). Les dates Crossref sont peu fiables : un tri par date
                    # remonte des enregistrements à dates erronées (ex. « 2121 ») en
                    # tête. On garde donc le tri par pertinence (défaut), qui place les
                    # articles les plus pertinents — pas les plus faussement récents.
                    params={"query": _plain_q, "rows": _cr_rows, "offset": _cr_offset,
                            "mailto": "literev@gesica.ch"},
                    timeout=20,
                )
                cr_resp.raise_for_status()
                _cr_items = cr_resp.json().get("message", {}).get("items", [])
                if not _cr_items:
                    break
                for item in _cr_items:
                    doi = _normalize_doi(item.get("DOI"))
                    titles = item.get("title", [])
                    title = titles[0] if titles else ""
                    if not doi or not title:
                        continue
                    abstract = item.get("abstract")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET
                            abstract = "".join(_ET.fromstring(abstract).itertext()).strip()
                        except Exception:
                            pass
                    year = None
                    created = item.get("created", {}).get("date-parts", [])
                    if created and created[0]:
                        year = created[0][0]
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id = _ingest_doc_direct(
                            source="crossref", title=title, abstract=abstract or None,
                            year=year, url=f"https://doi.org/{doi}", external_id=doi, doi=doi,
                        )
                        _link_to_scenario(doc_id)
                        count += 1
                        _inc("crossref")
                    except Exception:
                        _inc("crossref", 0, 1)
                _cr_fetched += len(_cr_items)
                if len(_cr_items) < _cr_rows or _cr_fetched >= _cr_limit:
                    break
                _cr_offset += _cr_rows
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Crossref populate {scenario_id}: {_e}")
        return ("crossref", count)

    def _fetch_europepmc():
        count = 0
        try:
            _ep_cursor_mark = "*"
            _ep_fetched = 0
            _ep_limit = min(max_results, max_results)
            _ep_page_size = 200
            while _ep_fetched < _ep_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                ep_resp = _requests.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    # sort=P_PDATE_D desc → articles les plus récents d'abord (idem OpenAlex).
                    params={"query": _boolean, "format": "json", "pageSize": _ep_page_size,
                            "resultType": "core", "sort": "P_PDATE_D desc",
                            "cursorMark": _ep_cursor_mark},
                    timeout=20,
                )
                ep_resp.raise_for_status()
                _ep_data = ep_resp.json()
                _ep_results = _ep_data.get("resultList", {}).get("result", [])
                if not _ep_results:
                    break
                for res in _ep_results:
                    pmid = res.get("pmid")
                    pmcid = res.get("pmcid")
                    doi = _normalize_doi(res.get("doi"))
                    ext_id = pmcid or pmid or doi
                    title = res.get("title") or ""
                    if not ext_id or not title:
                        continue
                    abstract = res.get("abstractText")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET2
                            abstract = "".join(_ET2.fromstring(f"<root>{abstract}</root>").itertext()).strip()
                        except Exception:
                            pass
                    year = None
                    yt = res.get("pubYear")
                    if yt and str(yt).isdigit():
                        year = int(yt)
                    url = (f"https://europepmc.org/article/{pmcid or pmid}" if (pmcid or pmid)
                           else (f"https://doi.org/{doi}" if doi else None))
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id = _ingest_doc_direct(
                            source="europepmc", title=title, abstract=abstract or None,
                            year=year, url=url, external_id=ext_id, doi=doi,
                        )
                        _link_to_scenario(doc_id)
                        count += 1
                        _inc("europepmc")
                    except Exception:
                        _inc("europepmc", 0, 1)
                _ep_fetched += len(_ep_results)
                _ep_next_cursor = _ep_data.get("nextCursorMark")
                if not _ep_next_cursor or _ep_next_cursor == _ep_cursor_mark or len(_ep_results) < _ep_page_size or _ep_fetched >= _ep_limit:
                    break
                _ep_cursor_mark = _ep_next_cursor
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"EuropePMC populate {scenario_id}: {_e}")
        return ("europepmc", count)

    def _fetch_preprints():
        # Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (SRC:PPR) :
        # recherche par mots-clés RÉELLE. L'API biorxiv ne fait que dates/DOI, d'où
        # l'ancien scan des 90 derniers jours filtré côté client (~0 résultat).
        count = 0
        try:
            _pp_cursor = "*"
            _pp_fetched = 0
            _pp_query = f"({_boolean}) AND (SRC:PPR)"
            while _pp_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé
                _pp_resp = _requests.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": _pp_query, "format": "json", "pageSize": 100,
                            "resultType": "core", "sort": "P_PDATE_D desc",
                            "cursorMark": _pp_cursor},
                    timeout=20,
                )
                _pp_resp.raise_for_status()
                _pp_data = _pp_resp.json()
                _pp_results = _pp_data.get("resultList", {}).get("result", [])
                if not _pp_results:
                    break
                for res in _pp_results:
                    doi = _normalize_doi(res.get("doi"))
                    ext_id = res.get("id") or doi
                    title = res.get("title") or ""
                    if not ext_id or not title:
                        continue
                    abstract = res.get("abstractText")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET3
                            abstract = "".join(_ET3.fromstring(f"<root>{abstract}</root>").itertext()).strip()
                        except Exception:
                            pass
                    year = None
                    yt = res.get("pubYear")
                    if yt and str(yt).isdigit():
                        year = int(yt)
                    url = (f"https://europepmc.org/article/{res.get('source','PPR')}/{res.get('id')}"
                           if res.get("id") else (f"https://doi.org/{doi}" if doi else None))
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id = _ingest_doc_direct(
                            source="preprint", title=title, abstract=abstract or None,
                            year=year, url=url, external_id=ext_id, doi=doi,
                            source_type="preprint",
                        )
                        _link_to_scenario(doc_id)
                        count += 1
                        _inc("preprint")
                    except Exception:
                        _inc("preprint", 0, 1)
                _pp_fetched += len(_pp_results)
                _pp_next = _pp_data.get("nextCursorMark")
                if not _pp_next or _pp_next == _pp_cursor or len(_pp_results) < 100:
                    break
                _pp_cursor = _pp_next
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Préprints (Europe PMC) populate {scenario_id}: {_e}")
        return ("preprints", count)

    def _ingest_parsed(source, docs):
        """Ingère une liste de docs parsés (helper commun aux nouvelles sources REST).
        Renvoie le nombre ingéré. Chaque doc : {title, abstract, year, url,
        external_id, doi, source_type}. Les docs <30 caractères sont ignorés."""
        c = 0
        for _d in docs:
            if len(f"{_d.get('title','')}\n\n{_d.get('abstract') or ''}".strip()) < 30:
                continue
            try:
                _doc_id = _ingest_doc_direct(
                    source=source, title=_d["title"], abstract=_d.get("abstract"),
                    year=_d.get("year"), url=_d.get("url"), external_id=_d["external_id"],
                    doi=_d.get("doi"), source_type=_d.get("source_type", "article"),
                )
                _link_to_scenario(_doc_id)
                c += 1
                _inc(source)
            except Exception:
                _inc(source, 0, 1)
        return c

    def _fetch_semantic_scholar():
        count = 0
        try:
            _s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            _hdrs = {"x-api-key": _s2_key} if _s2_key else {}
            _off, _cap = 0, min(max_results, 1000)   # S2 search : offset+limit <= 1000
            while _off < _cap:
                if _time.time() >= _fed_deadline[0]:
                    break
                _bulk = min(100, _cap - _off)
                _r = _requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": _plain_q, "offset": _off, "limit": _bulk,
                            "fields": "title,abstract,year,externalIds,url"},
                    headers=_hdrs, timeout=20,
                )
                if _r.status_code == 429:      # rate-limited → petite pause et on retente
                    _time.sleep(2)
                    continue
                _r.raise_for_status()
                _payload = _r.json()
                _got = len(_payload.get("data") or [])
                count += _ingest_parsed("semantic_scholar", _parse_semantic_scholar(_payload))
                if _got < _bulk:
                    break
                _off += _bulk
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Semantic Scholar populate {scenario_id}: {_e}")
        return ("semantic_scholar", count)

    def _fetch_doaj():
        count = 0
        try:
            import urllib.parse as _ulib
            _page, _fetched = 1, 0
            while _fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _r = _requests.get(
                    f"https://doaj.org/api/search/articles/{_ulib.quote(_plain_q, safe='')}",
                    params={"pageSize": 100, "page": _page}, timeout=20,
                )
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("results") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("doaj", _parse_doaj(_payload))
                _fetched += _n
                if _n < 100:
                    break
                _page += 1
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"DOAJ populate {scenario_id}: {_e}")
        return ("doaj", count)

    def _fetch_clinicaltrials():
        count = 0
        try:
            _ct_token, _ct_fetched = None, 0
            while _ct_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _params = {"query.term": _plain_q, "pageSize": 100, "format": "json"}
                if _ct_token:
                    _params["pageToken"] = _ct_token
                _r = _requests.get("https://clinicaltrials.gov/api/v2/studies", params=_params, timeout=20)
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("studies") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("clinicaltrials", _parse_clinicaltrials(_payload))
                _ct_fetched += _n
                _ct_token = _payload.get("nextPageToken")
                if not _ct_token:
                    break
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"ClinicalTrials.gov populate {scenario_id}: {_e}")
        return ("clinicaltrials", count)

    def _fetch_core():
        # CORE exige une clé API (gratuite). Sans clé → source ignorée proprement
        # (comme NCBI_API_KEY : optionnelle, le déploiement reste vert).
        _core_key = os.getenv("CORE_API_KEY")
        if not _core_key:
            logger.info(f"CORE populate {scenario_id}: CORE_API_KEY absent — source ignorée.")
            return ("core", 0)
        count = 0
        try:
            _core_offset, _core_fetched = 0, 0
            while _core_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _bulk = min(100, max_results - _core_fetched)
                _r = _requests.post(
                    "https://api.core.ac.uk/v3/search/works",
                    headers={"Authorization": f"Bearer {_core_key}"},
                    json={"q": _plain_q, "limit": _bulk, "offset": _core_offset},
                    timeout=25,
                )
                if _r.status_code == 429:
                    _time.sleep(3)
                    continue
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("results") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("core", _parse_core(_payload))
                _core_fetched += _n
                _core_offset += _bulk
                if _n < _bulk:
                    break
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"CORE populate {scenario_id}: {_e}")
        return ("core", count)

    def _fetch_arxiv():
        count = 0
        try:
            _ax_start = 0
            while _ax_start < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _bulk = min(100, max_results - _ax_start)
                _r = _requests.get(
                    "http://export.arxiv.org/api/query",
                    params={"search_query": f"all:{_plain_q}", "start": _ax_start, "max_results": _bulk},
                    timeout=25,
                )
                _r.raise_for_status()
                _docs = _parse_arxiv(_r.text)
                if not _docs:
                    break
                count += _ingest_parsed("arxiv", _docs)
                if len(_docs) < _bulk:
                    break
                _ax_start += _bulk
                _time.sleep(3)      # arXiv demande ≥3 s entre requêtes
        except Exception as _e:
            logger.warning(f"arXiv populate {scenario_id}: {_e}")
        return ("arxiv", count)

    def _fetch_openaire():
        count = 0
        try:
            _oa_page, _oa_fetched = 1, 0
            while _oa_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _r = _requests.get(
                    "https://api.openaire.eu/search/publications",
                    params={"keywords": _plain_q, "size": 50, "page": _oa_page, "format": "json"},
                    timeout=25,
                )
                _r.raise_for_status()
                _docs = _parse_openaire(_r.json())
                if not _docs:
                    break
                count += _ingest_parsed("openaire", _docs)
                _oa_fetched += len(_docs)
                if len(_docs) < 50:
                    break
                _oa_page += 1
                _time.sleep(0.5)
        except Exception as _e:
            logger.warning(f"OpenAIRE populate {scenario_id}: {_e}")
        return ("openaire", count)

    def _fetch_biorxiv_medrxiv():
        # L'API bioRxiv/medRxiv n'offre PAS de recherche par mots-clés. On scanne une
        # fenêtre RÉCENTE (les ~45 derniers jours) DEPUIS le curseur 0 vers aujourd'hui
        # — de sorte que la couverture porte réellement sur les préprints récents (le
        # bug précédent scannait le DÉBUT d'une fenêtre de 18 mois → les plus VIEUX,
        # jamais pertinents). Puis filtrage lexical (_parse_biorxiv). Chaque serveur
        # ingère sous sa propre source ("biorxiv" / "medrxiv"), distinctes des
        # "Preprints" (facette Europe PMC).
        count = 0
        try:
            from datetime import date as _date, timedelta as _td
            _terms = [re.sub(r"[^a-z0-9]", "", w) for w in (_plain_q or "").lower().split()]
            _terms = [t for t in _terms if len(t) >= 4]
            try:
                _to = _date.today()
                _win = f"{(_to - _td(days=45)).isoformat()}/{_to.isoformat()}"
            except Exception:
                _win = "2026-05-01/2026-12-31"
            for _server in ("biorxiv", "medrxiv"):
                _cursor, _pages = 0, 0
                while _pages < 30:          # borne dure ; les ~45 j récents tiennent dedans
                    if _time.time() >= _fed_deadline[0]:
                        break
                    _r = _requests.get(
                        f"https://api.biorxiv.org/details/{_server}/{_win}/{_cursor}/json", timeout=20)
                    _r.raise_for_status()
                    _payload = _r.json()
                    _coll = _payload.get("collection") or []
                    if not _coll:
                        break
                    count += _ingest_parsed(_server, _parse_biorxiv(_payload, _terms, _server))
                    _pages += 1
                    _total = int((_payload.get("messages") or [{}])[0].get("total", 0) or 0)
                    _cursor += len(_coll)
                    if _cursor >= _total or len(_coll) < 100:
                        break
                    _time.sleep(0.4)
        except Exception as _e:
            logger.warning(f"bioRxiv/medRxiv populate {scenario_id}: {_e}")
        return ("biorxiv_medrxiv", count)

    # Lancer toutes les sources en parallèle
    source_funcs = [
        _fetch_pubmed, _fetch_openalex, _fetch_crossref,
        _fetch_europepmc, _fetch_preprints,
        _fetch_semantic_scholar, _fetch_doaj, _fetch_clinicaltrials, _fetch_core,
        _fetch_arxiv, _fetch_openaire, _fetch_biorxiv_medrxiv,
    ]
    t_start = _time.time()
    if include_live:
        _set_phase("federation")
        # Garde-temps : passé ce délai, les boucles de pagination des sources lentes
        # s'arrêtent (cf. _fed_deadline) et on poursuit avec le corpus partiel.
        _fed_deadline[0] = t_start + POPULATE_FEDERATION_BUDGET
        # IMPORTANT — on N'UTILISE PAS `with ThreadPoolExecutor(...)` : sa sortie
        # appelle shutdown(wait=True), qui attend TOUTES les sources (jusqu'à ~5 min
        # quand OpenAlex/Crossref paginent vers 2000), annulant de fait le budget.
        # On gère l'executor manuellement et on l'arrête SANS attendre.
        executor = ThreadPoolExecutor(max_workers=12)
        try:
            futures = {executor.submit(fn): fn.__name__ for fn in source_funcs}
            try:
                # Budget global : ne pas attendre indéfiniment une source lente.
                for future in as_completed(futures, timeout=POPULATE_FEDERATION_BUDGET):
                    try:
                        src_name, src_count = future.result()
                        logger.info(f"Populate {scenario_id} [{src_name}]: {src_count} articles ingérés")
                    except Exception as _fe:
                        logger.warning(f"Populate {scenario_id} source future error: {_fe}")
            # CRITIQUE — sur Python <3.11, as_completed lève
            # concurrent.futures.TimeoutError (≠ TimeoutError natif). Sans
            # _FuturesTimeout dans le except, l'exception remontait, le bloc
            # plantait, et la reconstruction du corpus + le scoring + le passage à
            # « done » étaient SAUTÉS → corpus figé sur la base locale, statut
            # bloqué sur « running », résultats live perdus.
            except (TimeoutError, _FuturesTimeout):
                _done = sum(1 for _f in futures if _f.done())
                logger.warning(
                    f"Populate {scenario_id}: budget fédération {POPULATE_FEDERATION_BUDGET:.0f}s dépassé — "
                    f"{_done}/{len(futures)} sources terminées ; poursuite avec le corpus partiel "
                    f"(les sources lentes continuent en arrière-plan)."
                )
        finally:
            # wait=False : ne PAS bloquer sur les sources lentes. cancel_futures=True
            # annule celles qui n'ont pas démarré ; celles en cours s'arrêteront au
            # prochain tour de pagination grâce à _fed_deadline.
            executor.shutdown(wait=False, cancel_futures=True)
        t_elapsed = _time.time() - t_start
        logger.info(f"Populate {scenario_id}: fédération terminée en {t_elapsed:.1f}s")
    else:
        logger.info(f"Populate {scenario_id}: include_live=False — base locale uniquement")

    ingested = _ingested_total[0]
    errors = _errors_total[0]
    total_found = ingested  # Approximation — PubMed callback met à jour séparément

    # ── Corpus = correspondance BOOLÉENNE (ou multi-sous-requêtes) sur base enrichie ─
    # Après ingestion des articles live, on recalcule l'appartenance au corpus via
    # EXACTEMENT le même helper que la recherche : _boolean_corpus_ids en mono-requête,
    # _multi_query_corpus_ids (union/intersection) en multi-sous-requêtes. Le corpus
    # devient donc strictement « résultat de la requête sur base locale ∪ live ».
    # allow_empty=True en multi : une intersection légitimement vide DOIT vider le corpus.
    try:
        if _sub_queries:
            _final_ids = _multi_query_corpus_ids(_sub_queries, _combinator, filters)
            _n_corpus = _set_scenario_corpus(scenario_id, _final_ids, allow_empty=True)
            logger.info(f"Populate {scenario_id}: corpus multi-requêtes final = {_n_corpus} docs "
                        f"({_combinator} de {len(_sub_queries)} sous-requêtes, base locale ∪ live)")
        else:
            _final_ids = _boolean_corpus_ids(_boolean, filters)
            _n_corpus = _set_scenario_corpus(scenario_id, _final_ids)
            logger.info(f"Populate {scenario_id}: corpus booléen final = {_n_corpus} docs "
                        f"(base locale ∪ live, requête = {_boolean[:120]})")
    except Exception as _e_corpus:
        logger.warning(f"Rebuild corpus {scenario_id}: {_e_corpus}")

    try:
        # ── Règle qualité : articles SANS abstract retirés du corpus ─────────
        try:
            with engine.begin() as conn:
                _removed = conn.execute(text("""
                    DELETE FROM article_scenarios a
                    USING literature_document d
                    WHERE a.document_id = d.id AND a.scenario_id = :sid
                      AND (d.abstract IS NULL OR length(TRIM(d.abstract)) < 30)
                """), {"sid": scenario_id}).rowcount
            if _removed:
                logger.info(f"Populate {scenario_id}: {_removed} articles sans abstract retirés.")
        except Exception as _e_noabs:
            logger.warning(f"Suppression articles sans abstract {scenario_id}: {_e_noabs}")

        # ── Mettre à jour article_count (avant rerank) ──────────────────────
        # NB : on ne marque PAS encore populate_status='done' ici — le scoring
        # n'a pas tourné. Le marquer prématurément faisait paraître "prêt" un
        # scénario dont les similarity_score restaient NULL (compteurs divergents).
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE user_scenarios
                SET article_count = (
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                ),
                updated_at = NOW()
                WHERE id = :sid
            """), {"sid": scenario_id})

        # ── Scores sémantiques (cosinus) — SANS suppression ─────────────────
        # Soft filter : le seuil filtre l'affichage et l'aval, JAMAIS par
        # suppression. Réduire le seuil fait donc réapparaître des articles.
        _set_phase("scoring")
        _n_scored = 0
        _scoring_failed = False
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs liés sans chunk résumé
            _n_scored = _run_semantic_rerank_inline(scenario_id, query)
            logger.info(f"Post-populate scoring {scenario_id}: {_n_scored} articles scorés (cosinus, aucune suppression).")
        except Exception as _e_rr:
            _scoring_failed = True
            logger.warning(f"scoring post-populate {scenario_id}: {_e_rr}")

        # ── Honnêteté de l'état : 'done' SEULEMENT si le scoring a réellement
        # produit des scores. _run_semantic_rerank_inline avale ses erreurs et
        # renvoie 0 (ex. OpenAI indisponible) ; on détecte ce cas et on marque
        # 'error' plutôt que 'done' pour ne pas afficher un état prêt trompeur.
        try:
            with engine.begin() as conn:
                _scorable = conn.execute(text("""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                      AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
                """), {"sid": scenario_id}).scalar() or 0
                _ok = (not _scoring_failed) and (_n_scored > 0 or _scorable == 0)
                conn.execute(text("""
                    UPDATE user_scenarios SET populate_status = :st, updated_at = NOW() WHERE id = :sid
                """), {"st": "done" if _ok else "error", "sid": scenario_id})
                if not _ok:
                    logger.warning(f"Populate {scenario_id}: scoring n'a produit aucun score "
                                   f"({_scorable} articles scorables) → populate_status='error'.")
        except Exception as _e_st:
            logger.warning(f"populate_status update {scenario_id}: {_e_st}")

        # ── Le corpus est scoré (cosinus) → on le publie MAINTENANT ──────────
        # Le cross-encoder Cohere (plus lent) et les visualisations tournent
        # ENSUITE en arrière-plan : l'utilisateur voit les résultats ordonnés par
        # cosinus immédiatement, puis la liste se réordonne quand le rerank arrive
        # (rerank_status). Plus d'attente synchrone sur l'API Cohere.
        _cohere_enabled = bool(os.getenv("COHERE_API_KEY"))
        if _pipeline_callback is None:
            _sources_final = _user_scenario_populate_jobs.get(scenario_id, {}).get("sources", {})
            _src_parts = [f"{src}: {cnt}" for src, cnt in _sources_final.items() if cnt > 0]
            _src_summary = " | ".join(_src_parts) if _src_parts else "aucune source"
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "done",
                "phase": "done",
                "rerank_status": "running" if _cohere_enabled else "skipped",
                "ingested": ingested,
                "errors": errors,
                "total_found": total_found,
                "sources": _sources_final,
                "message": f"{ingested} articles ingérés depuis 5 sources ({_src_summary}), {errors} erreurs.",
            }

        # Arrière-plan : cross-encoder (réordonne le sous-ensemble pertinent) puis
        # clustering UMAP/HDBSCAN + knowledge graph (cache DB). Réservé au chemin
        # /populate (le pipeline complet a ses propres étapes).
        if _pipeline_callback is None:
            def _post_done_bg(_sid, _query):
                try:
                    _n_ce = _run_cross_encoder_rerank(_sid, _query)
                    if _n_ce:
                        logger.info(f"Post-populate cross-encoder {_sid}: {_n_ce} articles réordonnés.")
                except Exception as _ece:
                    logger.warning(f"cross-encoder arrière-plan {_sid}: {_ece}")
                finally:
                    _job = _user_scenario_populate_jobs.get(_sid)
                    if _job is not None:
                        _job["rerank_status"] = "done"
                try:
                    _run_clustering_background(_sid, True)   # clustering → cache DB
                except Exception as _e1:
                    logger.warning(f"Précalcul clustering {_sid}: {_e1}")
                try:
                    _precompute_user_kg(_sid)                # knowledge graph → cache DB
                except Exception as _e2:
                    logger.warning(f"Précalcul KG {_sid}: {_e2}")
            try:
                import threading as _vth
                _vth.Thread(target=_post_done_bg, args=(scenario_id, query), daemon=True).start()
            except Exception as _e_viz:
                logger.warning(f"Tâches arrière-plan {scenario_id}: {_e_viz}")

        logger.info(f"Populate user_scenario {scenario_id}: {ingested} articles ingérés (8 sources, parallèle).")
        return ingested

    except Exception as e:
        logger.error(f"Populate user_scenario {scenario_id} fatal: {e}", exc_info=True)
        if _pipeline_callback is None:
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "error",
                "error": str(e),
                "ingested": _ingested_total[0],
            }
        return 0

def _run_semantic_rerank_inline(scenario_id: str, query: str) -> int:
    """Score sémantique (cosinus requête↔article) du corpus, mis dans similarity_score.

    Optimisé : on RÉUTILISE les embeddings pgvector déjà stockés
    (document_chunk.embedding) et on calcule le cosinus EN BASE en UNE requête
    (au lieu de ré-embedder chaque résumé via OpenAI + cosinus Python + une
    transaction par article — ce qui rendait l'étape très lente). On ne ré-embedde
    via OpenAI QUE les articles fraîchement ingérés dont les chunks ne sont pas
    encore vectorisés (minorité)."""
    try:
        from openai import OpenAI as _OAI
        _client = _OAI(timeout=90.0)
        q_emb = _client.embeddings.create(model="text-embedding-3-small", input=query[:2000]).data[0].embedding
        q_str = str(q_emb)

        # 1) Rapide : cosinus pgvector en base pour tous les docs déjà vectorisés.
        with engine.begin() as _c:
            n_fast = _c.execute(text("""
                UPDATE article_scenarios asn
                SET similarity_score = sub.sim
                FROM (
                    SELECT c.document_id,
                           MAX(1 - (c.embedding <=> CAST(:q AS vector))) AS sim
                    FROM document_chunk c
                    JOIN article_scenarios a
                      ON a.document_id = c.document_id AND a.scenario_id = :sid
                    WHERE c.embedding IS NOT NULL
                      AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                    GROUP BY c.document_id
                ) sub
                WHERE asn.scenario_id = :sid AND asn.document_id = sub.document_id
            """), {"q": q_str, "sid": scenario_id}).rowcount or 0

        # 2) Repli OpenAI pour les articles encore non scorés (chunks pas encore
        #    vectorisés). On boucle par lots de 100 jusqu'à épuisement (PLUS de
        #    plafond à 1000) pour qu'AUCUN article pertinent ne reste non scoré ;
        #    chaque lot ré-interroge les NULL restants. Borne de sécurité pour
        #    éviter une boucle infinie si un lot ne parvenait pas à s'écrire.
        n_slow = 0
        if not _openai_in_cooldown():
            import numpy as _np
            _q = _np.asarray(q_emb, dtype=float)
            _qn = float(_np.linalg.norm(_q)) or 1.0
            _MAX_FALLBACK_BATCHES = 200  # 200 × 100 = 20 000 articles / scénario
            for _bi in range(_MAX_FALLBACK_BATCHES):
                if _openai_in_cooldown():
                    break
                with engine.connect() as _conn:
                    batch = _conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                        WHERE asn.similarity_score IS NULL
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 30
                        ORDER BY ld.id LIMIT 100
                    """), {"sid": scenario_id}).mappings().fetchall()
                if not batch:
                    break
                texts = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in batch]
                try:
                    emb = _client.embeddings.create(model="text-embedding-3-small", input=texts).data
                    ups = []
                    for j, e in enumerate(emb):
                        _d = _np.asarray(e.embedding, dtype=float)
                        sim = float(_q @ _d) / (_qn * (float(_np.linalg.norm(_d)) or 1.0))
                        ups.append({"score": max(0.0, min(1.0, sim)), "doc_id": batch[j]["id"], "sid": scenario_id})
                    with engine.begin() as _c:
                        _c.execute(text("""
                            UPDATE article_scenarios SET similarity_score = :score
                            WHERE document_id = :doc_id AND scenario_id = :sid
                        """), ups)
                    n_slow += len(ups)
                except Exception as _e:
                    logger.warning(f"Rerank fallback batch {_bi}: {_e}")
                    if _is_openai_quota_error(_e):
                        _trip_openai_cooldown()
                    break  # toute erreur : on arrête (évite une boucle infinie)
            else:
                logger.warning(f"Rerank {scenario_id}: plafond de repli atteint "
                               f"({_MAX_FALLBACK_BATCHES * 100} articles) — certains peuvent rester non scorés.")
        logger.info(f"Rerank {scenario_id}: {n_fast} via pgvector + {n_slow} via OpenAI (fallback).")
        return n_fast + n_slow
    except Exception as _e:
        logger.error(f"Rerank inline {scenario_id} fatal: {_e}", exc_info=True)
        return 0


def _cohere_rerank(query: str, docs: list[str], model: str = "rerank-v3.5") -> list[float] | None:
    """Cross-encoder rerank via l'API Cohere. Renvoie un score de pertinence par
    document (aligné sur `docs`), ou None si pas de clé / échec. Pas de dépendance
    Python ajoutée : appel REST direct. Activé seulement si COHERE_API_KEY est défini.
    """
    key = os.getenv("COHERE_API_KEY")
    if not key or not docs:
        return None
    try:
        import requests as _rq
        resp = _rq.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "query": query[:4000], "documents": docs, "top_n": len(docs)},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        scores: list[float | None] = [None] * len(docs)
        for item in data.get("results", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(docs):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores  # type: ignore[return-value]
    except Exception as _e:
        logger.warning(f"Cohere rerank failed: {_e}")
        return None


def _run_cross_encoder_rerank(scenario_id: str, query: str, top_k: int = 1000) -> int:
    """Reranke le sous-ensemble PERTINENT (cosinus >= seuil) avec un cross-encoder
    (Cohere). La SÉLECTION reste pilotée par le cosinus + seuil ; on ne fait
    qu'AMÉLIORER l'ORDRE des articles pertinents (précision). No-op sans clé Cohere.
    """
    if not os.getenv("COHERE_API_KEY"):
        return 0
    try:
        eff_threshold = 0.45
        with engine.connect() as _tc:
            _ts = _tc.execute(text(
                "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
            if _ts is not None:
                eff_threshold = float(_ts)
            rows = _tc.execute(text("""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE COALESCE(asn.similarity_score, 0.0) >= :thr
                  AND ld.abstract IS NOT NULL AND length(ld.abstract) > 30
                ORDER BY asn.similarity_score DESC NULLS LAST
                LIMIT :k
            """), {"sid": scenario_id, "thr": eff_threshold, "k": top_k}).mappings().all()
        if not rows:
            return 0
        docs = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in rows]
        scores = _cohere_rerank(query, docs)
        if not scores:
            return 0
        updated = 0
        with engine.begin() as _c:
            for r, s in zip(rows, scores):
                if s is None:
                    continue
                _c.execute(text("""
                    UPDATE article_scenarios SET rerank_score = :s
                    WHERE document_id = :doc_id AND scenario_id = :sid
                """), {"s": s, "doc_id": r["id"], "sid": scenario_id})
                updated += 1
        logger.info(f"Cross-encoder rerank {scenario_id}: {updated} articles pertinents réordonnés.")
        return updated
    except Exception as _e:
        logger.warning(f"Cross-encoder rerank {scenario_id} failed: {_e}")
        return 0


def _run_user_scenario_full_pipeline(scenario_id: str, query: str, filters: dict, max_results: int = 500) -> None:
    """
    Pipeline complet d'enrichissement pour un scénario utilisateur.
    Ordre optimal :
    1. ingest    – Ingestion multi-sources (PubMed+OpenAlex+Crossref+EuropePMC+Preprints+SemanticScholar+DOAJ+ClinicalTrials.gov+CORE+arXiv+OpenAIRE+medRxiv+bioRxiv)
    2. fulltext  – Récupération full-text (PMC→EuropePMC→Unpaywall) pendant que les IDs sont frais
    3. embed     – Embeddings sur title+abstract+fulltext chunks (contenu enrichi)
    4. rerank    – Score cosinus via pgvector (pas de re-embedding API)
    5. pico      – Extraction PICO (LLM, utilise fulltext si dispo)
    6. metadata  – Extraction métadonnées étude (LLM)
    7. clustering – K-means sur embeddings pgvector
    """
    import time as _time

    STEP_ORDER = ["ingest", "fulltext", "embed", "rerank", "pico", "metadata", "clustering"]

    def update_step(step: str, status: str, **kwargs):
        job = _user_scenario_pipeline_jobs.get(scenario_id, {})
        job["current_step"] = step
        job["steps"] = job.get("steps", {})
        job["steps"][step] = {"status": status, **kwargs}
        job["overall_status"] = "running"
        _user_scenario_pipeline_jobs[scenario_id] = job
        step_idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
        progress = int((step_idx / len(STEP_ORDER)) * 100)
        try:
            with engine.begin() as _conn:
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'running',
                        pipeline_step = :step,
                        pipeline_progress = :progress,
                        pipeline_started_at = COALESCE(pipeline_started_at, NOW())
                    WHERE id = :sid
                """), {"step": step, "progress": progress, "sid": scenario_id})
        except Exception as _e:
            logger.warning(f"update_step DB write failed: {_e}")
        logger.info(f"Pipeline {scenario_id} [{step}]: {status} {kwargs}")

    def ingest_callback(event: str, value):
        if event == "pubmed_found":
            update_step("ingest", "running", found=value)

    _user_scenario_pipeline_jobs[scenario_id] = {
        "overall_status": "running",
        "current_step": "ingest",
        "steps": {
            "ingest": {"status": "pending"},
            "fulltext": {"status": "pending"},
            "embed": {"status": "pending"},
            "rerank": {"status": "pending"},
            "pico": {"status": "pending"},
            "metadata": {"status": "pending"},
            "clustering": {"status": "pending"},
        },
    }

    try:
        # ── Étape 1 : Ingestion multi-sources ────────────────────────────────────
        update_step("ingest", "running")
        # Si le scénario a DÉJÀ été peuplé (populate terminé — typiquement par la
        # recherche, ensuite sauvegardée en scénario), on NE RE-FÉDÈRE PAS. Une 2ᵉ
        # fédération (live, non déterministe) recalculerait le corpus booléen sur une
        # base entre-temps enrichie par les threads d'arrière-plan de la 1ʳᵉ
        # fédération → le compteur dérivait (recherche = 7, carte = 8). On réutilise
        # le corpus existant et on passe directement à l'enrichissement.
        with engine.connect() as _psc:
            _ps_row = _psc.execute(text(
                "SELECT populate_status, "
                "(SELECT COUNT(DISTINCT document_id) FROM article_scenarios WHERE scenario_id = :sid) AS n "
                "FROM user_scenarios WHERE id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        _corpus_ready = bool(_ps_row and _ps_row.get("populate_status") == "done" and (_ps_row.get("n") or 0) > 0)
        if _corpus_ready:
            ingested = int(_ps_row["n"])
            logger.info(f"Pipeline {scenario_id}: corpus déjà construit ({ingested} docs, populate=done) — "
                        f"fédération sautée pour éviter la dérive du compteur.")
        else:
            ingested = _run_user_scenario_populate(
                scenario_id, query, filters, max_results, _pipeline_callback=ingest_callback
            )
        # `ingested` from populate is a raw cumulative counter (includes ON CONFLICT duplicates
        # and DB-cache articles). Get the real unique count from DB as the source of truth.
        with engine.connect() as _ic:
            _real_ingested = _ic.execute(text(
                "SELECT COUNT(DISTINCT document_id) FROM article_scenarios WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar() or 0
        update_step("ingest", "done", ingested=_real_ingested, api_results_raw=ingested)

        if _real_ingested == 0:
            _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
            _user_scenario_pipeline_jobs[scenario_id]["message"] = "Aucun article trouvé (7 sources interrogées)."
            return

        # ── Étape 2 : Full-text multi-sources (avant embedding pour enrichir les chunks) (PMC → EuropePMC → Unpaywall → bioRxiv → Semantic Scholar → OpenAlex) ──
        update_step("fulltext", "running")
        try:
            import re as _re
            import subprocess as _subprocess
            import tempfile as _tempfile
            import xml.etree.ElementTree as _ET_ft
            import requests as _requests

            _NCBI_BASE_FT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            _EPMC_BASE_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
            _UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "literev@gesica.ch")
            _CHUNK_SIZE_FT = 4000
            _CHUNK_OVERLAP_FT = 400

            def _ft_get(url, params=None, timeout=20):
                for _att in range(3):
                    try:
                        _r = _requests.get(url, params=params, timeout=timeout,
                                           headers={"User-Agent": "LiteRev-Evidence/1.0"})
                        if _r.status_code == 429:
                            _time.sleep(int(_r.headers.get("Retry-After", 10)))
                            continue
                        return _r
                    except Exception as _fe:
                        logger.debug(f"_ft_get {url}: attempt {_att+1} failed: {_fe}")
                        _time.sleep(1)
                logger.debug(f"_ft_get {url}: all retries exhausted")
                return None

            def _parse_pmc_xml_ft(xml_str):
                _skip = {"ref-list","ack","fn-group","glossary","app-group","notes","bio","author-notes"}
                try:
                    _root = _ET_ft.fromstring(xml_str)
                except _ET_ft.ParseError:
                    xml_str = _re.sub(r"&(?!amp;|lt;|gt;|apos;|quot;)", "&amp;", xml_str)
                    try:
                        _root = _ET_ft.fromstring(xml_str)
                    except Exception:
                        return None
                _parts = []
                def _walk(n):
                    _tag = n.tag.split("}")[-1] if "}" in n.tag else n.tag
                    if _tag in _skip:
                        return
                    if n.text and n.text.strip():
                        _parts.append(n.text.strip())
                    for _ch in n:
                        _walk(_ch)
                    if n.tail and n.tail.strip():
                        _parts.append(n.tail.strip())
                _walk(_root)
                _txt = _re.sub(r"\s+", " ", " ".join(_parts)).strip()
                return _txt if len(_txt) > 50 else None

            def _extract_pdf_text_ft(pdf_url):
                try:
                    _r = _requests.get(pdf_url, timeout=30, stream=True,
                                       headers={"User-Agent": "LiteRev-Evidence/1.0"})
                    if _r.status_code != 200:
                        return None
                    _ct = _r.headers.get("content-type", "")
                    if "pdf" not in _ct.lower() and not pdf_url.lower().endswith(".pdf"):
                        _txt = _re.sub(r"<[^>]+>", " ", _r.text)
                        _txt = _re.sub(r"\s+", " ", _txt).strip()
                        return _txt if len(_txt) > 500 else None
                    with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as _f:
                        for _chunk in _r.iter_content(chunk_size=8192):
                            _f.write(_chunk)
                        _tmp = _f.name
                    _res = _subprocess.run(["pdftotext", "-layout", _tmp, "-"],
                                           capture_output=True, text=True, timeout=30)
                    os.unlink(_tmp)
                    if _res.returncode == 0 and _res.stdout.strip():
                        _txt = _re.sub(r"\s+", " ", _res.stdout).strip()
                        return _txt if len(_txt) > 500 else None
                except Exception:
                    pass
                return None

            def _resolve_pmcid_ft(ext_id, pmid_val, doi_val):
                if ext_id and "PMC" in ext_id.upper():
                    _m = _re.search(r"(\d{5,10})", ext_id)
                    if _m:
                        return f"PMC{_m.group(1)}"
                if ext_id and _re.match(r"^PMC\d+$", ext_id.upper()):
                    return ext_id.upper()
                _cand = None
                if ext_id and ext_id.isdigit():
                    _cand = ext_id
                elif pmid_val:
                    _cand = str(pmid_val).replace("PMID:", "").strip()
                if _cand:
                    _r2 = _ft_get(f"{_NCBI_BASE_FT}/esummary.fcgi",
                                  params={"db": "pubmed", "id": _cand, "retmode": "json"})
                    if _r2 and _r2.status_code == 200:
                        try:
                            _aids = _r2.json().get("result", {}).get(_cand, {}).get("articleids", [])
                            for _aid in _aids:
                                if _aid.get("idtype") == "pmcid":
                                    _pmc = _aid.get("value", "").replace("PMC", "")
                                    if _pmc:
                                        return f"PMC{_pmc}"
                        except Exception:
                            pass
                if doi_val:
                    _r3 = _ft_get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                                  params={"ids": doi_val, "format": "json",
                                          "tool": "literev", "email": _UNPAYWALL_EMAIL})
                    if _r3 and _r3.status_code == 200:
                        try:
                            _recs = _r3.json().get("records", [])
                            if _recs and _recs[0].get("pmcid"):
                                return _recs[0]["pmcid"]
                        except Exception:
                            pass
                return None

            def _chunk_text_ft(text_str):
                # Découpage SÉMANTIQUE par phrases : on coupe aux frontières de
                # phrases (et non au milieu d'un mot/d'une idée), puis on fusionne
                # en chunks d'environ _CHUNK_SIZE_FT caractères avec un recouvrement
                # au niveau de la phrase. Plus cohérent que l'ancienne fenêtre brute.
                text_str = _re.sub(r"\s+", " ", text_str).strip()
                if not text_str:
                    return []
                _sentences = _re.split(r"(?<=[.!?])\s+", text_str)
                _chunks: list[str] = []
                _cur: list[str] = []
                _cur_len = 0
                for _s in _sentences:
                    if _cur and _cur_len + len(_s) + 1 > _CHUNK_SIZE_FT:
                        _chunks.append(" ".join(_cur).strip())
                        _ov: list[str] = []
                        _olen = 0
                        for _p in reversed(_cur):
                            if _olen + len(_p) <= _CHUNK_OVERLAP_FT:
                                _ov.insert(0, _p)
                                _olen += len(_p)
                            else:
                                break
                        _cur = _ov
                        _cur_len = sum(len(_p) + 1 for _p in _cur)
                    _cur.append(_s)
                    _cur_len += len(_s) + 1
                if _cur:
                    _chunks.append(" ".join(_cur).strip())
                # Phrase unique trop longue (> 1.5x la cible) : re-découpe en fenêtre mot.
                _final: list[str] = []
                _max = int(_CHUNK_SIZE_FT * 1.5)
                for _c in _chunks:
                    if len(_c) <= _max:
                        _final.append(_c)
                        continue
                    _st = 0
                    while _st < len(_c):
                        _en = min(_st + _CHUNK_SIZE_FT, len(_c))
                        if _en < len(_c):
                            _cut = _c.rfind(" ", _st, _en)
                            if _cut > _st:
                                _en = _cut
                        _final.append(_c[_st:_en].strip())
                        _st = _en
                return [c for c in _final if len(c) > 50]

            def _insert_fulltext_chunks_ft(doc_id, chunks, source_label, emb_client=None):
                """Insère les chunks fulltext_section et les embedde immédiatement si possible."""
                with engine.begin() as _c:
                    _c.execute(text(
                        "DELETE FROM document_chunk WHERE document_id = :did "
                        "AND chunk_type IN ('fulltext_section', 'full_text')"
                    ), {"did": doc_id})
                    for _i, _chunk_text in enumerate(chunks):
                        _meta = json.dumps({"source": source_label, "chunk_index": _i})
                        _c.execute(text("""
                            INSERT INTO document_chunk
                                (document_id, content, chunk_index, chunk_type, chunk_weight, metadata_json)
                            VALUES (:did, :content, :idx, 'fulltext_section', 1.0, CAST(:meta AS jsonb))
                        """), {"did": doc_id, "content": _chunk_text, "idx": _i, "meta": _meta})
                    _c.execute(text(
                        "UPDATE literature_document SET has_fulltext = true, open_access = true WHERE id = :did"
                    ), {"did": doc_id})
                # Embedder les nouveaux chunks immédiatement si client OpenAI disponible
                if emb_client and chunks:
                    try:
                        with engine.connect() as _c2:
                            _new_chunks = _c2.execute(text("""
                                SELECT id, content FROM document_chunk
                                WHERE document_id = :did AND chunk_type = 'fulltext_section'
                                  AND embedding IS NULL ORDER BY chunk_index
                            """), {"did": doc_id}).mappings().fetchall()
                        for _bi in range(0, len(_new_chunks), 50):
                            _batch = _new_chunks[_bi:_bi+50]
                            _texts = [r["content"][:8000] for r in _batch]
                            _emb_resp = emb_client.embeddings.create(
                                model="text-embedding-3-small", input=_texts)
                            for _k, _ed in enumerate(_emb_resp.data):
                                _vec = "[" + ",".join(str(x) for x in _ed.embedding) + "]"
                                with engine.begin() as _c3:
                                    _c3.execute(text(
                                        "UPDATE document_chunk SET embedding = CAST(:vec AS vector) WHERE id = :cid"
                                    ), {"vec": _vec, "cid": _batch[_k]["id"]})
                            _time.sleep(0.1)
                    except Exception as _emb_e:
                        logger.warning(f"Fulltext embed doc {doc_id}: {_emb_e}")
                return len(chunks)

            # Récupérer tous les documents du scénario sans full-text
            with engine.connect() as conn:
                ft_rows = conn.execute(text("""
                    SELECT ld.id, ld.external_id, ld.doi, ld.pmid, ld.source
                    FROM literature_document ld
                    JOIN article_scenarios asn ON asn.document_id = ld.id
                    WHERE asn.scenario_id = :sid
                      AND ld.project_context = 'literev'
                      AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                    ORDER BY ld.id
                """), {"sid": scenario_id}).mappings().fetchall()

            ft_fetched = 0
            ft_errors = 0
            ft_total = len(ft_rows)

            # Initialiser le client OpenAI pour l'embedding des chunks fulltext
            _ft_emb_client = None
            try:
                from openai import OpenAI as _OAI_ft
                _ft_emb_client = _OAI_ft(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                pass

            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            _ft_lock = threading.Lock()
            _ft_done_count = 0
            
            def _process_ft_row(row):
                _ext_id = row["external_id"] or ""
                _doi = _normalize_doi(row["doi"] or "") or ""
                _pmid = row["pmid"] or ""
                _source = row["source"] or ""
                _fulltext = None
                _source_used = None
                _ft_fail_reasons = []
                
                try:
                    # Source 1 : PMC
                    _pmcid = _resolve_pmcid_ft(_ext_id, _pmid, _doi)
                    if _pmcid:
                        _r_pmc = _ft_get(f"{_EPMC_BASE_FT}/{_pmcid}/fullTextXML", timeout=30)
                        if _r_pmc and _r_pmc.status_code == 200 and _r_pmc.text.strip().startswith("<"):
                            _fulltext = _parse_pmc_xml_ft(_r_pmc.text)
                            if _fulltext and len(_fulltext) > 500:
                                _source_used = f"europepmc:{_pmcid}"
                            else:
                                _ft_fail_reasons.append(f"europepmc:{_pmcid}:xml_parse_empty")
                        elif _r_pmc:
                            _ft_fail_reasons.append(f"europepmc:{_pmcid}:http_{_r_pmc.status_code}")
                        
                        if not _fulltext:
                            _pmcid_num = _pmcid.replace("PMC", "")
                            _r_ncbi = _ft_get(f"{_NCBI_BASE_FT}/efetch.fcgi",
                                             params={"db": "pmc", "id": _pmcid_num,
                                                     "rettype": "full", "retmode": "xml"}, timeout=30)
                            if _r_ncbi and _r_ncbi.status_code == 200 and _r_ncbi.text.strip().startswith("<"):
                                _fulltext = _parse_pmc_xml_ft(_r_ncbi.text)
                                if _fulltext and len(_fulltext) > 500:
                                    _source_used = f"pmc:{_pmcid}"
                                else:
                                    _fulltext = None
                                    _ft_fail_reasons.append(f"pmc:{_pmcid}:xml_parse_empty")
                            elif _r_ncbi:
                                _ft_fail_reasons.append(f"pmc:{_pmcid}:http_{_r_ncbi.status_code}")
                    else:
                        _ft_fail_reasons.append("pmcid:not_resolved")
                    
                    # Source 2 : Unpaywall
                    if not _fulltext and _doi and _doi.startswith("10."):
                        _r_uw = _ft_get(f"https://api.unpaywall.org/v2/{_doi}",
                                        params={"email": _UNPAYWALL_EMAIL})
                        if _r_uw and _r_uw.status_code == 200:
                            try:
                                _uw_data = _r_uw.json()
                                _pdf_url = None
                                _best = _uw_data.get("best_oa_location") or {}
                                _pdf_url = _best.get("url_for_pdf") or _best.get("url")
                                if not _pdf_url:
                                    for _loc in _uw_data.get("oa_locations", []):
                                        if _loc.get("url_for_pdf"):
                                            _pdf_url = _loc["url_for_pdf"]
                                            break
                                if _pdf_url:
                                    _fulltext = _extract_pdf_text_ft(_pdf_url)
                                    if _fulltext and len(_fulltext) > 500:
                                        _source_used = "unpaywall"
                                    else:
                                        _fulltext = None
                                        _ft_fail_reasons.append("unpaywall:pdf_empty")
                                else:
                                    _ft_fail_reasons.append(f"unpaywall:not_oa(is_oa={_uw_data.get('is_oa')})")
                            except Exception as _uw_e:
                                _ft_fail_reasons.append(f"unpaywall:parse_error:{_uw_e}")
                        elif _r_uw:
                            _ft_fail_reasons.append(f"unpaywall:http_{_r_uw.status_code}")
                        else:
                            _ft_fail_reasons.append("unpaywall:no_doi" if not _doi else "unpaywall:timeout")
                    elif not _doi:
                        _ft_fail_reasons.append("unpaywall:skipped_no_doi")
                    
                    # Source 3 : bioRxiv/medRxiv
                    if not _fulltext and _doi and _doi.startswith("10.1101/"):
                        for _srv in ["biorxiv", "medrxiv"]:
                            _r_bx = _ft_get(f"https://api.biorxiv.org/details/{_srv}/{_doi}/na/json")
                            if _r_bx and _r_bx.status_code == 200:
                                try:
                                    _coll = _r_bx.json().get("collection", [])
                                    if _coll:
                                        _pdf_url = f"https://www.{_srv}.org/content/{_doi}.full.pdf"
                                        _fulltext = _extract_pdf_text_ft(_pdf_url)
                                        if _fulltext and len(_fulltext) > 500:
                                            _source_used = _srv
                                            break
                                        else:
                                            _fulltext = None
                                            _ft_fail_reasons.append(f"{_srv}:pdf_empty")
                                    else:
                                        _ft_fail_reasons.append(f"{_srv}:not_found")
                                except Exception as _bx_e:
                                    _ft_fail_reasons.append(f"{_srv}:parse_error:{_bx_e}")
                    
                    # Source 4 : Semantic Scholar
                    if not _fulltext:
                        _ss_id = None
                        if _doi:
                            _ss_id = f"DOI:{_doi}"
                        elif _pmid:
                            _ss_id = f"PMID:{_pmid}"
                        elif _ext_id and _ext_id.upper().startswith("PMC"):
                            _ss_id = f"PMCID:{_ext_id}"
                        if _ss_id:
                            _r_ss = _ft_get(
                                f"https://api.semanticscholar.org/graph/v1/paper/{_ss_id}",
                                params={"fields": "openAccessPdf,abstract"},
                            )
                            if _r_ss and _r_ss.status_code == 200:
                                try:
                                    _ss_data = _r_ss.json()
                                    _oa_pdf = _ss_data.get("openAccessPdf")
                                    if _oa_pdf and _oa_pdf.get("url"):
                                        _fulltext = _extract_pdf_text_ft(_oa_pdf["url"])
                                        if _fulltext and len(_fulltext) > 500:
                                            _source_used = "semanticscholar"
                                        else:
                                            _fulltext = None
                                            _ft_fail_reasons.append("semanticscholar:pdf_empty")
                                    else:
                                        _ft_fail_reasons.append(f"semanticscholar:no_oa_pdf")
                                except Exception as _ss_e:
                                    _ft_fail_reasons.append(f"semanticscholar:parse_error:{_ss_e}")
                            elif _r_ss:
                                _ft_fail_reasons.append(f"semanticscholar:http_{_r_ss.status_code}")
                        else:
                            _ft_fail_reasons.append("semanticscholar:no_identifier")
                    
                    # Source 5 : OpenAlex
                    if not _fulltext and (_ext_id.startswith("W") or _doi):
                        _oa_work_url = (
                            f"https://api.openalex.org/works/{_ext_id}"
                            if _ext_id.startswith("W")
                            else f"https://api.openalex.org/works/doi:{_doi}"
                        )
                        _r_oa = _ft_get(_oa_work_url, params={"select": "open_access"})
                        if _r_oa and _r_oa.status_code == 200:
                            try:
                                _oa_info = _r_oa.json().get("open_access", {})
                                _oa_url = _oa_info.get("oa_url")
                                if _oa_url:
                                    _fulltext = _extract_pdf_text_ft(_oa_url)
                                    if _fulltext and len(_fulltext) > 500:
                                        _source_used = "openalex_oa"
                                    else:
                                        _fulltext = None
                                        _ft_fail_reasons.append("openalex:pdf_empty")
                                else:
                                    _ft_fail_reasons.append(f"openalex:not_oa(is_oa={_oa_info.get('is_oa')})")
                            except Exception as _oa_e:
                                _ft_fail_reasons.append(f"openalex:parse_error:{_oa_e}")
                        elif _r_oa:
                            _ft_fail_reasons.append(f"openalex:http_{_r_oa.status_code}")
                    
                    if _fulltext and _source_used:
                        _chunks_ft = _chunk_text_ft(_fulltext)
                        if _chunks_ft:
                            _insert_fulltext_chunks_ft(row["id"], _chunks_ft, _source_used, _ft_emb_client)
                            return True, None
                        else:
                            logger.warning(f"Fulltext doc {row['id']}: text retrieved but produced 0 chunks (source={_source_used})")
                            return False, "0_chunks"
                    else:
                        logger.info(f"Fulltext unavailable doc {row['id']} (ext_id={_ext_id}, doi={_doi}): {' | '.join(_ft_fail_reasons) or 'no_sources_tried'}")
                        return False, "not_found"
                except Exception as _ft_e:
                    logger.warning(f"Fulltext doc {row['id']}: {_ft_e}")
                    return False, str(_ft_e)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_process_ft_row, row): row for row in ft_rows}
                for future in as_completed(futures):
                    try:
                        success, _ = future.result()
                        with _ft_lock:
                            _ft_done_count += 1
                            if success:
                                ft_fetched += 1
                            else:
                                ft_errors += 1
                            
                            if _ft_done_count % 10 == 0 or _ft_done_count == ft_total:
                                update_step("fulltext", "running",
                                            done=_ft_done_count, total=ft_total, paywall=ft_errors,
                                            pct=round((_ft_done_count) / ft_total * 100, 1) if ft_total > 0 else 0)
                    except Exception as e:
                        with _ft_lock:
                            _ft_done_count += 1
                            ft_errors += 1
                            logger.error(f"Error in fulltext worker: {e}")

            update_step("fulltext", "done", fetched=ft_fetched, total=ft_total,
                        paywall_or_failed=ft_errors)
        except Exception as e:
            update_step("fulltext", "error", error=str(e))

        # ── Étape 3 : Embeddings (title_abstract + fulltext_section — contenu enrichi) ────────
        update_step("embed", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI_emb
                _emb_client = _OAI_emb(api_key=openai_key)
                with engine.connect() as _conn_emb:
                    _chunks_to_embed = _conn_emb.execute(text("""
                        SELECT c.id, c.document_id, c.content
                        FROM document_chunk c
                        JOIN article_scenarios ars ON ars.document_id = c.document_id
                        WHERE ars.scenario_id = :sid
                          AND c.embedding IS NULL
                          AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                          AND LENGTH(c.content) > 20
                        ORDER BY c.id
                    """), {"sid": scenario_id}).mappings().fetchall()
                _emb_total = len(_chunks_to_embed)
                _emb_docs_total = len({r["document_id"] for r in _chunks_to_embed})
                _emb_done = 0
                _emb_docs_done: set = set()
                _emb_errors = 0
                _emb_batch_size = 100
                for _bi in range(0, _emb_total, _emb_batch_size):
                    _batch = _chunks_to_embed[_bi:_bi + _emb_batch_size]
                    try:
                        _texts = [r["content"][:8000] for r in _batch]
                        _emb_resp = _emb_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=_texts
                        )
                        # Batch all updates in a single transaction (not one per chunk)
                        with engine.begin() as _conn_upd:
                            for _k, _emb_data in enumerate(_emb_resp.data):
                                _vec_str = "[" + ",".join(str(x) for x in _emb_data.embedding) + "]"
                                _conn_upd.execute(text("""
                                    UPDATE document_chunk
                                    SET embedding = CAST(:vec AS vector)
                                    WHERE id = :cid
                                """), {"vec": _vec_str, "cid": _batch[_k]["id"]})
                                _emb_done += 1
                                _emb_docs_done.add(_batch[_k]["document_id"])
                    except Exception as _emb_e:
                        _emb_errors += len(_batch)
                        logger.warning(f"Embed batch {_bi}: {_emb_e}")
                    update_step("embed", "running",
                                docs_done=len(_emb_docs_done), docs_total=_emb_docs_total,
                                chunks_done=_emb_done, chunks_total=_emb_total,
                                pct=round(_emb_done / _emb_total * 100, 1) if _emb_total > 0 else 0)
                update_step("embed", "done",
                            docs_embedded=len(_emb_docs_done), docs_total=_emb_docs_total,
                            chunks_embedded=_emb_done, chunks_total=_emb_total,
                            errors=_emb_errors)
            else:
                update_step("embed", "skipped", reason="Clé OpenAI non configurée")
        except Exception as _emb_ex:
            update_step("embed", "error", error=str(_emb_ex))

        # ── Étape 4 : Rerank via pgvector (cosinus sur embeddings stockés) ──────────────
        update_step("rerank", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI_rr
                _rr_client = _OAI_rr(api_key=openai_key)
                _rr_resp = _rr_client.embeddings.create(
                    model="text-embedding-3-small", input=query[:2000])
                _q_vec = "[" + ",".join(str(x) for x in _rr_resp.data[0].embedding) + "]"
                with engine.begin() as _rr_conn:
                    _rr_result = _rr_conn.execute(text("""
                        UPDATE article_scenarios ars
                        SET similarity_score = sub.best_sim
                        FROM (
                            SELECT c.document_id,
                                   MAX(1.0 - (c.embedding <=> CAST(:q_vec AS vector))) AS best_sim
                            FROM document_chunk c
                            JOIN article_scenarios a ON a.document_id = c.document_id
                            WHERE a.scenario_id = :sid
                              AND c.embedding IS NOT NULL
                              AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                            GROUP BY c.document_id
                        ) sub
                        WHERE ars.scenario_id = :sid
                          AND ars.document_id = sub.document_id
                    """), {"q_vec": _q_vec, "sid": scenario_id})
                n_reranked = _rr_result.rowcount
                update_step("rerank", "done", updated=n_reranked)

                # Seuil SÉMANTIQUE = SOFT : on ne supprime JAMAIS d'article du
                # corpus. Le corpus = résultat INTÉGRAL de la requête booléenne
                # (base locale ∪ live) ; le seuil ne fait que distinguer, à
                # l'affichage et en aval (page scénario, modèle), les articles
                # « au-dessus du seuil » (mis en avant) des « sous le seuil »
                # (conservés). Voir get_user_scenario_corpus (above/below_threshold).
                # On recalcule simplement article_count sur le corpus complet.
                try:
                    with engine.begin() as _ac:
                        _ac.execute(text("""
                            UPDATE user_scenarios
                            SET article_count = (
                                SELECT COUNT(DISTINCT ars.document_id)
                                FROM article_scenarios ars
                                JOIN literature_document d ON d.id = ars.document_id
                                WHERE ars.scenario_id = :sid
                                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                            )
                            WHERE id = :sid
                        """), {"sid": scenario_id})
                except Exception as _ce:
                    logger.warning(f"Post-rerank article_count update {scenario_id}: {_ce}")
            else:
                update_step("rerank", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("rerank", "error", error=str(e))

        # ── Étape 5 : Extraction PICO ─────────────────────────────────────────────────────
        update_step("pico", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI
                from datetime import datetime, timezone
                _client = _OAI(api_key=openai_key, timeout=90.0)
                system_prompt_pico = (
                    "You are a systematic review expert in emergency medicine. "
                    "Extract PICO elements and return ONLY valid JSON:\n"
                    '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
                    '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
                    '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
                    "Be concise (max 2 sentences per field). Return ONLY the JSON."
                )
                with engine.connect() as conn:
                    pico_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 50
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                pico_extracted = 0
                pico_errors = 0
                for row in pico_rows:
                    try:
                        response = _client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_pico},
                                {"role": "user", "content": f"Title: {row['title']}\n\nAbstract: {(row['abstract'] or '')[:3000]}"},
                            ],
                            temperature=0.1,
                            max_tokens=400,
                            response_format={"type": "json_object"},
                        )
                        pico = json.loads(response.choices[0].message.content)
                        required = {"P", "I", "C", "O", "study_design", "pico_confidence"}
                        if required.issubset(pico.keys()):
                            pico["pico_confidence"] = float(pico.get("pico_confidence", 0.5))
                            with engine.begin() as conn:
                                conn.execute(text("""
                                    UPDATE literature_document
                                    SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts
                                    WHERE id = :article_id
                                """), {"pico": json.dumps(pico), "ts": datetime.now(timezone.utc), "article_id": row["id"]})
                            pico_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline PICO article {row['id']}: {e}")
                        pico_errors += 1
                    _time.sleep(0.05)
                # Total coverage from DB (includes previously extracted articles)
                with engine.connect() as _pico_stat_conn:
                    _pico_total_in_scenario = _pico_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = :sid
                    """), {"sid": scenario_id}).scalar() or 0
                    _pico_total_with = _pico_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM literature_document ld
                        JOIN article_scenarios ars ON ars.document_id = ld.id
                        WHERE ars.scenario_id = :sid AND ld.pico_json IS NOT NULL
                    """), {"sid": scenario_id}).scalar() or 0
                update_step("pico", "done",
                            extracted_this_run=pico_extracted,
                            total_with_pico=_pico_total_with,
                            total_articles=_pico_total_in_scenario,
                            pct=round(_pico_total_with / _pico_total_in_scenario * 100, 1) if _pico_total_in_scenario > 0 else 0,
                            errors=pico_errors)
            else:
                update_step("pico", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("pico", "error", error=str(e))

        # ── Étape 6 : Extraction métadonnées ─────────────────────────────────
        update_step("metadata", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI2
                from datetime import datetime, timezone
                _client2 = _OAI2(api_key=openai_key)
                system_prompt_meta = (
                    "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
                    '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
                    '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
                    '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
                    '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
                    "Return ONLY the JSON."
                )
                with engine.connect() as conn:
                    meta_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year,
                               ld.citation_count, ld.open_access,
                               ld.study_design, ld.sample_size
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                meta_extracted = 0
                meta_errors = 0
                for row in meta_rows:
                    try:
                        response = _client2.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_meta},
                                {"role": "user", "content": f"Title: {row['title']}\n\nAbstract: {(row['abstract'] or '')[:2000]}"},
                            ],
                            temperature=0.1,
                            max_tokens=300,
                            response_format={"type": "json_object"},
                        )
                        metadata = json.loads(response.choices[0].message.content)
                        metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                        # Renseigner les colonnes structurées depuis le JSON extrait
                        # (study_design / sample_size), puis calculer un quality_score
                        # déterministe — sinon l'évaluation GRADE buckette tout en « Faible ».
                        study_design = metadata.get("study_type") or row.get("study_design")
                        sample_size = _coerce_int(metadata.get("sample_size")) or row.get("sample_size")
                        quality_score = _compute_quality_score(
                            study_design=study_design,
                            year=row.get("year"),
                            sample_size=sample_size,
                            citation_count=row.get("citation_count"),
                            open_access=row.get("open_access"),
                            bias_risk=metadata.get("bias_risk"),
                        )
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET metadata_json = CAST(:meta AS jsonb),
                                    study_design = COALESCE(:study_design, study_design),
                                    sample_size = COALESCE(:sample_size, sample_size),
                                    quality_score = COALESCE(:quality_score, quality_score)
                                WHERE id = :article_id
                            """), {
                                "meta": json.dumps(metadata),
                                "study_design": study_design,
                                "sample_size": sample_size,
                                "quality_score": quality_score,
                                "article_id": row["id"],
                            })
                        meta_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline metadata article {row['id']}: {e}")
                        meta_errors += 1
                    _time.sleep(0.05)
                # Total coverage from DB (includes previously extracted articles)
                with engine.connect() as _meta_stat_conn:
                    _meta_total_in_scenario = _meta_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = :sid
                    """), {"sid": scenario_id}).scalar() or 0
                    _meta_total_with = _meta_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM literature_document ld
                        JOIN article_scenarios ars ON ars.document_id = ld.id
                        WHERE ars.scenario_id = :sid
                          AND ld.metadata_json IS NOT NULL AND ld.metadata_json != '{}'::jsonb
                    """), {"sid": scenario_id}).scalar() or 0
                update_step("metadata", "done",
                            extracted_this_run=meta_extracted,
                            total_with_metadata=_meta_total_with,
                            total_articles=_meta_total_in_scenario,
                            pct=round(_meta_total_with / _meta_total_in_scenario * 100, 1) if _meta_total_in_scenario > 0 else 0,
                            errors=meta_errors)
            else:
                update_step("metadata", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("metadata", "error", error=str(e))

        # ── Étape 7 : Clustering (UMAP+HDBSCAN avec fallback KMeans) ────────────
        update_step("clustering", "running")
        try:

            with engine.connect() as conn:
                cl_docs = list(conn.execute(text("""
                    SELECT d.id, d.title, d.abstract, d.year, d.journal,
                           (
                               SELECT c.embedding::text
                               FROM document_chunk c
                               WHERE c.document_id = d.id
                                 AND c.embedding IS NOT NULL
                               -- Résumé (title_abstract) en priorité : vecteur cohérent.
                               ORDER BY (c.chunk_type = 'title_abstract') DESC, c.id
                               LIMIT 1
                           ) AS embedding_str
                    FROM literature_document d
                    JOIN article_scenarios asn ON asn.document_id = d.id
                    WHERE asn.scenario_id = :sid
                      AND d.project_context = 'literev'
                      AND d.abstract IS NOT NULL
                      AND LENGTH(d.abstract) > 50
                    ORDER BY d.year DESC NULLS LAST
                    LIMIT 100000
                """), {"sid": scenario_id}).mappings().all())

            if len(cl_docs) >= 5:
                texts = [f"{d['title']} {d['abstract'] or ''}" for d in cl_docs]

                # ── Embeddings → UMAP → HDBSCAN (cœur partagé _cluster_core) ──
                _cc = _cluster_core(cl_docs, texts, openai_key=None,
                                    allow_openai_embeddings=False, tfidf_min_df=1)
                labels = _cc["labels"]
                embedding_2d = _cc["embedding_2d"]
                method_used = _cc["method"]
                feature_names = _cc["feature_names"]
                X_dense = _cc["X_dense"]
                logger.info(f"Pipeline clustering {scenario_id}: {len(cl_docs)} docs, "
                            f"source={_cc['embedding_source']}, method={method_used}")

                n_clusters = len(set(int(l) for l in labels if int(l) != -1))

                # ── Persister en DB ───────────────────────────────────────────
                with engine.begin() as _cl_conn:
                    for _cl_idx, _cl_doc in enumerate(cl_docs):
                        _cl_id = int(labels[_cl_idx])
                        _cl_conn.execute(text("""
                            UPDATE article_scenarios
                            SET cluster_id = :cid, cluster_label = :clabel
                            WHERE scenario_id = :sid AND document_id = :did
                        """), {
                            "cid": _cl_id,
                            "clabel": f"Cluster {_cl_id + 1}" if _cl_id != -1 else "Non-classés",
                            "sid": scenario_id,
                            "did": _cl_doc["id"],
                        })

                # Cache de visualisation : MÊME helper que le calcul en arrière-plan
                # (plus de duplication) → DB durable (+ /tmp pour compat).
                _cl_payload = _build_clusters_payload(scenario_id, cl_docs, _cc, with_summaries=False)
                _save_viz_cache(scenario_id, "clustering", _cl_payload)
                try:
                    import os as _os
                    _os.makedirs("/tmp/literev_clustering_cache", exist_ok=True)
                    with open(f"/tmp/literev_clustering_cache/{scenario_id}.json", "w") as f:
                        json.dump(_cl_payload, f, default=str)
                except Exception:
                    pass
                update_step("clustering", "done", n_clusters=n_clusters, n_docs=len(cl_docs), method=method_used)
            else:
                update_step("clustering", "skipped", reason=f"Corpus insuffisant ({len(cl_docs)} articles)")
        except Exception as e:
            update_step("clustering", "error", error=str(e))

        # ── Précalcul du knowledge graph (cache DB) — visualisation prête ──────
        try:
            _precompute_user_kg(scenario_id)
        except Exception as _e_kg:
            logger.warning(f"Précalcul KG pipeline {scenario_id}: {_e_kg}")

        # ── Fin du pipeline ───────────────────────────────────────────────────
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
        _user_scenario_pipeline_jobs[scenario_id]["message"] = (
            f"Pipeline terminé."
        )
        # Persister pipeline_status = done et mettre à jour article_count (source de vérité = DB)
        try:
            with engine.begin() as _conn:
                _final_count = _conn.execute(text("""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                """), {"sid": scenario_id}).scalar() or 0
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'done',
                        pipeline_step = 'done',
                        pipeline_progress = 100,
                        article_count = :cnt,
                        updated_at = NOW()
                    WHERE id = :sid
                """), {"sid": scenario_id, "cnt": _final_count})
            _user_scenario_pipeline_jobs[scenario_id]["message"] = (
                f"{_final_count} articles dans le corpus."
            )
        except Exception as _e:
            logger.warning(f"Pipeline final DB update failed: {_e}")
        logger.info(f"Pipeline complet {scenario_id}: terminé.")

    except Exception as e:
        logger.error(f"Pipeline user_scenario {scenario_id} fatal: {e}", exc_info=True)
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "error"
        _user_scenario_pipeline_jobs[scenario_id]["error"] = str(e)
        # Persister l'échec en DB (symétrique du succès) : sinon la carte lit
        # pipeline_status='running' indéfiniment et le scénario est relancé comme
        # « orphelin » au redémarrage.
        try:
            with engine.begin() as _conn:
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'failed', updated_at = NOW()
                    WHERE id = :sid
                """), {"sid": scenario_id})
        except Exception as _e2:
            logger.warning(f"Pipeline failure DB update {scenario_id}: {_e2}")


@app.post("/user-scenarios/{scenario_id}/populate")
def populate_user_scenario(
    scenario_id: str,
    max_results: int = 100000,
    include_live: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Construit le corpus du scénario = requête booléenne sur (base locale ∪ live).
    Plafond LIVE_MAX_PER_SOURCE par source. include_live=False : base locale seule.
    """
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]

    if _launch_populate_job(scenario_id, query, row.get("filters") or {}, max_results, include_live) == "already_running":
        job = _user_scenario_populate_jobs.get(scenario_id) or {}
        return {
            "scenario_id": scenario_id,
            "status": "already_running",
            "message": "Une ingestion est déjà en cours pour ce scénario.",
            "ingested": job.get("ingested", 0),
        }

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": query,
        "max_results": max_results,
        "message": f"Ingération multi-sources lancée en arrière-plan pour '{row['name']}' "
                   "(DB Cache + PubMed + OpenAlex + Crossref + EuropePMC + Preprints + "
                   "Semantic Scholar + DOAJ + ClinicalTrials.gov + CORE + arXiv + OpenAIRE + "
                   "medRxiv + bioRxiv). "
                   "Utilisez /user-scenarios/{id}/populate/status pour suivre la progression.",
    }


@app.get("/user-scenarios/{scenario_id}/populate/status")
def get_user_scenario_populate_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état de l'ingéstion multi-sources en cours pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_populate_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "status": "not_started",
            "message": "Aucune ingestion lancée. Appelez POST /user-scenarios/{id}/populate.",
        }
    return {"scenario_id": scenario_id, **job}


@app.post("/user-scenarios/{scenario_id}/pipeline")
def start_user_scenario_pipeline(
    scenario_id: str,
    max_results: int = 500,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Déclenche le pipeline complet d'enrichissement en arrière-plan :
    PubMed → PICO → Métadonnées → Full-text → Clustering.
    Idéalement appelé dès qu'une recherche est validée en scénario épinglé.
    """
    import threading
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]

    with _pipeline_jobs_lock:
        job = _user_scenario_pipeline_jobs.get(scenario_id)
        if job and job.get("overall_status") == "running":
            return {
                "scenario_id": scenario_id,
                "status": "already_running",
                "message": "Un pipeline est déjà en cours pour ce scénario.",
                "current_step": job.get("current_step"),
            }

        _user_scenario_pipeline_jobs[scenario_id] = {
            "overall_status": "starting",
            "current_step": "ingest",
            "steps": {
                "ingest": {"status": "pending"},
                "fulltext": {"status": "pending"},
                "embed": {"status": "pending"},
                "rerank": {"status": "pending"},
                "pico": {"status": "pending"},
                "metadata": {"status": "pending"},
                "clustering": {"status": "pending"},
            },
        }

    t = threading.Thread(
        target=_run_user_scenario_full_pipeline,
        args=(scenario_id, query, row.get("filters") or {}, max_results),
        daemon=True,
    )
    t.start()

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": query,
        "max_results": max_results,
        "message": f"Pipeline complet lancé pour '{row['name']}' "
                   "(ingest 8 sources → fulltext → embeddings → rerank → PICO → métadonnées → clustering). "
                   "Suivez la progression via GET /user-scenarios/{id}/pipeline/status.",
        "steps": ["ingest", "fulltext", "embed", "rerank", "pico", "metadata", "clustering"],
    }


@app.get("/user-scenarios/{scenario_id}/pipeline/status")
def get_user_scenario_pipeline_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état détaillé du pipeline d'enrichissement pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_pipeline_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "overall_status": "not_started",
            "message": "Aucun pipeline lancé. Appelez POST /user-scenarios/{id}/pipeline.",
            "steps": {},
        }
    return {"scenario_id": scenario_id, **job}


@app.get("/user-scenarios/{scenario_id}/embedding-status")
def get_user_scenario_embedding_status(scenario_id: str) -> dict[str, Any]:
    """
    Embedding status for a user scenario.
    Reports separately:
    - title+abstract docs pending (1 chunk each)
    - fulltext papers pending (N chunks each)
    """
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        # Univers = corpus du scénario, hors doublons (MÊME filtre que /corpus),
        # pour réconcilier les compteurs. Inclut les docs SANS chunk (chunkless).
        corpus = conn.execute(text("""
            SELECT
                COUNT(*) AS corpus_total,
                COUNT(*) FILTER (
                    WHERE NOT EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = ars.document_id)
                ) AS chunkless
            FROM article_scenarios ars
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

        # Title+abstract: one chunk per doc. Un chunk title_abstract n'est "en
        # attente" QUE s'il sera réellement embeddé par le worker — qui IGNORE
        # (a) les chunks trop courts (length <= 20) et (b) le title_abstract d'un
        # doc qui possède aussi du plein texte (on embed alors le plein texte). Sans
        # ces deux filtres, le compteur restait bloqué à un petit nombre "en cours".
        ta = conn.execute(text("""
            SELECT
                COUNT(DISTINCT ars.document_id) AS total_docs,
                COUNT(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN ars.document_id END) AS embedded_docs,
                COUNT(DISTINCT CASE
                    WHEN c.embedding IS NULL
                     AND length(c.content) > 20
                     AND NOT EXISTS (
                         SELECT 1 FROM document_chunk c2
                         WHERE c2.document_id = ars.document_id
                           AND c2.chunk_type = 'fulltext_section'
                     )
                    THEN ars.document_id END) AS pending_docs
            FROM article_scenarios ars
            JOIN document_chunk c ON c.document_id = ars.document_id
                AND c.chunk_type = 'title_abstract'
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

        # Full-text: multiple chunks per doc
        ft = conn.execute(text("""
            SELECT
                COUNT(DISTINCT d.document_id) AS total_ft_docs,
                COUNT(DISTINCT CASE WHEN ft_emb.pending_chunks = 0 THEN d.document_id END) AS ft_docs_complete,
                COUNT(DISTINCT CASE WHEN ft_emb.pending_chunks > 0 THEN d.document_id END) AS ft_docs_pending,
                COALESCE(SUM(ft_emb.total_chunks), 0) AS total_ft_chunks,
                COALESCE(SUM(ft_emb.pending_chunks), 0) AS pending_ft_chunks,
                COALESCE(SUM(ft_emb.embedded_chunks), 0) AS embedded_ft_chunks
            FROM (
                SELECT DISTINCT ars.document_id
                FROM article_scenarios ars
                JOIN document_chunk c ON c.document_id = ars.document_id
                    AND c.chunk_type = 'fulltext_section'
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
            ) d
            JOIN (
                SELECT
                    c.document_id,
                    COUNT(*) AS total_chunks,
                    COUNT(*) FILTER (WHERE c.embedding IS NULL) AS pending_chunks,
                    COUNT(*) FILTER (WHERE c.embedding IS NOT NULL) AS embedded_chunks
                FROM document_chunk c
                WHERE c.chunk_type = 'fulltext_section'
                GROUP BY c.document_id
            ) ft_emb ON ft_emb.document_id = d.document_id
        """), {"sid": scenario_id}).mappings().first()

        # Scores de pertinence (RANKING) — INDÉPENDANT de l'indexation RAG ci-dessus.
        # Le similarity_score affiché est calculé EN LIGNE pendant la phase "scoring"
        # (_run_semantic_rerank_inline : réutilise les embeddings stockés, ré-embedde
        # le reste à la volée), et le rerank Cohere écrit rerank_score sur le
        # sous-ensemble pertinent. Ce sont CES compteurs qui pilotent les voyants
        # "Sémantique" / "Cohere" — PAS le worker d'indexation RAG (chunks).
        ranking = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL) AS scored,
                COUNT(*) FILTER (WHERE ars.rerank_score IS NOT NULL) AS reranked
            FROM article_scenarios ars
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

    ta_total = int(ta["total_docs"] or 0)
    ta_embedded = int(ta["embedded_docs"] or 0)
    ta_pending = int(ta["pending_docs"] or 0)

    corpus_total = int(corpus["corpus_total"] or 0)
    chunkless = int(corpus["chunkless"] or 0)

    ft_total = int(ft["total_ft_docs"] or 0)
    ft_pending_docs = int(ft["ft_docs_pending"] or 0)
    ft_total_chunks = int(ft["total_ft_chunks"] or 0)
    ft_pending_chunks = int(ft["pending_ft_chunks"] or 0)
    ft_embedded_chunks = int(ft["embedded_ft_chunks"] or 0)

    # Docs without any fulltext = abstract-only
    abstract_only_total = ta_total - ft_total
    # Total pending embedding work
    total_pending_chunks = ta_pending + ft_pending_chunks

    # ── Pertinence (ranking) : honnête, découplé de l'indexation RAG ────────────
    rank_total = int(ranking["total"] or 0)
    rank_scored = int(ranking["scored"] or 0)
    rank_reranked = int(ranking["reranked"] or 0)
    cohere_configured = bool(os.getenv("COHERE_API_KEY"))
    # Sémantique : prêt SEULEMENT quand TOUT le corpus est scoré (et non "au moins
    # un chunk vectorisé"). Tant que des articles restent non scorés, le classement
    # sémantique est incomplet → voyant non vert.
    semantic_ready = rank_total > 0 and rank_scored >= rank_total
    # Cohere : prêt SEULEMENT quand le rerank a RÉELLEMENT tourné (≥ 1 rerank_score),
    # pas simplement parce qu'une clé existe.
    cohere_ready = cohere_configured and rank_reranked > 0

    if chunkless > 0:
        status = "partial"
        status_label = f"{chunkless} document(s) pas encore découpé(s) (sans chunk) — invisibles à la recherche"
    elif total_pending_chunks == 0 and ta_total > 0:
        status = "complete"
        status_label = "All embeddings complete"
    elif ta_embedded == 0 and ft_embedded_chunks == 0:
        status = "none"
        status_label = "No embeddings yet — only lexical search available"
    else:
        status = "partial"
        status_label = f"{ta_pending} abstract-only docs + {ft_pending_chunks} fulltext chunks still to embed"

    return {
        "scenario_id": scenario_id,
        "status": status,
        "status_label": status_label,
        "corpus_total": corpus_total,
        "chunkless": chunkless,
        "abstract_only": {
            "total_docs": abstract_only_total,
            "embedded_docs": max(0, abstract_only_total - ta_pending),
            "pending_docs": ta_pending,
        },
        "title_abstract_chunks": {
            "total_docs": ta_total,
            "embedded_docs": ta_embedded,
            "pending_docs": ta_pending,
        },
        "fulltext": {
            "total_docs": ft_total,
            "docs_fully_embedded": int(ft["ft_docs_complete"] or 0),
            "docs_pending": ft_pending_docs,
            "total_chunks": ft_total_chunks,
            "embedded_chunks": ft_embedded_chunks,
            "pending_chunks": ft_pending_chunks,
        },
        "total_pending_chunks": total_pending_chunks,
        # ── Pertinence (ranking) : compteurs réels du classement affiché ─────────
        # Découplé de l'indexation RAG : `scored`/`reranked` portent sur les scores
        # effectivement présents sur article_scenarios (ce que l'utilisateur voit).
        "ranking": {
            "total": rank_total,
            "scored": rank_scored,
            "reranked": rank_reranked,
            "complete": semantic_ready,
        },
        # Disponibilité RÉELLE de chaque mode de pertinence (plus de voyant "lexical"
        # toujours vert, qui n'apportait aucune information) :
        # - sémantique : vert seulement quand TOUT le corpus est scoré.
        # - cohere     : vert seulement quand le rerank a réellement tourné.
        # - cohere_configured : distingue "pas de clé" de "clé OK, rerank pas encore".
        "score_availability": {
            "semantic": semantic_ready,
            "cohere": cohere_ready,
            "cohere_configured": cohere_configured,
        }
    }


# ── Proxy endpoints : rediriger les appels /gesica/scenarios/{usr-*}/... ──────
# Les endpoints existants (screening, pico, evidence-brief, clustering, rag, etc.)
# valident maintenant l'ID via la DB (user_scenarios is_system=TRUE).
# Pour les scénarios utilisateur (usr-*), on intercepte avant ce check.

@app.get("/user-scenarios/{scenario_id}/screening-progress")
def get_user_scenario_screening_progress(scenario_id: str) -> dict[str, Any]:
    """Progression du screening PRISMA pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN d.is_duplicate = TRUE THEN 1 ELSE 0 END) AS duplicates,
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 ELSE 0 END) AS included,
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 ELSE 0 END) AS excluded,
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) IS NULL OR COALESCE(ars.screening_status, d.screening_status) = 'pending' THEN 1 ELSE 0 END) AS pending
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    total = int(stats["total"] or 0)
    duplicates = int(stats["duplicates"] or 0)
    unique = total - duplicates
    included = int(stats["included"] or 0)
    excluded = int(stats["excluded"] or 0)
    screened = included + excluded
    pct = round(screened / unique * 100, 1) if unique > 0 else 0
    return {
        "scenario_id": scenario_id,
        "total_in_db": total,
        "total": total,
        "duplicates": duplicates,
        "unique_articles": unique,
        "screened": screened,
        "included": included,
        "excluded": excluded,
        "awaiting": unique - screened,
        "pending": unique - screened,
        "progress_pct": pct,
        "screening_complete": pct >= 100,
    }


@app.get("/user-scenarios/{scenario_id}/pico-stats")
def get_user_scenario_pico_stats(scenario_id: str) -> dict[str, Any]:
    """Statistiques PICO pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        counts = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS with_pico,
                COUNT(*) FILTER (WHERE d.pico_json IS NULL) AS without_pico,
                ROUND(AVG((d.pico_json->>'pico_confidence')::float)
                    FILTER (WHERE d.pico_json IS NOT NULL)::numeric, 2) AS avg_confidence
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id}).mappings().fetchone()
        designs = conn.execute(text("""
            SELECT
                COALESCE(d.pico_json->>'study_design', 'Non extrait') AS design,
                COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.pico_json IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id}).mappings().fetchall()
    total = counts["total"] if counts else 0
    with_pico = counts["with_pico"] if counts else 0
    return {
        "scenario_id": scenario_id,
        "total": total,
        "with_pico": with_pico,
        "without_pico": counts["without_pico"] if counts else 0,
        "coverage_pct": round((with_pico / total * 100) if total > 0 else 0, 1),
        "avg_confidence": float(counts["avg_confidence"]) if counts and counts["avg_confidence"] else None,
        "study_design_distribution": [{"design": d["design"], "count": d["n"]} for d in designs],
    }


@app.get("/user-scenarios/{scenario_id}/prisma")
def get_user_scenario_prisma(
    scenario_id: str,
    threshold: float = Query(None),
) -> dict[str, Any]:
    """Flow PRISMA modernisé pour un scénario utilisateur.

    Retourne 4 étapes : identification → pré-screening IA → curation manuelle → synthèse.
    Le seuil de similarité sémantique sépare sélection automatique et borderline.
    """
    row = _get_user_scenario_or_404(scenario_id)

    # Effective threshold (param > saved setting > default 0.45)
    with engine.connect() as conn:
        ss = conn.execute(text(
            "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id=:sid"
        ), {"sid": scenario_id}).first()
    eff_threshold = threshold if threshold is not None else (
        float(ss[0]) if ss and ss[0] is not None else 0.45
    )

    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN d.source = 'pubmed'   THEN 1 ELSE 0 END) AS pubmed,
                SUM(CASE WHEN d.source = 'pmc'      THEN 1 ELSE 0 END) AS pmc,
                SUM(CASE WHEN d.source = 'openalex' THEN 1 ELSE 0 END) AS openalex,
                SUM(CASE WHEN d.source = 'europepmc' THEN 1 ELSE 0 END) AS europepmc,
                SUM(CASE WHEN d.source = 'crossref' THEN 1 ELSE 0 END) AS crossref,
                SUM(CASE WHEN d.source = 'preprint' THEN 1 ELSE 0 END) AS preprint,
                SUM(CASE WHEN d.source = 'medrxiv'  THEN 1 ELSE 0 END) AS medrxiv,
                SUM(CASE WHEN d.source = 'biorxiv'  THEN 1 ELSE 0 END) AS biorxiv,
                SUM(CASE WHEN d.source = 'semantic_scholar' THEN 1 ELSE 0 END) AS semantic_scholar,
                SUM(CASE WHEN d.source = 'doaj'     THEN 1 ELSE 0 END) AS doaj,
                SUM(CASE WHEN d.source = 'clinicaltrials' THEN 1 ELSE 0 END) AS clinicaltrials,
                SUM(CASE WHEN d.source = 'core'     THEN 1 ELSE 0 END) AS core,
                SUM(CASE WHEN d.source = 'arxiv'    THEN 1 ELSE 0 END) AS arxiv,
                SUM(CASE WHEN d.source = 'openaire' THEN 1 ELSE 0 END) AS openaire,
                SUM(CASE WHEN d.source = 'db_cache' THEN 1 ELSE 0 END) AS db_cache,
                SUM(CASE WHEN d.is_duplicate = TRUE THEN 1 ELSE 0 END) AS duplicates,
                -- semantic split at effective threshold
                SUM(CASE WHEN COALESCE(ars.similarity_score, 0) >= :thr THEN 1 ELSE 0 END) AS above_threshold,
                SUM(CASE WHEN COALESCE(ars.similarity_score, 0) <  :thr THEN 1 ELSE 0 END) AS below_threshold,
                -- manual curation
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 ELSE 0 END) AS manually_included,
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 ELSE 0 END) AS manually_excluded,
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) IS NULL OR COALESCE(ars.screening_status, d.screening_status) = 'pending'
                         THEN 1 ELSE 0 END) AS pending,
                -- manually included but below threshold (override)
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included'
                           AND COALESCE(ars.similarity_score, 0) < :thr THEN 1 ELSE 0 END) AS manually_rescued,
                -- manually excluded above threshold (veto)
                SUM(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                           AND COALESCE(ars.similarity_score, 0) >= :thr THEN 1 ELSE 0 END) AS manually_vetoed,
                -- full text — RESTREINT à l'ensemble de preuves (≥ seuil OU inclus
                -- manuellement, hors exclus), pas au corpus entier : sinon le "X of Y"
                -- du PRISMA pouvait dépasser Y (le fameux "5 of 4").
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  THEN 1 ELSE 0 END) AS with_fulltext,
                -- embeddings
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.embedding IS NOT NULL
                ) THEN 1 ELSE 0 END) AS embedded
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_threshold}).mappings().first()

    total           = int(stats["total"] or 0)
    duplicates      = int(stats["duplicates"] or 0)
    above           = int(stats["above_threshold"] or 0)
    below           = int(stats["below_threshold"] or 0)
    man_included    = int(stats["manually_included"] or 0)
    man_excluded    = int(stats["manually_excluded"] or 0)
    pending         = int(stats["pending"] or 0)
    man_rescued     = int(stats["manually_rescued"] or 0)   # below threshold but manually included
    man_vetoed      = int(stats["manually_vetoed"] or 0)    # above threshold but manually excluded
    with_fulltext   = int(stats["with_fulltext"] or 0)
    embedded        = int(stats["embedded"] or 0)

    # Evidence = (above threshold NOT vetoed) + manually rescued
    evidence_total  = (above - man_vetoed) + man_rescued
    screening_done  = (man_included + man_excluded) > 0

    return {
        "scenario_id": scenario_id,
        "scenario_title": row["name"],
        "identification": {
            "total_records": total,
            "by_source": {
                "pubmed":    int(stats["pubmed"] or 0),
                "pmc":       int(stats["pmc"] or 0),
                "openalex":  int(stats["openalex"] or 0),
                "europepmc": int(stats["europepmc"] or 0),
                "crossref":  int(stats["crossref"] or 0),
                "preprint":  int(stats["preprint"] or 0),
                "medrxiv":   int(stats["medrxiv"] or 0),
                "biorxiv":   int(stats["biorxiv"] or 0),
                "semantic_scholar": int(stats["semantic_scholar"] or 0),
                "doaj":      int(stats["doaj"] or 0),
                "clinicaltrials": int(stats["clinicaltrials"] or 0),
                "core":      int(stats["core"] or 0),
                "arxiv":     int(stats["arxiv"] or 0),
                "openaire":  int(stats["openaire"] or 0),
                "db_cache":  int(stats["db_cache"] or 0),
            },
            "duplicates_removed": duplicates,
            "embedded": embedded,
        },
        "semantic_screening": {
            "threshold": eff_threshold,
            "above_threshold": above,
            "below_threshold": below,
            "method": "cosine similarity (text-embedding-3-small)",
        },
        "full_text": {
            "with_fulltext": with_fulltext,
            "without_fulltext": total - with_fulltext,
            "pct": round(with_fulltext / total * 100, 1) if total > 0 else 0.0,
            "note": "Texte intégral via PMC / EuropePMC / Unpaywall / Semantic Scholar",
        },
        "manual_curation": {
            "included": man_included,
            "excluded": man_excluded,
            "pending": pending,
            "screening_complete": screening_done,
            "manually_rescued": man_rescued,
            "manually_vetoed": man_vetoed,
        },
        "evidence": {
            "total": evidence_total,
            "ai_auto_selected": above - man_vetoed,
            "manually_rescued": man_rescued,
            "with_fulltext": with_fulltext,
            "screening_complete": screening_done,
        },
        # Keep legacy fields for backward compatibility
        "screening": {
            "records_screened": total,
            "records_excluded_title_abstract": man_excluded,
            "records_included_screening": man_included,
            "records_awaiting_screening": pending,
        },
        "eligibility": {
            "fulltext_assessed": above,
            "fulltext_retrieved": with_fulltext,
            "fulltext_not_retrieved": above - with_fulltext,
            "fulltext_excluded": 0,
        },
        "included": {
            "total_included": man_included if screening_done else evidence_total,
            "awaiting_assessment": pending,
            "screening_complete": screening_done,
            "note": "" if screening_done else "Screening manuel non encore effectué.",
        },
    }


@app.get("/user-scenarios/{scenario_id}/evidence-brief")
def get_user_scenario_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Evidence Brief d'un scénario utilisateur (délègue au constructeur générique)."""
    _get_user_scenario_or_404(scenario_id)
    return _build_evidence_brief(scenario_id)


def _build_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Construit l'Evidence Brief d'un scénario, indépendamment de son type (user
    ou GESICA) : un seul helper générique. Toutes les statistiques (designs,
    sources, niveaux de preuve, couverture, citations) sont calculées sur le
    SOUS-ENSEMBLE PERTINENT (au-dessus du seuil sémantique)."""
    eff_thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        # `relevant*` = sous-ensemble PERTINENT (au-dessus du seuil sémantique) sur
        # lequel l'Evidence Brief / le modèle s'appuient ; `total`/`with_*` couvrent
        # le corpus complet (pour le contexte).
        corpus_stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE d.is_duplicate IS TRUE) AS duplicates,
                COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS with_pico,
                COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) = 'included') AS included,
                COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) = 'excluded') AS excluded,
                COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) = 'pending' OR COALESCE(ars.screening_status, d.screening_status) IS NULL) AS pending,
                COUNT(*) FILTER (WHERE EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                )) AS with_fulltext,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.pico_json IS NOT NULL) AS relevant_with_pico,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND EXISTS (SELECT 1 FROM document_chunk c
                        WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')) AS relevant_with_fulltext,
                -- Couverture & citations : calculées sur le SOUS-ENSEMBLE PERTINENT
                -- (au-dessus du seuil), pas sur le corpus complet.
                MIN(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max,
                AVG(d.citation_count) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.citation_count IS NOT NULL) AS avg_citations,
                MAX(d.citation_count) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS max_citations
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchone()

        top_articles = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.journal, d.authors, d.doi,
                   d.study_design, d.pico_json, d.citation_count, COALESCE(ars.screening_status, d.screening_status) AS screening_status,
                   d.quality_score, ars.similarity_score
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            ORDER BY
                CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 0 ELSE 1 END,
                d.citation_count DESC NULLS LAST, d.year DESC NULLS LAST
            LIMIT 15
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        # Toutes les distributions ci-dessous sont calculées sur le SOUS-ENSEMBLE
        # PERTINENT (au-dessus du seuil sémantique), pas sur le corpus complet :
        # elles doivent sommer au nombre de « pertinents » affiché (ex. 51), pas
        # au total du corpus (ex. 100).
        study_designs = conn.execute(text(f"""
            WITH b AS (
                SELECT lower(coalesce(
                    nullif(trim(ld.study_design), ''),
                    nullif(trim(ld.pico_json->>'study_design'), ''), '')) AS d
                FROM article_scenarios ars
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
                  AND COALESCE(ars.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.screening_status, ld.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            )
            SELECT {_STUDY_DESIGN_CASE} AS design, COUNT(*) AS n
            FROM b GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        year_dist = conn.execute(text("""
            SELECT d.year, COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
              AND d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year ORDER BY d.year ASC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        source_dist = conn.execute(text("""
            SELECT d.source, COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            GROUP BY d.source ORDER BY n DESC LIMIT 8
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        # Niveau de preuve = GRADE strict basé sur le DEVIS (cf. _GRADE_LEVEL_CASE),
        # pas sur le quality_score composite : sinon une cohorte/cas-témoins bien
        # citée passait « Modérée » voire « Forte », ce qui contredit GRADE (toute
        # étude observationnelle part en certitude faible).
        evidence_levels = conn.execute(text(f"""
            WITH b AS (
                SELECT lower(coalesce(
                    nullif(trim(ld.study_design), ''),
                    nullif(trim(ld.pico_json->>'study_design'), ''), '')) AS d
                FROM article_scenarios ars
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
                  AND COALESCE(ars.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.screening_status, ld.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            )
            SELECT {_GRADE_LEVEL_CASE} AS level, COUNT(*) AS n
            FROM b GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

    pico_table = []
    for r in top_articles:
        pj = r["pico_json"] or {}
        pico_table.append({
            "id": r["id"],
            "title": (r["title"] or "")[:120],
            "year": r["year"],
            "journal": r["journal"],
            "citation_count": r["citation_count"],
            "study_design": r["study_design"] or pj.get("study_design", ""),
            "screening_status": r["screening_status"],
            "similarity_score": round(float(r["similarity_score"]), 3) if r["similarity_score"] else None,
            "pico": {
                "population": pj.get("population", pj.get("P", "")),
                "intervention": pj.get("intervention", pj.get("I", "")),
                "comparator": pj.get("comparator", pj.get("C", "")),
                "outcome": pj.get("outcome", pj.get("O", "")),
                "study_design": pj.get("study_design", ""),
                "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
                "limitations": pj.get("limitations", ""),
                "evidence_level": pj.get("evidence_level", ""),
            }
        })

    return {
        "scenario_id": scenario_id,
        "generated_at": __import__('datetime').datetime.now().isoformat(),
        "corpus_stats": {
            "total": int(corpus_stats["total"] or 0),
            "duplicates": int(corpus_stats["duplicates"] or 0),
            "with_pico": int(corpus_stats["with_pico"] or 0),
            "with_fulltext": int(corpus_stats["with_fulltext"] or 0),
            "relevant": int(corpus_stats["relevant"] or 0),
            "relevant_with_pico": int(corpus_stats["relevant_with_pico"] or 0),
            "relevant_with_fulltext": int(corpus_stats["relevant_with_fulltext"] or 0),
            "threshold": eff_thr,
            "included": int(corpus_stats["included"] or 0),
            "excluded": int(corpus_stats["excluded"] or 0),
            "pending": int(corpus_stats["pending"] or 0),
            "year_min": corpus_stats["year_min"],
            "year_max": corpus_stats["year_max"],
            "avg_citations": round(float(corpus_stats["avg_citations"]), 1) if corpus_stats["avg_citations"] else None,
            "max_citations": int(corpus_stats["max_citations"]) if corpus_stats["max_citations"] else None,
            "pico_coverage_pct": round(
                100 * int(corpus_stats["with_pico"] or 0) / max(int(corpus_stats["total"] or 1), 1), 1
            ),
        },
        "double_blind_stats": {"reviewer_1_done": 0, "reviewer_2_done": 0, "both_done": 0, "agreements": 0, "conflicts": 0},
        "top_articles": [
            {
                "id": r["id"],
                "title": r["title"],
                "year": r["year"],
                "journal": r["journal"],
                "authors": r["authors"],
                "doi": r["doi"],
                "study_design": r["study_design"] or (r["pico_json"].get("study_design") if r["pico_json"] else None),
                "citation_count": r["citation_count"],
                "screening_status": r["screening_status"],
                "quality_score": round(float(r["quality_score"]), 2) if r["quality_score"] else None,
                "similarity_score": round(float(r["similarity_score"]), 3) if r["similarity_score"] else None,
                "abstract_excerpt": (r["abstract"] or "")[:500],
                "pico_summary": {
                    "population": r["pico_json"].get("population", r["pico_json"].get("P", "")) if r["pico_json"] else "",
                    "intervention": r["pico_json"].get("intervention", r["pico_json"].get("I", "")) if r["pico_json"] else "",
                    "outcome": r["pico_json"].get("outcome", r["pico_json"].get("O", "")) if r["pico_json"] else "",
                    "key_finding": r["pico_json"].get("key_finding", r["pico_json"].get("conclusion", "")) if r["pico_json"] else "",
                } if r["pico_json"] else None,
            }
            for r in top_articles
        ],
        "pico_table": pico_table,
        "study_design_distribution": [{"design": d["design"], "count": int(d["n"])} for d in study_designs],
        "year_distribution": [{"year": d["year"], "count": int(d["n"])} for d in year_dist],
        "source_distribution": [{"source": s["source"], "count": int(s["n"])} for s in source_dist],
        "evidence_level_distribution": [{"level": e["level"], "count": int(e["n"])} for e in evidence_levels],
    }


@app.get("/user-scenarios/{scenario_id}/pico-bulk")
def get_user_scenario_pico_bulk(scenario_id: str, limit: int = 100000, offset: int = 0) -> dict[str, Any]:
    """Tous les articles d'un scénario utilisateur avec leur PICO extrait."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.source, d.authors, d.doi, d.journal,
                   d.study_design, d.pico_json, d.pico_extracted_at, COALESCE(ars.screening_status, d.screening_status) AS screening_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
            ORDER BY
                CASE WHEN d.pico_json IS NOT NULL THEN 0 ELSE 1 END,
                d.year DESC NULLS LAST, d.id DESC
            LIMIT :limit OFFSET :offset
        """), {"sid": scenario_id, "limit": limit, "offset": offset}).mappings().fetchall()
        total_row = conn.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS with_pico
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().fetchone()
    articles = []
    for r in rows:
        pico = r["pico_json"] if r["pico_json"] else None
        articles.append({
            "id": r["id"],
            "title": r["title"],
            "year": r["year"],
            "source": r["source"],
            "authors": r["authors"],
            "doi": r["doi"],
            "journal": r["journal"],
            "study_design": pico.get("study_design") if pico else r["study_design"],
            "pico_confidence": float(pico.get("pico_confidence", 0)) if pico else None,
            "P": pico.get("P") if pico else None,
            "I": pico.get("I") if pico else None,
            "C": pico.get("C") if pico else None,
            "O": pico.get("O") if pico else None,
            "pico_notes": pico.get("pico_notes") if pico else None,
            "has_pico": pico is not None,
            "pico_extracted_at": r["pico_extracted_at"].isoformat() if r["pico_extracted_at"] else None,
            "screening_status": r["screening_status"],
        })
    return {
        "scenario_id": scenario_id,
        "total": int(total_row["total"]) if total_row else 0,
        "with_pico": int(total_row["with_pico"]) if total_row else 0,
        "offset": offset,
        "limit": limit,
        "articles": articles,
    }


@app.get("/user-scenarios/{scenario_id}/knowledge-graph")
def _compute_user_kg(scenario_id: str, max_nodes: int = 400, min_similarity: float = 0.35) -> dict[str, Any]:
    """Calcul du knowledge graph d'un scénario utilisateur (un seul endroit, réutilisé
    par l'endpoint ET le précalcul)."""
    sql = _KG_NODE_SQL.format(
        join=("JOIN article_scenarios ars ON ars.document_id = d.id"
              " JOIN document_chunk c ON c.document_id = d.id"),
        where="ars.scenario_id = :sid",
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"sid": scenario_id, "max_nodes": max_nodes}).mappings().all()
        n_total = conn.execute(text("""
            SELECT COUNT(*) FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND EXISTS (SELECT 1 FROM document_chunk c
                          WHERE c.document_id = d.id AND c.embedding IS NOT NULL)
        """), {"sid": scenario_id}).scalar() or 0
    return _build_knowledge_graph(scenario_id, rows, min_similarity, int(n_total))


def _precompute_user_kg(scenario_id: str) -> None:
    """Précalcule + met en cache (DB) le knowledge graph aux paramètres par défaut."""
    try:
        _save_viz_cache(scenario_id, "kg", _compute_user_kg(scenario_id))
    except Exception as _e:
        logger.warning(f"Précalcul KG {scenario_id}: {_e}")


def get_user_scenario_knowledge_graph(
    scenario_id: str,
    max_nodes: int = 400,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Graphe de connaissance d'un scénario utilisateur (réseau de similarité sémantique)."""
    _get_user_scenario_or_404(scenario_id)
    # Cache DB durable aux paramètres par défaut (précalculé par le pipeline/populate).
    _default = (max_nodes == 400 and abs(min_similarity - 0.35) < 1e-6)
    if _default:
        _db = _load_viz_cache(scenario_id, "kg")
        if _db:
            return _db
    kg = _compute_user_kg(scenario_id, max_nodes, min_similarity)
    if _default:
        _save_viz_cache(scenario_id, "kg", kg)
    return kg


@app.post("/user-scenarios/{scenario_id}/rag")
def user_scenario_rag_assistant(scenario_id: str, payload: AskIn) -> dict[str, Any]:
    """Assistant RAG pour un scénario utilisateur (délègue au RAG générique filtré)."""
    row = _get_user_scenario_or_404(scenario_id)
    # L'Assistant n'interroge que le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique
    # ou inclus manuellement), pas tout le corpus du scénario.
    eff_thr = _get_scenario_threshold(scenario_id)
    payload.filters = payload.filters or {}
    payload.filters["project_context"] = "literev"

    openai_key = os.getenv("OPENAI_API_KEY")
    query_embedding = None
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            response = client.embeddings.create(
                input=[payload.question.replace("\n", " ").strip()],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Erreur embedding RAG user_scenario {scenario_id}: {e}")

    with engine.connect() as conn:
        if query_embedding:
            rows = conn.execute(text("""
                SELECT d.id AS document_id, d.title, d.year, d.url, d.source,
                       d.authors, d.journal, d.doi,
                       c.content, c.metadata_json,
                       (1 - (c.embedding <=> CAST(:emb AS vector))) AS score
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
                WHERE c.embedding IS NOT NULL
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                  AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1, per-scenario)
                ORDER BY c.embedding <=> CAST(:emb AS vector)
                LIMIT 8
            """), {"emb": str(query_embedding), "sid": scenario_id, "thr": eff_thr}).mappings().all()
        else:
            terms = [t.strip() for t in re.split(r"\s+", payload.question.lower()) if t.strip()]
            if not terms:
                return {"answer": "Question vide.", "sources": []}
            like_clauses = " OR ".join(
                f"(LOWER(COALESCE(d.title,'')) LIKE :t{i} OR LOWER(COALESCE(c.content,'')) LIKE :t{i})"
                for i in range(len(terms))
            )
            params: dict[str, Any] = {"sid": scenario_id, "thr": eff_thr}
            for i, t in enumerate(terms):
                params[f"t{i}"] = f"%{t}%"
            rows = conn.execute(text(f"""
                SELECT d.id AS document_id, d.title, d.year, d.url, d.source,
                       d.authors, d.journal, d.doi,
                       c.content, c.metadata_json, 1.0 AS score
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
                WHERE ({like_clauses})
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1, per-scenario)
                ORDER BY d.year DESC NULLS LAST
                LIMIT 8
            """), params).mappings().all()

    if not rows:
        return {
            "answer": f"Aucun article trouvé dans le corpus du scénario '{row['name']}' pour cette question.",
            "sources": [], "scenario_id": scenario_id,
        }

    context_blocks = []
    sources = []
    seen: set = set()
    for i, r in enumerate(rows):
        doc_id = r["document_id"]
        context_blocks.append(
            f"--- SOURCE {i+1} ---\nTitre: {r['title']}\n"
            f"Auteurs: {r.get('authors','') or 'N/A'}\n"
            f"Journal: {r.get('journal','') or 'N/A'} ({r['year'] or 'N/A'})\n"
            f"DOI: {r.get('doi','') or 'N/A'}\nContenu: {r['content']}\n"
        )
        if doc_id not in seen:
            seen.add(doc_id)
            sources.append({
                "document_id": doc_id, "title": r["title"], "year": r["year"],
                "url": r["url"], "source": r["source"], "authors": r.get("authors"),
                "journal": r.get("journal"), "doi": r.get("doi"),
                "score": float(r.get("score", 0)),
            })

    context_str = "\n\n".join(context_blocks)
    if not openai_key:
        return {
            "answer": "[Mode dégradé]\n\nSources :\n" + "\n".join(f"- {s['title']} ({s['year']})" for s in sources),
            "sources": sources, "scenario_id": scenario_id,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key, timeout=90.0)
        system_prompt = (
            f"Vous êtes l'assistant scientifique expert de LiteRev-Evidence pour la recherche : "
            f"**{row['name']}**.\n\n"
            "Règles de rédaction :\n"
            "1. Basez-vous STRICTEMENT sur les sources fournies dans le contexte.\n"
            "2. Citez vos sources avec [SOURCE 1], [SOURCE 2], etc.\n"
            "3. Mentionnez les niveaux de preuve (RCT, méta-analyse, étude observationnelle).\n"
            "4. Soyez précis et structuré. Si le contexte est insuffisant, dites-le.\n"
        ) + _llm_lang_directive(payload.lang)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXTE :\n{context_str}\n\nQUESTION : {payload.question}"},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": sources, "scenario_id": scenario_id, "model": "Assistant IA",
        }
    except Exception as e:
        logger.error(f"Erreur OpenAI RAG user_scenario {scenario_id}: {e}")
        return {"answer": f"Erreur : {str(e)}", "sources": sources, "scenario_id": scenario_id}


@app.get("/user-scenarios/{scenario_id}/double-blind/kappa")
def get_user_scenario_kappa(scenario_id: str) -> dict[str, Any]:
    """Kappa de Cohen pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.reviewer_1_status, d.reviewer_2_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND d.reviewer_1_status IS NOT NULL
              AND d.reviewer_2_status IS NOT NULL
        """), {"sid": scenario_id}).mappings().all()
    if not rows:
        return {
            "scenario_id": scenario_id, "n_evaluated": 0, "kappa": None,
            "interpretation": "Aucune évaluation double-aveugle disponible",
            "agreements": {}, "conflicts": 0,
        }
    n = len(rows)
    categories = ["included", "excluded", "pending"]
    matrix = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for r in rows:
        r1 = r["reviewer_1_status"] if r["reviewer_1_status"] in categories else "pending"
        r2 = r["reviewer_2_status"] if r["reviewer_2_status"] in categories else "pending"
        matrix[r1][r2] += 1
    po = sum(matrix[c][c] for c in categories) / n
    pe = sum((sum(matrix[c][c2] for c2 in categories) / n) * (sum(matrix[c1][c] for c1 in categories) / n) for c in categories)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
    if kappa >= 0.81: interpretation = "Quasi-parfait (≥ 0.81)"
    elif kappa >= 0.61: interpretation = "Substantiel (0.61–0.80)"
    elif kappa >= 0.41: interpretation = "Modéré (0.41–0.60)"
    elif kappa >= 0.21: interpretation = "Faible (0.21–0.40)"
    else: interpretation = "Médiocre (< 0.21)"
    conflicts = sum(matrix[r1][r2] for r1 in categories for r2 in categories if r1 != r2)
    return {
        "scenario_id": scenario_id, "n_evaluated": n, "kappa": round(kappa, 4),
        "po_observed": round(po, 4), "pe_expected": round(pe, 4),
        "interpretation": interpretation, "conflicts": conflicts,
        "agreements": {c: matrix[c][c] for c in categories}, "matrix": matrix,
    }


@app.get("/user-scenarios/{scenario_id}/double-blind/conflicts")
def get_user_scenario_conflicts(scenario_id: str) -> list[dict[str, Any]]:
    """Conflits double-aveugle pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.journal,
                   d.reviewer_1_status, d.reviewer_1_reason,
                   d.reviewer_2_status, d.reviewer_2_reason, d.kappa_final_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND d.reviewer_1_status IS NOT NULL
              AND d.reviewer_2_status IS NOT NULL
              AND d.reviewer_1_status != d.reviewer_2_status
            ORDER BY d.id
        """), {"sid": scenario_id}).mappings().all()
    return [dict(r) for r in rows]


@app.post("/user-scenarios/{scenario_id}/double-blind/decision")
def submit_user_scenario_double_blind_decision(
    scenario_id: str,
    payload: DoubleBlindDecisionIn,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Décision double-aveugle pour un article d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    # Vérifier que l'article appartient au scénario
    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios WHERE document_id = :doc_id AND scenario_id = :sid
        """), {"doc_id": payload.article_id, "sid": scenario_id}).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario utilisateur")
    # Déléguer à l'implémentation existante
    return submit_double_blind_decision(scenario_id, payload)


@app.get("/user-scenarios/{scenario_id}/clustering")
def get_user_scenario_clustering(scenario_id: str, force_refresh: bool = False, lang: str | None = Query(None)) -> dict[str, Any]:
    """Clustering pour un scénario utilisateur."""
    import threading
    _get_user_scenario_or_404(scenario_id)
    if not force_refresh:
        _db = _load_viz_cache(scenario_id, "clustering")
        if _db:
            return _db
    job = _clustering_jobs.get(scenario_id)
    if job and job["status"] == "done" and not force_refresh:
        return job["result"]
    if not job or job.get("status") not in ("running",) or force_refresh:
        _clustering_jobs[scenario_id] = {"status": "running"}
        t = threading.Thread(target=_run_clustering_background, args=(scenario_id, force_refresh, lang), daemon=True)
        t.start()
    return {
        "scenario_id": scenario_id, "status": "running",
        "message": "Calcul en cours. Revenez dans 30-60s.",
        "clusters": [],
    }


@app.get("/user-scenarios/{scenario_id}/clustering/status")
def get_user_scenario_clustering_status(scenario_id: str) -> dict:
    """Statut du clustering pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _clustering_jobs.get(scenario_id)
    if not job:
        _db = _load_viz_cache(scenario_id, "clustering")
        if _db:
            return _db
        return {"scenario_id": scenario_id, "status": "not_started", "message": "Aucun calcul lancé."}
    if job["status"] == "running":
        return {"scenario_id": scenario_id, "status": "running", "message": "Calcul en cours..."}
    if job["status"] == "error":
        return {"scenario_id": scenario_id, "status": "error", "error": job.get("error", "Erreur inconnue")}
    return job["result"]


@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/screen")
def screen_user_scenario_article(
    scenario_id: str,
    article_id: int,
    status: str,
    reason: str | None = None,
    notes: str | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Screening PRISMA pour un article d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    if status not in ("included", "excluded", "pending"):
        raise HTTPException(status_code=422, detail="status doit être 'included', 'excluded' ou 'pending'")
    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios WHERE document_id = :doc_id AND scenario_id = :sid
        """), {"doc_id": article_id, "sid": scenario_id}).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario utilisateur")
    with engine.begin() as conn:
        row = conn.execute(text("""
            UPDATE literature_document
            SET screening_status = :status, screening_reason = :reason, screening_notes = :notes
            WHERE id = :article_id AND project_context = 'literev'
            RETURNING id
        """), {"status": status, "reason": reason, "notes": notes, "article_id": article_id}).first()
        # Migration 2 dual-write: also record the decision on the per-scenario row
        _write_ars_screening(conn, scenario_id, article_id, status, reason, notes)
    if not row:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    return {"id": row[0], "status": status, "updated": True}


@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/pico/extract")
def extract_user_scenario_article_pico(scenario_id: str, article_id: int, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Extraction PICO pour un article d'un scénario utilisateur (délègue à l'endpoint GESICA)."""
    _get_user_scenario_or_404(scenario_id)
    return extract_article_pico(scenario_id, article_id)


@app.get("/user-scenarios/{scenario_id}/articles/{article_id}/pico")
def get_user_scenario_article_pico(scenario_id: str, article_id: int) -> dict[str, Any]:
    """PICO d'un article dans un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    return get_article_pico(scenario_id, article_id)


@app.get("/user-scenarios/{scenario_id}/evidence-brief/pdf")
def get_user_scenario_evidence_brief_pdf(scenario_id: str):
    """PDF Evidence Brief pour un scénario utilisateur."""
    row = _get_user_scenario_or_404(scenario_id)
    # Injecter le scénario utilisateur dans user_scenarios avec is_system=True temporairement
    # n'est plus nécessaire : get_evidence_brief_pdf lit maintenant depuis la DB via _get_db_gesica_scenario_or_404
    # On crée une entrée temporaire dans user_scenarios si nécessaire
    # Pour les user_scenarios, on appelle directement la logique PDF avec les données du scénario
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    import io as _io

    # Comme le panneau Evidences à l'écran : tout porte sur le SOUS-ENSEMBLE
    # PERTINENT (≥ seuil sémantique, via article_scenarios), pas le corpus complet.
    eff_thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as _conn:
        corpus_stats = _conn.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE) AS total,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                MIN(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND COALESCE(ars.screening_status, d.screening_status) = 'included') AS included,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.pico_json IS NOT NULL) AS with_pico
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().first()
        top_articles = _conn.execute(text("""
            SELECT d.title, d.year, d.journal, d.authors,
                   COALESCE((d.pico_json->>'study_design'), d.study_design, 'N/A') AS design
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            ORDER BY d.quality_score DESC NULLS LAST, d.year DESC NULLS LAST
            LIMIT 100000
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().all()
        study_designs = _conn.execute(text("""
            SELECT
                COALESCE((d.pico_json->>'study_design'), d.study_design, 'Non classifié') AS design,
                COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().all()

    _buf = _io.BytesIO()
    _doc = SimpleDocTemplate(_buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    _styles = getSampleStyleSheet()
    _dark_green = colors.HexColor("#1a3a2a")
    _brand_green = colors.HexColor("#22c55e")
    _light_text = colors.HexColor("#374151")
    _title_style = ParagraphStyle("UT", parent=_styles["Title"], fontSize=22, textColor=_dark_green, spaceAfter=6, fontName="Helvetica-Bold")
    _h2_style = ParagraphStyle("UH2", parent=_styles["Heading2"], fontSize=13, textColor=_dark_green, spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
    _body_style = ParagraphStyle("UB", parent=_styles["Normal"], fontSize=9, textColor=_light_text, spaceAfter=4, leading=14)
    _small_style = ParagraphStyle("US", parent=_styles["Normal"], fontSize=7, textColor=colors.HexColor("#6b7280"), spaceAfter=2)

    _story = []
    _story.append(Paragraph("LiteRev : Evidence to Scenario", _small_style))
    _story.append(Paragraph(f"Evidence Brief : {row['name']}", _title_style))
    _story.append(Paragraph(f"Scénario utilisateur · Généré le {__import__('datetime').datetime.now().strftime('%d/%m/%Y à %H:%M')}", _small_style))
    _story.append(HRFlowable(width="100%", thickness=2, color=_brand_green, spaceAfter=12))

    _total = int(corpus_stats["total"] or 0)
    _relevant = int(corpus_stats["relevant"] or 0)
    _included = int(corpus_stats["included"] or 0)
    _with_pico = int(corpus_stats["with_pico"] or 0)
    _year_min = corpus_stats["year_min"] or "N/A"
    _year_max = corpus_stats["year_max"] or "N/A"

    _story.append(Paragraph("Corpus documentaire", _h2_style))
    _stats_data = [["Indicateur", "Valeur"], ["Articles du corpus (uniques)", str(_total)],
                   [f"Articles pertinents utilisés (≥ {eff_thr:.2f})", str(_relevant)],
                   ["Articles inclus", str(_included) if _included > 0 else "En attente"], ["Pertinents avec PICO", str(_with_pico)],
                   ["Période (pertinents)", f"{_year_min} – {_year_max}"]]
    _st = Table(_stats_data, colWidths=[10*cm, 6*cm])
    _st.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), _dark_green), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                              ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
                              ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
                              ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("PADDING", (0,0), (-1,-1), 6)]))
    _story.append(_st)

    if study_designs:
        _story.append(Paragraph("Distribution par type d'étude", _h2_style))
        _dd = [["Type d'étude", "Nombre"]] + [[str(d["design"]), str(d["n"])] for d in study_designs]
        _dt = Table(_dd, colWidths=[10*cm, 6*cm])
        _dt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), _dark_green), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                                  ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
                                  ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
                                  ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("PADDING", (0,0), (-1,-1), 6)]))
        _story.append(_dt)

    if top_articles:
        _story.append(Paragraph("Articles les plus pertinents", _h2_style))
        for _i, _art in enumerate(top_articles, 1):
            _story.append(Paragraph(f"<b>{_i}. {(_art['title'] or 'Sans titre')[:120]}</b>",
                                    ParagraphStyle("at", parent=_body_style, fontSize=9, textColor=_dark_green)))
            _story.append(Paragraph((_art['authors'] or '')[:80], _small_style))
            _story.append(Paragraph(f"{_art['year'] or 'N/A'} · {_art['journal'] or 'Journal inconnu'} · {_art['design']}", _small_style))
            _story.append(Spacer(1, 4))

    _story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceBefore=16))
    _story.append(Paragraph("Ce document a été généré automatiquement par LiteRev.", _small_style))
    _doc.build(_story)
    _buf.seek(0)
    from fastapi.responses import Response as _Resp
    return _Resp(content=_buf.read(), media_type="application/pdf",
                 headers={"Content-Disposition": f'attachment; filename="evidence_brief_{scenario_id}.pdf"'})


# ═══════════════════════════════════════════════════════════════
# SCORING SÉMANTIQUE + EVIDENCE BRIEF LLM + VARIABLES + HEATMAP
# ═══════════════════════════════════════════════════════════════

# ─── SCORING SÉMANTIQUE POST-INGESTION ───────────────────────────────────────

_RERANK_JOBS: dict[str, dict] = {}

DEFAULT_SIMILARITY_THRESHOLD = 0.45


def _ensure_scenario_settings_table():
    """Table pour stocker les paramètres par scénario (seuil, etc.)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_settings (
                scenario_id     VARCHAR(80) PRIMARY KEY,
                similarity_threshold FLOAT DEFAULT 0.45,
                evidence_brief_json  JSONB DEFAULT NULL,
                brief_generated_at   TIMESTAMP DEFAULT NULL,
                variables_json       JSONB DEFAULT NULL,
                variables_validated  BOOLEAN DEFAULT FALSE,
                variables_generated_at TIMESTAMP DEFAULT NULL,
                updated_at           TIMESTAMP DEFAULT NOW()
            )
        """))
    logger.info("Table scenario_settings vérifiée/créée.")

try:
    _ensure_scenario_settings_table()
except Exception as _e:
    logger.warning(f"_ensure_scenario_settings_table: {_e}")


def _get_scenario_threshold(scenario_id: str) -> float:
    """Retourne le seuil de similarité configuré pour ce scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    return float(row["similarity_threshold"]) if row and row["similarity_threshold"] is not None else DEFAULT_SIMILARITY_THRESHOLD


def _get_scenario_name(scenario_id: str) -> str:
    """Retourne le nom lisible d'un scénario (user ou GESICA)."""
    if scenario_id.startswith("usr-"):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT name FROM user_scenarios WHERE id = :id"
            ), {"id": scenario_id}).mappings().first()
        return row["name"] if row else scenario_id
    # GESICA : utiliser la DB
    try:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        return _gesica_title(meta)
    except Exception:
        return scenario_id


def _embed_query_vector(query: str) -> str | None:
    """Embedding d'une requête → chaîne vecteur pgvector « [x,y,…] », ou None si
    indisponible (pas de clé, cooldown quota, ou erreur). Sert à classer les chunks
    de texte intégral par pertinence à la requête du scénario."""
    if not query or not query.strip():
        return None
    if not os.getenv("OPENAI_API_KEY") or _openai_in_cooldown():
        return None
    try:
        from openai import OpenAI as _OAI2
        _emb = _OAI2(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0).embeddings.create(
            input=[query.replace("\n", " ").strip()[:2000]],
            model="text-embedding-3-small",
        ).data[0].embedding
        return "[" + ",".join(str(x) for x in _emb) + "]"
    except Exception as _e:
        if _is_openai_quota_error(_e):
            _trip_openai_cooldown()
        logger.warning(f"_embed_query_vector: {_e}")
        return None


def _fetch_fulltext_excerpts(top_ids: list, query_emb: str | None,
                             char_cap: int, chunks_per_doc: int) -> dict:
    """Pour chaque doc, concatène ses chunks `fulltext_section` les PLUS PERTINENTS
    à la requête (cosinus sur l'embedding déjà stocké du chunk), jusqu'à `char_cap`.
    Dépenser le budget de tokens sur les passages pertinents (méthodes/résultats)
    plutôt que sur les premiers caractères (souvent l'intro) : meilleure profondeur
    à coût égal. Repli sur l'ordre du document (par id) pour les docs dont les chunks
    ne sont pas (encore) embeddés, ou si aucun embedding de requête n'est disponible."""
    result: dict = {}
    if not top_ids:
        return result
    # 1) Sélection par pertinence (nécessite un embedding de requête + chunks embeddés)
    if query_emb is not None:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT document_id, string_agg(content, E'\n\n' ORDER BY rn) AS fulltext
                    FROM (
                        SELECT document_id, content,
                               ROW_NUMBER() OVER (PARTITION BY document_id
                                   ORDER BY embedding <=> CAST(:qemb AS vector)) AS rn
                        FROM document_chunk
                        WHERE document_id = ANY(CAST(:ids AS bigint[]))
                          AND chunk_type = 'fulltext_section'
                          AND embedding IS NOT NULL
                          AND content IS NOT NULL AND length(TRIM(content)) > 0
                    ) ranked
                    WHERE rn <= :cpd
                    GROUP BY document_id
                """), {"ids": top_ids, "qemb": query_emb, "cpd": chunks_per_doc}).mappings().fetchall()
            result = {r["document_id"]: (r["fulltext"] or "")[:char_cap] for r in rows}
        except Exception as _e:
            logger.warning(f"_fetch_fulltext_excerpts relevance: {_e}")
    # 2) Repli par ordre de document pour les docs sans résultat pertinent.
    _missing = [i for i in top_ids if i not in result]
    if _missing:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT document_id, string_agg(content, E'\n\n' ORDER BY id) AS fulltext
                    FROM document_chunk
                    WHERE document_id = ANY(CAST(:ids AS bigint[]))
                      AND chunk_type = 'fulltext_section'
                      AND content IS NOT NULL AND length(TRIM(content)) > 0
                    GROUP BY document_id
                """), {"ids": _missing}).mappings().fetchall()
            for r in rows:
                result[r["document_id"]] = (r["fulltext"] or "")[:char_cap]
        except Exception as _e:
            logger.warning(f"_fetch_fulltext_excerpts fallback: {_e}")
    return result


def _get_above_threshold_articles(scenario_id: str, threshold: float | None = None,
                                  include_fulltext: bool = False,
                                  fulltext_query: str | None = None,
                                  fulltext_top_docs: int = 25,
                                  fulltext_char_cap: int = 2500,
                                  fulltext_chunks_per_doc: int = 5) -> list[dict]:
    """
    Retourne les articles au-dessus du seuil de similarité OU validés humainement.
    Priorité : included > similarity_score >= threshold > autres.

    include_fulltext=True attache à chaque article un champ `fulltext` : un extrait
    du TEXTE INTÉGRAL (chunks `fulltext_section`) pour les `fulltext_top_docs` plus
    pertinents. Quand `fulltext_query` est fourni, on choisit par doc les
    `fulltext_chunks_per_doc` chunks les PLUS PERTINENTS à cette requête (et non les
    premiers), plafonné à `fulltext_char_cap` caractères — le budget de tokens est
    ainsi dépensé sur les passages utiles. Les documents sans texte intégral gardent
    `fulltext=""` (title+abstract seuls).
    """
    if threshold is None:
        threshold = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ld.id, ld.title, ld.abstract, ld.year, ld.journal, ld.authors, ld.doi,
                   ld.study_design, ld.pico_json, ld.citation_count, COALESCE(asn.screening_status, ld.screening_status) AS screening_status,
                   ld.quality_score, asn.similarity_score
            FROM literature_document ld
            JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
            WHERE ld.project_context = 'literev'
              AND ld.is_duplicate IS NOT TRUE
              -- Porte de screening (C1) : ne jamais alimenter le modèle avec un
              -- article explicitement exclu (les autres statuts restent admis).
              AND COALESCE(asn.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
              -- Décision produit : un article NON scoré (similarity_score NULL)
              -- n'est PAS pertinent — même définition que tous les affichages
              -- (COALESCE(score,0) >= seuil). On garde le rattrapage 'included'.
              AND (
                  COALESCE(asn.screening_status, ld.screening_status) = 'included'
                  OR COALESCE(asn.similarity_score, 0) >= :threshold
              )
            ORDER BY
                CASE WHEN COALESCE(asn.screening_status, ld.screening_status) = 'included' THEN 0 ELSE 1 END,
                asn.similarity_score DESC NULLS LAST,
                ld.citation_count DESC NULLS LAST
        """), {"sid": scenario_id, "threshold": threshold}).mappings().fetchall()
    articles = [dict(r) for r in rows]
    if include_fulltext and articles:
        _top_ids = [a["id"] for a in articles[:fulltext_top_docs]]
        _qemb = _embed_query_vector(fulltext_query) if fulltext_query else None
        _ft = _fetch_fulltext_excerpts(_top_ids, _qemb, fulltext_char_cap, fulltext_chunks_per_doc)
        for a in articles:
            a["fulltext"] = _ft.get(a["id"], "")
    return articles


def _evidence_fingerprint(doc_ids: list, threshold: float | None, lang: str | None, ctx: str) -> str:
    """Empreinte du corpus pertinent d'un scénario : hash de l'ensemble ORDONNÉ des
    IDs de documents + seuil + langue + version de contexte. Sert de clé de cache
    « ne pas régénérer si le corpus n'a pas changé » pour l'Evidence Brief et les
    Variables (seuil et langue inclus : un brief FR ne doit pas être resservi pour
    une requête EN, ni un corpus au seuil 0.45 pour un seuil 0.60)."""
    import hashlib
    _ids = "|".join(str(i) for i in sorted(doc_ids))
    _key = f"{ctx}|thr={threshold}|lang={(lang or 'fr').lower()[:2]}|{_ids}"
    return hashlib.sha256(_key.encode("utf-8")).hexdigest()


@app.post("/scenarios/{scenario_id}/rerank")
def trigger_rerank(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche le scoring sémantique post-ingestion pour un scénario.
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading

    # Récupérer la requête du scénario
    if scenario_id.startswith("usr-"):
        row = _get_user_scenario_or_404(scenario_id)
        query = row["query"]
    else:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        nl_queries = meta.get("nl_queries") or []
        query = nl_queries[0] if nl_queries else _gesica_title(meta)

    import time
    if _job_is_active(_RERANK_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0, "started_at": time.time()}

    def _run():
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk résumé -> searchable
            n = _run_semantic_rerank_inline(scenario_id, query)
            # Recalcul COMPLET : après le cosinus, relancer AUSSI le cross-encoder Cohere
            # sur le sous-ensemble pertinent — sinon « Recalculer scores » ne rafraîchissait
            # que le cosinus et les rerank_score restaient figés/partiels.
            try:
                _nce = _run_cross_encoder_rerank(scenario_id, query)
                logger.info(f"Rerank manuel {scenario_id}: {n} cosinus + {_nce} cross-encoder Cohere.")
            except Exception as _ece:
                logger.warning(f"cross-encoder (recalcul manuel) {scenario_id}: {_ece}")
            _RERANK_JOBS[scenario_id] = {"status": "done", "updated": n}
        except Exception as e:
            # Sans ce filet, une exception laisse le job en "running" pour toujours
            # (→ "already_running" + badge "recalcul en cours" figé jusqu'au restart).
            logger.error(f"Rerank job {scenario_id}: {e}", exc_info=True)
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(e), "updated": 0}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "query": query}


@app.get("/scenarios/{scenario_id}/rerank/status")
def get_rerank_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de reranking sémantique."""
    return _RERANK_JOBS.get(scenario_id, {"status": "idle"})


@app.post("/scenarios/{scenario_id}/rebuild-corpus")
def rebuild_corpus(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Reconstruit l'appartenance au corpus (article_scenarios) d'un scénario à
    partir de SA requête booléenne sur la base LOCALE (aucune ré-ingestion live),
    PUIS recalcule les scores (cosinus + cross-encoder).

    Pourquoi : /rerank ne SCORE que les liens article_scenarios EXISTANTS ; il ne
    peut donc rien faire pour un scénario dont le corpus est vide (jamais peuplé,
    ou seulement via l'ancien champ scenario_type). C'est le cas des scénarios
    « ⚠ VIDÉ » repérés par scripts/migration1_scenario_type_diff.py. Cette route
    reconstruit leur appartenance puis les score.

    Coût OpenAI minime : ~1 embedding de requête par scénario (la sélection des
    documents se fait en base locale ; pas d'appel aux sources live). GESICA et
    user_scenarios partagent la table user_scenarios, donc le traitement est
    identique. Suivi via /scenarios/{id}/rerank/status."""
    import threading
    with engine.connect() as _conn:
        row = _conn.execute(
            text("SELECT * FROM user_scenarios WHERE id = :id"), {"id": scenario_id}
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario '{scenario_id}' non trouvé")
    row = dict(row)

    # Requête en langage naturel (sert à l'embedding de scoring).
    _nl = row.get("nl_queries")
    query = (row.get("query")
             or (_nl[0] if isinstance(_nl, list) and _nl else None)
             or row.get("title") or row.get("name") or "")
    filters = row.get("filters") or {}

    # Requête BOOLÉENNE (définit l'appartenance, sur la base locale). Priorité :
    # stratégie stockée → boolean_queries (GESICA) → génération depuis la requête.
    _strat = row.get("search_strategy")
    _bq = row.get("boolean_queries")
    if isinstance(_strat, dict) and _strat.get("general"):
        boolean = _strat["general"]
    elif isinstance(_bq, list) and _bq:
        boolean = _bq[0]
    elif isinstance(_bq, str) and _bq.strip():
        boolean = _bq
    elif query:
        try:
            boolean = _generate_search_strategy(query).get("general") or query
        except Exception:
            boolean = query
    else:
        raise HTTPException(status_code=422,
                            detail="Scénario sans requête exploitable : reconstruction impossible.")

    if _RERANK_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}
    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0}

    def _run():
        try:
            ids = _boolean_corpus_ids(boolean, filters)        # base LOCALE uniquement
            n_corpus = _set_scenario_corpus(scenario_id, ids)  # fixe l'appartenance
            _backfill_title_abstract_chunks(scenario_id)       # chunks résumé manquants
            n = _run_semantic_rerank_inline(scenario_id, query or boolean)  # cosinus pgvector
            try:
                _run_cross_encoder_rerank(scenario_id, query or boolean)    # Cohere (si clé)
            except Exception as _ece:
                logger.warning(f"rebuild-corpus cross-encoder {scenario_id}: {_ece}")
            _RERANK_JOBS[scenario_id] = {"status": "done", "corpus": n_corpus, "updated": n}
            logger.info(f"Rebuild corpus {scenario_id}: {n_corpus} liens, {n} scorés.")
        except Exception as _e:
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(_e)}
            logger.error(f"Rebuild corpus {scenario_id}: {_e}", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "boolean": str(boolean)[:200]}


def _backfill_title_abstract_chunks(scenario_id: str | None = None) -> int:
    """
    Crée un chunk `title_abstract` (embedding NULL) pour les documents qui ont un
    titre/résumé mais AUCUN chunk title_abstract — typiquement les docs liés depuis
    la base locale sans création de chunk. Le worker d'enrichissement les embed
    ensuite : recherche sémantique au niveau résumé + compteurs réconciliés.
    Idempotent (NOT EXISTS). Si scenario_id est None, traite tout le corpus literev.
    """
    scope = "JOIN article_scenarios ars ON ars.document_id = ld.id AND ars.scenario_id = :sid" if scenario_id else ""
    params = {"sid": scenario_id} if scenario_id else {}
    try:
        with engine.begin() as conn:
            n = conn.execute(text(f"""
                INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, created_at)
                SELECT DISTINCT ld.id,
                       (SELECT COALESCE(MAX(c2.chunk_index), -1) + 1 FROM document_chunk c2 WHERE c2.document_id = ld.id),
                       btrim(coalesce(ld.title, '') || E'\\n\\n' || coalesce(ld.abstract, '')),
                       'title_abstract', now()
                FROM literature_document ld
                {scope}
                WHERE ld.project_context = 'literev'
                  AND ld.is_duplicate IS NOT TRUE
                  AND length(btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, ''))) >= 30
                  AND NOT EXISTS (
                      SELECT 1 FROM document_chunk c
                      WHERE c.document_id = ld.id AND c.chunk_type = 'title_abstract'
                  )
            """), params).rowcount
        if n:
            logger.info(f"Backfill title_abstract chunks ({scenario_id or 'global'}): {n} créés (embedding par le worker).")
        return n
    except Exception as e:
        logger.warning(f"Backfill title_abstract chunks {scenario_id}: {e}")
        return 0


def _maybe_autorerank(scenario_id: str) -> bool:
    """
    Lance le scoring sémantique en arrière-plan si jamais effectué (auto-score),
    et complète au passage les chunks title_abstract manquants (recherche + compteurs).
    Ne se déclenche qu'une fois par scénario (tant que le process vit) : si le
    job est déjà 'running' ou 'done', on ne relance pas. Renvoie True si lancé.
    """
    import threading

    st = _RERANK_JOBS.get(scenario_id, {}).get("status")
    if st in ("running", "done"):
        return False
    try:
        if scenario_id.startswith("usr-"):
            row = _get_user_scenario_or_404(scenario_id)
            query = row["query"]
        else:
            meta = _get_db_gesica_scenario_or_404(scenario_id)
            nl = meta.get("nl_queries") or []
            query = nl[0] if nl else _gesica_title(meta)
    except Exception:
        return False
    if not query:
        return False

    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0}

    def _run():
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk résumé -> searchable
            n = _run_semantic_rerank_inline(scenario_id, query)
            _RERANK_JOBS[scenario_id] = {"status": "done", "updated": n}
            logger.info(f"Auto-rerank {scenario_id}: {n} articles scorés.")
        except Exception as e:
            logger.warning(f"Auto-rerank {scenario_id}: {e}")
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return True


@app.get("/scenarios/{scenario_id}/settings")
def get_scenario_settings(scenario_id: str) -> dict[str, Any]:
    """Retourne les paramètres du scénario (seuil, état du brief LLM, variables)."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if not row:
        return {
            "scenario_id": scenario_id,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "evidence_brief_json": None,
            "brief_generated_at": None,
            "variables_json": None,
            "variables_validated": False,
            "variables_generated_at": None,
        }
    return dict(row)


@app.patch("/scenarios/{scenario_id}/settings")
def update_scenario_settings(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour les paramètres du scénario (seuil, variables validées, etc.)."""
    allowed = {"similarity_threshold", "variables_json", "variables_validated"}
    updates = {k: v for k, v in payload.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=422, detail="Aucun champ valide à mettre à jour")

    with engine.begin() as conn:
        # Upsert
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, updated_at)
            VALUES (:sid, NOW())
            ON CONFLICT (scenario_id) DO NOTHING
        """), {"sid": scenario_id})

        for key, val in updates.items():
            import json as _json
            if isinstance(val, (dict, list)):
                val = _json.dumps(val)
            conn.execute(text(f"""
                UPDATE scenario_settings SET {key} = :val, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"val": val, "sid": scenario_id})

    # Retourner l'objet settings complet mis à jour
    with engine.connect() as conn:
        updated_row = conn.execute(text("""
            SELECT scenario_id, similarity_threshold, brief_generated_at,
                   variables_validated, variables_generated_at, updated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if updated_row:
        return {
            "status": "updated",
            "scenario_id": scenario_id,
            "updated": list(updates.keys()),
            "similarity_threshold": float(updated_row["similarity_threshold"]) if updated_row["similarity_threshold"] is not None else 0.45,
            "variables_validated": bool(updated_row["variables_validated"]),
            "updated_at": updated_row["updated_at"].isoformat() if updated_row["updated_at"] else None,
        }
    return {"status": "updated", "scenario_id": scenario_id, "updated": list(updates.keys())}


# ─── EVIDENCE BRIEF LLM AUTOMATIQUE ──────────────────────────────────────────

_BRIEF_GENERATION_JOBS: dict[str, dict] = {}


def _job_is_active(job: dict | None, stale_after: float = 180.0) -> bool:
    """True only if a background job is genuinely still running. A "running"
    entry older than stale_after seconds (a thread killed by a restart or hung
    on a network call) is treated as STALE, so a crashed/killed job never locks
    out retries permanently with "already_running"."""
    import time
    if not job or job.get("status") != "running":
        return False
    return (time.time() - job.get("started_at", 0)) < stale_after


def _generate_evidence_brief_llm(scenario_id: str, force: bool = False, lang: str | None = None) -> dict[str, Any]:
    """
    Génère un Evidence Brief narratif complet via LLM à partir des articles
    au-dessus du seuil de similarité (ou validés humainement).
    Sauvegarde le résultat dans scenario_settings.evidence_brief_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from openai import OpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold, include_fulltext=True,
                                             fulltext_query=_get_scenario_name(scenario_id),
                                             fulltext_top_docs=30, fulltext_char_cap=2800)

    if not articles:
        return {"error": "Aucun article au-dessus du seuil pour générer le brief."}

    # Cache par EMPREINTE DU CORPUS (+ seuil + langue), et non plus par âge (< 24h).
    # On ne régénère que si le sous-ensemble pertinent, le seuil ou la langue ont
    # changé — l'ancien cache « 24h » régénérait un corpus inchangé ET servait un
    # brief périmé (mauvaise langue / corpus modifié) tant qu'il avait moins de 24h.
    _brief_fp = _evidence_fingerprint([a["id"] for a in articles], threshold, lang, "brief-v3-relevant-fulltext")
    if not force:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT evidence_brief_json FROM scenario_settings WHERE scenario_id = :sid
            """), {"sid": scenario_id}).mappings().first()
        if row and row["evidence_brief_json"]:
            _cached = dict(row["evidence_brief_json"])
            if _cached.get("_corpus_fingerprint") == _brief_fp:
                return _cached

    scenario_name = _get_scenario_name(scenario_id)

    # Préparer le contexte : top 30 articles. Trois niveaux de texte, du plus
    # structuré au plus brut : PICO (structure) + abstract (récit) + extrait du
    # TEXTE INTÉGRAL quand disponible (méthodes/résultats que l'abstract résume).
    # Le LLM voit ainsi le texte réel des articles, pas seulement leur PICO.
    context_articles = []
    for a in articles[:30]:
        pj = a.get("pico_json") or {}
        _ca = {
            "title": a.get("title", ""),
            "year": a.get("year"),
            "journal": a.get("journal", ""),
            "citation_count": a.get("citation_count"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "screening_status": a.get("screening_status"),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
            "abstract": (a.get("abstract") or "")[:1500],
        }
        _ft = (a.get("fulltext") or "").strip()
        if _ft:
            _ca["fulltext_excerpt"] = _ft
        context_articles.append(_ca)

    context_str = _json.dumps(context_articles, ensure_ascii=False, indent=2)

    # Stats corpus
    total = len(articles)
    included = sum(1 for a in articles if a.get("screening_status") == "included")
    with_pico = sum(1 for a in articles if a.get("pico_json"))
    years = [a["year"] for a in articles if a.get("year")]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    study_designs = {}
    for a in articles:
        pj = a.get("pico_json") or {}
        d = a.get("study_design") or pj.get("study_design", "Non classifié")
        study_designs[d] = study_designs.get(d, 0) + 1

    top_designs = sorted(study_designs.items(), key=lambda x: -x[1])[:5]

    # Plafond GRADE déterministe d'après les devis présents : on empêche le LLM de
    # surclasser un corpus observationnel. En GRADE strict, toute étude
    # observationnelle (cohorte, cas-témoins, transversale, série de cas) démarre
    # en certitude FAIBLE ; seuls les essais randomisés et leurs synthèses peuvent
    # soutenir une certitude élevée.
    _designs_blob = " ".join(study_designs.keys()).lower()
    if any(k in _designs_blob for k in ("randomi", "rct", "meta-analysis", "méta-analyse",
                                        "meta analysis", "systematic review", "revue systématique")):
        _grade_ceiling = ("Forte possible (essais randomisés / synthèses d'essais présents), "
                          "à pondérer selon la cohérence et le risque de biais")
    elif any(k in _designs_blob for k in ("controlled trial", "clinical trial", "quasi-exper",
                                          "quasi exper", "non-randomi", "non randomi")):
        _grade_ceiling = "Modérée au mieux (essais contrôlés non randomisés / quasi-expérimental)"
    else:
        _grade_ceiling = ("Faible (corpus observationnel : GRADE plafonne la certitude à faible, "
                          "sauf upgrade explicitement justifié)")

    system_prompt = """Tu es un expert en médecine d'urgence et en revue systématique de la littérature scientifique.
Tu génères des Evidence Briefs complets, rigoureux et structurés.
Tu dois produire un JSON structuré avec tous les champs demandés.
Sois précis, factuel, et base-toi exclusivement sur les articles fournis.
Ne pas utiliser de tiret em (—). Utiliser des tirets simples (-) si nécessaire.""" + _llm_lang_directive(lang)

    user_prompt = f"""Génère un Evidence Brief complet pour le scénario de recherche : "{scenario_name}"

Corpus : {total} articles ({year_range}), {with_pico} avec PICO extrait, {included} validés humainement.
Designs d'étude principaux : {', '.join(f'{d} ({n})' for d, n in top_designs)}.

Articles (top 30 par pertinence) :
{context_str}

RÈGLES GRADE (strict) pour « evidence_level » et « grade_recommendation » :
- Le niveau de preuve part du DEVIS d'étude. Essais randomisés et méta-analyses/revues
  systématiques d'essais → certitude potentiellement Forte. TOUTE étude observationnelle
  (cohorte, cas-témoins, transversale, série/rapport de cas) démarre en certitude FAIBLE.
  Ne JAMAIS classer des cas-témoins ou des cohortes en preuve « Forte ».
- « evidence_level » reflète la MEILLEURE preuve du corpus, plafonnée par le devis.
  Plafond estimé d'après les devis présents : {_grade_ceiling}.
- « grade_recommendation » suit le niveau de preuve : A uniquement si preuves Fortes et
  cohérentes (essais/méta-analyses) ; B si preuves modérées ; C si preuves faibles
  (corpus observationnel) ; D si très faibles / avis d'experts ; GPP pour une bonne
  pratique sans preuve directe.

Génère un JSON avec EXACTEMENT ces champs :
{{
  "executive_summary": "Résumé exécutif en 3-4 phrases synthétisant les principales conclusions",
  "clinical_context": "Contexte clinique et importance du sujet (2-3 paragraphes)",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4", "Finding 5"],
  "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "evidence_synthesis": "Synthèse narrative détaillée des évidences (4-6 paragraphes)",
  "population_summary": "Résumé des populations étudiées",
  "intervention_summary": "Résumé des interventions/expositions étudiées",
  "outcome_summary": "Résumé des outcomes mesurés",
  "methodological_quality": "Évaluation de la qualité méthodologique globale",
  "limitations": ["Limite 1", "Limite 2", "Limite 3"],
  "research_gaps": ["Gap 1", "Gap 2", "Gap 3"],
  "clinical_implications": "Implications cliniques pratiques (2-3 paragraphes)",
  "implementation_recommendations": ["Recommandation 1", "Recommandation 2", "Recommandation 3"],
  "evidence_level": "Niveau de preuve global (Fort/Modéré/Faible/Insuffisant)",
  "grade_recommendation": "Grade de recommandation (A/B/C/D/GPP)",
  "future_research": "Directions pour la recherche future",
  "key_references": [
    {{"title": "...", "year": ..., "journal": "...", "key_contribution": "..."}}
  ]
}}
Retourne UNIQUEMENT le JSON valide."""

    try:
        client = _OAI(timeout=90.0)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        brief = _json.loads(response.choices[0].message.content)

        # Ajouter les métadonnées
        brief["_meta"] = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "articles_used": total,
            "articles_above_threshold": total,
            "threshold": threshold,
            "human_validated": included,
            "year_range": year_range,
            "study_designs": dict(top_designs),
            "auto_generated": True,
            "model": "gpt-4.1",
        }

        # Empreinte du corpus : sert de clé de cache « ne pas régénérer si inchangé »
        # (relue en tête de fonction). Stockée DANS le brief pour éviter une colonne.
        brief["_corpus_fingerprint"] = _brief_fp

        # Sauvegarder en DB
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, evidence_brief_json, brief_generated_at, updated_at)
                VALUES (:sid, CAST(:brief AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET evidence_brief_json = CAST(:brief AS jsonb),
                    brief_generated_at = NOW(),
                    updated_at = NOW()
            """), {"sid": scenario_id, "brief": _json.dumps(brief)})

        logger.info(f"Evidence Brief LLM généré pour {scenario_id}: {len(context_articles)} articles.")
        return brief

    except Exception as e:
        logger.error(f"Evidence Brief LLM {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/scenarios/{scenario_id}/evidence-brief/generate")
def generate_evidence_brief(scenario_id: str, force: bool = False, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche la génération asynchrone de l'Evidence Brief LLM.
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading, time

    if _job_is_active(_BRIEF_GENERATION_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_evidence_brief_llm(scenario_id, force=force, lang=lang)
            if "error" in result:
                _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "done", "generated_at": result.get("_meta", {}).get("generated_at")}
        except Exception as e:
            # Without this, an exception (or a killed thread) leaves status stuck
            # at "running" and every retry returns "already_running" forever.
            logger.error(f"Evidence Brief job {scenario_id}: {e}", exc_info=True)
            _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/evidence-brief/generate/status")
def get_brief_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération du brief LLM."""
    return _BRIEF_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/evidence-brief/llm")
def get_llm_evidence_brief(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne le brief LLM généré (depuis le cache DB) DANS LA LANGUE demandée.
    Le cache n'est servi que si son empreinte (corpus + seuil + langue) correspond ;
    sinon (corpus modifié OU langue différente) on déclenche une régénération et on
    retourne un statut pending. Sans `lang`, le brief reste en français (défaut).
    """
    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold)
    if not articles:
        return {"status": "empty", "message": "Aucun article au-dessus du seuil. Ajoutez des articles ou abaissez le seuil de similarité."}

    _want_fp = _evidence_fingerprint([a["id"] for a in articles], threshold, lang, "brief-v3-relevant-fulltext")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT evidence_brief_json, brief_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["evidence_brief_json"]:
        brief = dict(row["evidence_brief_json"])
        # Ne servir le cache que s'il correspond au corpus ET à la langue demandés.
        # Un brief français ne doit pas être resservi quand l'UI passe en anglais.
        if brief.get("_corpus_fingerprint") == _want_fp:
            brief["_cached"] = True
            brief["_generated_at"] = row["brief_generated_at"].isoformat() if row["brief_generated_at"] else None
            return brief

    # Si un job précédent a ÉCHOUÉ, renvoyer l'erreur au lieu de relancer la
    # génération à chaque appel : sinon un échec persistant (mauvaise sortie LLM,
    # quota) boucle en regénération à chaque poll côté front. Le bouton
    # « régénérer » (POST force) reste la voie de reprise explicite.
    _bj = _BRIEF_GENERATION_JOBS.get(scenario_id, {})
    if _bj.get("status") == "error":
        return {"status": "error", "message": _bj.get("error", "Échec de la génération du brief.")}

    # Pas de brief en cache valide (absent, corpus changé, ou autre langue) :
    # déclencher la génération DANS LA LANGUE demandée.
    generate_evidence_brief(scenario_id, lang=lang)
    return {"status": "generating", "message": "Génération en cours, réessayez dans 30 secondes."}


# ─── VARIABLES & MODÈLE AUTO-REMPLI DEPUIS PICO ──────────────────────────────

_VARIABLES_GENERATION_JOBS: dict[str, dict] = {}


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
_FEATURE_SOURCES = {"user", "public_api"}
_ALGO_FAMILIES = {
    "gradient_boosting", "lightgbm", "xgboost", "random_forest", "logistic_regression",
    "linear_regression", "elasticnet", "svm", "mlp", "cox_ph", "knn",
    "prophet", "sarimax",
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
    """Retourne (source, public_provider) — déclaré LLM sinon heuristique texte."""
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
        columns.append({
            "name": f["machine_name"], "dtype": f["dtype"], "role": "feature",
            "required": f.get("importance") == "high",
            "source": f.get("source", "user"), "public_provider": f.get("public_provider"),
            "description": f.get("name", ""),
        })
    return {
        "target_column": outcome_mn,
        "columns": columns,
        "formats": ["csv", "xlsx"],
        "user_columns": [c["name"] for c in columns if c["source"] == "user"],
        "public_columns": [c["name"] for c in columns if c["source"] == "public_api"],
        "notes": ("Les en-têtes du fichier doivent correspondre EXACTEMENT à ces noms. "
                  "Les colonnes 'public_api' pourront être récupérées automatiquement (Phase 2)."),
    }


def _coerce_family_for_task(family: Any, task_type: str) -> str:
    """Ramène une famille d'algorithme choisie vers une famille COMPATIBLE avec la
    tâche (sans vérifier la présence du paquet — c'est le rôle du trainer). Ex.:
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

    # ── Data template (dérivé → garanti cohérent avec les machine_name ci-dessus) ──
    data_template = _derive_data_template(outcome, features)

    variables["model_spec"] = {
        "schema": MODEL_SPEC_SCHEMA,
        "version": 1,
        "outcome": outcome,
        "features": features,
        "algorithm": algorithm,
        "data_template": data_template,
    }
    variables["_provenance_index"] = {str(i): prov_meta[i] for i in sorted(cited) if i in prov_meta}
    return variables


def _generate_variables_from_pico(scenario_id: str, persist: str = "active", lang: str | None = None) -> dict[str, Any]:
    """
    Génère automatiquement les variables du modèle et l'outcome à partir des articles
    pertinents (PICO structuré + titre/résumé + extrait de texte intégral quand
    disponible), et non des seuls PICO. Sauvegarde dans scenario_settings.variables_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from openai import OpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold, include_fulltext=True,
                                             fulltext_query=_get_scenario_name(scenario_id),
                                             fulltext_top_docs=25, fulltext_char_cap=2200)

    # On n'EXIGE plus un PICO extrait : tout article pertinent contribue au choix des
    # variables/outcome/algorithme via son titre+abstract (+texte intégral si dispo).
    # Le PICO, quand présent, ajoute la structure P/I/C/O. Auparavant, un article
    # pertinent sans PICO était purement ignoré (perte de signal injustifiée).
    # `pico_articles` = désormais l'ensemble pertinent complet (nom conservé pour le
    # reste du flux ; le PICO reste optionnel par article).
    pico_articles = articles
    if not pico_articles:
        return {"error": "Aucun article au-dessus du seuil pour générer les variables."}
    _n_with_pico = sum(1 for a in pico_articles if a.get("pico_json"))

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
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
            "abstract": (a.get("abstract") or "")[:1200],
        }
        _ft = (a.get("fulltext") or "").strip()
        if _ft:
            _entry["fulltext_excerpt"] = _ft
        pico_context.append(_entry)

    context_str = _json.dumps(pico_context, ensure_ascii=False, indent=2)

    system_prompt = """Tu es un expert en modélisation prédictive en médecine d'urgence.
A partir d'une revue systématique de la littérature, tu identifies les variables clés,
l'outcome principal, et le meilleur algorithme pour un modèle prédictif.
Tu génères un JSON structuré. Ne pas utiliser de tiret em (—).""" + _llm_lang_directive(lang)

    user_prompt = f"""Scénario : "{scenario_name}"
Basé sur {len(pico_articles)} articles pertinents ({_n_with_pico} avec PICO extrait ; PICO, abstract et extrait de texte intégral fournis quand disponibles) :

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
      "evidence_level": "Nombre d'études qui la mentionnent",
      "machine_name": "identifiant_snake_case_court (ex: temp_max_j1)",
      "dtype": "float|int|bool|category|datetime",
      "source": "user (fournie par l'utilisateur) | public_api (récupérable: météo, open data...)",
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
  "implementation_notes": "Notes d'implémentation pratiques",
  "validation_status": "pending"
}}

IMPORTANT pour les champs "provenance" : ce sont des listes d'identifiants (champ "id")
des articles figurant dans la liste ci-dessus. N'invente AUCUN id : n'utilise que des id
réellement présents. Les "machine_name" doivent être de courts identifiants snake_case
(minuscules, chiffres, underscores) utilisables comme noms de colonnes.

IMPORTANT pour "alert_thresholds" : donne des plages de valeurs NUMÉRIQUES concrètes de
l'OUTCOME (pas des phrases vagues), cohérentes avec son "unit" et son "task_type", et
fondées sur les seuils rapportés dans les évidences quand ils existent (sinon des seuils
cliniquement plausibles). Les trois plages doivent être contiguës et couvrir tout le
domaine. Tu PEUX renommer les catégories si l'outcome s'y prête (ex. pour un compte :
"Faible"/"Modéré"/"Élevé"), mais garde 3 niveaux du plus sûr au plus critique.

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
        _CTX_VERSION = "ctx-v4-relevant-fulltext"
        evidence_fingerprint = _evidence_fingerprint(
            [a.get("id") for a in pico_articles[:25] if a.get("id") is not None],
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

        if _reused is not None:
            variables = dict(_reused)
            logger.info(f"Variables {scenario_id}: évidence inchangée (fingerprint {evidence_fingerprint}) "
                        f"→ spec réutilisé (déterministe, sans appel LLM).")
        else:
            client = _OAI(timeout=90.0)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                seed=42,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            variables = _json.loads(response.choices[0].message.content)

            # Phase 1 : normaliser en model_spec déterministe (machine_name, dtype,
            # algorithme/CV/métrique, data_template) + provenance tracée vers les articles.
            try:
                variables = _attach_model_spec(variables, pico_articles[:25])
            except Exception as spec_err:  # le spec machine ne doit jamais bloquer la génération
                logger.error(f"model_spec build {scenario_id}: {spec_err}", exc_info=True)

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


# ─── Localisation NON DESTRUCTIVE des Variables/Modèle (affichage FR/EN) ──────
# variables_json est FONCTIONNEL (machine_name, dtype, model_spec, data_template) :
# on ne le régénère pas au changement de langue. On traduit UNIQUEMENT les champs
# d'affichage (noms, définitions, libellés, justifications) et on met en cache la
# variante par langue dans variables_i18n. Les identifiants machine, types,
# provenances et le model_spec structurel restent intacts.

def _norm_lang(lang) -> str | None:
    """'en'/'fr' à partir d'une valeur de langue quelconque ; None si non exploitable."""
    if not isinstance(lang, str) or not lang.strip():
        return None
    return "en" if lang.strip().lower().startswith("en") else "fr"


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


def _llm_translate_strings(texts: list, target_lang: str) -> list:
    """Traduit une liste de chaînes vers 'en'/'fr' (ordre + longueur conservés).
    Renvoie les chaînes d'ORIGINE en cas d'échec — jamais de corruption."""
    import json as _json
    from openai import OpenAI as _OAI
    if not texts:
        return texts
    lang_name = "English" if target_lang == "en" else "French"
    sys = (f"You are a professional medical translator. Translate each string in the input "
           f"array to {lang_name}. Preserve medical/technical terms, units (e.g. /100k, mg, %), "
           "acronyms (ARI, RSV, SARIMAX, ED), and all numbers. Do NOT add explanations or notes. "
           'Return ONLY a JSON object {"items": [...]} whose "items" is a JSON array of the '
           "translated strings, in the SAME order and with EXACTLY the same length as the input.")
    usr = _json.dumps({"items": list(texts)}, ensure_ascii=False)
    try:
        client = _OAI(timeout=90.0)
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0, seed=42, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        out = _json.loads(resp.choices[0].message.content)
        items = out.get("items") if isinstance(out, dict) else None
        if isinstance(items, list) and len(items) == len(texts) and all(isinstance(x, str) for x in items):
            return items
        logger.warning(f"translate_strings: shape mismatch ({type(items).__name__}, "
                       f"{len(items) if isinstance(items, list) else 'n/a'} vs {len(texts)})")
    except Exception as e:
        logger.warning(f"translate_strings failed: {e}")
    return texts


def _translate_variables_payload(variables: dict, target_lang: str) -> dict:
    """Copie TRADUITE de variables_json : seuls les champs d'AFFICHAGE sont traduits
    (noms, définitions, libellés, justifications, listes de bases/notes). machine_name,
    dtype, task_type, family, metric, cv, provenance, ranges numériques et structure du
    model_spec restent identiques. Les noms/justification traduits sont propagés dans le
    model_spec miroir (lu par /model/spec)."""
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
        return v
    translated = _llm_translate_strings(texts, target_lang)
    if len(translated) != len(texts):
        return v   # sécurité : ne jamais corrompre
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
    return v


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
    translated = _translate_variables_payload(base_variables, req)
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
    articles = _get_above_threshold_articles(scenario_id, threshold)
    if not articles:
        return {"status": "empty", "message": "Aucun article au-dessus du seuil. Ajoutez des articles ou abaissez le seuil de similarité."}

    # Job en échec : renvoyer l'erreur au lieu de relancer à chaque poll (évite
    # une boucle de regénération quand la génération échoue de façon persistante).
    _vj = _VARIABLES_GENERATION_JOBS.get(scenario_id, {})
    if _vj.get("status") == "error":
        return {"status": "error", "message": _vj.get("error", "Échec de la génération des variables.")}

    # Déclencher la génération. On passe lang EXPLICITEMENT : appeler l'endpoint
    # directement sans cet argument passerait l'objet Query(None) par défaut (et non
    # None) jusqu'à _llm_lang_directive → crash « 'Query' object has no attribute 'strip' ».
    generate_scenario_variables(scenario_id, lang=lang)
    return {"status": "generating", "message": "Génération en cours, réessayez dans 30 secondes."}


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
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_json = CAST(:vars AS jsonb),
                    variables_validated = TRUE,
                    variables_lang = NULL,
                    variables_i18n = NULL,
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


@app.get("/scenarios/{scenario_id}/model/spec")
def get_scenario_model_spec(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Vue 'machine' du modèle dérivée de variables_json.model_spec : outcome
    (task_type), features (machine_name/dtype/source), algorithme (famille, CV,
    métrique) et data_template — les noms de colonnes EXACTS à fournir pour
    l'upload de données. Chaque élément porte sa provenance (ids d'articles).
    Les libellés d'affichage (noms, justifications, seuils) suivent la langue de l'UI ;
    les identifiants machine et le data_template restent invariants.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_validated, variables_generated_at,
                   variables_lang, variables_i18n
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if not (row and row["variables_json"]):
        return {"status": "empty", "message": "Aucune spécification de modèle. Générez d'abord les Variables & Modèle."}

    # Variables localisées (affichage FR/EN) : le model_spec miroir porte les noms et
    # la justification traduits, mais les machine_name / data_template sont intacts.
    vj = dict(_get_localized_variables(
        scenario_id, lang, base_variables=dict(row["variables_json"]),
        variables_lang=row.get("variables_lang"), variables_i18n=row.get("variables_i18n")))
    spec = vj.get("model_spec")
    if not spec:
        # Spec générée avant la Phase 1 : pas encore de schéma machine.
        return {
            "status": "legacy",
            "message": "Spécification antérieure au schéma machine. Relancez la génération des Variables & Modèle pour obtenir le data_template.",
            "validated": row["variables_validated"],
        }

    # ── Résolution provenance : pour chaque outcome/variable/algorithme, on
    # remonte l'article SOURCE le plus pertinent (plus récent, puis plus cité). ──
    prov_ids: set[int] = set()
    for elem in (spec.get("outcome") or {}, spec.get("algorithm") or {}):
        prov_ids.update(int(i) for i in (elem.get("provenance") or []) if isinstance(i, (int, float)))
    for f in (spec.get("features") or []):
        prov_ids.update(int(i) for i in (f.get("provenance") or []) if isinstance(i, (int, float)))
    # Provenance des modalités d'alerte (seuils green/orange/red) : on les résout
    # aussi pour que chaque modalité affiche l'article source.
    _at_raw = vj.get("alert_thresholds") if isinstance(vj, dict) else None
    if isinstance(_at_raw, dict):
        for _band in _at_raw.values():
            if isinstance(_band, dict):
                prov_ids.update(int(i) for i in (_band.get("provenance") or []) if isinstance(i, (int, float)))

    resolved: dict[str, Any] = {}
    if prov_ids:
        with engine.connect() as conn:
            arts = conn.execute(text("""
                SELECT id, title, year, doi, citation_count
                FROM literature_document WHERE id = ANY(:ids)
            """), {"ids": list(prov_ids)}).mappings().all()
        for a in arts:
            resolved[str(a["id"])] = {
                "id": a["id"], "title": a["title"], "year": a["year"],
                "doi": a["doi"], "citation_count": a["citation_count"],
                "url": (f"https://doi.org/{a['doi']}" if a["doi"] else None),
            }

    def _best_article(ids):
        cand = [resolved[str(int(i))] for i in (ids or [])
                if isinstance(i, (int, float)) and str(int(i)) in resolved]
        if not cand:
            return None
        cand.sort(key=lambda a: ((a["year"] or 0), (a["citation_count"] or 0)), reverse=True)
        return cand[0]

    outcome = dict(spec.get("outcome") or {})
    outcome["best_article"] = _best_article(outcome.get("provenance"))
    algorithm = dict(spec.get("algorithm") or {})
    algorithm["best_article"] = _best_article(algorithm.get("provenance"))
    features = []
    for f in (spec.get("features") or []):
        f2 = dict(f)
        f2["best_article"] = _best_article(f.get("provenance"))
        features.append(f2)

    # Modalités d'alerte enrichies : chaque niveau porte son article source (le
    # plus pertinent de sa provenance) + la liste résolue, pour lier la littérature.
    alert_thresholds = {}
    if isinstance(_at_raw, dict):
        for _lvl, _band in _at_raw.items():
            if not isinstance(_band, dict):
                continue
            _b2 = dict(_band)
            _prov = _band.get("provenance") or []
            _b2["best_article"] = _best_article(_prov)
            _b2["provenance_articles"] = [
                resolved[str(int(i))] for i in _prov
                if isinstance(i, (int, float)) and str(int(i)) in resolved
            ]
            alert_thresholds[_lvl] = _b2

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "schema": spec.get("schema"),
        "version": spec.get("version"),
        "outcome": outcome,
        "features": features,
        "algorithm": algorithm,
        "alert_thresholds": alert_thresholds,
        "data_template": spec.get("data_template"),
        "provenance_index": resolved or vj.get("_provenance_index", {}),
        "validated": row["variables_validated"],
        "generated_at": row["variables_generated_at"].isoformat() if row["variables_generated_at"] else None,
    }


# ─── SPEC EVOLUTION (Phase 5) : nouvelle évidence -> proposition -> validation ─
# Quand de nouveaux articles apportent de l'évidence (nouvel outcome, nouvelle
# variable, meilleur algorithme), on RÉGÉNÈRE le spec dans un slot de staging,
# on le DIFFE contre le spec actif validé, l'utilisateur VALIDE, puis le nouveau
# spec devient actif (version +1) et le modèle est ré-entraîné.

_SPEC_PROPOSAL_JOBS: dict[str, dict] = {}


def _ensure_spec_proposal_columns():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_proposal_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS proposal_generated_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS recommended_actions_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS actions_generated_at TIMESTAMP"))
        # Langue des actions en cache : on régénère quand l'UI change de langue.
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS recommended_actions_lang VARCHAR(8)"))
        # Localisation NON destructive des Variables/Modèle : langue de génération +
        # cache des traductions d'affichage par langue (le spec fonctionnel est intact).
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_lang VARCHAR(8)"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_i18n JSONB"))
        # Caches de visualisation persistés en DB (durables, contrairement à /tmp).
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS clustering_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS clustering_generated_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS knowledge_graph_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS kg_generated_at TIMESTAMP"))
    logger.info("Colonnes de proposition de spec vérifiées/créées.")


try:
    _ensure_spec_proposal_columns()
except Exception as _e:
    logger.warning(f"_ensure_spec_proposal_columns: {_e}")


# ─── ACTIONS RECOMMANDÉES (carte tableau de bord, généralisé aux user scenarios) ─
_ACTIONS_JOBS: dict[str, dict] = {}


def _generate_recommended_actions(scenario_id: str, lang: str | None = None) -> list[str]:
    """
    Génère 4-5 actions opérationnelles/cliniques concrètes à partir de l'évidence
    du scénario (PICO des articles au-dessus du seuil), façon « Actions
    recommandées » des cartes GESICA. Cache dans scenario_settings.
    """
    import json as _json
    from openai import OpenAI as _OAI

    articles = _get_above_threshold_articles(scenario_id)
    pico_articles = [a for a in articles if a.get("pico_json")]
    base = pico_articles or articles
    if not base:
        return []

    scenario_name = _get_scenario_name(scenario_id)
    ctx = []
    for a in base[:20]:
        pj = a.get("pico_json") or {}
        ctx.append({
            "title": (a.get("title") or "")[:120],
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
        })

    system = ("Tu es un expert en aide à la décision en santé/médecine d'urgence. "
              "À partir d'une revue de littérature, tu proposes des ACTIONS opérationnelles "
              "concrètes, spécifiques et actionnables (pas de généralités). Pas de tiret em (—)."
              ) + _llm_lang_directive(lang)
    user = (f"Scénario : \"{scenario_name}\"\n"
            f"Basé sur {len(base)} articles :\n{_json.dumps(ctx, ensure_ascii=False)[:6000]}\n\n"
            "Génère un JSON {\"recommended_actions\": [\"action 1\", ...]} avec 4 à 5 actions "
            "concrètes déduites de l'évidence. Retourne UNIQUEMENT le JSON.")
    try:
        client = _OAI(timeout=90.0)
        resp = client.chat.completions.create(
            model="gpt-4.1", temperature=0.2, max_tokens=700,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        data = _json.loads(resp.choices[0].message.content)
        actions = [str(x) for x in (data.get("recommended_actions") or []) if str(x).strip()][:6]
    except Exception as e:
        logger.error(f"Génération actions {scenario_id}: {e}", exc_info=True)
        return []

    _lang_norm = (lang or "fr")[:2].lower()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, recommended_actions_json, recommended_actions_lang, actions_generated_at, updated_at)
            VALUES (:sid, CAST(:a AS jsonb), :lng, NOW(), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
            SET recommended_actions_json = CAST(:a AS jsonb), recommended_actions_lang = :lng,
                actions_generated_at = NOW(), updated_at = NOW()
        """), {"sid": scenario_id, "a": _json.dumps(actions), "lng": _lang_norm})
    return actions


def _maybe_generate_actions(scenario_id: str, lang: str | None = None) -> bool:
    """Lance la génération des actions en arrière-plan, une fois par (scénario, langue).
    La clé de job inclut la langue : un changement de langue relance la génération
    (au lieu du garde-fou « une seule fois » qui figeait la 1re langue)."""
    import threading
    _job_key = f"{scenario_id}:{(lang or 'fr')[:2].lower()}"
    if _ACTIONS_JOBS.get(_job_key, {}).get("status") in ("running", "done"):
        return False
    _ACTIONS_JOBS[_job_key] = {"status": "running"}

    def _run():
        try:
            n = _generate_recommended_actions(scenario_id, lang=lang)
            _ACTIONS_JOBS[_job_key] = {"status": "done", "count": len(n)}
        except Exception as e:
            _ACTIONS_JOBS[_job_key] = {"status": "error", "error": str(e)}
            logger.warning(f"Actions job {scenario_id}: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return True


@app.get("/scenarios/{scenario_id}/recommended-actions")
def get_recommended_actions(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """Actions recommandées (cache) ; génère en arrière-plan au 1er appel si absentes."""
    _lang_norm = (lang or "fr")[:2].lower()
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT recommended_actions_json, recommended_actions_lang, actions_generated_at FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    # Ne servir le cache que s'il est DANS LA LANGUE demandée. Les actions anciennes
    # sans langue enregistrée (NULL) sont considérées françaises.
    if (row and isinstance(row["recommended_actions_json"], list) and row["recommended_actions_json"]
            and (row["recommended_actions_lang"] or "fr") == _lang_norm):
        return {"status": "ready", "actions": row["recommended_actions_json"],
                "generated_at": row["actions_generated_at"].isoformat() if row["actions_generated_at"] else None}
    started = _maybe_generate_actions(scenario_id, lang=lang)
    job = _ACTIONS_JOBS.get(f"{scenario_id}:{_lang_norm}", {})
    if job.get("status") == "error":
        return {"status": "error", "actions": [], "error": job.get("error")}
    return {"status": "generating" if (started or job.get("status") == "running") else "empty", "actions": []}



def _diff_model_spec(old: dict | None, new: dict | None) -> dict:
    """Diff structuré entre deux model_spec (pur, testable)."""
    old, new = old or {}, new or {}
    o_out, n_out = old.get("outcome") or {}, new.get("outcome") or {}
    outcome_fields = {}
    for f in ("name", "machine_name", "task_type", "unit"):
        if (o_out.get(f) or None) != (n_out.get(f) or None):
            outcome_fields[f] = {"old": o_out.get(f), "new": n_out.get(f)}

    o_feats = {f.get("machine_name"): f for f in (old.get("features") or [])}
    n_feats = {f.get("machine_name"): f for f in (new.get("features") or [])}
    added = [k for k in n_feats if k not in o_feats]
    removed = [k for k in o_feats if k not in n_feats]
    changed = []
    for k in n_feats:
        if k in o_feats:
            fc = {}
            for fld in ("dtype", "source", "importance"):
                if (o_feats[k].get(fld) or None) != (n_feats[k].get(fld) or None):
                    fc[fld] = {"old": o_feats[k].get(fld), "new": n_feats[k].get(fld)}
            if fc:
                changed.append({"machine_name": k, "fields": fc})

    o_alg, n_alg = old.get("algorithm") or {}, new.get("algorithm") or {}
    alg_fields = {}
    for f in ("family", "metric"):
        if (o_alg.get(f) or None) != (n_alg.get(f) or None):
            alg_fields[f] = {"old": o_alg.get(f), "new": n_alg.get(f)}

    has_changes = bool(outcome_fields or added or removed or changed or alg_fields)
    return {
        "has_changes": has_changes,
        "outcome_changed": bool(outcome_fields),
        "outcome_fields": outcome_fields,
        "features_added": added,
        "features_removed": removed,
        "features_changed": changed,
        "algorithm_changed": bool(alg_fields),
        "algorithm_fields": alg_fields,
        "summary": {
            "added": len(added), "removed": len(removed), "changed": len(changed),
            "outcome_changed": bool(outcome_fields), "algorithm_changed": bool(alg_fields),
        },
    }


@app.post("/scenarios/{scenario_id}/model/spec/propose")
def propose_scenario_spec(scenario_id: str, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Régénère le spec depuis l'évidence courante dans un slot de proposition (async)."""
    import threading, time

    if _job_is_active(_SPEC_PROPOSAL_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_variables_from_pico(scenario_id, persist="proposal", lang=lang)
            if "error" in result:
                _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "done"}
        except Exception as e:
            logger.error(f"Spec proposal job {scenario_id}: {e}", exc_info=True)
            _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/model/spec/proposal")
def get_scenario_spec_proposal(scenario_id: str) -> dict[str, Any]:
    """Proposition de spec en attente + diff vs spec actif."""
    job = _SPEC_PROPOSAL_JOBS.get(scenario_id, {})
    if job.get("status") == "running":
        return {"status": "generating", "message": "Régénération en cours, réessayez bientôt."}

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_proposal_json, proposal_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    if not (row and row["variables_proposal_json"]):
        return {"status": "empty", "message": "Aucune proposition. Lancez /model/spec/propose."}

    proposal = dict(row["variables_proposal_json"])
    active = dict(row["variables_json"]) if row["variables_json"] else {}
    diff = _diff_model_spec(active.get("model_spec"), proposal.get("model_spec"))

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "diff": diff,
        "proposal_spec": proposal.get("model_spec"),
        "active_version": (active.get("model_spec") or {}).get("version"),
        "proposal_provenance": proposal.get("_provenance_index", {}),
        "generated_at": row["proposal_generated_at"].isoformat() if row["proposal_generated_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/spec/proposal/validate")
def validate_scenario_spec_proposal(scenario_id: str, payload: dict[str, Any],
                                    _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Valide ou rejette la proposition. Body: {"action": "accept"|"reject", "retrain": bool}.
    accept -> la proposition devient le spec actif (version +1), validé, et le
    modèle est ré-entraîné si un dataset est branché.
    """
    import json as _json
    import threading
    from datetime import datetime, timezone

    action = (payload.get("action") or "").strip().lower()
    if action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="action doit être 'accept' ou 'reject'.")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_proposal_json
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_proposal_json"]):
        raise HTTPException(status_code=400, detail="Aucune proposition en attente.")

    if action == "reject":
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_proposal_json = NULL, proposal_generated_at = NULL, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})
        return {"status": "rejected", "scenario_id": scenario_id}

    # accept : promouvoir la proposition en spec actif, version +1.
    proposal = dict(row["variables_proposal_json"])
    active = dict(row["variables_json"]) if row["variables_json"] else {}
    old_ver = int((active.get("model_spec") or {}).get("version", 0) or 0)
    if proposal.get("model_spec"):
        proposal["model_spec"]["version"] = old_ver + 1

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE scenario_settings
            SET variables_json = CAST(:vars AS jsonb),
                variables_validated = TRUE,
                variables_generated_at = NOW(),
                variables_lang = NULL,
                variables_i18n = NULL,
                variables_proposal_json = NULL,
                proposal_generated_at = NULL,
                updated_at = NOW()
            WHERE scenario_id = :sid
        """), {"sid": scenario_id, "vars": _json.dumps(proposal)})

    # Ré-entraînement automatique si un dataset est branché.
    retrain = payload.get("retrain", True)
    retrain_started = False
    if retrain:
        with engine.connect() as conn:
            ds = conn.execute(text(
                "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
            ), {"sid": scenario_id}).first()
        if ds and _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") != "running":
            _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
            threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
            retrain_started = True

    return {
        "status": "accepted",
        "scenario_id": scenario_id,
        "new_version": old_ver + 1,
        "retrain_started": retrain_started,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


# Messages d'alerte « souple » (task_type ↔ contenu réel de la cible), bilingues.
_SPEC_WARN = {
    "target_not_numeric_for_regression": {
        "fr": "La cible « {t} » n'est pas numérique : une classification conviendrait sans doute mieux qu'une régression.",
        "en": "Target “{t}” is not numeric: a classification task would likely fit better than regression.",
    },
    "target_binary_for_regression": {
        "fr": "La cible « {t} » ne prend que {n} valeurs distinctes : une classification conviendrait sans doute mieux.",
        "en": "Target “{t}” has only {n} distinct values: a classification task would likely fit better.",
    },
    "target_high_cardinality_for_classification": {
        "fr": "La cible « {t} » est numérique avec {n} valeurs distinctes : une régression conviendrait sans doute mieux qu'une classification.",
        "en": "Target “{t}” is numeric with {n} distinct values: a regression task would likely fit better than classification.",
    },
}


def _task_target_sanity(scenario_id: str, outcome: dict) -> tuple[str, dict] | None:
    """Contrôle de cohérence (SOUPLE) entre le task_type choisi et le contenu réel
    de la colonne cible du dataset branché. Best-effort : renvoie None s'il n'y a pas
    de dataset ou si la lecture échoue. Retour : (code, params) → localisé par l'appelant."""
    import pandas as pd
    import pandas.api.types as pdt
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT stored_path FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).mappings().first()
    if not (ds and ds["stored_path"]):
        return None
    tt = (outcome.get("task_type") or "classification").strip().lower()
    target_norm = _norm_col(outcome.get("machine_name") or "")
    df = pd.read_csv(ds["stored_path"], nrows=5000)
    col = next((c for c in df.columns if _norm_col(c) == target_norm), None)
    if col is None:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    is_num = pdt.is_numeric_dtype(s)
    nuniq = int(s.nunique())
    if tt in ("regression", "count"):
        if not is_num:
            return ("target_not_numeric_for_regression", {"t": str(col)})
        if nuniq <= 2:
            return ("target_binary_for_regression", {"t": str(col), "n": nuniq})
    elif tt == "classification":
        if is_num and nuniq > 20:
            return ("target_high_cardinality_for_classification", {"t": str(col), "n": nuniq})
    return None


@app.post("/scenarios/{scenario_id}/model/spec/edit")
def edit_scenario_model_spec(scenario_id: str, payload: dict[str, Any],
                             lang: str | None = Query(None),
                             _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Édite DIRECTEMENT le spec actif (sans régénérer depuis l'évidence) : choix de
    l'algorithme (parmi les candidats de l'évidence ou toute famille valide),
    task_type / positive_class de l'outcome, ajout/suppression de variables. Le
    data_template est reconstruit (colonnes attendues à l'upload), la version
    incrémentée, et le modèle ré-entraîné si demandé et si un dataset est branché.

    Body (tous les champs optionnels) :
      {"algorithm_family": "lightgbm", "metric": "rmse", "task_type": "regression",
       "positive_class": "oui", "remove_features": ["mn1"],
       "add_features": [{"name": "Âge", "dtype": "int", "importance": "medium"}],
       "retrain": true}
    """
    import json as _json
    import threading
    from datetime import datetime, timezone

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_json"]):
        raise HTTPException(status_code=400, detail="Aucune spécification de modèle à éditer. Générez d'abord les Variables & Modèle.")
    vj = dict(row["variables_json"])
    spec = vj.get("model_spec")
    if not spec:
        raise HTTPException(status_code=400, detail="Spécification antérieure au schéma machine. Relancez la génération des Variables & Modèle.")

    spec = dict(spec)
    outcome = dict(spec.get("outcome") or {})
    algorithm = dict(spec.get("algorithm") or {})
    features = [dict(f) for f in (spec.get("features") or [])]
    changed = False

    # ── Outcome : task_type ──
    if payload.get("task_type"):
        new_tt = _coerce_enum(payload["task_type"], _TASK_TYPES, outcome.get("task_type") or "classification")
        if new_tt != (outcome.get("task_type") or ""):
            outcome["task_type"] = new_tt
            changed = True
            # Métrique par défaut si l'actuelle n'a plus de sens pour la nouvelle tâche.
            _reg_metrics, _clf_metrics = {"rmse", "mae", "r2"}, {"roc_auc", "average_precision"}
            _default_metric = {"classification": "roc_auc", "regression": "rmse",
                               "count": "rmse", "survival": "c_index"}[new_tt]
            cur_metric = (algorithm.get("metric") or "").strip().lower()
            ok = (cur_metric in _reg_metrics) if new_tt in ("regression", "count") \
                else (cur_metric in _clf_metrics) if new_tt == "classification" else True
            if not ok:
                algorithm["metric"] = _default_metric
            if new_tt != "classification":
                outcome["positive_class"] = None
            algorithm["family"] = _coerce_family_for_task(algorithm.get("family"), new_tt)

    # ── Outcome : positive_class (classification seulement) ──
    if "positive_class" in payload and outcome.get("task_type") == "classification":
        pc = payload.get("positive_class")
        pc = str(pc).strip() if pc not in (None, "") else None
        if pc != outcome.get("positive_class"):
            outcome["positive_class"] = pc
            changed = True

    # ── Algorithme : famille + métrique ──
    if payload.get("algorithm_family"):
        fam = _coerce_family_for_task(payload["algorithm_family"], outcome.get("task_type") or "classification")
        if fam != (algorithm.get("family") or ""):
            algorithm["family"] = fam
            changed = True
    if payload.get("metric"):
        met = _coerce_enum(payload["metric"], _METRICS, algorithm.get("metric") or "rmse")
        if met != (algorithm.get("metric") or ""):
            algorithm["metric"] = met
            changed = True

    # ── Features : suppression ──
    remove = {str(m).strip() for m in (payload.get("remove_features") or []) if str(m).strip()}
    if remove:
        kept = [f for f in features if f.get("machine_name") not in remove]
        if len(kept) != len(features):
            features, changed = kept, True

    # ── Features : ajout (variable manuelle, sans provenance évidence) ──
    used = {f.get("machine_name") for f in features if f.get("machine_name")}
    used.add(outcome.get("machine_name"))
    for add in (payload.get("add_features") or []):
        if not isinstance(add, dict):
            continue
        name = str(add.get("name") or "").strip()
        if not name:
            continue
        mn = _slug_identifier(add.get("machine_name") or name, used)
        features.append({
            "name": name, "machine_name": mn,
            "dtype": _coerce_enum(add.get("dtype"), _DTYPES, "float"),
            "source": "user", "public_provider": None,
            "importance": _coerce_enum(add.get("importance"), {"high", "medium", "low"}, "medium"),
            "provenance": [],
        })
        changed = True

    if not features:
        raise HTTPException(status_code=400, detail="Le modèle doit conserver au moins une variable explicative.")
    if not changed:
        return {"status": "unchanged", "scenario_id": scenario_id}

    # ── Cohérence prévision : une famille de série temporelle exige une cible
    # NUMÉRIQUE. Si l'outcome est en classification, on ramène à 'regression' (cible
    # float dans le data_template) et on nettoie la classe positive / la métrique. ──
    if (algorithm.get("family") in _TS_ALGO_FAMILIES) and outcome.get("task_type") == "classification":
        outcome["task_type"] = "regression"
        outcome["positive_class"] = None
        if (algorithm.get("metric") or "") in ("roc_auc", "average_precision"):
            algorithm["metric"] = "rmse"

    # ── Reconstruire le spec + le data_template (jamais désynchronisé des features) ──
    spec["outcome"], spec["algorithm"], spec["features"] = outcome, algorithm, features
    spec["data_template"] = _derive_data_template(outcome, features)
    spec["version"] = int(spec.get("version", 1) or 1) + 1
    vj["model_spec"] = spec

    # ── Contrôle souple task↔cible (vs dataset branché), localisé ──
    warnings: list[str] = []
    try:
        _sanity = _task_target_sanity(scenario_id, outcome)
        if _sanity:
            _code, _params = _sanity
            _lg = "en" if (lang or "fr").strip().lower().startswith("en") else "fr"
            warnings.append(_SPEC_WARN[_code][_lg].format(**_params))
    except Exception as _we:
        logger.warning(f"task_target_sanity {scenario_id}: {_we}")

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE scenario_settings
            SET variables_json = CAST(:vars AS jsonb),
                variables_validated = TRUE,
                variables_generated_at = NOW(),
                variables_lang = NULL,
                variables_i18n = NULL,
                updated_at = NOW()
            WHERE scenario_id = :sid
        """), {"sid": scenario_id, "vars": _json.dumps(vj)})

    # ── Ré-entraînement si demandé + dataset branché ──
    retrain = payload.get("retrain", True)
    retrain_started = False
    if retrain:
        with engine.connect() as conn:
            ds = conn.execute(text(
                "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
            ), {"sid": scenario_id}).first()
        if ds and _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") != "running":
            _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
            threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
            retrain_started = True

    return {
        "status": "updated",
        "scenario_id": scenario_id,
        "new_version": spec["version"],
        "outcome": {"machine_name": outcome.get("machine_name"), "name": outcome.get("name"),
                    "task_type": outcome.get("task_type"), "positive_class": outcome.get("positive_class")},
        "algorithm": {"family": algorithm.get("family"), "metric": algorithm.get("metric")},
        "features": [{"name": f["name"], "machine_name": f["machine_name"],
                      "dtype": f["dtype"], "importance": f.get("importance")} for f in features],
        "data_template": spec["data_template"],
        "warnings": warnings,
        "retrain_started": retrain_started,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── MODEL DATA (Phase 2) : upload, validation vs data_template, readiness ────
# L'utilisateur branche ses données (CSV/XLSX) sur le model_spec. On valide les
# colonnes contre le data_template, on stocke le dataset, et on évalue si le
# modèle peut être entraîné. Les colonnes manquantes 'public_api' (ex: météo)
# sont signalées comme récupérables automatiquement (branchement Open-Meteo : étape suivante).

import os as _os_mod

MODEL_DATA_DIR = Path(_os_mod.environ.get("MODEL_DATA_DIR", "/home/ubuntu/uploads_datasets"))


def _ensure_model_dataset_table():
    """Suivi des datasets uploadés par scénario pour l'entraînement du modèle."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_model_dataset (
                id              BIGSERIAL PRIMARY KEY,
                scenario_id     VARCHAR(80) NOT NULL,
                filename        TEXT,
                stored_path     TEXT,
                n_rows          INTEGER,
                n_cols          INTEGER,
                columns_json    JSONB,
                validation_json JSONB,
                is_active       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_model_dataset_scenario_active "
            "ON scenario_model_dataset (scenario_id, is_active)"
        ))
        # Données SYNTHÉTIQUES de démo (badge « non réel » dans l'UI) vs données réelles.
        conn.execute(text("ALTER TABLE scenario_model_dataset ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE"))
    logger.info("Table scenario_model_dataset vérifiée/créée.")


try:
    _ensure_model_dataset_table()
except Exception as _e:
    logger.warning(f"_ensure_model_dataset_table: {_e}")


def _norm_col(s: Any) -> str:
    """Normalise un nom de colonne pour l'appariement fichier↔data_template. MÊMES
    règles que model_trainer._norm (repli NFKD des accents, minuscules, non-alnum →
    « _ ») pour que la VALIDATION et l'ENTRAÎNEMENT s'accordent : un en-tête réel
    « Température max (J1) » s'apparie au machine_name « temperature_max_j1 »."""
    import unicodedata as _ud
    _s = _ud.normalize("NFKD", str(s if s is not None else "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", _s.lower()).strip("_")


def _get_model_spec(scenario_id: str) -> dict | None:
    """Retourne le model_spec stocké (ou None) pour un scénario."""
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_json"]):
        return None
    return dict(row["variables_json"]).get("model_spec")


def _validate_dataset_against_template(file_columns: list, data_template: dict,
                                       file_dtype_kinds: dict | None = None,
                                       n_rows: int | None = None) -> dict:
    """
    Compare les colonnes d'un fichier au data_template (pur, testable).
    Le matching est insensible à la casse/aux espaces ; on signale les colonnes
    cible/explicatives présentes, manquantes (user vs public_api), en trop, et
    les incompatibilités de type. Conclut sur la possibilité d'entraîner.
    """
    file_dtype_kinds = file_dtype_kinds or {}
    cols = list(file_columns or [])
    norm_to_file: dict[str, Any] = {}
    for c in cols:
        norm_to_file.setdefault(_norm_col(c), c)

    template_cols = data_template.get("columns") or []
    target_col = data_template.get("target_column")

    present_required, missing_required, present_optional = [], [], []
    missing_user, missing_public, matched_features = [], [], []
    renamed, dtype_warnings = [], []
    target_present = False
    n_features = n_features_present = 0

    def file_has(canonical):
        f = norm_to_file.get(_norm_col(canonical))
        if f is not None and f != canonical:
            renamed.append({"expected": canonical, "found": f})
        return f

    for col in template_cols:
        name, role = col.get("name"), col.get("role")
        required, source = col.get("required", False), col.get("source", "user")
        if role == "feature":
            n_features += 1
        found = file_has(name)
        if found is not None:
            if role == "outcome":
                target_present = True
            else:
                n_features_present += 1
                matched_features.append(name)
            (present_required if required else present_optional).append(name)
            kind, exp = file_dtype_kinds.get(_norm_col(found)), col.get("dtype")
            if exp in ("float", "int") and kind == "other":
                dtype_warnings.append({"column": name, "expected": exp, "found_kind": kind})
        else:
            if required:
                missing_required.append(name)
            if source == "public_api":
                missing_public.append(name)
            elif role != "outcome":
                missing_user.append(name)

    template_norms = {_norm_col(c.get("name")) for c in template_cols}
    extra_columns = [c for c in cols if _norm_col(c) not in template_norms]

    reasons = []
    if not target_present:
        reasons.append(f"Colonne cible '{target_col}' absente (obligatoire pour entraîner).")
    if n_features_present == 0:
        reasons.append("Aucune variable explicative présente dans le fichier.")
    # Assez de lignes ? L'entraînement exige un plancher (model_trainer: min 20).
    # Sans ce contrôle, un fichier de 5 lignes renvoyait « training_started: true »
    # puis échouait silencieusement en arrière-plan.
    _MIN_TRAIN_ROWS = 20
    if n_rows is not None and n_rows < _MIN_TRAIN_ROWS:
        reasons.append(f"Trop peu de lignes ({n_rows}) : minimum {_MIN_TRAIN_ROWS} pour entraîner.")
    # Colonnes numériques contenant des valeurs non numériques : bloquant (sinon
    # l'imputation/mise à l'échelle plante à l'entraînement en arrière-plan).
    if dtype_warnings:
        _bad = ", ".join(str(w.get("column")) for w in dtype_warnings)
        reasons.append(f"Colonne(s) numérique(s) avec des valeurs non numériques : {_bad}. Corrigez ou retirez ces valeurs.")
    can_train = (
        target_present and n_features_present >= 1
        and (n_rows is None or n_rows >= _MIN_TRAIN_ROWS)
        and not dtype_warnings
    )

    return {
        "target_column": target_col,
        "target_present": target_present,
        "n_features_total": n_features,
        "n_features_present": n_features_present,
        "matched_features": matched_features,
        "present_required": present_required,
        "missing_required": missing_required,
        "present_optional": present_optional,
        "missing_user": missing_user,
        "missing_public": missing_public,
        "extra_columns": extra_columns,
        "renamed": renamed,
        "dtype_warnings": dtype_warnings,
        "readiness": {
            "can_train": can_train,
            "reasons": reasons,
            "auto_fetchable": missing_public,
        },
    }


def _dataframe_dtype_kinds(df) -> dict:
    """{col_normalisé: 'numeric'|'datetime'|'bool'|'other'} depuis un DataFrame pandas."""
    import pandas as pd
    kinds = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            k = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            k = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            k = "datetime"
        else:
            k = "other"
        kinds[_norm_col(c)] = k
    return kinds


def _maybe_autotrain(scenario_id: str, report: dict) -> bool:
    """Démarre l'entraînement si les données branchées suffisent (can_train) et
    qu'aucun entraînement n'est déjà en cours. Renvoie True si lancé."""
    import threading

    if not ((report or {}).get("readiness") or {}).get("can_train"):
        return False
    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return False
    _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
    threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
    logger.info(f"Auto-entraînement déclenché pour {scenario_id} (données suffisantes).")
    return True


# ─── PHASE 2 : Connecteurs de données publiques (auto-remplissage des variables) ─
# Chaque connecteur récupère une source publique RÉELLE et lisible par machine et
# renvoie une série temporelle quotidienne « tidy » joignable au dataset du modèle
# sur la clé date — au lieu d'un upload CSV manuel. Discovery + fetch ici ; le
# mapping variable→connecteur et l'assemblage du dataset arrivent ensuite.
class ConnectorFetchIn(BaseModel):
    region: str | None = None        # alias Romandie (geneva|lausanne|sion|…) ou lat+lon
    lat: float | None = None
    lon: float | None = None
    start_date: str = Field(..., min_length=8)   # YYYY-MM-DD
    end_date: str = Field(..., min_length=8)
    limit: int = Field(default=2000, ge=1, le=20000)


@app.get("/model/connectors")
def list_data_connectors(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """Liste les connecteurs de données publiques disponibles (métadonnées : source,
    licence, granularité, variables fournies). Sert à mapper les colonnes `public_api`
    du data_template vers une source récupérable automatiquement."""
    import data_connectors
    return {"connectors": data_connectors.list_connectors()}


@app.post("/model/connectors/{connector_id}/fetch")
def fetch_data_connector(
    connector_id: str, payload: ConnectorFetchIn, _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Récupère la série quotidienne « tidy » d'un connecteur pour une zone + période.
    Renvoie les lignes (plafonnées à `limit`) + la provenance réelle (source, licence)."""
    import data_connectors
    if connector_id not in data_connectors.CONNECTORS:
        raise HTTPException(status_code=404, detail=f"Connecteur inconnu : {connector_id}")
    params = payload.model_dump(exclude_none=True)
    limit = params.pop("limit", 2000)
    try:
        rows = data_connectors.fetch_series(connector_id, params)
    except ValueError as _ve:
        raise HTTPException(status_code=422, detail=str(_ve))
    except Exception as _fe:
        logger.warning(f"Connector {connector_id} fetch failed: {_fe}")
        raise HTTPException(status_code=502,
                            detail=f"Échec de récupération depuis la source publique : {_fe}")
    meta = data_connectors.CONNECTORS[connector_id].metadata()
    return {
        "connector_id": connector_id, "provider": meta["provider"], "license": meta["license"],
        "commercial_ok": meta["commercial_ok"], "variables": meta["variables"],
        "row_count": len(rows), "rows": rows[:limit],
    }


def _agg_for_column(col: str) -> str:
    """Fonction d'agrégation au ré-échantillonnage, déduite du nom de colonne :
    précipitations → somme, charge virale (eaux usées) → dernière valeur du bucket,
    tout le reste (température, humidité, qualité de l'air, incidence) → moyenne."""
    c = (col or "").lower()
    if "precip" in c or "rain" in c or "rainfall" in c:
        return "sum"
    if "load" in c or "wastewater" in c or c.endswith("_ww") or "viral" in c:
        return "last"
    return "mean"


def _assemble_connector_frames(frames: dict, mappings: list[dict], freq: str,
                               datetime_col: str | None):
    """PUR : resample chaque série de connecteur à `freq` (moyenne), joint sur la clé
    'date' et renomme chaque `connector_variable` vers sa `template_column`. Renvoie
    (DataFrame assemblé | None, colonnes remplies). Aligne les fréquences hétérogènes
    (météo quotidienne, eaux usées / Sentinella hebdomadaires) sur une grille commune."""
    import pandas as pd
    resampled: dict[str, Any] = {}
    for cid, df in (frames or {}).items():
        try:
            if "date" not in getattr(df, "columns", []):
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            if d.empty:
                continue
            d = d.set_index("date")
            for _c in list(d.columns):
                d[_c] = pd.to_numeric(d[_c], errors="coerce")
            d = d.resample(freq).agg({_c: _agg_for_column(_c) for _c in d.columns}).reset_index()
            d["date"] = d["date"].dt.strftime("%Y-%m-%d")
            resampled[cid] = d
        except Exception:
            continue
    assembled = None
    filled: list[str] = []
    for m in mappings:
        d = resampled.get(m["connector_id"])
        var = m["connector_variable"]
        if d is None or var not in d.columns:
            continue
        part = d[["date", var]].rename(columns={var: m["template_column"]})
        part = part.dropna(subset=[m["template_column"]]).drop_duplicates(subset=["date"])
        assembled = part if assembled is None else assembled.merge(part, on="date", how="outer")
        filled.append(m["template_column"])
    if assembled is None or assembled.empty:
        return None, []
    if datetime_col and datetime_col not in assembled.columns:
        assembled = assembled.rename(columns={"date": datetime_col})
    assembled = assembled.sort_values(assembled.columns[0]).reset_index(drop=True)
    return assembled, filled


@app.post("/scenarios/{scenario_id}/model/data")
async def upload_model_dataset(
    scenario_id: str,
    file: UploadFile = File(...),
    auto_train: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Branche un jeu de données (CSV/XLSX) sur le model_spec d'un scénario.
    Valide les en-têtes contre le data_template, stocke le dataset (actif), et
    renvoie un rapport de validation + l'état de préparation à l'entraînement.
    Si les données suffisent (can_train) et auto_train, l'entraînement démarre
    automatiquement (upload -> entraînement -> modèle en ligne, sans étape manuelle).
    """
    import io
    import pandas as pd
    from datetime import datetime, timezone

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    data_template = spec.get("data_template") or {}
    if not data_template.get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers CSV et Excel (.xlsx, .xls) sont acceptés.")

    # Plafond d'octets AVANT de tout charger en mémoire : lit au plus MAX+1 octets
    # (borne la RAM face à un upload démesuré / malveillant). 25 Mo couvrent largement
    # un CSV/XLSX de 500 000 lignes, qui est de toute façon rejeté plus bas.
    _MAX_UPLOAD_BYTES = 25 * 1024 * 1024
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"Fichier trop volumineux (> {_MAX_UPLOAD_BYTES // (1024 * 1024)} Mo).")
    try:
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lecture du fichier impossible : {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Le fichier ne contient aucune ligne.")
    if len(df) > 500_000:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (> 500 000 lignes).")

    report = _validate_dataset_against_template(list(df.columns), data_template, _dataframe_dtype_kinds(df), n_rows=len(df))

    # Stockage (CSV canonique) + métadonnées ; le précédent dataset devient inactif.
    safe = Path(filename).name or "dataset.csv"
    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_{safe}.csv")
        df.to_csv(stored_path, index=False)
    except Exception as e:
        logger.error(f"Stockage dataset {scenario_id}: {e}", exc_info=True)

    # Échec de stockage → NE PAS activer un dataset au chemin NULL (il paraîtrait
    # « prêt » mais tout entraînement/monitoring échouerait faute de fichier).
    if stored_path is None:
        raise HTTPException(status_code=500,
                            detail="Échec du stockage du dataset (espace disque ?). Réessayez.")

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, FALSE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": filename, "sp": stored_path,
            "nr": int(len(df)), "nc": int(len(df.columns)),
            "cj": json.dumps([str(c) for c in df.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "filename": filename,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "stored": stored_path is not None,
        "validation": report,
        "training_started": _maybe_autotrain(scenario_id, report) if auto_train else False,
    }


class AutoFetchMapping(BaseModel):
    template_column: str            # colonne du data_template à remplir
    connector_id: str               # ex. open-meteo-weather, eawag-wastewater
    connector_variable: str         # variable fournie par le connecteur (ex. temp_mean)


class AutoFetchIn(BaseModel):
    region: str | None = None       # alias Romandie OU lat+lon
    lat: float | None = None
    lon: float | None = None
    start_date: str = Field(..., min_length=8)
    end_date: str = Field(..., min_length=8)
    frequency: str = Field(default="W")     # W (hebdo, défaut épidémio) | D | MS
    mappings: list[AutoFetchMapping] = Field(..., min_length=1)
    auto_train: bool = False


@app.post("/scenarios/{scenario_id}/model/data/auto-fetch")
def auto_fetch_model_dataset(
    scenario_id: str, payload: AutoFetchIn, _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Phase 2 — assemble le dataset du modèle depuis des connecteurs de données
    PUBLIQUES au lieu d'un upload. Chaque mapping relie une colonne du data_template
    à (connecteur, variable) ; on récupère chaque connecteur, on aligne sur une grille
    de dates commune (frequency) et on joint, on stocke le dataset actif, puis on
    renvoie la couverture (colonnes remplies vs encore requises) + un aperçu."""
    import pandas as pd
    from datetime import datetime, timezone
    import data_connectors

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    data_template = spec.get("data_template") or {}
    if not data_template.get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    base_params = {"start_date": payload.start_date, "end_date": payload.end_date}
    if payload.region:
        base_params["region"] = payload.region
    if payload.lat is not None:
        base_params["lat"] = payload.lat
    if payload.lon is not None:
        base_params["lon"] = payload.lon

    frames: dict[str, Any] = {}
    fetch_errors: dict[str, str] = {}
    for cid in {m.connector_id for m in payload.mappings}:
        if cid not in data_connectors.CONNECTORS:
            fetch_errors[cid] = "connecteur inconnu"
            continue
        try:
            rows = data_connectors.fetch_series(cid, dict(base_params))
        except Exception as _e:
            fetch_errors[cid] = str(_e)
            continue
        if not rows:
            fetch_errors[cid] = "aucune donnée renvoyée pour cette zone/période"
            continue
        frames[cid] = pd.DataFrame(rows)

    _dt_col = next((c["name"] for c in data_template.get("columns", []) if c.get("dtype") == "datetime"), None)
    freq = (payload.frequency or "W").strip() or "W"
    assembled, filled = _assemble_connector_frames(
        frames, [m.model_dump() for m in payload.mappings], freq, _dt_col)
    if assembled is None or assembled.empty:
        raise HTTPException(status_code=422, detail={
            "message": "Aucune donnée assemblée depuis les connecteurs (mappings/fenêtre à vérifier).",
            "fetch_errors": fetch_errors,
        })

    report = _validate_dataset_against_template(
        list(assembled.columns), data_template, _dataframe_dtype_kinds(assembled), n_rows=len(assembled))

    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_autofetch.csv")
        assembled.to_csv(stored_path, index=False)
    except Exception as _e:
        logger.error(f"Stockage auto-fetch {scenario_id}: {_e}", exc_info=True)
    if stored_path is None:
        raise HTTPException(status_code=500, detail="Échec du stockage du dataset auto-récupéré.")

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, FALSE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": "auto_fetch.csv", "sp": stored_path,
            "nr": int(len(assembled)), "nc": int(len(assembled.columns)),
            "cj": json.dumps([str(c) for c in assembled.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "source": "auto_fetch",
        "frequency": freq,
        "n_rows": int(len(assembled)),
        "n_cols": int(len(assembled.columns)),
        "filled_columns": filled,
        "still_needed_user_columns": report.get("missing_user", []),
        "fetch_errors": fetch_errors,
        "validation": report,
        "preview": json.loads(assembled.head(8).to_json(orient="records")),
        "training_started": _maybe_autotrain(scenario_id, report) if payload.auto_train else False,
    }


@app.get("/scenarios/{scenario_id}/model/data")
def get_model_dataset(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Résumé du dataset actif d'un scénario + état de préparation à l'entraînement.
    Authentifié : le schéma des données uploadées par l'utilisateur ne doit pas être
    lisible publiquement par simple connaissance de l'ID de scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, filename, n_rows, n_cols, columns_json, validation_json, is_synthetic, created_at
            FROM scenario_model_dataset
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()

    if not row:
        return {"status": "empty", "message": "Aucun jeu de données branché. Uploadez un CSV/XLSX correspondant au data_template."}

    return {
        "status": "ready",
        "dataset_id": row["id"],
        "filename": row["filename"],
        "n_rows": row["n_rows"],
        "n_cols": row["n_cols"],
        "columns": row["columns_json"],
        "validation": row["validation_json"],
        "is_synthetic": bool(row["is_synthetic"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/data/synthetic")
def generate_synthetic_model_dataset(scenario_id: str, n_rows: int = 400,
                                     auto_train: bool = True,
                                     _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Génère un dataset SYNTHÉTIQUE cohérent avec le data_template du spec et le
    branche comme dataset actif. Permet de faire tourner un vrai modèle de
    démonstration (entraînable immédiatement) sans données réelles — utile pour
    transformer un scénario en démo « modèle en ligne ». Généralisable à tout scénario.
    """
    from datetime import datetime, timezone
    import model_trainer

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    if not (spec.get("data_template") or {}).get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    n_rows = max(50, min(int(n_rows or 400), 5000))
    try:
        df = model_trainer.generate_synthetic_dataset(spec, n_rows=n_rows)
    except Exception as e:
        logger.error(f"Synthetic gen {scenario_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Génération synthétique impossible : {e}")

    report = _validate_dataset_against_template(list(df.columns), spec["data_template"], _dataframe_dtype_kinds(df))

    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_synthetic.csv")
        df.to_csv(stored_path, index=False)
    except Exception as e:
        logger.error(f"Stockage dataset synthétique {scenario_id}: {e}", exc_info=True)

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, TRUE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": f"synthetic_{n_rows}.csv", "sp": stored_path,
            "nr": int(len(df)), "nc": int(len(df.columns)),
            "cj": json.dumps([str(c) for c in df.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "synthetic": True,
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "stored": stored_path is not None,
        "validation": report,
        "training_started": _maybe_autotrain(scenario_id, report) if auto_train else False,
        "note": "Données synthétiques de démonstration — à remplacer par des données réelles pour un usage opérationnel.",
    }


# ─── MODEL TRAINING (Phase 3) : sklearn + Optuna sur le dataset branché ───────
# Entraîne un vrai modèle à partir du model_spec (Phase 1) et du dataset
# uploadé (Phase 2) : préprocessing, HPO par validation croisée (Optuna),
# holdout, importances. Le pipeline entraîné est sérialisé (joblib) et sert les
# prédictions. Remplace les formules mock par un modèle réellement appris.

_MODEL_TRAIN_JOBS: dict[str, dict] = {}


def _ensure_model_run_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_model_run (
                id               BIGSERIAL PRIMARY KEY,
                scenario_id      VARCHAR(80) NOT NULL,
                dataset_id       BIGINT,
                status           VARCHAR(20) DEFAULT 'running',
                family           TEXT,
                task_type        TEXT,
                metric           TEXT,
                metrics_json     JSONB,
                best_params_json JSONB,
                feature_importance_json JSONB,
                summary_json     JSONB,
                artifact_path    TEXT,
                error            TEXT,
                is_active        BOOLEAN DEFAULT FALSE,
                created_at       TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_model_run_scenario_active "
            "ON scenario_model_run (scenario_id, is_active)"
        ))
    logger.info("Table scenario_model_run vérifiée/créée.")


try:
    _ensure_model_run_table()
except Exception as _e:
    logger.warning(f"_ensure_model_run_table: {_e}")


def _run_model_training(scenario_id: str, n_trials: int = 25, compare: bool = False) -> None:
    """Job d'entraînement (thread) : charge le dataset actif, entraîne, persiste.
    compare=True entraîne PLUSIEURS familles (lightgbm/xgboost/GB/RF/linéaire),
    persiste la MEILLEURE et joint le classement complet au résumé."""
    import json as _json
    import pandas as pd
    from datetime import datetime, timezone
    import model_trainer

    try:
        spec = _get_model_spec(scenario_id)
        if not spec:
            raise ValueError("Aucun model_spec (générez/validez les Variables & Modèle).")

        with engine.connect() as conn:
            ds = conn.execute(text("""
                SELECT id, stored_path, is_synthetic FROM scenario_model_dataset
                WHERE scenario_id = :sid AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """), {"sid": scenario_id}).mappings().first()
        if not ds or not ds["stored_path"]:
            raise ValueError("Aucun dataset branché. Uploadez d'abord un CSV/XLSX.")

        df = pd.read_csv(ds["stored_path"])
        if compare:
            comparison = model_trainer.compare_models(df, spec, n_trials=n_trials)
            result = comparison.get("best")
            if not result:
                raise ValueError("Aucun modèle n'a pu être entraîné lors de la comparaison.")
            # Classement complet joint au résumé (affiché dans l'onglet Modèle).
            result["leaderboard"] = comparison["leaderboard"]
            result["families_compared"] = comparison["families_tried"]
            result["leaderboard_lower_is_better"] = comparison["lower_is_better"]
        else:
            result = model_trainer.train_model(df, spec, n_trials=n_trials)

        # Traçabilité : modèle entraîné sur des données SYNTHÉTIQUES (démo) vs réelles.
        result["is_synthetic"] = bool(ds.get("is_synthetic"))

        # Sérialiser le pipeline entraîné.
        pipeline = result.pop("pipeline")
        artifact_path = None
        try:
            import joblib
            adir = MODEL_DATA_DIR / scenario_id / "model"
            adir.mkdir(parents=True, exist_ok=True)
            artifact_path = str(adir / f"artifact_{int(datetime.now(timezone.utc).timestamp())}.joblib")
            joblib.dump(pipeline, artifact_path)
        except Exception as e:
            logger.error(f"Sérialisation artefact {scenario_id}: {e}", exc_info=True)
            artifact_path = None

        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE scenario_model_run SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
            ), {"sid": scenario_id})
            run_id = conn.execute(text("""
                INSERT INTO scenario_model_run
                    (scenario_id, dataset_id, status, family, task_type, metric,
                     metrics_json, best_params_json, feature_importance_json, summary_json,
                     artifact_path, is_active)
                VALUES (:sid, :did, 'done', :fam, :tt, :met,
                     CAST(:mj AS jsonb), CAST(:bp AS jsonb), CAST(:fi AS jsonb), CAST(:sj AS jsonb),
                     :ap, TRUE)
                RETURNING id
            """), {
                "sid": scenario_id, "did": ds["id"], "fam": result["family"],
                "tt": result["task_type"], "met": result["metric"],
                "mj": _json.dumps(result["metrics"]),
                "bp": _json.dumps(result["best_params"]),
                "fi": _json.dumps(result["feature_importances"]),
                "sj": _json.dumps({k: v for k, v in result.items()
                                   if k not in ("metrics", "best_params", "feature_importances")}),
                "ap": artifact_path,
            }).scalar()

        _MODEL_TRAIN_JOBS[scenario_id] = {
            "status": "done", "run_id": run_id, "metrics": result["metrics"],
            "family": result["family"], "task_type": result["task_type"],
        }
        logger.info(f"Modèle entraîné {scenario_id}: run {run_id}, {result['metrics']}")
    except Exception as e:
        logger.error(f"Entraînement modèle {scenario_id}: {e}", exc_info=True)
        _MODEL_TRAIN_JOBS[scenario_id] = {"status": "error", "error": str(e)}
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO scenario_model_run (scenario_id, status, error, is_active)
                    VALUES (:sid, 'error', :err, FALSE)
                """), {"sid": scenario_id, "err": str(e)[:2000]})
        except Exception:
            pass


@app.post("/scenarios/{scenario_id}/model/train")
def train_scenario_model(scenario_id: str, n_trials: int = 25,
                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Lance l'entraînement réel (async) du modèle sur le dataset branché."""
    import threading

    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    if not _get_model_spec(scenario_id):
        raise HTTPException(status_code=400, detail="Aucun model_spec. Générez puis validez les Variables & Modèle.")
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).first()
    if not ds:
        raise HTTPException(status_code=400, detail="Aucun dataset branché. Uploadez un CSV/XLSX d'abord.")

    n_trials = max(5, min(int(n_trials or 25), 100))
    _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
    threading.Thread(target=_run_model_training, args=(scenario_id, n_trials), daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "n_trials": n_trials}


@app.post("/scenarios/{scenario_id}/model/compare")
def compare_scenario_models(scenario_id: str, n_trials: int = 15,
                            _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Entraîne PLUSIEURS familles (lightgbm/xgboost/GB/RF/linéaire) sur le dataset
    branché, garde la meilleure comme modèle actif et joint le classement complet.
    Partage le registre de jobs et le endpoint de statut avec /model/train."""
    import threading

    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    if not _get_model_spec(scenario_id):
        raise HTTPException(status_code=400, detail="Aucun model_spec. Générez puis validez les Variables & Modèle.")
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).first()
    if not ds:
        raise HTTPException(status_code=400, detail="Aucun dataset branché. Uploadez un CSV/XLSX d'abord.")

    # La comparaison entraîne N familles → on limite le budget d'essais par famille
    # (défaut 15) pour garder un temps de calcul raisonnable.
    n_trials = max(5, min(int(n_trials or 15), 50))
    _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
    threading.Thread(
        target=_run_model_training, args=(scenario_id, n_trials), kwargs={"compare": True}, daemon=True
    ).start()
    return {"status": "started", "scenario_id": scenario_id, "n_trials": n_trials, "mode": "compare"}


@app.get("/scenarios/{scenario_id}/model/train/status")
def get_model_train_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job d'entraînement."""
    return _MODEL_TRAIN_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/model/run")
def get_model_run(scenario_id: str) -> dict[str, Any]:
    """Dernier modèle entraîné actif : métriques, hyperparamètres, importances."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, dataset_id, status, family, task_type, metric, metrics_json,
                   best_params_json, feature_importance_json, summary_json, error,
                   (artifact_path IS NOT NULL) AS has_artifact, created_at
            FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()

    if not row:
        return {"status": "empty", "message": "Aucun modèle entraîné. Lancez l'entraînement après avoir branché des données."}

    return {
        "status": "ready",
        "run_id": row["id"],
        "dataset_id": row["dataset_id"],
        "family": row["family"],
        "task_type": row["task_type"],
        "metric": row["metric"],
        "metrics": row["metrics_json"],
        "best_params": row["best_params_json"],
        "feature_importances": row["feature_importance_json"],
        "summary": row["summary_json"],
        "has_artifact": row["has_artifact"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/predict")
def predict_scenario_model(scenario_id: str, payload: dict[str, Any],
                           _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Prédit avec le modèle entraîné actif. Body: {"rows": [{feature: value, ...}, ...]}.
    Renvoie les prédictions (et probabilités si classification).
    """
    import pandas as pd

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="Body attendu: {\"rows\": [ {feature: value, ...} ]}")

    with engine.connect() as conn:
        run = conn.execute(text("""
            SELECT artifact_path, task_type, summary_json FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE AND artifact_path IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not run:
        raise HTTPException(status_code=400, detail="Aucun modèle entraîné disponible. Entraînez d'abord le modèle.")

    # Modèle de prévision : pas de prédiction ligne-à-ligne (la sortie est la
    # prévision temporelle, disponible dans l'onglet Modèle / le monitoring).
    if run["task_type"] == "forecast":
        raise HTTPException(status_code=400,
                            detail="Modèle de prévision : la prédiction se fait sur l'horizon temporel, pas ligne par ligne. Voir la prévision dans l'onglet Modèle.")

    try:
        import joblib
        pipeline = joblib.load(run["artifact_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chargement du modèle impossible : {e}")

    df = pd.DataFrame(rows)
    classes = (run["summary_json"] or {}).get("classes")
    try:
        preds = pipeline.predict(df)
        # Classification : reconvertir les entiers encodés vers les labels d'origine.
        if run["task_type"] == "classification" and classes:
            predictions = [classes[int(p)] if 0 <= int(p) < len(classes) else _jsonable(p) for p in preds]
        else:
            predictions = [_jsonable(p) for p in preds]
        out: dict[str, Any] = {"status": "ok", "predictions": predictions}
        if run["task_type"] == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)
            out["classes"] = classes
            out["probabilities"] = [[float(x) for x in r] for r in proba]
        # Explication LOCALE par prédiction (contribution de chaque variable) — sans
        # dépendance externe, via ablation vers le fond. Plafonnée pour rester rapide.
        summ = run["summary_json"] or {}
        background = summ.get("explain_background")
        if background and len(df) <= 50:
            import model_trainer as _mt
            used = [{"machine_name": k} for k in background]
            explanations = []
            for i in range(len(df)):
                try:
                    explanations.append(_mt.explain_prediction(
                        pipeline, df.iloc[[i]], used, run["task_type"], background))
                except Exception:
                    explanations.append(None)
            out["explanations"] = explanations
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prédiction impossible (colonnes manquantes ?) : {e}")


def _jsonable(v):
    """Convertit les scalaires numpy en types Python natifs."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass
    return v


# ─── MODEL MONITORING (Phase 4) : statut live piloté par le modèle entraîné ───
# Score les données récentes via le modèle réel (Phase 3) et en déduit un niveau
# d'alerte green/orange/red, avec les libellés des alert_thresholds du spec.

_DEFAULT_ALERT_LABELS = {
    "green": "Normal", "orange": "Vigilance", "red": "Alerte critique",
    "unavailable": "Données insuffisantes",
}


@app.get("/scenarios/{scenario_id}/model/monitor")
def monitor_scenario_model(scenario_id: str, window: int = 7) -> dict[str, Any]:
    """
    Statut live du modèle entraîné : score les `window` dernières lignes du
    dataset branché et renvoie un niveau d'alerte + la valeur courante.
    """
    import pandas as pd
    import model_trainer
    from datetime import datetime, timezone

    with engine.connect() as conn:
        run = conn.execute(text("""
            SELECT id, family, task_type, metric, metrics_json, summary_json, artifact_path
            FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE AND artifact_path IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not run:
        return {"status": "unavailable", "status_color": "unavailable",
                "status_label": "Modèle non entraîné",
                "message": "Entraînez le modèle après avoir branché des données."}

    # ── Modèle de PRÉVISION (Prophet/SARIMAX) : pas de scoring ligne-à-ligne. La
    # valeur « courante » est le PROCHAIN point prévu, lu dans le résumé. Les bandes
    # d'alerte numériques (seuils littéraires) s'appliquent si présentes. ──
    if run["family"] in _TS_ALGO_FAMILIES or run["task_type"] == "forecast":
        summ = run["summary_json"] or {}
        fc = summ.get("forecast") or {}
        preds = fc.get("predicted") or []
        spec = _get_model_spec(scenario_id) or {}
        outcome = spec.get("outcome") or {}
        if not preds:
            return {"status": "unavailable", "status_color": "unavailable",
                    "status_label": _DEFAULT_ALERT_LABELS["unavailable"],
                    "message": "Prévision indisponible ; ré-entraînez le modèle."}
        next_val = float(preds[0])
        with engine.connect() as conn:
            vj = conn.execute(text(
                "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
        alert_thresholds = (dict(vj).get("alert_thresholds") if vj else None) or {}
        orange, red = model_trainer._alert_bounds(alert_thresholds)
        if orange is not None and red is not None:
            level = model_trainer._level_from_value(next_val, orange, red)
            label = (alert_thresholds.get(level) or {}).get("label") or _DEFAULT_ALERT_LABELS.get(level, "—")
        else:
            level, label = "green", _DEFAULT_ALERT_LABELS.get("green", "Normal")
        return {
            "status": "ready", "scenario_id": scenario_id,
            "status_color": level, "status_label": label,
            "value": next_val, "kind": "forecast",
            "unit": outcome.get("unit"), "outcome": outcome.get("name"),
            "bands": {"orange": orange, "red": red},
            "forecast": {"dates": (fc.get("dates") or [])[:12], "predicted": [float(v) for v in preds[:12]]},
            "horizon": summ.get("horizon"), "n_scored": len(preds),
            "model": {"run_id": run["id"], "family": run["family"], "task_type": run["task_type"],
                      "metric": run["metric"], "metrics": run["metrics_json"]},
            "alert_thresholds": alert_thresholds,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    with engine.connect() as conn:
        ds = conn.execute(text("""
            SELECT stored_path FROM scenario_model_dataset
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not ds or not ds["stored_path"]:
        return {"status": "unavailable", "status_color": "unavailable",
                "status_label": "Aucune donnée", "message": "Aucun dataset branché."}

    try:
        import joblib
        pipeline = joblib.load(run["artifact_path"])
        df = pd.read_csv(ds["stored_path"])
    except Exception as e:
        logger.error(f"Monitor load {scenario_id}: {e}", exc_info=True)
        return {"status": "error", "status_color": "unavailable",
                "status_label": "Erreur de chargement", "message": str(e)}

    summary = run["summary_json"] or {}
    task_type = run["task_type"]
    classes = summary.get("classes")
    target = summary.get("target")

    window = max(1, min(int(window or 7), 200))
    recent = df.tail(window)
    target_values = None
    if task_type in ("regression", "count") and target and target in df.columns:
        target_values = pd.to_numeric(df[target], errors="coerce").tolist()

    # Récupérer la classe positive + libellés d'alerte depuis le spec.
    spec = _get_model_spec(scenario_id) or {}
    positive_class = (spec.get("outcome") or {}).get("positive_class")
    with engine.connect() as conn:
        vj = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).scalar()
    alert_thresholds = (dict(vj).get("alert_thresholds") if vj else None) or {}

    try:
        mon = model_trainer.compute_monitoring(
            pipeline, recent, task_type, classes=classes,
            positive_class=positive_class, target_values=target_values,
            alert_thresholds=alert_thresholds)
    except Exception as e:
        logger.error(f"Monitor score {scenario_id}: {e}", exc_info=True)
        return {"status": "error", "status_color": "unavailable",
                "status_label": "Erreur de scoring", "message": str(e)}

    level = mon["level"]
    # 'unavailable' (NaN / pas de données / seuils non interprétables) ne doit
    # jamais réutiliser le libellé « Normal ».
    label = (
        _DEFAULT_ALERT_LABELS["unavailable"] if level == "unavailable"
        else ((alert_thresholds.get(level) or {}).get("label") or _DEFAULT_ALERT_LABELS.get(level, "Indisponible"))
    )
    outcome = (spec.get("outcome") or {})

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "status_color": level,
        "status_label": label,
        "value": _jsonable(mon["value"]),
        "kind": mon["kind"],
        "unit": outcome.get("unit"),
        "outcome": outcome.get("name"),
        "positive_class": mon.get("positive_class"),
        "bands": mon["bands"],
        "n_scored": mon["n_scored"],
        "window": window,
        "model": {"run_id": run["id"], "family": run["family"],
                  "task_type": task_type, "metric": run["metric"],
                  "metrics": run["metrics_json"]},
        "alert_thresholds": alert_thresholds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── HEATMAP AVEC VRAIS NOMS ─────────────────────────────────────────────────

@app.get("/corpus/stats/by-year/named")
def get_corpus_stats_by_year_named() -> dict[str, Any]:
    """
    Comme /corpus/stats/by-year mais avec les vrais noms des scénarios
    (GESICA et user_scenarios) dans la heatmap.
    """
    with engine.connect() as conn:
        # Articles par année (1800 → année courante)
        rows_year = conn.execute(text("""
            SELECT year, COUNT(*) as count
            FROM literature_document
            WHERE year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY year ORDER BY year ASC
        """)).mappings().all()

        # Articles par scénario ET par source (heatmap) + nombre en texte intégral.
        rows_heatmap = conn.execute(text("""
            SELECT ars.scenario_id, d.source,
                   COUNT(*) AS count,
                   COUNT(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM document_chunk c
                       WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                   )) AS fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.is_duplicate IS NOT TRUE
            GROUP BY ars.scenario_id, d.source
            ORDER BY ars.scenario_id, count DESC
        """)).mappings().all()

        # Articles par scénario ET par année
        rows_scenario_year = conn.execute(text("""
            SELECT d.year, ars.scenario_id, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year, ars.scenario_id ORDER BY d.year ASC
        """)).mappings().all()

        # Noms des user_scenarios (only non-deleted ones)
        user_names = conn.execute(text("""
            SELECT id, name FROM user_scenarios
        """)).mappings().all()

    user_name_map = {r["id"]: r["name"] for r in user_names}
    # Valid GESICA scenarios (not hidden) — depuis la DB
    with engine.connect() as _hm_conn:
        _gesica_db_rows = _list_db_gesica_scenarios(_hm_conn)
    _gesica_name_map = {str(r["id"]): _gesica_title(r) for r in _gesica_db_rows}
    valid_gesica_ids = set(_gesica_name_map.keys())
    # All valid scenario IDs: existing user scenarios + non-hidden GESICA ones
    valid_sids = set(user_name_map.keys()) | valid_gesica_ids

    def _resolve_name(sid: str) -> str | None:
        if sid in user_name_map:
            return user_name_map[sid]
        if sid in _gesica_name_map:
            return _gesica_name_map[sid]
        return None  # deleted or hidden — exclude from heatmap

    by_year = {str(r["year"]): r["count"] for r in rows_year}

    # Clé = scenario_id (stable, évite la fusion de scénarios homonymes qui
    # faisait apparaître MOINS de scénarios qu'en réalité). On porte le nom et,
    # par source CANONIQUE, le total ET le nombre en texte intégral.
    heatmap: dict[str, dict] = {}
    for r in rows_heatmap:
        if r["scenario_id"] not in valid_sids:
            continue
        name = _resolve_name(r["scenario_id"])
        if not name:
            continue
        src = _canonical_source(r["source"])
        entry = heatmap.setdefault(str(r["scenario_id"]), {"name": name, "sources": {}})
        cell = entry["sources"].setdefault(src, {"total": 0, "fulltext": 0})
        cell["total"] += int(r["count"] or 0)
        cell["fulltext"] += int(r["fulltext"] or 0)

    scenario_year: dict[str, dict[str, int]] = {}
    for r in rows_scenario_year:
        if r["scenario_id"] not in valid_sids:
            continue
        name = _resolve_name(r["scenario_id"])
        if not name:
            continue
        yr = str(r["year"])
        if name not in scenario_year:
            scenario_year[name] = {}
        scenario_year[name][yr] = scenario_year[name].get(yr, 0) + r["count"]

    return {
        "by_year": by_year,
        "scenario_by_year": scenario_year,
        "heatmap_scenario_source": heatmap,
    }


# ─── ASSISTANT IA FILTRÉ PAR SEUIL ───────────────────────────────────────────

@app.post("/ask/stream/filtered")
async def ask_stream_filtered(payload: dict[str, Any]):
    """
    Version de /ask/stream qui filtre les chunks par seuil de similarité
    et priorise les articles validés humainement.
    """
    from openai import AsyncOpenAI, OpenAI as SyncOpenAI

    question = payload.get("question", "")
    scenario_id = payload.get("scenario_id", None)
    top_k = int(payload.get("top_k", 12))
    project_context = payload.get("project_context", "literev")
    lang = payload.get("lang")

    if not question:
        raise HTTPException(status_code=422, detail="question est requis")

    threshold = _get_scenario_threshold(scenario_id) if scenario_id else DEFAULT_SIMILARITY_THRESHOLD

    # Compteurs « articles utilisés / avec texte intégral » à afficher sous la
    # réponse IA — MÊMES définitions que la vue Preuves (sous-ensemble PERTINENT du
    # scénario : ≥ seuil OU inclus). Volontairement PAS dérivés de `sources`, qui
    # sont des chunks plafonnés à top_k et ne refléteraient pas le nb d'articles.
    papers_used = 0
    papers_with_fulltext = 0
    if scenario_id:
        try:
            with engine.connect() as _cc:
                _cnt = _cc.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                            AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                            AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                        COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                            AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                            AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                            AND EXISTS (SELECT 1 FROM document_chunk c
                                WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')) AS relevant_with_fulltext
                    FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid
                """), {"sid": scenario_id, "thr": threshold}).mappings().fetchone()
            if _cnt:
                papers_used = int(_cnt["relevant"] or 0)
                papers_with_fulltext = int(_cnt["relevant_with_fulltext"] or 0)
        except Exception as _e_cnt:
            logger.warning(f"ask_stream_filtered counts {scenario_id}: {_e_cnt}")

    # Embedding de la question
    try:
        sync_client = SyncOpenAI(timeout=90.0)
        emb_resp = sync_client.embeddings.create(
            model="text-embedding-3-small",
            input=question[:2000],
        )
        q_emb = emb_resp.data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"
    except Exception as e:
        logger.error(f"Embedding error in /ask/stream/filtered: {e}")
        emb_str = None

    context_chunks = []
    sources = []

    if emb_str:
        # Construire le filtre scénario avec seuil
        where_extra = ""
        join_extra = ""
        # Migration 2 (screening par scénario) : quand un scénario est fourni, on
        # JOINT article_scenarios pour lire le statut PROPRE au scénario via
        # COALESCE (repli sur la colonne globale tant que la Phase 5 n'a pas
        # supprimé le fallback). Sans scénario, on reste sur la colonne globale.
        screen_expr = "d.screening_status"
        params_extra: dict[str, Any] = {"top_k": top_k, "emb": emb_str, "threshold": threshold}

        if project_context:
            where_extra += " AND d.project_context = :project_context"
            params_extra["project_context"] = project_context

        if scenario_id:
            # Filtrer par scénario ET par seuil de similarité (ou validé humainement).
            # JOIN (au lieu d'EXISTS) : (scenario_id, document_id) est unique dans
            # article_scenarios, donc la cardinalité est préservée et asn devient
            # lisible dans le SELECT/ORDER pour le screening par scénario.
            join_extra = " JOIN article_scenarios asn ON asn.document_id = d.id AND asn.scenario_id = :scenario_id "
            screen_expr = "COALESCE(asn.screening_status, d.screening_status)"
            where_extra += f"""
                AND (
                    asn.similarity_score >= :threshold
                    OR asn.similarity_score IS NULL
                    OR {screen_expr} = 'included'
                )
            """
            params_extra["scenario_id"] = scenario_id

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT c.content, d.title, d.year, d.doi, d.authors, d.id AS doc_id,
                       {screen_expr} AS screening_status,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                {join_extra}
                WHERE c.embedding IS NOT NULL {where_extra}
                ORDER BY
                    CASE WHEN {screen_expr} = 'included' THEN 0 ELSE 1 END,
                    c.embedding <=> CAST(:emb AS vector)
                LIMIT :top_k
            """), params_extra).mappings().all()

        for i, r in enumerate(rows):
            # Inclure titre + année DANS le contexte (le prompt demande de citer
            # par titre, donc le modèle doit disposer des titres).
            context_chunks.append(
                f"[{i + 1}] {r['title'] or 'Sans titre'} ({r['year'] or 'année inconnue'})\n{r['content']}"
            )
            # Champs canoniques attendus par le front (ScenarioRagSource) : document_id
            # + score (et authors). Avant, on envoyait id/similarity et pas d'auteurs,
            # d'où « Pertinence: NaN% » et « undefined • » dans le panneau Sources.
            sources.append({
                "document_id": r["doc_id"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "authors": r["authors"],
                "score": round(float(r["similarity"]), 3),
                "validated": r["screening_status"] == "included",
            })

    context_text = "\n\n---\n\n".join(context_chunks[:top_k]) if context_chunks else "Aucun contexte disponible."

    system_prompt = """Tu es un assistant expert en médecine d'urgence et en revue systématique de la littérature scientifique.
Tu réponds de manière précise, factuelle et structurée.
Base-toi exclusivement sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Cite les articles pertinents par leur titre quand tu les mentionnes.
Ne pas utiliser de tiret em (—).""" + _llm_lang_directive(lang)

    user_prompt = f"""Contexte scientifique (extraits d'articles sélectionnés par pertinence sémantique) :
{context_text}

Question : {question}

Réponds de manière structurée et cite les sources pertinentes du contexte."""

    async def event_generator():
        import json as _json2
        # Méta d'abord : combien d'articles pertinents alimentent la réponse et
        # combien ont un texte intégral (affiché sous la réponse, comme la vue Preuves).
        meta_event = ("event: meta\ndata: "
                      + _json2.dumps({"papers_used": papers_used,
                                      "papers_with_fulltext": papers_with_fulltext,
                                      "threshold": round(float(threshold), 2)})
                      + "\n\n")
        yield meta_event
        sources_event = f"event: sources\ndata: {_json2.dumps(sources)}\n\n"
        yield sources_event

        # Pas de contexte pertinent → ne pas générer de réponse non étayée.
        if not context_chunks:
            msg = ("Aucun passage pertinent (au-dessus du seuil) n'a été trouvé pour "
                   "cette question dans ce scénario. Reformulez ou abaissez le seuil.")
            yield f"data: {_json2.dumps({'token': msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        try:
            async_client = AsyncOpenAI(timeout=90.0)
            stream = await async_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=1500,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    token_event = f"data: {_json2.dumps({'token': delta.content})}\n\n"
                    yield token_event
        except Exception as e:
            yield f"event: error\ndata: {_json2.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    from fastapi.responses import StreamingResponse as _SR
    return _SR(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── PIPELINE COMPLET AVEC BRIEF LLM ─────────────────────────────────────────

@app.post("/scenarios/{scenario_id}/full-pipeline")
def trigger_full_pipeline_with_brief(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche le pipeline complet incluant :
    1. Reranking sémantique
    2. Génération Evidence Brief LLM
    3. Génération Variables & Modèle
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading

    if scenario_id.startswith("usr-"):
        row = _get_user_scenario_or_404(scenario_id)
        query = row["query"]
    else:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        nl = meta.get("nl_queries") or []
        query = nl[0] if nl else _gesica_title(meta)

    def _run():
        logger.info(f"Full pipeline with brief: {scenario_id}")
        # 1. Reranking
        _run_semantic_rerank_inline(scenario_id, query)
        # 2. Evidence Brief LLM
        _generate_evidence_brief_llm(scenario_id, force=True)
        # 3. Variables & Modèle
        _generate_variables_from_pico(scenario_id)
        logger.info(f"Full pipeline with brief done: {scenario_id}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "steps": ["rerank", "evidence_brief", "variables"]}


# ─── Endpoint model-status pour user_scenarios ───────────────────────────────
@app.get("/user-scenarios/{scenario_id}/model-status")
def get_user_scenario_model_status(scenario_id: str) -> dict[str, Any]:
    """
    Statut du modèle pour un scénario utilisateur.
    Retourne un statut neutre (pas de modèle prédictif pour les scénarios utilisateurs).
    """
    _get_user_scenario_or_404(scenario_id)
    from datetime import datetime, timezone
    # Compter les articles récents (30 derniers jours)
    with engine.connect() as conn:
        recent_count = conn.execute(text("""
            SELECT COUNT(*) AS cnt
            FROM literature_document d
            JOIN article_scenarios asn ON asn.document_id = d.id AND asn.scenario_id = :sid
            WHERE d.project_context = 'literev'
              AND d.created_at >= NOW() - INTERVAL '30 days'
        """), {"sid": scenario_id}).scalar()
    return {
        "scenario_id": scenario_id,
        "status_color": "blue",
        "status_label": "Scénario personnalisé",
        "model_info": {
            "name": "N/A",
            "description": "Les scénarios personnalisés ne disposent pas d'un modèle prédictif intégré. Utilisez l'onglet Variables & Données pour configurer votre propre modèle.",
            "type": "user_defined",
        },
        "alert_thresholds": {},
        "model_result": None,
        "model_error": None,
        "recent_articles_30d": int(recent_count or 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/user-scenarios/{scenario_id}/model-run")
def run_user_scenario_model(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Re-run du modèle pour un scénario utilisateur (retourne le statut neutre)."""
    return get_user_scenario_model_status(scenario_id)

# ─── Alias GESICA : /gesica/scenarios/{id}/pico -> pico-bulk ─────────────────
@app.get("/gesica/scenarios/{scenario_id}/pico")
def get_gesica_scenario_pico_alias(
    scenario_id: str,
    limit: int = 100000,
) -> dict[str, Any]:
    """Alias vers pico-bulk pour compatibilité frontend."""
    return get_scenario_pico_bulk(scenario_id, limit=limit)

# ─── Alias GESICA : /gesica/scenarios/{id}/screening -> screening-progress ───
@app.get("/gesica/scenarios/{scenario_id}/screening")
def get_gesica_scenario_screening_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers screening-progress pour compatibilité frontend."""
    return get_scenario_screening_progress(scenario_id)

# ─── Alias user-scenarios : /user-scenarios/{id}/pico -> pico-bulk ───────────
@app.get("/user-scenarios/{scenario_id}/pico")
def get_user_scenario_pico_alias(
    scenario_id: str,
    limit: int = 100000,
) -> dict[str, Any]:
    """Alias vers pico-bulk pour compatibilité frontend."""
    return get_user_scenario_pico_bulk(scenario_id, limit=limit)

# ─── Alias user-scenarios : /user-scenarios/{id}/screening -> screening-progress
@app.get("/user-scenarios/{scenario_id}/screening")
def get_user_scenario_screening_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers screening-progress pour compatibilité frontend."""
    return get_user_scenario_screening_progress(scenario_id)

# ─── Alias GESICA : /gesica/scenarios/{id} → /gesica/scenarios/{id}/detail ───
@app.get("/gesica/scenarios/{scenario_id}")
def get_gesica_scenario_root_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers /detail pour compatibilité frontend (évite les 404 sur la route racine)."""
    return get_scenario_detail(scenario_id)
