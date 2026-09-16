"""Configuration, database engine, FastAPI app, middleware, auth and shared helpers.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import logging
import os
import re
import secrets as _secrets
import time
import time as _time_mod
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine


def _msg(lang, fr: str, en: str) -> str:
    """Texte utilisateur dans la langue demandée (français par défaut) : pour les
    messages d'état renvoyés par l'API et affichés tels quels par l'interface."""
    return en if (isinstance(lang, str) and lang.strip().lower().startswith("en")) else fr

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

# Charger aussi le fichier secrets hors-repo (jamais commité)
for _ep in ["/etc/literev/secrets", "/opt/literev-api/secrets.env"]:
    _load_env_file(_ep)

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("La variable d'environnement DB_URL est requise et n'est pas configurée.")

# ReliefWeb : appname OBLIGATOIRE et PRÉ-APPROUVÉ depuis le 1er novembre 2025
# (« From 1 November 2025, API users will require a pre-approved appname », apidoc.reliefweb.int).
# Vide = flux ReliefWeb désactivé ; l'ingestion répond alors 503 en l'expliquant.
RELIEFWEB_APPNAME = os.getenv("RELIEFWEB_APPNAME") or ""
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

# Limiteur de débit in-memory robuste par IP (H-2)
class InMemoryRateLimiter:
    def __init__(self, requests_limit: int, window_seconds: int):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        # Stocke les timestamps des requêtes pour chaque IP
        self.history: dict[str, list[float]] = defaultdict(list)
        self._last_sweep = time.time()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Balayage périodique : purge les IP sans requête récente. Sans cela, la dict
        # `history` ne perdait JAMAIS ses clés (une IP qui ne revient pas gardait son
        # entrée vide indéfiniment) → croissance mémoire non bornée avec des IP
        # distinctes / des X-Forwarded-For tournants.
        if now - self._last_sweep > self.window_seconds:
            self._last_sweep = now
            cutoff = now - self.window_seconds
            self.history = defaultdict(list, {
                k: recent for k, v in self.history.items()
                if (recent := [t for t in v if t > cutoff])
            })
        # Filtrer les anciens timestamps hors de la fenêtre
        self.history[ip] = [t for t in self.history[ip] if now - t < self.window_seconds]
        if len(self.history[ip]) >= self.requests_limit:
            return False
        self.history[ip].append(now)
        return True

# Le frontend est volubile (tableau de bord = nombreux appels, recherche = gros
# payloads) : limites généreuses pour éviter les faux positifs, plus strictes
# sur les endpoints coûteux (RAG, recherche, génération de briefs).
def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Integer setting read from the environment; the default when unset or invalid."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# Per-IP limits, requests per minute. Overridable for an event where a whole room
# shares one public IP (a presentation, a workshop): RATE_LIMIT_GENERAL_PER_MIN and
# RATE_LIMIT_EXPENSIVE_PER_MIN in the API environment, then restart the service.
# /health reports the values in force.
RATE_LIMIT_GENERAL_PER_MIN = _env_int("RATE_LIMIT_GENERAL_PER_MIN", 600, minimum=1)
RATE_LIMIT_EXPENSIVE_PER_MIN = _env_int("RATE_LIMIT_EXPENSIVE_PER_MIN", 30, minimum=1)
general_limiter = InMemoryRateLimiter(requests_limit=RATE_LIMIT_GENERAL_PER_MIN, window_seconds=60)
expensive_limiter = InMemoryRateLimiter(requests_limit=RATE_LIMIT_EXPENSIVE_PER_MIN, window_seconds=60)
_PROCESS_STARTED_AT = _time_mod.time()
try:
    _SLOW_REQUEST_MS = max(0, int(os.getenv("SLOW_REQUEST_MS", "2000")))
except ValueError:
    _SLOW_REQUEST_MS = 2000


def _process_stats() -> dict[str, Any]:
    """Mémoire résidente (courante et pic), threads, uptime et état du pool DB du
    processus API - lisible dans /health sans accès au serveur."""
    out: dict[str, Any] = {"uptime_s": int(_time_mod.time() - _PROCESS_STARTED_AT)}
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    out["rss_mb"] = round(int(_line.split()[1]) / 1024.0, 1)
                elif _line.startswith("VmHWM:"):
                    out["rss_peak_mb"] = round(int(_line.split()[1]) / 1024.0, 1)
                elif _line.startswith("Threads:"):
                    out["threads"] = int(_line.split()[1])
    except Exception:                                  # noqa: BLE001 - non Linux
        pass
    try:
        out["db_pool"] = engine.pool.status()
    except Exception:                                  # noqa: BLE001
        pass
    return out

# Nb de sauts de proxy DE CONFIANCE devant l'app (défaut 1 = un seul nginx). L'IP
# client réelle est le N-ième saut en partant de la FIN du X-Forwarded-For ; les
# entrées avant ce point sont contrôlables par le client (spoofing). Passer à 2 si
# un CDN/LB s'ajoute devant nginx - sinon on limiterait sur l'IP du CDN (limite
# globale) ou on ferait confiance à un XFF spoofé. Borné à >= 1 ; parse tolérant.
try:
    _TRUSTED_PROXY_HOPS = max(1, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))
except (TypeError, ValueError):
    _TRUSTED_PROXY_HOPS = 1

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
    # IP réelle derrière le reverse proxy : on ne fait confiance qu'aux
    # _TRUSTED_PROXY_HOPS derniers sauts (notre infra) et on prend le client réel
    # juste avant. Les entrées antérieures du X-Forwarded-For sont spoofables. On
    # ignore les segments vides et on retombe sur request.client si l'en-tête est
    # absent ou malformé (évite un bucket de rate-limit partagé sur IP "" ).
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = ""
    if forwarded_for:
        _parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if _parts:
            client_ip = _parts[-_TRUSTED_PROXY_HOPS] if len(_parts) >= _TRUSTED_PROXY_HOPS else _parts[0]
    if not client_ip:
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
    # la relancer telle quelle (Starlette renvoie son 500 habituel - aucun
    # changement de comportement). Les HTTPException sont déjà converties en
    # réponses en amont et ne remontent donc pas ici.
    _t0 = _time_mod.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            f"Unhandled error on {request.method} {path} from {client_ip}: {exc}",
            exc_info=True,
        )
        raise
    _ms = (_time_mod.perf_counter() - _t0) * 1000.0
    if _ms >= _SLOW_REQUEST_MS:
        logger.warning(
            f"slow request: {request.method} {path} {int(_ms)} ms status={response.status_code} "
            f"bytes={response.headers.get('content-length', '?')} ip={client_ip}"
        )
    return response

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
# d'attente quand une source est lente/bloquée (PubMed efetch timeout=90). Les sources
# non terminées continuent en arrière-plan ; le corpus est reconstruit avec ce qui est
# déjà ingéré. Réglable via l'env POPULATE_FEDERATION_BUDGET (mettre p. ex. 600 pour un
# corpus plus complet). Défaut 180 s : le populate tourne dans un THREAD de fond, donc
# un budget plus large ne bloque pas la requête - il retarde juste le passage à « done ».
# On NE le met pas à l'infini : une source réellement bloquée maintiendrait le job en vie.
try:
    POPULATE_FEDERATION_BUDGET = float(os.getenv("POPULATE_FEDERATION_BUDGET", "180"))
except (TypeError, ValueError):
    POPULATE_FEDERATION_BUDGET = 180.0


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


def _job_is_active(job: dict | None, stale_after: float = 180.0) -> bool:
    """True only if a background job is genuinely still running. A "running"
    entry older than stale_after seconds (a thread killed by a restart or hung
    on a network call) is treated as STALE, so a crashed/killed job never locks
    out retries permanently with "already_running"."""
    import time
    if not job or job.get("status") != "running":
        return False
    return (time.time() - job.get("started_at", 0)) < stale_after


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
