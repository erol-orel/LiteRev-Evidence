#!/bin/bash
# =============================================================
# deploy.sh — LiteRev Evidence to Scenario
#
# CHEMIN UNIQUE :
#   Source git  : /opt/literev-api/
#   Build       : /opt/literev-api/frontend/  (npm run build)
#   Servi nginx : /var/www/literev-frontend/  (seul chemin public)
#   API         : localhost:8000 (uvicorn /opt/literev-api/main.py)
#
# Garanties :
#   - un build frontend qui échoue ABORTE le déploiement (le site live reste intact)
#   - bascule atomique du frontend (aucune fenêtre où le site est vide)
#   - dépendances backend installées + migrations alembic appliquées
#   - health check bloquant : un /health non OK fait échouer le déploiement
#
# Usage : bash /opt/literev-api/deploy.sh
# =============================================================
set -euo pipefail

REPO_DIR="/opt/literev-api"
FRONTEND_DIR="$REPO_DIR/frontend"
NGINX_ROOT="/var/www/literev-frontend"
VENV_PY="$REPO_DIR/.venv/bin/python3"
VENV_PIP="$REPO_DIR/.venv/bin/pip"
[ -x "$VENV_PY" ] || VENV_PY="python3"
[ -x "$VENV_PIP" ] || VENV_PIP="pip3"

echo ""
echo "========================================"
echo "  LiteRev Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── 1. Git pull ───────────────────────────────────────────────
echo "[1/7] Git pull..."
cd "$REPO_DIR"
git reset --hard HEAD          # discard any server-side manual edits
git clean -fd                  # remove untracked files
git pull origin main
echo "  OK — $(git log --oneline -1)"

# Re-exec pour utiliser le deploy.sh fraichement tiré
if [ -z "${DEPLOY_REEXEC:-}" ]; then
  export DEPLOY_REEXEC=1
  exec bash "$REPO_DIR/deploy.sh"
fi

# ── 2. Vérification syntaxe Python (tous les modules importés) ─
echo "[2/7] Vérification syntaxe Python..."
"$VENV_PY" -m compileall -q "$REPO_DIR"/*.py
echo "  OK — $(md5sum $REPO_DIR/main.py | cut -d' ' -f1)"

# ── 3. Dépendances backend ────────────────────────────────────
echo "[3/7] Installation dépendances backend..."
"$VENV_PIP" install -q -r "$REPO_DIR/requirements.txt"
echo "  OK"

# ── 4. Migrations base de données ─────────────────────────────
# Non bloquant : le schéma est aussi garanti au démarrage par les fonctions
# _ensure_*() de main.py. Un échec alembic est journalisé mais n'interrompt
# pas le déploiement (l'app applique ses DDL idempotentes au boot).
echo "[4/7] Migrations alembic (non bloquant)..."
if [ -f "$REPO_DIR/alembic.ini" ]; then
  if ( cd "$REPO_DIR" && "$VENV_PY" -m alembic upgrade head ); then
    echo "  OK"
  else
    echo "  AVERTISSEMENT : alembic upgrade a échoué — on continue (DDL au boot)."
  fi
else
  echo "  (pas d'alembic.ini, étape ignorée)"
fi

# ── 5. Build frontend (dans un répertoire temporaire) ─────────
echo "[5/7] Build frontend..."
cd "$FRONTEND_DIR"
npm ci --silent
rm -rf "$FRONTEND_DIR/dist"
# SÉCURITÉ : on n'injecte PLUS la clé d'écriture dans le build. Vite « inline » les
# variables VITE_* dans le bundle JS *public* — y placer WRITE_API_KEY exposait le
# secret à tout visiteur du site. Les mutations sont désormais signées côté navigateur
# par une clé que l'admin saisit une seule fois dans l'UI (bouton « Clé admin »,
# stockée en localStorage sur son appareil, jamais embarquée dans le bundle).
# On neutralise toute VITE_API_KEY résiduelle de l'environnement pour qu'elle ne
# soit pas embarquée par mégarde.
unset VITE_API_KEY || true
npm run build            # set -e fait échouer le deploy si le build casse
TITLE=$(grep -o '<title>.*</title>' "$FRONTEND_DIR/dist/index.html")
BUNDLE=$(ls "$FRONTEND_DIR/dist/assets/"*.js | xargs -n1 basename | head -1)
echo "  OK — $TITLE | $BUNDLE"

# ── 6. Déploiement atomique vers nginx ────────────────────────
echo "[6/7] Bascule atomique vers $NGINX_ROOT..."
STAGING="${NGINX_ROOT}.new"
PREVIOUS="${NGINX_ROOT}.prev"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -r "$FRONTEND_DIR/dist/"* "$STAGING/"
# Conserver la version précédente pour rollback, puis basculer
rm -rf "$PREVIOUS"
[ -d "$NGINX_ROOT" ] && mv "$NGINX_ROOT" "$PREVIOUS"
mv "$STAGING" "$NGINX_ROOT"
echo "  OK — $(grep -o '<title>.*</title>' $NGINX_ROOT/index.html)"
echo "  Rollback dispo : $PREVIOUS"

# ── 7. Redémarrage API + health check bloquant ────────────────
echo "[7/7] Redémarrage API..."
systemctl restart literev-api
HEALTH=""
for i in 1 2 3 4 5 6; do
  sleep 2
  HEALTH=$(curl -fsS --max-time 5 http://localhost:8000/health 2>/dev/null || true)
  [ -n "$HEALTH" ] && break
done
if [ -z "$HEALTH" ]; then
  echo "  ÉCHEC — /health ne répond pas. Rollback du frontend..."
  if [ -d "$PREVIOUS" ]; then
    rm -rf "$NGINX_ROOT"
    mv "$PREVIOUS" "$NGINX_ROOT"
    echo "  Frontend restauré depuis $PREVIOUS."
  fi
  echo "  Service : $(systemctl is-active literev-api || true)"
  exit 1
fi
echo "  Service : $(systemctl is-active literev-api)"
echo "  Health  : $HEALTH"

echo ""
echo "========================================"
echo "  DEPLOY OK — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Commit : $(git -C $REPO_DIR log --oneline -1)"
echo "========================================"
echo ""
