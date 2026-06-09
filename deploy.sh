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
# Usage : bash /opt/literev-api/deploy.sh
# =============================================================
set -e

REPO_DIR="/opt/literev-api"
FRONTEND_DIR="$REPO_DIR/frontend"
NGINX_ROOT="/var/www/literev-frontend"

echo ""
echo "========================================"
echo "  LiteRev Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── 1. Git pull ───────────────────────────────────────────────
echo "[1/5] Git pull..."
cd "$REPO_DIR"
git pull origin main
echo "  OK — $(git log --oneline -1)"

# Re-exec pour utiliser le deploy.sh fraichement tiré
if [ -z "$DEPLOY_REEXEC" ]; then
  export DEPLOY_REEXEC=1
  exec bash "$REPO_DIR/deploy.sh"
fi

# ── 2. Vérification syntaxe Python ───────────────────────────
echo "[2/5] Vérification syntaxe Python..."
python3 -m py_compile "$REPO_DIR/main.py"
echo "  OK — main.py ($(md5sum $REPO_DIR/main.py | cut -d' ' -f1))"

# ── 3. Build frontend ─────────────────────────────────────────
echo "[3/5] Build frontend..."
cd "$FRONTEND_DIR"
npm install --silent
# Nettoyer les anciens assets AVANT le build
rm -rf "$FRONTEND_DIR/dist/assets/"
npm run build 2>&1 | grep -E "built in|error|Error" || true
TITLE=$(grep -o '<title>.*</title>' "$FRONTEND_DIR/dist/index.html" 2>/dev/null || echo "TITRE NON TROUVE")
BUNDLE=$(ls "$FRONTEND_DIR/dist/assets/"*.js 2>/dev/null | xargs basename || echo "ERREUR")
echo "  OK — $TITLE | $BUNDLE"

# ── 4. Déploiement vers nginx (chemin unique) ─────────────────
echo "[4/5] Déploiement vers $NGINX_ROOT..."
# Supprimer TOUT pour éviter les résidus
rm -rf "$NGINX_ROOT"/*
# Copier le nouveau build
cp -r "$FRONTEND_DIR/dist/"* "$NGINX_ROOT/"
# Vérification
echo "  OK — $(grep -o '<title>.*</title>' $NGINX_ROOT/index.html)"
echo "  Assets: $(ls $NGINX_ROOT/assets/)"

# ── 5. Redémarrage API + health check ────────────────────────
echo "[5/5] Redémarrage API..."
systemctl restart literev-api
sleep 3
echo "  Service : $(systemctl is-active literev-api)"
echo "  Health  : $(curl -s --max-time 5 http://localhost:8000/health)"

echo ""
echo "========================================"
echo "  DEPLOY OK — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Commit : $(git -C $REPO_DIR log --oneline -1)"
echo "========================================"
echo ""
