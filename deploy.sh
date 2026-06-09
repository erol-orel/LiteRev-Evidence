#!/bin/bash
# =============================================================
# deploy.sh — LiteRev Evidence to Scenario
# Source unique de vérité : /opt/literev-api (repo git)
# Build dans                : /opt/literev-frontend
# Servi par nginx depuis    : /var/www/literev-frontend/
# API uvicorn sur           : localhost:8000
# =============================================================
set -e

REPO_DIR="/opt/literev-api"
FRONTEND_SRC_DIR="$REPO_DIR/frontend"
FRONTEND_BUILD_DIR="/opt/literev-frontend"
NGINX_ROOT="/var/www/literev-frontend"

echo ""
echo "========================================"
echo "  LiteRev Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── 1. Git pull ───────────────────────────────────────────────
echo "[1/6] Git pull..."
cd "$REPO_DIR"
git pull origin main
echo "  OK — $(git log --oneline -1)"

# ── 2. Re-exec pour utiliser le deploy.sh mis à jour ─────────
if [ -z "$DEPLOY_REEXEC" ]; then
  echo "  Re-lancement avec le deploy.sh mis a jour..."
  export DEPLOY_REEXEC=1
  exec bash "$REPO_DIR/deploy.sh"
fi

# ── 3. Vérification syntaxe Python ───────────────────────────
echo "[2/6] Vérification syntaxe Python..."
python3 -m py_compile "$REPO_DIR/main.py"
echo "  OK — main.py valide (hash: $(md5sum $REPO_DIR/main.py | cut -d' ' -f1))"

# ── 4. Sync sources frontend ──────────────────────────────────
echo "[3/6] Sync sources frontend..."
rsync -a --delete "$FRONTEND_SRC_DIR/src/"    "$FRONTEND_BUILD_DIR/src/"
rsync -a --delete "$FRONTEND_SRC_DIR/public/" "$FRONTEND_BUILD_DIR/public/" 2>/dev/null || true
for f in index.html package.json tsconfig.app.json tsconfig.json vite.config.ts tailwind.config.js postcss.config.js; do
  [ -f "$FRONTEND_SRC_DIR/$f" ] && cp -f "$FRONTEND_SRC_DIR/$f" "$FRONTEND_BUILD_DIR/$f"
done
cp -f "$FRONTEND_SRC_DIR/pnpm-lock.yaml" "$FRONTEND_BUILD_DIR/pnpm-lock.yaml" 2>/dev/null || true
# Vérification critique
[ -f "$FRONTEND_BUILD_DIR/src/components/ScenarioDetailPage.tsx" ] || { echo "ERREUR: ScenarioDetailPage.tsx manquant!"; exit 1; }
echo "  OK — sources synchronisées"

# ── 5. Build frontend ─────────────────────────────────────────
echo "[4/6] Build frontend..."
cd "$FRONTEND_BUILD_DIR"
npm install --silent
# Nettoyer les anciens assets AVANT le build
rm -rf dist/assets/
npm run build 2>&1 | grep -E "built in|error TS|Error" || true
TITLE=$(grep -o '<title>.*</title>' dist/index.html 2>/dev/null || echo "TITRE NON TROUVE")
echo "  OK — $TITLE"
echo "  Bundle: $(ls dist/assets/*.js 2>/dev/null | xargs basename)"

# ── 6. Déploiement nginx (source unique de vérité) ───────────
echo "[5/6] Déploiement vers nginx ($NGINX_ROOT)..."
# Supprimer TOUT pour éviter les résidus d'anciens builds
rm -rf "$NGINX_ROOT"/*
# Copier le nouveau build
rsync -a dist/ "$NGINX_ROOT/"
# Vérification
DEPLOYED_TITLE=$(grep -o '<title>.*</title>' "$NGINX_ROOT/index.html" 2>/dev/null || echo "ERREUR")
DEPLOYED_JS=$(ls "$NGINX_ROOT/assets/"*.js 2>/dev/null | xargs basename || echo "ERREUR")
echo "  OK — $DEPLOYED_TITLE"
echo "  JS: $DEPLOYED_JS"
echo "  Assets: $(ls $NGINX_ROOT/assets/ | tr '\n' ' ')"

# Synchroniser aussi /opt/literev-api/frontend/dist/ pour cohérence
rsync -a --delete dist/ "$REPO_DIR/frontend/dist/"

# ── 7. Redémarrage API + health check ────────────────────────
echo "[6/6] Redémarrage API..."
systemctl restart literev-api 2>/dev/null || echo "(pas de systemd)"
sleep 3
STATUS=$(systemctl is-active literev-api 2>/dev/null || echo "unknown")
HEALTH=$(curl -s --max-time 5 http://localhost:8000/health 2>/dev/null || echo "unreachable")
echo "  Service : $STATUS"
echo "  Health  : $HEALTH"

echo ""
echo "========================================"
echo "  DEPLOY TERMINE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Commit  : $(git -C $REPO_DIR log --oneline -1)"
echo "  Titre   : $DEPLOYED_TITLE"
echo "  JS      : $DEPLOYED_JS"
echo "========================================"
echo ""
