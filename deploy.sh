#!/bin/bash
# LiteRev deploy script
# Safe pattern: git pull then re-exec itself so bash always runs the latest version.
set -e

REPO_DIR="/opt/literev-api"
FRONTEND_SRC_DIR="$REPO_DIR/frontend"
FRONTEND_BUILD_DIR="/opt/literev-frontend"
NGINX_DIST="/var/www/literev-frontend/dist"

# ── Step 1: pull latest code ──────────────────────────────────────────────────
echo "==> [1/6] Git pull"
cd "$REPO_DIR"
git pull origin main

# ── Step 2: re-exec this script so bash uses the freshly pulled version ───────
# Only re-exec once (guard via env var DEPLOY_REEXEC=1)
if [ -z "$DEPLOY_REEXEC" ]; then
  echo "==> Re-launching with updated deploy.sh..."
  export DEPLOY_REEXEC=1
  exec bash "$REPO_DIR/deploy.sh"
fi

# ── Step 3: verify Python syntax ─────────────────────────────────────────────
echo "==> [2/6] Python syntax check"
python3 -m py_compile "$REPO_DIR/main.py"
echo "  OK — main.py valide"

# ── Step 4: sync frontend source from repo to build folder ───────────────────
echo "==> [3/6] Sync frontend sources"
# Sync complet du dossier src/ (inclut components/, lib/, etc.)
rsync -a --delete "$FRONTEND_SRC_DIR/src/"    "$FRONTEND_BUILD_DIR/src/"
rsync -a --delete "$FRONTEND_SRC_DIR/public/" "$FRONTEND_BUILD_DIR/public/" 2>/dev/null || true
cp -f "$FRONTEND_SRC_DIR/index.html"          "$FRONTEND_BUILD_DIR/index.html" 2>/dev/null || true
cp -f "$FRONTEND_SRC_DIR/package.json"        "$FRONTEND_BUILD_DIR/package.json"
cp -f "$FRONTEND_SRC_DIR/tsconfig.app.json"   "$FRONTEND_BUILD_DIR/tsconfig.app.json"
cp -f "$FRONTEND_SRC_DIR/tsconfig.json"       "$FRONTEND_BUILD_DIR/tsconfig.json"
cp -f "$FRONTEND_SRC_DIR/vite.config.ts"      "$FRONTEND_BUILD_DIR/vite.config.ts"
cp -f "$FRONTEND_SRC_DIR/tailwind.config.js"  "$FRONTEND_BUILD_DIR/tailwind.config.js"
cp -f "$FRONTEND_SRC_DIR/postcss.config.js"   "$FRONTEND_BUILD_DIR/postcss.config.js"
cp -f "$FRONTEND_SRC_DIR/pnpm-lock.yaml"      "$FRONTEND_BUILD_DIR/pnpm-lock.yaml" 2>/dev/null || true

# Vérification : s'assurer que ScenarioDetailPage est bien présent
if [ ! -f "$FRONTEND_BUILD_DIR/src/components/ScenarioDetailPage.tsx" ]; then
  echo "ERREUR : ScenarioDetailPage.tsx manquant apres sync !" >&2
  exit 1
fi
echo "  OK — ScenarioDetailPage.tsx present"

# ── Step 5: build ─────────────────────────────────────────────────────────────
echo "==> [4/6] Frontend build"
cd "$FRONTEND_BUILD_DIR"
npm install --silent
# Nettoyer les anciens assets AVANT le build pour eviter l'accumulation
rm -rf dist/assets/
npm run build
echo "  OK — build termine"

# Verifier le titre dans index.html
TITLE=$(grep -o '<title>.*</title>' dist/index.html || echo "non trouve")
echo "  Titre : $TITLE"

# ── Step 6: deploy to nginx ───────────────────────────────────────────────────
echo "==> [5/6] Deploy to Nginx (nettoyage cache)"
mkdir -p "$NGINX_DIST"
# Supprimer TOUS les anciens assets avant de copier les nouveaux
rm -rf "$NGINX_DIST/assets/"
rsync -a --delete dist/ "$NGINX_DIST/"
echo "  Assets deployes : $(ls $NGINX_DIST/assets/ | tr '\n' '  ')"

# ── Step 7: restart API + health check ───────────────────────────────────────
echo "==> [6/6] Restart API + health check"
systemctl restart literev-api 2>/dev/null || echo "(no systemd service — skipping)"
sleep 3
STATUS=$(systemctl is-active literev-api 2>/dev/null || echo "unknown")
echo "  Service : $STATUS"
HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "unreachable")
echo "  Health  : $HEALTH"

echo ""
echo "Deploy complete — $(date)"
echo "  main.py hash : $(md5sum $REPO_DIR/main.py | cut -d' ' -f1)"
echo "  JS bundle    : $(ls $NGINX_DIST/assets/*.js 2>/dev/null | xargs basename 2>/dev/null || echo 'non trouve')"
echo ""
