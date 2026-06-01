#!/bin/bash
# LiteRev deploy script
# Safe pattern: git pull then re-exec itself so bash always runs the latest version.
set -e

REPO_DIR="/opt/literev-api"
FRONTEND_SRC_DIR="$REPO_DIR/frontend"
FRONTEND_BUILD_DIR="/opt/literev-frontend"
NGINX_DIST="/var/www/literev-frontend/dist"

# ── Step 1: pull latest code ──────────────────────────────────────────────────
echo "==> [1/5] Git pull"
cd "$REPO_DIR"
git pull origin main

# ── Step 2: re-exec this script so bash uses the freshly pulled version ───────
# Only re-exec once (guard via env var DEPLOY_REEXEC=1)
if [ -z "$DEPLOY_REEXEC" ]; then
  echo "==> Re-launching with updated deploy.sh..."
  export DEPLOY_REEXEC=1
  exec bash "$REPO_DIR/deploy.sh"
fi

# ── Step 3: sync frontend source from repo to build folder ───────────────────
echo "==> [2/5] Sync frontend sources"
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
# Copier le lock file pour éviter les réinstallations inutiles
cp -f "$FRONTEND_SRC_DIR/pnpm-lock.yaml"      "$FRONTEND_BUILD_DIR/pnpm-lock.yaml" 2>/dev/null || true

# Vérification : s'assurer que ScenarioDetailPage est bien présent
if [ ! -f "$FRONTEND_BUILD_DIR/src/components/ScenarioDetailPage.tsx" ]; then
  echo "ERREUR : ScenarioDetailPage.tsx manquant après sync !" >&2
  exit 1
fi
echo "  ✓ ScenarioDetailPage.tsx présent"

# ── Step 4: build ─────────────────────────────────────────────────────────────
echo "==> [3/5] Frontend build"
cd "$FRONTEND_BUILD_DIR"
npm install --silent
npm run build

# ── Step 5: deploy to nginx ───────────────────────────────────────────────────
echo "==> [4/5] Deploy to Nginx"
mkdir -p "$NGINX_DIST"
rsync -a --delete dist/ "$NGINX_DIST/"

# ── Step 6: restart API ───────────────────────────────────────────────────────
echo "==> [5/5] Restart API"
systemctl restart literev-api 2>/dev/null || echo "(no systemd service — skipping)"

echo ""
echo "✅ Deploy complete — $(date)"
