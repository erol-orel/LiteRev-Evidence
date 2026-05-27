#!/bin/bash
set -e

echo "==> Git pull"
cd /opt/literev-api
git pull origin main

echo "==> Sync frontend source from repo to build folder"
cp -r /opt/literev-api/frontend/src /opt/literev-frontend/
cp -r /opt/literev-api/frontend/public /opt/literev-frontend/ 2>/dev/null || true
cp /opt/literev-api/frontend/index.html /opt/literev-frontend/ 2>/dev/null || true

echo "==> Frontend build"
cd /opt/literev-frontend
npm install --silent
npm run build

echo "==> Deploy to Nginx"
cp -r dist/* /var/www/literev-frontend/dist/

echo "==> Restart API"
systemctl restart literev-api 2>/dev/null || echo "(no systemd service found)"

echo "✅ Done — $(date)"
