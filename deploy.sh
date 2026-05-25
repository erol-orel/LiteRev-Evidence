#!/bin/bash
set -e

echo "==> Git pull"
cd /opt/literev-api
git pull origin main

echo "==> Frontend build"
cd /opt/literev-frontend
npm install --silent
npm run build

echo "==> Deploy to Nginx"
cp -r dist/* /var/www/literev-frontend/dist/

echo "==> Sync frontend source to repo"
cp -r /opt/literev-frontend/src /opt/literev-api/frontend/

echo "==> Restart API"
systemctl restart literev-api 2>/dev/null || echo "(no systemd service found)"

echo "✅ Done — $(date)"
