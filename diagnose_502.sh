#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Script de diagnostic 502 — LiteRev API
# Exécuter sur app-01 (62.238.39.50) avec : sudo bash diagnose_502.sh
# ═══════════════════════════════════════════════════════════════════════
set -e
echo "═══════════════════════════════════════════════════════"
echo "  DIAGNOSTIC 502 — LiteRev API"
echo "═══════════════════════════════════════════════════════"

echo ""
echo "1. État du service literev-api :"
systemctl status literev-api --no-pager -l 2>&1 | tail -30

echo ""
echo "2. Dernières lignes de log du service :"
journalctl -u literev-api --no-pager -n 50 2>&1

echo ""
echo "3. Processus Python actifs :"
ps aux | grep -E "python|uvicorn|gunicorn" | grep -v grep

echo ""
echo "4. Ports en écoute :"
ss -tlnp | grep -E "8000|8001|8080|8765|5000"

echo ""
echo "5. Version du code sur le serveur :"
cd /opt/literev-api && git log --oneline -5

echo ""
echo "6. Vérification syntaxe main.py :"
python3 -m py_compile /opt/literev-api/main.py && echo "  ✓ Syntaxe OK" || echo "  ✗ ERREUR DE SYNTAXE"

echo ""
echo "7. Vérification syntaxe gesica_scenario_enriched_metadata.py :"
python3 -m py_compile /opt/literev-api/gesica_scenario_enriched_metadata.py && echo "  ✓ Syntaxe OK" || echo "  ✗ ERREUR DE SYNTAXE"

echo ""
echo "8. Test import des dépendances critiques :"
python3 -c "import fastapi; print('  ✓ fastapi OK')" 2>&1
python3 -c "import sqlalchemy; print('  ✓ sqlalchemy OK')" 2>&1
python3 -c "import openai; print('  ✓ openai OK')" 2>&1
python3 -c "import sklearn; print('  ✓ sklearn OK')" 2>&1
python3 -c "import umap; print('  ✓ umap-learn OK')" 2>&1 || echo "  ✗ umap-learn manquant — installer avec: pip3 install umap-learn"
python3 -c "import hdbscan; print('  ✓ hdbscan OK')" 2>&1 || echo "  ✗ hdbscan manquant — installer avec: pip3 install hdbscan"

echo ""
echo "9. Connexion à la base de données :"
python3 -c "
from sqlalchemy import create_engine, text
try:
    engine = create_engine('postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev')
    with engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM literature_document'))
        print(f'  ✓ DB OK — {result.scalar()} documents')
except Exception as e:
    print(f'  ✗ ERREUR DB : {e}')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  RÉSOLUTION : Exécuter les commandes suivantes :"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  # 1. Installer les dépendances manquantes"
echo "  pip3 install umap-learn hdbscan"
echo ""
echo "  # 2. Mettre à jour le code"
echo "  cd /opt/literev-api && git pull origin main"
echo ""
echo "  # 3. Redémarrer le service"
echo "  systemctl restart literev-api"
echo ""
echo "  # 4. Vérifier que le service est bien démarré"
echo "  systemctl status literev-api"
echo "  curl -s http://localhost:8000/gesica/scenarios | head -100"
