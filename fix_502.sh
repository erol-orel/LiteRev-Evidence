#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Script de correction 502 — LiteRev API
# Exécuter sur app-01 avec : sudo bash fix_502.sh
# ═══════════════════════════════════════════════════════════════════════
VENV="/opt/literev-api/.venv"
API_DIR="/opt/literev-api"

echo "═══════════════════════════════════════════════════════"
echo "  CORRECTION 502 — LiteRev API"
echo "═══════════════════════════════════════════════════════"

echo ""
echo "1. Vérification de psycopg (v3) dans le .venv :"
if "$VENV/bin/python3" -c "import psycopg; print('  ✓ psycopg v3 OK')" 2>/dev/null; then
    echo "  → psycopg v3 présent"
else
    echo "  ✗ psycopg v3 manquant — installation..."
    "$VENV/bin/pip" install "psycopg[binary]" "psycopg[pool]" 2>&1 | tail -5
    echo "  ✓ psycopg v3 installé"
fi

echo ""
echo "2. Installation de python-multipart (requis pour l'upload de fichiers) :"
"$VENV/bin/pip" install python-multipart 2>&1 | tail -3
echo "  ✓ python-multipart OK"

echo ""
echo "3. Vérification des autres dépendances :"
"$VENV/bin/pip" install umap-learn hdbscan 2>&1 | grep -E "Successfully|already" | head -5
echo "  ✓ umap-learn + hdbscan OK"

echo ""
echo "4. Test de démarrage de main.py (import seulement) :"
cd "$API_DIR"
"$VENV/bin/python3" - 2>&1 << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    import main
    print("  ✓ main.py importé sans erreur")
except Exception as e:
    print(f"  ✗ ERREUR : {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "5. Redémarrage du service :"
systemctl restart literev-api
sleep 4
systemctl status literev-api --no-pager -l | tail -20

echo ""
echo "6. Test de l'endpoint /gesica/scenarios :"
sleep 2
result=$(curl -s --max-time 15 http://localhost:8000/gesica/scenarios 2>&1)
if echo "$result" | python3 -m json.tool > /dev/null 2>&1; then
    echo "  ✓ Endpoint répond avec du JSON valide !"
    echo "$result" | python3 -m json.tool | head -30
else
    echo "  ✗ Réponse invalide :"
    echo "$result" | head -20
fi

echo ""
echo "7. Logs récents du service :"
journalctl -u literev-api --no-pager -n 40 --since "3 minutes ago"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  TERMINÉ"
echo "═══════════════════════════════════════════════════════"
