#!/usr/bin/env python3
"""
Script de diagnostic pour l'endpoint /gesica/scenarios.
À exécuter sur app-01 : .venv/bin/python3 debug_scenarios.py
"""
import os
import sys
import traceback

# Simuler exactement ce que fait l'endpoint
DB_URL = os.getenv("DB_URL", "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev")
print(f"[1] DB_URL = {DB_URL[:40]}...")

try:
    from sqlalchemy import create_engine, text
    print("[2] SQLAlchemy importé OK")
except ImportError as e:
    print(f"[ERREUR] SQLAlchemy manquant : {e}")
    sys.exit(1)

try:
    engine = create_engine(DB_URL, pool_pre_ping=True)
    print("[3] Engine créé OK")
except Exception as e:
    print(f"[ERREUR] Création engine : {e}")
    sys.exit(1)

try:
    with engine.connect() as conn:
        print("[4] Connexion DB OK")
        
        # Test 1 : colonnes de literature_document
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'literature_document'
            ORDER BY ordinal_position
        """)).fetchall()
        col_names = [c[0] for c in cols]
        print(f"[5] Colonnes literature_document : {col_names}")
        
        # Test 2 : vérifier scenario_type
        if 'scenario_type' not in col_names:
            print("[ERREUR] La colonne 'scenario_type' N'EXISTE PAS dans literature_document !")
        else:
            print("[6] Colonne scenario_type : OK")
        
        # Test 3 : compter les documents GESICA
        count = conn.execute(text("""
            SELECT COUNT(*) FROM literature_document WHERE project_context = 'gesica'
        """)).scalar()
        print(f"[7] Documents GESICA : {count}")
        
        # Test 4 : distribution des scénarios
        rows = conn.execute(text("""
            SELECT scenario_type, COUNT(*) as n
            FROM literature_document
            WHERE project_context = 'gesica'
            GROUP BY scenario_type
            ORDER BY n DESC
            LIMIT 10
        """)).fetchall()
        print(f"[8] Top 10 scénarios en DB :")
        for r in rows:
            print(f"    {r[0]} : {r[1]}")
        
        # Test 5 : simuler la requête exacte de l'endpoint
        sql_counts = text("""
            SELECT scenario_type, COUNT(*) as article_count
            FROM literature_document
            WHERE project_context = 'gesica' 
              AND scenario_type IS NOT NULL 
              AND scenario_type != 'unassigned'
            GROUP BY scenario_type;
        """)
        db_counts = {row[0]: row[1] for row in conn.execute(sql_counts).fetchall()}
        print(f"[9] db_counts (scénarios avec articles) : {len(db_counts)} scénarios")
        
        # Test 6 : vérifier GESICA_SCENARIO_METADATA dans main.py
        print("[10] Test import main.py...")
        sys.path.insert(0, '/opt/literev-api')
        try:
            # On importe juste la constante sans démarrer le serveur
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_check", "/opt/literev-api/main.py")
            # Ne pas exécuter le module entier, juste vérifier la syntaxe
            print("[10] main.py trouvé. Vérification de GESICA_SCENARIO_METADATA...")
            
            # Lire le fichier et chercher GESICA_SCENARIO_METADATA
            with open('/opt/literev-api/main.py', 'r') as f:
                content = f.read()
            if 'GESICA_SCENARIO_METADATA' in content:
                # Compter les entrées
                count_entries = content.count('"title":')
                print(f"[11] GESICA_SCENARIO_METADATA trouvé avec ~{count_entries} entrées 'title'")
            else:
                print("[ERREUR] GESICA_SCENARIO_METADATA introuvable dans main.py !")
                
            if '/gesica/scenarios' in content:
                print("[12] Endpoint /gesica/scenarios trouvé dans main.py : OK")
            else:
                print("[ERREUR] Endpoint /gesica/scenarios introuvable dans main.py !")
                
        except Exception as e:
            print(f"[ERREUR] Import main.py : {e}")
            
except Exception as e:
    print(f"[ERREUR] Connexion DB : {e}")
    traceback.print_exc()

print("\n--- DIAGNOSTIC TERMINÉ ---")
print("Copiez-collez ce résultat pour identifier le problème exact.")
