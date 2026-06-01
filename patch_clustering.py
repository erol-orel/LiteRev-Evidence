#!/usr/bin/env python3
"""Remplace l'ancien corps de get_scenario_clustering (lignes 2971-3176) par la nouvelle version async."""
import re

with open("main.py", "r") as f:
    content = f.read()

# Nouveau corps de get_scenario_clustering
new_body = '''    import threading
    meta = GESICA_SCENARIO_METADATA.get(scenario_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Scénario '{scenario_id}' non trouvé")

    # Si résultat en cache mémoire, retourner immédiatement
    job = _clustering_jobs.get(scenario_id)
    if job and job["status"] == "done" and not force_refresh:
        return job["result"]
    if job and job["status"] == "error" and not force_refresh:
        return {"scenario_id": scenario_id, "status": "error", "error": job.get("error"), "clusters": []}

    # Lancer le calcul en arrière-plan si pas déjà en cours
    if not job or job.get("status") not in ("running",) or force_refresh:
        _clustering_jobs[scenario_id] = {"status": "running"}
        t = threading.Thread(target=_run_clustering_background, args=(scenario_id, force_refresh), daemon=True)
        t.start()

    return {"scenario_id": scenario_id, "status": "running",
            "message": "Calcul en cours (embeddings + UMAP + HDBSCAN). Revenez dans 30-60s ou utilisez /clustering/status.",
            "clusters": []}


@app.get("/gesica/scenarios/{scenario_id}/clustering/status")
def get_clustering_status(scenario_id: str) -> dict:
    """Vérifie si le clustering est terminé et retourne le résultat si disponible."""
    job = _clustering_jobs.get(scenario_id)
    if not job:
        return {"scenario_id": scenario_id, "status": "not_started",
                "message": "Aucun calcul lancé. Appelez GET /clustering d'abord."}
    if job["status"] == "running":
        return {"scenario_id": scenario_id, "status": "running",
                "message": "Calcul en cours..."}
    if job["status"] == "error":
        return {"scenario_id": scenario_id, "status": "error",
                "error": job.get("error", "Erreur inconnue")}
    # done
    return job["result"]
'''

# Pattern pour trouver l'ancien corps (entre la def et le prochain @app)
old_pattern = r'(    import json as _json\n    import time as _time\n    import threading\n.*?    return result\n)'

# Remplacer avec re.DOTALL
new_content = re.sub(old_pattern, new_body, content, count=1, flags=re.DOTALL)

if new_content == content:
    print("ERREUR: Pattern non trouvé, essai avec un pattern plus simple")
    # Trouver les lignes 2971-3176 et les remplacer
    lines = content.split('\n')
    # Trouver la ligne "    import json as _json" après la def get_scenario_clustering
    start_line = None
    end_line = None
    in_func = False
    for i, line in enumerate(lines):
        if '@app.get("/gesica/scenarios/{scenario_id}/clustering")' in line:
            in_func = True
        if in_func and '    import json as _json' in line and start_line is None:
            start_line = i
        if in_func and start_line and '    return result' in line:
            end_line = i
            break
    
    if start_line and end_line:
        print(f"Remplacement des lignes {start_line+1} à {end_line+1}")
        new_lines = lines[:start_line] + new_body.split('\n') + lines[end_line+1:]
        new_content = '\n'.join(new_lines)
        print("OK: Remplacement effectué")
    else:
        print(f"Lignes trouvées: start={start_line}, end={end_line}")
        exit(1)
else:
    print("OK: Pattern trouvé et remplacé")

with open("main.py", "w") as f:
    f.write(new_content)

print("main.py mis à jour")
