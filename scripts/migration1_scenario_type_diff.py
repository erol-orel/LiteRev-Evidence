#!/usr/bin/env python3
"""
migration1_scenario_type_diff.py — APERÇU LECTURE SEULE (aucune écriture).

Migration 1 (scenario_type → article_scenarios, « Way B ») : avant de réécrire les
~29 endpoints qui scopent par `d.scenario_type = :sid` (appartenance « ingestion »),
ce script montre, PAR SCÉNARIO, comment le corpus changerait si on scopait plutôt
par `article_scenarios` (appartenance « scorée »).

Il NE MODIFIE RIEN. Il fait uniquement des SELECT.

Pour chaque scénario il affiche :
  A   = nb de documents avec scenario_type = sid          (définition actuelle)
  B   = nb de documents liés via article_scenarios        (définition cible « Way B »)
  ∩   = documents présents dans les DEUX
  A\\B = documents qui DISPARAÎTRAIENT du scénario (ingestion sans lien scoré)
  B\\A = documents qui APPARAÎTRAIENT en plus (scorés mais ingérés ailleurs)

⚠ DANGER signalé si A>0 et B==0 : le scénario serait VIDÉ par « Way B ». Ces
scénarios nécessitent d'abord un backfill (créer les lignes article_scenarios
manquantes depuis scenario_type) avant de basculer les endpoints.

Usage :
    DATABASE_URL='postgresql+psycopg://user:pwd@host:5432/literev' \
        python3 scripts/migration1_scenario_type_diff.py

    # Idéalement contre une COPIE/snapshot en lecture seule, pas la prod live.
    # (Un rôle en lecture seule suffit : le script n'émet que des SELECT.)
"""
import os
import sys

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if not DB_URL:
    sys.exit("DATABASE_URL (ou DB_URL) requis. Ex: "
             "DATABASE_URL='postgresql+psycopg://user:pwd@host:5432/literev' python3 "
             "scripts/migration1_scenario_type_diff.py")

from sqlalchemy import create_engine, text  # noqa: E402

# Une seule requête : pour chaque scénario, le recouvrement entre l'appartenance
# « ingestion » (scenario_type) et « scorée » (article_scenarios). Filtre doublons
# identique des deux côtés pour une comparaison juste.
DIFF_SQL = text("""
WITH a AS (  -- ingestion membership (current definition: scenario_type)
    SELECT scenario_type AS sid, id AS doc_id
    FROM literature_document
    WHERE project_context = 'literev'
      AND scenario_type IS NOT NULL
      AND (is_duplicate IS NULL OR is_duplicate = FALSE)
), b AS (    -- scored membership (target definition: article_scenarios / Way B)
    SELECT ars.scenario_id AS sid, ars.document_id AS doc_id
    FROM article_scenarios ars
    JOIN literature_document d ON d.id = ars.document_id
    WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
)
SELECT COALESCE(a.sid, b.sid) AS sid,
       COUNT(*) FILTER (WHERE a.doc_id IS NOT NULL AND b.doc_id IS NOT NULL) AS both,
       COUNT(*) FILTER (WHERE a.doc_id IS NOT NULL AND b.doc_id IS NULL)     AS a_only,
       COUNT(*) FILTER (WHERE a.doc_id IS NULL     AND b.doc_id IS NOT NULL) AS b_only
FROM a
FULL OUTER JOIN b ON a.sid = b.sid AND a.doc_id = b.doc_id
GROUP BY COALESCE(a.sid, b.sid)
ORDER BY sid
""")


def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True, client_encoding="utf8")
    with engine.connect() as conn:
        rows = conn.execute(DIFF_SQL).mappings().all()

    if not rows:
        print("Aucun scénario trouvé (ni scenario_type, ni article_scenarios).")
        return

    hdr = f"{'scénario':40} {'A (actuel)':>11} {'B (cible)':>10} {'communs':>8} {'perdus(A)':>11} {'gagnés(B)':>11}"
    print(hdr)
    print("-" * len(hdr))

    danger, totals = [], {"a": 0, "b": 0, "both": 0, "a_only": 0, "b_only": 0}
    for r in rows:
        sid = str(r["sid"])
        both, a_only, b_only = int(r["both"]), int(r["a_only"]), int(r["b_only"])
        a_count, b_count = both + a_only, both + b_only
        totals["a"] += a_count; totals["b"] += b_count; totals["both"] += both
        totals["a_only"] += a_only; totals["b_only"] += b_only
        flag = "  ⚠ VIDÉ" if (a_count > 0 and b_count == 0) else ""
        if flag:
            danger.append(sid)
        print(f"{sid[:40]:40} {a_count:>11} {b_count:>10} {both:>8} {a_only:>11} {b_only:>11}{flag}")

    print("-" * len(hdr))
    print(f"{'TOTAL':40} {totals['a']:>11} {totals['b']:>10} {totals['both']:>8} "
          f"{totals['a_only']:>11} {totals['b_only']:>11}")
    print()
    print(f"Scénarios : {len(rows)}")
    if danger:
        print(f"⚠ {len(danger)} scénario(s) seraient VIDÉS par « Way B » "
              f"(backfill article_scenarios requis avant bascule) : {', '.join(danger[:20])}"
              + (" …" if len(danger) > 20 else ""))
    else:
        print("✓ Aucun scénario vidé : « Way B » est sûr sans backfill préalable.")
    print()
    print("Rappel : ce script n'a RIEN modifié (SELECT uniquement).")


if __name__ == "__main__":
    main()
