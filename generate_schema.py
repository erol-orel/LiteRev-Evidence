#!/usr/bin/env python3
"""
generate_schema.py — Extrait le schéma SQL complet depuis la base PostgreSQL
et le sauvegarde dans schema.sql pour versionnement dans GitHub.

Usage:
    DB_URL='postgresql+psycopg://literev:...@10.10.1.10:5432/literev' python3 generate_schema.py
"""

import os
import sys
from sqlalchemy import create_engine, inspect, text

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")


def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True)
    inspector = inspect(engine)

    lines: list[str] = [
        "-- ============================================================",
        "-- LiteRev-Evidence — Schéma PostgreSQL",
        "-- Généré automatiquement par generate_schema.py",
        "-- ============================================================",
        "",
        "-- Extensions requises",
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
        "",
    ]

    tables = inspector.get_table_names()
    for table in sorted(tables):
        cols = inspector.get_columns(table)
        pk_constraint = inspector.get_pk_constraint(table)
        fks = inspector.get_foreign_keys(table)
        indexes = inspector.get_indexes(table)
        unique_constraints = inspector.get_unique_constraints(table)

        lines.append(f"-- ──────────────────────────────────────────────────────────")
        lines.append(f"-- Table: {table}")
        lines.append(f"-- ──────────────────────────────────────────────────────────")
        lines.append(f"CREATE TABLE IF NOT EXISTS {table} (")

        col_defs: list[str] = []
        for col in cols:
            col_type = str(col["type"])
            nullable = "" if col["nullable"] else " NOT NULL"
            default = ""
            if col.get("default") is not None:
                default = f" DEFAULT {col['default']}"
            col_defs.append(f"    {col['name']} {col_type}{nullable}{default}")

        # Primary key
        pk_cols = pk_constraint.get("constrained_columns", [])
        if pk_cols:
            pk_name = pk_constraint.get("name") or f"pk_{table}"
            col_defs.append(f"    CONSTRAINT {pk_name} PRIMARY KEY ({', '.join(pk_cols)})")

        # Foreign keys
        for fk in fks:
            fk_cols = ", ".join(fk["constrained_columns"])
            ref_table = fk["referred_table"]
            ref_cols = ", ".join(fk["referred_columns"])
            fk_name = fk.get("name") or f"fk_{table}_{ref_table}"
            col_defs.append(
                f"    CONSTRAINT {fk_name} FOREIGN KEY ({fk_cols}) REFERENCES {ref_table} ({ref_cols})"
            )

        lines.append(",\n".join(col_defs))
        lines.append(");")
        lines.append("")

        # Indexes
        for idx in indexes:
            if idx.get("unique"):
                uc_name = idx["name"]
                uc_cols = ", ".join(idx["column_names"])
                lines.append(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {uc_name} ON {table} ({uc_cols});"
                )
            else:
                idx_name = idx["name"]
                idx_cols = ", ".join(idx["column_names"])
                lines.append(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({idx_cols});"
                )

        # Unique constraints (not already covered by unique indexes)
        for uc in unique_constraints:
            uc_name = uc.get("name") or f"uc_{table}"
            uc_cols = ", ".join(uc["column_names"])
            lines.append(
                f"ALTER TABLE {table} ADD CONSTRAINT IF NOT EXISTS {uc_name} UNIQUE ({uc_cols});"
            )

        lines.append("")

    # Récupérer les triggers et fonctions via SQL brut
    with engine.connect() as conn:
        triggers = conn.execute(text("""
            SELECT trigger_name, event_manipulation, event_object_table, action_statement
            FROM information_schema.triggers
            WHERE trigger_schema = 'public'
            ORDER BY event_object_table, trigger_name
        """)).mappings().all()

        if triggers:
            lines.append("-- ──────────────────────────────────────────────────────────")
            lines.append("-- Triggers")
            lines.append("-- ──────────────────────────────────────────────────────────")
            for t in triggers:
                lines.append(
                    f"-- Trigger: {t['trigger_name']} ON {t['event_object_table']} ({t['event_manipulation']})"
                )
                lines.append(f"-- Action: {t['action_statement']}")
                lines.append("")

    schema_sql = "\n".join(lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(schema_sql)

    print(f"[OK] Schéma exporté dans {OUTPUT_FILE} ({len(lines)} lignes)")
    print(f"[INFO] Tables trouvées : {', '.join(sorted(tables))}")


if __name__ == "__main__":
    main()
