"""Les tables qu'un test de la recherche ou de l'ingestion lit, créées si la base de test
ne les a pas encore.

La suite amorce une base vide et chaque test d'intégration crée ce qu'il lit. Ceux de la
recherche passent tôt dans l'ordre alphabétique (`test_a_...`), avant les tests qui créent
`literature_document` : sans ceci, ils échouaient sur une base neuve et passaient sur une
base déjà servie. Mêmes colonnes que la production pour ce qui est lu ; le reste n'est
pas créé, et rien n'est supprimé (une table existante garde ses colonnes et ses données).
"""
import main
from conftest import ensure_document_columns


def ensure_search_tables(cur) -> bool:
    """Crée ce qui manque. Renvoie True quand `document_chunk` a dû être créée (l'appelant
    la retire à la fin, comme le demande `ensure_document_columns`)."""
    cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
    if cur.fetchone()[0]:
        main._ensure_user_scenarios_table()
    cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
    cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
    created_chunk_table = ensure_document_columns(cur)
    for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE"),
                     ("external_id", "TEXT"), ("doi", "TEXT"), ("title_norm", "TEXT"),
                     ("authors", "TEXT"), ("journal", "TEXT"), ("source_type", "TEXT")):
        cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
    for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                     ("screening_notes", "TEXT"), ("screened_at", "TIMESTAMP")):
        cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
    cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
    return created_chunk_table
