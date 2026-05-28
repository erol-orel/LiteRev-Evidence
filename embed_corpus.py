import logging
import os
import json
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed-corpus")

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def generate_embedding(client, text_to_embed: str) -> list[float]:
    """Génère un embedding 1536-dim via l'API OpenAI."""
    # Nettoyer le texte pour éviter les sauts de ligne excessifs
    cleaned_text = text_to_embed.replace("\n", " ").strip()
    if not cleaned_text:
        return [0.0] * 1536
        
    response = client.embeddings.create(
        input=[cleaned_text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def main():
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY est requise pour générer les embeddings.")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logger.error("Le package 'openai' est requis. Installez-le avec pip.")
        return

    engine = create_engine(DB_URL, pool_pre_ping=True)

    # 1. Récupérer les chunks sans embedding
    sql_fetch = text("""
        SELECT id, content 
        FROM document_chunk 
        WHERE embedding IS NULL
        ORDER BY id ASC
    """)
    
    with engine.connect() as conn:
        chunks = conn.execute(sql_fetch).mappings().all()

    if not chunks:
        logger.info("Tous les chunks ont déjà un embedding. Rien à faire.")
        return

    logger.info(f"Trouvé {len(chunks)} chunks sans embedding à traiter.")

    # 2. Générer et mettre à jour par lots (batch)
    batch_size = 50
    sql_update = text("""
        UPDATE document_chunk 
        SET embedding = CAST(:embedding AS vector) 
        WHERE id = :id
    """)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        logger.info(f"Traitement du lot {i//batch_size + 1} ({i} à {i+len(batch)})...")
        
        updates = []
        for r in batch:
            try:
                # Concaténer un peu de contexte pour améliorer l'embedding si nécessaire
                emb = generate_embedding(client, r["content"])
                updates.append({
                    "id": r["id"],
                    "embedding": str(emb) # psycopg3 attend la string de liste pour le cast vector
                })
            except Exception as e:
                logger.error(f"Erreur lors de la génération de l'embedding pour le chunk {r['id']}: {e}")

        if updates:
            with engine.begin() as conn:
                conn.execute(sql_update, updates)
            logger.info(f"Mis à jour {len(updates)} chunks avec succès.")

    logger.info("Génération des embeddings terminée.")

if __name__ == "__main__":
    main()
