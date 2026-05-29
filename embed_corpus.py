import logging
import os
from pathlib import Path
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed-corpus")

# ─── Chargement de la clé API ─────────────────────────────────────────────────
# Priorité : variable d'environnement > fichier .env > fichier /etc/literev/env
def _load_env_file(path: str) -> None:
    """Charge les variables KEY=VALUE d'un fichier dans os.environ (sans dépendance python-dotenv)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

# Chercher le .env dans le répertoire courant, le répertoire du script, puis /etc/literev/env
for _env_path in [".env", str(Path(__file__).parent / ".env"), "/etc/literev/env", "/opt/literev-api/.env"]:
    _load_env_file(_env_path)

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-emUeiQDcxW3dPn4Qo4WTG9"

# text-embedding-3-small : max 8192 tokens ≈ 32 000 chars
# On tronque à 7000 tokens ≈ 28 000 chars pour rester confortablement sous la limite
MAX_CHARS = 28_000


def truncate_text(text_to_embed: str) -> str:
    """Tronque le texte à MAX_CHARS caractères si nécessaire."""
    cleaned = text_to_embed.replace("\n", " ").strip()
    if len(cleaned) > MAX_CHARS:
        logger.warning(f"Chunk tronqué : {len(cleaned)} chars → {MAX_CHARS} chars")
        return cleaned[:MAX_CHARS]
    return cleaned


def generate_embedding(client, text_to_embed: str) -> list[float]:
    """Génère un embedding 1536-dim via l'API OpenAI (text-embedding-3-small)."""
    cleaned = truncate_text(text_to_embed)
    if not cleaned:
        return [0.0] * 1536
    response = client.embeddings.create(
        input=[cleaned],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def main():
    if not OPENAI_API_KEY:
        logger.error(
            "OPENAI_API_KEY est requise pour générer les embeddings.\n"
            "Solutions (par ordre de priorité) :\n"
            "  1. Variable d'environnement : export OPENAI_API_KEY=sk-...  && python3 embed_corpus.py\n"
            "  2. Fichier /opt/literev-api/.env contenant : OPENAI_API_KEY=sk-...\n"
            "  3. Service systemd : sudo systemctl edit literev-api → [Service] Environment=OPENAI_API_KEY=sk-...\n"
            "  4. Inline : OPENAI_API_KEY=sk-... python3 embed_corpus.py --project gesica"
        )
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

    # 2. Générer et mettre à jour par lots de 50
    batch_size = 50
    sql_update = text("""
        UPDATE document_chunk
        SET embedding = CAST(:embedding AS vector)
        WHERE id = :id
    """)

    total_ok = 0
    total_err = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        logger.info(
            f"Lot {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
            f" — chunks {i}–{i + len(batch) - 1}"
        )

        updates = []
        for r in batch:
            try:
                emb = generate_embedding(client, r["content"] or "")
                updates.append({
                    "id": r["id"],
                    "embedding": str(emb),
                })
            except Exception as e:
                logger.error(f"Erreur embedding chunk {r['id']}: {e}")
                total_err += 1

        if updates:
            with engine.begin() as conn:
                conn.execute(sql_update, updates)
            total_ok += len(updates)
            logger.info(f"  → {len(updates)} embeddings sauvegardés.")

    logger.info(
        f"Terminé — {total_ok} embeddings générés, {total_err} erreurs."
    )


if __name__ == "__main__":
    main()
