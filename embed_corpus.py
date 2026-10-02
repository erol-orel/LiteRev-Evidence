import logging
import os
import sys

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed-corpus")

# ─── Configuration ───────────────────────────────────────────────────────────
# Un seul chargeur pour tout le dépôt : env_files. Celui-ci lisait six fichiers dans son
# propre ordre, SANS /etc/literev-api.env - le fichier que systemd passe au service -, et
# chargeait la moitié de la liste après avoir déjà exigé DB_URL. Voir env_files.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_files import CANONICAL, load_env  # noqa: E402

load_env()

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# text-embedding-3-small : max 8192 tokens
# 1 token ≈ 3–4 chars en anglais, mais certains chunks contiennent du HTML/XML dense
# On tronque à 7500 tokens via tiktoken (fallback : 20 000 chars)
MAX_TOKENS = 7_500
MAX_CHARS_FALLBACK = 20_000

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # encodage utilisé par text-embedding-3-small
    _USE_TIKTOKEN = True
except ImportError:
    _USE_TIKTOKEN = False
    logger.warning("tiktoken non disponible - troncature par caractères (moins précise)")


def truncate_text(text_to_embed: str) -> str:
    """Tronque le texte à MAX_TOKENS tokens (ou MAX_CHARS_FALLBACK chars si tiktoken absent)."""
    cleaned = text_to_embed.replace("\n", " ").strip()
    if _USE_TIKTOKEN:
        tokens = _enc.encode(cleaned)
        if len(tokens) > MAX_TOKENS:
            logger.warning(f"Chunk tronqué : {len(tokens)} tokens → {MAX_TOKENS} tokens")
            cleaned = _enc.decode(tokens[:MAX_TOKENS])
    else:
        if len(cleaned) > MAX_CHARS_FALLBACK:
            logger.warning(f"Chunk tronqué : {len(cleaned)} chars → {MAX_CHARS_FALLBACK} chars")
            cleaned = cleaned[:MAX_CHARS_FALLBACK]
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
            f"  1. Le fichier de configuration du service : {CANONICAL}\n"
            "  2. Inline : OPENAI_API_KEY=sk-... python3 embed_corpus.py --project gesica\n"
            "  3. Variable d'environnement : export OPENAI_API_KEY=sk-...\n"
            "Les fichiers lus, dans l'ordre : python3 scripts/env_audit.py"
        )
        return

    try:
        # Via llm_usage : ce script embède le corpus entier, donc il DÉPENSE. Le laisser
        # hors comptabilité rendrait la table trompeuse (« le worker est tout le coût »).
        from llm_usage import MeteredOpenAI as OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, purpose="embed_corpus")
    except ImportError:
        logger.error("Le package 'openai' est requis. Installez-le avec pip.")
        return

    engine = create_engine(DB_URL, pool_pre_ping=True)
    import llm_usage as _llm_usage
    _llm_usage.configure(engine)          # sinon les appels ci-dessous ne sont pas comptés

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
            f" - chunks {i}–{i + len(batch) - 1}"
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
        f"Terminé - {total_ok} embeddings générés, {total_err} erreurs."
    )


if __name__ == "__main__":
    main()
