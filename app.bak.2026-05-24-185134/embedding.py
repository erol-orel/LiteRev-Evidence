import logging

from sentence_transformers import SentenceTransformer

from app.config import MODEL_NAME

logger = logging.getLogger("literev-api")

model = SentenceTransformer(MODEL_NAME)


def embed_text(text_value: str) -> list[float]:
    return model.encode([text_value], normalize_embeddings=True)[0].tolist()
