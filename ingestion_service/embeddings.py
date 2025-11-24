# ingestion_service/embeddings.py

from typing import List
from openai import OpenAI

from api_service.config import get_settings, Settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small


def get_embedding_client(settings: Settings | None = None) -> OpenAI:
    if settings is None:
        settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str], settings: Settings | None = None) -> List[List[float]]:
    """
    Embed a batch of texts using OpenAI embeddings.
    """
    if not texts:
        return []

    settings = settings or get_settings()
    client = get_embedding_client(settings)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # response.data[i].embedding is already a list[float]
    return [item.embedding for item in response.data]
