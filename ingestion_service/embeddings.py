# ingestion_service/embeddings.py

from typing import List
# CHANGE: Use AsyncOpenAI for all embedding operations
from openai import AsyncOpenAI 

from api_service.config import get_settings, Settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  
# ... (rest of globals)


# The client factory now returns the async client
def get_embedding_client(settings: Settings | None = None) -> AsyncOpenAI: 
    if settings is None:
        settings = get_settings()
    # CHANGE: Instantiate AsyncOpenAI
    return AsyncOpenAI(api_key=settings.openai_api_key) 


# Make the function asynchronous
async def embed_texts(texts: List[str], settings: Settings | None = None) -> List[List[float]]:
    """
    Embed a batch of texts using OpenAI embeddings asynchronously.
    """
    if not texts:
        return []

    settings = settings or get_settings()
    client = get_embedding_client(settings)

    # Await the API call
    response = await client.embeddings.create( 
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]