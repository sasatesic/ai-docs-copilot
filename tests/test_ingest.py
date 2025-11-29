# tests/test_ingest.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ingestion_service.ingest import main 
from ingestion_service.embeddings import EMBEDDING_DIM 
from api_service.config import Settings 

# --- Constants ---
MOCK_DOCS = [
    {"text": "text1", "meta": {"source_id": "doc1"}},
    {"text": "text2", "meta": {"source_id": "doc2"}},
]
MOCK_TEXTS = ["text1", "text2"]
MOCK_METADATAS = [{"source_id": "doc1"}, {"source_id": "doc2"}]
MOCK_EMBEDDINGS = [[0.1] * 1536, [0.2] * 1536]

# --- Fixtures ---

@pytest.fixture
def mock_settings_obj():
    """A controlled Settings object to use across patches."""
    return Settings(openai_api_key="test-key", qdrant_host="test-host")

# --- Tests ---

@pytest.mark.asyncio
async def test_ingestion_pipeline_success(mock_settings_obj):
    """
    Tests the main ingestion pipeline logic.
    """
    # Patch dependencies
    # 1. Patch get_settings to return our specific object
    with patch("ingestion_service.ingest.get_settings", return_value=mock_settings_obj), \
         patch("ingestion_service.ingest.load_documents", return_value=MOCK_DOCS) as mock_load, \
         patch("ingestion_service.ingest.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("ingestion_service.ingest.VectorStoreClient") as MockVectorStoreClient:

        # Configure mocks
        mock_embed.return_value = MOCK_EMBEDDINGS
        
        # Mock vector store instance methods
        mock_vs_instance = MockVectorStoreClient.return_value
        mock_vs_instance.ensure_collection = AsyncMock()
        mock_vs_instance.upsert_embeddings = AsyncMock()

        # Run main
        await main()

        # Assertions
        mock_load.assert_called_once()
        
        # Verify embed_texts was called with the specific settings object
        mock_embed.assert_called_once_with(MOCK_TEXTS, settings=mock_settings_obj)
        
        # Verify Vector Store interactions
        MockVectorStoreClient.assert_called_once_with(mock_settings_obj, collection_name="docs")
        mock_vs_instance.ensure_collection.assert_called_once_with(vector_size=EMBEDDING_DIM)
        mock_vs_instance.upsert_embeddings.assert_called_once_with(MOCK_EMBEDDINGS, MOCK_TEXTS, MOCK_METADATAS)

@pytest.mark.asyncio
async def test_ingestion_pipeline_no_documents():
    """
    Tests that the pipeline exits early if no documents are found.
    """
    with patch("ingestion_service.ingest.load_documents", return_value=[]) as mock_load, \
         patch("ingestion_service.ingest.embed_texts", new_callable=AsyncMock) as mock_embed:
        
        await main()
        
        mock_load.assert_called_once()
        mock_embed.assert_not_called()