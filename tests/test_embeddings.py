# tests/test_embeddings.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api_service.config import Settings
from ingestion_service.embeddings import embed_texts, EMBEDDING_MODEL

# --- Fixtures ---

@pytest.fixture
def mock_settings():
    return Settings(openai_api_key="test-key")

@pytest.fixture
def mock_openai_response():
    """Creates a mock response object structure matching OpenAI's."""
    # OpenAI response has a .data attribute, which is a list of objects with an .embedding attribute
    mock_item1 = MagicMock()
    mock_item1.embedding = [0.1, 0.2, 0.3]
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.4, 0.5, 0.6]
    
    mock_resp = MagicMock()
    mock_resp.data = [mock_item1, mock_item2]
    return mock_resp

# --- Tests ---

@pytest.mark.asyncio
async def test_embed_texts_success(mock_settings, mock_openai_response):
    """Test successful embedding generation."""
    
    # Patch the AsyncOpenAI client class where it is imported in the module
    with patch("ingestion_service.embeddings.AsyncOpenAI") as MockClientClass:
        # Configure the mock client instance
        mock_client_instance = MockClientClass.return_value
        mock_client_instance.embeddings.create = AsyncMock(return_value=mock_openai_response)
        
        inputs = ["text1", "text2"]
        results = await embed_texts(inputs, settings=mock_settings)
        
        # Verify result format
        assert results == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Verify API call arguments
        mock_client_instance.embeddings.create.assert_called_once_with(
            model=EMBEDDING_MODEL,
            input=inputs
        )
        
        # Verify client was initialized with correct key
        MockClientClass.assert_called_once_with(api_key="test-key")

@pytest.mark.asyncio
async def test_embed_texts_empty(mock_settings):
    """Test that empty input returns empty list without API call."""
    with patch("ingestion_service.embeddings.AsyncOpenAI") as MockClientClass:
        results = await embed_texts([], settings=mock_settings)
        assert results == []
        MockClientClass.assert_not_called()