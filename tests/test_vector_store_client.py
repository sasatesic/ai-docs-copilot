# tests/test_vector_store_client.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from api_service.clients.vector_store_client import VectorStoreClient
from api_service.config import Settings
from qdrant_client.models import Filter

# Mock Payloads and Data Structures
MOCK_PAYLOAD_A = {"text": "Qdrant is fast", "source_id": "doc_a"}
MOCK_PAYLOAD_B = {"text": "FastAPI is great", "source_id": "doc_b"}

# Mock objects matching Qdrant responses
class MockQueryPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

class MockPointScroll:
    def __init__(self, payload):
        self.payload = payload

class MockQueryResponse:
    def __init__(self, points):
        self.points = points

class MockCollectionResponse:
    def __init__(self, collections):
        self.collections = collections

# --- Fixtures ---

@pytest.fixture
def mock_settings():
    """Provides a minimal Settings object."""
    return Settings(openai_api_key="dummy", qdrant_host="dummy", qdrant_port=6333)


@pytest.fixture
def mock_qdrant_instance():
    """Provides a fully configured AsyncMock of the Qdrant client instance."""
    mock_client = AsyncMock()
    
    # Mock for get_collections (used by ensure_collection)
    mock_client.get_collections.return_value = MockCollectionResponse(collections=[])

    # Mock for upsert (used by upsert_embeddings)
    mock_client.upsert.return_value = MagicMock()
    
    # Mock for delete (used by delete_by_source_id)
    mock_client.delete.return_value = MagicMock(status="completed")
    
    # Mock for collection existence check
    mock_client.get_collection = AsyncMock() 

    return mock_client


@patch("api_service.clients.vector_store_client.AsyncQdrantClient")
def test_vector_store_client_initialization(MockAsyncQdrantClient, mock_settings):
    """
    Tests that the client initializes AsyncQdrantClient with the correct settings.
    """
    MockAsyncQdrantClient.return_value = MagicMock()
    VectorStoreClient(mock_settings, collection_name="test")
    
    MockAsyncQdrantClient.assert_called_once_with(host="dummy", port=6333)


@pytest.mark.asyncio
@patch("api_service.clients.vector_store_client.AsyncQdrantClient")
async def test_vector_search_returns_correct_format(MockAsyncQdrantClient, mock_settings, mock_qdrant_instance):
    """
    Tests the main search method (pure vector search).
    """
    MockAsyncQdrantClient.return_value = mock_qdrant_instance
    client = VectorStoreClient(mock_settings)

    # Configure mock response for query_points
    points = [
        MockQueryPoint(1, 0.95, MOCK_PAYLOAD_A), 
        MockQueryPoint(2, 0.80, MOCK_PAYLOAD_B),
    ]
    mock_qdrant_instance.query_points.return_value = MockQueryResponse(points)

    query_vector = [0.1] * 1536
    results = await client.search(query_vector, top_k=2)

    # Assertions
    mock_qdrant_instance.query_points.assert_called_once()
    assert len(results) == 2
    assert results[0] == (0.95, MOCK_PAYLOAD_A)
    assert results[1] == (0.80, MOCK_PAYLOAD_B)


@pytest.mark.asyncio
@patch("api_service.clients.vector_store_client.AsyncQdrantClient")
async def test_sparse_search_calculates_synthetic_score(MockAsyncQdrantClient, mock_settings, mock_qdrant_instance):
    """
    Tests the sparse_search method and verifies the synthetic RRF score calculation.
    """
    MockAsyncQdrantClient.return_value = mock_qdrant_instance
    client = VectorStoreClient(mock_settings)

    # Mock Qdrant scroll response (returns points and next offset)
    scroll_points = [
        MockPointScroll(MOCK_PAYLOAD_A), # Rank 1 (index 0)
        MockPointScroll(MOCK_PAYLOAD_B), # Rank 2 (index 1)
    ]
    # Qdrant scroll returns a tuple: (points, next_offset)
    mock_qdrant_instance.scroll.return_value = (scroll_points, None)

    query_text = "test keyword"
    top_k = 2
    results = await client.sparse_search(query_text, top_k=top_k)

    # Assertions
    mock_qdrant_instance.scroll.assert_called_once()
    assert len(results) == 2
    
    # Check synthetic score calculation: 1.0 - (index / (top_k * 2.0))
    # Rank 1 (index 0): 1.0 - (0 / 4.0) = 1.0
    # Rank 2 (index 1): 1.0 - (1 / 4.0) = 0.75
    
    assert results[0][0] == 1.0 
    assert results[0][1] == MOCK_PAYLOAD_A
    assert results[1][0] == 0.75
    assert results[1][1] == MOCK_PAYLOAD_B


@pytest.mark.asyncio
@patch("api_service.clients.vector_store_client.AsyncQdrantClient")
async def test_list_source_ids_and_delete(MockAsyncQdrantClient, mock_settings, mock_qdrant_instance):
    """
    Tests list_source_ids and delete_by_source_id methods.
    """
    MockAsyncQdrantClient.return_value = mock_qdrant_instance
    client = VectorStoreClient(mock_settings)

    # 1. Mock for list_source_ids (scroll)
    mock_qdrant_instance.get_collection.return_value = MagicMock() # Mock collection exists check
    scroll_points = [
        MockPointScroll({"source_id": "doc_a"}),
        MockPointScroll({"source_id": "doc_b"}),
        MockPointScroll({"source_id": "doc_a"}), # Duplicate
    ]
    # Qdrant scroll returns a tuple: (points, next_offset)
    mock_qdrant_instance.scroll.return_value = (scroll_points, None)

    source_ids = await client.list_source_ids()

    assert source_ids == ["doc_a", "doc_b"]

    # 2. Test delete_by_source_id
    deleted = await client.delete_by_source_id("doc_b")
    
    assert deleted == 1 # Status completed returns 1
    mock_qdrant_instance.delete.assert_called_once()
    
    # Verify the filter in the delete call
    args, kwargs = mock_qdrant_instance.delete.call_args
    # Check positional arguments for the points_selector
    delete_filter = args[1] if len(args) > 1 else kwargs.get('points_selector')
    
    # Extract the key/value from the filter structure
    assert delete_filter.filter.must[0].key == "source_id"
    assert delete_filter.filter.must[0].match.value == "doc_b"