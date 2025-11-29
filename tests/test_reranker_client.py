# tests/test_reranker_client.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from api_service.clients.reranker_client import RerankerClient
from api_service.config import Settings
from typing import Dict, Any, List, Tuple

# Mock Payloads
MOCK_DOCUMENTS: List[Tuple[float, Dict[str, Any]]] = [
    (0.9, {"text": "Document about FastAPI speed.", "source_id": "doc_a"}),
    (0.7, {"text": "Document about database clusters.", "source_id": "doc_b"}),
    (0.5, {"text": "Document about Python chunking.", "source_id": "doc_c"}),
]

# Define a mock Cohere Rerank result structure
class MockRerankResult:
    def __init__(self, index, score):
        self.index = index
        self.relevance_score = score

class MockRerankResponse:
    def __init__(self, results):
        self.results = results

@pytest.fixture
def mock_settings():
    """Provides a minimal Settings object needed for client initialization."""
    return Settings(openai_api_key="dummy-key", cohere_api_key="test-cohere-key")


@pytest.fixture
def mock_cohere_client():
    """Mocks the internal Cohere AsyncClient instance."""
    mock_client = AsyncMock()
    
    # Configure the mock response: push the FastAPI doc (index 0) to the second spot 
    # and the database doc (index 1) to the first spot.
    mock_response = MockRerankResponse(
        results=[
            # Result 1: Original document at index 1 (Doc B) is now highest score
            MockRerankResult(index=1, score=0.98), 
            # Result 2: Original document at index 0 (Doc A) is second highest score
            MockRerankResult(index=0, score=0.95),
            # Result 3: Original document at index 2 (Doc C) is lowest score
            MockRerankResult(index=2, score=0.90),
        ]
    )
    mock_client.rerank.return_value = mock_response
    return mock_client


# Patch the external Cohere AsyncClient class itself
@patch("api_service.clients.reranker_client.AsyncClient")
def test_rerank_client_rerank_call_and_mapping(MockAsyncClient, mock_settings, mock_cohere_client):
    """
    Tests that the rerank method correctly calls Cohere API and maps the new score 
    back to the original payload.
    """
    # 1. Inject our mock client instance
    MockAsyncClient.return_value = mock_cohere_client

    # 2. Instantiate the RerankerClient
    client = RerankerClient(mock_settings)
    
    test_query = "How can I improve database performance?"
    test_top_n = 3
    
    # 3. Run the async method
    reranked_results = asyncio.run(client.rerank(
        query=test_query,
        documents=MOCK_DOCUMENTS,
        top_n=test_top_n
    ))

    # 4. Assertions

    # Verify the underlying API method was called once
    mock_cohere_client.rerank.assert_called_once()
    
    # Verify arguments passed to the API call
    args, kwargs = mock_cohere_client.rerank.call_args
    
    # Expected texts passed to Cohere (order matters here!)
    expected_texts = [d[1]["text"] for d in MOCK_DOCUMENTS] 
    
    assert kwargs["query"] == test_query
    assert kwargs["documents"] == expected_texts
    assert kwargs["top_n"] == test_top_n
    
    # 5. Verify the output mapping
    assert len(reranked_results) == 3
    
    # Result 1 should be Doc B (index 1) with score 0.98
    assert reranked_results[0][0] == 0.98
    assert reranked_results[0][1]["source_id"] == "doc_b"
    
    # Result 2 should be Doc A (index 0) with score 0.95
    assert reranked_results[1][0] == 0.95
    assert reranked_results[1][1]["source_id"] == "doc_a"

def test_rerank_client_no_key_fallback(mock_settings):
    """
    Tests that the client falls back to vector score and limits top_n if no COHERE_API_KEY is provided.
    """
    # Set settings with no API key
    mock_settings.cohere_api_key = None 
    client = RerankerClient(mock_settings)

    test_query = "Fallback query"
    test_top_n = 2
    
    # Run the async method
    reranked_results = asyncio.run(client.rerank(
        query=test_query,
        documents=MOCK_DOCUMENTS, # 3 documents in this list
        top_n=test_top_n
    ))

    # Assertions
    assert len(reranked_results) == test_top_n
    
    # The output should just be the first two elements of the input list (sorted by vector score implicitly)
    assert reranked_results[0][1]["source_id"] == "doc_a"
    assert reranked_results[1][1]["source_id"] == "doc_b"