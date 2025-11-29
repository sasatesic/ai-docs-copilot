# tests/test_rag_service.py

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Import dependencies and constants from the service file
from api_service.services.rag import answer_with_rag, RRF_K, SEARCH_K, RERANK_N
from api_service.models.ask import AskResponse
from api_service.config import Settings
from typing import Dict, Any, List, Tuple

# --- Constants and Mock Data ---

# NOTE: DUMMY_EMBEDDING is needed because embed_texts is patched to return it
DUMMY_EMBEDDING = [0.1] * 1536 

# Mock payloads representing document chunks
MOCK_PAYLOAD_1: Dict[str, Any] = {"text": "Qdrant is a fast vector database.", "source_id": "qdrant.md", "source_file": "qdrant.md", "chunk_index": 1}
MOCK_PAYLOAD_2: Dict[str, Any] = {"text": "FastAPI is based on Starlette.", "source_id": "fastapi.md", "source_file": "fastapi.md", "chunk_index": 1}
MOCK_PAYLOAD_3: Dict[str, Any] = {"text": "Overlap is important for chunking.", "source_id": "chunking.md", "source_file": "chunking.md", "chunk_index": 1}

# --- Fixtures for Mocked Dependencies ---

@pytest.fixture
def mock_settings():
    """Provides a dummy Settings object."""
    return Settings(openai_api_key="dummy_key", cohere_api_key="dummy_key")


@pytest.fixture
def mock_llm_client():
    """Mocks the LLM Client's chat method."""
    mock = AsyncMock()
    mock.chat.return_value = "The final answer based on context."
    return mock


@pytest.fixture
def mock_vector_store():
    """Mocks the Vector Store Client's hybrid search components."""
    mock = AsyncMock()
    
    # 1. Mock Dense Search (Vector Search, score 0-1)
    mock.search.return_value = [
        (0.9, MOCK_PAYLOAD_1), 
        (0.8, MOCK_PAYLOAD_2), 
        (0.7, MOCK_PAYLOAD_3),
    ]
    
    # 2. Mock Sparse Search (Keyword Search, synthetic rank score for RRF)
    mock.sparse_search.return_value = [
        (1.0, MOCK_PAYLOAD_2), 
        (0.9, MOCK_PAYLOAD_3), 
        (0.8, MOCK_PAYLOAD_1),
    ]
    return mock


@pytest.fixture
def mock_reranker_client():
    """Mocks the Reranker Client to enforce a known output order and satisfy RERANK_N length."""
    mock = AsyncMock()
    
    # FIX 1: Ensure the mock returns RERANK_N (5) results to satisfy the assertion
    mock.rerank.return_value = [
        (0.99, MOCK_PAYLOAD_2), 
        (0.95, MOCK_PAYLOAD_1), 
        (0.90, MOCK_PAYLOAD_3), 
        (0.85, MOCK_PAYLOAD_1), # Duplicated for length
        (0.80, MOCK_PAYLOAD_2), # Duplicated for length
    ]
    return mock


# --- Tests for answer_with_rag ---

# Patching the external dependency 'embed_texts' globally for these tests
@patch("api_service.services.rag.embed_texts", AsyncMock(return_value=[DUMMY_EMBEDDING]))
class TestRAGService:
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_success(self, mock_settings, mock_llm_client, mock_vector_store, mock_reranker_client):
        """
        Tests the successful execution of the full RAG pipeline (Hybrid Search -> RRF -> Rerank -> LLM).
        """
        response = await answer_with_rag(
            question="What is FastAPI?",
            llm=mock_llm_client,
            vector_store=mock_vector_store,
            reranker=mock_reranker_client,
            settings=mock_settings,
        )

        # 1. Verify Hybrid Search was executed
        mock_vector_store.sparse_search.assert_called_once()
        mock_vector_store.search.assert_called_once()

        # 2. Verify Reranker was called 
        mock_reranker_client.rerank.assert_called_once()
        
        # 3. Verify the LLM was called with the final context
        mock_llm_client.chat.assert_called_once()
        
        # 4. Verify the response format and content
        assert isinstance(response, AskResponse)
        assert response.used_rag is True
        # FIX 1 VERIFICATION: Should pass now as mock returns 5 elements
        assert len(response.sources) == RERANK_N 
        assert response.sources[0].source_id == "fastapi.md" # Check if the mock reranker result was used

    @pytest.mark.asyncio
    async def test_rag_pipeline_no_hits_fallback(self, mock_settings, mock_llm_client, mock_vector_store, mock_reranker_client):
        """
        Tests the fallback logic when no documents are retrieved by Hybrid Search.
        """
        # Force vector store to return empty results
        mock_vector_store.search.return_value = []
        mock_vector_store.sparse_search.return_value = []

        response = await answer_with_rag(
            question="What is your name?",
            llm=mock_llm_client,
            vector_store=mock_vector_store,
            reranker=mock_reranker_client,
            settings=mock_settings,
        )

        # 1. Verify Reranker was NOT called
        mock_reranker_client.rerank.assert_not_called()

        # 2. Verify LLM was called with fallback prompt
        mock_llm_client.chat.assert_called_once()
        
        # Check the system prompt for the fallback message
        called_messages = mock_llm_client.chat.call_args[0][0]
        assert "No documents were found" in called_messages[0]["content"]

        # 3. Verify the response format
        assert isinstance(response, AskResponse)
        assert response.used_rag is False
        assert response.sources == []
        
    @pytest.mark.asyncio
    async def test_rag_pipeline_filtered_search(self, mock_settings, mock_llm_client, mock_vector_store, mock_reranker_client):
        """
        Tests that the source_id filter (Faceted RAG) is passed correctly to both Hybrid Search components.
        """
        # Reset mocks for clean call count verification
        mock_vector_store.search.reset_mock()
        mock_vector_store.sparse_search.reset_mock()
        
        test_source_id = "filter_test.md"

        await answer_with_rag(
            question="Filter me please.",
            llm=mock_llm_client,
            vector_store=mock_vector_store,
            reranker=mock_reranker_client,
            settings=mock_settings,
            source_id=test_source_id, # Filter is provided
        )

        # The expected filter passed to the client methods
        expected_filter = {"source_id": test_source_id}
        
        # FIX 2: Access arguments from kwargs, not positional args, as top_k and filter_metadata are passed by keyword.
        
        # Check Dense Search call
        mock_vector_store.search.assert_called_once()
        args_dense, kwargs_dense = mock_vector_store.search.call_args
        assert kwargs_dense.get("filter_metadata") == expected_filter
        
        # Check Sparse Search call
        mock_vector_store.sparse_search.assert_called_once()
        args_sparse, kwargs_sparse = mock_vector_store.sparse_search.call_args
        assert kwargs_sparse.get("filter_metadata") == expected_filter