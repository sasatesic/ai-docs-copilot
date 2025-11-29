# tests/test_models.py

import pytest
from pydantic import ValidationError

# Import all models from the target file
from api_service.models.ask import (
    RAGSource, AskRequest, AskResponse, 
    SearchRequest, SearchResponse, SearchHit
)

# --- Test Data ---

RAG_SOURCE_DATA = {
    "score": 0.95,
    "text": "sample text",
    "source_file": "doc.pdf",
    "chunk_index": 5,
    "source_id": "doc_123"
}

# --- Tests for RAGSource ---

def test_rag_source_full_data():
    """Test RAGSource with all optional fields provided."""
    source = RAGSource(**RAG_SOURCE_DATA)
    assert source.score == 0.95
    assert source.text == "sample text"
    assert source.source_file == "doc.pdf"

def test_rag_source_minimal_data():
    """Test RAGSource with only required fields (score, text)."""
    minimal_data = {"score": 0.1, "text": "minimal"}
    source = RAGSource(**minimal_data)
    assert source.score == 0.1
    assert source.text == "minimal"
    assert source.source_file is None

def test_rag_source_invalid_score_type():
    """Test that RAGSource fails on invalid score type."""
    with pytest.raises(ValidationError):
        RAGSource(score="high", text="sample text")

# --- Tests for AskRequest ---

def test_ask_request_minimal():
    """Test AskRequest with just the question."""
    request = AskRequest(question="What is RAG?")
    assert request.question == "What is RAG?"
    assert request.source_id is None

def test_ask_request_with_filter():
    """Test AskRequest with optional source_id filter (Faceted RAG)."""
    request = AskRequest(question="What is RAG?", source_id="doc_a")
    assert request.source_id == "doc_a"

# --- Tests for AskResponse ---

def test_ask_response_success():
    """Test AskResponse with valid sources and answer."""
    source = RAGSource(**RAG_SOURCE_DATA)
    response = AskResponse(answer="RAG is good.", sources=[source], used_rag=True)
    assert response.answer == "RAG is good."
    assert len(response.sources) == 1
    assert response.used_rag is True

def test_ask_response_fallback():
    """Test AskResponse for fallback scenario (no sources)."""
    response = AskResponse(answer="I don't know.", sources=[], used_rag=False)
    assert response.used_rag is False

# --- Tests for SearchRequest ---

def test_search_request_defaults():
    """Test SearchRequest defaults."""
    request = SearchRequest(query="find me")
    assert request.query == "find me"
    assert request.top_k == 5
    assert request.source_id is None

def test_search_request_custom():
    """Test SearchRequest with custom values."""
    request = SearchRequest(query="find me", top_k=10, source_id="doc_xyz")
    assert request.top_k == 10
    assert request.source_id == "doc_xyz"

# --- Tests for SearchResponse ---

def test_search_response():
    """Test SearchResponse with mock hits."""
    hit_data = {
        "score": 0.99,
        "text": "hit text",
        "source_file": "doc.md",
        "source_id": "doc_123",
        "chunk_index": 0
    }
    hit = SearchHit(**hit_data)
    response = SearchResponse(hits=[hit])
    assert len(response.hits) == 1
    assert response.hits[0].score == 0.99