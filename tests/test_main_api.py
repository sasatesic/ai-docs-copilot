# tests/test_main_api.py

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from api_service.main import app

# --- Mock Data ---

MOCK_ASK_RESPONSE = {
    "answer": "RAG is successfully working.",
    "sources": [{"score": 0.99, "text": "source text", "source_id": "test_doc"}],
    "used_rag": True,
}

MOCK_EMBEDDING = [0.1] * 1536 

# --- Fixtures ---

@pytest.fixture
async def async_client():
    """Provides an asynchronous test client for FastAPI using httpx."""
    # Use ASGITransport for httpx >= 0.28.0 compatibility
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

# --- Tests ---

@pytest.mark.asyncio
async def test_health_check(async_client):
    """Tests the basic /health endpoint."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_ask_endpoint_success(async_client):
    """Tests the /ask endpoint and verifies it calls the RAG service correctly."""
    
    # Use context manager to patch and get a reference to the mock
    with patch("api_service.main.answer_with_rag", new_callable=AsyncMock) as mock_service:
        mock_service.return_value = MOCK_ASK_RESPONSE
        
        # 1. Successful request with filter
        response = await async_client.post("/ask", json={"question": "Test question?", "source_id": "test_doc"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["answer"] == "RAG is successfully working."
        assert data["used_rag"] is True
        
        # Verify call
        mock_service.assert_called_once()
        args, kwargs = mock_service.call_args
        assert kwargs['question'] == "Test question?"
        assert kwargs['source_id'] == "test_doc"

@pytest.mark.asyncio
async def test_list_documents_success(async_client):
    """Tests the /documents GET endpoint."""
    with patch("api_service.clients.vector_store_client.VectorStoreClient.list_source_ids", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = ["doc_a", "doc_b"]
        
        response = await async_client.get("/documents")
        assert response.status_code == 200
        assert response.json()["source_ids"] == ["doc_a", "doc_b"]

@pytest.mark.asyncio
async def test_delete_document_success(async_client):
    """Tests the /documents/{source_id} DELETE endpoint."""
    with patch("api_service.clients.vector_store_client.VectorStoreClient.delete_by_source_id", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = 1
        
        response = await async_client.delete("/documents/test_doc_id")
        assert response.status_code == 200
        assert response.json()["deleted"] is True
        assert response.json()["source_id"] == "test_doc_id"
        
        mock_delete.assert_called_once_with("test_doc_id")

@pytest.mark.asyncio
async def test_upload_document_success(async_client):
    """Tests the /documents POST endpoint for file upload."""
    
    # We patch multiple dependencies
    with patch("api_service.main.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("api_service.main.VectorStoreClient") as MockVectorStoreClient:
        
        # Configure mocks
        mock_embed.return_value = [MOCK_EMBEDDING]
        
        mock_client_instance = MockVectorStoreClient.return_value
        mock_client_instance.ensure_collection = AsyncMock()
        mock_client_instance.upsert_embeddings = AsyncMock()
        
        file_content = "This is a document to upload." * 10 

        response = await async_client.post(
            "/documents",
            params={"source_id": "test_upload"},
            files={"file": ("new_doc.md", file_content.encode("utf-8"), "text/markdown")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["ingested_chunks"] > 0
        assert data["source_id"] == "test_upload"
        
        # Verify interactions
        mock_client_instance.ensure_collection.assert_called_once()
        mock_client_instance.upsert_embeddings.assert_called_once()

@pytest.mark.asyncio
async def test_upload_document_unsupported_file(async_client):
    """Tests that the /documents POST endpoint rejects unsupported file types."""
    # We mock embed_texts just in case logic slips through, though it shouldn't
    with patch("api_service.main.embed_texts", new_callable=AsyncMock):
        
        response = await async_client.post(
            "/documents",
            files={"file": ("image.jpg", b"binary data", "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "Only .md or .txt files are supported" in response.json()["detail"]