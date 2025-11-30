from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid

from api_service.config import Settings, get_settings
from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.clients.reranker_client import RerankerClient
from api_service.models.ask import AskRequest, AskResponse, SearchRequest, SearchResponse, SearchHit
from api_service.services.rag import answer_with_rag, stream_answer_with_rag
from ingestion_service.chunking import chunk_text
from ingestion_service.embeddings import embed_texts, EMBEDDING_DIM


app = FastAPI(title="AI Docs Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_and_timing(request, call_next):
    start = time.time()
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    return response


# ---- Dependency factories ----

def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return LLMClient(settings)

def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreClient:
    return VectorStoreClient(settings, collection_name="docs")

def get_reranker_client(settings: Settings = Depends(get_settings)) -> RerankerClient:
    return RerankerClient(settings)


# ---- Endpoints ----

@app.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.openai_model,
    }

@app.post("/debug/llm")
async def debug_llm(prompt: str, llm: LLMClient = Depends(get_llm_client)):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    answer = await llm.chat(messages, max_tokens=128)
    return {"answer": answer}

@app.post("/ask", response_model=AskResponse)
async def ask_docs_copilot(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    vector_store: VectorStoreClient = Depends(get_vector_store),
    reranker: RerankerClient = Depends(get_reranker_client),
):
    try:
        resp = await answer_with_rag(
            question=body.question,
            llm=llm,
            vector_store=vector_store,
            reranker=reranker,
            settings=settings,
            source_id=body.source_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {type(e).__name__}: {e}")
    return resp

@app.post("/chat_stream")
async def chat_stream_endpoint(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    vector_store: VectorStoreClient = Depends(get_vector_store),
    reranker: RerankerClient = Depends(get_reranker_client),
):
    """
    Streams the RAG response: sources first, then text tokens.
    """
    async def generator():
        async for chunk in stream_answer_with_rag(
            question=body.question,
            llm=llm,
            vector_store=vector_store,
            reranker=reranker,
            settings=settings,
            source_id=body.source_id,
        ):
            yield chunk

    return StreamingResponse(generator(), media_type="application/x-ndjson")

@app.get("/documents")
async def list_documents(vector_store: VectorStoreClient = Depends(get_vector_store)):
    try:
        items = await vector_store.list_source_ids()
        return {"source_ids": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List error: {type(e).__name__}: {e}")

@app.delete("/documents/{source_id}")
async def delete_document(source_id: str, vector_store: VectorStoreClient = Depends(get_vector_store)):
    try:
        deleted = await vector_store.delete_by_source_id(source_id)
        return {"deleted": bool(deleted), "source_id": source_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {type(e).__name__}: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_preview(
    body: SearchRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreClient = Depends(get_vector_store),
):
    try:
        [qvec] = await embed_texts([body.query], settings=settings)
        filt = {"source_id": body.source_id} if body.source_id else None
        results = await vector_store.raw_search(qvec, top_k=body.top_k, filter_metadata=filt)
        hits = [
            SearchHit(
                score=score,
                text=payload.get("text", ""),
                source_file=payload.get("source_file"),
                source_id=payload.get("source_id"),
                chunk_index=payload.get("chunk_index"),
            )
            for score, payload in results
        ]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {type(e).__name__}: {e}")

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    source_id: str | None = None,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreClient = Depends(get_vector_store),
):
    name = (file.filename or "").strip()
    if not name.lower().endswith((".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # We read the file content as bytes for the parsers
    raw = await file.read()
    
    # We import parsers here to avoid circular imports or just for cleanliness
    # In a real app, this logic might be in a service
    from ingestion_service.parsers import parse_pdf, parse_docx, parse_pptx, parse_xlsx
    
    text = ""
    try:
        ext = name.lower().split('.')[-1]
        if ext in ["md", "txt"]:
            text = raw.decode("utf-8")
        elif ext == "pdf":
            text = parse_pdf(raw)
        elif ext == "docx":
            text = parse_docx(raw)
        elif ext == "pptx":
            text = parse_pptx(raw)
        elif ext == "xlsx":
            text = parse_xlsx(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    chunks = chunk_text(text)
    if not chunks:
        return {"ingested_chunks": 0, "source_id": source_id or name, "filename": name}

    embeddings = await embed_texts(chunks, settings=settings)
    await vector_store.ensure_collection(vector_size=EMBEDDING_DIM)

    sid = source_id or name
    metadatas = [
        {"source_file": name, "chunk_index": i, "source_id": sid}
        for i in range(len(chunks))
    ]
    await vector_store.upsert_embeddings(embeddings, chunks, metadatas)

    return {"ingested_chunks": len(chunks), "source_id": sid, "filename": name}