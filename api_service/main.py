from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
import time
import uuid

from api_service.config import Settings, get_settings
from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.models.ask import AskRequest, AskResponse
from api_service.services.rag import answer_with_rag # now an async function
from api_service.models.ask import AskRequest, AskResponse, SearchRequest, SearchResponse, SearchHit
from ingestion_service.chunking import chunk_text
from ingestion_service.embeddings import embed_texts, EMBEDDING_DIM # now an async function


app = FastAPI(title="AI Docs Copilot API")

# Middleware remains async
@app.middleware("http")
async def add_request_id_and_timing(request, call_next):
    start = time.time()
    request_id = str(uuid.uuid4())

    response = await call_next(request) # Await the next call

    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    return response


# ---- Dependency factories ----

# These remain SYNCHRONOUS as they only instantiate objects, no I/O is performed.
def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return LLMClient(settings)


def get_vector_store(
    settings: Settings = Depends(get_settings),
) -> VectorStoreClient:
    return VectorStoreClient(settings, collection_name="docs")


# ---- Endpoints ----


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    # This is an easy place to miss, but async is necessary for consistent application
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.openai_model,
    }


@app.post("/debug/llm")
async def debug_llm(
    prompt: str,
    llm: LLMClient = Depends(get_llm_client),
):
    """
    Simple endpoint to verify LLM client works.
    It just echoes the prompt with a short answer.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    answer = await llm.chat(messages, max_tokens=128)
    return {"answer": answer}

@app.post("/ask", response_model=AskResponse)
async def ask_docs_copilot(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    vector_store: VectorStoreClient = Depends(get_vector_store),
):
    """
    Main RAG endpoint:
    - embeds question
    - retrieves context from Qdrant
    - asks LLM to answer using that context
    """
    try:
        resp = await answer_with_rag(
            question=body.question,
            llm=llm,
            vector_store=vector_store,
            settings=settings,
        )
    except Exception as e:
        # Simple error surface for now
        raise HTTPException(
            status_code=500,
            detail=f"RAG error: {type(e).__name__}: {e}",
        )

    return resp

@app.get("/documents")
async def list_documents(vector_store: VectorStoreClient = Depends(get_vector_store)):
    try:
        items = await vector_store.list_source_ids()
        return {"source_ids": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List error: {type(e).__name__}: {e}")

@app.delete("/documents/{source_id}")
async def delete_document(
    source_id: str,
    vector_store: VectorStoreClient = Depends(get_vector_store),
):
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
    # This reads the file asynchronously, but we keep the main function async
    name = (file.filename or "").strip()
    if not name.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md or .txt files are supported.")

    raw = await file.read() # Await is already here for FastAPI's UploadFile
    try:
        text = raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text.")

    chunks = chunk_text(text)
    if not chunks:
        return {"ingested_chunks": 0, "source_id": source_id or name, "filename": name}

    # embed + upsert
    embeddings = await embed_texts(chunks, settings=settings)
    await vector_store.ensure_collection(vector_size=EMBEDDING_DIM)

    sid = source_id or name  # default source_id = filename
    metadatas = [
        {"source_file": name, "chunk_index": i, "source_id": sid}
        for i in range(len(chunks))
    ]
    await vector_store.upsert_embeddings(embeddings, chunks, metadatas)

    return {"ingested_chunks": len(chunks), "source_id": sid, "filename": name}