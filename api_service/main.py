from fastapi import FastAPI, Depends, HTTPException
import time
import uuid

from api_service.config import Settings, get_settings
from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.models.ask import AskRequest, AskResponse
from api_service.services.rag import answer_with_rag


app = FastAPI(title="AI Docs Copilot API")


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


def get_vector_store(
    settings: Settings = Depends(get_settings),
) -> VectorStoreClient:
    # collection name is fixed for now; later we can parametrize
    return VectorStoreClient(settings, collection_name="docs")


# ---- Endpoints ----


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)):
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
    answer = llm.chat(messages, max_tokens=128)
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
        resp = answer_with_rag(
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
