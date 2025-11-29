# api_service/services/rag.py

from typing import List, Tuple, Optional # Added Optional
from pydantic import BaseModel # Added BaseModel if not imported

from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
# NEW: Import RerankerClient
from api_service.clients.reranker_client import RerankerClient 
from api_service.models.ask import RAGSource, AskResponse
from api_service.config import Settings
from ingestion_service.embeddings import embed_texts  # reuse embeddings from ingestion

# Constants for retrieval and re-ranking
SEARCH_K = 20 # Retrieve more documents from vector search
RERANK_N = 5  # Select the top N relevant documents for the LLM context


def build_context_from_hits(
    hits: List[Tuple[float, dict]],
    max_chars: int = 2000,
) -> Tuple[str, List[RAGSource]]:
    """
    Turn Qdrant/Reranker hits into a context string + list of RAGSource objects.
    """
    context_parts: List[str] = []
    sources: List[RAGSource] = []

    current_len = 0
    for score, payload in hits:
        text = payload.get("text", "")
        if not text:
            continue

        addition = text + "\n\n"
        if current_len + len(addition) > max_chars:
            break

        current_len += len(addition)
        context_parts.append(addition)

        sources.append(
            RAGSource(
                score=score,
                text=text,
                source_file=payload.get("source_file"),
                chunk_index=payload.get("chunk_index"),
                source_id=payload.get("source_id"),
            )
        )

    context = "\n".join(context_parts).strip()
    return context, sources


async def answer_with_rag(
    question: str,
    llm: LLMClient,
    vector_store: VectorStoreClient,
    # NEW: RerankerClient dependency
    reranker: RerankerClient,
    settings: Settings,
    top_k: int = RERANK_N, # Use the RERANK_N constant here for context size
    source_id: str | None = None,
) -> AskResponse:
    """
    Full RAG pipeline with **Re-ranking**:
    - embed question
    - search in Qdrant for a large set (SEARCH_K)
    - re-rank the search results to find the best subset (RERANK_N)
    - build context
    - call LLM with question + context
    """
    # 1) Embed the question
    [query_embedding] = await embed_texts([question], settings=settings)

    # Prepare metadata filter for Qdrant
    filter_metadata = {"source_id": source_id} if source_id else None

    # 2) Initial Search in Qdrant (Retrieve a larger set of documents)
    hits = await vector_store.search(
        query_vector=query_embedding, 
        top_k=SEARCH_K, # Use SEARCH_K here
        filter_metadata=filter_metadata
    )

    if not hits:
        # No RAG context – fall back to direct LLM answer
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. No documents were found, "
                           "so answer from your general knowledge.",
            },
            {"role": "user", "content": question},
        ]
        answer = await llm.chat(messages, max_tokens=512)
        return AskResponse(answer=answer, sources=[], used_rag=False)

    # 2.5) Rerank the initial hits to find the most relevant N
    # NEW STEP: Await the rerank client call
    final_hits = await reranker.rerank(
        query=question, 
        documents=hits, 
        top_n=RERANK_N
    )

    # 3) Build context + sources (using the final, re-ranked hits)
    context, sources = build_context_from_hits(final_hits)

    # 4) Ask LLM using the context
    system_prompt = (
        "You are an AI assistant that answers questions using the provided context.\n"
        "Use ONLY the context to answer. If the context does not contain the answer, say "
        "'I don't know based on the provided documents.'\n"
        "Cite relevant points in your own words; do not invent sources."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = await llm.chat(messages, max_tokens=512)

    return AskResponse(answer=answer, sources=sources, used_rag=True)