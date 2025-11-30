# api_service/services/rag.py

import json
from typing import List, Tuple, Dict, Any, AsyncGenerator, Optional
from collections import defaultdict

from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.clients.reranker_client import RerankerClient
from api_service.models.ask import RAGSource, AskResponse
from api_service.config import Settings
from ingestion_service.embeddings import embed_texts

SEARCH_K = 20  # Retrieve more documents from vector search
RERANK_N = 5   # Select the top N relevant documents for the LLM context
RRF_K = 60     # A constant for Reciprocal Rank Fusion


def build_context_from_hits(
    hits: List[Tuple[float, dict]],
    max_chars: int = 2000,
) -> Tuple[str, List[RAGSource]]:
    """
    Turn Qdrant hits into a context string + list of RAGSource objects.
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


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[float, Dict[str, Any]]]],
    k: int = RRF_K
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Combines multiple ranked lists (e.g., vector and sparse search) using RRF.
    """
    fused_scores = defaultdict(float)
    document_map = {}

    for ranked_list in ranked_lists:
        for rank, (score, payload) in enumerate(ranked_list, start=1):
            doc_key = payload.get("source_file", "") + payload.get("text", "")
            document_map[doc_key] = payload
            fused_scores[doc_key] += 1.0 / (k + rank)

    final_results = []
    for doc_key, score in fused_scores.items():
        final_results.append((score, document_map[doc_key]))

    final_results.sort(key=lambda x: x[0], reverse=True)
    return final_results


async def answer_with_rag(
    question: str,
    llm: LLMClient,
    vector_store: VectorStoreClient,
    reranker: RerankerClient,
    settings: Settings,
    top_k: int = RERANK_N,
    source_id: str | None = None,
) -> AskResponse:
    """
    Full RAG pipeline (Non-Streaming).
    """
    [query_embedding] = await embed_texts([question], settings=settings)
    filter_metadata = {"source_id": source_id} if source_id else None

    dense_hits = await vector_store.search(
        query_vector=query_embedding,
        top_k=SEARCH_K,
        filter_metadata=filter_metadata
    )
    sparse_hits = await vector_store.sparse_search(
        query_text=question,
        top_k=SEARCH_K,
        filter_metadata=filter_metadata
    )

    hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=RRF_K)

    if not hits:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. No documents found."},
            {"role": "user", "content": question},
        ]
        answer = await llm.chat(messages, max_tokens=512)
        return AskResponse(answer=answer, sources=[], used_rag=False)

    final_hits = await reranker.rerank(query=question, documents=hits, top_n=top_k)
    context, sources = build_context_from_hits(final_hits)

    system_prompt = (
        "You are an AI assistant that answers questions using the provided context.\n"
        "Use ONLY the context to answer. If the context does not contain the answer, say "
        "'I don't know based on the provided documents.'\n"
        "Cite relevant points in your own words; do not invent sources."
    )
    user_prompt = f"Question:\n{question}\n\nContext:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = await llm.chat(messages, max_tokens=512)
    return AskResponse(answer=answer, sources=sources, used_rag=True)


async def stream_answer_with_rag(
    question: str,
    llm: LLMClient,
    vector_store: VectorStoreClient,
    reranker: RerankerClient,
    settings: Settings,
    top_k: int = RERANK_N,
    source_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming RAG pipeline. Yields NDJSON lines.
    Line 1: Sources (JSON)
    Lines 2+: Content tokens (JSON)
    """
    # 1. Retrieval (Identical to answer_with_rag)
    [query_embedding] = await embed_texts([question], settings=settings)
    filter_metadata = {"source_id": source_id} if source_id else None

    dense_hits = await vector_store.search(
        query_vector=query_embedding,
        top_k=SEARCH_K,
        filter_metadata=filter_metadata
    )
    sparse_hits = await vector_store.sparse_search(
        query_text=question,
        top_k=SEARCH_K,
        filter_metadata=filter_metadata
    )

    hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=RRF_K)

    # Handle no hits
    if not hits:
        # Yield empty sources
        yield json.dumps({"type": "sources", "data": []}) + "\n"
        
        # Stream fallback answer
        messages = [
            {"role": "system", "content": "You are a helpful assistant. No documents found."},
            {"role": "user", "content": question},
        ]
        async for token in llm.stream_chat(messages, max_tokens=512):
            yield json.dumps({"type": "content", "data": token}) + "\n"
        return

    # 2. Rerank & Build Context
    final_hits = await reranker.rerank(query=question, documents=hits, top_n=top_k)
    context, sources = build_context_from_hits(final_hits)

    # 3. Yield Sources FIRST
    if sources:
        sources_data = [s.model_dump() for s in sources]
        yield json.dumps({"type": "sources", "data": sources_data}) + "\n"

    # 4. Stream LLM Response
    system_prompt = (
        "You are an AI assistant that answers questions using the provided context.\n"
        "Use ONLY the context to answer. If the context does not contain the answer, say "
        "'I don't know based on the provided documents.'\n"
        "Cite relevant points in your own words; do not invent sources."
    )
    user_prompt = f"Question:\n{question}\n\nContext:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    async for token in llm.stream_chat(messages, max_tokens=512):
        yield json.dumps({"type": "content", "data": token}) + "\n"