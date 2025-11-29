# api_service/services/rag.py

from typing import List, Tuple, Dict, Any
from collections import defaultdict # NEW: defaultdict for RRF

from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.clients.reranker_client import RerankerClient 
from api_service.models.ask import RAGSource, AskResponse
from api_service.config import Settings
from ingestion_service.embeddings import embed_texts  # reuse embeddings from ingestion

SEARCH_K = 20 # Retrieve more documents from vector search
RERANK_N = 5  # Select the top N relevant documents for the LLM context
# NEW: Constant for RRF
RRF_K = 60    # A constant for Reciprocal Rank Fusion


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


# NEW FUNCTION: Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[float, Dict[str, Any]]]],
    k: int = RRF_K
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Combines multiple ranked lists (e.g., vector and sparse search) using RRF.
    """
    fused_scores = defaultdict(float)
    document_map = {} # Maps unique key (e.g., source_id + text) to its payload

    for ranked_list in ranked_lists:
        for rank, (score, payload) in enumerate(ranked_list, start=1):
            
            # Create a unique key for the document payload
            # NOTE: We use the raw text, as chunks are guaranteed to be unique
            doc_key = payload.get("source_file", "") + payload.get("text", "") 
            document_map[doc_key] = payload
            
            # RRF formula: 1 / (k + rank)
            fused_scores[doc_key] += 1.0 / (k + rank)

    # Convert the fused scores back to a list of (score, payload)
    final_results = []
    for doc_key, score in fused_scores.items():
        final_results.append((score, document_map[doc_key]))

    # Sort by the new fused RRF score in descending order
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
    Full RAG pipeline with **Hybrid Search (RRF)** and Re-ranking.
    """
    # 1) Embed the question
    [query_embedding] = await embed_texts([question], settings=settings)

    # Prepare metadata filter for Qdrant
    filter_metadata = {"source_id": source_id} if source_id else None

    # 2) Initial Search (Hybrid RRF)
    # A) Dense Search (Vector)
    dense_hits = await vector_store.search(
        query_vector=query_embedding, 
        top_k=SEARCH_K, 
        filter_metadata=filter_metadata
    )
    
    # B) Sparse Search (Keyword)
    sparse_hits = await vector_store.sparse_search(
        query_text=question, 
        top_k=SEARCH_K, 
        filter_metadata=filter_metadata
    )

    # C) Combine ranks using RRF
    hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=RRF_K)


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