# api_service/services/rag.py

from typing import List, Tuple

from api_service.clients.llm_client import LLMClient
from api_service.clients.vector_store_client import VectorStoreClient
from api_service.models.ask import RAGSource, AskResponse
from api_service.config import Settings
from ingestion_service.embeddings import embed_texts  # reuse embeddings from ingestion


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


def answer_with_rag(
    question: str,
    llm: LLMClient,
    vector_store: VectorStoreClient,
    settings: Settings,
    top_k: int = 5,
) -> AskResponse:
    """
    Full RAG pipeline:
    - embed question
    - search in Qdrant
    - build context
    - call LLM with question + context
    """
    # 1) Embed the question
    [query_embedding] = embed_texts([question], settings=settings)

    # 2) Search in Qdrant
    hits = vector_store.search(query_vector=query_embedding, top_k=top_k)

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
        answer = llm.chat(messages, max_tokens=512)
        return AskResponse(answer=answer, sources=[], used_rag=False)

    # 3) Build context + sources
    context, sources = build_context_from_hits(hits)

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

    answer = llm.chat(messages, max_tokens=512)

    return AskResponse(answer=answer, sources=sources, used_rag=True)
