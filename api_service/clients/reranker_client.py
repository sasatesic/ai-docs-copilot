# api_service/clients/reranker_client.py

from typing import List, Tuple
import cohere
from cohere import AsyncClient

from api_service.config import Settings


class RerankerClient:
    """
    Client wrapper for Cohere's re-ranking model (Rerank v3).
    """
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # The cohere client handles async/await automatically when using AsyncClient
        self._client = AsyncClient(api_key=settings.cohere_api_key)
        # Use a high-quality model for best results
        self._model = "rerank-english-v3.0" 

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[float, dict]],
        top_n: int = 5,
    ) -> List[Tuple[float, dict]]:
        """
        Re-ranks retrieved documents based on the query, keeping the original payload.

        Args:
            query: The user's question.
            documents: A list of (score, payload) tuples from the vector search.
            top_n: The number of top documents to keep after re-ranking.

        Returns:
            A new list of (re-rank_score, payload) tuples.
        """
        if not self._settings.cohere_api_key:
            # Fallback to simple vector score if API key is missing
            return documents[:top_n]
        
        # 1. Extract just the document text for the Cohere API
        texts = [d[1].get("text", "") for d in documents]
        
        # 2. Await the Cohere API call
        response = await self._client.rerank(
            query=query,
            documents=texts,
            model=self._model,
            top_n=top_n
        )
        
        # 3. Map the re-ranked results back to the original payload
        reranked_results = []
        for rank in response.results:
            # Cohere returns the index of the original document
            original_index = rank.index 
            
            # The payload contains all the necessary metadata
            original_payload = documents[original_index][1]
            
            # Use the new re-rank score
            new_score = rank.relevance_score 
            
            reranked_results.append((new_score, original_payload))
            
        return reranked_results