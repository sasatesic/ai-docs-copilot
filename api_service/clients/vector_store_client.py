# api_service/clients/vector_store_client.py

from typing import List, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


from api_service.config import Settings


class VectorStoreClient:
    """
    Wrapper around Qdrant for storing and querying document embeddings.
    """

    def __init__(self, settings: Settings, collection_name: str = "docs") -> None:
        self._settings = settings
        self._collection_name = collection_name
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    def ensure_collection(self, vector_size: int) -> None:
        """
        Create the collection if it doesn't exist.
        """
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        """
        Upsert a batch of embeddings + their raw text + metadata.
        """
        points = []
        for emb, text, meta in zip(embeddings, texts, metadatas):
            payload = {"text": text, **meta}
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=emb,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    def search(
    self,
    query_vector: List[float],
    top_k: int = 5,
    filter_metadata: dict | None = None,
) -> List[Tuple[float, dict]]:
        """
        Search by vector. Optionally filter by simple metadata equality.
        Returns (score, payload) tuples.
        """
        qdrant_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            qdrant_filter = Filter(must=conditions)

        # Use query_points instead of search
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,  # Note: 'query' instead of 'query_vector'
            limit=top_k,
            query_filter=qdrant_filter,
        )

        results: List[Tuple[float, dict]] = []
        for point in response.points:
            results.append((point.score, point.payload or {}))

        return results
