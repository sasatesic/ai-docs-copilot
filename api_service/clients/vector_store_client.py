# api_service/clients/vector_store_client.py

from typing import List, Optional, Tuple
from uuid import uuid4
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, FilterSelector
from typing import List, Tuple, Dict, Set
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



    def _collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def list_source_ids(self, limit: int = 1000) -> List[str]:
        """
        Return up to `limit` unique source_id values.
        Works across qdrant-client variants (tuple or object scroll).
        """
        if not self._collection_exists():
            return []

        unique: Set[str] = set()
        offset: Optional[int] = None

        while True:
            res = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )

            # Support both return shapes:
            #  - tuple: (points, next_offset)
            #  - object: res.points, res.next_page_offset
            if isinstance(res, tuple):
                points, offset = res
            else:
                points = getattr(res, "points", None) or []
                offset = getattr(res, "next_page_offset", None)

            if not points:
                break

            for p in points:
                payload = getattr(p, "payload", None) or {}
                sid = payload.get("source_id")
                if sid:
                    unique.add(str(sid))
                    if len(unique) >= limit:
                        return sorted(unique)

            if offset is None:
                break

        return sorted(unique)




    def delete_by_source_id(self, source_id: str) -> int:
        flt = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
        try:
            from qdrant_client.models import FilterSelector  # may not exist in some versions
            selector = FilterSelector(filter=flt)
            res = self._client.delete(
                collection_name=self._collection_name,
                points_selector=selector,
                wait=True,
            )
        except Exception:
            res = self._client.delete(
                collection_name=self._collection_name,
                points_selector=flt,  # older clients accept Filter directly
                wait=True,
            )
        return 1 if getattr(res, "status", None) == "completed" else 0



    def raw_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Dict | None = None,
    ) -> List[Tuple[float, Dict]]:
        """
        Same as search(), but explicitly returns score + payload (preview).
        Uses query_points (works across client versions).
        """
        qdrant_filter = None
        if filter_metadata:
            qdrant_filter = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_metadata.items()
            ])
        resp = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [(pt.score, pt.payload or {}) for pt in resp.points]
