from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.qdrant_ids import point_id_for_chunk_id


def _distance(d: str) -> qm.Distance:
    d = d.lower()
    if d == "cosine":
        return qm.Distance.COSINE
    if d == "dot":
        return qm.Distance.DOT
    if d in ("euclid", "euclidean"):
        return qm.Distance.EUCLID
    raise ValueError(f"Unsupported vector distance: {d}")


class QdrantFacade:
    def __init__(self, url: str, api_key: str | None, collection: str, vector_size: int, distance: str):
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(url=url, api_key=api_key, timeout=10.0)

    def health(self) -> bool:
        # qdrant-client API differs between versions; simplest liveness probe is listing collections.
        self.client.get_collections()
        return True

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.vector_size, distance=_distance(self.distance)),
        )

    def retrieve(self, chunk_id: str) -> dict[str, Any] | None:
        pid = point_id_for_chunk_id(chunk_id)
        pts = self.client.retrieve(collection_name=self.collection, ids=[pid], with_payload=True, with_vectors=False)
        if not pts:
            return None
        p0 = pts[0]
        payload = p0.payload or {}
        # Ensure we keep original chunk_id in payload; if missing (legacy), reconstruct it.
        if "chunk_id" not in payload:
            payload["chunk_id"] = chunk_id
        return {"id": str(p0.id), "payload": payload}

    def upsert_points(self, points: list[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        pid = point_id_for_chunk_id(chunk_id)
        self.client.delete(collection_name=self.collection, points_selector=qm.PointIdsList(points=[pid]), wait=True)

    def delete_by_doc_id(self, doc_id: str) -> None:
        flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        self.client.delete(collection_name=self.collection, points_selector=qm.FilterSelector(filter=flt), wait=True)

    def search(self, vector: list[float], flt: qm.Filter | None, limit: int) -> list[qm.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    def get_chunks_by_page(self, doc_id: str, page: int) -> list[dict[str, Any]]:
        """Get all chunks for a specific (doc_id, page)."""
        # Build filter for doc_id and page
        must_conditions = [
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ]
        
        # Page is stored in locator.page (nested object)
        # Qdrant supports nested field access with dot notation
        must_conditions.append(
            qm.FieldCondition(key="locator.page", match=qm.MatchValue(value=page))
        )
        
        flt = qm.Filter(must=must_conditions)
        
        try:
            # Use scroll to get all points (up to reasonable limit)
            result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=1000,  # Should be enough for one page
                with_payload=True,
                with_vectors=False,
            )
            
            chunks = []
            points = result[0]  # First element is list of points
            # Sort by chunk_index if available
            points.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
            
            for point in points:
                payload = point.payload or {}
                # Ensure chunk_id is present
                if "chunk_id" not in payload:
                    # Try to reconstruct from point id (fallback)
                    continue
                chunks.append(payload)
            
            return chunks
        except Exception:
            # If nested field access doesn't work, try alternative approach
            # Fallback: get all chunks for doc_id and filter by page in Python
            try:
                doc_filter = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
                result = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=doc_filter,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                
                chunks = []
                points = result[0]
                for point in points:
                    payload = point.payload or {}
                    locator = payload.get("locator") or {}
                    if isinstance(locator, dict) and locator.get("page") == page:
                        if "chunk_id" not in payload:
                            continue
                        chunks.append(payload)
                
                # Sort by chunk_index
                chunks.sort(key=lambda c: c.get("chunk_index", 0))
                return chunks
            except Exception:
                return []


