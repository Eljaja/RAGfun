from __future__ import annotations

import logging

from store import QdrantStore, BM25Store
from embedder import Embedder
from reranker import Reranker
from models import ChunkResult

logger = logging.getLogger(__name__)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantStore,
        bm25: BM25Store,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker

    async def search(
        self,
        query: str,
        collection: str,
        *,
        top_k: int = 5,
        strategy: str = "hybrid",
        rerank: bool = True,
        rerank_top_n: int = 5,
    ) -> list[ChunkResult]:
        vector = (await self.embedder.embed([query]))[0]

        if strategy == "vector":
            return await self._vector_search(vector, collection, top_k)
        if strategy == "bm25":
            return await self._bm25_search(query, collection, top_k)

        # Hybrid: vector + BM25 with RRF fusion
        chunks = await self._hybrid_search(query, vector, collection, top_k)

        if rerank and self.reranker and chunks:
            try:
                chunks = await self.reranker.rerank(query, chunks, top_n=rerank_top_n)
            except Exception:
                logger.warning("Reranker failed, returning un-reranked results", exc_info=True)

        return chunks

    async def _vector_search(
        self, vector: list[float], collection: str, top_k: int,
    ) -> list[ChunkResult]:
        hits = await self.qdrant.search(collection, vector, limit=top_k)
        return [ChunkResult(**h) for h in hits]

    async def _bm25_search(
        self, query: str, collection: str, top_k: int,
    ) -> list[ChunkResult]:
        hits = await self.bm25.search(query, collection, top_k)
        results = []
        for doc_id, score in hits[:top_k]:
            doc = await self.bm25.get(doc_id, collection)
            if doc:
                results.append(ChunkResult(
                    text=doc["text"],
                    source_id=doc.get("source_id") or doc.get("doc_id", ""),
                    chunk_index=doc.get("chunk_index", 0),
                    score=score,
                ))
        return results

    async def _hybrid_search(
        self, query: str, vector: list[float], collection: str, top_k: int,
    ) -> list[ChunkResult]:
        fetch_k = top_k * 2

        vector_hits = await self.qdrant.search(collection, vector, limit=fetch_k)
        bm25_hits = await self.bm25.search(query, collection, fetch_k)

        # Build payload map from vector results
        payloads: dict[str, dict] = {
            f"{h['source_id']}_{h['chunk_index']}": h for h in vector_hits
        }

        # Add BM25-only payloads
        for doc_id, _ in bm25_hits:
            if doc_id not in payloads:
                doc = await self.bm25.get(doc_id, collection)
                if doc:
                    payloads[doc_id] = {
                        "text": doc["text"],
                        "source_id": doc.get("source_id") or doc.get("doc_id", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "score": 0.0,
                    }

        fused = rrf([
            [f"{h['source_id']}_{h['chunk_index']}" for h in vector_hits],
            [doc_id for doc_id, _ in bm25_hits],
        ])

        return [
            ChunkResult(
                text=payloads[doc_id]["text"],
                source_id=payloads[doc_id]["source_id"],
                chunk_index=payloads[doc_id]["chunk_index"],
                score=score,
            )
            for doc_id, score in fused[:top_k]
            if doc_id in payloads
        ]
