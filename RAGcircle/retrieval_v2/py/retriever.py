from __future__ import annotations

import logging

from store import QdrantStore, BM25Store
from embedder import Embedder
from reranker import Reranker
from models import (
    AdaptiveKStep,
    BM25SearchStep,
    ChunkResult,
    ExecutionPlan,
    FuseStep,
    RerankStep,
    ScoreSource,
    TrimStep,
    VectorSearchStep,
)

logger = logging.getLogger(__name__)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def adaptive_k_cutoff(
    chunks: list[ChunkResult],
    *,
    min_k: int = 3,
    max_k: int = 24,
) -> list[ChunkResult]:
    if len(chunks) <= 1:
        return chunks
    scores = [c.score for c in chunks]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    if not gaps:
        return chunks
    best_i = max(range(len(gaps)), key=lambda i: gaps[i])
    effective_k = max(min_k, min(max_k, best_i + 1))
    return chunks[:effective_k]


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

    async def execute_plan(
        self,
        query: str,
        collection: str,
        plan: ExecutionPlan,
    ) -> list[ChunkResult]:
        round_plan = plan.round

        retrieval_outputs: list[list[ChunkResult]] = []
        for step in round_plan.retrieve:
            if isinstance(step, VectorSearchStep):
                q = step.query.strip() if step.query.strip() else query
                vector = (await self.embedder.embed([q]))[0]
                retrieval_outputs.append(await self._vector_search(vector, collection, step.top_k))
            elif isinstance(step, BM25SearchStep):
                q = step.query.strip() if step.query.strip() else query
                retrieval_outputs.append(await self._bm25_search(q, collection, step.top_k))

        if not retrieval_outputs:
            return []

        chunks: list[ChunkResult]
        if len(retrieval_outputs) == 1:
            chunks = retrieval_outputs[0]
        else:
            combine = round_plan.combine or FuseStep()
            chunks = self._fuse_rrf(retrieval_outputs, combine.rrf_k)

        for step in round_plan.rank:
            if isinstance(step, RerankStep):
                if not chunks or not self.reranker:
                    continue
                rq = step.query.strip() if step.query.strip() else query
                try:
                    chunks = await self.reranker.rerank(rq, chunks, top_n=step.top_n)
                except Exception:
                    logger.warning("Reranker failed in execution plan", exc_info=True)
            elif isinstance(step, AdaptiveKStep):
                chunks = adaptive_k_cutoff(chunks, min_k=step.min_k, max_k=step.max_k)

        for step in round_plan.finalize:
            if isinstance(step, TrimStep):
                chunks = chunks[: step.top_k]

        return chunks

    async def _vector_search(
        self, vector: list[float], collection: str, top_k: int,
    ) -> list[ChunkResult]:
        hits = await self.qdrant.search(collection, vector, limit=top_k)
        return [ChunkResult(**h, score_source=ScoreSource.RETRIEVAL) for h in hits]

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
                    score_source=ScoreSource.RETRIEVAL,
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
                score_source=ScoreSource.RRF,
            )
            for doc_id, score in fused[:top_k]
            if doc_id in payloads
        ]

    def _fuse_rrf(
        self,
        retrieval_outputs: list[list[ChunkResult]],
        rrf_k: int = 60,
    ) -> list[ChunkResult]:
        payloads: dict[str, ChunkResult] = {}
        rankings: list[list[str]] = []

        for output in retrieval_outputs:
            ranking_ids: list[str] = []
            for c in output:
                cid = f"{c.source_id}_{c.chunk_index}"
                ranking_ids.append(cid)
                if cid not in payloads:
                    payloads[cid] = c
            rankings.append(ranking_ids)

        fused = rrf(rankings, k=rrf_k)
        return [
            ChunkResult(
                text=payloads[cid].text,
                source_id=payloads[cid].source_id,
                chunk_index=payloads[cid].chunk_index,
                score=score,
                score_source=ScoreSource.RRF,
            )
            for cid, score in fused
            if cid in payloads
        ]
