from __future__ import annotations

import asyncio
import logging

import httpx
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import (
    ConnectionError as OSConnectionError,
    ConnectionTimeout as OSConnectionTimeout,
    NotFoundError as OSNotFoundError,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse as QdrantUnexpectedResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import (
    AdaptiveKStep,
    BM25SearchStep,
    ChunkResult,
    EmbeddingResponse,
    ExecutionPlan,
    RetrievalStep,
    RerankStep,
    ScoreSource,
    VectorSearchStep,
)
from ranking import (
    adaptive_k_cutoff,
    apply_finalize,
    combine_sources,
    dedupe_keep_best,
)

logger = logging.getLogger(__name__)
_RETRYABLE_HTTP = (httpx.TransportError, httpx.TimeoutException)
_RETRYABLE_QDRANT = (ConnectionError, QdrantUnexpectedResponse)
_RETRYABLE_OS = (OSConnectionError, OSConnectionTimeout)


class HybridRetriever:
    def __init__(
        self,
        *,
        qdrant: AsyncQdrantClient,
        opensearch: AsyncOpenSearch,
        embed_http: httpx.AsyncClient,
        embed_model: str,
        rerank_http: httpx.AsyncClient | None = None,
        rerank_model: str = "",
    ):
        self.qdrant = qdrant
        self.opensearch = opensearch
        self.embed_http = embed_http
        self.embed_model = embed_model
        self.rerank_http = rerank_http
        self.rerank_model = rerank_model

    # ── network helpers ───────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(_RETRYABLE_HTTP),
        reraise=True,
    )
    async def _model_post(
        self, client: httpx.AsyncClient, path: str, model: str, payload: dict,
    ) -> dict:
        resp = await client.post(path, json={"model": model, **payload})
        resp.raise_for_status()
        return resp.json()

    async def _embed(self, text: str) -> list[float]:
        data = await self._model_post(
            self.embed_http, "/embeddings", self.embed_model, {"input": text},
        )
        parsed = EmbeddingResponse.model_validate(data)
        return parsed.data[0].embedding

    async def _rerank(
        self, query: str, chunks: list[ChunkResult], top_n: int,
    ) -> list[ChunkResult]:
        data = await self._model_post(
            self.rerank_http, "/rerank", self.rerank_model,
            {"query": query, "documents": [c.text for c in chunks]},
        )
        ranked = sorted(
            data["results"], key=lambda x: x["relevance_score"], reverse=True,
        )
        return [
            chunks[r["index"]].model_copy(update={
                "score": float(r["relevance_score"]),
                "score_source": ScoreSource.RERANK,
            })
            for r in ranked[:top_n]
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, max=2),
        retry=retry_if_exception_type(_RETRYABLE_QDRANT),
        reraise=True,
    )
    async def _vector_search(
        self, vector: list[float], collection: str, top_k: int,
    ) -> list[ChunkResult]:
        results = await self.qdrant.query_points(
            collection, query=vector, limit=top_k, with_payload=True,
        )
        return [
            ChunkResult(
                text=r.payload["text"],
                source_id=r.payload.get("source_id") or r.payload.get("doc_id", ""),
                chunk_index=r.payload.get("chunk_index", 0),
                score=r.score,
            )
            for r in results.points
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, max=2),
        retry=retry_if_exception_type(_RETRYABLE_OS),
        reraise=True,
    )
    async def _bm25_search(
        self, query: str, collection: str, top_k: int,
    ) -> list[ChunkResult]:
        resp = await self.opensearch.search(
            index=collection,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
        hits = resp["hits"]["hits"][:top_k]
        docs = await asyncio.gather(
            *(self._os_get(hit["_id"], collection) for hit in hits),
        )
        return [
            ChunkResult(
                text=doc["text"],
                source_id=doc.get("source_id") or doc.get("doc_id", ""),
                chunk_index=doc.get("chunk_index", 0),
                score=hit["_score"],
            )
            for hit, doc in zip(hits, docs, strict=False)
            if doc
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, max=2),
        retry=retry_if_exception_type(_RETRYABLE_OS),
        reraise=True,
    )
    async def _os_get(self, doc_id: str, index: str) -> dict | None:
        try:
            resp = await self.opensearch.get(index=index, id=doc_id)
            return resp["_source"]
        except OSNotFoundError:
            return None

    # ── plan execution ────────────────────────────────────────

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
        plan = ExecutionPlan.from_legacy(
            strategy=strategy,
            top_k=top_k,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
        )
        chunks = await self.execute_plan(query=query, collection=collection, plan=plan)
        return chunks

    def _resolve_step_query(
        self,
        *,
        base_query: str,
        step,
    ) -> str:
        query = (getattr(step, "query", "") or "").strip()
        return query or base_query

    async def _search_step(
        self,
        *,
        base_query: str,
        collection: str,
        step,
    ) -> list[ChunkResult]:
        step_query = self._resolve_step_query(
            base_query=base_query, step=step,
        )
        match step:
            case BM25SearchStep():
                return await self._bm25_search(step_query, collection, step.top_k)
            case VectorSearchStep():
                vector = await self._embed(step_query)
                return await self._vector_search(vector, collection, step.top_k)
            case _:
                raise RuntimeError(f"Unsupported retrieval step: {type(step).__name__}")

    async def _fetch_sources(
        self,
        *,
        query: str,
        collection: str,
        steps: list[RetrievalStep],
    ) -> list[list[ChunkResult]]:
        return await asyncio.gather(*(
            self._search_step(base_query=query, collection=collection, step=s)
            for s in steps
        ))

    async def _apply_rank_steps(
        self,
        *,
        base_query: str,
        chunks: list[ChunkResult],
        rank_steps: list[RerankStep | AdaptiveKStep],
    ) -> list[ChunkResult]:
        out = chunks
        for rank_step in rank_steps:
            match rank_step:
                case RerankStep(top_n=top_n):
                    rerank_query = self._resolve_step_query(
                        base_query=base_query, step=rank_step,
                    )
                    out = await self._rerank(rerank_query, out, top_n=min(top_n, len(out)))
                case AdaptiveKStep(min_k=min_k, max_k=max_k):
                    out = adaptive_k_cutoff(out, min_k=min_k, max_k=max_k)
        return out

    async def execute_plan(
        self,
        *,
        query: str,
        collection: str,
        plan: ExecutionPlan,
    ) -> list[ChunkResult]:
        """Execute a typed retrieval plan. Returns deduped results sorted desc by score."""
        rnd = plan.round
        source_lists = await self._fetch_sources(
            query=query, collection=collection, steps=rnd.retrieve,
        )
        chunks = combine_sources(source_lists, rnd.combine)
        chunks = await self._apply_rank_steps(
            base_query=query, chunks=chunks, rank_steps=rnd.rank,
        )
        chunks = apply_finalize(chunks, rnd.finalize)
        return dedupe_keep_best(chunks)
