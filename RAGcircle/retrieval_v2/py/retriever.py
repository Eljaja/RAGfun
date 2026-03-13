from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter as CollCounter

from store import QdrantStore, BM25Store
from embedder import Embedder
from reranker import Reranker
from models import (
    AdaptiveKStep,
    BM25SearchStep,
    ChunkResult,
    ExecutionPlan,
    FuseStep,
    PlanRound,
    RetrievalStep,
    RerankStep,
    StepBase,
    TrimStep,
    VectorSearchStep,
)

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _keyword_query(q: str) -> str:
    toks = [t.lower() for t in _TOKEN_RE.findall(q or "")]
    if not toks:
        return ""
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
        "are", "was", "were", "be", "been", "with", "what", "which", "who", "when",
        "where", "why", "how", "many", "much", "did", "does", "do", "have", "has",
        "что", "какой", "какая", "какие", "кто", "где", "когда", "как", "сколько",
        "это", "эта", "эти", "этот", "для", "или", "а", "и", "в", "на", "по", "с",
    }
    toks = [t for t in toks if len(t) >= 3 and t not in stop]
    if not toks:
        return ""
    c = CollCounter(toks)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    keep = [t for t, _ in ranked[:10]]
    return " ".join(keep)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _chunk_key(chunk: ChunkResult) -> str:
    return f"{chunk.source_id}:{chunk.chunk_index}"


def _dedupe_keep_best(chunks: list[ChunkResult]) -> list[ChunkResult]:
    best: dict[str, ChunkResult] = {}
    for chunk in chunks:
        key = _chunk_key(chunk)
        current = best.get(key)
        if current is None or chunk.score > current.score:
            best[key] = chunk
    out = list(best.values())
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _adaptive_k_cutoff(
    chunks: list[ChunkResult],
    *,
    min_k: int = 3,
    max_k: int = 24,
) -> list[ChunkResult]:
    if len(chunks) <= 1:
        return chunks
    scores = [float(c.score) for c in chunks]
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

    @staticmethod
    def build_legacy_plan(
        *,
        strategy: str,
        top_k: int,
        rerank: bool,
        rerank_top_n: int,
    ) -> ExecutionPlan:
        strategy = (strategy or "hybrid").lower().strip()
        top_k = max(1, int(top_k))
        rerank_top_n = max(1, int(rerank_top_n))

        if strategy == "bm25":
            return ExecutionPlan(
                rounds=[
                    PlanRound(
                        retrieve=[BM25SearchStep(top_k=top_k)],
                        finalize=[TrimStep(top_k=top_k)],
                    )
                ]
            )
        if strategy == "vector":
            return ExecutionPlan(
                rounds=[
                    PlanRound(
                        retrieve=[VectorSearchStep(top_k=top_k)],
                        finalize=[TrimStep(top_k=top_k)],
                    )
                ]
            )

        # Legacy hybrid behavior:
        # - retrieve from both backends with a wider pool
        # - fuse via RRF
        # - optional rerank
        # - trim to requested top_k
        fetch_k = max(top_k * 2, top_k)
        rank_steps = [RerankStep(top_n=min(rerank_top_n, fetch_k))] if rerank else []
        return ExecutionPlan(
            rounds=[
                PlanRound(
                    retrieve=[
                        VectorSearchStep(top_k=fetch_k),
                        BM25SearchStep(top_k=fetch_k),
                    ],
                    combine=FuseStep(rrf_k=60),
                    rank=rank_steps,
                    finalize=[TrimStep(top_k=top_k)],
                )
            ]
        )

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
        plan = self.build_legacy_plan(
            strategy=strategy,
            top_k=top_k,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
        )
        chunks, _ = await self.execute_plan(query=query, collection=collection, plan=plan)
        return chunks

    @staticmethod
    def _append_warning(warnings: list[str], code: str) -> None:
        if code not in warnings:
            warnings.append(code)

    def _resolve_step_query(
        self,
        *,
        base_query: str,
        step: RetrievalStep,
    ) -> str:
        query = (step.query or "").strip()
        if not query:
            if step.query_mode == "keyword":
                query = _keyword_query(base_query) or base_query
            else:
                query = base_query
        return query

    async def _search_step(
        self,
        *,
        base_query: str,
        collection: str,
        step: BM25SearchStep | VectorSearchStep,
    ) -> list[ChunkResult]:
        match step:
            case BM25SearchStep():
                step_query = self._resolve_step_query(
                    base_query=base_query,
                    step=step,
                )
                return await self._bm25_search(step_query, collection, step.top_k)
            case VectorSearchStep():
                step_query = self._resolve_step_query(
                    base_query=base_query,
                    step=step,
                )
                vector = (await self.embedder.embed([step_query]))[0]
                return await self._vector_search(vector, collection, step.top_k)
            case _:
                raise RuntimeError(f"Unsupported retrieval step: {type(step).__name__}")
 

    async def _run_retrieval_steps(
        self,
        *,
        base_query: str,
        collection: str,
        steps: list[RetrievalStep],
    ) -> list[list[ChunkResult]]:
        return await asyncio.gather(
            *[
                self._search_step(
                    base_query=base_query,
                    collection=collection,
                    step=step,
                )
                for step in steps
            ]
        )

    def _combine_round_sources(
        self,
        *,
        source_lists: list[list[ChunkResult]],
        combine: FuseStep | None,
    ) -> list[ChunkResult]:
        if combine is not None:
            return self._fuse_sources(source_lists, combine)
        flat = [c for chunks in source_lists for c in chunks]
        return _dedupe_keep_best(flat)

    async def _search_round_sources(
        self,
        *,
        base_query: str,
        collection: str,
        round_spec: PlanRound,
    ) -> list[list[ChunkResult]]:
        return await self._run_retrieval_steps(
            base_query=base_query,
            collection=collection,
            steps=round_spec.retrieve,
        )

    async def _apply_rank_steps(
        self,
        *,
        query: str,
        chunks: list[ChunkResult],
        rank_steps: list[RerankStep | AdaptiveKStep],
        warnings: list[str],
    ) -> list[ChunkResult]:
        out = chunks
        for rank_step in rank_steps:
            if isinstance(rank_step, RerankStep):
                if not out:
                    continue
                if self.reranker is None:
                    self._append_warning(warnings, "rerank_unavailable")
                    continue
                try:
                    top_n = min(rank_step.top_n, len(out))
                    out = await self.reranker.rerank(query, out, top_n=top_n)
                except Exception:
                    logger.warning("Reranker step failed", exc_info=True)
                    self._append_warning(warnings, "rerank_failed")
            elif isinstance(rank_step, AdaptiveKStep):
                out = _adaptive_k_cutoff(
                    out,
                    min_k=rank_step.min_k,
                    max_k=rank_step.max_k,
                )
        return out

    @staticmethod
    def _apply_finalize_steps(
        *,
        chunks: list[ChunkResult],
        finalize_steps: list[TrimStep],
    ) -> list[ChunkResult]:
        out = chunks
        for final_step in finalize_steps:
            out = out[: final_step.top_k]
        return out

    async def execute_plan(
        self,
        *,
        query: str,
        collection: str,
        plan: ExecutionPlan,
    ) -> tuple[list[ChunkResult], list[str]]:
        """
        Execute a typed retrieval plan.

        Returns:
        - chunks: merged results from all rounds (deduped, sorted desc by score)
        - warnings: non-fatal issues (e.g. rerank unavailable/failed)
        """
        merged_across_rounds: list[ChunkResult] = []
        warnings: list[str] = []

        for round_spec in plan.rounds:
            # pipeline!!!
            source_lists = await self._search_round_sources(
                base_query=query,
                collection=collection,
                round_spec=round_spec,
            )
            round_chunks = self._combine_round_sources(
                source_lists=source_lists,
                combine=round_spec.combine,
            )
            round_chunks = await self._apply_rank_steps(
                query=query,
                chunks=round_chunks,
                rank_steps=round_spec.rank,
                warnings=warnings,
            )
            round_chunks = self._apply_finalize_steps(
                chunks=round_chunks,
                finalize_steps=round_spec.finalize,
            )

            merged_across_rounds = _dedupe_keep_best(merged_across_rounds + round_chunks)

        return merged_across_rounds, warnings

    async def _vector_search(
        self, vector: list[float], collection: str, top_k: int,
    ) -> list[ChunkResult]:
        hits = await self.qdrant.search(collection, vector, limit=top_k)
        return [ChunkResult(**h) for h in hits]

    async def _bm25_search(
        self, query: str, collection: str, top_k: int,
    ) -> list[ChunkResult]:
        ids_scores = list((await self.bm25.search(query, collection, top_k))[:top_k])
        docs = await asyncio.gather(
            *(self.bm25.get(doc_id, collection) for doc_id, _ in ids_scores),
        )
        return [
            ChunkResult(
                text=doc["text"],
                source_id=doc.get("source_id") or doc.get("doc_id", ""),
                chunk_index=doc.get("chunk_index", 0),
                score=score,
            )
            for (_, score), doc in zip(ids_scores, docs, strict=False)
            if doc
        ]

    def _fuse_sources(
        self,
        source_lists: list[list[ChunkResult]],
        fuse_step: FuseStep,
    ) -> list[ChunkResult]:
        if not source_lists:
            return []
        if len(source_lists) == 1:
            return _dedupe_keep_best(source_lists[0])

        rankings = [[_chunk_key(chunk) for chunk in chunks] for chunks in source_lists]
        keyed_chunks = [
            (_chunk_key(chunk), chunk)
            for chunks in source_lists
            for chunk in chunks
        ]
        # Preserve "first seen" payload semantics without mutating in-loop.
        payloads = {cid: chunk for cid, chunk in reversed(keyed_chunks)}

        return [
            base.model_copy(update={"score": float(fused_score)})
            for cid, fused_score in rrf(rankings, k=fuse_step.rrf_k)
            if (base := payloads.get(cid)) is not None
        ]
