"""Retrieval pipeline: initial_expand -> fetch -> loop(check, expand, fetch) -> finalize.

Spec reference: Layer 2b — Retrieval Pipeline (ARCHITECTURE_v2.md:58-91).
Black-box for chunk acquisition. The generator pipeline calls run_retrieval()
and gets back RetrievalResult (chunks + traces).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import Settings
from lib.context import merge_chunks, stitch_segments
from engine.brain_budget import BudgetCounter
from engine.trace_collector import TraceCollector
from clients.llm import LLMClient
from retrieval_contract import ChunkResult, ExecutionPlan, ScoreSource
from retrieval_contract import from_preset
from models.plan import ConfigMeta, RetrievalRequest, RetrievalResult
from models.steps import (
    BM25AnchorStep,
    FactoidExpandStep,
    FactQueryStep,
    HyDEStep,
    InitialExpandStep,
    KeywordStep,
    LoopExpandStep,
    QualityCheckStep,
    QueryVariantsStep,
    StitchStep,
    TwoPassStep,
)
from steps.expand import (
    bm25_anchor_expand,
    fact_queries,
    factoid_expand,
    hyde,
    keywords,
    query_variants_expand,
    two_pass_expand,
)
from steps.query_heuristics import unique_source_count
from steps.retrieve import fetch_all

logger = logging.getLogger(__name__)


# ── Public entry point ───────────────────────────────────


async def run_retrieval(
    *,
    query: str,
    meta: ConfigMeta,
    history_text: str,
    initial_expand_steps: list[InitialExpandStep],
    loop_check_steps: list[QualityCheckStep],
    loop_expand_steps: list[LoopExpandStep],
    finalize_steps: list[StitchStep],
    max_rounds: int = 2,
    default_preset: str = "hybrid",
    default_top_k: int = 10,
    default_rerank: bool = True,
    project_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    collector: TraceCollector,
) -> RetrievalResult:
    """Run the full retrieval pipeline, return chunks + traces."""

    # TODO: think about this one 
    # I think we should be able to merge plans if they are the same 
    # ah shi
    if not meta.retrieval_plan:
        meta.retrieval_plan = from_preset(
        default_preset, top_k=default_top_k, rerank=default_rerank,
    )
    default_config = meta.retrieval_plan

    # -- Initial expand: original query + expand-step queries
    requests = [RetrievalRequest(query=query)]
    expand_reqs = await _initial_expand(
        initial_expand_steps,
        query=query,
        meta=meta,
        history_text=history_text,
        budget=budget,
        llm=llm,
        model=model,
        collector=collector,
    )
    requests.extend(expand_reqs)

    # -- First fetch
    chunks = await fetch_all(
        requests,
        default_plan=default_config,
        project_id=project_id,
        http_client=http_client,
        settings=settings,
    )

    if not chunks:
        return RetrievalResult(chunks=[], traces=list(collector.entries))

    # -- Loop: check -> expand -> fetch
    for round_idx in range(max_rounds - 1):
        if not _should_continue(loop_check_steps, chunks=chunks):
            break

        loop_reqs = _loop_expand(
            loop_expand_steps,
            query=query,
            chunks=chunks,
            is_factoid=meta.is_factoid,
            retrieval_plan=meta.retrieval_plan,
            collector=collector,
        )

        if not loop_reqs:
            break

        extra = await fetch_all(
            loop_reqs,
            default_plan=default_config,
            project_id=project_id,
            http_client=http_client,
            settings=settings,
        )
        chunks = merge_chunks([chunks, extra])
        collector._entries.append({
            "kind": "action", "label": "RetrievalLoop",
            "content": f"Round {round_idx + 2}: fetched {len(extra)} extra chunks",
        })

    # -- Finalize
    chunks = _finalize(finalize_steps, chunks=chunks, collector=collector)

    return RetrievalResult(chunks=chunks, traces=list(collector.entries))


# ── Initial expand ───────────────────────────────────────


async def _initial_expand(
    steps: list[InitialExpandStep],
    *,
    query: str,
    meta: ConfigMeta,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    collector: TraceCollector,
) -> list[RetrievalRequest]:
    """Dispatch expand steps to leaf functions, collect retrieval requests."""
    requests: list[RetrievalRequest] = []

    for step in steps:
        match step:
            case HyDEStep(num_passages=num_passages):
                queries = await hyde(
                    query=query, lang=meta.lang, budget=budget,
                    llm=llm, model=model, num_passages=num_passages,
                    collector=collector,
                )
                requests.extend(RetrievalRequest(query=q) for q in queries)

            case FactQueryStep(max_queries=max_q):
                queries = await fact_queries(
                    query=query, history_text=history_text,
                    max_queries=max_q, budget=budget,
                    retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                    collector=collector,
                )
                logger.warning(f" this is a raw JSON SEemingly {queries}")
                requests.extend(RetrievalRequest(query=q) for q in queries)

            case KeywordStep():
                queries = await keywords(
                    query=query, history_text=history_text,
                    budget=budget, retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                    collector=collector,
                )
                requests.extend(RetrievalRequest(query=q) for q in queries)

            case QueryVariantsStep():
                reqs = query_variants_expand(query=query)
                if reqs:
                    await collector.emit({
                        "kind": "action", "label": "QueryVariants",
                        "content": f"Generated {len(reqs)} variants",
                    })
                requests.extend(reqs)

            case BM25AnchorStep(top_k=anchor_k):
                reqs = bm25_anchor_expand(query=query, top_k=anchor_k)
                if reqs:
                    kw = reqs[0].query
                    await collector.emit({
                        "kind": "action", "label": "BM25Anchor",
                        "content": f"kw={kw[:60]}",
                    })
                requests.extend(reqs)

    return requests


# ── Loop helpers ─────────────────────────────────────────

# TODO
# implement actual methods that would work here 
# and do the outline on the paper pls 
# what steps we have and how do they connect 
def _should_continue(
    checks: list[QualityCheckStep],
    *,
    chunks: list[ChunkResult],
) -> bool:
    """Return True if more retrieval rounds are needed."""
    for step in checks:
        if not chunks or len(chunks) < step.min_hits:
            return True
        if (
            chunks[0].score_source == ScoreSource.RERANK
            and chunks[0].score < step.min_score
        ):
            return True
    return False


async def _loop_expand(
    steps: list[LoopExpandStep],
    *,
    query: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
    retrieval_plan: ExecutionPlan | None,
    collector: TraceCollector,
) -> list[RetrievalRequest]:
    """Dispatch loop-expand steps to leaf functions, collect requests."""
    requests: list[RetrievalRequest] = []

    for step in steps:
        match step:
            case TwoPassStep(min_unique_sources=min_src):
                reqs = two_pass_expand(
                    query=query, chunks=chunks,
                    min_unique_sources=min_src,
                    retrieval_plan=retrieval_plan,
                )
                if reqs:
                    n_unique = unique_source_count(chunks)
                    await collector.emit({
                        "kind": "action", "label": "TwoPass",
                        "content": f"Only {n_unique} unique sources, follow-up query queued",
                    })
                requests.extend(reqs)

            case FactoidExpandStep():
                reqs = factoid_expand(
                    query=query, chunks=chunks, is_factoid=is_factoid,
                )
                if reqs:
                    await collector.emit({
                        "kind": "action", "label": "FactoidExpand",
                        "content": "Expanding within top sources",
                    })
                requests.extend(reqs)

    return requests


# ── Finalize ─────────────────────────────────────────────


def _finalize(
    steps: list[StitchStep],
    *,
    chunks: list[ChunkResult],
    collector: TraceCollector,
) -> list[ChunkResult]:
    """Run finalize steps (stitching, etc.) on the final chunk set."""

    for step in steps:
        before = len(chunks)
        chunks = stitch_segments(chunks, max_per_segment=step.max_per_segment)
        after = len(chunks)
        if before != after:
            collector._entries.append({
                "kind": "action", "label": "Stitch",
                "content": f"Stitched {before} chunks into {after} segments",
            })

    return chunks
