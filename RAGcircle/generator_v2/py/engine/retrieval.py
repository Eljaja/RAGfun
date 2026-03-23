"""Retrieval pipeline: initial_expand -> fetch -> loop(check, expand, fetch) -> finalize.

Spec reference: Layer 2b — Retrieval Pipeline (ARCHITECTURE_v2.md:58-91).
Black-box for chunk acquisition. The generator pipeline calls run_retrieval()
and gets back RetrievalResult (chunks + traces).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import Settings
from context import merge_chunks, stitch_segments
from engine.budget import BudgetCounter
from llm import LLMClient
from retrieval_contract import ChunkResult, ScoreSource
from models.plan import ConfigMeta, RetrievalRequest, RetrievalResult
from retrieval_contract import ExecutionPlan
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
from plan_builder import from_preset
from query_variants import (
    extract_hint_terms,
    keyword_query,
    query_variants as heuristic_variants,
    unique_source_count,
)
from steps.expand import _fact_queries, _hyde, _keywords
from steps.retrieve import _safe_retrieve

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
) -> RetrievalResult:
    """Run the full retrieval pipeline, return chunks + traces."""
    traces: list[dict[str, Any]] = []
    default_config = meta.retrieval_plan or from_preset(
        default_preset, top_k=default_top_k, rerank=default_rerank,
    )

    # -- Initial expand: original query + expand-step queries
    requests = [RetrievalRequest(query=query)]
    expand_reqs, expand_traces = await _initial_expand(
        initial_expand_steps,
        query=query,
        meta=meta,
        history_text=history_text,
        budget=budget,
        llm=llm,
        model=model,
    )
    requests.extend(expand_reqs)
    traces.extend(expand_traces)

    # -- First fetch
    chunks = await _fetch_all(
        requests,
        default_config=default_config,
        project_id=project_id,
        http_client=http_client,
        settings=settings,
    )

    if not chunks:
        return RetrievalResult(chunks=[], traces=traces)

    # -- Loop: check -> expand -> fetch
    for round_idx in range(max_rounds - 1):
        if not _should_continue(loop_check_steps, chunks=chunks):
            break

        loop_reqs, loop_traces = _loop_expand(
            loop_expand_steps,
            query=query,
            chunks=chunks,
            is_factoid=meta.is_factoid,
            retrieval_plan=meta.retrieval_plan,
        )
        traces.extend(loop_traces)

        if not loop_reqs:
            break

        extra = await _fetch_all(
            loop_reqs,
            default_config=default_config,
            project_id=project_id,
            http_client=http_client,
            settings=settings,
        )
        chunks = merge_chunks([chunks, extra])
        traces.append({
            "kind": "action", "label": "RetrievalLoop",
            "content": f"Round {round_idx + 2}: fetched {len(extra)} extra chunks",
        })

    # -- Finalize
    chunks, finalize_traces = _finalize(finalize_steps, chunks=chunks)
    traces.extend(finalize_traces)

    return RetrievalResult(chunks=chunks, traces=traces)


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
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Run expand steps, produce retrieval requests (no I/O yet)."""
    requests: list[RetrievalRequest] = []
    traces: list[dict[str, Any]] = []

    for step in steps:
        match step:
            case HyDEStep(num_passages=num_passages):
                hyde_queries, t = await _hyde(
                    query=query, lang=meta.lang, budget=budget,
                    llm=llm, model=model, num_passages=num_passages,
                )
                requests.extend(RetrievalRequest(query=q) for q in hyde_queries)
                traces.extend(t)

            case FactQueryStep(max_queries=max_q):
                sub_queries, t = await _fact_queries(
                    query=query, history_text=history_text,
                    max_queries=max_q, budget=budget,
                    retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                )
                requests.extend(RetrievalRequest(query=q) for q in sub_queries)
                traces.extend(t)

            case KeywordStep():
                kw_queries, t = await _keywords(
                    query=query, history_text=history_text,
                    budget=budget, retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                )
                requests.extend(RetrievalRequest(query=q) for q in kw_queries)
                traces.extend(t)

            case QueryVariantsStep():
                variants = heuristic_variants(query)
                if len(variants) > 1:
                    requests.extend(
                        RetrievalRequest(query=v) for v in variants[1:]
                    )
                    traces.append({
                        "kind": "action", "label": "QueryVariants",
                        "content": f"Generated {len(variants)} variants",
                    })

            case BM25AnchorStep(top_k=anchor_k):
                kw = keyword_query(query)
                if kw:
                    bm25_plan = from_preset("fast", top_k=anchor_k, rerank=False)
                    requests.append(RetrievalRequest(query=kw, plan_override=bm25_plan))
                    traces.append({
                        "kind": "action", "label": "BM25Anchor",
                        "content": f"kw={kw[:60]}",
                    })

    return requests, traces


# ── Fetch ────────────────────────────────────────────────


async def _fetch_all(
    requests: list[RetrievalRequest],
    *,
    default_config: ExecutionPlan,
    project_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> list[ChunkResult]:
    """Fetch all retrieval requests in parallel, merge results."""
    if not requests:
        return []

    if len(requests) == 1:
        req = requests[0]
        return await _safe_retrieve(
            req.query, req.plan_override or default_config,
            project_id=project_id, http_client=http_client, settings=settings,
        )

    results = await asyncio.gather(*(
        _safe_retrieve(
            req.query, req.plan_override or default_config,
            project_id=project_id, http_client=http_client, settings=settings,
        )
        for req in requests
    ))
    return merge_chunks(list(results))


# ── Loop helpers ─────────────────────────────────────────


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


def _loop_expand(
    steps: list[LoopExpandStep],
    *,
    query: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
    retrieval_plan: ExecutionPlan | None,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Produce retrieval requests for the next loop iteration."""
    requests: list[RetrievalRequest] = []
    traces: list[dict[str, Any]] = []

    for step in steps:
        match step:
            case TwoPassStep(min_unique_sources=min_src):
                n_unique = unique_source_count(chunks)
                if n_unique < min_src:
                    hints = extract_hint_terms(chunks, max_terms=3)
                    if hints:
                        follow_up = f"{query} {' '.join(hints)}"
                        plan = retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
                        requests.append(RetrievalRequest(query=follow_up, plan_override=plan))
                        traces.append({
                            "kind": "action", "label": "TwoPass",
                            "content": f"Only {n_unique} unique sources, follow-up query queued",
                        })

            case FactoidExpandStep():
                if is_factoid and chunks:
                    expand_plan = from_preset("fast", top_k=5, rerank=False)
                    requests.append(RetrievalRequest(query=query, plan_override=expand_plan))
                    traces.append({
                        "kind": "action", "label": "FactoidExpand",
                        "content": "Expanding within top sources",
                    })

    return requests, traces


# ── Finalize ─────────────────────────────────────────────


def _finalize(
    steps: list[StitchStep],
    *,
    chunks: list[ChunkResult],
) -> tuple[list[ChunkResult], list[dict[str, Any]]]:
    """Run finalize steps (stitching, etc.) on the final chunk set."""
    traces: list[dict[str, Any]] = []

    for step in steps:
        before = len(chunks)
        chunks = stitch_segments(chunks, max_per_segment=step.max_per_segment)
        after = len(chunks)
        if before != after:
            traces.append({
                "kind": "action", "label": "Stitch",
                "content": f"Stitched {before} chunks into {after} segments",
            })

    return chunks, traces
