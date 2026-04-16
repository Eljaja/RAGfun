"""Retrieval pipeline: initial_expand -> fetch -> loop(check, expand, fetch) -> finalize.

Uses run_stage for sub-stages; owns the loop logic and fetch calls.
Returns (RetrievalResult, traces) — traces are a parallel channel.
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import Any

import httpx

from config import Settings
from lib.context import merge_chunks, stitch_segments
from engine.brain_budget import BudgetCounter
from engine.stage import run_stage, StageResult
from clients.llm import LLMClient
from retrieval_contract import ChunkResult, ExecutionPlan, ScoreSource
from retrieval_contract import from_preset
from models.plan import ConfigMeta, RetrievalRequest, RetrievalResult
from models.steps import (
    InitialExpandStep,
    LoopExpandStep,
    QualityCheckStep,
    StitchStep,
)
from steps.expand import make_expand_dispatch, make_loop_expand_dispatch
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
) -> tuple[RetrievalResult, list[dict[str, Any]]]:
    """Run the full retrieval pipeline. Returns (result, traces)."""
    traces: list[dict[str, Any]] = []

    if not meta.retrieval_plan:
        meta.retrieval_plan = from_preset(
            default_preset, top_k=default_top_k, rerank=default_rerank,
        )
    default_config = meta.retrieval_plan

    # -- Initial expand: original query + expand-step queries
    requests = [RetrievalRequest(query=query)]

    expand_result = await run_stage(
        initial_expand_steps,
        make_expand_dispatch(
            query=query, meta=meta, history_text=history_text,
            budget=budget, llm=llm, model=model,
        ),
        parallel=True,
    )
    if expand_result.has_errors:
        logger.warning(
            "initial_expand: %d/%d steps failed",
            len(expand_result.errors), len(initial_expand_steps),
        )
    requests.extend(chain.from_iterable(expand_result.domain))
    traces.extend(expand_result.traces)

    # -- First fetch
    chunks = await fetch_all(
        requests,
        default_plan=default_config,
        project_id=project_id,
        http_client=http_client,
        settings=settings,
    )

    if not chunks:
        return RetrievalResult(chunks=[]), traces

    # -- Loop: check -> expand -> fetch
    for round_idx in range(max_rounds - 1):
        if not _should_continue(loop_check_steps, chunks=chunks):
            break

        loop_result = await run_stage(
            loop_expand_steps,
            make_loop_expand_dispatch(
                query=query, chunks=chunks,
                is_factoid=meta.is_factoid,
                retrieval_plan=meta.retrieval_plan,
            ),
            parallel=False,
        )
        traces.extend(loop_result.traces)

        loop_reqs = list(chain.from_iterable(loop_result.domain))
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
        traces.append({
            "kind": "action", "label": "RetrievalLoop",
            "content": f"Round {round_idx + 2}: fetched {len(extra)} extra chunks",
        })

    # -- Finalize
    chunks, finalize_traces = _finalize(finalize_steps, chunks=chunks)
    traces.extend(finalize_traces)

    return RetrievalResult(chunks=chunks), traces


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


# ── Finalize ─────────────────────────────────────────────


def _finalize(
    steps: list[StitchStep],
    *,
    chunks: list[ChunkResult],
) -> tuple[list[ChunkResult], list[dict[str, Any]]]:
    """Run finalize steps (stitching) on the final chunk set."""
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
