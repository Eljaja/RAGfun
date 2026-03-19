"""Post-retrieve step handlers: chunk enrichment after retrieval.

Handlers: quality_check, two_pass, bm25_anchor, factoid_expand, stitch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from context import merge_chunks, stitch_segments
from engine.env import StepEnv
from engine.registry import step_handler
from models.chunks import ScoreSource
from models.events import Event, TraceEvent
from plan_builder import from_preset
from query_variants import extract_hint_terms, keyword_query, unique_source_count
from retrieval_client import retrieve

logger = logging.getLogger(__name__)


async def _safe_retrieve(env: StepEnv, query: str, plan: Any) -> list[Any]:
    try:
        return await retrieve(
            env.http_client, env.settings.retrieval_url,
            project_id=env.ctx.project_id, query=query, plan=plan,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


@step_handler("quality_check")
async def run_quality_check(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    chunks = env.ctx.chunks
    poor = (
        not chunks
        or len(chunks) < step.min_hits
        or (chunks[0].score_source == ScoreSource.RERANK and chunks[0].score < step.min_score)
    )
    if poor:
        yield TraceEvent(kind="thought", label="Quality", content="Retrieval quality is poor")
    return
    yield  # noqa


@step_handler("two_pass")
async def run_two_pass(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    n_unique = unique_source_count(env.ctx.chunks)
    if n_unique >= step.min_unique_sources:
        return

    yield TraceEvent(
        kind="action", label="TwoPass",
        content=f"Only {n_unique} unique sources, generating follow-up query",
    )

    hints = extract_hint_terms(env.ctx.chunks, max_terms=3)
    if not hints:
        return

    follow_up = f"{env.ctx.query} {' '.join(hints)}"
    plan = env.ctx.retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
    extra = await _safe_retrieve(env, follow_up, plan)
    if extra:
        env.ctx.chunks = merge_chunks([env.ctx.chunks, extra])


@step_handler("bm25_anchor")
async def run_bm25_anchor(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    kw = keyword_query(env.ctx.query)
    if not kw:
        return

    yield TraceEvent(kind="action", label="BM25Anchor", content=f"kw={kw[:60]}")

    bm25_plan = from_preset("fast", top_k=step.top_k, rerank=False)
    bm25_hits = await _safe_retrieve(env, kw, bm25_plan)
    if bm25_hits:
        env.ctx.chunks = merge_chunks([env.ctx.chunks, bm25_hits])


@step_handler("factoid_expand")
async def run_factoid_expand(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.is_factoid or not env.ctx.chunks:
        return

    top_sources = list({c.source_id for c in env.ctx.chunks[:3]})
    if not top_sources:
        return

    yield TraceEvent(
        kind="action", label="FactoidExpand",
        content=f"Expanding within {len(top_sources)} top sources",
    )

    expand_plan = from_preset("fast", top_k=5, rerank=False)
    expand_results = await asyncio.gather(*(
        _safe_retrieve(env, env.ctx.query, expand_plan)
        for _ in top_sources
    ))
    env.ctx.chunks = merge_chunks([env.ctx.chunks, *expand_results])


@step_handler("stitch")
async def run_stitch(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    before = len(env.ctx.chunks)
    env.ctx.chunks = stitch_segments(env.ctx.chunks, max_per_segment=step.max_per_segment)
    after = len(env.ctx.chunks)
    if before != after:
        yield TraceEvent(
            kind="action", label="Stitch",
            content=f"Stitched {before} chunks into {after} segments",
        )
    return
    yield  # noqa
