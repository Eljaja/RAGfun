"""Retrieve-phase step handler."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from context import merge_chunks
from engine.env import StepEnv
from engine.registry import step_handler
from models.events import Event, RetrievalEvent, TraceEvent
from plan_builder import from_preset
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


@step_handler("retrieve")
async def run_retrieve(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    plan = env.ctx.retrieval_plan or from_preset(step.preset, top_k=step.top_k, rerank=step.rerank)

    yield TraceEvent(
        kind="action", label="Retrieving",
        content=f"queries={len(env.ctx.search_queries)}, first={env.ctx.search_query[:80]}...",
    )

    if len(env.ctx.search_queries) <= 1:
        new_chunks = await _safe_retrieve(env, env.ctx.search_query, plan)
    else:
        results = await asyncio.gather(*(
            _safe_retrieve(env, sq, plan)
            for sq in env.ctx.search_queries
        ))
        new_chunks = merge_chunks(list(results))

    if env.ctx.chunks:
        env.ctx.chunks = merge_chunks([env.ctx.chunks, new_chunks])
    else:
        env.ctx.chunks = new_chunks

    yield RetrievalEvent(
        mode=env.ctx.retrieval_mode,
        partial=False,
        degraded=[],
        context=[
            {"source_id": c.source_id, "text": c.text[:200], "score": c.score}
            for c in env.ctx.chunks[:10]
        ],
    )
