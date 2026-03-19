"""The main execute loop — walks a BrainPlan, dispatches via registry.

This replaces the 300+ line brain_executor.py with a ~40 line loop.
The domain logic lives in steps/*.py; this is pure plumbing.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import Settings
from context import extract_source_details, history_summary
from engine.budget import BudgetCounter
from engine.context import RunContext
from engine.env import StepEnv
from engine.registry import run_steps
from llm import LLMClient
from models.events import DoneEvent, ErrorEvent, Event, ProgressEvent

import steps.expand  # noqa: F401 — registers handlers
import steps.retrieve  # noqa: F401
import steps.enrich  # noqa: F401
import steps.generate  # noqa: F401
import steps.evaluate  # noqa: F401

logger = logging.getLogger(__name__)


async def execute(
    plan: Any,
    *,
    project_id: str,
    query: str,
    history: list[dict[str, str]] | None = None,
    include_sources: bool = True,
    llm: LLMClient,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[Event]:
    model = settings.agent_llm_model or settings.llm_model
    raw_history = history or []

    ctx = RunContext(
        project_id=project_id,
        query=query,
        history=raw_history,
        history_text=history_summary(raw_history),
        budget=BudgetCounter(plan.max_llm_calls),
        include_sources=include_sources,
    )

    env = StepEnv(ctx=ctx, llm=llm, model=model, http_client=http_client, settings=settings)

    for round_idx, brain_round in enumerate(plan.rounds):
        ctx.round_index = round_idx
        yield ProgressEvent(stage="round", content=f"Round {round_idx + 1}/{len(plan.rounds)}")

        async for ev in run_steps(brain_round.expand, env):
            yield ev
            if isinstance(ev, ErrorEvent):
                return

        async for ev in run_steps([brain_round.retrieve], env):
            yield ev
            if isinstance(ev, ErrorEvent):
                return

        if not ctx.chunks:
            yield ErrorEvent(error="No results from retrieval service")
            return

        async for ev in run_steps(brain_round.post_retrieve, env):
            yield ev
            if isinstance(ev, ErrorEvent):
                return

        async for ev in run_steps([brain_round.generate], env):
            yield ev
            if isinstance(ev, ErrorEvent):
                return

        async for ev in run_steps(brain_round.evaluate, env):
            yield ev
            if isinstance(ev, ErrorEvent):
                return

    sources: list[dict[str, Any]] = []
    if ctx.include_sources:
        sources = extract_source_details(ctx.chunks, source_meta=ctx.source_meta)

    yield DoneEvent(
        answer=ctx.answer,
        mode=ctx.retrieval_mode,
        partial=False,
        degraded=[],
        sources=sources,
        context=[
            {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
            for c in ctx.chunks[:10]
        ],
    )
