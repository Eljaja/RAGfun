"""Pipeline: configure -> retrieve -> generate -> evaluate -> result.

Single async function, no streaming, no mutable context.
The caller (endpoints.py) owns retry and presentation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import Settings
from context import extract_source_details, history_summary
from engine.budget import BudgetCounter
from engine.retrieval import run_retrieval
from llm import LLMClient
from models.plan import BrainRound, PipelineResult
from steps.configure import configure
from steps.evaluate import evaluate
from steps.generate import generate

logger = logging.getLogger(__name__)


async def run_pipeline(
    plan: BrainRound,
    *,
    project_id: str,
    query: str,
    history: list[dict[str, str]] | None = None,
    include_sources: bool = True,
    llm: LLMClient,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> PipelineResult:
    """Execute a single pipeline pass. Returns a PipelineResult."""
    raw_history = history or []
    history_text = history_summary(raw_history)
    model = settings.agent_llm_model or settings.llm_model
    budget = BudgetCounter(plan.max_llm_calls)
    traces: list[dict[str, Any]] = []

    # 1. Configure
    meta = await configure(
        plan.configure,
        query=query, history_text=history_text, budget=budget,
        llm=llm, model=model, settings=settings,
    )
    traces.extend(meta.traces)

    # 2. Retrieve
    ret = await run_retrieval(
        query=query,
        meta=meta,
        history_text=history_text,
        initial_expand_steps=list(plan.retrieval.initial_expand),
        loop_check_steps=list(plan.retrieval.loop_check),
        loop_expand_steps=list(plan.retrieval.loop_expand),
        finalize_steps=list(plan.retrieval.finalize),
        max_rounds=plan.retrieval.max_rounds,
        default_preset=plan.retrieval.default_config.preset,
        default_top_k=plan.retrieval.default_config.top_k,
        default_rerank=plan.retrieval.default_config.rerank,
        project_id=project_id,
        http_client=http_client,
        settings=settings,
        budget=budget,
        llm=llm,
        model=model,
    )
    traces.extend(ret.traces)

    if not ret.chunks:
        return PipelineResult(
            answer="", chunks=[], sources=[],
            mode=meta.retrieval_mode, lang=meta.lang,
            is_factoid=meta.is_factoid, traces=traces,
        )

    # 3. Generate
    answer = await generate(
        plan.generate,
        chunks=ret.chunks, query=query, lang=meta.lang,
        history=raw_history, source_meta={},
        max_context_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        llm=llm, model=model,
    )

    # 4. Evaluate
    verdict = await evaluate(
        plan.evaluate,
        answer=answer, chunks=ret.chunks, query=query,
        history_text=history_text,
        is_factoid=meta.is_factoid, source_meta={},
        budget=budget, llm=llm, model=model,
        settings=settings,
    )
    traces.extend(verdict.traces)

    final_answer = verdict.answer if verdict.answer is not None else answer
    final_chunks = verdict.chunks if verdict.chunks is not None else ret.chunks

    sources: list[dict[str, Any]] = []
    if include_sources:
        sources = extract_source_details(final_chunks)

    return PipelineResult(
        answer=final_answer,
        chunks=final_chunks,
        sources=sources,
        mode=meta.retrieval_mode,
        lang=meta.lang,
        is_factoid=meta.is_factoid,
        needs_retry=verdict.needs_retry,
        missing_terms=verdict.missing_terms,
        requery=verdict.requery,
        traces=traces,
    )
