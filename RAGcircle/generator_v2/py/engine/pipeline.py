"""Pipeline: expand -> retrieve -> enrich -> generate -> evaluate -> result.

Single async function, no streaming, no mutable context.
The caller (endpoints.py) owns retry and presentation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import Settings
from context import extract_source_details, history_summary, merge_chunks
from engine.budget import BudgetCounter
from llm import LLMClient
from models.chunks import ChunkResult
from models.plan import BrainRound, PipelineResult
from steps.enrich import enrich
from steps.evaluate import evaluate
from steps.expand import expand
from steps.generate import generate
from steps.retrieve import retrieve_all

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

    # 1. Expand
    exp = await expand(
        plan.expand,
        query=query, project_id=project_id, lang="English",
        history_text=history_text, budget=budget,
        llm=llm, model=model, http_client=http_client, settings=settings,
    )
    traces.extend(exp.traces)

    # 2. Retrieve
    chunks = await retrieve_all(
        exp.queries,
        plan=exp.retrieval_plan,
        preset=plan.retrieve.preset,
        top_k=plan.retrieve.top_k,
        rerank=plan.retrieve.rerank,
        project_id=project_id,
        http_client=http_client, settings=settings,
    )

    if exp.extra_chunks:
        chunks = merge_chunks([chunks, exp.extra_chunks])

    if not chunks:
        return PipelineResult(
            answer="", chunks=[], sources=[],
            mode=exp.retrieval_mode, lang=exp.lang,
            is_factoid=exp.is_factoid, traces=traces,
        )

    # 3. Enrich
    chunks, enrich_traces = await enrich(
        plan.post_retrieve,
        chunks=chunks, query=query, project_id=project_id,
        is_factoid=exp.is_factoid, retrieval_plan=exp.retrieval_plan,
        http_client=http_client, settings=settings,
    )
    traces.extend(enrich_traces)

    # 4. Generate
    answer = await generate(
        plan.generate,
        chunks=chunks, query=query, lang=exp.lang,
        history=raw_history, source_meta={},
        max_context_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        llm=llm, model=model,
    )

    # 5. Evaluate
    verdict = await evaluate(
        plan.evaluate,
        answer=answer, chunks=chunks, query=query,
        project_id=project_id, history=raw_history,
        history_text=history_text, lang=exp.lang,
        is_factoid=exp.is_factoid, source_meta={},
        budget=budget, llm=llm, model=model,
        http_client=http_client, settings=settings,
    )
    traces.extend(verdict.traces)

    final_answer = verdict.answer if verdict.answer is not None else answer
    final_chunks = verdict.chunks if verdict.chunks is not None else chunks

    # Build sources
    sources: list[dict[str, Any]] = []
    if include_sources:
        sources = extract_source_details(final_chunks)

    return PipelineResult(
        answer=final_answer,
        chunks=final_chunks,
        sources=sources,
        mode=exp.retrieval_mode,
        lang=exp.lang,
        is_factoid=exp.is_factoid,
        needs_retry=verdict.needs_retry,
        missing_terms=verdict.missing_terms,
        requery=verdict.requery,
        traces=traces,
    )
