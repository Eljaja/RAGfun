"""Expand phase: query preparation before retrieval.

Pure functions: (query, lang, history, budget, deps) -> ExpandResult.
No mutable context, no env god object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from config import Settings
from context import merge_chunks
from engine.budget import BudgetCounter
from llm import LLMClient
from models.chunks import ChunkResult
from models.plan import ExpandResult
from models.retrieval import ExecutionPlan
from models.steps import (
    DetectLangStep,
    ExpandStep,
    FactQueryStep,
    HyDEStep,
    KeywordStep,
    PlanLLMStep,
    QueryVariantsStep,
)
from plan_builder import from_llm_plan, from_preset
from prompts import (
    DETECT_LANG_SYSTEM,
    DETECT_LANG_USER,
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
    PLAN_SYSTEM,
    PLAN_USER,
)
from query_variants import is_factoid_question, query_variants as heuristic_variants
from retrieval_client import retrieve as retrieval_call
from shared import strip_thinking

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_PLAN_DICT = {
    "retrieval_mode": "hybrid",
    "top_k": 10,
    "rerank": True,
    "use_hyde": False,
    "reason": "fallback",
}


async def expand(
    steps: list[ExpandStep],
    *,
    query: str,
    project_id: str,
    lang: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> ExpandResult:
    """Run all expand steps, return the accumulated result."""
    queries = [query]
    is_factoid = False
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode = "hybrid"
    extra_chunks: list[ChunkResult] = []
    traces: list[dict[str, Any]] = []

    for step in steps:
        match step:
            case PlanLLMStep():
                rp, rm, t = await _plan_retrieval(
                    query=query, history_text=history_text, budget=budget,
                    llm=llm, model=model, settings=settings,
                )
                retrieval_plan = rp
                retrieval_mode = rm
                traces.extend(t)

            case DetectLangStep():
                lang, is_factoid, t = await _detect_lang(
                    query=query, budget=budget, llm=llm, model=model,
                )
                traces.extend(t)

            case HyDEStep():
                new_query, t = await _hyde(
                    query=query, lang=lang, budget=budget,
                    llm=llm, model=model,
                )
                if new_query:
                    queries[0] = new_query
                traces.extend(t)

            case FactQueryStep(max_queries=max_q):
                chunks, t = await _fact_queries(
                    query=query, project_id=project_id,
                    history_text=history_text,
                    max_queries=max_q, budget=budget,
                    retrieval_plan=retrieval_plan,
                    existing_chunks=extra_chunks,
                    llm=llm, model=model,
                    http_client=http_client, settings=settings,
                )
                extra_chunks = chunks
                traces.extend(t)

            case KeywordStep():
                chunks, t = await _keywords(
                    query=query, project_id=project_id,
                    history_text=history_text,
                    budget=budget, retrieval_plan=retrieval_plan,
                    existing_chunks=extra_chunks,
                    llm=llm, model=model,
                    http_client=http_client, settings=settings,
                )
                extra_chunks = chunks
                traces.extend(t)

            case QueryVariantsStep():
                variants = heuristic_variants(query)
                if len(variants) > 1:
                    queries = variants
                    traces.append({
                        "kind": "action", "label": "QueryVariants",
                        "content": f"Generated {len(variants)} variants",
                    })

    return ExpandResult(
        queries=queries,
        lang=lang,
        is_factoid=is_factoid,
        retrieval_plan=retrieval_plan,
        retrieval_mode=retrieval_mode,
        extra_chunks=extra_chunks,
        traces=traces,
    )


# ── Individual step implementations ──────────────────────


async def _plan_retrieval(
    *, query: str, history_text: str, budget: BudgetCounter,
    llm: LLMClient, model: str, settings: Settings,
) -> tuple[ExecutionPlan, str, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.plan", "payload": {"query": query}}]

    if not budget.try_consume():
        return from_preset("hybrid", top_k=10, rerank=True), "hybrid", traces

    try:
        raw = await asyncio.wait_for(
            _plan_retrieval_llm(llm, model, query, history_text=history_text),
            timeout=settings.agent_llm_timeout + 10,
        )
    except Exception as e:
        logger.warning("Plan LLM failed: %s", e)
        return from_preset("hybrid", top_k=10, rerank=True), "hybrid", traces

    retrieval_mode = str(raw.get("retrieval_mode") or "hybrid")
    reason = str(raw.get("reason") or "")
    plan = from_llm_plan(
        raw,
        top_k_min=settings.agent_top_k_min,
        top_k_max=settings.agent_top_k_max,
    )
    traces.append({
        "kind": "thought", "label": "Plan",
        "content": f"mode={retrieval_mode}. Reason: {reason}",
    })
    return plan, retrieval_mode, traces


async def _plan_retrieval_llm(
    llm: LLMClient, model: str, query: str, *, history_text: str = "",
) -> dict:
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": PLAN_USER.format(history=history_text, query=query)},
        ],
        temperature=0.0,
    )
    cleaned = strip_thinking(raw or "")
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        logger.warning("Failed to parse LLM plan output, using fallback")
        return dict(_DEFAULT_PLAN_DICT)


async def _detect_lang(
    *, query: str, budget: BudgetCounter, llm: LLMClient, model: str,
) -> tuple[str, bool, list[dict[str, Any]]]:
    lang = "English"
    if budget.try_consume():
        raw = await llm.complete(
            model,
            [
                {"role": "system", "content": DETECT_LANG_SYSTEM},
                {"role": "user", "content": DETECT_LANG_USER.format(text=query)},
            ],
            temperature=0.0,
        )
        lang = (raw or "").strip() or "English"
    is_factoid = is_factoid_question(query)
    return lang, is_factoid, []


async def _hyde(
    *, query: str, lang: str, budget: BudgetCounter,
    llm: LLMClient, model: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    if not budget.try_consume():
        return None, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.hyde", "payload": {"query": query}}]
    try:
        passage = await llm.complete(
            model,
            [
                {"role": "system", "content": HYDE_SYSTEM.format(lang=lang)},
                {"role": "user", "content": HYDE_USER.format(query=query, lang=lang)},
            ],
            temperature=0.2,
        )
        return (passage.strip() or None), traces
    except Exception:
        logger.warning("HyDE failed, using original query")
        return None, traces


async def _safe_retrieve(
    query: str, plan: ExecutionPlan, *,
    project_id: str, http_client: httpx.AsyncClient, settings: Settings,
) -> list[ChunkResult]:
    try:
        return await retrieval_call(
            http_client, settings.retrieval_url,
            project_id=project_id, query=query, plan=plan,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


async def _fact_queries(
    *, query: str, project_id: str, history_text: str, max_queries: int,
    budget: BudgetCounter, retrieval_plan: ExecutionPlan | None,
    existing_chunks: list[ChunkResult],
    llm: LLMClient, model: str,
    http_client: httpx.AsyncClient, settings: Settings,
) -> tuple[list[ChunkResult], list[dict[str, Any]]]:
    if not budget.try_consume():
        return existing_chunks, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.fact_split", "payload": {"query": query}}]
    try:
        raw = await llm.complete(
            model,
            [
                {"role": "system", "content": FACT_QUERIES_SYSTEM},
                {"role": "user", "content": FACT_QUERIES_USER.format(
                    history=history_text, query=query,
                )},
            ],
            temperature=0.2,
        )
        data = json.loads(strip_thinking(raw))
        sub_queries = [str(q).strip() for q in (data.get("fact_queries") or []) if str(q).strip()]
        sub_queries = sub_queries[:max_queries]
    except Exception:
        return existing_chunks, traces

    if not sub_queries or retrieval_plan is None:
        return existing_chunks, traces

    fact_results = await asyncio.gather(*(
        _safe_retrieve(fq, retrieval_plan,
                       project_id=project_id, http_client=http_client, settings=settings)
        for fq in sub_queries
    ))
    chunks = merge_chunks([existing_chunks, *fact_results])
    return chunks, traces


async def _keywords(
    *, query: str, project_id: str, history_text: str,
    budget: BudgetCounter, retrieval_plan: ExecutionPlan | None,
    existing_chunks: list[ChunkResult],
    llm: LLMClient, model: str,
    http_client: httpx.AsyncClient, settings: Settings,
) -> tuple[list[ChunkResult], list[dict[str, Any]]]:
    if not budget.try_consume():
        return existing_chunks, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.keywords", "payload": {"query": query}}]
    try:
        raw = await llm.complete(
            model,
            [
                {"role": "system", "content": KEYWORD_QUERIES_SYSTEM},
                {"role": "user", "content": KEYWORD_QUERIES_USER.format(
                    history=history_text, query=query,
                )},
            ],
            temperature=0.0,
        )
        data = json.loads(strip_thinking(raw))
        kw = [str(q).strip() for q in (data.get("keywords") or []) if str(q).strip()]
    except Exception:
        return existing_chunks, traces

    if not kw or retrieval_plan is None:
        return existing_chunks, traces

    kw_results = await asyncio.gather(*(
        _safe_retrieve(k, retrieval_plan,
                       project_id=project_id, http_client=http_client, settings=settings)
        for k in kw[:4]
    ))
    chunks = merge_chunks([existing_chunks, *kw_results])
    return chunks, traces
