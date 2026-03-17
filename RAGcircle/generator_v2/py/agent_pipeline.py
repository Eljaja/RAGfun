"""Advanced /agent pipeline: plan -> expand -> retrieve -> quality -> generate -> assess -> retry.

Async generator yielding SSE events. Composes domain modules.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from assessment import assess_completeness, detect_language, quality_is_poor
from config import Settings
from context import (
    build_context,
    extract_source_details,
    history_summary,
    merge_chunks,
)
from llm import LLMClient
from models import AgentRequest, ChunkResult, ExecutionPlan
from plan_builder import DEFAULT_PLAN, from_llm_plan, from_preset
from planning import BudgetCounter, plan_retrieval
from prompts import ANSWER_SYSTEM, ANSWER_SYSTEM_WITH_TOOLS, ANSWER_USER
from query_expansion import fact_queries, hyde, keyword_queries
from retrieval_client import retrieve
from tools import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

AGENT_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {
        "use_hyde": False, "use_fact_queries": False,
        "use_retry": False, "max_llm_calls": 4, "max_fact_queries": 0,
    },
    "conservative": {
        "use_hyde": False, "use_fact_queries": False,
        "use_retry": False, "max_llm_calls": 6, "max_fact_queries": 0,
    },
    "aggressive": {
        "use_hyde": True, "use_fact_queries": True,
        "use_retry": True, "max_llm_calls": 16, "max_fact_queries": 4,
    },
}


def _resolve_option(
    payload_val: Any,
    preset_val: Any,
    settings_val: Any,
) -> Any:
    """Resolve a setting: payload override > preset > settings default."""
    if payload_val is not None:
        return payload_val
    if preset_val is not None:
        return preset_val
    return settings_val


async def _retrieve_with_plan(
    http_client: httpx.AsyncClient,
    retrieval_url: str,
    project_id: str,
    query: str,
    plan: ExecutionPlan,
) -> list[ChunkResult]:
    try:
        return await retrieve(
            http_client, retrieval_url,
            project_id=project_id, query=query, plan=plan,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


async def agent_pipeline(
    request: AgentRequest,
    *,
    llm: LLMClient,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    # ── Resolve options from payload / preset / settings ──
    preset = AGENT_MODE_PRESETS.get((request.mode or "").lower(), {})

    max_llm_calls = _resolve_option(request.max_llm_calls, preset.get("max_llm_calls"), settings.agent_max_llm_calls)
    use_hyde = _resolve_option(request.use_hyde, preset.get("use_hyde"), settings.agent_use_hyde)
    use_fact_q = _resolve_option(request.use_fact_queries, preset.get("use_fact_queries"), settings.agent_use_fact_queries)
    use_retry = _resolve_option(request.use_retry, preset.get("use_retry"), settings.agent_use_retry)
    use_tools = _resolve_option(request.use_tools, None, settings.agent_use_tools)
    max_fact_q = _resolve_option(request.max_fact_queries, preset.get("max_fact_queries"), settings.agent_max_fact_queries)
    hyde_num = max(1, min(7, request.hyde_num or settings.agent_hyde_num))

    model = settings.agent_llm_model or settings.llm_model
    retrieval_url = settings.retrieval_url
    hist = request.history or []
    hist_text = history_summary(hist)
    budget = BudgetCounter(max_llm_calls)

    # ── 1. Plan ──────────────────────────────────────────
    yield {"type": "trace", "kind": "tool", "name": "llm.plan", "payload": {"model": model, "query": request.query}}

    if budget.can_call():
        try:
            plan_dict = await asyncio.wait_for(
                plan_retrieval(llm, model, request.query, history_text=hist_text),
                timeout=settings.agent_llm_timeout + 10,
            )
        except Exception as e:
            yield {"type": "error", "error": f"Plan error: {e!s}"}
            return
    else:
        plan_dict = {"retrieval_mode": "hybrid", "top_k": 10, "rerank": True, "use_hyde": False, "reason": "budget"}

    mode = str(plan_dict.get("retrieval_mode") or "hybrid")
    raw_k = int(request.top_k) if request.top_k is not None else int(plan_dict.get("top_k") or 10)
    top_k = max(settings.agent_top_k_min, min(settings.agent_top_k_max, raw_k))
    rerank = bool(plan_dict.get("rerank", True))
    reason = str(plan_dict.get("reason") or "")

    plan = from_llm_plan(plan_dict, top_k_min=settings.agent_top_k_min, top_k_max=settings.agent_top_k_max)

    yield {
        "type": "trace", "kind": "thought", "label": "Plan",
        "content": f"mode={mode}, top_k={top_k}, rerank={rerank}, hyde={use_hyde}. Reason: {reason}",
    }

    # ── 2. Language detection ────────────────────────────
    if budget.can_call():
        answer_lang = await detect_language(llm, model, request.query)
    else:
        answer_lang = "English"

    # ── 3. HyDE (optional) ───────────────────────────────
    search_query = request.query
    if use_hyde and budget.can_call():
        yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"query": request.query}}
        try:
            hyde_passage = await hyde(llm, model, request.query, lang=answer_lang)
            if hyde_passage.strip():
                search_query = hyde_passage
        except Exception:
            logger.warning("HyDE failed, using original query")

    # ── 4. Retrieve ──────────────────────────────────────
    yield {"type": "trace", "kind": "action", "label": "Retrieving", "content": f"query={search_query[:80]}..."}

    chunks = await _retrieve_with_plan(
        http_client, retrieval_url, request.project_id, search_query, plan,
    )

    if not chunks:
        yield {"type": "error", "error": "No results from retrieval service"}
        return

    # ── 5. Quality check + fact queries ──────────────────
    run_fact = use_fact_q and quality_is_poor(chunks) and budget.can_call()
    if not run_fact and use_fact_q and budget.can_call():
        run_fact = True

    if run_fact and max_fact_q > 0 and budget.can_call():
        yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"query": request.query}}
        try:
            sub_queries = await fact_queries(llm, model, request.query, history_text=hist_text)
            sub_queries = sub_queries[:max_fact_q]
        except Exception:
            sub_queries = []

        if sub_queries:
            fact_plan = from_preset("hybrid", top_k=top_k, rerank=rerank)
            fact_results = await asyncio.gather(*(
                _retrieve_with_plan(
                    http_client, retrieval_url, request.project_id, fq, fact_plan,
                )
                for fq in sub_queries
            ))
            chunks = merge_chunks([chunks, *fact_results])

    yield {
        "type": "retrieval",
        "mode": mode,
        "partial": False,
        "degraded": [],
        "context": [{"source_id": c.source_id, "text": c.text[:200], "score": c.score} for c in chunks[:10]],
    }

    # ── 6. Generate answer ───────────────────────────────
    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
    )

    if use_tools:
        system_prompt = ANSWER_SYSTEM_WITH_TOOLS.format(lang=answer_lang)
    else:
        system_prompt = ANSWER_SYSTEM.format(lang=answer_lang)

    answer_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ANSWER_USER.format(
            history=hist_text, query=request.query, context=context_text,
        )},
    ]

    full_answer = ""
    if use_tools:
        full_answer = await llm.complete_with_tools(model, answer_messages, TOOL_DEFINITIONS)
        yield {"type": "token", "content": full_answer}
    else:
        async for token in llm.stream(model, answer_messages):
            full_answer += token
            yield {"type": "token", "content": token}

    # ── 7. Assess completeness ───────────────────────────
    if use_retry and budget.can_call():
        yield {"type": "trace", "kind": "tool", "name": "llm.assess", "payload": {}}
        assessment = await assess_completeness(
            llm, model, request.query, full_answer, history_text=hist_text,
        )

        # ── 8. Retry if incomplete ───────────────────────
        if assessment.incomplete and budget.can_call():
            yield {
                "type": "trace", "kind": "thought", "label": "Retry",
                "content": f"Answer incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
            }

            retry_queries = list(assessment.missing_terms)
            if budget.can_call():
                try:
                    kw = await keyword_queries(llm, model, request.query, history_text=hist_text)
                    retry_queries.extend(kw)
                except Exception:
                    pass

            if retry_queries:
                retry_plan = from_preset("thorough", top_k=top_k, rerank=True)
                retry_results = await asyncio.gather(*(
                    _retrieve_with_plan(
                        http_client, retrieval_url, request.project_id, rq, retry_plan,
                    )
                    for rq in retry_queries[:4]
                ))
                chunks = merge_chunks([chunks, *retry_results])

                context_text = build_context(
                    chunks,
                    max_chars=settings.max_context_chars,
                    max_chunk_chars=settings.max_chunk_chars,
                )
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ANSWER_USER.format(
                        history=hist_text, query=request.query, context=context_text,
                    )},
                ]
                full_answer = ""
                async for token in llm.stream(model, retry_messages):
                    full_answer += token
                    yield {"type": "token", "content": token}

    # ── Done ─────────────────────────────────────────────
    sources = extract_source_details(chunks) if request.include_sources else []

    yield {
        "type": "done",
        "answer": full_answer,
        "mode": mode,
        "partial": False,
        "degraded": [],
        "sources": sources,
        "context": [{"source_id": c.source_id, "text": c.text[:300], "score": c.score} for c in chunks[:10]],
    }
