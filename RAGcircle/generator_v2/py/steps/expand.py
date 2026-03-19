"""Expand-phase step handlers: query preparation before retrieval.

Handlers: plan_retrieval, detect_lang, hyde, fact_queries, keywords, query_variants.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from context import merge_chunks
from engine.env import StepEnv
from engine.registry import step_handler
from llm import LLMClient
from models.events import Event, TraceEvent
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
from retrieval_client import retrieve
from shared import strip_thinking

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


# ── plan_retrieval ───────────────────────────────────────

@step_handler("plan_retrieval")
async def run_plan_llm(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    yield TraceEvent(kind="tool", name="llm.plan", payload={"query": env.ctx.query})

    if not env.ctx.budget.try_consume():
        env.ctx.retrieval_plan = from_preset("hybrid", top_k=10, rerank=True)
        return

    try:
        raw = await asyncio.wait_for(
            _plan_retrieval_llm(env.llm, env.model, env.ctx.query, history_text=env.ctx.history_text),
            timeout=env.settings.agent_llm_timeout + 10,
        )
    except Exception as e:
        from models.events import ErrorEvent
        yield ErrorEvent(error=f"Plan error: {e!s}")
        return

    env.ctx.retrieval_mode = str(raw.get("retrieval_mode") or "hybrid")
    reason = str(raw.get("reason") or "")
    env.ctx.retrieval_plan = from_llm_plan(
        raw,
        top_k_min=env.settings.agent_top_k_min,
        top_k_max=env.settings.agent_top_k_max,
    )
    yield TraceEvent(
        kind="thought", label="Plan",
        content=f"mode={env.ctx.retrieval_mode}. Reason: {reason}",
    )


_DEFAULT_PLAN_DICT = {
    "retrieval_mode": "hybrid",
    "top_k": 10,
    "rerank": True,
    "use_hyde": False,
    "reason": "fallback",
}


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


# ── detect_lang ──────────────────────────────────────────

@step_handler("detect_lang")
async def run_detect_lang(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if env.ctx.budget.try_consume():
        raw = await env.llm.complete(
            env.model,
            [
                {"role": "system", "content": DETECT_LANG_SYSTEM},
                {"role": "user", "content": DETECT_LANG_USER.format(text=env.ctx.query)},
            ],
            temperature=0.0,
        )
        env.ctx.lang = (raw or "").strip() or "English"
    env.ctx.is_factoid = is_factoid_question(env.ctx.query)
    return
    yield  # noqa: make this an async generator


# ── hyde ─────────────────────────────────────────────────

@step_handler("hyde")
async def run_hyde(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.budget.try_consume():
        return

    yield TraceEvent(kind="tool", name="llm.hyde", payload={"query": env.ctx.query})
    try:
        passage = await env.llm.complete(
            env.model,
            [
                {"role": "system", "content": HYDE_SYSTEM.format(lang=env.ctx.lang)},
                {"role": "user", "content": HYDE_USER.format(query=env.ctx.query, lang=env.ctx.lang)},
            ],
            temperature=0.2,
        )
        if passage.strip():
            env.ctx.search_query = passage
    except Exception:
        logger.warning("HyDE failed, using original query")


# ── fact_queries ─────────────────────────────────────────

@step_handler("fact_queries")
async def run_fact_queries(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.budget.try_consume():
        return

    yield TraceEvent(kind="tool", name="llm.fact_split", payload={"query": env.ctx.query})

    try:
        raw = await env.llm.complete(
            env.model,
            [
                {"role": "system", "content": FACT_QUERIES_SYSTEM},
                {"role": "user", "content": FACT_QUERIES_USER.format(
                    history=env.ctx.history_text, query=env.ctx.query,
                )},
            ],
            temperature=0.2,
        )
        data = json.loads(strip_thinking(raw))
        sub_queries = [str(q).strip() for q in (data.get("fact_queries") or []) if str(q).strip()]
        sub_queries = sub_queries[:step.max_queries]
    except Exception:
        return

    if not sub_queries or env.ctx.retrieval_plan is None:
        return

    fact_results = await asyncio.gather(*(
        _safe_retrieve(env, fq, env.ctx.retrieval_plan)
        for fq in sub_queries
    ))
    env.ctx.chunks = merge_chunks([env.ctx.chunks, *fact_results])


# ── keywords ─────────────────────────────────────────────

@step_handler("keywords")
async def run_keywords(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.budget.try_consume():
        return

    yield TraceEvent(kind="tool", name="llm.keywords", payload={"query": env.ctx.query})

    try:
        raw = await env.llm.complete(
            env.model,
            [
                {"role": "system", "content": KEYWORD_QUERIES_SYSTEM},
                {"role": "user", "content": KEYWORD_QUERIES_USER.format(
                    history=env.ctx.history_text, query=env.ctx.query,
                )},
            ],
            temperature=0.0,
        )
        data = json.loads(strip_thinking(raw))
        kw = [str(q).strip() for q in (data.get("keywords") or []) if str(q).strip()]
    except Exception:
        return

    if not kw or env.ctx.retrieval_plan is None:
        return

    kw_results = await asyncio.gather(*(
        _safe_retrieve(env, k, env.ctx.retrieval_plan)
        for k in kw[:4]
    ))
    env.ctx.chunks = merge_chunks([env.ctx.chunks, *kw_results])
    return
    yield  # noqa


# ── query_variants ───────────────────────────────────────

@step_handler("query_variants")
async def run_query_variants(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    variants = heuristic_variants(env.ctx.query)
    if len(variants) > 1:
        env.ctx.search_queries = variants
        yield TraceEvent(
            kind="action", label="QueryVariants",
            content=f"Generated {len(variants)} variants",
        )
    return
    yield  # noqa
