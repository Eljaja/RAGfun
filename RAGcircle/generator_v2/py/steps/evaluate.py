"""Evaluate-phase step handlers: answer quality checks.

Handlers: reflect, assess, supplemental_retrieve, factoid_retry.

Key change from old code: AssessStep now ONLY does the LLM assessment and
writes missing_terms to ctx. The re-retrieval + re-generation are separate
composable steps (supplemental_retrieve + a second generate round).

FactoidRetryStep now only does the grounding check + within-doc expand.
If grounding fails, it expands chunks and re-generates inline (kept atomic
because the grounding swap logic is tightly coupled to the re-generation).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from context import build_context, history_as_messages, merge_chunks
from engine.env import StepEnv
from engine.registry import step_handler
from models.assessment import AssessmentResult, ReflectionResult
from models.events import Event, TokenEvent, TraceEvent
from plan_builder import from_preset
from prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    ASSESS_SYSTEM,
    ASSESS_USER,
    REFLECT_SYSTEM,
    REFLECT_USER,
)
from query_variants import answer_is_grounded
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


# ── reflect ──────────────────────────────────────────────

@step_handler("reflect")
async def run_reflect(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.budget.try_consume() or not env.ctx.answer:
        return

    yield TraceEvent(kind="tool", name="llm.reflect", payload={})

    context_text = build_context(
        env.ctx.chunks,
        max_chars=env.settings.max_context_chars,
        max_chunk_chars=env.settings.max_chunk_chars,
        source_meta=env.ctx.source_meta,
    )

    try:
        raw = await env.llm.complete(
            env.model,
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": REFLECT_USER.format(
                    query=env.ctx.query, context=context_text, answer=env.ctx.answer,
                )},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reflection-result",
                    "schema": ReflectionResult.model_json_schema(),
                },
            },
        )
        data = json.loads(raw)
        result = ReflectionResult(
            complete=data.get("complete", True),
            missing_context=data.get("missing_context"),
            requery=data.get("requery"),
        )
    except Exception:
        logger.warning("Reflection failed, continuing")
        return

    if not result.complete and result.requery:
        env.ctx.search_query = result.requery
        yield TraceEvent(
            kind="thought", label="Reflect",
            content=f"Incomplete. Requery: {result.requery}",
        )
    elif result.complete:
        yield TraceEvent(kind="thought", label="Reflect", content="Answer is complete")

    return
    yield  # noqa


# ── assess (atomic: just the LLM check) ─────────────────

@step_handler("assess")
async def run_assess(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.budget.try_consume() or not env.ctx.answer:
        return

    yield TraceEvent(kind="tool", name="llm.assess", payload={})

    raw = await env.llm.complete(
        env.model,
        [
            {"role": "system", "content": ASSESS_SYSTEM},
            {"role": "user", "content": ASSESS_USER.format(
                history=env.ctx.history_text,
                question=env.ctx.query,
                answer=env.ctx.answer,
            )},
        ],
        temperature=0.0,
    )

    try:
        data = json.loads(strip_thinking(raw))
        if not isinstance(data, dict):
            return
        missing = data.get("missing_terms") or []
        if not isinstance(missing, list):
            missing = []
        assessment = AssessmentResult(
            incomplete=bool(data.get("incomplete")),
            missing_terms=[str(t).strip() for t in missing if str(t).strip()],
            reason=str(data.get("reason") or ""),
        )
    except Exception:
        logger.warning("Failed to parse assessment result")
        return

    if not assessment.incomplete:
        return

    env.ctx.missing_terms = assessment.missing_terms
    yield TraceEvent(
        kind="thought", label="Assess",
        content=f"Incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
    )
    return
    yield  # noqa


# ── supplemental_retrieve (uses missing_terms from assess) ─

@step_handler("supplemental_retrieve")
async def run_supplemental_retrieve(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.missing_terms:
        return

    retry_queries = list(env.ctx.missing_terms)

    if env.ctx.budget.try_consume():
        try:
            raw = await env.llm.complete(
                env.model,
                [
                    {"role": "system", "content": "Extract short keyword queries from the user request."},
                    {"role": "user", "content": (
                        f'{env.ctx.history_text}'
                        f'Return JSON: {{"keywords": [..]}} with 3-6 short keyword phrases. '
                        f'Query: {env.ctx.query}'
                    )},
                ],
                temperature=0.0,
            )
            data = json.loads(strip_thinking(raw))
            kw = [str(q).strip() for q in (data.get("keywords") or []) if str(q).strip()]
            retry_queries.extend(kw)
        except Exception:
            pass

    retry_plan = from_preset("thorough", top_k=10, rerank=True)
    retry_results = await asyncio.gather(*(
        _safe_retrieve(env, rq, retry_plan)
        for rq in retry_queries[:step.max_queries]
    ))
    env.ctx.chunks = merge_chunks([env.ctx.chunks, *retry_results])

    context_text = build_context(
        env.ctx.chunks,
        max_chars=env.settings.max_context_chars,
        max_chunk_chars=env.settings.max_chunk_chars,
        source_meta=env.ctx.source_meta,
    )
    system_prompt = ANSWER_SYSTEM.format(lang=env.ctx.lang)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(env.ctx.history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=env.ctx.query, context=context_text,
    )})

    env.ctx.answer = ""
    async for token in env.llm.stream(env.model, messages):
        env.ctx.answer += token
        yield TokenEvent(content=token)

    env.ctx.missing_terms = []


# ── factoid_retry ────────────────────────────────────────

@step_handler("factoid_retry")
async def run_factoid_retry(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.is_factoid or not env.ctx.answer or not env.ctx.chunks:
        return

    context_text = build_context(
        env.ctx.chunks,
        max_chars=env.settings.max_context_chars,
        max_chunk_chars=env.settings.max_chunk_chars,
        source_meta=env.ctx.source_meta,
    )

    if answer_is_grounded(answer=env.ctx.answer, context_text=context_text):
        return

    yield TraceEvent(
        kind="thought", label="FactoidRetry",
        content="Answer not grounded in context, attempting recovery",
    )

    expand_plan = from_preset("fast", top_k=5, rerank=False)
    top_sources = list({c.source_id for c in env.ctx.chunks[:3]})
    if top_sources:
        expand_results = await asyncio.gather(*(
            _safe_retrieve(env, env.ctx.query, expand_plan)
            for _ in top_sources
        ))
        env.ctx.chunks = merge_chunks([env.ctx.chunks, *expand_results])

    new_context = build_context(
        env.ctx.chunks,
        max_chars=env.settings.max_context_chars,
        max_chunk_chars=env.settings.max_chunk_chars,
        source_meta=env.ctx.source_meta,
    )
    system_prompt = ANSWER_SYSTEM.format(lang=env.ctx.lang)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(env.ctx.history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=env.ctx.query, context=new_context,
    )})

    new_answer = ""
    async for token in env.llm.stream(env.model, messages):
        new_answer += token

    if answer_is_grounded(answer=new_answer, context_text=new_context):
        env.ctx.answer = new_answer
        yield TokenEvent(content=env.ctx.answer)
        yield TraceEvent(
            kind="thought", label="FactoidRetry",
            content="Recovery succeeded, answer replaced",
        )
    else:
        yield TraceEvent(
            kind="thought", label="FactoidRetry",
            content="Recovery did not improve grounding, keeping original",
        )
