"""Evaluate phase: answer + chunks in, verdict out.

Pure function: (...) -> Verdict. May re-generate the answer internally
if supplemental retrieval or factoid retry changes the context.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from config import Settings
from context import build_context, history_as_messages, merge_chunks
from engine.budget import BudgetCounter
from llm import LLMClient
from models.assessment import AssessmentResult, ReflectionResult
from models.chunks import ChunkResult
from models.plan import Verdict
from models.steps import (
    AssessStep,
    EvalStep,
    FactoidRetryStep,
    ReflectStep,
    SupplementalRetrieveStep,
)
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
from shared import strip_thinking
from steps.retrieve import _safe_retrieve

logger = logging.getLogger(__name__)


async def evaluate(
    steps: list[EvalStep],
    *,
    answer: str,
    chunks: list[ChunkResult],
    query: str,
    project_id: str,
    history: list[dict[str, str]],
    history_text: str,
    lang: str,
    is_factoid: bool,
    source_meta: dict[str, dict[str, Any]],
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> Verdict:
    """Run all evaluate steps. Returns a Verdict with possible answer replacement."""
    traces: list[dict[str, Any]] = []
    missing_terms: list[str] = []
    requery: str | None = None
    current_answer = answer
    current_chunks = chunks

    for step in steps:
        match step:
            case ReflectStep():
                rq, t = await _reflect(
                    answer=current_answer, chunks=current_chunks, query=query,
                    source_meta=source_meta, budget=budget,
                    llm=llm, model=model, settings=settings,
                )
                if rq:
                    requery = rq
                traces.extend(t)

            case AssessStep():
                mt, t = await _assess(
                    answer=current_answer, query=query, history_text=history_text,
                    budget=budget, llm=llm, model=model,
                )
                missing_terms = mt
                traces.extend(t)

            case SupplementalRetrieveStep(max_queries=max_q):
                if missing_terms:
                    new_answer, new_chunks, t = await _supplemental_retrieve(
                        answer=current_answer, chunks=current_chunks,
                        query=query, project_id=project_id,
                        history=history, history_text=history_text,
                        lang=lang, source_meta=source_meta,
                        missing_terms=missing_terms, max_queries=max_q,
                        budget=budget, llm=llm, model=model,
                        http_client=http_client, settings=settings,
                    )
                    current_answer = new_answer
                    current_chunks = new_chunks
                    missing_terms = []
                    traces.extend(t)

            case FactoidRetryStep():
                new_answer, new_chunks, t = await _factoid_retry(
                    answer=current_answer, chunks=current_chunks,
                    query=query, project_id=project_id,
                    history=history, lang=lang,
                    is_factoid=is_factoid, source_meta=source_meta,
                    llm=llm, model=model,
                    http_client=http_client, settings=settings,
                )
                current_answer = new_answer
                current_chunks = new_chunks
                traces.extend(t)

    answer_changed = current_answer != answer
    return Verdict(
        needs_retry=bool(missing_terms) or bool(requery),
        missing_terms=missing_terms,
        requery=requery,
        answer=current_answer if answer_changed else None,
        chunks=current_chunks if answer_changed else None,
        traces=traces,
    )


# ── Individual step implementations ──────────────────────


async def _reflect(
    *, answer: str, chunks: list[ChunkResult], query: str,
    source_meta: dict[str, dict[str, Any]],
    budget: BudgetCounter, llm: LLMClient, model: str,
    settings: Settings,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Returns (requery_or_None, traces)."""
    if not budget.try_consume() or not answer:
        return None, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.reflect", "payload": {}}]
    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )

    try:
        raw = await llm.complete(
            model,
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": REFLECT_USER.format(
                    query=query, context=context_text, answer=answer,
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
        return None, traces

    if not result.complete and result.requery:
        traces.append({
            "kind": "thought", "label": "Reflect",
            "content": f"Incomplete. Requery: {result.requery}",
        })
        return result.requery, traces

    if result.complete:
        traces.append({"kind": "thought", "label": "Reflect", "content": "Answer is complete"})
    return None, traces


async def _assess(
    *, answer: str, query: str, history_text: str,
    budget: BudgetCounter, llm: LLMClient, model: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Returns (missing_terms, traces)."""
    if not budget.try_consume() or not answer:
        return [], []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.assess", "payload": {}}]

    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": ASSESS_SYSTEM},
            {"role": "user", "content": ASSESS_USER.format(
                history=history_text, question=query, answer=answer,
            )},
        ],
        temperature=0.0,
    )

    try:
        data = json.loads(strip_thinking(raw))
        if not isinstance(data, dict):
            return [], traces
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
        return [], traces

    if not assessment.incomplete:
        return [], traces

    traces.append({
        "kind": "thought", "label": "Assess",
        "content": f"Incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
    })
    return assessment.missing_terms, traces


async def _supplemental_retrieve(
    *, answer: str, chunks: list[ChunkResult],
    query: str, project_id: str,
    history: list[dict[str, str]], history_text: str,
    lang: str, source_meta: dict[str, dict[str, Any]],
    missing_terms: list[str], max_queries: int,
    budget: BudgetCounter, llm: LLMClient, model: str,
    http_client: httpx.AsyncClient, settings: Settings,
) -> tuple[str, list[ChunkResult], list[dict[str, Any]]]:
    """Re-retrieve and re-generate. Returns (new_answer, new_chunks, traces)."""
    retry_queries = list(missing_terms)

    if budget.try_consume():
        try:
            raw = await llm.complete(
                model,
                [
                    {"role": "system", "content": "Extract short keyword queries from the user request."},
                    {"role": "user", "content": (
                        f'{history_text}'
                        f'Return JSON: {{"keywords": [..]}} with 3-6 short keyword phrases. '
                        f'Query: {query}'
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
        _safe_retrieve(rq, retry_plan,
                       project_id=project_id, http_client=http_client, settings=settings)
        for rq in retry_queries[:max_queries]
    ))
    new_chunks = merge_chunks([chunks, *retry_results])

    context_text = build_context(
        new_chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )
    system_prompt = ANSWER_SYSTEM.format(lang=lang)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=query, context=context_text,
    )})

    new_answer = await llm.complete(model, messages)
    return new_answer, new_chunks, []


async def _factoid_retry(
    *, answer: str, chunks: list[ChunkResult],
    query: str, project_id: str,
    history: list[dict[str, str]], lang: str,
    is_factoid: bool, source_meta: dict[str, dict[str, Any]],
    llm: LLMClient, model: str,
    http_client: httpx.AsyncClient, settings: Settings,
) -> tuple[str, list[ChunkResult], list[dict[str, Any]]]:
    """Check grounding, optionally re-generate. Returns (answer, chunks, traces)."""
    if not is_factoid or not answer or not chunks:
        return answer, chunks, []

    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )

    if answer_is_grounded(answer=answer, context_text=context_text):
        return answer, chunks, []

    traces: list[dict[str, Any]] = [{
        "kind": "thought", "label": "FactoidRetry",
        "content": "Answer not grounded in context, attempting recovery",
    }]

    expand_plan = from_preset("fast", top_k=5, rerank=False)
    extra = await _safe_retrieve(
        query, expand_plan,
        project_id=project_id, http_client=http_client, settings=settings,
    )
    new_chunks = merge_chunks([chunks, extra]) if extra else chunks

    new_context = build_context(
        new_chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )
    system_prompt = ANSWER_SYSTEM.format(lang=lang)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=query, context=new_context,
    )})

    new_answer = await llm.complete(model, messages)
    if answer_is_grounded(answer=new_answer, context_text=new_context):
        traces.append({
            "kind": "thought", "label": "FactoidRetry",
            "content": "Recovery succeeded, answer replaced",
        })
        return new_answer, new_chunks, traces

    traces.append({
        "kind": "thought", "label": "FactoidRetry",
        "content": "Recovery did not improve grounding, keeping original",
    })
    return answer, chunks, traces
