"""Evaluate phase: answer + chunks in, verdict out.

Leaf functions return (dict, traces).  The dict carries partial Verdict
fields; the caller merges them with merge_partials.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from config import Settings
from lib.context import build_context
from engine.brain_budget import BudgetCounter
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.assessment import AssessmentResult, ReflectionResult
from retrieval_contract import ChunkResult
from models.steps import (
    AssessStep,
    EvalStep,
    GroundingCheckStep,
    ReflectStep,
)
from steps.prompts import (
    ASSESS_SYSTEM,
    ASSESS_USER,
    REFLECT_SYSTEM,
    REFLECT_USER,
)
from steps.query_heuristics import answer_is_grounded

logger = logging.getLogger(__name__)


def make_eval_dispatch(
    *,
    answer: str,
    chunks: list[ChunkResult],
    query: str,
    history_text: str,
    is_factoid: bool,
    source_meta: dict[str, dict[str, Any]],
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> Callable[[EvalStep], Any]:
    """Return a dispatch closure for evaluate steps."""

    def dispatch(step: EvalStep):
        match step:
            case ReflectStep():
                return _reflect(
                    answer=answer, chunks=chunks, query=query,
                    source_meta=source_meta, budget=budget,
                    llm=llm, model=model, settings=settings,
                )
            case AssessStep():
                return _assess(
                    answer=answer, query=query, history_text=history_text,
                    budget=budget, llm=llm, model=model,
                )
            case GroundingCheckStep():
                return _grounding_check(
                    answer=answer, chunks=chunks,
                    is_factoid=is_factoid, source_meta=source_meta,
                    settings=settings,
                )
            case _:
                raise TypeError(f"unknown eval step: {type(step).__name__}")

    return dispatch


# ── Individual step implementations ──────────────────────


async def _reflect(
    *, answer: str, chunks: list[ChunkResult], query: str,
    source_meta: dict[str, dict[str, Any]],
    budget: BudgetCounter, llm: LLMClient, model: str,
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not budget.try_consume() or not answer:
        return {}, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.reflect", "payload": {}}]
    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
    )

    try:
        result = await llm.complete_model(
            model,
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": REFLECT_USER.format(
                    query=query, context=context_text, answer=answer,
                )},
            ],
            ReflectionResult,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reflection-result",
                    "schema": ReflectionResult.model_json_schema(),
                },
            },
        )
    except LLMTransportError as exc:
        logger.warning("Reflection LLM unavailable: %s", exc)
        return {}, traces
    except LLMParseError as exc:
        logger.warning("Reflection parse failed: %s — raw: %s", exc, exc.raw[:300])
        return {}, traces

    if not result.complete and result.requery:
        traces.append({
            "kind": "thought", "label": "Reflect",
            "content": f"Incomplete. Requery: {result.requery}",
        })
        return {"requery": result.requery}, traces

    if result.complete:
        traces.append({"kind": "thought", "label": "Reflect", "content": "Answer is complete"})
    return {}, traces


async def _assess(
    *, answer: str, query: str, history_text: str,
    budget: BudgetCounter, llm: LLMClient, model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not budget.try_consume() or not answer:
        return {}, []

    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.assess", "payload": {}}]

    try:
        assessment = await llm.complete_model(
            model,
            [
                {"role": "system", "content": ASSESS_SYSTEM},
                {"role": "user", "content": ASSESS_USER.format(
                    history=history_text, question=query, answer=answer,
                )},
            ],
            AssessmentResult,
            temperature=0.0,
        )
    except LLMTransportError as exc:
        logger.warning("Assessment LLM unavailable: %s", exc)
        return {}, traces
    except LLMParseError as exc:
        logger.warning("Assessment parse failed: %s — raw: %s", exc, exc.raw[:300])
        return {}, traces

    if not assessment.incomplete:
        return {}, traces

    traces.append({
        "kind": "thought", "label": "Assess",
        "content": f"Incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
    })
    return {"missing_terms": assessment.missing_terms}, traces


def _grounding_check(
    *,
    answer: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
    source_meta: dict[str, dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check if the answer is grounded in chunks."""
    if not is_factoid or not answer or not chunks:
        return {}, []

    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )

    if answer_is_grounded(answer=answer, context_text=context_text):
        return {}, []

    return (
        {"missing_terms": ["answer not grounded"]},
        [{"kind": "thought", "label": "GroundingCheck",
          "content": "Answer not grounded in retrieved context"}],
    )
