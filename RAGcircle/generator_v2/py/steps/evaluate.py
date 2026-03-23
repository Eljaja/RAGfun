"""Evaluate phase: answer + chunks in, verdict out.

Pure judging: no retrieval, no generation. Returns a Verdict.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings
from lib.context import build_context
from engine.budget import BudgetCounter
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.assessment import AssessmentResult, ReflectionResult
from retrieval_contract import ChunkResult
from models.plan import Verdict
from models.steps import (
    AssessStep,
    EvalStep,
    GroundingCheckStep,
    ReflectStep,
)
from lib.prompts import (
    ASSESS_SYSTEM,
    ASSESS_USER,
    REFLECT_SYSTEM,
    REFLECT_USER,
)
from lib.query_variants import answer_is_grounded

logger = logging.getLogger(__name__)


async def evaluate(
    steps: list[EvalStep],
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
) -> Verdict:
    """Run all evaluate steps. Returns a Verdict — no retrieval, no generation."""
    traces: list[dict[str, Any]] = []
    missing_terms: list[str] = []
    requery: str | None = None

    for step in steps:
        match step:
            case ReflectStep():
                rq, t = await _reflect(
                    answer=answer, chunks=chunks, query=query,
                    source_meta=source_meta, budget=budget,
                    llm=llm, model=model, settings=settings,
                )
                if rq:
                    requery = rq
                traces.extend(t)

            case AssessStep():
                mt, t = await _assess(
                    answer=answer, query=query, history_text=history_text,
                    budget=budget, llm=llm, model=model,
                )
                missing_terms = mt
                traces.extend(t)

            case GroundingCheckStep():
                grounded, t = await _grounding_check(
                    answer=answer, chunks=chunks,
                    is_factoid=is_factoid, source_meta=source_meta,
                    settings=settings,
                )
                if not grounded:
                    missing_terms = ["answer not grounded"]
                traces.extend(t)

    return Verdict(
        needs_retry=bool(missing_terms) or bool(requery),
        missing_terms=missing_terms,
        requery=requery,
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
        return None, traces
    except LLMParseError as exc:
        logger.warning("Reflection parse failed: %s — raw: %s", exc, exc.raw[:300])
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
        return [], traces
    except LLMParseError as exc:
        logger.warning("Assessment parse failed: %s — raw: %s", exc, exc.raw[:300])
        return [], traces

    if not assessment.incomplete:
        return [], traces

    traces.append({
        "kind": "thought", "label": "Assess",
        "content": f"Incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
    })
    return assessment.missing_terms, traces


async def _grounding_check(
    *,
    answer: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
    source_meta: dict[str, dict[str, Any]],
    settings: Settings,
) -> tuple[bool, list[dict[str, Any]]]:
    """Check if the answer is grounded in chunks. Returns (grounded, traces)."""
    if not is_factoid or not answer or not chunks:
        return True, []

    context_text = build_context(
        chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=source_meta,
    )

    if answer_is_grounded(answer=answer, context_text=context_text):
        return True, []

    return False, [{
        "kind": "thought", "label": "GroundingCheck",
        "content": "Answer not grounded in retrieved context",
    }]
