"""Configure phase: detect language, plan retrieval strategy.

Pure functions: (query, history, budget, deps) -> ConfigMeta.
Runs before retrieval — no chunks yet, no generation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import Settings
from engine.brain_budget import BudgetCounter
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.plan import ConfigMeta
from retrieval_contract import ExecutionPlan
from models.steps import ConfigStep, DetectLangStep, PlanLLMStep
from retrieval_contract import from_llm_plan, from_preset
from steps.prompts import DETECT_LANG_SYSTEM, DETECT_LANG_USER, PLAN_SYSTEM, PLAN_USER
from steps.query_heuristics import is_factoid_question

logger = logging.getLogger(__name__)

_DEFAULT_PLAN_DICT: dict[str, Any] = {
    "retrieval_mode": "hybrid",
    "top_k": 10,
    "rerank": True,
    "use_hyde": False,
    "reason": "fallback",
}

_FALLBACK = ("hybrid", from_preset("hybrid", top_k=10, rerank=True))


async def configure(
    steps: list[ConfigStep],
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> ConfigMeta:
    """Run all configure steps, return metadata for downstream phases."""
    lang = "English"
    is_factoid = False
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode = "hybrid"
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

    return ConfigMeta(
        lang=lang,
        is_factoid=is_factoid,
        retrieval_plan=retrieval_plan,
        retrieval_mode=retrieval_mode,
        traces=traces,
    )


# ── Individual step implementations ──────────────────────


async def _plan_retrieval(
    *, query: str, history_text: str, budget: BudgetCounter,
    llm: LLMClient, model: str, settings: Settings,
) -> tuple[ExecutionPlan, str, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = [{"kind": "tool", "name": "llm.plan", "payload": {"query": query}}]
    fallback_mode, fallback_plan = _FALLBACK

    if not budget.try_consume():
        return fallback_plan, fallback_mode, traces

    try:
        raw = await asyncio.wait_for(
            llm.complete_json(
                model,
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user", "content": PLAN_USER.format(
                        history=history_text, query=query,
                    )},
                ],
                temperature=0.0,
            ),
            timeout=settings.agent_llm_timeout + 10,
        )
    except LLMTransportError as exc:
        logger.warning("Plan LLM unavailable: %s", exc)
        return fallback_plan, fallback_mode, traces
    except LLMParseError as exc:
        logger.warning("Plan LLM parse failed: %s — raw: %s", exc, exc.raw[:300])
        return fallback_plan, fallback_mode, traces
    except asyncio.TimeoutError:
        logger.warning("Plan LLM timed out")
        return fallback_plan, fallback_mode, traces

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


async def _detect_lang(
    *, query: str, budget: BudgetCounter, llm: LLMClient, model: str,
) -> tuple[str, bool, list[dict[str, Any]]]:
    lang = "English"
    if budget.try_consume():
        try:
            raw = await llm.complete(
                model,
                [
                    {"role": "system", "content": DETECT_LANG_SYSTEM},
                    {"role": "user", "content": DETECT_LANG_USER.format(text=query)},
                ],
                temperature=0.0,
            )
            lang = (raw or "").strip() or "English"
        except LLMTransportError as exc:
            logger.warning("detect_lang LLM unavailable: %s", exc)
    is_factoid = is_factoid_question(query)
    return lang, is_factoid, []
