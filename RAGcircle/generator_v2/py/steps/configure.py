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
from engine.trace_collector import TraceCollector
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


# TODO 
# method is too opinionated in fact
# so what would be nice is to secure the structure of dsl more and make some steps 
# compulsory 
async def configure(
    steps: list[ConfigStep],
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
    collector: TraceCollector,
) -> ConfigMeta:
    """Run all configure steps, return metadata for downstream phases."""
    lang = "English"
    is_factoid = False
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode = "hybrid"

    for step in steps:
        match step:
            case PlanLLMStep():
                rp, rm = await _plan_retrieval(
                    query=query, history_text=history_text, budget=budget,
                    llm=llm, model=model, settings=settings, collector=collector,
                )
                retrieval_plan = rp
                retrieval_mode = rm

            case DetectLangStep():
                lang, is_factoid = await _detect_lang(
                    query=query, budget=budget, llm=llm, model=model,
                    collector=collector,
                )

    return ConfigMeta(
        lang=lang,
        is_factoid=is_factoid,
        retrieval_plan=retrieval_plan,
        retrieval_mode=retrieval_mode,
    )


# ── Individual step implementations ──────────────────────


async def _plan_retrieval(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
    collector: TraceCollector,
) -> tuple[ExecutionPlan, str]:
    fallback_mode, fallback_plan = _FALLBACK

    await collector.emit(
        {"kind": "tool", "name": "llm.plan", "payload": {"query": query}},
    )

    # TODO: I do not like this method at all 
    # And I did not like that it created a bug when it was not executed
    # but some basic config was provided
    # Oh and it should definetly return the actual plan 
    if not budget.try_consume():
        return fallback_plan, fallback_mode

    try:
        raw = await asyncio.wait_for(
            llm.complete_json(
                model,
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {
                        "role": "user",
                        "content": PLAN_USER.format(
                            history=history_text,
                            query=query,
                        ),
                    },
                ],
                temperature=0.0,
            ),
            timeout=settings.agent_llm_timeout + 10,
        )
    except LLMTransportError as exc:
        logger.warning("Plan LLM unavailable: %s", exc)
        return fallback_plan, fallback_mode
    except LLMParseError as exc:
        logger.warning(
            "Plan LLM parse failed: %s — raw: %s",
            exc,
            exc.raw[:300],
        )
        return fallback_plan, fallback_mode
    except asyncio.TimeoutError:
        logger.warning("Plan LLM timed out")
        return fallback_plan, fallback_mode

    retrieval_mode = str(raw.get("retrieval_mode") or "hybrid")
    reason = str(raw.get("reason") or "")
    plan = from_llm_plan(
        raw,
        top_k_min=settings.agent_top_k_min,
        top_k_max=settings.agent_top_k_max,
    )
    await collector.emit(
        {
            "kind": "thought",
            "label": "Plan",
            "content": f"mode={retrieval_mode}. Reason: {reason}",
        },
    )
    return plan, retrieval_mode


async def _detect_lang(
    *,
    query: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    collector: TraceCollector,
) -> tuple[str, bool]:
    lang = "English"
    used_llm = False
    attempted_llm = budget.try_consume()

    if attempted_llm:
        await collector.emit(
            {
                "kind": "tool",
                "name": "llm.detect_lang",
                "payload": {"query_preview": query[:240]},
            },
        )
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
            used_llm = True
        except LLMTransportError as exc:
            logger.warning("detect_lang LLM unavailable: %s", exc)

    is_factoid = is_factoid_question(query)

    if used_llm:
        note = ""
    elif attempted_llm:
        note = " [lang default after LLM failure]"
    else:
        note = " [lang default: no LLM budget]"

    logger.info(
        "detect_lang: lang=%r is_factoid=%s via_llm=%s",
        lang,
        is_factoid,
        used_llm,
    )
    await collector.emit(
        {
            "kind": "thought",
            "label": "DetectLang",
            "content": f"lang={lang}, is_factoid={is_factoid}{note}",
        },
    )

    return lang, is_factoid
