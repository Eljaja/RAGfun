"""Configure phase: detect language, plan retrieval strategy.

Leaf functions return (dict, traces).  The dict carries partial ConfigMeta
fields; the caller merges them with merge_partials.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from config import Settings
from engine.brain_budget import BudgetCounter
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.steps import ConfigStep, DetectLangStep, PlanLLMStep
from retrieval_contract import from_llm_plan, from_preset
from steps.prompts import DETECT_LANG_SYSTEM, DETECT_LANG_USER, PLAN_SYSTEM, PLAN_USER
from steps.query_heuristics import is_factoid_question

logger = logging.getLogger(__name__)

_FALLBACK = ("hybrid", from_preset("hybrid", top_k=10, rerank=True))


def make_config_dispatch(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> Callable[[ConfigStep], Any]:
    """Return a dispatch closure for configure steps."""

    def dispatch(step: ConfigStep):
        match step:
            case PlanLLMStep():
                return _plan_retrieval(
                    query=query, history_text=history_text, budget=budget,
                    llm=llm, model=model, settings=settings,
                )
            case DetectLangStep():
                return _detect_lang(
                    query=query, budget=budget, llm=llm, model=model,
                )
            case _:
                raise TypeError(f"unknown config step: {type(step).__name__}")

    return dispatch


# ── Individual step implementations ──────────────────────


async def _plan_retrieval(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.plan", "payload": {"query": query}},
    ]
    fallback_mode, fallback_plan = _FALLBACK

    if not budget.try_consume():
        return (
            {"retrieval_plan": fallback_plan, "retrieval_mode": fallback_mode},
            traces,
        )

    try:
        raw = await asyncio.wait_for(
            llm.complete_json(
                model,
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {
                        "role": "user",
                        "content": PLAN_USER.format(
                            history=history_text, query=query,
                        ),
                    },
                ],
                temperature=0.0,
            ),
            timeout=settings.agent_llm_timeout + 10,
        )
    except LLMTransportError as exc:
        logger.warning("Plan LLM unavailable: %s", exc)
        return (
            {"retrieval_plan": fallback_plan, "retrieval_mode": fallback_mode},
            traces,
        )
    except LLMParseError as exc:
        logger.warning("Plan LLM parse failed: %s — raw: %s", exc, exc.raw[:300])
        return (
            {"retrieval_plan": fallback_plan, "retrieval_mode": fallback_mode},
            traces,
        )
    except asyncio.TimeoutError:
        logger.warning("Plan LLM timed out")
        return (
            {"retrieval_plan": fallback_plan, "retrieval_mode": fallback_mode},
            traces,
        )

    retrieval_mode = str(raw.get("retrieval_mode") or "hybrid")
    reason = str(raw.get("reason") or "")
    plan = from_llm_plan(
        raw,
        top_k_min=settings.agent_top_k_min,
        top_k_max=settings.agent_top_k_max,
    )
    traces.append({
        "kind": "thought",
        "label": "Plan",
        "content": f"mode={retrieval_mode}. Reason: {reason}",
    })
    return {"retrieval_plan": plan, "retrieval_mode": retrieval_mode}, traces


async def _detect_lang(
    *,
    query: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    lang = "English"
    used_llm = False
    attempted_llm = budget.try_consume()

    if attempted_llm:
        traces.append({
            "kind": "tool",
            "name": "llm.detect_lang",
            "payload": {"query_preview": query[:240]},
        })
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
        "detect_lang: lang=%r is_factoid=%s via_llm=%s", lang, is_factoid, used_llm,
    )
    traces.append({
        "kind": "thought",
        "label": "DetectLang",
        "content": f"lang={lang}, is_factoid={is_factoid}{note}",
    })

    return {"lang": lang, "is_factoid": is_factoid}, traces
