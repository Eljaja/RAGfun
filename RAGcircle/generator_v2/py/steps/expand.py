"""Expand step implementations: LLM-based query expansion.

These are the leaf functions called by engine/retrieval._initial_expand().
Each takes (query, budget, llm, ...) and returns (list[str], traces).
No I/O, no retrieval — pure LLM query generators.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from engine.budget import BudgetCounter
from llm import LLMClient
from models.retrieval import ExecutionPlan
from prompts import (
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
)
from shared import strip_thinking

logger = logging.getLogger(__name__)


async def _hyde(
    *, query: str, lang: str, budget: BudgetCounter,
    llm: LLMClient, model: str, num_passages: int = 1,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate hypothetical passages for retrieval.

    When num_passages > 1, generates N passages at escalating temperatures
    (0.2, 0.35, 0.50, ...) for recall diversity, matching the old agent-search
    multi-HyDE strategy.
    """
    actual_n = min(num_passages, budget.remaining)
    if actual_n < 1 or not budget.try_consume():
        return [], []

    messages = [
        {"role": "system", "content": HYDE_SYSTEM.format(lang=lang)},
        {"role": "user", "content": HYDE_USER.format(query=query, lang=lang)},
    ]
    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.hyde",
         "payload": {"query": query, "num": actual_n}},
    ]

    if actual_n == 1:
        try:
            passage = await llm.complete(model, messages, temperature=0.2)
            result = [passage.strip()] if passage and passage.strip() else []
            return result, traces
        except Exception:
            logger.warning("HyDE failed, using original query")
            return [], traces

    for _ in range(actual_n - 1):
        budget.try_consume()

    temps = [0.2 + 0.15 * i for i in range(actual_n)]
    tasks = [llm.complete(model, messages, temperature=t) for t in temps]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    passages = []
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("HyDE passage failed: %s", r)
            continue
        text = (r or "").strip()
        if text:
            passages.append(text)

    traces.append({
        "kind": "thought", "label": "HyDE",
        "content": f"Generated {len(passages)}/{actual_n} hypothetical passages",
    })
    return passages, traces


async def _fact_queries(
    *,
    query: str,
    history_text: str,
    max_queries: int,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    if retrieval_plan is None or not budget.try_consume():
        return [], []

    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.fact_split", "payload": {"query": query}},
    ]

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
        return [], traces

    return sub_queries, traces


async def _keywords(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    if retrieval_plan is None or not budget.try_consume():
        return [], []

    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.keywords", "payload": {"query": query}},
    ]

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
        return [], traces

    return kw[:4], traces
