"""LLM-driven retrieval planning and budget tracking."""

from __future__ import annotations

import json
import logging
import re

from llm import LLMClient
from prompts import PLAN_SYSTEM, PLAN_USER

logger = logging.getLogger(__name__)

_THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

_DEFAULT_PLAN_DICT = {
    "retrieval_mode": "hybrid",
    "top_k": 10,
    "rerank": True,
    "use_hyde": False,
    "reason": "fallback",
}


async def plan_retrieval(
    llm: LLMClient,
    model: str,
    query: str,
    *,
    history_text: str = "",
) -> dict:
    """Ask the LLM to choose retrieval strategy. Returns a plan dict."""
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": PLAN_USER.format(history=history_text, query=query)},
        ],
        temperature=0.0,
    )
    cleaned = _THINKING_RE.sub("", raw or "").strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        logger.warning("Failed to parse LLM plan output, using fallback")
        return dict(_DEFAULT_PLAN_DICT)


class BudgetCounter:
    """Tracks LLM call budget for a single request."""

    def __init__(self, max_calls: int):
        self._max = max_calls
        self._used = 0

    def can_call(self) -> bool:
        self._used += 1
        return self._used <= self._max

    @property
    def remaining(self) -> int:
        return max(0, self._max - self._used)

    @property
    def used(self) -> int:
        return self._used
