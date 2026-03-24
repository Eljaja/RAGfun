"""LLM-driven query expansion: HyDE, fact queries, keyword queries.

Each function takes an LLMClient + prompt args, returns structured output.
"""

from __future__ import annotations

import json
import logging
import re

from llm import LLMClient
from prompts import (
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
)

logger = logging.getLogger(__name__)

_THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINKING_RE.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


async def hyde(
    llm: LLMClient,
    model: str,
    query: str,
    *,
    lang: str = "English",
    temperature: float = 0.2,
) -> str:
    """Generate a hypothetical answer passage for retrieval."""
    return await llm.complete(
        model,
        [
            {"role": "system", "content": HYDE_SYSTEM.format(lang=lang)},
            {"role": "user", "content": HYDE_USER.format(query=query, lang=lang)},
        ],
        temperature=temperature,
    )


async def fact_queries(
    llm: LLMClient,
    model: str,
    query: str,
    *,
    history_text: str = "",
) -> list[str]:
    """Extract 2-3 fact-oriented sub-queries from the user request."""
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": FACT_QUERIES_SYSTEM},
            {"role": "user", "content": FACT_QUERIES_USER.format(history=history_text, query=query)},
        ],
        temperature=0.2,
    )
    try:
        data = json.loads(_strip_thinking(raw))
        out = data.get("fact_queries") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        logger.warning("Failed to parse fact queries from LLM response")
        return []


async def keyword_queries(
    llm: LLMClient,
    model: str,
    query: str,
    *,
    history_text: str = "",
) -> list[str]:
    """Extract 3-6 short keyword phrases from the user request."""
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": KEYWORD_QUERIES_SYSTEM},
            {"role": "user", "content": KEYWORD_QUERIES_USER.format(history=history_text, query=query)},
        ],
        temperature=0.0,
    )
    try:
        data = json.loads(_strip_thinking(raw))
        out = data.get("keywords") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        logger.warning("Failed to parse keyword queries from LLM response")
        return []
