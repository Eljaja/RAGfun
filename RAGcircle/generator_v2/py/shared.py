"""Shared utilities — consolidates duplicated logic."""

from __future__ import annotations

import re

_THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text.

    Single canonical implementation — imported everywhere instead of
    duplicated across llm.py, query_expansion.py, query_variants.py, planning.py.
    """
    if not text or not text.strip():
        return text
    return _THINKING_RE.sub("", text).strip()
