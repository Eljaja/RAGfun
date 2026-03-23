"""Shared utilities — consolidates duplicated logic."""

from __future__ import annotations

import re

_THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:\w*)\n?(.*?)```", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text.

    Single canonical implementation — imported everywhere instead of
    duplicated across llm.py, query_expansion.py, query_variants.py, planning.py.
    """
    if not text or not text.strip():
        return text
    return _THINKING_RE.sub("", text).strip()


def strip_fences(text: str) -> str:
    """Extract content from markdown code fences if present.

    Handles ```json ... ```, ```  ... ```, etc.  Returns the inner
    content when exactly one fenced block is found, otherwise the
    original text (stripped).
    """
    text = text.strip()
    matches = _FENCE_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    return text
