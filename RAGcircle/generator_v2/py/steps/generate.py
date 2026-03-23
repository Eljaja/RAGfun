"""Generate phase: chunks + query in, answer out.

Pure function: (...) -> str. No streaming — the caller streams the final answer.
"""

from __future__ import annotations

from typing import Any

from lib.context import build_context, history_as_messages
from clients.llm import LLMClient
from retrieval_contract import ChunkResult
from models.steps import GenerateStep
from lib.prompts import ANSWER_SYSTEM, ANSWER_SYSTEM_WITH_TOOLS, ANSWER_USER
from lib.tools import TOOL_DEFINITIONS


async def generate(
    step: GenerateStep,
    *,
    chunks: list[ChunkResult],
    query: str,
    lang: str,
    history: list[dict[str, str]],
    source_meta: dict[str, dict[str, Any]],
    max_context_chars: int = 6000,
    max_chunk_chars: int = 1200,
    llm: LLMClient,
    model: str,
) -> str:
    """Generate an answer from chunks. Returns the full answer string."""
    if not chunks:
        return ""

    context_text = build_context(
        chunks,
        max_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        source_meta=source_meta,
    )

    if step.use_tools:
        system_prompt = ANSWER_SYSTEM_WITH_TOOLS.format(lang=lang)
    else:
        system_prompt = ANSWER_SYSTEM.format(lang=lang)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=query, context=context_text,
    )})

    if step.use_tools:
        return await llm.complete_with_tools(
            model, messages, TOOL_DEFINITIONS, temperature=step.temperature,
        )

    return await llm.complete(model, messages, temperature=step.temperature)
