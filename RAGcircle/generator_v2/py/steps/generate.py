"""Generate-phase step handler."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from context import build_context, history_as_messages
from engine.env import StepEnv
from engine.registry import step_handler
from models.events import ErrorEvent, Event, TokenEvent
from prompts import ANSWER_SYSTEM, ANSWER_SYSTEM_WITH_TOOLS, ANSWER_USER
from tools import TOOL_DEFINITIONS


@step_handler("generate")
async def run_generate(step: Any, env: StepEnv) -> AsyncIterator[Event]:
    if not env.ctx.chunks:
        yield ErrorEvent(error="No chunks available for generation")
        return

    context_text = build_context(
        env.ctx.chunks,
        max_chars=env.settings.max_context_chars,
        max_chunk_chars=env.settings.max_chunk_chars,
        source_meta=env.ctx.source_meta,
    )

    if step.use_tools:
        system_prompt = ANSWER_SYSTEM_WITH_TOOLS.format(lang=env.ctx.lang)
    else:
        system_prompt = ANSWER_SYSTEM.format(lang=env.ctx.lang)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(env.ctx.history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=env.ctx.query, context=context_text,
    )})

    env.ctx.answer = ""
    if step.use_tools:
        env.ctx.answer = await env.llm.complete_with_tools(
            env.model, messages, TOOL_DEFINITIONS, temperature=step.temperature,
        )
        yield TokenEvent(content=env.ctx.answer)
    elif step.stream:
        async for token in env.llm.stream(env.model, messages, temperature=step.temperature):
            env.ctx.answer += token
            yield TokenEvent(content=token)
    else:
        env.ctx.answer = await env.llm.complete(env.model, messages, temperature=step.temperature)
        yield TokenEvent(content=env.ctx.answer)
