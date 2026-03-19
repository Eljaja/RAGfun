"""Step registry — the dispatch table that replaces the match/case blocks.

Usage:
    @step_handler("hyde")
    async def run_hyde(step: HyDEStep, env: StepEnv) -> AsyncIterator[Event]:
        ...

The executor calls run_steps() which looks up the handler by step.kind.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from models.events import ErrorEvent, Event

logger = logging.getLogger(__name__)

STEP_REGISTRY: dict[str, Callable[..., AsyncIterator[Event]]] = {}


def step_handler(kind: str) -> Callable:
    def decorator(fn: Callable[..., AsyncIterator[Event]]) -> Callable[..., AsyncIterator[Event]]:
        if kind in STEP_REGISTRY:
            logger.warning("Overwriting step handler for kind=%r", kind)
        STEP_REGISTRY[kind] = fn
        return fn
    return decorator


async def run_steps(steps: list[Any], env: Any) -> AsyncIterator[Event]:
    for step in steps:
        handler = STEP_REGISTRY.get(step.kind)
        if handler is None:
            yield ErrorEvent(error=f"No handler for step kind={step.kind!r}")
            return
        async for event in handler(step, env):
            yield event
            if isinstance(event, ErrorEvent):
                return
