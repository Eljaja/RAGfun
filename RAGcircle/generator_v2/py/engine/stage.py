"""Generic step execution framework.

run_stage dispatches steps (parallel or sequential), captures outcomes,
flattens traces, and returns a StageResult the caller post-processes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class StepOutcome(Generic[T]):
    step_name: str
    value: T | None = None
    error: Exception | None = None
    tb: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class StageResult(Generic[T]):
    domain: list[T]
    traces: list[dict[str, Any]]
    errors: list[StepOutcome]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


async def execute_steps(
    steps: list,
    dispatch: Callable,
    *,
    parallel: bool = False,
) -> list[StepOutcome]:
    """Run each step through dispatch, wrapping results in StepOutcome."""

    async def _safe(step: Any) -> StepOutcome:
        name = type(step).__name__
        try:
            result = dispatch(step)
            if inspect.isawaitable(result):
                result = await result
            return StepOutcome(step_name=name, value=result)
        except Exception as e:
            return StepOutcome(
                step_name=name, error=e, tb=traceback.format_exc(),
            )

    if parallel:
        return list(await asyncio.gather(*(_safe(s) for s in steps)))
    return [await _safe(s) for s in steps]


def collect(outcomes: list[StepOutcome]) -> StageResult:
    """Separate successes from failures, flatten traces."""
    successes = [o for o in outcomes if o.ok]
    errors = [o for o in outcomes if not o.ok]

    values = [o.value for o in successes]
    domain = [r for r, _ in values]
    traces = list(chain.from_iterable(t for _, t in values))

    return StageResult(domain=domain, traces=traces, errors=errors)


async def run_stage(
    steps: list,
    dispatch: Callable,
    *,
    parallel: bool = False,
) -> StageResult:
    """Execute steps + collect. Returns StageResult."""
    outcomes = await execute_steps(steps, dispatch, parallel=parallel)
    return collect(outcomes)


def merge_partials(partials: list[dict]) -> dict:
    """Merge partial dicts, right-biased."""
    return {k: v for p in partials for k, v in p.items()}
