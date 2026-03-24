"""LLM call budget tracking.

BudgetCounter.can_call() was a query that mutated — a guaranteed bug source.
Replaced with explicit has_budget() / consume() and a convenience try_consume().
"""

from __future__ import annotations


class BudgetCounter:
    __slots__ = ("_max", "_used")

    def __init__(self, max_calls: int) -> None:
        self._max = max_calls
        self._used = 0

    def has_budget(self) -> bool:
        return self._used < self._max

    def consume(self) -> None:
        self._used += 1

    def try_consume(self) -> bool:
        """Atomically check + consume. Returns True if budget was available."""
        if self._used >= self._max:
            return False
        self._used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self._max - self._used)

    @property
    def used(self) -> int:
        return self._used
