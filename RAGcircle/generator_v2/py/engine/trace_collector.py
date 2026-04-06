from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

TraceCallback = Callable[[dict[str, Any]], None | Awaitable[None]]


@dataclass
class TraceCollector:
    _entries: list[dict[str, Any]] = field(default_factory=list)
    _callbacks: list[TraceCallback] = field(default_factory=list)

    def add_callback(self, fn: TraceCallback) -> None:
        self._callbacks.append(fn)

    async def emit(self, trace: dict[str, Any]) -> None:
        self._entries.append(trace)
        for fn in self._callbacks:
            result = fn(trace)
            if inspect.isawaitable(result):
                await result

    async def emit_many(self, traces: list[dict[str, Any]]) -> None:
        for t in traces:
            await self.emit(t)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)