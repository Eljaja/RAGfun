"""Typed event dataclasses for the SSE stream.

Step handlers yield these instead of raw dicts. The executor serialises
them for the wire. Pattern-matching on Event replaces `.get("type")`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class TokenEvent:
    content: str
    type: Literal["token"] = "token"


@dataclass(slots=True)
class DoneEvent:
    answer: str
    mode: str
    partial: bool
    degraded: list[str]
    sources: list[dict[str, Any]]
    context: list[dict[str, Any]]
    needs_retry: bool = False
    missing_terms: list[str] = field(default_factory=list)
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    error: str
    type: Literal["error"] = "error"


@dataclass(slots=True)
class TraceEvent:
    kind: str
    label: str = ""
    name: str = ""
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    type: Literal["trace"] = "trace"


@dataclass(slots=True)
class ProgressEvent:
    stage: str
    content: str = ""
    type: Literal["progress"] = "progress"


@dataclass(slots=True)
class RetrievalEvent:
    mode: str
    partial: bool
    degraded: list[str]
    context: list[dict[str, Any]]
    type: Literal["retrieval"] = "retrieval"


@dataclass(slots=True)
class InitEvent:
    trace_id: str
    type: Literal["init"] = "init"


Event = TokenEvent | DoneEvent | ErrorEvent | TraceEvent | ProgressEvent | RetrievalEvent | InitEvent
