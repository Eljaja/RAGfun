"""SSE event type definitions for the agent streaming endpoint."""

from __future__ import annotations

from typing import Literal

EventType = Literal[
    "trace",
    "retrieval",
    "token",
    "progress",
    "done",
    "error",
]

TraceKind = Literal["thought", "tool", "action", "mood"]

ProgressStage = Literal["init", "plan", "scope", "research", "write", "done", "compiled"]
