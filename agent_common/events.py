"""Unified event format for agent-search and deep-research SSE streams."""

from __future__ import annotations

from typing import Literal

# Event types for UI compatibility
EventType = Literal[
    "trace",      # thought, tool, action, mood
    "retrieval",  # hits, context, mode, partial, degraded
    "token",      # streaming answer chunk
    "progress",   # stage, percent, message (deep-research)
    "done",       # answer, sources, context
    "error",      # error message
]

# Trace kinds
TraceKind = Literal["thought", "tool", "action", "mood"]

# Progress stages (deep-research)
ProgressStage = Literal["init", "plan", "scope", "research", "write", "done", "compiled"]
