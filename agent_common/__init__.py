"""Shared library for agent-search."""

from agent_common.prompts import (
    ANSWER_SYSTEM,
    ANSWER_SYSTEM_WITH_TOOLS,
    ANSWER_USER,
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
    PLAN_SYSTEM,
    PLAN_USER,
)
from agent_common.retrieval import (
    build_context,
    context_from_hits,
    merge_hits,
    quality_is_poor,
    sources_from_context,
    strip_thinking,
)
from agent_common.gate_client import AsyncGateClient
from agent_common.events import EventType
from agent_common.tools import run_calculator, run_execute_code

__all__ = [
    "ANSWER_SYSTEM",
    "ANSWER_SYSTEM_WITH_TOOLS",
    "ANSWER_USER",
    "FACT_QUERIES_SYSTEM",
    "FACT_QUERIES_USER",
    "HYDE_SYSTEM",
    "HYDE_USER",
    "KEYWORD_QUERIES_SYSTEM",
    "KEYWORD_QUERIES_USER",
    "PLAN_SYSTEM",
    "PLAN_USER",
    "quality_is_poor",
    "merge_hits",
    "build_context",
    "context_from_hits",
    "sources_from_context",
    "strip_thinking",
    "AsyncGateClient",
    "EventType",
    "run_calculator",
    "run_execute_code",
]
