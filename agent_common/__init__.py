"""Shared library for agent-search and deep-research."""

from agent_common.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    DEEP_FACT_QUERIES,
    DEEP_HYDE,
    DEEP_KEYWORD_QUERIES,
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
)
from agent_common.gate_client import AsyncGateClient, SyncGateClient
from agent_common.events import EventType

__all__ = [
    "ANSWER_SYSTEM",
    "ANSWER_USER",
    "DEEP_FACT_QUERIES",
    "DEEP_HYDE",
    "DEEP_KEYWORD_QUERIES",
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
    "AsyncGateClient",
    "SyncGateClient",
    "EventType",
]
