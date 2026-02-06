"""Shared library for agent-search and deep-research."""

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
    "quality_is_poor",
    "merge_hits",
    "build_context",
    "context_from_hits",
    "sources_from_context",
    "AsyncGateClient",
    "SyncGateClient",
    "EventType",
]
