"""Shared schema between generator and gate services."""

from generator_contract.api import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    SearchFilters,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "AgentRequest",
    "AgentResponse",
    "SearchFilters",
]
