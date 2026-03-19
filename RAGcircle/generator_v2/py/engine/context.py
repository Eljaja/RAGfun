"""RunContext — mutable pipeline state carried across rounds.

Immutable request fields are above the line; mutable pipeline state below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.budget import BudgetCounter
from models.chunks import ChunkResult
from models.retrieval import ExecutionPlan


@dataclass
class RunContext:
    # ── Immutable request context ────────────────────────
    project_id: str
    query: str
    history: list[dict[str, str]]
    history_text: str
    include_sources: bool = True

    # ── Mutable pipeline state ───────────────────────────
    lang: str = "English"
    is_factoid: bool = False
    search_queries: list[str] = field(default_factory=list)
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode: str = "hybrid"
    chunks: list[ChunkResult] = field(default_factory=list)
    answer: str = ""
    budget: BudgetCounter = field(default_factory=lambda: BudgetCounter(12))
    round_index: int = 0
    source_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.search_queries:
            self.search_queries = [self.query]

    @property
    def search_query(self) -> str:
        return self.search_queries[0] if self.search_queries else self.query

    @search_query.setter
    def search_query(self, value: str) -> None:
        if self.search_queries:
            self.search_queries[0] = value
        else:
            self.search_queries = [value]
