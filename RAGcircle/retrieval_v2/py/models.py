"""Models for the retrieval service.

Shared contract types (plan steps, chunks, API models) are imported from
the retrieval_contract package. Internal-only types live here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Re-export shared contract types so existing `from models import X` keeps working.
from retrieval_contract import (  # noqa: F401
    AdaptiveKStep,
    BM25SearchStep,
    ChunkResult,
    ExecuteRequest,
    ExecuteResponse,
    ExecutionPlan,
    FinalizeStep,
    FuseStep,
    PlanRound,
    QueryStepBase,
    RankStep,
    RerankStep,
    RetrievalStep,
    RetrieveRequest,
    RetrieveResponse,
    ScoreSource,
    StepBase,
    TrimStep,
    VectorSearchStep,
)


# ── Internal types (not part of the shared contract) ─────


class EmbeddingResponseData(BaseModel):
    embedding: list[float]
    index: int
    object: Literal["embedding"] = Field(...)


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingResponseData]
    model: str
    object: Literal["list"] = Field(...)
    usage: dict[str, Any] | None = None
