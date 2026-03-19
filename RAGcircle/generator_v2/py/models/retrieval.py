"""Retrieval execution plan models — mirrors the retrieval_v2 service."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BM25SearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bm25_search"] = "bm25_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="")


class VectorSearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["vector_search"] = "vector_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="")


class FuseStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fuse"] = "fuse"
    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=5000)


class RerankStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rerank"] = "rerank"
    top_n: int = Field(default=8, ge=1, le=1000)
    query: str = Field(default="")


class AdaptiveKStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["adaptive_k"] = "adaptive_k"
    min_k: int = Field(default=3, ge=1, le=1000)
    max_k: int = Field(default=24, ge=1, le=1000)


class TrimStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["trim"] = "trim"
    top_k: int = Field(default=8, ge=1, le=2000)


RetrievalStep = Annotated[BM25SearchStep | VectorSearchStep, Field(discriminator="kind")]
RankStep = Annotated[RerankStep | AdaptiveKStep, Field(discriminator="kind")]
FinalizeStep = Annotated[TrimStep, Field(discriminator="kind")]


class PlanRound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieve: list[RetrievalStep] = Field(default_factory=list)
    combine: FuseStep | None = None
    rank: list[RankStep] = Field(default_factory=list)
    finalize: list[FinalizeStep] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: PlanRound
