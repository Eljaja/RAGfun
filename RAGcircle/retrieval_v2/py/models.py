from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class StepBase(BaseModel):
    """Base model for retrieval plan steps with strict field checks."""

    model_config = ConfigDict(extra="forbid")


class BM25SearchStep(StepBase):
    kind: Literal["bm25_search"] = "bm25_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str | None = None
    query_mode: Literal["raw", "keyword"] = "raw"


class VectorSearchStep(StepBase):
    kind: Literal["vector_search"] = "vector_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str | None = None
    query_mode: Literal["raw", "keyword"] = "raw"


class FuseStep(StepBase):
    kind: Literal["fuse"] = "fuse"
    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=5000)


class RerankStep(StepBase):
    kind: Literal["rerank"] = "rerank"
    top_n: int = Field(default=8, ge=1, le=1000)


class AdaptiveKStep(StepBase):
    kind: Literal["adaptive_k"] = "adaptive_k"
    min_k: int = Field(default=3, ge=1, le=1000)
    max_k: int = Field(default=24, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.min_k > self.max_k:
            raise ValueError("adaptive_k.min_k must be <= adaptive_k.max_k")
        return self


class TrimStep(StepBase):
    kind: Literal["trim"] = "trim"
    top_k: int = Field(default=8, ge=1, le=2000)


RetrievalStep = Annotated[BM25SearchStep | VectorSearchStep, Field(discriminator="kind")]
RankStep = Annotated[RerankStep | AdaptiveKStep, Field(discriminator="kind")]
FinalizeStep = Annotated[TrimStep, Field(discriminator="kind")]


class PlanRound(BaseModel):
    """
    One retrieval round with explicit phases.

    This shape provides basic composition checks:
    - retrieval comes first (bm25/vector only),
    - optional fuse step after retrieval,
    - ranking/post-processing next.
    """

    model_config = ConfigDict(extra="forbid")

    retrieve: list[RetrievalStep] = Field(default_factory=list)
    combine: FuseStep | None = None
    rank: list[RankStep] = Field(default_factory=list)
    finalize: list[FinalizeStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_structure(self):
        if not self.retrieve:
            raise ValueError("plan round must include at least one retrieval step")
        if self.combine is not None and len(self.retrieve) < 2:
            raise ValueError("fuse step requires at least two retrieval steps in the round")
        trim_steps = [step for step in self.finalize if step.kind == "trim"]
        if len(trim_steps) > 1:
            raise ValueError("at most one trim step is allowed per round")
        return self


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: list[PlanRound] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_rounds(self):
        if not self.rounds:
            raise ValueError("execution plan must include at least one round")
        return self


class RetrieveRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=2000)
    rerank: bool = True
    rerank_top_n: int = Field(default=5, ge=1, le=2000)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")
    plan: ExecutionPlan | None = None


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    strategy: str
    query: str
    plan_used: ExecutionPlan | None = None
    rounds_executed: int = 0
    warnings: list[str] = Field(default_factory=list)
