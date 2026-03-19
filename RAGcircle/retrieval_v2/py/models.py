from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float
    score_source: "ScoreSource" = "retrieval"


class ScoreSource(StrEnum):
    RETRIEVAL = "retrieval"
    RRF = "rrf"
    RERANK = "rerank"


class RetrieveRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    strategy: str
    query: str
    plan_used: "ExecutionPlan | None" = None


class StepBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryStepBase(StepBase):
    query: str = ""


class BM25SearchStep(QueryStepBase):
    kind: Literal["bm25_search"] = "bm25_search"
    top_k: int = Field(default=20, ge=1, le=2000)


class VectorSearchStep(QueryStepBase):
    kind: Literal["vector_search"] = "vector_search"
    top_k: int = Field(default=20, ge=1, le=2000)


class FuseStep(StepBase):
    kind: Literal["fuse"] = "fuse"
    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=5000)


class RerankStep(QueryStepBase):
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
    model_config = ConfigDict(extra="forbid")

    retrieve: list[RetrievalStep] = Field(default_factory=list)
    combine: FuseStep | None = None
    rank: list[RankStep] = Field(default_factory=list)
    finalize: list[FinalizeStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_structure(self):
        if not self.retrieve:
            raise ValueError("plan round must include at least one retrieval step")
        if len(self.retrieve) >= 2 and self.combine is None:
            raise ValueError("multiple retrieval steps require combine step")
        if self.combine is not None and len(self.retrieve) < 2:
            raise ValueError("combine step requires at least two retrieval steps")
        return self


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: PlanRound

    @classmethod
    def from_legacy(
        cls,
        *,
        strategy: str,
        top_k: int,
        rerank: bool,
        rerank_top_n: int,
    ) -> "ExecutionPlan":
        strategy = (strategy or "hybrid").lower().strip()
        top_k = max(1, int(top_k))
        rerank_top_n = max(1, int(rerank_top_n))

        if strategy == "bm25":
            return cls(
                round=PlanRound(
                    retrieve=[BM25SearchStep(top_k=top_k)],
                    finalize=[TrimStep(top_k=top_k)],
                )
            )
        if strategy == "vector":
            return cls(
                round=PlanRound(
                    retrieve=[VectorSearchStep(top_k=top_k)],
                    finalize=[TrimStep(top_k=top_k)],
                )
            )

        fetch_k = max(top_k * 2, top_k)
        rank_steps = [RerankStep(top_n=min(rerank_top_n, fetch_k))] if rerank else []
        return cls(
            round=PlanRound(
                retrieve=[
                    VectorSearchStep(top_k=fetch_k),
                    BM25SearchStep(top_k=fetch_k),
                ],
                combine=FuseStep(rrf_k=60),
                rank=rank_steps,
                finalize=[TrimStep(top_k=top_k)],
            )
        )


class ExecuteRequest(BaseModel):
    project_id: str
    query: str
    plan: ExecutionPlan


class ExecuteResponse(BaseModel):
    chunks: list[ChunkResult]
    query: str
    plan: ExecutionPlan
