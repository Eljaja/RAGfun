from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_QUERY_LEN = 2000


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class EmbeddingResponseData(BaseModel):
    embedding: list[float]
    index: int
    object: Literal["embedding"] = Field(...)


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingResponseData]
    model: str
    object: Literal["list"] = Field(...)
    usage: dict[str, Any] | None = None


class StepBase(BaseModel):
    """Base model for retrieval plan steps with strict field checks."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("query", mode="before", check_fields=False)
    @classmethod
    def _normalize_and_validate_query(cls, value: object) -> str:
        """
        Query handling rules:
        - missing field -> defaults to "", and retriever falls back to request query
        - explicit null is rejected
        - explicit empty/whitespace values are rejected
        - hidden control chars are rejected
        """
        if value is None:
            raise ValueError("query cannot be null; omit the field to use request query")
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        text = value.strip()
        if not text:
            raise ValueError("query cannot be empty; omit the field to use request query")
        if len(text) > _MAX_QUERY_LEN:
            raise ValueError(f"query is too long (max {_MAX_QUERY_LEN} chars)")
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            raise ValueError("query contains unsupported control characters")
        return text


class BM25SearchStep(StepBase):
    kind: Literal["bm25_search"] = "bm25_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="", max_length=_MAX_QUERY_LEN)


class VectorSearchStep(StepBase):
    kind: Literal["vector_search"] = "vector_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="", max_length=_MAX_QUERY_LEN)


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

    @classmethod
    def from_legacy(
        cls,
        *,
        strategy: str,
        top_k: int,
        rerank: bool,
        rerank_top_n: int,
    ) -> ExecutionPlan:
        strategy = (strategy or "hybrid").lower().strip()
        top_k = max(1, int(top_k))
        rerank_top_n = max(1, int(rerank_top_n))

        if strategy == "bm25":
            return cls(
                rounds=[
                    PlanRound(
                        retrieve=[BM25SearchStep(top_k=top_k)],
                        finalize=[TrimStep(top_k=top_k)],
                    )
                ]
            )
        if strategy == "vector":
            return cls(
                rounds=[
                    PlanRound(
                        retrieve=[VectorSearchStep(top_k=top_k)],
                        finalize=[TrimStep(top_k=top_k)],
                    )
                ]
            )

        fetch_k = max(top_k * 2, top_k)
        rank_steps = [RerankStep(top_n=min(rerank_top_n, fetch_k))] if rerank else []
        return cls(
            rounds=[
                PlanRound(
                    retrieve=[
                        VectorSearchStep(top_k=fetch_k),
                        BM25SearchStep(top_k=fetch_k),
                    ],
                    combine=FuseStep(rrf_k=60),
                    rank=rank_steps,
                    finalize=[TrimStep(top_k=top_k)],
                )
            ]
        )


class RetrieveRequest(BaseModel):
    project_id: str
    query: str = Field(min_length=1, max_length=_MAX_QUERY_LEN)
    top_k: int = Field(default=5, ge=1, le=2000)
    rerank: bool = True
    rerank_top_n: int = Field(default=5, ge=1, le=2000)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: object) -> str:
        if value is None:
            raise ValueError("query cannot be null")
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        text = value.strip()
        if not text:
            raise ValueError("query cannot be empty")
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            raise ValueError("query contains unsupported control characters")
        return text


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    strategy: str
    query: str
    plan_used: ExecutionPlan | None = None
    rounds_executed: int = 0


class ExecuteRequest(BaseModel):
    project_id: str
    query: str = Field(min_length=1, max_length=_MAX_QUERY_LEN)
    plan: ExecutionPlan

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: object) -> str:
        if value is None:
            raise ValueError("query cannot be null")
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        text = value.strip()
        if not text:
            raise ValueError("query cannot be empty")
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            raise ValueError("query contains unsupported control characters")
        return text


class ExecuteResponse(BaseModel):
    chunks: list[ChunkResult]
    query: str
    plan: ExecutionPlan
    rounds_executed: int
