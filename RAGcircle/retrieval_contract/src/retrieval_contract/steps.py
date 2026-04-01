from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUERY_LEN = 2000


def validate_query_str(value: object, *, required: bool) -> str:
    """Shared query validation.

    required=True  -> null/empty is an error (request-level queries).
    required=False -> empty string is accepted as the "no override" sentinel
                      (means "use the request-level query"); null is still
                      rejected because omitting the field achieves the same.
    """
    if value is None:
        if required:
            raise ValueError("query cannot be null")
        raise ValueError("query cannot be null; omit the field to use request query")
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    text = value.strip()
    if not text:
        if required:
            raise ValueError("query cannot be empty")
        return ""
    if len(text) > MAX_QUERY_LEN:
        raise ValueError(f"query is too long (max {MAX_QUERY_LEN} chars)")
    if any(ord(ch) < 32 and ch not in ("\t", "\n", "\r") for ch in text):
        raise ValueError("query contains unsupported control characters")
    return text


class StepBase(BaseModel):
    """Base for all plan steps."""

    model_config = ConfigDict(extra="forbid")


class QueryStepBase(StepBase):
    """Base for steps that accept an optional per-step query override."""

    query: str = Field(default="", max_length=MAX_QUERY_LEN)

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, v: object) -> str:
        return validate_query_str(v, required=False)


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
    """One retrieval round with explicit phases.

    Validates:
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
        if len(self.retrieve) >= 2 and self.combine is None:
            raise ValueError(
                "multiple retrieval steps require a combine (fuse) step "
                "to produce comparable scores"
            )
        if self.combine is not None and len(self.retrieve) < 2:
            raise ValueError("fuse step requires at least two retrieval steps in the round")

        has_adaptive_k = any(s.kind == "adaptive_k" for s in self.rank)
        if has_adaptive_k:
            rerank_seen = False
            for step in self.rank:
                if step.kind == "rerank":
                    rerank_seen = True
                if step.kind == "adaptive_k" and not rerank_seen:
                    raise ValueError(
                        "adaptive_k requires a preceding rerank step "
                        "to produce meaningful score gaps"
                    )

        trim_steps = [step for step in self.finalize if step.kind == "trim"]
        if len(trim_steps) > 1:
            raise ValueError("at most one trim step is allowed per round")
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
    ) -> ExecutionPlan:
        strategy = (strategy or "hybrid").lower().strip()
        top_k = max(1, int(top_k))
        rerank_top_n = max(1, int(rerank_top_n))

        if strategy == "bm25":
            return cls(
                round=PlanRound(
                    retrieve=[BM25SearchStep(top_k=top_k)],
                    finalize=[TrimStep(top_k=top_k)],
                ),
            )
        if strategy == "vector":
            return cls(
                round=PlanRound(
                    retrieve=[VectorSearchStep(top_k=top_k)],
                    finalize=[TrimStep(top_k=top_k)],
                ),
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
            ),
        )
