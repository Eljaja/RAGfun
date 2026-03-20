from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.chunks import ChunkResult
from models.retrieval import ExecutionPlan
from models.steps import (
    BrainRetrieveStep,
    EvalStep,
    ExpandStep,
    GenerateStep,
    PostRetrieveStep,
)


class BrainRound(BaseModel):
    """A single pipeline pass: expand -> retrieve -> enrich -> generate -> evaluate."""

    model_config = ConfigDict(extra="forbid")

    expand: list[ExpandStep] = Field(default_factory=list)
    retrieve: BrainRetrieveStep = Field(default_factory=BrainRetrieveStep)
    post_retrieve: list[PostRetrieveStep] = Field(default_factory=list)
    generate: GenerateStep = Field(default_factory=GenerateStep)
    evaluate: list[EvalStep] = Field(default_factory=list)
    max_llm_calls: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_structure(self):
        expand_kinds = [s.kind for s in self.expand]
        if expand_kinds.count("plan_retrieval") > 1:
            raise ValueError("at most one plan_retrieval step per round")
        if expand_kinds.count("detect_lang") > 1:
            raise ValueError("at most one detect_lang step per round")
        if expand_kinds.count("query_variants") > 1:
            raise ValueError("at most one query_variants step per round")
        eval_kinds = [s.kind for s in self.evaluate]
        if "reflect" in eval_kinds and "assess" in eval_kinds:
            raise ValueError("reflect and assess are mutually exclusive in a single round")
        post_kinds = [s.kind for s in self.post_retrieve]
        if post_kinds.count("stitch") > 1:
            raise ValueError("at most one stitch step per round")
        return self


BrainPlan = BrainRound


@dataclass
class EnrichRetrievalRequest:
    """A deferred retrieval request emitted by an enrich step."""

    query: str
    plan: ExecutionPlan | None = None


@dataclass
class ExpandResult:
    """Output of the expand phase."""

    queries: list[str]
    lang: str = "English"
    is_factoid: bool = False
    retrieval_plan: Any = None
    retrieval_mode: str = "hybrid"
    traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Verdict:
    """Output of the evaluate phase."""

    needs_retry: bool = False
    missing_terms: list[str] = field(default_factory=list)
    requery: str | None = None
    answer: str | None = None
    chunks: list[ChunkResult] | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Output of a single pipeline run."""

    answer: str
    chunks: list[ChunkResult]
    sources: list[dict[str, Any]]
    mode: str = "hybrid"
    lang: str = "English"
    is_factoid: bool = False
    needs_retry: bool = False
    missing_terms: list[str] = field(default_factory=list)
    requery: str | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)
