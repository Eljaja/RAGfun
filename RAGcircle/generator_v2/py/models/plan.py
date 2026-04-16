from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval_contract import ChunkResult, ExecutionPlan
from models.steps import (
    BrainRetrieveStep,
    ConfigStep,
    EvalStep,
    GenerateStep,
    InitialExpandStep,
    LoopExpandStep,
    QualityCheckStep,
    StitchStep,
)


class RetrievalPlan(BaseModel):
    """Full retrieval pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    default_config: BrainRetrieveStep = Field(default_factory=BrainRetrieveStep)
    initial_expand: list[InitialExpandStep] = Field(default_factory=list)
    loop_check: list[QualityCheckStep] = Field(default_factory=list)
    loop_expand: list[LoopExpandStep] = Field(default_factory=list)
    finalize: list[StitchStep] = Field(default_factory=list)
    max_rounds: int = Field(default=2, ge=1, le=10)


class BrainRound(BaseModel):
    """A single pipeline pass: configure -> retrieve -> generate -> evaluate."""

    model_config = ConfigDict(extra="forbid")

    configure: list[ConfigStep] = Field(default_factory=list)
    retrieval: RetrievalPlan = Field(default_factory=RetrievalPlan)
    generate: GenerateStep = Field(default_factory=GenerateStep)
    evaluate: list[EvalStep] = Field(default_factory=list)
    max_llm_calls: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_structure(self):
        config_kinds = [s.kind for s in self.configure]
        if config_kinds.count("plan_retrieval") > 1:
            raise ValueError("at most one plan_retrieval step per round")
        if config_kinds.count("detect_lang") > 1:
            raise ValueError("at most one detect_lang step per round")
        expand_kinds = [s.kind for s in self.retrieval.initial_expand]
        if expand_kinds.count("query_variants") > 1:
            raise ValueError("at most one query_variants step per round")
        eval_kinds = [s.kind for s in self.evaluate]
        if "reflect" in eval_kinds and "assess" in eval_kinds:
            raise ValueError("reflect and assess are mutually exclusive in a single round")
        finalize_kinds = [s.kind for s in self.retrieval.finalize]
        if finalize_kinds.count("stitch") > 1:
            raise ValueError("at most one stitch step per round")
        return self


BrainPlan = BrainRound


@dataclass
class ConfigMeta:
    """Output of the configure phase."""

    lang: str = "English"
    is_factoid: bool = False
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode: str = "hybrid"


@dataclass
class RetrievalRequest:
    """A single retrieval request — query plus optional config override."""

    query: str
    plan_override: ExecutionPlan | None = None


@dataclass
class RetrievalResult:
    """Output of the retrieval pipeline."""

    chunks: list[ChunkResult]


@dataclass
class Verdict:
    """Output of the evaluate phase."""

    missing_terms: list[str] = field(default_factory=list)
    requery: str | None = None
    answer: str | None = None
    chunks: list[ChunkResult] | None = None

    @property
    def needs_retry(self) -> bool:
        return bool(self.missing_terms) or bool(self.requery)


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
