"""All BrainPlan step types (the DSL).

Each step carries its own `kind` literal, enabling discriminated unions and
the step-registry dispatch in engine/registry.py.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanLLMStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["plan_retrieval"] = "plan_retrieval"


class DetectLangStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["detect_lang"] = "detect_lang"


class HyDEStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["hyde"] = "hyde"
    num_passages: int = Field(default=1, ge=1, le=7)


class FactQueryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fact_queries"] = "fact_queries"
    max_queries: int = Field(default=2, ge=1, le=10)


class KeywordStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["keywords"] = "keywords"


class QueryVariantsStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["query_variants"] = "query_variants"


class BrainRetrieveStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["retrieve"] = "retrieve"
    preset: str = Field(default="hybrid", pattern="^(fast|hybrid|thorough|budget)$")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = True


class QualityCheckStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["quality_check"] = "quality_check"
    min_hits: int = Field(default=3, ge=1, le=50)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TwoPassStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["two_pass"] = "two_pass"
    min_unique_sources: int = Field(default=3, ge=1, le=20)


class BM25AnchorStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bm25_anchor"] = "bm25_anchor"
    top_k: int = Field(default=15, ge=1, le=100)


class FactoidExpandStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["factoid_expand"] = "factoid_expand"


class StitchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["stitch"] = "stitch"
    max_per_segment: int = Field(default=4, ge=1, le=10)


class GenerateStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["generate"] = "generate"
    use_tools: bool = False
    stream: bool = True
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ReflectStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["reflect"] = "reflect"


class AssessStep(BaseModel):
    """LLM completeness check — writes assessment to ctx. Does NOT re-retrieve."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["assess"] = "assess"


class GroundingCheckStep(BaseModel):
    """Heuristic grounding check — is the answer grounded in chunks?"""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["grounding_check"] = "grounding_check"


# ── Discriminated unions ─────────────────────────────────

# Old unions (used by current pipeline, removed in Session 2)
ExpandStep = Annotated[
    PlanLLMStep | DetectLangStep | HyDEStep | FactQueryStep | KeywordStep | QueryVariantsStep,
    Field(discriminator="kind"),
]

PostRetrieveStep = Annotated[
    QualityCheckStep | TwoPassStep | BM25AnchorStep | FactoidExpandStep | StitchStep,
    Field(discriminator="kind"),
]

EvalStep = Annotated[
    ReflectStep | AssessStep | GroundingCheckStep,
    Field(discriminator="kind"),
]

# New unions (spec-aligned, used by retrieval pipeline)
ConfigStep = Annotated[
    PlanLLMStep | DetectLangStep,
    Field(discriminator="kind"),
]

InitialExpandStep = Annotated[
    HyDEStep | FactQueryStep | KeywordStep | QueryVariantsStep | BM25AnchorStep,
    Field(discriminator="kind"),
]

LoopExpandStep = Annotated[
    TwoPassStep | FactoidExpandStep,
    Field(discriminator="kind"),
]
