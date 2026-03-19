from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.steps import (
    BrainRetrieveStep,
    EvalStep,
    ExpandStep,
    GenerateStep,
    PostRetrieveStep,
)


class BrainRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expand: list[ExpandStep] = Field(default_factory=list)
    retrieve: BrainRetrieveStep = Field(default_factory=BrainRetrieveStep)
    post_retrieve: list[PostRetrieveStep] = Field(default_factory=list)
    generate: GenerateStep = Field(default_factory=GenerateStep)
    evaluate: list[EvalStep] = Field(default_factory=list)

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


class BrainPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: list[BrainRound] = Field(min_length=1)
    max_llm_calls: int = Field(default=12, ge=1, le=100)
