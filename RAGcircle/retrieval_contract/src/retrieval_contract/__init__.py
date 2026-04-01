"""Shared schema between retrieval and generator services."""

from retrieval_contract.chunks import ChunkResult, ScoreSource
from retrieval_contract.steps import (
    AdaptiveKStep,
    BM25SearchStep,
    ExecutionPlan,
    FinalizeStep,
    FuseStep,
    PlanRound,
    QueryStepBase,
    RankStep,
    RerankStep,
    RetrievalStep,
    StepBase,
    TrimStep,
    VectorSearchStep,
)
from retrieval_contract.api import (
    ExecuteRequest,
    ExecuteResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from retrieval_contract.plan_builder import (
    DEFAULT_PLAN,
    from_llm_plan,
    from_preset,
)

__all__ = [
    "ChunkResult",
    "ScoreSource",
    "StepBase",
    "QueryStepBase",
    "AdaptiveKStep",
    "BM25SearchStep",
    "ExecutionPlan",
    "FinalizeStep",
    "FuseStep",
    "PlanRound",
    "RankStep",
    "RerankStep",
    "RetrievalStep",
    "TrimStep",
    "VectorSearchStep",
    "ExecuteRequest",
    "ExecuteResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "from_preset",
    "from_llm_plan",
    "DEFAULT_PLAN",
]
