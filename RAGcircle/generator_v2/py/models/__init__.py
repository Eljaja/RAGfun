"""Models package — re-exports everything for backwards compatibility."""

from __future__ import annotations

from retrieval_contract import (
    AdaptiveKStep,
    BM25SearchStep,
    ChunkResult,
    ExecutionPlan,
    FinalizeStep,
    FuseStep,
    PlanRound,
    RankStep,
    RerankStep,
    RetrievalStep,
    ScoreSource,
    TrimStep,
    VectorSearchStep,
)
from models.steps import (
    AssessStep,
    BM25AnchorStep,
    BrainRetrieveStep,
    ConfigStep,
    DetectLangStep,
    EvalStep,
    FactoidExpandStep,
    FactQueryStep,
    GenerateStep,
    GroundingCheckStep,
    HyDEStep,
    InitialExpandStep,
    KeywordStep,
    LoopExpandStep,
    PlanLLMStep,
    QualityCheckStep,
    QueryVariantsStep,
    ReflectStep,
    StitchStep,
    TwoPassStep,
)
from models.plan import (
    BrainPlan,
    BrainRound,
    ConfigMeta,
    PipelineResult,
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResult,
    Verdict,
)
from models.api import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    ExecuteRequest,
    PlanRunRequest,
    PlanRunResponse,
    SearchFilters,
)
from models.assessment import AssessmentResult, ReflectionResult
from models.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    InitEvent,
    ProgressEvent,
    RetrievalEvent,
    TokenEvent,
    TraceEvent,
)

__all__ = [
    "ChunkResult", "ScoreSource",
    "AdaptiveKStep", "BM25SearchStep", "ExecutionPlan", "FinalizeStep",
    "FuseStep", "PlanRound", "RankStep", "RerankStep", "RetrievalStep",
    "TrimStep", "VectorSearchStep",
    "AssessStep", "BM25AnchorStep", "BrainRetrieveStep", "ConfigStep",
    "DetectLangStep", "EvalStep", "FactoidExpandStep",
    "FactQueryStep", "GenerateStep", "GroundingCheckStep", "HyDEStep",
    "InitialExpandStep", "KeywordStep", "LoopExpandStep", "PlanLLMStep",
    "QualityCheckStep", "QueryVariantsStep",
    "ReflectStep", "StitchStep", "TwoPassStep",
    "BrainPlan", "BrainRound", "ConfigMeta",
    "PipelineResult", "RetrievalPlan", "RetrievalRequest", "RetrievalResult",
    "Verdict",
    "AgentRequest", "AgentResponse", "ChatRequest", "ChatResponse",
    "ExecuteRequest", "PlanRunRequest", "PlanRunResponse", "SearchFilters",
    "AssessmentResult", "ReflectionResult",
    "DoneEvent", "ErrorEvent", "Event", "InitEvent", "ProgressEvent",
    "RetrievalEvent", "TokenEvent", "TraceEvent",
]
