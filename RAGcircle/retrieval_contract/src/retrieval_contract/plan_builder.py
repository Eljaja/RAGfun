"""ExecutionPlan construction from presets or LLM planner output."""

from __future__ import annotations

from retrieval_contract.steps import (
    AdaptiveKStep,
    BM25SearchStep,
    ExecutionPlan,
    FuseStep,
    PlanRound,
    RerankStep,
    TrimStep,
    VectorSearchStep,
)


def from_preset(
    name: str,
    *,
    top_k: int = 5,
    rerank: bool = True,
    rerank_top_n: int | None = None,
) -> ExecutionPlan:
    rerank_top_n = rerank_top_n or top_k
    name = (name or "hybrid").lower().strip()

    if name == "fast":
        return ExecutionPlan(round=PlanRound(
            retrieve=[VectorSearchStep(top_k=top_k)],
            finalize=[TrimStep(top_k=top_k)],
        ))

    if name == "budget":
        fetch_k = max(top_k * 2, 10)
        return ExecutionPlan(round=PlanRound(
            retrieve=[
                VectorSearchStep(top_k=fetch_k),
                BM25SearchStep(top_k=fetch_k),
            ],
            combine=FuseStep(rrf_k=60),
            finalize=[TrimStep(top_k=top_k)],
        ))

    if name == "thorough":
        fetch_k = max(top_k * 3, 30)
        rtn = min(rerank_top_n * 2, fetch_k)
        return ExecutionPlan(round=PlanRound(
            retrieve=[
                VectorSearchStep(top_k=fetch_k),
                BM25SearchStep(top_k=fetch_k),
            ],
            combine=FuseStep(rrf_k=60),
            rank=[
                RerankStep(top_n=rtn),
                AdaptiveKStep(min_k=3, max_k=rtn),
            ],
            finalize=[TrimStep(top_k=rtn)],
        ))

    # "hybrid" (default)
    fetch_k = max(top_k * 2, top_k)
    rank_steps = [RerankStep(top_n=min(rerank_top_n, fetch_k))] if rerank else []
    return ExecutionPlan(round=PlanRound(
        retrieve=[
            VectorSearchStep(top_k=fetch_k),
            BM25SearchStep(top_k=fetch_k),
        ],
        combine=FuseStep(rrf_k=60),
        rank=rank_steps,
        finalize=[TrimStep(top_k=top_k)],
    ))


def from_llm_plan(
    plan_dict: dict,
    *,
    top_k_min: int = 5,
    top_k_max: int = 24,
) -> ExecutionPlan:
    """Translate LLM planner JSON into a validated ExecutionPlan.

    The LLM may request bm25/vector/hybrid modes, but the retrieval
    service doesn't support standalone bm25-only search, so we always
    use hybrid with the LLM's top_k and rerank preferences.
    """
    raw_k = int(plan_dict.get("top_k") or 10)
    top_k = max(top_k_min, min(top_k_max, raw_k))
    rerank = bool(plan_dict.get("rerank", True))
    # TODO: change the misleading method cause this is very bad actually
    return from_preset("hybrid", top_k=top_k, rerank=rerank)


DEFAULT_PLAN = from_preset("hybrid", top_k=10, rerank=True)
