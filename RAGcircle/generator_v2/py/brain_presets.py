"""Named BrainRound constructors — each returns a single pipeline pass."""

from __future__ import annotations

from models import (
    AssessStep,
    BM25AnchorStep,
    BrainRetrieveStep,
    BrainRound,
    DetectLangStep,
    FactoidExpandStep,
    FactoidRetryStep,
    FactQueryStep,
    GenerateStep,
    HyDEStep,
    KeywordStep,
    PlanLLMStep,
    QualityCheckStep,
    QueryVariantsStep,
    ReflectStep,
    StitchStep,
    SupplementalRetrieveStep,
    TwoPassStep,
)


def simple(
    *,
    preset: str = "hybrid",
    top_k: int = 5,
    rerank: bool = True,
    reflection_enabled: bool = True,
    max_llm_calls: int = 4,
) -> BrainRound:
    """Simple /chat pipeline: retrieve -> generate, with optional reflect."""
    return BrainRound(
        expand=[QueryVariantsStep()],
        retrieve=BrainRetrieveStep(preset=preset, top_k=top_k, rerank=rerank),
        post_retrieve=[StitchStep()],
        generate=GenerateStep(stream=False),
        evaluate=[ReflectStep()] if reflection_enabled else [],
        max_llm_calls=max_llm_calls,
    )


def agent(
    *,
    use_hyde: bool = False,
    use_fact_queries: bool = True,
    use_retry: bool = True,
    use_tools: bool = False,
    use_query_variants: bool = True,
    use_two_pass: bool = True,
    use_bm25_anchor: bool = False,
    use_factoid_expand: bool = True,
    use_factoid_retry: bool = True,
    use_stitch: bool = True,
    max_llm_calls: int = 12,
    max_fact_queries: int = 2,
    top_k: int = 10,
    rerank: bool = True,
) -> BrainRound:
    """Full agent pipeline: plan -> expand -> retrieve -> generate -> evaluate."""
    expand: list = [PlanLLMStep(), DetectLangStep()]
    if use_hyde:
        expand.append(HyDEStep())
    if use_query_variants:
        expand.append(QueryVariantsStep())

    post_retrieve: list = []
    if use_fact_queries:
        expand.append(FactQueryStep(max_queries=max_fact_queries))
        post_retrieve.append(QualityCheckStep())
    if use_two_pass:
        post_retrieve.append(TwoPassStep())
    if use_bm25_anchor:
        post_retrieve.append(BM25AnchorStep())
    if use_factoid_expand:
        post_retrieve.append(FactoidExpandStep())
    if use_stitch:
        post_retrieve.append(StitchStep())

    evaluate: list = []
    if use_retry:
        evaluate.append(AssessStep())
        evaluate.append(SupplementalRetrieveStep())
    if use_factoid_retry:
        evaluate.append(FactoidRetryStep())

    return BrainRound(
        expand=expand,
        retrieve=BrainRetrieveStep(preset="hybrid", top_k=top_k, rerank=rerank),
        post_retrieve=post_retrieve,
        generate=GenerateStep(use_tools=use_tools, stream=True),
        evaluate=evaluate,
        max_llm_calls=max_llm_calls,
    )


def retry_round(
    *,
    top_k: int = 10,
    use_stitch: bool = True,
    use_hyde: bool = True,
    use_fact_queries: bool = True,
    max_fact_queries: int = 2,
) -> BrainRound:
    """Retry round: HyDE + keywords + fact queries -> thorough retrieve -> generate.

    Mirrors the old agent-search retry which used HyDE, missing-term queries,
    and keyword expansion before re-retrieving at 2x top_k.
    """
    expand: list = []
    if use_hyde:
        expand.append(HyDEStep())
    expand.append(KeywordStep())
    if use_fact_queries:
        expand.append(FactQueryStep(max_queries=max_fact_queries))

    return BrainRound(
        expand=expand,
        retrieve=BrainRetrieveStep(preset="thorough", top_k=max(12, top_k * 2), rerank=True),
        post_retrieve=[StitchStep()] if use_stitch else [],
        generate=GenerateStep(stream=True),
        max_llm_calls=6,
    )


def minimal() -> BrainRound:
    return agent(
        use_hyde=False,
        use_fact_queries=False,
        use_retry=False,
        use_query_variants=False,
        use_two_pass=False,
        use_bm25_anchor=False,
        use_factoid_expand=False,
        use_factoid_retry=False,
        use_stitch=False,
        max_llm_calls=4,
        max_fact_queries=0,
    )


def conservative() -> BrainRound:
    return agent(
        use_hyde=False,
        use_fact_queries=False,
        use_retry=False,
        use_query_variants=True,
        use_two_pass=False,
        use_bm25_anchor=False,
        use_factoid_expand=True,
        use_factoid_retry=True,
        use_stitch=True,
        max_llm_calls=6,
        max_fact_queries=0,
    )


def aggressive() -> BrainRound:
    return agent(
        use_hyde=True,
        use_fact_queries=True,
        use_retry=True,
        use_query_variants=True,
        use_two_pass=True,
        use_bm25_anchor=True,
        use_factoid_expand=True,
        use_factoid_retry=True,
        use_stitch=True,
        max_llm_calls=16,
        max_fact_queries=4,
    )


def gate(
    *,
    preset: str = "hybrid",
    top_k: int = 20,
    rerank: bool = True,
    use_bm25_anchor: bool = True,
    use_two_pass: bool = True,
    use_stitch: bool = True,
) -> BrainRound:
    """Replicates the gate /v1/chat pipeline.

    Single-round, zero LLM expansion calls. All retrieval improvements
    are heuristic.
    """
    post_retrieve: list = [QualityCheckStep()]
    if use_bm25_anchor:
        post_retrieve.append(BM25AnchorStep(top_k=min(top_k, 30)))
    if use_two_pass:
        post_retrieve.append(TwoPassStep(min_unique_sources=3))
    post_retrieve.append(FactoidExpandStep())
    if use_stitch:
        post_retrieve.append(StitchStep(max_per_segment=4))

    return BrainRound(
        expand=[QueryVariantsStep()],
        retrieve=BrainRetrieveStep(preset=preset, top_k=top_k, rerank=rerank),
        post_retrieve=post_retrieve,
        generate=GenerateStep(stream=False, use_tools=False),
        evaluate=[FactoidRetryStep()],
        max_llm_calls=3,
    )


AGENT_PRESET_BUILDERS = {
    "minimal": minimal,
    "conservative": conservative,
    "aggressive": aggressive,
    "gate": gate,
}
