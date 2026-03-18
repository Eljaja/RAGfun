"""Named BrainPlan constructors replacing hardcoded pipelines."""

from __future__ import annotations

from models import (
    AssessStep,
    BM25AnchorStep,
    BrainPlan,
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
    TwoPassStep,
)


def simple(
    *,
    preset: str = "hybrid",
    top_k: int = 5,
    rerank: bool = True,
    max_retries: int = 1,
    reflection_enabled: bool = True,
) -> BrainPlan:
    """Simple /chat pipeline: retrieve -> generate, with optional reflect + retry rounds."""
    retrieve = BrainRetrieveStep(preset=preset, top_k=top_k, rerank=rerank)
    generate = GenerateStep(stream=False)

    first_round = BrainRound(
        expand=[QueryVariantsStep()],
        retrieve=retrieve,
        post_retrieve=[StitchStep()],
        generate=generate,
        evaluate=[ReflectStep()] if reflection_enabled and max_retries > 0 else [],
    )

    rounds = [first_round]
    for _ in range(max_retries if reflection_enabled else 0):
        rounds.append(BrainRound(
            retrieve=retrieve,
            generate=generate,
            evaluate=[ReflectStep()],
        ))

    max_llm_calls = 1 + (max_retries * 2 if reflection_enabled else 0)
    return BrainPlan(rounds=rounds, max_llm_calls=max(max_llm_calls, 2))


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
) -> BrainPlan:
    """Full agent pipeline: plan -> expand -> retrieve -> generate -> assess."""
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

    retrieve = BrainRetrieveStep(preset="hybrid", top_k=top_k, rerank=rerank)
    generate = GenerateStep(use_tools=use_tools, stream=True)

    evaluate: list = []
    if use_retry:
        evaluate.append(AssessStep())
    if use_factoid_retry:
        evaluate.append(FactoidRetryStep())

    first_round = BrainRound(
        expand=expand,
        retrieve=retrieve,
        post_retrieve=post_retrieve,
        generate=generate,
        evaluate=evaluate,
    )

    rounds = [first_round]
    if use_retry:
        retry_round = BrainRound(
            expand=[KeywordStep()],
            retrieve=BrainRetrieveStep(preset="thorough", top_k=top_k, rerank=True),
            post_retrieve=[StitchStep()] if use_stitch else [],
            generate=GenerateStep(stream=True),
        )
        rounds.append(retry_round)

    return BrainPlan(rounds=rounds, max_llm_calls=max_llm_calls)


def minimal() -> BrainPlan:
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


def conservative() -> BrainPlan:
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


def aggressive() -> BrainPlan:
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
) -> BrainPlan:
    """Replicates the gate /v1/chat pipeline.

    Single-round, zero LLM expansion calls. All retrieval improvements
    are heuristic: multi-query variants, BM25 anchor, two-pass hint
    expansion, factoid within-doc expand, segment stitching, and
    factoid post-generation grounding retry.

    Gate flow mapped to BrainPlan steps:
      _query_variants()           -> QueryVariantsStep  (expand)
      hybrid search fan-out       -> _exec_retrieve multi-query
      _apply_bm25_anchor_pass()   -> BM25AnchorStep     (post-retrieve)
      two_pass (hint terms)       -> TwoPassStep         (post-retrieve)
      factoid pre-expansion       -> FactoidExpandStep   (post-retrieve)
      segment stitching           -> StitchStep          (post-retrieve)
      build_messages (history)    -> history_as_messages  (generate)
      _strip_thinking_text        -> ThinkStripper        (llm.stream)
      factoid grounding retry     -> FactoidRetryStep    (evaluate)
    """
    post_retrieve: list = [QualityCheckStep()]
    if use_bm25_anchor:
        post_retrieve.append(BM25AnchorStep(top_k=min(top_k, 30)))
    if use_two_pass:
        post_retrieve.append(TwoPassStep(min_unique_sources=3))
    post_retrieve.append(FactoidExpandStep())
    if use_stitch:
        post_retrieve.append(StitchStep(max_per_segment=4))

    return BrainPlan(
        rounds=[
            BrainRound(
                expand=[QueryVariantsStep()],
                retrieve=BrainRetrieveStep(preset=preset, top_k=top_k, rerank=rerank),
                post_retrieve=post_retrieve,
                generate=GenerateStep(stream=False, use_tools=False),
                evaluate=[FactoidRetryStep()],
            ),
        ],
        max_llm_calls=3,
    )


AGENT_PRESET_BUILDERS = {
    "minimal": minimal,
    "conservative": conservative,
    "aggressive": aggressive,
    "gate": gate,
}
