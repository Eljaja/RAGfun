# Generator v2 Architecture

Target architecture for the generator pipeline. This document captures the design decisions
from the architecture review and serves as the blueprint for refactoring.

## Three Layers of Control

```
Layer 3: Planning + Recovery   PlanAgent + run_with_recovery
         initial plan and replan are the same agent, different context
         N attempts, honest "I don't know" when nothing works

Layer 2: Generator Pipeline    run_pipeline
         configure → retrieve → generate → evaluate
         one plan, one pass, four honest phases

Layer 2b: Retrieval Pipeline   run_retrieval (inside retrieve phase)
          initial_expand → fetch → loop(filter → expand → fetch) → finalize
          loop expansion can be agent-driven on retry

Layer 1: Sub-steps             fold over narrow steps within each phase
         one LLM call, one job, one guardrail
```

Future layer 4: Orchestration — multi-plan, query decomposition, parallel strategies.

## Generator Pipeline (Layer 2)

Four phases, explicit typed wiring. Early halt when retrieval finds nothing —
don't waste LLM calls generating from empty context.

```python
async def run_pipeline(plan, *, query, ...) -> PipelineResult:
    meta = await configure(plan.configure, query=query, ...)
    ret  = await run_retrieval(plan.retrieval, query=query, meta=meta, ...)

    if not ret.chunks:
        return PipelineResult(answer=None, chunks=[], halted=True,
                              halt_reason="retrieval returned nothing")

    answer  = await generate(plan.generate, chunks=ret.chunks, lang=meta.lang, ...)
    verdict = await evaluate(plan.evaluate, answer=answer, chunks=ret.chunks, ...)
    return PipelineResult(answer=answer, chunks=ret.chunks, verdict=verdict)
```

### Phase contracts

| Phase     | Input                | Output             | IO allowed         |
|-----------|----------------------|--------------------|---------------------|
| configure | query, history       | metadata           | LLM calls           |
| retrieve  | query, metadata      | chunks             | LLM + retrieval     |
| generate  | chunks, query, lang  | answer             | LLM calls           |
| evaluate  | answer, chunks       | verdict            | LLM calls           |

No phase does retrieval except retrieve. No phase produces an answer except generate.
Evaluate only judges — it never retrieves or generates.

## Retrieval Pipeline (Layer 2b)

All chunk acquisition lives here. The generator calls it as a black box.

```python
async def run_retrieval(plan, *, query, meta, ...) -> RetrievalResult:
    default_config = meta.retrieval_plan or from_preset(...)

    # Initial expand: one-time query generation (no chunks yet)
    requests = [RetrievalRequest(query=query)]
    requests += await initial_expand(plan.initial_expand, query=query, ...)

    # First fetch — each request may override the default config
    chunks = await fetch_all(requests, default_config=default_config, ...)

    # Always filter at least once
    chunks = await filter_chunks(plan.loop_filter, chunks=chunks, query=query, ...)

    # Loop: check → expand → fetch → filter
    for _ in range(plan.max_rounds - 1):
        if not await should_continue(plan.loop_check, query=query, chunks=chunks, ...):
            break
        requests = await loop_expand(plan.loop_expand, query=query, chunks=chunks, ...)
        if not requests:
            break
        extra = await fetch_all(requests, default_config=default_config, ...)
        chunks = merge_chunks([chunks, extra])
        chunks = await filter_chunks(plan.loop_filter, chunks=chunks, query=query, ...)

    # Finalize: stitch, annotate, sufficiency check
    chunks = await finalize(plan.finalize, chunks=chunks, query=query, ...)

    return RetrievalResult(chunks=chunks, ...)
```

`fetch_all` uses `request.plan_override` when present, `default_config` when not:

```python
async def fetch_all(requests, *, default_config, ...) -> list[ChunkResult]:
    all_chunks = []
    for req in requests:
        config = req.plan_override or default_config
        result = await retrieval_call(..., query=req.query, plan=config)
        all_chunks.extend(result)
    return all_chunks
```

### Who decides what to retrieve

```
1. RetrievalPlan.default_config        static default (preset, top_k, rerank)
       ↑ overridden by
2. PlanAgent (produces the BrainRound)  decides retrieval strategy, step selection
       ↑ overridden per-request by
3. Any expand step via plan_override   step-specific (BM25Anchor → fast; Reflect → thorough)
```

retrieve_all is a dumb executor. It runs whatever requests it's given with whatever config.

On run 1, the PlanAgent (or preset) produces a RetrievalPlan with heuristic
loop_check + loop_expand steps. On run 2+, the PlanAgent has failure context
and can wire in a RetrievalReflectStep that reasons about what to search for
next (see Retrieval subagent below).

## Planning + Recovery (Layer 3)

Separate from the pipeline. Produces plans and runs complete pipelines.

### Unified PlanAgent (planned)

Initial planning and replanning are the same operation with different context:

```python
class PlanAgent:
    async def plan(
        self,
        query: str,
        history: list[Message],
        prev_result: PipelineResult | None = None,
    ) -> BrainRound:
        ...
```

`prev_result=None` → initial planning (only knows the query).
`prev_result=<failed>` → replanning (knows what went wrong, what chunks were found,
what strategy was used). The agent can change anything in the new plan — including
wiring in agent-driven retrieval steps for the retry (see Retrieval subagent below).

Now: presets produce the initial BrainRound. PlanLLMStep in configure adjusts
retrieval settings. Later: PlanAgent replaces both — one agent, one call.

### Recovery loop

One concept: a **retry strategy** looks at a failed result and either returns a new
plan to try, or `None` meaning "I can't help here." When no strategy can help, that's
the honest "I don't know." The only hard stop is the budget — a constraint, not a strategy.

```python
class RetryStrategy(Protocol):
    async def replan(
        self,
        result: PipelineResult,
        history: list[PipelineResult],
        current_plan: BrainRound,
    ) -> BrainRound | None:
        """Return a new plan, or None if this strategy can't help."""
        ...
```

```python
async def run_with_recovery(
    plan: BrainRound,
    strategies: list[RetryStrategy],
    *,
    max_attempts: int = 3,
    budget: BudgetCounter,
    **kwargs,
) -> PipelineResult:
    history: list[PipelineResult] = []
    current_plan = plan

    for _ in range(max_attempts):
        result = await run_pipeline(current_plan, budget=budget, **kwargs)
        history.append(result)

        if result.verdict and result.verdict.acceptable:
            return result

        if budget.remaining < budget.min_for_attempt:
            return _honest_failure("budget exhausted", history)

        new_plan = None
        for strategy in strategies:
            if new_plan := await strategy.replan(result, history, current_plan):
                break

        if new_plan is None:
            return _honest_failure("no strategy could help", history)

        current_plan = new_plan

    return _honest_failure("max attempts reached", history)


def _honest_failure(reason: str, history: list[PipelineResult]) -> PipelineResult:
    return PipelineResult(
        answer=None,
        chunks=_best_chunks_across(history),
        verdict=Verdict(acceptable=False, reason=reason),
        gave_up=True,
        attempts=history,
    )
```

Three ways the loop ends honestly:

1. `budget exhausted` — hard constraint, checked before consulting strategies
2. `no strategy could help` — every strategy returned None for this failure
3. `max attempts reached` — tried N times, none were acceptable

### FallbackPreset (now)

The working v1 strategy. Switches to a different plan on first failure.

```python
class FallbackPreset:
    """First attempt failed → try a different preset entirely."""
    def __init__(self, fallback: BrainRound):
        self.fallback = fallback

    async def replan(self, result, history, current_plan):
        if len(history) > 1:
            return None                     # already tried fallback
        return self.fallback
```

### Agent replanner (planned)

The strategy that replaces FallbackPreset once built. An LLM agent receives
full context about what happened and reasons about what to try differently.

```python
@dataclass
class ReplanContext:
    query:        str
    verdict:      Verdict                   # what went wrong
    chunks:       list[ChunkResult]         # what we found (or didn't)
    history:      list[PipelineResult]      # everything tried so far
    current_plan: BrainRound               # the plan that just failed
    attempt:      int                       # which attempt this was
    budget_remaining: int                   # how much budget is left

class AgentReplanner:
    def __init__(self, llm, budget: BudgetCounter):
        self.llm = llm
        self.budget = budget

    async def replan(self, result, history, current_plan):
        if len(history) >= 2:
            prev = history[-2].verdict
            if prev and prev.reason == result.verdict.reason:
                return None                 # same failure twice, replanning won't help

        ctx = ReplanContext(
            query=result.query,
            verdict=result.verdict,
            chunks=result.chunks,
            history=history,
            current_plan=current_plan,
            attempt=len(history),
            budget_remaining=self.budget.remaining,
        )

        return await self.llm.structured_output(
            prompt=REPLAN_PROMPT,
            context=ctx,
            response_model=BrainRound,      # agent outputs a full plan
        )
```

The agent can change anything in the plan: retrieval strategy, expand steps,
generation prompt, which evaluate steps to run. It's not limited to "widen top_k."
The structured output constraint (`response_model=BrainRound`) means the agent
produces valid plans by construction — Pydantic validates the output.

This is the same agent as PlanAgent — `AgentReplanner.replan()` is just
`PlanAgent.plan(prev_result=result)` wrapped in the `RetryStrategy` interface.

### Retrieval subagent (planned)

On run 2+, the PlanAgent knows what went wrong and can wire a smarter retrieval
loop. `RetrievalReflectStep` is the LLM-driven loop expand step — same interface
as the heuristic ones, smarter inside:

```python
async def _retrieval_reflect(step, *, query, chunks, budget, llm, ...) -> list[RetrievalRequest]:
    decision = await llm.structured_output(
        prompt=RETRIEVAL_REFLECT_PROMPT,
        context=RetrievalReflectContext(
            query=query,
            chunks_so_far=chunks,
            round=current_round,
            budget_remaining=budget.remaining,
        ),
        response_model=RetrievalReflectDecision,
    )
    if decision.stop:
        return []                               # "enough chunks"
    return [
        RetrievalRequest(query=q, plan_override=decision.config_override)
        for q in decision.queries
    ]
```

The retrieval pipeline doesn't change structure — `run_retrieval` still loops over
`plan.loop_expand` steps. The step just happens to be LLM-driven on retry:

```
Run 1 (preset):    loop_expand = [TwoPassStep(), QualityCheckStep()]
Run 2 (replanned): loop_expand = [RetrievalReflectStep()]
```

The PlanAgent decides which version to use based on what failed.

### Strategy list is configurable

The strategy list is not hardcoded — it's part of the recovery configuration.

```python
# Now: FallbackPreset is the only strategy.
strategies = [FallbackPreset(retry_plan)]

# Later: agent replanner replaces it. The agent can output any BrainRound —
# including a known preset if it thinks a swap is the right move.
strategies = [AgentReplanner(llm, budget)]
```

The interface stays the same. When the agent replanner is ready, swap it into
the strategy list. For hard queries, the agent uses the verdict and chunk history
to design a targeted plan instead of a blind preset swap.

Recovery strategies are named, testable, composable. Not inline code in endpoints.

## Plan Types

```python
class BrainRound(BaseModel):
    configure:      list[ConfigStep]
    retrieval:      RetrievalPlan
    generate:       list[GenerateStep]
    evaluate:       list[EvalStep]
    max_llm_calls:  int = 30

class RetrievalPlan(BaseModel):
    default_config:  BrainRetrieveStep
    initial_expand:  list[ExpandStep]       # HyDE, FactQuery, Keywords, etc.
    loop_check:      list[LoopCheckStep]    # should we keep retrieving?
    loop_expand:     list[LoopExpandStep]   # what to search for next
    loop_filter:     list[FilterStep]       # chunk quality filter
    finalize:        list[FinalizeStep]     # Stitch, annotate, sufficiency
    max_rounds:      int = 2

@dataclass
class RetrievalRequest:
    query: str
    plan_override: ExecutionPlan | None = None

@dataclass
class PipelineResult:
    answer:      str | None           # None when halted or gave up
    chunks:      list[ChunkResult]
    verdict:     Verdict | None
    halted:      bool = False         # pipeline stopped early (e.g. empty retrieval)
    halt_reason: str | None = None
    gave_up:     bool = False         # recovery exhausted, honest "I don't know"
    attempts:    list[PipelineResult] = field(default_factory=list)
```

`answer=None` is the signal. The endpoint checks `gave_up` or `halted` and formats
an honest response ("I wasn't able to find relevant information") instead of
forcing a hallucinated answer from nothing.

## Primitive Operations

| Operation | Type signature                                  | Where                        |
|-----------|-------------------------------------------------|------------------------------|
| Configure | `query → metadata`                              | configure phase              |
| Expand    | `(query, chunks?) → [RetrievalRequest]`         | initial_expand, loop_expand  |
| Check     | `(query, chunks) → bool`                        | loop_check                   |
| Filter    | `chunks → chunks`                               | loop_filter, finalize        |
| Fetch     | `[RetrievalRequest] → chunks`                   | fetch_all (per-request config) |
| Halt      | `chunks=[] → PipelineResult(answer=None)`       | pipeline (early exit)        |
| Generate  | `chunks → answer`                               | generate phase               |
| Judge     | `answer → verdict`                              | evaluate phase               |
| Replan    | `(result, history, plan) → BrainRound \| None`  | recovery strategies          |

No step crosses these boundaries. Halt is control flow. Replan is the only operation
that produces a new plan — heuristic strategies do it mechanically, the agent replanner
does it by reasoning about what went wrong.

## Step Inventory

### Configure steps (produce metadata, no queries)

| Step           | LLM/heuristic | Produces                          |
|----------------|---------------|-----------------------------------|
| PlanLLMStep    | LLM           | retrieval_plan, retrieval_mode    |
| DetectLangStep | LLM           | lang, is_factoid                  |

Future: QueryClassifyStep, RelevanceGateStep, AmbiguityCheckStep.

### Expand steps — initial (produce queries, no chunk dependency)

| Step              | LLM/heuristic | Produces                                      |
|-------------------|---------------|-----------------------------------------------|
| HyDEStep          | LLM           | hypothetical passage as query                 |
| FactQueryStep     | LLM           | sub-queries from query decomposition          |
| KeywordStep       | LLM           | keyword queries                               |
| QueryVariantsStep | heuristic     | reformulated queries                          |
| BM25AnchorStep    | heuristic     | keyword query with plan_override=fast         |

### Loop check steps (should we keep retrieving?)

| Step                  | LLM/heuristic | Returns false when                         |
|-----------------------|---------------|--------------------------------------------|
| QualityCheckStep      | heuristic     | enough unique sources, scores above threshold |
| BudgetCheckStep       | heuristic     | LLM budget too low for another round       |

Future: LLMSufficiencyCheck — ask an LLM "do these chunks cover the query?"

### Expand steps — loop (produce queries, need chunks)

| Step                   | LLM/heuristic | Produces                                   |
|------------------------|---------------|--------------------------------------------|
| TwoPassStep            | heuristic     | follow-up query from chunk hint terms      |
| FactoidExpandStep      | heuristic     | re-retrieval request with fast preset      |
| RetrievalReflectStep   | LLM           | targeted queries + config overrides (planned) |

### Filter steps (chunks → fewer/better chunks)

| Step                  | LLM/heuristic | What it does                        |
|-----------------------|---------------|-------------------------------------|
| (none today)          |               |                                     |
| ChunkRelevanceFilter  | heuristic     | score-based relevance threshold     |
| LLMChunkFilter        | LLM           | per-chunk relevance check           |

### Finalize steps (end-of-retrieval processing)

| Step              | LLM/heuristic | What it does                         |
|-------------------|---------------|--------------------------------------|
| StitchStep        | heuristic     | merge adjacent chunks from same doc  |
| ChunkAnnotateStep | LLM           | annotate each chunk with claims      |
| SufficiencyCheck  | LLM           | enough context to answer?            |

### Generate steps (produce answer)

| Step              | LLM/heuristic | What it does                         |
|-------------------|---------------|--------------------------------------|
| GenerateStep      | LLM           | draft answer from chunks + query     |

Future: DraftStep, GroundingCheckStep, ToneCheckStep, ReviseStep (split the single
GenerateStep into narrow sub-steps with guardrails).

### Evaluate steps (judge answer, produce verdict only)

| Step               | LLM/heuristic | What it does                        |
|--------------------|---------------|-------------------------------------|
| ReflectStep        | LLM           | is the answer complete? → requery   |
| AssessStep         | LLM           | what's missing? → missing_terms     |
| GroundingCheckStep | heuristic     | is answer grounded in chunks?       |

Evaluate never retrieves. Evaluate never generates. Recovery handles retry.

## Refactoring Plan

### Phase 1: Fix dishonest steps

Six functions do retrieval or generation outside their phase. Each task is one file,
one specific change — small enough for a fast model with precise instructions.

**expand.py — make query-only:**

| # | Function | Keep | Delete | Return type change |
|---|----------|------|--------|--------------------|
| 1 | `_fact_queries` | LLM call that generates sub-queries | `_safe_retrieve` calls on each sub-query | `list[ChunkResult]` → `list[str]` |
| 2 | `_keywords` | LLM call that generates keywords | `_safe_retrieve` calls on each keyword | `list[ChunkResult]` → `list[str]` |
| 3 | `_safe_retrieve` | — | entire function | (deleted) |

After: `expand.py` has no `http_client`, no `settings`, no retrieval imports.

**evaluate.py — evaluate only:**

| # | Function | Keep | Delete |
|---|----------|------|--------|
| 4 | `_supplemental_retrieve` | — | entire function (keyword extraction + retrieve + generate) |
| 5 | `_factoid_retry` | extract `answer_is_grounded` check → new `GroundingCheckStep` | rest of function (retrieve + generate) |

After: `evaluate.py` has no `http_client`, no `settings`, no retrieval imports.
`_reflect` and `_assess` stay unchanged.

**enrich.py — dissolve into correct phases:**

| # | Step branch | Query logic to extract | Delete |
|---|-------------|------------------------|--------|
| 6a | `TwoPassStep` | hint term extraction + follow-up query building → `LoopExpandStep` | `_safe_retrieve` call |
| 6b | `BM25AnchorStep` | keyword query building → `ExpandStep` (initial, with `plan_override`) | `_safe_retrieve` call |
| 6c | `FactoidExpandStep` | original query reuse → `LoopExpandStep` (with `plan_override`) | `_safe_retrieve` call |
| 6d | `StitchStep` | — (already honest) → `FinalizeStep` in retrieval pipeline | — |

After: `enrich.py` deleted. Logic lives in expand.py (query producers) and
retrieval.py (stitch in finalize).

### Phase 1b: Move honest functions to correct phases

| Function | From | To | Why |
|----------|------|----|-----|
| `_plan_retrieval` + `_plan_retrieval_llm` | expand.py | steps/configure.py | produces metadata, not queries |
| `_detect_lang` | expand.py | steps/configure.py | produces metadata, not queries |
| `StitchStep` logic | enrich.py | retrieval pipeline finalize | chunk transformation, not enrichment |

### Phase 1c: Structural wiring

7. **Create `engine/retrieval.py`** with `run_retrieval`. All retrieval IO moves here.
   The cleaned expand steps produce queries; retrieval.py executes them.

8. **Create `engine/recovery.py`** with `run_with_recovery`, `RetryStrategy` protocol,
   `FallbackPreset`, `_honest_failure`. Duplicated retry logic in endpoints.py replaced.

9. **Fix type holes.** `retrieval_plan: Any` → `ExecutionPlan | None`.
   `brain_plan: Any` → `BrainRound`.

10. **Extract shared models.** `ChunkResult`, `ScoreSource`, retrieval step types,
    `ExecutionPlan`, `PlanRound` into a shared `rag_types` package.

### Phase 2: Fill gaps

11. **Real streaming.** generate returns `AsyncIterator[str]`. Pipeline yields tokens
    as they arrive instead of batching.

12. **Prometheus metrics.** Per-phase latency, LLM call counts by stage, retrieval
    round counts, retry rates.

13. **Mood traces.** Restore MEME_GRUMPS. 10 lines in endpoints.

### Phase 3: Add narrow steps (ongoing)

14. Add steps one at a time as failure modes are identified. Each step is: define model,
    add to union, add case to match, write prompt. Architecture doesn't change.

    Priority candidates:
    - RetrievalReflectStep (LLM-driven loop expansion, planned)
    - ChunkRelevanceFilter (heuristic, in loop_filter)
    - SufficiencyCheck (LLM, in finalize)
    - DraftStep + ReviseStep (split generate into narrow sub-steps)

### Phase 4: Higher layers (when needed)

15. PlanAgent — unified initial planning + replanning.
16. Orchestration layer for multi-plan, query decomposition, parallel strategies.
17. Ingestion-time enrichment in doc_processor_v2 (topic annotation, question generation).

## File Structure After Refactoring

```
engine/
  pipeline.py          ~30 lines    generator pipeline: configure → retrieve → generate → evaluate
  retrieval.py         ~60 lines    retrieval pipeline: expand → fetch → loop → finalize
  recovery.py          ~60 lines    run_with_recovery + RetryStrategy protocol + _honest_failure
  budget.py            (unchanged)

steps/
  configure.py         ~80 lines    _plan_retrieval, _plan_retrieval_llm, _detect_lang
  expand.py            ~150 lines   HyDE, FactQuery (query-only), Keywords (query-only),
                                    QueryVariants, BM25Anchor, TwoPass, FactoidExpand,
                                    QualityCheck (+ future RetrievalReflect)
  generate.py          ~60 lines    GenerateStep (future: Draft, Grounding, Tone, Revise)
  evaluate.py          ~80 lines    Reflect, Assess, GroundingCheck
  enrich.py            (deleted — dissolved into expand.py and retrieval pipeline)

models/
  steps.py             step unions: ConfigStep, ExpandStep, LoopCheckStep, LoopExpandStep,
                       FilterStep, FinalizeStep, GenerateStep, EvalStep
  plan.py              BrainRound, RetrievalPlan, RetrievalRequest, RetrievalResult,
                       ConfigMeta, Verdict, PipelineResult
  (rest unchanged)

endpoints.py           calls run_with_recovery, no inline retry logic
brain_presets.py       updated to new BrainRound shape
```

Total pipeline code: ~460 lines, down from ~900.
Every phase is honest. All retrieval IO in one file. Recovery in one file.
enrich.py deleted — its logic lives where it belongs.

## Design Principles

1. **Steps are honest.** Each step does exactly one primitive operation. No step crosses
   phase boundaries. If a step wants to both filter chunks AND produce queries, it's
   two steps.

2. **Plans are data.** BrainRound and RetrievalPlan are values that can be inspected,
   serialized, logged, diffed, and generated by an LLM. The interpreter is dumb.

3. **Phases have typed contracts.** configure produces metadata. expand produces queries.
   filter produces fewer chunks. generate produces an answer. evaluate produces a verdict.
   The type signatures enforce the boundaries.

4. **Retrieval is a black box.** The generator pipeline says "give me chunks" and gets
   chunks. It doesn't know about multi-round, augmentation, filtering, or stitching.

5. **Recovery is above the pipeline.** The pipeline is a pure single-pass function. Retry
   strategies live in the recovery layer, not inside steps. Heuristic strategies handle
   common patterns cheaply; the agent replanner handles hard cases by reasoning about
   what went wrong. When nobody can help, recovery says "I don't know."

6. **Add bricks, not framework.** New capabilities come from adding narrow steps to
   existing phases. The architecture doesn't change. Each new step is: model + case + prompt.
