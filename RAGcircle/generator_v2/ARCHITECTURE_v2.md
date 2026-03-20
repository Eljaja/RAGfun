# Generator v2 Architecture

Blueprint for refactoring the generator pipeline. Captures every design decision
from the architecture review.

## Layers

```
Layer 3  Planning + Recovery     PlanAgent + run_with_recovery
         produces plans, runs N attempts, honest "I don't know"

Layer 2  Generator Pipeline      run_pipeline
         configure → retrieve → generate → evaluate

Layer 2b Retrieval Pipeline      run_retrieval (inside retrieve phase)
         expand → fetch → loop(check → expand → fetch → filter) → finalize

Layer 1  Sub-steps               fold over narrow steps within each phase
         one LLM call, one job, one guardrail
```

Future: Layer 4 Orchestration — multi-plan, query decomposition, parallel strategies.

---

## Layer 2: Generator Pipeline

Four phases, typed wiring. Early halt when retrieval finds nothing.

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

| Phase     | Input                | Output   | IO            |
|-----------|----------------------|----------|---------------|
| configure | query, history       | metadata | LLM           |
| retrieve  | query, metadata      | chunks   | LLM + HTTP    |
| generate  | chunks, query, lang  | answer   | LLM           |
| evaluate  | answer, chunks       | verdict  | LLM           |

No phase does retrieval except retrieve. No phase produces an answer except generate.
Evaluate only judges — it never retrieves or generates.

---

## Layer 2b: Retrieval Pipeline

All chunk acquisition lives here. The generator calls it as a black box.

```python
async def run_retrieval(plan, *, query, meta, ...) -> RetrievalResult:
    default_config = meta.retrieval_plan or from_preset(...)

    # Initial expand (no chunks yet)
    requests = [RetrievalRequest(query=query)]
    requests += await initial_expand(plan.initial_expand, query=query, ...)

    # First fetch — each request may override default_config via plan_override
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

    # Finalize: stitch, annotate, sufficiency
    chunks = await finalize(plan.finalize, chunks=chunks, query=query, ...)

    return RetrievalResult(chunks=chunks, ...)
```

### fetch_all respects per-request config

```python
async def fetch_all(requests, *, default_config, ...) -> list[ChunkResult]:
    all_chunks = []
    for req in requests:
        config = req.plan_override or default_config
        result = await retrieval_call(..., query=req.query, plan=config)
        all_chunks.extend(result)
    return all_chunks
```

Most expand steps only produce queries — `HyDEStep` returns
`RetrievalRequest(query="hypothetical passage...")` with no override.
Steps that care about config set `plan_override` — e.g., `BM25AnchorStep`
returns `RetrievalRequest(query="keywords...", plan_override=fast_config)`.

### Who decides retrieval config

```
1. RetrievalPlan.default_config       static default (preset, top_k, rerank)
       ↑ overridden by
2. PlanAgent (produces the BrainRound) decides strategy, step selection
       ↑ overridden per-request by
3. Any expand step via plan_override  step-specific override
```

---

## Layer 3: Planning + Recovery

### PlanAgent (planned)

Initial planning and replanning are the same operation, different context:

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

`prev_result=None` → initial plan (only knows the query).
`prev_result=<failed>` → replan (knows verdict, chunks, strategy used). Can change
anything — including wiring in LLM-driven retrieval steps for the retry.

Now: presets produce the initial BrainRound. Later: PlanAgent replaces presets.

### Recovery loop

A **retry strategy** looks at a failed result and either returns a new plan or `None`
("I can't help"). When no strategy can help, honest "I don't know." The only hard
stop is the budget.

```python
class RetryStrategy(Protocol):
    async def replan(
        self,
        result: PipelineResult,
        history: list[PipelineResult],
        current_plan: BrainRound,
    ) -> BrainRound | None: ...

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

Three honest exits: budget exhausted, no strategy could help, max attempts reached.

### FallbackPreset (now)

Working v1 strategy. Switches to a different plan on first failure.

```python
class FallbackPreset:
    def __init__(self, fallback: BrainRound):
        self.fallback = fallback

    async def replan(self, result, history, current_plan):
        if len(history) > 1:
            return None
        return self.fallback
```

### AgentReplanner (planned)

Replaces FallbackPreset. Same agent as PlanAgent, wrapped in `RetryStrategy`.

```python
@dataclass
class ReplanContext:
    query:            str
    verdict:          Verdict
    chunks:           list[ChunkResult]
    history:          list[PipelineResult]
    current_plan:     BrainRound
    attempt:          int
    budget_remaining: int

class AgentReplanner:
    def __init__(self, llm, budget: BudgetCounter):
        self.llm = llm
        self.budget = budget

    async def replan(self, result, history, current_plan):
        if len(history) >= 2:
            prev = history[-2].verdict
            if prev and prev.reason == result.verdict.reason:
                return None

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
            response_model=BrainRound,
        )
```

`response_model=BrainRound` — the agent outputs a valid plan by construction.
Can change anything: retrieval strategy, expand steps, generation prompt.

### Retrieval subagent (planned)

On run 2+, the PlanAgent can wire `RetrievalReflectStep` into `loop_expand` —
an LLM-driven step that replaces the heuristic ones. Same interface, smarter inside:

```python
async def _retrieval_reflect(step, *, query, chunks, budget, llm, ...) -> list[RetrievalRequest]:
    decision = await llm.structured_output(
        prompt=RETRIEVAL_REFLECT_PROMPT,
        context=RetrievalReflectContext(
            query=query, chunks_so_far=chunks,
            round=current_round, budget_remaining=budget.remaining,
        ),
        response_model=RetrievalReflectDecision,
    )
    if decision.stop:
        return []
    return [
        RetrievalRequest(query=q, plan_override=decision.config_override)
        for q in decision.queries
    ]
```

The retrieval pipeline structure doesn't change — only the steps plugged in:

```
Run 1 (preset):    loop_expand = [TwoPassStep(), FactoidExpandStep()]
Run 2 (replanned): loop_expand = [RetrievalReflectStep()]
```

### Strategy list is configurable

```python
# Now
strategies = [FallbackPreset(retry_plan)]

# Later — agent replanner, can output any BrainRound including a known preset
strategies = [AgentReplanner(llm, budget)]
```

---

## Types

```python
class BrainRound(BaseModel):
    configure:      list[ConfigStep]
    retrieval:      RetrievalPlan
    generate:       list[GenerateStep]
    evaluate:       list[EvalStep]
    max_llm_calls:  int = 30

class RetrievalPlan(BaseModel):
    default_config:  BrainRetrieveStep
    initial_expand:  list[ExpandStep]
    loop_check:      list[LoopCheckStep]
    loop_expand:     list[LoopExpandStep]
    loop_filter:     list[FilterStep]
    finalize:        list[FinalizeStep]
    max_rounds:      int = 2

@dataclass
class RetrievalRequest:
    query: str
    plan_override: ExecutionPlan | None = None

@dataclass
class PipelineResult:
    answer:      str | None
    chunks:      list[ChunkResult]
    verdict:     Verdict | None
    halted:      bool = False
    halt_reason: str | None = None
    gave_up:     bool = False
    attempts:    list[PipelineResult] = field(default_factory=list)
```

`answer=None` signals "I don't know." The endpoint checks `gave_up` or `halted`
and formats an honest response instead of forcing a hallucinated answer.

---

## Primitive Operations

| Operation | Signature                                       | Where                       |
|-----------|-------------------------------------------------|-----------------------------|
| Configure | `query → metadata`                              | configure phase             |
| Expand    | `(query, chunks?) → [RetrievalRequest]`         | initial_expand, loop_expand |
| Check     | `(query, chunks) → bool`                        | loop_check                  |
| Fetch     | `[RetrievalRequest] → chunks`                   | fetch_all                   |
| Filter    | `chunks → chunks`                               | loop_filter, finalize       |
| Halt      | `chunks=[] → PipelineResult(answer=None)`       | pipeline early exit         |
| Generate  | `chunks → answer`                               | generate phase              |
| Judge     | `answer → verdict`                              | evaluate phase              |
| Replan    | `(result, history, plan) → BrainRound \| None`  | recovery strategies         |

No step crosses these boundaries.

---

## Step Inventory

### Configure (metadata, no queries)

| Step           | LLM/heuristic | Produces                       |
|----------------|---------------|--------------------------------|
| PlanLLMStep    | LLM           | retrieval_plan, retrieval_mode |
| DetectLangStep | LLM           | lang, is_factoid               |

Future: QueryClassifyStep, RelevanceGateStep, AmbiguityCheckStep.

### Expand — initial (queries, no chunk dependency)

| Step              | LLM/heuristic | Produces                              |
|-------------------|---------------|---------------------------------------|
| HyDEStep          | LLM           | hypothetical passage as query         |
| FactQueryStep     | LLM           | sub-queries from query decomposition  |
| KeywordStep       | LLM           | keyword queries                       |
| QueryVariantsStep | heuristic     | reformulated queries                  |
| BM25AnchorStep    | heuristic     | keyword query with plan_override=fast |

### Loop check (should we keep retrieving?)

| Step             | LLM/heuristic | Stops when                                 |
|------------------|---------------|--------------------------------------------|
| QualityCheckStep | heuristic     | enough unique sources, scores above threshold |
| BudgetCheckStep  | heuristic     | LLM budget too low for another round       |

Future: LLMSufficiencyCheck.

### Expand — loop (queries, need chunks)

| Step                 | LLM/heuristic | Produces                                |
|----------------------|---------------|-----------------------------------------|
| TwoPassStep          | heuristic     | follow-up query from chunk hint terms   |
| FactoidExpandStep    | heuristic     | re-retrieval request with fast preset   |
| RetrievalReflectStep | LLM           | targeted queries + config overrides (planned) |

### Filter (chunks → fewer chunks)

| Step                 | LLM/heuristic | What it does                    |
|----------------------|---------------|---------------------------------|
| ChunkRelevanceFilter | heuristic     | score-based relevance threshold (planned) |
| LLMChunkFilter       | LLM           | per-chunk relevance check (planned) |

### Finalize (end-of-retrieval processing)

| Step              | LLM/heuristic | What it does                        |
|-------------------|---------------|-------------------------------------|
| StitchStep        | heuristic     | merge adjacent chunks from same doc |
| ChunkAnnotateStep | LLM           | annotate each chunk with claims (planned) |
| SufficiencyCheck  | LLM           | enough context to answer? (planned) |

### Generate (produce answer)

| Step         | LLM/heuristic | What it does                    |
|--------------|---------------|---------------------------------|
| GenerateStep | LLM           | draft answer from chunks + query |

Future: DraftStep, GroundingCheckStep, ToneCheckStep, ReviseStep.

### Evaluate (judge answer, verdict only)

| Step               | LLM/heuristic | What it does                  |
|--------------------|---------------|-------------------------------|
| ReflectStep        | LLM           | is the answer complete?       |
| AssessStep         | LLM           | what's missing?               |
| GroundingCheckStep | heuristic     | is answer grounded in chunks? |

Evaluate never retrieves. Evaluate never generates.

---

## Refactoring Plan

### Phase 1: Fix dishonest steps

Six functions currently do retrieval or generation outside their phase.

**expand.py — make query-only:**

| # | Function         | Keep                    | Delete                          | Return type change              |
|---|------------------|-------------------------|---------------------------------|---------------------------------|
| 1 | `_fact_queries`  | LLM → sub-queries       | `_safe_retrieve` per sub-query  | `list[ChunkResult]` → `list[str]` |
| 2 | `_keywords`      | LLM → keywords          | `_safe_retrieve` per keyword    | `list[ChunkResult]` → `list[str]` |
| 3 | `_safe_retrieve` | —                       | entire function                 | (deleted)                       |

After: expand.py has no `http_client`, no `settings`, no retrieval imports.

**evaluate.py — evaluate only:**

| # | Function                 | Keep                                        | Delete                          |
|---|--------------------------|---------------------------------------------|---------------------------------|
| 4 | `_supplemental_retrieve` | —                                           | entire function                 |
| 5 | `_factoid_retry`         | `answer_is_grounded` → new GroundingCheckStep | rest (retrieve + generate)     |

After: evaluate.py has no `http_client`, no `settings`, no retrieval imports.

**enrich.py — dissolve:**

| #  | Branch           | Extract to                                       | Delete              |
|----|------------------|--------------------------------------------------|---------------------|
| 6a | TwoPassStep      | hint terms + follow-up query → LoopExpandStep    | `_safe_retrieve`    |
| 6b | BM25AnchorStep   | keyword query → ExpandStep (with plan_override)  | `_safe_retrieve`    |
| 6c | FactoidExpandStep| query reuse → LoopExpandStep (with plan_override)| `_safe_retrieve`    |
| 6d | StitchStep       | → FinalizeStep in retrieval pipeline             | —                   |

After: enrich.py deleted.

### Phase 1b: Move honest functions

| Function                              | From      | To                | Why                  |
|---------------------------------------|-----------|-------------------|----------------------|
| `_plan_retrieval`, `_plan_retrieval_llm` | expand.py | steps/configure.py | metadata, not queries |
| `_detect_lang`                        | expand.py | steps/configure.py | metadata, not queries |
| StitchStep logic                      | enrich.py | retrieval finalize | chunk transform       |

### Phase 1c: Structural wiring

1. Create `engine/retrieval.py` — `run_retrieval` + `fetch_all`.
2. Create `engine/recovery.py` — `run_with_recovery` + `RetryStrategy` + `FallbackPreset`.
3. Fix type holes: `retrieval_plan: Any` → `ExecutionPlan | None`, `brain_plan: Any` → `BrainRound`.
4. Extract shared models to `rag_types` package (ChunkResult, ExecutionPlan, etc.).

### Phase 2: Fill gaps

5. Real streaming — generate returns `AsyncIterator[str]`.
6. Prometheus metrics — per-phase latency, LLM call counts, retry rates.
7. Mood traces — restore MEME_GRUMPS.

### Phase 3: Add narrow steps (ongoing)

8. One step at a time: define model, add to union, add case, write prompt.

   Priority: RetrievalReflectStep, ChunkRelevanceFilter, SufficiencyCheck,
   DraftStep + ReviseStep.

### Phase 4: Higher layers (when needed)

9. PlanAgent — unified initial planning + replanning.
10. Orchestration — multi-plan, query decomposition, parallel strategies.
11. Ingestion-time enrichment in doc_processor_v2.

---

## File Structure After Refactoring

```
engine/
  pipeline.py       ~30 lines   configure → retrieve → generate → evaluate
  retrieval.py      ~70 lines   expand → fetch → loop(check → expand → fetch → filter) → finalize
  recovery.py       ~60 lines   run_with_recovery + RetryStrategy + FallbackPreset
  budget.py         (unchanged)

steps/
  configure.py      ~80 lines   _plan_retrieval, _plan_retrieval_llm, _detect_lang
  expand.py         ~150 lines  HyDE, FactQuery, Keywords, QueryVariants, BM25Anchor,
                                TwoPass, FactoidExpand (all query-only)
  generate.py       ~60 lines   GenerateStep
  evaluate.py       ~80 lines   Reflect, Assess, GroundingCheck

models/
  steps.py          ConfigStep, ExpandStep, LoopCheckStep, LoopExpandStep,
                    FilterStep, FinalizeStep, GenerateStep, EvalStep
  plan.py           BrainRound, RetrievalPlan, RetrievalRequest, RetrievalResult,
                    ConfigMeta, Verdict, PipelineResult

endpoints.py        calls run_with_recovery
brain_presets.py    updated to new BrainRound shape
```

~460 lines total, down from ~900. enrich.py deleted.

---

## Design Principles

1. **Steps are honest.** One primitive operation per step. No step crosses phase
   boundaries.

2. **Plans are data.** BrainRound and RetrievalPlan can be inspected, serialized,
   logged, diffed, and generated by an LLM.

3. **Typed contracts.** Configure → metadata. Expand → queries. Filter → fewer chunks.
   Generate → answer. Evaluate → verdict. Types enforce boundaries.

4. **Retrieval is a black box.** The generator says "give me chunks" and gets chunks.

5. **Recovery is above the pipeline.** The pipeline is a pure single-pass function.
   Retry strategies live in the recovery layer. When nobody can help, recovery says
   "I don't know."

6. **Add bricks, not framework.** New capabilities = new steps in existing phases.
   Each new step: model + case + prompt. Architecture doesn't change.
