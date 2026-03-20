The 20-40 LLM calls claim
That's the interesting number. Your system in its full aggressive preset uses roughly 6-8 LLM calls: plan, detect_lang, HyDE, maybe fact_queries, maybe keywords, generate, assess, maybe reflect. RunLLM claims 3-5x more.

Where do those extra calls go? Reading between the lines of their description, each of these is probably its own narrowly-scoped LLM call:

Is this question relevant to this product?
Is this question answerable from the available data?
What type of question is this? (troubleshooting, how-to, conceptual, comparison)
What data sources are most likely to contain the answer?
Is this query ambiguous? If so, what are the interpretations?
For each retrieved chunk: is this chunk actually relevant to the question?
For each retrieved chunk: what specific claim does this chunk support?
Is there enough context to answer confidently?
Draft answer
Does the answer actually address the question asked?
Does every claim in the answer have a supporting chunk?
Is the answer consistent with the product's terminology?
Is the answer actionable (not just "it depends")?
Is the tone appropriate for technical support?
Are there caveats or version-specific notes to add?
Final answer refinement
That's ~16 calls and I'm probably still under-counting. The point isn't the exact number — it's the philosophy: each call has one job and a guardrail on its output. No call is "generate an answer from this context" with 47 implicit instructions. Each call does one narrow thing, and the output is validated before the next call runs.




# Future Generator Steps

Brain-layer improvements for generator_v2. Prerequisites and dependencies noted inline.

## Adaptive Replanning (assess -> replan -> retry)

Instead of using a static `retry_round()` preset when the assessor flags an incomplete answer, feed the `Verdict` (missing_terms, reason, original chunks summary) back into the planning LLM and let it design a targeted recovery `BrainRound`.

**Why it matters.** The current retry is blind -- it always does HyDE + keywords + fact_queries at 2x top_k regardless of *why* the answer failed. An LLM-generated retry plan can target the gap: missing specifics -> factoid expansion; wrong topic coverage -> keyword focus on the missing subtopic; vague answer -> higher top_k + two-pass.

**Design (moderate control).** The replanning LLM returns structured parameters, not raw step lists:

```json
{
  "use_hyde": true,
  "top_k": 20,
  "use_fact_queries": true,
  "max_fact_queries": 3,
  "use_stitch": true,
  "focus_terms": ["the specific thing the assessor said was missing"],
  "reason": "answer lacked specifics about X"
}
```

Mapped to a `BrainRound` via the existing `agent()` builder. Pydantic validates the structure.

**Where it lives.** Endpoint-level, not pipeline-level. The pipeline stays a pure single-pass function. The endpoint owns the retry loop and the replanning decision:

```python
for attempt in range(max_retries):
    result = await run_pipeline(plan, ...)
    if not result.needs_retry:
        break
    plan = await replan(
        original_query=query,
        verdict=result,
        attempt=attempt,
        llm=llm, model=model, settings=settings,
    )
```

**Cost.** One extra LLM call per retry (cheap structured-output call). Replaces wasted blind retrieval with targeted retrieval.

**Prerequisite.** Refine single-round quality first -- better expand, enrich, and evaluate steps. Adaptive replanning amplifies a good pipeline; it can't rescue a weak one.

**Estimated scope.** ~60 lines: `replan()` function, a prompt template, endpoint loop update.

## Critique-Revise Step

A new `EvalStep` that catches *generation* failures (hallucinated facts, missed caveats in context) without re-retrieving. Complementary to retry, which catches *retrieval* failures.

1. Critique: "Given these chunks and this answer, what's wrong?"
2. Revise: "Rewrite the answer using the same chunks, fixing the issues."

Returns a `Verdict` with the revised answer. No `needs_retry`, no re-retrieval. Fits naturally into the existing evaluate phase -- `pipeline.py` already picks up `verdict.answer` as a replacement.

**Cost.** 2 LLM calls (critique + revise). Cheap relative to a full retry cycle.

**Estimated scope.** ~50 lines: new step type in `models/steps.py`, handler in `steps/evaluate.py`, prompt template.

## Iterative Assessment

Add `AssessStep()` to `retry_round()`'s evaluate list so retried answers get assessed too. Currently the retry is fire-and-forget -- if it also fails, we don't know.

Combined with adaptive replanning, this enables: assess -> replan -> retry -> assess -> (give up or try once more).

**Estimated scope.** ~10 lines: add step to preset, add `max_retries` loop in endpoints.

## Summary

| Feature | Catches | Cost | Scope | Depends on |
|---|---|---|---|---|
| Critique-revise | Generation failures | 2 LLM calls | ~50 lines | Nothing |
| Iterative assessment | Silent retry failures | 1 LLM call/retry | ~10 lines | Nothing |
| Adaptive replanning | Wrong retry strategy | 1 LLM call/retry | ~60 lines | Good single-round quality |

## Agent Mood / Personality Traces

The old agent-search emitted a random "mood" trace at the start of each request -- a grumpy one-liner from a hardcoded list (`MEME_GRUMPS`):

```
"Sigh. Fine. I will do science."
"This better be worth the tokens."
"I am not mad. I am just disappointed in entropy."
"Okay, okay, I will carry this search. Again."
"One more query and I start charging by the sigh."
```

Emitted as `{"type": "trace", "kind": "mood", "content": "..."}` before the plan step. Zero LLM cost, purely cosmetic -- but the frontend displayed it and users noticed its absence.

**Where it lives.** Either in `endpoints.py` (emit a mood trace before calling `run_pipeline`) or in the pipeline itself as a trivial first trace entry. Endpoint-level is cleaner since it's presentation, not computation.

**Estimated scope.** ~10 lines: the list, a `random.choice`, and one trace dict appended to the stream.

## Summary

| Feature | Catches | Cost | Scope | Depends on |
|---|---|---|---|---|
| Critique-revise | Generation failures | 2 LLM calls | ~50 lines | Nothing |
| Iterative assessment | Silent retry failures | 1 LLM call/retry | ~10 lines | Nothing |
| Adaptive replanning | Wrong retry strategy | 1 LLM call/retry | ~60 lines | Good single-round quality |
| Agent mood traces | UX regression | 0 (no LLM) | ~10 lines | Nothing |

Recommended order: refine single-round quality -> critique-revise -> iterative assessment -> adaptive replanning. Agent mood can go in anytime.

---

# Design Context for Later Stages

Notes from the architecture review that didn't fit in ARCHITECTURE_v2.md but will
be useful when working on phases 3-4.

## What agent-search did (and why we changed it)

The old monolithic `_run_agent()` in `RAGfun/agent-search/app/main.py` had:

- **One hardcoded retry.** If `assessment["incomplete"]` was true: force hybrid mode,
  double top_k (`max(12, top_k * 2)`), rerun HyDE + keywords + fact_queries. No loop.
- **No "I don't know."** If retry also failed, the original (possibly bad) answer was
  returned. The system never admitted it couldn't answer.
- **Hardcoded plan fallback.** If plan LLM returned garbage: `hybrid, top_k=10, rerank=True`.
  Same when LLM budget was exhausted mid-pipeline.
- **No preset switching on failure.** Presets (minimal/conservative/aggressive) chosen
  at request time only. `use_retry=True` only in aggressive.
- **quality_is_poor() as inline heuristic.** Checked min_hits, min_score, partial/degraded
  flags. Triggered extra fact queries before answering — not a retry, just extra retrieval
  shoved into the middle of the flow.
- **Fake streaming.** Generated the full answer, then emitted tokens one by one.
- **MEME_GRUMPS.** Random grumpy personality trace. Users noticed when it disappeared.

Key lesson: the old system's retry was essentially a hardcoded `BroaderRetrieval` strategy
(the one we discarded as redundant with the retrieval loop). It doubled knobs mechanically
without understanding why the answer failed. The new `run_with_recovery` with
`RetryStrategy` protocol is strictly better.

## Pydantic discriminated unions for step types

We chose Pydantic discriminated unions (`Annotated[Union[...], Field(discriminator="kind")]`)
over plain dataclasses for step types because:

1. **Serialization.** A `BrainRound` can be JSON-serialized and deserialized. This means
   plans can be logged, diffed, stored in a database, and — critically — generated by an
   LLM via structured output (`response_model=BrainRound`).
2. **Validation.** Pydantic validates field ranges, required fields, enum values. A step
   with `top_k=-1` is caught at construction time.
3. **JSON Schema.** Pydantic generates JSON schemas that can be passed to LLMs for
   structured output. The LLM sees the schema and produces valid plans by construction.
4. **Pattern matching.** `match step:` with `case HyDEStep(): ...` works naturally.

Plain dataclasses are used for internal data (`RetrievalRequest`, `PipelineResult`) that
doesn't need serialization or LLM generation. Use Pydantic for API boundaries and plan
types; dataclasses for internal plumbing.

## Why PipelineState was rejected

Early design proposed threading all intermediate data through a single god object:

```python
@dataclass
class PipelineState:
    query: str
    queries: list[str]
    chunks: list[ChunkResult]
    answer: str | None
    lang: str
    is_factoid: bool
    retrieval_plan: ExecutionPlan | None
    verdict: Verdict | None
    traces: list[dict]
    budget: BudgetCounter
```

Problems:
- Every phase can read and write everything — no boundaries.
- Adding a field to one phase requires changing the shared state type.
- Testing a single phase requires constructing the entire state.
- The type says nothing about what each phase actually needs or produces.

Fix: explicit function signatures for each phase. `configure(query=...) → ConfigMeta`.
`generate(chunks=..., lang=...) → str`. Each function declares exactly what it needs.
Local variables in `run_pipeline` handle data flow. No shared mutable state.

## Pipeline as a fold over phases (Haskell perspective)

The generator pipeline is a left fold:

```
result = foldl (flip applyPhase) initialState [configure, retrieve, generate, evaluate]
```

Each phase takes the accumulated result and produces more data. The `BrainRound` is the
program; `run_pipeline` is the interpreter. This is essentially a Free Monad pattern
where the DSL operations are the primitive operations table (Configure, Expand, Check,
Fetch, Filter, Generate, Judge, Replan) and the interpreter runs them.

The retrieval pipeline is a different fold — a fold with a loop (catamorphism with
recursion), which is why it has its own interpreter (`run_retrieval`) rather than being
inlined into `run_pipeline`.

Recovery is a fold over attempts — unfold until acceptable or give up.

## Builder DSL (sugar, not architecture)

To make plan construction feel more "framework-like" without adding a registry:

```python
plan = (
    RetrievalPlanBuilder(preset="aggressive")
    .expand(HyDEStep(), FactQueryStep(max=3), KeywordStep())
    .check(QualityCheckStep(min_sources=3))
    .loop_with(TwoPassStep(), FactoidExpandStep())
    .filter(ChunkRelevanceFilter(threshold=0.15))
    .finalize(StitchStep(), SufficiencyCheck())
    .max_rounds(3)
    .build()
)
```

Same typed slots underneath. Same interpreter. Just a friendlier surface for assembling
plans. Build this after the refactoring, not before — it's presentation, not architecture.

## Toward generic pipelines (long-term)

Three concepts needed to move from the current "slotted topology" to a LlamaIndex/n8n-style
generic pipeline:

1. **Typed ports.** Each step declares input/output types as data, not just as function
   signatures. This makes wiring automatic and enables visualization.

2. **DAG scheduling.** A generic interpreter that topologically sorts nodes and runs them,
   handling fan-out (one output feeds multiple inputs) and fan-in (multiple outputs merge
   into one input).

3. **Nested pipelines.** A pipeline is itself a node with typed ports. The retrieval
   pipeline inside the generator pipeline is already this — just hand-wired.

The current architecture doesn't fight this evolution. `BrainRound` is already "pipeline
as data." The step unions are already "typed nodes." The gap is: the interpreter
(`run_pipeline`, `run_retrieval`) is hand-written for a fixed topology instead of being
generic. Closing that gap requires making the topology itself data — not just the steps.

Not needed now. The fixed topology handles the domain well. But when the number of
pipeline variants grows or when visual editing is wanted, this is the path.

## Per-step fine-tuned models (RunLLM insight)

RunLLM uses 20-40 narrowly-scoped LLM calls per answer. Each call has one job and a
guardrail. The key insight: they can use smaller, cheaper, fine-tuned models for each
narrow task instead of one large model doing everything.

In our architecture, this means each step in a `BrainRound` could specify its own model:

```python
class HyDEStep(BaseModel):
    kind: Literal["hyde"] = "hyde"
    num_passages: int = 1
    model_override: str | None = None    # use a fine-tuned model for HyDE
```

Steps that are well-defined and high-volume (language detection, chunk relevance filtering,
grounding checks) are candidates for fine-tuning. Steps that require general reasoning
(planning, generation, assessment) stay on the full model.

Prerequisite: enough logged data to fine-tune on. The trace system already logs every
LLM call — that's the training data.

## Ingestion-time enrichment (separate workstream)

RunLLM annotates documents at ingestion time with:
- Topics and categories
- Potential questions the document answers
- Key entities and relationships
- Summary at multiple granularities

This is a doc_processor_v2 project (`RAGfun/RAGcircle/doc_processor_v2`), not a generator
project. It makes retrieval better (reducing the need for query-time heuristics like
TwoPass and FactoidExpand) but the pipeline architecture doesn't change.

The connection: if ingestion-time enrichment is good enough, many of the loop_expand
steps become unnecessary. The pipeline simplifies naturally — not by removing slots,
but by leaving them empty.

## Multi-plan orchestration (Layer 4)

When a query decomposes into multiple sub-tasks:

```
"Compare authentication methods in v2 vs v3"
→ sub-task 1: "authentication in v2"
→ sub-task 2: "authentication in v3"
→ merge: compare the two answers
```

Each sub-task gets its own `BrainRound` and runs through `run_with_recovery`
independently. A merge step combines the results.

This is Layer 4 — above recovery. The orchestrator decomposes queries, runs parallel
pipelines, and synthesizes results. It doesn't change the pipeline architecture; it
runs multiple pipelines.

Design sketch:

```python
class Orchestrator:
    async def run(self, query, ...) -> OrchestratorResult:
        tasks = await self.decompose(query)      # LLM: split into sub-tasks
        if len(tasks) == 1:
            return await run_with_recovery(tasks[0].plan, ...)
        results = await asyncio.gather(*[
            run_with_recovery(task.plan, ...) for task in tasks
        ])
        return await self.merge(query, results)  # LLM: synthesize
```

Not needed until query complexity demands it. The single-pipeline path handles most
queries.
