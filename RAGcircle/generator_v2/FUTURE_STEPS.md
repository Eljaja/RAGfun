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
