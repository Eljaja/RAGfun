"""Plan-driven brain engine. Walks a BrainPlan dispatching to domain modules."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from assessment import assess_completeness, detect_language, quality_is_poor, reflect
from config import Settings
from context import (
    build_context,
    extract_source_details,
    history_as_messages,
    history_summary,
    merge_chunks,
    stitch_segments,
)
from llm import LLMClient
from models import (
    AssessStep,
    BM25AnchorStep,
    BrainPlan,
    BrainRetrieveStep,
    ChunkResult,
    DetectLangStep,
    ExecutionPlan,
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
from plan_builder import from_llm_plan, from_preset
from planning import BudgetCounter, plan_retrieval
from prompts import ANSWER_SYSTEM, ANSWER_SYSTEM_WITH_TOOLS, ANSWER_USER
from query_expansion import fact_queries, hyde, keyword_queries
from query_variants import (
    answer_is_grounded,
    extract_hint_terms,
    is_factoid_question,
    keyword_query,
    query_variants,
    unique_source_count,
)
from retrieval_client import retrieve
from tools import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Mutable state carried across rounds."""

    project_id: str
    query: str
    history: list[dict[str, str]]
    history_text: str
    lang: str = "English"
    search_queries: list[str] = field(default_factory=list)
    retrieval_plan: ExecutionPlan | None = None
    retrieval_mode: str = "hybrid"
    chunks: list[ChunkResult] = field(default_factory=list)
    answer: str = ""
    budget: BudgetCounter = field(default_factory=lambda: BudgetCounter(12))
    include_sources: bool = True
    round_index: int = 0
    is_factoid: bool = False
    source_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.search_queries:
            self.search_queries = [self.query]

    @property
    def search_query(self) -> str:
        return self.search_queries[0] if self.search_queries else self.query

    @search_query.setter
    def search_query(self, value: str) -> None:
        if self.search_queries:
            self.search_queries[0] = value
        else:
            self.search_queries = [value]


async def _safe_retrieve(
    http_client: httpx.AsyncClient,
    retrieval_url: str,
    project_id: str,
    query: str,
    plan: ExecutionPlan,
) -> list[ChunkResult]:
    try:
        return await retrieve(
            http_client, retrieval_url,
            project_id=project_id, query=query, plan=plan,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


# ── Step dispatchers ─────────────────────────────────────


async def _exec_plan_llm(
    step: PlanLLMStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "trace", "kind": "tool", "name": "llm.plan", "payload": {"query": ctx.query}}

    if not ctx.budget.can_call():
        ctx.retrieval_plan = from_preset("hybrid", top_k=10, rerank=True)
        return

    try:
        plan_dict = await asyncio.wait_for(
            plan_retrieval(llm, model, ctx.query, history_text=ctx.history_text),
            timeout=settings.agent_llm_timeout + 10,
        )
    except Exception as e:
        yield {"type": "error", "error": f"Plan error: {e!s}"}
        return

    ctx.retrieval_mode = str(plan_dict.get("retrieval_mode") or "hybrid")
    reason = str(plan_dict.get("reason") or "")
    ctx.retrieval_plan = from_llm_plan(
        plan_dict,
        top_k_min=settings.agent_top_k_min,
        top_k_max=settings.agent_top_k_max,
    )

    yield {
        "type": "trace", "kind": "thought", "label": "Plan",
        "content": f"mode={ctx.retrieval_mode}. Reason: {reason}",
    }


async def _exec_detect_lang(
    step: DetectLangStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
) -> AsyncIterator[dict[str, Any]]:
    if ctx.budget.can_call():
        ctx.lang = await detect_language(llm, model, ctx.query)
    ctx.is_factoid = is_factoid_question(ctx.query)
    return
    yield  # noqa: make this an async generator


async def _exec_hyde(
    step: HyDEStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.budget.can_call():
        return

    yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"query": ctx.query}}
    try:
        passage = await hyde(llm, model, ctx.query, lang=ctx.lang)
        if passage.strip():
            ctx.search_query = passage
    except Exception:
        logger.warning("HyDE failed, using original query")


async def _exec_fact_queries(
    step: FactQueryStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.budget.can_call():
        return

    yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"query": ctx.query}}

    try:
        sub_queries = await fact_queries(llm, model, ctx.query, history_text=ctx.history_text)
        sub_queries = sub_queries[:step.max_queries]
    except Exception:
        return

    if not sub_queries or ctx.retrieval_plan is None:
        return

    fact_plan = ctx.retrieval_plan
    fact_results = await asyncio.gather(*(
        _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, fq, fact_plan)
        for fq in sub_queries
    ))
    ctx.chunks = merge_chunks([ctx.chunks, *fact_results])


async def _exec_keywords(
    step: KeywordStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.budget.can_call():
        return

    yield {"type": "trace", "kind": "tool", "name": "llm.keywords", "payload": {"query": ctx.query}}

    try:
        kw = await keyword_queries(llm, model, ctx.query, history_text=ctx.history_text)
    except Exception:
        return

    if not kw or ctx.retrieval_plan is None:
        return

    kw_plan = ctx.retrieval_plan
    kw_results = await asyncio.gather(*(
        _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, k, kw_plan)
        for k in kw[:4]
    ))
    ctx.chunks = merge_chunks([ctx.chunks, *kw_results])
    return
    yield


async def _exec_query_variants(
    step: QueryVariantsStep,
    ctx: RunContext,
) -> AsyncIterator[dict[str, Any]]:
    """Zero-cost heuristic: generate keyword, phrase, year variants."""
    variants = query_variants(ctx.query)
    if len(variants) > 1:
        ctx.search_queries = variants
        yield {
            "type": "trace", "kind": "action", "label": "QueryVariants",
            "content": f"Generated {len(variants)} variants",
        }
    return
    yield


async def _exec_retrieve(
    step: BrainRetrieveStep,
    ctx: RunContext,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Retrieve phase -- fans out if multiple search_queries exist."""
    plan = ctx.retrieval_plan or from_preset(step.preset, top_k=step.top_k, rerank=step.rerank)

    yield {
        "type": "trace", "kind": "action", "label": "Retrieving",
        "content": f"queries={len(ctx.search_queries)}, first={ctx.search_query[:80]}...",
    }

    if len(ctx.search_queries) <= 1:
        new_chunks = await _safe_retrieve(
            http_client, settings.retrieval_url, ctx.project_id, ctx.search_query, plan,
        )
    else:
        results = await asyncio.gather(*(
            _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, sq, plan)
            for sq in ctx.search_queries
        ))
        new_chunks = merge_chunks(list(results))

    if ctx.chunks:
        ctx.chunks = merge_chunks([ctx.chunks, new_chunks])
    else:
        ctx.chunks = new_chunks

    yield {
        "type": "retrieval",
        "mode": ctx.retrieval_mode,
        "partial": False,
        "degraded": [],
        "context": [{"source_id": c.source_id, "text": c.text[:200], "score": c.score} for c in ctx.chunks[:10]],
    }


async def _exec_quality_check(
    step: QualityCheckStep,
    ctx: RunContext,
) -> AsyncIterator[dict[str, Any]]:
    poor = quality_is_poor(ctx.chunks, min_hits=step.min_hits, min_score=step.min_score)
    if poor:
        yield {"type": "trace", "kind": "thought", "label": "Quality", "content": "Retrieval quality is poor"}
    return
    yield


async def _exec_two_pass(
    step: TwoPassStep,
    ctx: RunContext,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Conditional second retrieval pass when unique source count is low."""
    n_unique = unique_source_count(ctx.chunks)
    if n_unique >= step.min_unique_sources:
        return

    yield {
        "type": "trace", "kind": "action", "label": "TwoPass",
        "content": f"Only {n_unique} unique sources, generating follow-up query",
    }

    hints = extract_hint_terms(ctx.chunks, max_terms=3)
    if not hints:
        return

    follow_up = f"{ctx.query} {' '.join(hints)}"
    plan = ctx.retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
    extra = await _safe_retrieve(
        http_client, settings.retrieval_url, ctx.project_id, follow_up, plan,
    )
    if extra:
        ctx.chunks = merge_chunks([ctx.chunks, extra])


async def _exec_bm25_anchor(
    step: BM25AnchorStep,
    ctx: RunContext,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """BM25-only keyword anchor search, merged with existing results."""
    kw = keyword_query(ctx.query)
    if not kw:
        return

    yield {
        "type": "trace", "kind": "action", "label": "BM25Anchor",
        "content": f"kw={kw[:60]}",
    }

    bm25_plan = from_preset("fast", top_k=step.top_k, rerank=False)
    bm25_hits = await _safe_retrieve(
        http_client, settings.retrieval_url, ctx.project_id, kw, bm25_plan,
    )
    if bm25_hits:
        ctx.chunks = merge_chunks([ctx.chunks, bm25_hits])


async def _exec_factoid_expand(
    step: FactoidExpandStep,
    ctx: RunContext,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Pre-generation factoid expansion: within-doc BM25 for factoid questions."""
    if not ctx.is_factoid or not ctx.chunks:
        return

    top_sources = list({c.source_id for c in ctx.chunks[:3]})
    if not top_sources:
        return

    yield {
        "type": "trace", "kind": "action", "label": "FactoidExpand",
        "content": f"Expanding within {len(top_sources)} top sources",
    }

    expand_plan = from_preset("fast", top_k=5, rerank=False)
    expand_results = await asyncio.gather(*(
        _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, ctx.query, expand_plan)
        for _ in top_sources
    ))
    ctx.chunks = merge_chunks([ctx.chunks, *expand_results])


async def _exec_stitch(
    step: StitchStep,
    ctx: RunContext,
) -> AsyncIterator[dict[str, Any]]:
    """Post-retrieval segment stitching."""
    before = len(ctx.chunks)
    ctx.chunks = stitch_segments(ctx.chunks, max_per_segment=step.max_per_segment)
    after = len(ctx.chunks)
    if before != after:
        yield {
            "type": "trace", "kind": "action", "label": "Stitch",
            "content": f"Stitched {before} chunks into {after} segments",
        }
    return
    yield


async def _exec_generate(
    step: GenerateStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.chunks:
        yield {"type": "error", "error": "No chunks available for generation"}
        return

    context_text = build_context(
        ctx.chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=ctx.source_meta,
    )

    if step.use_tools:
        system_prompt = ANSWER_SYSTEM_WITH_TOOLS.format(lang=ctx.lang)
    else:
        system_prompt = ANSWER_SYSTEM.format(lang=ctx.lang)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(ctx.history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=ctx.query, context=context_text,
    )})

    ctx.answer = ""
    if step.use_tools:
        ctx.answer = await llm.complete_with_tools(
            model, messages, TOOL_DEFINITIONS, temperature=step.temperature,
        )
        yield {"type": "token", "content": ctx.answer}
    elif step.stream:
        async for token in llm.stream(model, messages, temperature=step.temperature):
            ctx.answer += token
            yield {"type": "token", "content": token}
    else:
        ctx.answer = await llm.complete(model, messages, temperature=step.temperature)
        yield {"type": "token", "content": ctx.answer}


async def _exec_reflect(
    step: ReflectStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.budget.can_call() or not ctx.answer:
        return

    yield {"type": "trace", "kind": "tool", "name": "llm.reflect", "payload": {}}

    context_text = build_context(
        ctx.chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=ctx.source_meta,
    )
    try:
        result = await reflect(llm, model, ctx.query, context_text, ctx.answer)
        if not result.complete and result.requery:
            ctx.search_query = result.requery
            yield {
                "type": "trace", "kind": "thought", "label": "Reflect",
                "content": f"Incomplete. Requery: {result.requery}",
            }
        elif result.complete:
            yield {"type": "trace", "kind": "thought", "label": "Reflect", "content": "Answer is complete"}
    except Exception:
        logger.warning("Reflection failed, continuing")
    return
    yield


async def _exec_assess(
    step: AssessStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    if not ctx.budget.can_call() or not ctx.answer:
        return

    yield {"type": "trace", "kind": "tool", "name": "llm.assess", "payload": {}}

    assessment = await assess_completeness(
        llm, model, ctx.query, ctx.answer, history_text=ctx.history_text,
    )

    if not assessment.incomplete:
        return

    yield {
        "type": "trace", "kind": "thought", "label": "Assess",
        "content": f"Incomplete: {assessment.reason}. Missing: {assessment.missing_terms}",
    }

    retry_queries = list(assessment.missing_terms)
    if ctx.budget.can_call():
        try:
            kw = await keyword_queries(llm, model, ctx.query, history_text=ctx.history_text)
            retry_queries.extend(kw)
        except Exception:
            pass

    if retry_queries and ctx.retrieval_plan is not None:
        retry_plan = from_preset("thorough", top_k=10, rerank=True)
        retry_results = await asyncio.gather(*(
            _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, rq, retry_plan)
            for rq in retry_queries[:4]
        ))
        ctx.chunks = merge_chunks([ctx.chunks, *retry_results])

        context_text = build_context(
            ctx.chunks,
            max_chars=settings.max_context_chars,
            max_chunk_chars=settings.max_chunk_chars,
            source_meta=ctx.source_meta,
        )
        system_prompt = ANSWER_SYSTEM.format(lang=ctx.lang)
        retry_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        retry_messages.extend(history_as_messages(ctx.history))
        retry_messages.append({"role": "user", "content": ANSWER_USER.format(
            history="", query=ctx.query, context=context_text,
        )})
        ctx.answer = ""
        async for token in llm.stream(model, retry_messages):
            ctx.answer += token
            yield {"type": "token", "content": token}


async def _exec_factoid_retry(
    step: FactoidRetryStep,
    ctx: RunContext,
    llm: LLMClient,
    model: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Post-generation grounding check for factoid answers.

    If the answer is not grounded in context, expand within-doc,
    re-generate, and conditionally swap.
    """
    if not ctx.is_factoid or not ctx.answer or not ctx.chunks:
        return

    context_text = build_context(
        ctx.chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=ctx.source_meta,
    )

    if answer_is_grounded(answer=ctx.answer, context_text=context_text):
        return

    yield {
        "type": "trace", "kind": "thought", "label": "FactoidRetry",
        "content": "Answer not grounded in context, attempting recovery",
    }

    expand_plan = from_preset("fast", top_k=5, rerank=False)
    top_sources = list({c.source_id for c in ctx.chunks[:3]})
    if top_sources:
        expand_results = await asyncio.gather(*(
            _safe_retrieve(http_client, settings.retrieval_url, ctx.project_id, ctx.query, expand_plan)
            for _ in top_sources
        ))
        ctx.chunks = merge_chunks([ctx.chunks, *expand_results])

    new_context = build_context(
        ctx.chunks,
        max_chars=settings.max_context_chars,
        max_chunk_chars=settings.max_chunk_chars,
        source_meta=ctx.source_meta,
    )
    system_prompt = ANSWER_SYSTEM.format(lang=ctx.lang)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(ctx.history))
    messages.append({"role": "user", "content": ANSWER_USER.format(
        history="", query=ctx.query, context=new_context,
    )})

    new_answer = ""
    async for token in llm.stream(model, messages):
        new_answer += token

    if answer_is_grounded(answer=new_answer, context_text=new_context):
        ctx.answer = new_answer
        yield {"type": "token", "content": ctx.answer}
        yield {
            "type": "trace", "kind": "thought", "label": "FactoidRetry",
            "content": "Recovery succeeded, answer replaced",
        }
    else:
        yield {
            "type": "trace", "kind": "thought", "label": "FactoidRetry",
            "content": "Recovery did not improve grounding, keeping original",
        }


# ── Main executor ────────────────────────────────────────


async def execute(
    plan: BrainPlan,
    *,
    project_id: str,
    query: str,
    history: list[dict[str, str]] | None = None,
    include_sources: bool = True,
    llm: LLMClient,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Walk a BrainPlan, dispatch each step, yield SSE events."""
    model = settings.agent_llm_model or settings.llm_model
    raw_history = history or []

    ctx = RunContext(
        project_id=project_id,
        query=query,
        history=raw_history,
        history_text=history_summary(raw_history),
        budget=BudgetCounter(plan.max_llm_calls),
        include_sources=include_sources,
    )

    for round_idx, brain_round in enumerate(plan.rounds):
        ctx.round_index = round_idx

        yield {
            "type": "progress", "stage": "round",
            "content": f"Round {round_idx + 1}/{len(plan.rounds)}",
        }

        # ── Expand phase ─────────────────────────────────
        for step in brain_round.expand:
            match step:
                case PlanLLMStep():
                    async for ev in _exec_plan_llm(step, ctx, llm, model, settings):
                        yield ev
                        if ev.get("type") == "error":
                            return
                case DetectLangStep():
                    async for ev in _exec_detect_lang(step, ctx, llm, model):
                        yield ev
                case HyDEStep():
                    async for ev in _exec_hyde(step, ctx, llm, model):
                        yield ev
                case FactQueryStep():
                    async for ev in _exec_fact_queries(step, ctx, llm, model, http_client, settings):
                        yield ev
                case KeywordStep():
                    async for ev in _exec_keywords(step, ctx, llm, model, http_client, settings):
                        yield ev
                case QueryVariantsStep():
                    async for ev in _exec_query_variants(step, ctx):
                        yield ev

        # ── Retrieve phase ───────────────────────────────
        async for ev in _exec_retrieve(brain_round.retrieve, ctx, http_client, settings):
            yield ev

        if not ctx.chunks:
            yield {"type": "error", "error": "No results from retrieval service"}
            return

        # ── Post-retrieve phase ──────────────────────────
        for step in brain_round.post_retrieve:
            match step:
                case QualityCheckStep():
                    async for ev in _exec_quality_check(step, ctx):
                        yield ev
                case TwoPassStep():
                    async for ev in _exec_two_pass(step, ctx, http_client, settings):
                        yield ev
                case BM25AnchorStep():
                    async for ev in _exec_bm25_anchor(step, ctx, http_client, settings):
                        yield ev
                case FactoidExpandStep():
                    async for ev in _exec_factoid_expand(step, ctx, http_client, settings):
                        yield ev
                case StitchStep():
                    async for ev in _exec_stitch(step, ctx):
                        yield ev

        # ── Generate phase ───────────────────────────────
        async for ev in _exec_generate(brain_round.generate, ctx, llm, model, settings):
            yield ev
            if ev.get("type") == "error":
                return

        # ── Evaluate phase ───────────────────────────────
        for step in brain_round.evaluate:
            match step:
                case ReflectStep():
                    async for ev in _exec_reflect(step, ctx, llm, model, settings):
                        yield ev
                case AssessStep():
                    async for ev in _exec_assess(step, ctx, llm, model, http_client, settings):
                        yield ev
                case FactoidRetryStep():
                    async for ev in _exec_factoid_retry(step, ctx, llm, model, http_client, settings):
                        yield ev

    # ── Done ─────────────────────────────────────────────
    sources = extract_source_details(ctx.chunks, source_meta=ctx.source_meta) if ctx.include_sources else []

    yield {
        "type": "done",
        "answer": ctx.answer,
        "mode": ctx.retrieval_mode,
        "partial": False,
        "degraded": [],
        "sources": sources,
        "context": [
            {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
            for c in ctx.chunks[:10]
        ],
    }
