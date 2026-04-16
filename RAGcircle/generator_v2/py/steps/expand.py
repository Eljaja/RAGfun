"""Expand step implementations: query expansion (LLM-based and heuristic).

Leaf functions return (list[RetrievalRequest], traces).
Dispatch factories build closures used by run_stage.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from engine.brain_budget import BudgetCounter
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.plan import ConfigMeta, RetrievalRequest
from models.steps import (
    BM25AnchorStep,
    FactoidExpandStep,
    FactQueryStep,
    HyDEStep,
    InitialExpandStep,
    KeywordStep,
    LoopExpandStep,
    QueryVariantsStep,
    TwoPassStep,
)
from retrieval_contract import ChunkResult, ExecutionPlan
from retrieval_contract import from_preset
from steps.prompts import (
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
)
from steps.query_heuristics import (
    extract_hint_terms,
    keyword_query,
    query_variants as heuristic_variants,
    unique_source_count,
)

logger = logging.getLogger(__name__)


# ── Dispatch factories ───────────────────────────────────


def make_expand_dispatch(
    *,
    query: str,
    meta: ConfigMeta,
    history_text: str,
    budget: BudgetCounter,
    llm: LLMClient,
    model: str,
) -> Callable[[InitialExpandStep], Any]:
    """Return a dispatch closure for initial-expand steps."""

    def dispatch(step: InitialExpandStep):
        match step:
            case HyDEStep(num_passages=n):
                return hyde(
                    query=query, lang=meta.lang, budget=budget,
                    llm=llm, model=model, num_passages=n,
                )
            case FactQueryStep(max_queries=n):
                return fact_queries(
                    query=query, history_text=history_text,
                    max_queries=n, budget=budget,
                    retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                )
            case KeywordStep():
                return keywords(
                    query=query, history_text=history_text,
                    budget=budget, retrieval_plan=meta.retrieval_plan,
                    llm=llm, model=model,
                )
            case QueryVariantsStep():
                return query_variants_expand(query=query)
            case BM25AnchorStep(top_k=k):
                return bm25_anchor_expand(query=query, top_k=k)
            case _:
                raise TypeError(f"unknown expand step: {type(step).__name__}")

    return dispatch


def make_loop_expand_dispatch(
    *,
    query: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
    retrieval_plan: ExecutionPlan | None,
) -> Callable[[LoopExpandStep], Any]:
    """Return a dispatch closure for loop-expand steps."""

    def dispatch(step: LoopExpandStep):
        match step:
            case TwoPassStep(min_unique_sources=n):
                return two_pass_expand(
                    query=query, chunks=chunks,
                    min_unique_sources=n, retrieval_plan=retrieval_plan,
                )
            case FactoidExpandStep():
                return factoid_expand(
                    query=query, chunks=chunks, is_factoid=is_factoid,
                )
            case _:
                raise TypeError(f"unknown loop expand step: {type(step).__name__}")

    return dispatch


# ── LLM-based expand leaves ──────────────────────────────


async def hyde(
    *, query: str, lang: str, budget: BudgetCounter,
    llm: LLMClient, model: str, num_passages: int = 1,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Generate hypothetical passages for retrieval.

    When num_passages > 1, generates N passages at escalating temperatures
    (0.2, 0.35, 0.50, ...) for recall diversity.
    """
    actual_n = min(num_passages, budget.remaining)
    if actual_n < 1 or not budget.try_consume():
        return [], []

    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.hyde",
         "payload": {"query": query, "num": actual_n}},
    ]
    messages = [
        {"role": "system", "content": HYDE_SYSTEM.format(lang=lang)},
        {"role": "user", "content": HYDE_USER.format(query=query, lang=lang)},
    ]

    if actual_n == 1:
        try:
            passage = await llm.complete(model, messages, temperature=0.2)
            text = (passage or "").strip()
            reqs = [RetrievalRequest(query=text)] if text else []
            traces.append({
                "kind": "thought", "label": "HyDE",
                "content": f"Generated {len(reqs)}/1 hypothetical passages",
            })
            return reqs, traces
        except LLMTransportError as exc:
            logger.warning("HyDE failed (transport): %s", exc)
            return [], traces

    for _ in range(actual_n - 1):
        budget.try_consume()

    temps = [0.2 + 0.15 * i for i in range(actual_n)]
    tasks = [llm.complete(model, messages, temperature=t) for t in temps]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    reqs: list[RetrievalRequest] = []
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("HyDE passage failed: %s", r)
            continue
        text = (r or "").strip()
        if text:
            reqs.append(RetrievalRequest(query=text))

    traces.append({
        "kind": "thought", "label": "HyDE",
        "content": f"Generated {len(reqs)}/{actual_n} hypothetical passages",
    })
    return reqs, traces


async def fact_queries(
    *,
    query: str,
    history_text: str,
    max_queries: int,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    if retrieval_plan is None or not budget.try_consume():
        return [], []

    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.fact_split", "payload": {"query": query}},
    ]

    try:
        data = await llm.complete_json(
            model,
            [
                {"role": "system", "content": FACT_QUERIES_SYSTEM},
                {"role": "user", "content": FACT_QUERIES_USER.format(
                    history=history_text, query=query,
                )},
            ],
            temperature=0.2,
        )
    except LLMTransportError as exc:
        logger.warning("fact_queries LLM unavailable: %s", exc)
        return [], traces
    except LLMParseError as exc:
        logger.warning("fact_queries parse failed: %s — raw: %s", exc, exc.raw[:300])
        return [], traces

    sub = [str(q).strip() for q in (data.get("fact_queries") or []) if str(q).strip()]
    reqs = [RetrievalRequest(query=q) for q in sub[:max_queries]]
    traces.append({
        "kind": "thought", "label": "FactQueries",
        "content": f"Split into {len(reqs)} sub-queries",
    })
    return reqs, traces


async def keywords(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    if retrieval_plan is None or not budget.try_consume():
        return [], []

    traces: list[dict[str, Any]] = [
        {"kind": "tool", "name": "llm.keywords", "payload": {"query": query}},
    ]

    try:
        data = await llm.complete_json(
            model,
            [
                {"role": "system", "content": KEYWORD_QUERIES_SYSTEM},
                {"role": "user", "content": KEYWORD_QUERIES_USER.format(
                    history=history_text, query=query,
                )},
            ],
            temperature=0.0,
        )
    except LLMTransportError as exc:
        logger.warning("keywords LLM unavailable: %s", exc)
        return [], traces
    except LLMParseError as exc:
        logger.warning("keywords parse failed: %s — raw: %s", exc, exc.raw[:300])
        return [], traces

    kw = [str(q).strip() for q in (data.get("keywords") or []) if str(q).strip()]
    reqs = [RetrievalRequest(query=q) for q in kw[:4]]
    traces.append({
        "kind": "thought", "label": "Keywords",
        "content": f"Extracted {len(reqs)} keyword queries",
    })
    return reqs, traces


# ── Heuristic expand leaves ──────────────────────────────


def query_variants_expand(
    *, query: str,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Generate syntactic query variants (no LLM call)."""
    variants = heuristic_variants(query)
    if len(variants) <= 1:
        return [], []
    reqs = [RetrievalRequest(query=v) for v in variants[1:]]
    return reqs, []


def bm25_anchor_expand(
    *, query: str, top_k: int,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Create a BM25-only anchor request from keyword extraction."""
    kw = keyword_query(query)
    if not kw:
        return [], []
    plan = from_preset("fast", top_k=top_k, rerank=False)
    reqs = [RetrievalRequest(query=kw, plan_override=plan)]
    return reqs, []


def two_pass_expand(
    *,
    query: str,
    chunks: list[ChunkResult],
    min_unique_sources: int,
    retrieval_plan: ExecutionPlan | None,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Follow-up query when unique-source count is below threshold."""
    n_unique = unique_source_count(chunks)
    if n_unique >= min_unique_sources:
        return [], []
    hints = extract_hint_terms(chunks, max_terms=3)
    if not hints:
        return [], []
    follow_up = f"{query} {' '.join(hints)}"
    plan = retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
    reqs = [RetrievalRequest(query=follow_up, plan_override=plan)]
    return reqs, []


def factoid_expand(
    *,
    query: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
) -> tuple[list[RetrievalRequest], list[dict[str, Any]]]:
    """Fast shallow expansion for factoid-type questions."""
    if not is_factoid or not chunks:
        return [], []
    plan = from_preset("fast", top_k=5, rerank=False)
    reqs = [RetrievalRequest(query=query, plan_override=plan)]
    return reqs, []
