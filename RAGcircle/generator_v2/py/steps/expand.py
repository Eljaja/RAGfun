"""Expand step implementations: query expansion (LLM-based and heuristic).

Leaf functions called by engine/retrieval's dispatch loops.
Each function takes a TraceCollector and emits traces directly.
Return values are pure results (no traces).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engine.brain_budget import BudgetCounter
from engine.trace_collector import TraceCollector
from clients.llm import LLMClient, LLMParseError, LLMTransportError
from models.plan import RetrievalRequest
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


async def hyde(
    *, query: str, lang: str, budget: BudgetCounter,
    llm: LLMClient, model: str, num_passages: int = 1,
    collector: TraceCollector,
) -> list[str]:
    """Generate hypothetical passages for retrieval.

    When num_passages > 1, generates N passages at escalating temperatures
    (0.2, 0.35, 0.50, ...) for recall diversity, matching the old agent-search
    multi-HyDE strategy.
    """
    actual_n = min(num_passages, budget.remaining)
    if actual_n < 1 or not budget.try_consume():
        return []

    messages = [
        {"role": "system", "content": HYDE_SYSTEM.format(lang=lang)},
        {"role": "user", "content": HYDE_USER.format(query=query, lang=lang)},
    ]
    await collector.emit(
        {"kind": "tool", "name": "llm.hyde",
         "payload": {"query": query, "num": actual_n}},
    )

    if actual_n == 1:
        try:
            passage = await llm.complete(model, messages, temperature=0.2)
            result = [passage.strip()] if passage and passage.strip() else []
            return result
        except LLMTransportError as exc:
            logger.warning("HyDE failed (transport): %s", exc)
            return []

    for _ in range(actual_n - 1):
        budget.try_consume()

    temps = [0.2 + 0.15 * i for i in range(actual_n)]
    tasks = [llm.complete(model, messages, temperature=t) for t in temps]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    passages = []
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("HyDE passage failed: %s", r)
            continue
        text = (r or "").strip()
        if text:
            passages.append(text)

    await collector.emit({
        "kind": "thought", "label": "HyDE",
        "content": f"Generated {len(passages)}/{actual_n} hypothetical passages",
    })
    return passages


async def fact_queries(
    *,
    query: str,
    history_text: str,
    max_queries: int,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
    collector: TraceCollector,
) -> list[str]:
    if retrieval_plan is None or not budget.try_consume():
        return []

    await collector.emit(
        {"kind": "tool", "name": "llm.fact_split", "payload": {"query": query}},
    )

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
        return []
    except LLMParseError as exc:
        logger.warning("fact_queries parse failed: %s — raw: %s", exc, exc.raw[:300])
        return []

    sub_queries = [str(q).strip() for q in (data.get("fact_queries") or []) if str(q).strip()]
    return sub_queries[:max_queries]


async def keywords(
    *,
    query: str,
    history_text: str,
    budget: BudgetCounter,
    retrieval_plan: ExecutionPlan | None,
    llm: LLMClient, model: str,
    collector: TraceCollector,
) -> list[str]:
    if retrieval_plan is None or not budget.try_consume():
        return []

    await collector.emit(
        {"kind": "tool", "name": "llm.keywords", "payload": {"query": query}},
    )

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
        return []
    except LLMParseError as exc:
        logger.warning("keywords parse failed: %s — raw: %s", exc, exc.raw[:300])
        return []

    kw = [str(q).strip() for q in (data.get("keywords") or []) if str(q).strip()]
    return kw[:4]


# ── Heuristic expand steps ───────────────────────────────


def query_variants_expand(
    *, query: str,
) -> list[RetrievalRequest]:
    """Generate syntactic query variants (no LLM call)."""
    variants = heuristic_variants(query)
    if len(variants) <= 1:
        return []
    reqs = [RetrievalRequest(query=v) for v in variants[1:]]
    return reqs


def bm25_anchor_expand(
    *, query: str, top_k: int,
) -> list[RetrievalRequest]:
    """Create a BM25-only anchor request from keyword extraction."""
    kw = keyword_query(query)
    if not kw:
        return []
    plan = from_preset("fast", top_k=top_k, rerank=False)
    reqs = [RetrievalRequest(query=kw, plan_override=plan)]
    return reqs


def two_pass_expand(
    *,
    query: str,
    chunks: list[ChunkResult],
    min_unique_sources: int,
    retrieval_plan: ExecutionPlan | None,
) -> list[RetrievalRequest]:
    """Follow-up query when unique-source count is below threshold."""
    n_unique = unique_source_count(chunks)
    if n_unique >= min_unique_sources:
        return []
    hints = extract_hint_terms(chunks, max_terms=3)
    if not hints:
        return []
    follow_up = f"{query} {' '.join(hints)}"
    plan = retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
    reqs = [RetrievalRequest(query=follow_up, plan_override=plan)]
    return reqs


def factoid_expand(
    *,
    query: str,
    chunks: list[ChunkResult],
    is_factoid: bool,
) -> list[RetrievalRequest]:
    """Fast shallow expansion for factoid-type questions."""
    if not is_factoid or not chunks:
        return []
    plan = from_preset("fast", top_k=5, rerank=False)
    reqs = [RetrievalRequest(query=query, plan_override=plan)]
    return reqs
