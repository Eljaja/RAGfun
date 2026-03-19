from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from llm import LLMClient, generate_answer, reflect
from models import AgentRequest, ChunkResult, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def chunks_to_agent_context(chunks: list[ChunkResult]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for c in chunks:
        chunk_id = f"{c.source_id}:{c.chunk_index}"
        context.append(
            {
                "chunk_id": chunk_id,
                "doc_id": c.source_id,
                "text": c.text,
                "score": c.score,
                "source": {"doc_id": c.source_id},
            }
        )
    return context


def sources_from_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for c in context:
        doc_id = str((c.get("source") or {}).get("doc_id") or c.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append(
            {
                "ref": len(sources) + 1,
                "doc_id": doc_id,
                "title": None,
                "uri": None,
                "locator": None,
            }
        )
    return sources


def build_plan_from_legacy(strategy: str, top_k: int, rerank: bool) -> dict[str, Any]:
    strategy = (strategy or "hybrid").lower().strip()
    top_k = max(1, int(top_k))
    if strategy == "bm25":
        return {
            "round": {
                "retrieve": [{"kind": "bm25_search", "top_k": top_k}],
                "finalize": [{"kind": "trim", "top_k": top_k}],
            }
        }
    if strategy == "vector":
        return {
            "round": {
                "retrieve": [{"kind": "vector_search", "top_k": top_k}],
                "finalize": [{"kind": "trim", "top_k": top_k}],
            }
        }

    fetch_k = max(top_k * 2, top_k)
    rank_steps: list[dict[str, Any]] = []
    if rerank:
        rank_steps.append({"kind": "rerank", "top_n": min(fetch_k, max(top_k, 8))})
    return {
        "round": {
            "retrieve": [
                {"kind": "vector_search", "top_k": fetch_k},
                {"kind": "bm25_search", "top_k": fetch_k},
            ],
            "combine": {"kind": "fuse", "method": "rrf", "rrf_k": 60},
            "rank": rank_steps,
            "finalize": [{"kind": "trim", "top_k": top_k}],
        }
    }


def build_agent_plan(strategy: str, top_k: int, rerank: bool, mode: str | None) -> dict[str, Any]:
    mode_norm = (mode or "conservative").lower().strip()
    base = build_plan_from_legacy(strategy, top_k, rerank)
    round_obj = base["round"]
    round_obj.setdefault("rank", [])
    if mode_norm in ("conservative", "aggressive"):
        if rerank and not any(s.get("kind") == "rerank" for s in round_obj["rank"]):
            round_obj["rank"].append({"kind": "rerank", "top_n": max(top_k, min(top_k * 2, 24))})
        round_obj["rank"].append({"kind": "adaptive_k", "min_k": 3, "max_k": max(24, top_k * 2)})
    return base


async def call_retrieval(
    client: httpx.AsyncClient,
    retrieval_url: str,
    project_id: str,
    query: str,
    top_k: int,
    strategy: str,
    rerank: bool,
) -> list[ChunkResult]:
    """Call the retrieval service over HTTP."""
    resp = await client.post(
        f"{retrieval_url.rstrip('/')}/retrieve",
        json={
            "project_id": project_id,
            "query": query,
            "top_k": top_k,
            "strategy": strategy,
            "rerank": rerank,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [ChunkResult(**c) for c in data["chunks"]]


async def call_retrieval_plan(
    client: httpx.AsyncClient,
    retrieval_url: str,
    project_id: str,
    query: str,
    plan: dict[str, Any],
) -> list[ChunkResult]:
    resp = await client.post(
        f"{retrieval_url.rstrip('/')}/plan/retrieve",
        json={
            "project_id": project_id,
            "query": query,
            "plan": plan,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [ChunkResult(**c) for c in data["chunks"]]


def merge_chunks(chunks_by_query: list[list[ChunkResult]], limit: int) -> list[ChunkResult]:
    by_key: dict[str, ChunkResult] = {}
    for group in chunks_by_query:
        for c in group:
            key = f"{c.source_id}:{c.chunk_index}"
            prev = by_key.get(key)
            if prev is None or c.score > prev.score:
                by_key[key] = c
    return sorted(by_key.values(), key=lambda c: c.score, reverse=True)[:limit]


async def generate_fact_queries(
    llm: LLMClient,
    model: str,
    query: str,
    max_queries: int,
) -> list[str]:
    raw = await llm.complete(
        model,
        [
            {
                "role": "system",
                "content": "Split the user request into small factual sub-queries. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Return JSON object with field fact_queries (array of strings). "
                    f"Give at most {max_queries} items.\n"
                    f"Question: {query}"
                ),
            },
        ],
    )
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out = data.get("fact_queries") or []
    return [str(q).strip() for q in out if str(q).strip()][:max_queries]


async def rag_pipeline(
    request: ChatRequest,
    retrieval_url: str,
    http_client: httpx.AsyncClient,
    llm: LLMClient,
    gen_model: str,
    reflection_model: str,
    reflection_enabled: bool,
) -> ChatResponse:
    """
    Full RAG pipeline:
    1. Call retrieval service for chunks
    2. Generate answer via LLM
    3. Optionally reflect and re-query if answer is incomplete
    """
    query = request.query
    do_reflect = request.reflection_enabled and reflection_enabled
    max_retries = request.max_retries if do_reflect else 0
    retries_used = 0
    answer = ""
    chunks: list[ChunkResult] = []

    for attempt in range(max_retries + 1):
        chunks = await call_retrieval(
            client=http_client,
            retrieval_url=retrieval_url,
            project_id=request.project_id,
            query=query,
            top_k=request.top_k,
            strategy=request.strategy,
            rerank=request.rerank,
        )

        if not chunks:
            return ChatResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                chunks_used=0,
                retries_used=retries_used,
                query=request.query,
            )

        answer = await generate_answer(llm, gen_model, query, chunks)

        if not do_reflect:
            break

        try:
            reflection = await reflect(llm, reflection_model, query, chunks, answer)
            logger.info(
                "Reflection attempt %d: complete=%s, requery=%s",
                attempt, reflection.complete, reflection.requery,
            )

            if reflection.complete:
                break

            query = reflection.requery or query
            retries_used += 1
        except Exception:
            logger.warning("Reflection failed on attempt %d, using current answer", attempt, exc_info=True)
            break

    sources = list({c.source_id for c in chunks})

    return ChatResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(chunks),
        retries_used=retries_used,
        query=request.query,
    )


async def rag_agent_pipeline(
    request: AgentRequest,
    retrieval_url: str,
    http_client: httpx.AsyncClient,
    llm: LLMClient,
    gen_model: str,
) -> dict[str, Any]:
    plan = build_agent_plan(request.strategy, request.top_k, request.rerank, request.mode)
    chunks = await call_retrieval_plan(
        client=http_client,
        retrieval_url=retrieval_url,
        project_id=request.project_id,
        query=request.query,
        plan=plan,
    )

    mode_norm = (request.mode or "conservative").lower().strip()
    if mode_norm == "aggressive":
        try:
            fact_queries = await generate_fact_queries(llm, gen_model, request.query, max_queries=2)
        except Exception:
            fact_queries = []
        if fact_queries:
            all_chunks = [chunks]
            for fq in fact_queries:
                try:
                    fq_chunks = await call_retrieval_plan(
                        client=http_client,
                        retrieval_url=retrieval_url,
                        project_id=request.project_id,
                        query=fq,
                        plan=plan,
                    )
                    all_chunks.append(fq_chunks)
                except Exception:
                    logger.warning("Fact query retrieval failed: %s", fq, exc_info=True)
            chunks = merge_chunks(all_chunks, limit=max(8, request.top_k * 2))

    if not chunks:
        return {
            "answer": "",
            "context": [],
            "sources": [],
            "mode": request.strategy,
            "partial": True,
            "degraded": ["no_context"],
            "plan": plan,
        }

    answer = await generate_answer(llm, gen_model, request.query, chunks)
    context = chunks_to_agent_context(chunks)
    sources = sources_from_context(context) if request.include_sources else []
    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "mode": request.strategy,
        "partial": False,
        "degraded": [],
        "plan": plan,
    }
