"""
Agent-search service: LLM-driven retrieval with plan, HyDE, fact queries, retry.

Flow: plan → HyDE (opt) → gate.chat → quality check → fact_queries (if poor/always) → answer → assess → retry if incomplete.
Endpoints: POST /v1/agent/stream (SSE), POST /v1/agent (JSON).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from agent_common import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    AsyncGateClient,
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    KEYWORD_QUERIES_SYSTEM,
    KEYWORD_QUERIES_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    build_context,
    context_from_hits,
    merge_hits,
    quality_is_poor,
    sources_from_context,
)

# Prometheus metrics
AGENT_REQS = Counter("agent_requests_total", "Agent requests", ["endpoint", "status"])
AGENT_LAT = Histogram(
    "agent_request_duration_seconds",
    "Request duration (seconds)",
    ["endpoint"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 90, 120),
)
AGENT_LLM_CALLS = Counter("agent_llm_calls_total", "LLM calls per request", ["stage"])
AGENT_GATE_CALLS = Counter("agent_gate_calls_total", "Gate chat calls per request")
AGENT_RETRY = Counter("agent_retry_total", "Retry triggered (incomplete answer)")


def _history_summary(history: list[dict[str, str]], max_turns: int = 3) -> str:
    """Format last N turns for prompt context."""
    if not history:
        return ""
    turns = history[-(max_turns * 2) :]
    parts = []
    for m in turns:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content[:200]}{'...' if len(content) > 200 else ''}")
    if not parts:
        return ""
    return "Recent conversation:\n" + "\n".join(parts) + "\n\n"


MEME_GRUMPS = [
    "Sigh. Fine. I will do science.",
    "This better be worth the tokens.",
    "I am not mad. I am just disappointed in entropy.",
    "Okay, okay, I will carry this search. Again.",
    "One more query and I start charging by the sigh.",
]


class GateFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    project_ids: list[str] | None = None


class AgentRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: GateFilters | None = None
    include_sources: bool = True


app = FastAPI(title="Agent Search", version="0.1.0")


def _env_get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _llm_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def _llm_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = await client.post(url, json=payload, headers=_llm_headers(api_key), timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


async def _llm_chat_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> AsyncIterator[str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
    headers = _llm_headers(api_key)
    async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout_s) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = payload.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content")
            if chunk:
                yield str(chunk)


async def _plan_retrieval(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    timeout_s: float,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    hist = _history_summary(history or [])
    system = {"role": "system", "content": PLAN_SYSTEM}
    user = {"role": "user", "content": PLAN_USER.format(history=hist, query=query)}
    raw = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.0,
        timeout_s=timeout_s,
    )
    try:
        return json.loads(raw)
    except Exception:
        return {"retrieval_mode": "hybrid", "top_k": 8, "rerank": True, "use_hyde": False, "reason": "fallback"}


async def _make_hyde(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    timeout_s: float,
    lang_hint: str = "English",
) -> str:
    system = {"role": "system", "content": HYDE_SYSTEM.format(lang=lang_hint)}
    user = {"role": "user", "content": HYDE_USER.format(query=query, lang=lang_hint)}
    return await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.2,
        timeout_s=timeout_s,
    )


async def _fact_queries(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    timeout_s: float,
    history: list[dict[str, str]] | None = None,
) -> list[str]:
    hist = _history_summary(history or [])
    system = {"role": "system", "content": FACT_QUERIES_SYSTEM}
    user = {"role": "user", "content": FACT_QUERIES_USER.format(history=hist, query=query)}
    raw = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.2,
        timeout_s=timeout_s,
    )
    try:
        data = json.loads(raw)
        out = data.get("fact_queries") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        return []


async def _keyword_queries(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    timeout_s: float,
    history: list[dict[str, str]] | None = None,
) -> list[str]:
    hist = _history_summary(history or [])
    system = {"role": "system", "content": KEYWORD_QUERIES_SYSTEM}
    user = {"role": "user", "content": KEYWORD_QUERIES_USER.format(history=hist, query=query)}
    raw = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.0,
        timeout_s=timeout_s,
    )
    try:
        data = json.loads(raw)
        out = data.get("keywords") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        return []


async def _assess_answer_completeness(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    question: str,
    answer: str,
    timeout_s: float,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You evaluate if the answer fully addresses the user's question. "
            "Return a single JSON object only."
        ),
    }
    hist = _history_summary(history or [])
    user = {
        "role": "user",
        "content": (
            f"{hist}"
            "Return JSON: {\"incomplete\": true|false, \"missing_terms\": [\"term1\", ...], "
            "\"reason\": \"short\"}. "
            "Mark incomplete if the answer says the info is missing or refuses to answer. "
            "missing_terms should be the concrete concepts that need more retrieval."
            f"\nQuestion: {question}\nAnswer: {answer}"
        ),
    }
    raw = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.0,
        timeout_s=timeout_s,
    )
    try:
        data = json.loads(raw)
    except Exception:
        return {"incomplete": False, "missing_terms": [], "reason": "parse_error"}
    if not isinstance(data, dict):
        return {"incomplete": False, "missing_terms": [], "reason": "bad_format"}
    missing_terms = data.get("missing_terms") or []
    if not isinstance(missing_terms, list):
        missing_terms = []
    return {
        "incomplete": bool(data.get("incomplete")),
        "missing_terms": [str(t).strip() for t in missing_terms if str(t).strip()],
        "reason": str(data.get("reason") or ""),
    }


async def _detect_language(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    text: str,
    timeout_s: float,
) -> str:
    system = {
        "role": "system",
        "content": "Detect the language of the text. Return a short language name in English only.",
    }
    user = {"role": "user", "content": f"Text:\n{text}"}
    raw = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.0,
        timeout_s=timeout_s,
    )
    return (raw or "").strip() or "English"


def _with_trace_id(event: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """Inject trace_id into event for OTEL/Jaeger correlation."""
    return {**event, "trace_id": trace_id}


async def _run_agent(payload: AgentRequest, client: httpx.AsyncClient) -> AsyncIterator[dict[str, Any]]:
    gate_url = _env_get("AGENT_GATE_URL", _env_get("GATE_URL", "http://rag-gate:8090"))
    llm_base = _env_get("AGENT_LLM_BASE_URL", _env_get("GATE_LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model = _env_get("AGENT_LLM_MODEL", _env_get("GATE_LLM_MODEL", "gpt-4o-mini"))
    llm_key = os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("GATE_LLM_API_KEY")
    llm_timeout = float(_env_get("AGENT_LLM_TIMEOUT_S", "60"))
    gate_timeout = float(_env_get("AGENT_GATE_TIMEOUT_S", "60"))
    max_llm_calls = int(_env_get("AGENT_MAX_LLM_CALLS", "12"))
    llm_calls = [0]

    def _can_llm() -> bool:
        llm_calls[0] += 1
        return llm_calls[0] <= max_llm_calls

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None
    gate = AsyncGateClient(gate_url, timeout_s=gate_timeout)

    hist = [m for m in payload.history] if payload.history else None
    yield {"type": "trace", "kind": "mood", "content": random.choice(MEME_GRUMPS)}
    yield {"type": "trace", "kind": "tool", "name": "llm.plan", "payload": {"model": llm_model, "query": payload.query}}
    if _can_llm():
        AGENT_LLM_CALLS.labels(stage="plan").inc()
        plan = await _plan_retrieval(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist)
    else:
        plan = {"retrieval_mode": "hybrid", "top_k": 8, "rerank": True, "use_hyde": False, "reason": "max_llm_calls"}

    mode = str(plan.get("retrieval_mode") or "hybrid")
    top_k = int(plan.get("top_k") or 8)
    rerank = bool(plan.get("rerank") if plan.get("rerank") is not None else True)
    use_hyde = bool(plan.get("use_hyde") or False)
    reason = str(plan.get("reason") or "no_reason")

    yield {
        "type": "trace",
        "kind": "thought",
        "label": "Plan",
        "content": f"mode={mode}, top_k={top_k}, rerank={rerank}, hyde={use_hyde}. Reason: {reason}",
    }

    if _can_llm():
        AGENT_LLM_CALLS.labels(stage="detect_lang").inc()
        answer_lang = await _detect_language(
            client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout
        )
    else:
        answer_lang = "English"

    search_query = payload.query
    if use_hyde and _can_llm():
        AGENT_LLM_CALLS.labels(stage="hyde").inc()
        yield {"type": "trace", "kind": "thought", "label": "HyDE", "content": f"Generating hypothetical passage in {answer_lang}."}
        yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"model": llm_model, "query": payload.query, "lang": answer_lang}}
        hyde = await _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, lang_hint=answer_lang)
        search_query = hyde.strip() or payload.query

    yield {
        "type": "trace",
        "kind": "tool",
        "name": "gate.chat",
        "payload": {"query": search_query, "retrieval_mode": mode, "top_k": top_k, "rerank": rerank},
    }
    AGENT_GATE_CALLS.inc()
    primary = await gate.chat(
        client,
        search_query,
        history=[m for m in payload.history] if payload.history else None,
        retrieval_mode=mode,
        top_k=top_k,
        rerank=rerank,
        filters=filters,
        include_sources=payload.include_sources,
    )

    min_hits = int(_env_get("AGENT_QUALITY_MIN_HITS", "3"))
    min_score = float(_env_get("AGENT_QUALITY_MIN_SCORE", "0.15"))
    always_fact = _env_get("AGENT_ALWAYS_FACT_QUERIES", "false").lower() in ("1", "true", "yes")
    responses = [primary]
    hits = list(primary.get("hits") or [])
    context_chunks = list(primary.get("context") or [])
    degraded = set(primary.get("degraded") or [])
    partial = bool(primary.get("partial"))

    run_fact = (quality_is_poor(primary, min_hits=min_hits, min_score=min_score) or always_fact) and _can_llm()
    if run_fact:
        AGENT_LLM_CALLS.labels(stage="fact_split").inc()
        msg = "Search looks weak. Splitting into fact queries." if quality_is_poor(primary, min_hits=min_hits, min_score=min_score) else "Multi-query fusion: adding fact queries."
        yield {"type": "trace", "kind": "thought", "label": "Quality", "content": msg}
        yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"model": llm_model, "query": payload.query}}
        fact_qs = await _fact_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist)
        if fact_qs:
            for fq in fact_qs:
                yield {"type": "trace", "kind": "action", "content": f"Fact query: {fq}"}
                yield {"type": "trace", "kind": "tool", "name": "gate.chat", "payload": {"query": fq, "retrieval_mode": mode, "top_k": max(4, top_k // 2), "rerank": rerank}}
            fact_tasks = [
                gate.chat(
                    client,
                    fq,
                    history=hist,
                    retrieval_mode=mode,
                    top_k=max(4, top_k // 2),
                    rerank=rerank,
                    filters=filters,
                    include_sources=payload.include_sources,
                )
                for fq in fact_qs
            ]
            fact_responses = await asyncio.gather(*fact_tasks)
            AGENT_GATE_CALLS.inc(len(fact_responses))
            responses.extend(fact_responses)
            for r2 in fact_responses:
                degraded.update(r2.get("degraded") or [])
                partial = partial or bool(r2.get("partial"))
            hits = merge_hits(responses, cap=max(12, top_k))
            context_chunks = context_from_hits(hits, context_chunks)
        else:
            yield {"type": "trace", "kind": "thought", "label": "Quality", "content": "No useful fact queries found."}

    if not context_chunks:
        context_chunks = context_from_hits(hits, context_chunks)

    retrieval_payload = {
        "hits": hits,
        "partial": partial,
        "degraded": list(degraded),
        "mode": mode,
    }
    yield {
        "type": "retrieval",
        "mode": mode,
        "partial": partial,
        "degraded": list(degraded),
        "context": context_chunks,
        "retrieval": retrieval_payload,
    }

    context_text = build_context(hits, limit=8, max_chars=4000)
    if not context_text and context_chunks:
        context_text = build_context(context_chunks, limit=8, max_chars=4000)
    if not context_text:
        yield {"type": "error", "error": "no_context"}
        return

    hist_ctx = _history_summary(hist) if hist else ""
    system = {"role": "system", "content": ANSWER_SYSTEM.format(lang=answer_lang)}
    user = {"role": "user", "content": ANSWER_USER.format(history=hist_ctx, query=payload.query, context=context_text)}

    AGENT_LLM_CALLS.labels(stage="answer").inc()
    yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": True}}
    answer_parts: list[str] = []
    async for chunk in _llm_chat_stream(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=0.2,
        timeout_s=llm_timeout,
    ):
        answer_parts.append(chunk)
        yield {"type": "token", "content": chunk}

    full_answer = "".join(answer_parts)
    sources = sources_from_context(context_chunks) if context_chunks else (primary.get("sources") or [])

    if _can_llm():
        AGENT_LLM_CALLS.labels(stage="assess").inc()
        assessment = await _assess_answer_completeness(
            client,
            llm_base,
            llm_model,
            llm_key,
            payload.query,
            full_answer,
            timeout_s=llm_timeout,
            history=hist,
        )
    else:
        assessment = {"incomplete": False, "missing_terms": [], "reason": "max_llm_calls"}

    if assessment.get("incomplete") and _can_llm():
        AGENT_RETRY.inc()
        yield {
            "type": "trace",
            "kind": "thought",
            "label": "Retry",
            "content": "Answer looks underspecified. Trying alternate retrieval strategy.",
        }

        retry_mode = "hybrid"
        retry_top_k = max(12, top_k * 2)
        retry_rerank = True

        missing_terms = list(assessment.get("missing_terms") or [])
        retry_query = payload.query
        if _can_llm():
            AGENT_LLM_CALLS.labels(stage="hyde_retry").inc()
            yield {
                "type": "trace",
                "kind": "tool",
                "name": "llm.hyde",
                "payload": {"model": llm_model, "query": payload.query, "lang": answer_lang},
            }
            hyde_retry = await _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, lang_hint=answer_lang)
            if hyde_retry.strip():
                retry_query = hyde_retry.strip()

        responses = []
        queries = [retry_query]
        for term in missing_terms:
            if term and term.lower() not in retry_query.lower():
                queries.append(f"{payload.query} {term}")

        if _can_llm():
            AGENT_LLM_CALLS.labels(stage="keywords").inc()
            yield {"type": "trace", "kind": "tool", "name": "llm.keywords", "payload": {"model": llm_model, "query": payload.query}}
            keyword_qs = await _keyword_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist)
        else:
            keyword_qs = []
        for kw in keyword_qs:
            if kw and kw.lower() not in retry_query.lower():
                queries.append(kw)

        for q in queries:
            yield {"type": "trace", "kind": "tool", "name": "gate.chat", "payload": {"query": q, "retrieval_mode": retry_mode, "top_k": retry_top_k, "rerank": retry_rerank}}
        retry_tasks = [
            gate.chat(client, q, history=hist, retrieval_mode=retry_mode, top_k=retry_top_k, rerank=retry_rerank, filters=filters, include_sources=payload.include_sources)
            for q in queries
        ]
        retry_responses = await asyncio.gather(*retry_tasks)
        AGENT_GATE_CALLS.inc(len(retry_responses))

        retry_resp = retry_responses[0] if retry_responses else {}
        retry_hits = list(retry_resp.get("hits") or [])
        retry_context = list(retry_resp.get("context") or [])
        retry_degraded = set(retry_resp.get("degraded") or [])
        retry_partial = bool(retry_resp.get("partial"))

        fact_qs = []
        if _can_llm():
            AGENT_LLM_CALLS.labels(stage="fact_split_retry").inc()
            yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"model": llm_model, "query": payload.query}}
            fact_qs = await _fact_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist)
        if fact_qs:
            for fq in fact_qs:
                yield {"type": "trace", "kind": "action", "content": f"Fact query: {fq}"}
                yield {"type": "trace", "kind": "tool", "name": "gate.chat", "payload": {"query": fq, "retrieval_mode": retry_mode, "top_k": max(4, retry_top_k // 2), "rerank": retry_rerank}}
            fact_retry_tasks = [
                gate.chat(client, fq, history=hist, retrieval_mode=retry_mode, top_k=max(4, retry_top_k // 2), rerank=retry_rerank, filters=filters, include_sources=payload.include_sources)
                for fq in fact_qs
            ]
            fact_retry_responses = await asyncio.gather(*fact_retry_tasks)
            AGENT_GATE_CALLS.inc(len(fact_retry_responses))
            responses = [retry_resp] + list(fact_retry_responses)
            for r2 in fact_retry_responses:
                retry_degraded.update(r2.get("degraded") or [])
                retry_partial = retry_partial or bool(r2.get("partial"))
            retry_hits = merge_hits(responses, cap=max(12, retry_top_k))
            retry_context = context_from_hits(retry_hits, retry_context)

        if not retry_context:
            retry_context = context_from_hits(retry_hits, retry_context)

        yield {
            "type": "retrieval",
            "mode": retry_mode,
            "partial": retry_partial,
            "degraded": list(retry_degraded),
            "context": retry_context,
            "retrieval": {
                "hits": retry_hits,
                "partial": retry_partial,
                "degraded": list(retry_degraded),
                "mode": retry_mode,
            },
        }

        retry_context_text = build_context(retry_hits, limit=8, max_chars=4000)
        if not retry_context_text and retry_context:
            retry_context_text = build_context(retry_context, limit=8, max_chars=4000)

        if retry_context_text:
            hist_ctx = _history_summary(hist) if hist else ""
            system = {"role": "system", "content": ANSWER_SYSTEM.format(lang=answer_lang)}
            user = {"role": "user", "content": ANSWER_USER.format(history=hist_ctx, query=payload.query, context=retry_context_text)}
            AGENT_LLM_CALLS.labels(stage="answer_retry").inc()
            yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": False}}
            full_answer = await _llm_chat(
                client,
                llm_base,
                llm_model,
                llm_key,
                [system, user],
                temperature=0.2,
                timeout_s=llm_timeout,
            )
            sources = sources_from_context(retry_context) if retry_context else (retry_resp.get("sources") or [])
            context_chunks = retry_context
            degraded = retry_degraded
            partial = retry_partial

    yield {
        "type": "done",
        "answer": full_answer,
        "mode": mode,
        "partial": partial,
        "degraded": list(degraded),
        "sources": sources if payload.include_sources else [],
        "context": context_chunks,
    }


@app.get("/v1/readyz")
async def readyz():
    return {"ready": True}


@app.get("/v1/metrics")
async def metrics():
    """Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/agent")
async def agent_non_streaming(payload: AgentRequest):
    """Non-streaming: collect events and return full response as JSON."""
    t0 = time.perf_counter()
    trace_id = uuid.uuid4().hex
    timeout_s = float(_env_get("AGENT_REQUEST_TIMEOUT_S", "120"))
    answer = ""
    sources: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    mode = "hybrid"
    partial = False
    degraded: list[str] = []
    error: str | None = None

    async def _collect() -> None:
        nonlocal answer, sources, context, mode, partial, degraded, error
        async with httpx.AsyncClient() as client:
            async for event in _run_agent(payload, client):
                if event.get("type") == "retrieval":
                    context = list(event.get("context") or [])
                elif event.get("type") == "token":
                    answer += event.get("content", "")
                elif event.get("type") == "done":
                    answer = event.get("answer", answer)
                    sources = list(event.get("sources") or [])
                    context = list(event.get("context") or context)
                    mode = event.get("mode", mode)
                    partial = event.get("partial", partial)
                    degraded = list(event.get("degraded") or [])
                elif event.get("type") == "error":
                    error = event.get("error", "unknown")
                    break

    try:
        await asyncio.wait_for(_collect(), timeout=timeout_s)
    except asyncio.TimeoutError:
        error = "request_timeout"
    except Exception as exc:
        error = str(exc)
    AGENT_LAT.labels(endpoint="/v1/agent").observe(time.perf_counter() - t0)
    if error:
        AGENT_REQS.labels(endpoint="/v1/agent", status="error").inc()
        raise HTTPException(status_code=500, detail=error)
    AGENT_REQS.labels(endpoint="/v1/agent", status="ok").inc()
    return {
        "trace_id": trace_id,
        "answer": answer,
        "sources": sources if payload.include_sources else [],
        "context": context,
        "mode": mode,
        "partial": partial,
        "degraded": degraded,
    }


@app.post("/v1/agent/stream")
async def agent_stream(request: Request, payload: AgentRequest):
    t0 = time.perf_counter()
    trace_id = uuid.uuid4().hex
    timeout_s = float(_env_get("AGENT_REQUEST_TIMEOUT_S", "120"))

    async def event_stream() -> AsyncIterator[str]:
        deadline = asyncio.get_event_loop().time() + timeout_s
        stream_error: str | None = None
        try:
            async with httpx.AsyncClient() as client:
                try:
                    async for event in _run_agent(payload, client):
                        if getattr(request, "is_disconnected", False):
                            stream_error = "client_disconnected"
                            return
                        if asyncio.get_event_loop().time() > deadline:
                            stream_error = "request_timeout"
                            yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': 'request_timeout'}, trace_id))}\n\n"
                            return
                        yield f"data: {json.dumps(_with_trace_id(event, trace_id))}\n\n"
                except asyncio.CancelledError:
                    stream_error = "client_disconnected"
                    raise
                except asyncio.TimeoutError:
                    stream_error = "request_timeout"
                    yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': 'request_timeout'}, trace_id))}\n\n"
                except Exception as exc:
                    stream_error = str(exc)
                    yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': str(exc)}, trace_id))}\n\n"
        except asyncio.CancelledError:
            stream_error = "client_disconnected"
        finally:
            AGENT_LAT.labels(endpoint="/v1/agent/stream").observe(time.perf_counter() - t0)
            AGENT_REQS.labels(endpoint="/v1/agent/stream", status="ok" if not stream_error else "error").inc()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
