from __future__ import annotations

import json
import os
import random
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


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
) -> dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You are a retrieval strategist for a RAG system. "
            "Return a single JSON object only. Keep 'reason' short."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "Decide per-request retrieval knobs.\n"
            "JSON fields: retrieval_mode (bm25|vector|hybrid), top_k (1..40), "
            "rerank (true/false), use_hyde (true/false), reason.\n"
            f"Query: {query}"
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
) -> str:
    system = {"role": "system", "content": "Write a short hypothetical answer passage for retrieval. English only."}
    user = {"role": "user", "content": f"Query: {query}\nReturn a 3-5 sentence passage."}
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
) -> list[str]:
    system = {"role": "system", "content": "Extract fact-oriented sub-queries from the user request."}
    user = {
        "role": "user",
        "content": (
            "Return JSON: {\"fact_queries\": [..]} with 2-3 short queries.\n"
            f"Query: {query}"
        ),
    }
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
) -> list[str]:
    system = {"role": "system", "content": "Extract short keyword queries from the user request."}
    user = {
        "role": "user",
        "content": (
            "Return JSON: {\"keywords\": [..]} with 3-6 short keyword phrases. "
            f"Query: {query}"
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
        out = data.get("keywords") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        return []


async def _gate_search(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    mode: str,
    top_k: int,
    rerank: bool,
    filters: dict[str, Any] | None,
    include_sources: bool,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat"
    payload = {
        "query": query,
        "history": [],
        "retrieval_mode": mode,
        "top_k": int(top_k),
        "rerank": bool(rerank),
        "include_sources": bool(include_sources),
    }
    if filters:
        payload["filters"] = filters
    resp = await client.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    retrieval = data.get("retrieval") or {}
    hits = list(retrieval.get("hits") or [])
    context_chunks = list(data.get("context") or [])
    if context_chunks:
        by_cid = {str(c.get("chunk_id")): c for c in context_chunks if c.get("chunk_id")}
        for h in hits:
            if h.get("text"):
                continue
            cid = str(h.get("chunk_id") or "")
            c = by_cid.get(cid)
            if c:
                h["text"] = c.get("text")
                h["source"] = c.get("source")
                if h.get("score") is None:
                    h["score"] = c.get("score")
    if not hits and context_chunks:
        hits = [
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "score": c.get("score"),
                "text": c.get("text"),
                "source": c.get("source"),
            }
            for c in context_chunks
        ]

    return {
        "ok": bool(retrieval.get("ok", True)),
        "mode": retrieval.get("mode", mode),
        "partial": bool(retrieval.get("partial")),
        "degraded": retrieval.get("degraded") or [],
        "hits": hits,
        "context": context_chunks,
        "sources": data.get("sources") or [],
        "retrieval": retrieval,
    }


def _quality_is_poor(resp: dict[str, Any]) -> bool:
    hits = list(resp.get("hits") or [])
    if not hits:
        return True
    if len(hits) < 3:
        return True
    if resp.get("partial") or (resp.get("degraded") or []):
        return True
    top = hits[0]
    score = top.get("rerank_score") if top.get("rerank_score") is not None else top.get("score")
    if isinstance(score, (int, float)) and score < 0.15:
        return True
    return False


def _merge_hits(responses: list[dict[str, Any]], cap: int = 20) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for r in responses:
        for h in r.get("hits") or []:
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else 0.0
            if cid not in merged or score_f > scores.get(cid, -1.0):
                merged[cid] = h
                scores[cid] = score_f
    out = sorted(merged.values(), key=lambda h: scores.get(str(h.get("chunk_id") or ""), 0.0), reverse=True)
    return out[:cap]


def _build_context(hits: list[dict[str, Any]], limit: int = 8, max_chars: int = 4000) -> str:
    blocks: list[str] = []
    total = 0
    for i, h in enumerate(hits[:limit], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(h.get("doc_id") or "-")
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        max_block = max(300, min(1200, max_chars - total))
        if len(text) > max_block:
            text = text[: max_block - 3].rstrip() + "..."
        block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        if total + len(block) > max_chars:
            remaining = max(0, max_chars - total - 60)
            if remaining <= 0:
                break
            text = text[:remaining].rstrip() + "..."
            block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _context_from_hits(hits: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cid = {str(c.get("chunk_id")): c for c in contexts if c.get("chunk_id")}
    out: list[dict[str, Any]] = []
    for h in hits:
        cid = str(h.get("chunk_id") or "")
        if cid and cid in by_cid:
            out.append(by_cid[cid])
            continue
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        out.append(
            {
                "chunk_id": cid,
                "doc_id": str(h.get("doc_id") or ""),
                "text": h.get("text"),
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "source": h.get("source"),
            }
        )
    return out


def _sources_from_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for c in context:
        src = c.get("source") or {}
        doc_id = src.get("doc_id") or c.get("doc_id")
        if not doc_id:
            continue
        key = (doc_id, src.get("uri"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "ref": src.get("ref"),
                "doc_id": doc_id,
                "title": src.get("title"),
                "uri": src.get("uri"),
                "locator": src.get("locator"),
            }
        )
    return out


async def _assess_answer_completeness(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    question: str,
    answer: str,
    timeout_s: float,
) -> dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You evaluate if the answer fully addresses the user's question. "
            "Return a single JSON object only."
        ),
    }
    user = {
        "role": "user",
        "content": (
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


async def _run_agent(payload: AgentRequest, client: httpx.AsyncClient) -> AsyncIterator[dict[str, Any]]:
    gate_url = _env_get("AGENT_GATE_URL", _env_get("GATE_URL", "http://rag-gate:8090"))
    llm_base = _env_get("AGENT_LLM_BASE_URL", _env_get("GATE_LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model = _env_get("AGENT_LLM_MODEL", _env_get("GATE_LLM_MODEL", "gpt-4o-mini"))
    llm_key = os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("GATE_LLM_API_KEY")
    llm_timeout = float(_env_get("AGENT_LLM_TIMEOUT_S", "60"))
    gate_timeout = float(_env_get("AGENT_GATE_TIMEOUT_S", "60"))

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    yield {"type": "trace", "kind": "mood", "content": random.choice(MEME_GRUMPS)}
    yield {"type": "trace", "kind": "tool", "name": "llm.plan", "payload": {"model": llm_model, "query": payload.query}}
    plan = await _plan_retrieval(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)

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

    search_query = payload.query
    if use_hyde:
        yield {"type": "trace", "kind": "thought", "label": "HyDE", "content": "Generating a hypothetical passage."}
        yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"model": llm_model, "query": payload.query}}
        hyde = await _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)
        search_query = hyde.strip() or payload.query

    yield {
        "type": "trace",
        "kind": "tool",
        "name": "gate.chat",
        "payload": {"query": search_query, "retrieval_mode": mode, "top_k": top_k, "rerank": rerank},
    }
    primary = await _gate_search(
        client,
        gate_url,
        search_query,
        mode,
        top_k,
        rerank,
        filters,
        payload.include_sources,
        timeout_s=gate_timeout,
    )

    responses = [primary]
    hits = list(primary.get("hits") or [])
    context_chunks = list(primary.get("context") or [])
    degraded = set(primary.get("degraded") or [])
    partial = bool(primary.get("partial"))

    if _quality_is_poor(primary):
        yield {"type": "trace", "kind": "thought", "label": "Quality", "content": "Search looks weak. Splitting into fact queries."}
        yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"model": llm_model, "query": payload.query}}
        fact_qs = await _fact_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)
        if fact_qs:
            for fq in fact_qs:
                yield {
                    "type": "trace",
                    "kind": "action",
                    "content": f"Fact query: {fq}",
                }
                yield {
                    "type": "trace",
                    "kind": "tool",
                    "name": "gate.chat",
                    "payload": {"query": fq, "retrieval_mode": mode, "top_k": max(4, top_k // 2), "rerank": rerank},
                }
                r2 = await _gate_search(
                    client,
                    gate_url,
                    fq,
                    mode,
                    max(4, top_k // 2),
                    rerank,
                    filters,
                    payload.include_sources,
                    timeout_s=gate_timeout,
                )
                responses.append(r2)
                degraded.update(r2.get("degraded") or [])
                partial = partial or bool(r2.get("partial"))

            hits = _merge_hits(responses, cap=max(12, top_k))
            context_chunks = _context_from_hits(hits, context_chunks)
        else:
            yield {"type": "trace", "kind": "thought", "label": "Quality", "content": "No useful fact queries found."}

    if not context_chunks:
        context_chunks = _context_from_hits(hits, context_chunks)

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

    context_text = _build_context(hits, limit=8, max_chars=4000)
    if not context_text and context_chunks:
        context_text = _build_context(context_chunks, limit=8, max_chars=4000)
    if not context_text:
        yield {"type": "error", "error": "no_context"}
        return

    answer_lang = await _detect_language(
        client,
        llm_base,
        llm_model,
        llm_key,
        payload.query,
        timeout_s=llm_timeout,
    )
    system = {
        "role": "system",
        "content": (
            "You answer using the provided context only. "
            f"If the context is insufficient, say what is missing. Reply in {answer_lang}."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Question:\n{payload.query}\n\nContext:\n{context_text}\n\n"
            "Answer in the same language as the question."
        ),
    }

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
    sources = primary.get("sources") or _sources_from_context(context_chunks)

    assessment = await _assess_answer_completeness(
        client,
        llm_base,
        llm_model,
        llm_key,
        payload.query,
        full_answer,
        timeout_s=llm_timeout,
    )

    if assessment.get("incomplete"):
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
        yield {
            "type": "trace",
            "kind": "tool",
            "name": "llm.hyde",
            "payload": {"model": llm_model, "query": payload.query},
        }
        hyde_retry = await _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)
        if hyde_retry.strip():
            retry_query = hyde_retry.strip()

        responses = []
        queries = [retry_query]
        for term in missing_terms:
            if term and term.lower() not in retry_query.lower():
                queries.append(f"{payload.query} {term}")

        yield {"type": "trace", "kind": "tool", "name": "llm.keywords", "payload": {"model": llm_model, "query": payload.query}}
        keyword_qs = await _keyword_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)
        for kw in keyword_qs:
            if kw and kw.lower() not in retry_query.lower():
                queries.append(kw)

        for q in queries:
            yield {
                "type": "trace",
                "kind": "tool",
                "name": "gate.chat",
                "payload": {
                    "query": q,
                    "retrieval_mode": retry_mode,
                    "top_k": retry_top_k,
                    "rerank": retry_rerank,
                },
            }
            r0 = await _gate_search(
                client,
                gate_url,
                q,
                retry_mode,
                retry_top_k,
                retry_rerank,
                filters,
                payload.include_sources,
                timeout_s=gate_timeout,
            )
            responses.append(r0)

        retry_resp = responses[0] if responses else {}

        retry_hits = list(retry_resp.get("hits") or [])
        retry_context = list(retry_resp.get("context") or [])
        retry_degraded = set(retry_resp.get("degraded") or [])
        retry_partial = bool(retry_resp.get("partial"))

        yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"model": llm_model, "query": payload.query}}
        fact_qs = await _fact_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout)
        if fact_qs:
            responses = [retry_resp]
            for fq in fact_qs:
                yield {"type": "trace", "kind": "action", "content": f"Fact query: {fq}"}
                r2 = await _gate_search(
                    client,
                    gate_url,
                    fq,
                    retry_mode,
                    max(4, retry_top_k // 2),
                    retry_rerank,
                    filters,
                    payload.include_sources,
                    timeout_s=gate_timeout,
                )
                responses.append(r2)
                retry_degraded.update(r2.get("degraded") or [])
                retry_partial = retry_partial or bool(r2.get("partial"))

            retry_hits = _merge_hits(responses, cap=max(12, retry_top_k))
            retry_context = _context_from_hits(retry_hits, retry_context)

        if not retry_context:
            retry_context = _context_from_hits(retry_hits, retry_context)

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

        retry_context_text = _build_context(retry_hits, limit=8, max_chars=4000)
        if not retry_context_text and retry_context:
            retry_context_text = _build_context(retry_context, limit=8, max_chars=4000)

        if retry_context_text:
            system = {
                "role": "system",
                "content": (
                    "You answer using the provided context only. "
                    f"If the context is insufficient, say what is missing. Reply in {answer_lang}."
                ),
            }
            user = {
                "role": "user",
                "content": (
                    f"Question:\n{payload.query}\n\nContext:\n{retry_context_text}\n\n"
                    "Answer in the same language as the question."
                ),
            }
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
            sources = retry_resp.get("sources") or _sources_from_context(retry_context)
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


@app.post("/v1/agent/stream")
async def agent_stream(payload: AgentRequest):
    async def event_stream() -> AsyncIterator[str]:
        async with httpx.AsyncClient() as client:
            try:
                async for event in _run_agent(payload, client):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
