"""
Deep-research service: LangGraph-based iterative research with structured report.

Flow: plan → scope (plan + queries) → research loop (batch Gate calls, distilled notes,
next_queries) → early stop on min gain → write (streaming report).
Scope/research fallback: empty queries → use question.
Endpoint: POST /v1/deep-research/stream (SSE).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import time
import logging
from typing import Any, Dict, List, TypedDict, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from agent_common import (
    DEEP_FACT_QUERIES,
    DEEP_HYDE,
    DEEP_KEYWORD_QUERIES,
    SyncGateClient,
    build_context,
    context_from_hits,
    merge_hits,
    quality_is_poor,
    sources_from_context,
)
from agent_common.web_search import web_search_sync


_LOG_LEVEL = os.environ.get("DEEP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger("deep-research")


DEFAULT_REPORT_TEMPLATE = (
    "Report template:\n"
    "1) Executive summary (3-6 bullets)\n"
    "2) Sections mapped to the plan (short paragraphs + bullet facts)\n"
    "3) Open questions and risks\n"
    "4) Sources list\n"
)


class GateFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    project_ids: list[str] | None = None


class DeepResearchRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: GateFilters | None = None
    include_sources: bool = True
    max_iterations: int = 2
    retrieval_mode: str | None = None
    top_k: int | None = None
    rerank: bool | None = None
    use_web_search: bool | None = None
    web_search_num: int | None = None
    web_search_timeout_s: float | None = None


class ResearchState(TypedDict):
    question: str
    plan: List[str]
    queries: List[str]
    notes: List[str]
    sources: List[Dict[str, Any]]
    context: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    report: str
    filters: Dict[str, Any] | None
    include_sources: bool
    retrieval_mode: str
    top_k: int
    rerank: bool
    plan_reason: str
    history: List[Dict[str, str]] | None


class ScopeOutput(BaseModel):
    plan: List[str] = Field(description="Short research plan")
    queries: List[str] = Field(description="Initial retrieval queries")


class ResearchStepOutput(BaseModel):
    next_queries: List[str] = Field(description="Follow-up queries")
    distilled_notes: List[str] = Field(description="Short, verifiable notes")


class RetrievalPlanOutput(BaseModel):
    retrieval_mode: str = Field(description="bm25|vector|hybrid")
    top_k: int = Field(description="1..40")
    rerank: bool = Field(description="true/false")
    reason: str | None = Field(default=None, description="Short reasoning")


class MCPToolConfig(BaseModel):
    method: str = "POST"
    path: str
    description: str | None = None


class MCPServerConfig(BaseModel):
    name: str
    base_url: str
    tools: dict[str, MCPToolConfig]


class EventEmitter:
    def __init__(self, event_queue: queue.Queue[object]) -> None:
        self._queue = event_queue

    def emit(self, event: dict[str, Any]) -> None:
        if event.get("type") in {"progress", "trace", "retrieval"}:
            logger.info("event.emit", extra={"type": event.get("type"), "stage": event.get("stage")})
        self._queue.put(event)


SENTINEL = object()


def _env_get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _build_llm():
    provider = os.environ.get("DEEP_LLM_PROVIDER") or os.environ.get("GATE_LLM_PROVIDER") or ""
    model = (
        os.environ.get("DEEP_LLM_MODEL")
        or os.environ.get("LLM_MODEL")
        or os.environ.get("GATE_LLM_MODEL")
        or "gpt-4o-mini"
    )
    temp = float(_env_get("DEEP_LLM_TEMPERATURE", "0"))
    base_url = os.environ.get("DEEP_LLM_BASE_URL") or os.environ.get("GATE_LLM_BASE_URL")
    api_key = os.environ.get("DEEP_LLM_API_KEY") or os.environ.get("GATE_LLM_API_KEY") or ""

    if model and ":" in model:
        logger.info("llm.init provider=auto model=%s base_url=%s key=%s", model, base_url or "-", "set" if api_key else "unset")
        return init_chat_model(model, temperature=temp)

    provider_norm = provider.lower().strip()
    if provider_norm in {"openai", "openai_compat", "openai-compatible", "openai_compatible"}:
        logger.info("llm.init provider=%s model=%s base_url=%s key=%s", provider_norm, model, base_url or "default", "set" if api_key else "unset")
        return ChatOpenAI(model=model, temperature=temp, base_url=base_url, api_key=api_key or "EMPTY")

    if provider_norm:
        candidate = f"{provider_norm}:{model}"
        try:
            logger.info("llm.init provider=%s model=%s base_url=%s key=%s", provider_norm, candidate, base_url or "-", "set" if api_key else "unset")
            return init_chat_model(candidate, temperature=temp)
        except Exception:
            logger.warning("llm.init fallback provider=%s model=%s", provider_norm, model)
            return init_chat_model(f"openai:{model}", temperature=temp)

    logger.info("llm.init provider=openai model=%s base_url=%s key=%s", model, base_url or "default", "set" if api_key else "unset")
    return init_chat_model(f"openai:{model}", temperature=temp)


def _safe_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _llm_text(llm, prompt: str) -> str:
    try:
        result = llm.invoke(prompt)
    except Exception:
        logger.exception("llm.invoke failed")
        return ""
    if hasattr(result, "content"):
        return str(result.content or "")
    return str(result or "")


def _llm_json(llm, prompt: str) -> dict[str, Any]:
    raw = _llm_text(llm, prompt)
    if not raw:
        return {}
    return _safe_json(raw)


def _default_plan(question: str) -> list[str]:
    if question:
        return [
            "Definition and scope",
            "Architecture and components",
            "Key capabilities and limitations",
            "Operational considerations",
            "Ecosystem and alternatives",
        ]
    return ["Background", "Key findings", "Implications", "Risks", "Next steps"]


def _plan_retrieval(llm, question: str) -> dict[str, Any]:
    prompt = (
        "You are a retrieval strategist for a RAG system. Return JSON only.\n"
        "Fields: retrieval_mode (bm25|vector|hybrid), top_k (1..40), rerank (true|false), use_hyde (true|false), reason.\n"
        f"Question: {question}"
    )
    data = _llm_json(llm, prompt)
    return {
        "retrieval_mode": str(data.get("retrieval_mode") or "hybrid"),
        "top_k": int(data.get("top_k") or 8),
        "rerank": bool(data.get("rerank") if data.get("rerank") is not None else True),
        "use_hyde": bool(data.get("use_hyde") if data.get("use_hyde") is not None else False),
        "reason": str(data.get("reason") or ""),
    }


def _infer_query_lang(query: str) -> str:
    """Heuristic: infer language from script for HyDE generation."""
    q = (query or "").strip()
    if not q:
        return "English"
    # Cyrillic (Russian, Ukrainian, etc.)
    if any("\u0400" <= c <= "\u04FF" for c in q):
        return "Russian"
    # CJK
    if any("\u4E00" <= c <= "\u9FFF" or "\u3040" <= c <= "\u30FF" for c in q):
        return "the same language as the query"
    return "English"


def _make_hyde(llm, query: str) -> str:
    lang = _infer_query_lang(query)
    prompt = DEEP_HYDE.format(query=query, lang=lang)
    return _llm_text(llm, prompt)


def _fact_queries(llm, query: str) -> list[str]:
    prompt = DEEP_FACT_QUERIES.format(query=query)
    data = _llm_json(llm, prompt)
    out = data.get("fact_queries") or []
    return [str(q).strip() for q in out if str(q).strip()]


def _keyword_queries(llm, query: str) -> list[str]:
    prompt = DEEP_KEYWORD_QUERIES.format(query=query)
    data = _llm_json(llm, prompt)
    out = data.get("keywords") or []
    return [str(q).strip() for q in out if str(q).strip()]



def _load_mcp_config() -> MCPServerConfig:
    default_path = os.path.join(os.path.dirname(__file__), "mcp_gate.json")
    cfg_path = os.environ.get("DEEP_MCP_CONFIG", default_path)
    data: dict[str, Any] = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    base_url = os.environ.get("DEEP_GATE_URL") or os.environ.get("GATE_URL") or data.get("base_url")
    if not base_url:
        base_url = "http://rag-gate:8090"
    data["base_url"] = base_url
    data.setdefault("name", "rag-gate")
    data.setdefault("tools", {"chat": {"method": "POST", "path": "/v1/chat"}})
    return MCPServerConfig(**data)


def _merge_sources(existing: list[dict[str, Any]], new_sources: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for source in existing + new_sources:
        doc_id = source.get("doc_id")
        uri = source.get("uri")
        key = (doc_id, uri)
        if not doc_id or key in seen:
            continue
        seen.add(key)
        out.append(source)
        if len(out) >= cap:
            break
    return out


def _merge_context(existing: list[dict[str, Any]], new_context: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for ctx in existing + new_context:
        cid = ctx.get("chunk_id")
        key = cid or id(ctx)
        if key in seen:
            continue
        seen.add(key)
        out.append(ctx)
        if len(out) >= cap:
            break
    return out


def _normalize_context(data: dict[str, Any]) -> list[dict[str, Any]]:
    context = list(data.get("context") or [])
    retrieval = data.get("retrieval") or {}
    hits = list(retrieval.get("hits") or [])
    if not context and hits:
        context = [
            {
                "chunk_id": h.get("chunk_id"),
                "doc_id": h.get("doc_id"),
                "score": h.get("score"),
                "text": h.get("text"),
                "source": h.get("source"),
            }
            for h in hits
        ]
    sources = data.get("sources") or []
    by_doc = {str(s.get("doc_id")): s for s in sources if s.get("doc_id")}
    for chunk in context:
        if chunk.get("source"):
            continue
        doc_id = chunk.get("doc_id")
        if doc_id is not None and str(doc_id) in by_doc:
            chunk["source"] = by_doc[str(doc_id)]
    return context


def _documents_from_context(
    context: list[dict[str, Any]],
    query: str,
    max_docs: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    out = []
    for chunk in context:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        src = chunk.get("source") or {}
        doc_id = str(chunk.get("doc_id") or src.get("doc_id") or "")
        title = src.get("title") or doc_id
        uri = src.get("uri")
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        out.append(
            {
                "query": query,
                "doc_id": doc_id,
                "title": title,
                "uri": uri,
                "content": text,
            }
        )
        if len(out) >= max_docs:
            break
    return out


def _dedupe_queries(queries: list[str], cap: int) -> list[str]:
    seen = set()
    out = []
    for q in queries:
        q_norm = q.strip()
        if not q_norm:
            continue
        key = q_norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q_norm)
        if len(out) >= cap:
            break
    return out


def _build_graph(
    llm,
    gate: SyncGateClient,
    emitter: EventEmitter,
    template: str,
) -> Any:
    logger.info("graph.build.enter")
    def emit_progress(stage: str, iteration: int, max_iterations: int, message: str) -> None:
        safe_max = max(1, max_iterations)
        if stage == "plan":
            percent = 0.02
        elif stage == "scope":
            percent = 0.05
        elif stage == "research":
            percent = 0.05 + 0.85 * min(iteration, safe_max) / safe_max
        elif stage == "write":
            percent = 0.95
        elif stage == "done":
            percent = 1.0
        else:
            percent = 0.0
        emitter.emit(
            {
                "type": "progress",
                "stage": stage,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "percent": round(percent, 4),
                "message": message,
            }
        )

    def plan_node(state: dict[str, Any]) -> dict[str, Any]:
        max_iters = int(state.get("max_iterations") or 0)
        emit_progress("plan", 0, max_iters, "Plan: choosing retrieval settings")
        logger.info("plan.start")
        defaults = {
            "retrieval_mode": state.get("retrieval_mode") or "hybrid",
            "top_k": int(state.get("top_k") or 8),
            "rerank": bool(state.get("rerank") if state.get("rerank") is not None else True),
            "use_hyde": bool(state.get("use_hyde") if state.get("use_hyde") is not None else False),
            "reason": "",
        }
        try:
            data = _plan_retrieval(llm, state.get("question") or "")
            mode = str(data.get("retrieval_mode") or defaults["retrieval_mode"])
            top_k = int(data.get("top_k") or defaults["top_k"])
            rerank = bool(data.get("rerank") if data.get("rerank") is not None else defaults["rerank"])
            use_hyde = bool(data.get("use_hyde") if data.get("use_hyde") is not None else defaults["use_hyde"])
            reason = str(data.get("reason") or "")
        except Exception:
            logger.exception("plan.failed")
            mode = str(defaults["retrieval_mode"])
            top_k = int(defaults["top_k"])
            rerank = bool(defaults["rerank"])
            use_hyde = bool(defaults["use_hyde"])
            reason = ""

        if mode not in {"bm25", "vector", "hybrid"}:
            mode = str(defaults["retrieval_mode"])
        top_k = max(1, min(40, int(top_k)))

        emitter.emit(
            {
                "type": "trace",
                "kind": "thought",
                "label": "Retrieval plan",
                "content": f"mode={mode}, top_k={top_k}, rerank={rerank}, hyde={use_hyde}. {reason}".strip(),
            }
        )
        logger.info("plan.done", extra={"mode": mode, "top_k": top_k, "rerank": rerank, "hyde": use_hyde})

        return {
            "retrieval_mode": mode,
            "top_k": top_k,
            "rerank": rerank,
            "use_hyde": use_hyde,
            "plan_reason": reason,
            "question": state.get("question") or "",  # preserve for downstream
        }

    def scope_node(state: dict[str, Any]) -> dict[str, Any]:
        max_iters = int(state.get("max_iterations") or 0)
        emit_progress("scope", 0, max_iters, "Scope: drafting plan")
        logger.info("scope.start")
        emitter.emit(
            {
                "type": "trace",
                "kind": "thought",
                "label": "Scope",
                "content": "Drafting a plan and initial queries.",
            }
        )
        prompt = (
            "You are a deep research planner. Build a concise plan and initial queries.\n"
            f"Question: {state.get('question') or ''}\n\n"
            "Requirements:\n"
            "1) Plan: 5-10 bullets.\n"
            "2) Queries: 6-10 specific queries with key entities, versions, dates, or comparisons.\n\n"
            f"Report template:\n{template}\n"
        )
        try:
            out = llm.with_structured_output(ScopeOutput).invoke(prompt)
            if out is None:
                raw = llm.invoke(prompt).content
                logger.warning("scope.raw_fallback", extra={"chars": len(str(raw))})
                data = _safe_json(str(raw))
                plan = list(data.get("plan") or [])
                queries = list(data.get("queries") or [])
            else:
                plan = list(out.plan or [])
                queries = list(out.queries or [])
        except Exception:
            logger.exception("scope.failed")
            plan = []
            queries = []
        question = (state.get("question") or "").strip()
        logger.info("scope.state_question=%s", question[:80] if question else "(empty)")
        if not plan:
            plan = _default_plan(question)
        if not queries:
            queries = [question] if question else []
        queries = _dedupe_queries(queries, cap=12)
        if not queries and question:
            queries = [question]
        logger.info("scope.queries=%s", queries)
        logger.info("scope.done", extra={"plan_items": len(plan), "queries": len(queries)})
        return {
            "plan": plan,
            "queries": queries,
            "notes": [],
            "sources": [],
            "context": [],
            "iteration": 0,
            "question": (question or state.get("question") or "").strip(),  # preserve for research (prefer local)
            "web_search_num": state.get("web_search_num"),
            "web_search_timeout_s": state.get("web_search_timeout_s"),
            "use_web_search": state.get("use_web_search"),
        }

    def research_node(state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration") or 0) + 1
        max_iters = int(state.get("max_iterations") or 0)
        emit_progress(
            "research",
            iteration,
            max_iters,
            f"Research iteration {iteration}/{max_iters}",
        )
        logger.info("research.start", extra={"iteration": iteration, "remaining": len(state.get("queries", [])), "question": (state.get("question") or "")[:50]})
        batch_size = int(_env_get("DEEP_RESEARCH_BATCH", "5"))
        question = (state.get("question") or state.get("initial_question") or "").strip()
        # Explicit fallback: filter empty, fallback to question when scope/merge yields empty queries
        queries_list = [q for q in (state.get("queries") or []) if q and str(q).strip()]
        if not queries_list and question:
            queries_list = [question]
            logger.warning("research.fallback_queries from question", extra={"question": question[:80]})
        elif not queries_list:
            logger.error("research.no_queries_no_question")
        batch = list(queries_list[:batch_size])
        remaining = list(queries_list[batch_size:])
        logger.info("research.batch=%s", batch)
        new_context: list[dict[str, Any]] = []
        new_sources: list[dict[str, Any]] = []
        gathered_docs: list[dict[str, Any]] = []

        if not batch:
            emitter.emit(
                {
                    "type": "trace",
                    "kind": "thought",
                    "label": "Queries",
                    "content": "No queries to run in this iteration.",
                }
            )
            # Fallback: when no queries but web search enabled, run web search with question
            web_provider = _env_get("WEB_SEARCH_PROVIDER", "").lower().strip()
            web_key = os.environ.get("WEB_SEARCH_API_KEY") or os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
            _req_web = state.get("use_web_search")
            _web_enabled = _req_web is not False and (_req_web is True or _env_get("DEEP_USE_WEB_SEARCH", "false").lower() in ("1", "true", "yes"))
            if question and web_provider in ("serper", "tavily") and (web_key or "").strip() and _web_enabled:
                web_num = int(state.get("web_search_num") or _env_get("WEB_SEARCH_NUM", "5"))
                web_timeout = float(state.get("web_search_timeout_s") or _env_get("WEB_SEARCH_TIMEOUT_S", "15"))
                emitter.emit({"type": "trace", "kind": "tool", "name": "web.search", "payload": {"query": question, "provider": web_provider, "num": web_num}})
                web_hits = web_search_sync(question, provider=web_provider, api_key=web_key, num=web_num, timeout_s=web_timeout)
                if web_hits:
                    max_docs_val = int(_env_get("DEEP_RESEARCH_MAX_DOCS", "8"))
                    max_chars_val = int(_env_get("DEEP_RESEARCH_MAX_CHARS", "4000"))
                    for i, h in enumerate(web_hits[:max_docs_val]):
                        src = h.get("source") or {}
                        gathered_docs.append({
                            "query": question,
                            "doc_id": h.get("doc_id", ""),
                            "title": src.get("title", ""),
                            "uri": src.get("uri", ""),
                            "content": (h.get("text") or "")[:max_chars_val],
                        })
                        new_sources.append({"ref": i + 1, "doc_id": h.get("doc_id", ""), "title": src.get("title", ""), "uri": src.get("uri", ""), "locator": None})
                    emitter.emit({"type": "trace", "kind": "action", "content": f"Web search: {len(web_hits)} results (no gate queries)"})

        max_docs = int(_env_get("DEEP_RESEARCH_MAX_DOCS", "8"))
        max_chars = int(_env_get("DEEP_RESEARCH_MAX_CHARS", "4000"))
        mode = state.get("retrieval_mode") or "hybrid"
        top_k = int(state.get("top_k") or 8)
        rerank = bool(state.get("rerank") if state.get("rerank") is not None else True)
        use_hyde = bool(state.get("use_hyde") if state.get("use_hyde") is not None else False)
        filters = state.get("filters")
        include_sources = state.get("include_sources", True)

        for q in batch:
            search_query = q
            if use_hyde:
                emitter.emit(
                    {
                        "type": "trace",
                        "kind": "thought",
                        "label": "HyDE",
                        "content": "Generating a hypothetical passage for retrieval.",
                    }
                )
                hyde = _make_hyde(llm, q)
                if hyde.strip():
                    search_query = hyde.strip()
            emitter.emit(
                {
                    "type": "trace",
                    "kind": "tool",
                    "name": "mcp.gate.chat",
                    "payload": {
                        "query": search_query,
                        "retrieval_mode": mode,
                        "top_k": top_k,
                        "rerank": rerank,
                    },
                }
            )
            history = state.get("history") or []
            primary = gate.chat(
                search_query,
                history=history,
                retrieval_mode=mode,
                top_k=top_k,
                rerank=rerank,
                filters=filters,
                include_sources=include_sources,
            )

            responses = [primary]
            hits = list(primary.get("hits") or [])
            context_chunks = list(primary.get("context") or [])
            degraded = set(primary.get("degraded") or [])
            partial = bool(primary.get("partial"))

            if quality_is_poor(primary):
                emitter.emit(
                    {
                        "type": "trace",
                        "kind": "thought",
                        "label": "Quality",
                        "content": "Search looks weak. Expanding into fact queries.",
                    }
                )
                fact_qs = _fact_queries(llm, q)
                for fq in fact_qs:
                    emitter.emit(
                        {
                            "type": "trace",
                            "kind": "action",
                            "content": f"Fact query: {fq}",
                        }
                    )
                    emitter.emit(
                        {
                            "type": "trace",
                            "kind": "tool",
                            "name": "mcp.gate.chat",
                            "payload": {"query": fq, "retrieval_mode": mode, "top_k": max(4, top_k // 2), "rerank": rerank},
                        }
                    )
                    r2 = gate.chat(
                        fq,
                        history=history,
                        retrieval_mode=mode,
                        top_k=max(4, top_k // 2),
                        rerank=rerank,
                        filters=filters,
                        include_sources=include_sources,
                    )
                    responses.append(r2)
                    degraded.update(r2.get("degraded") or [])
                    partial = partial or bool(r2.get("partial"))

                if mode != "hybrid":
                    emitter.emit(
                        {
                            "type": "trace",
                            "kind": "tool",
                            "name": "mcp.gate.chat",
                            "payload": {"query": search_query, "retrieval_mode": "hybrid", "top_k": top_k, "rerank": rerank},
                        }
                    )
                    alt = gate.chat(
                        search_query,
                        history=history,
                        retrieval_mode="hybrid",
                        top_k=top_k,
                        rerank=rerank,
                        filters=filters,
                        include_sources=include_sources,
                    )
                    responses.append(alt)
                    degraded.update(alt.get("degraded") or [])
                    partial = partial or bool(alt.get("partial"))
                for alt_mode in ("bm25", "vector"):
                    if alt_mode == mode:
                        continue
                    emitter.emit(
                        {
                            "type": "trace",
                            "kind": "tool",
                            "name": "mcp.gate.chat",
                            "payload": {"query": search_query, "retrieval_mode": alt_mode, "top_k": top_k, "rerank": rerank},
                        }
                    )
                    alt = gate.chat(
                        search_query,
                        history=history,
                        retrieval_mode=alt_mode,
                        top_k=top_k,
                        rerank=rerank,
                        filters=filters,
                        include_sources=include_sources,
                    )
                    responses.append(alt)
                    degraded.update(alt.get("degraded") or [])
                    partial = partial or bool(alt.get("partial"))

                hits = merge_hits(responses, cap=max(12, top_k))
                context_chunks = context_from_hits(hits, context_chunks)

                if not hits:
                    keyword_qs = _keyword_queries(llm, q)
                    for kw in keyword_qs:
                        emitter.emit(
                            {
                                "type": "trace",
                                "kind": "tool",
                                "name": "mcp.gate.chat",
                                "payload": {"query": kw, "retrieval_mode": mode, "top_k": max(4, top_k // 2), "rerank": rerank},
                            }
                        )
                        r3 = gate.chat(
                            kw,
                            history=history,
                            retrieval_mode=mode,
                            top_k=max(4, top_k // 2),
                            rerank=rerank,
                            filters=filters,
                            include_sources=include_sources,
                        )
                        responses.append(r3)
                    hits = merge_hits(responses, cap=max(12, top_k))
                    context_chunks = context_from_hits(hits, context_chunks)

            web_provider = _env_get("WEB_SEARCH_PROVIDER", "").lower().strip()
            web_key = os.environ.get("WEB_SEARCH_API_KEY") or os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
            _req_web = state.get("use_web_search")
            _web_enabled = _req_web is not False and (_req_web is True or _env_get("DEEP_USE_WEB_SEARCH", "false").lower() in ("1", "true", "yes"))
            if web_provider in ("serper", "tavily") and (web_key or "").strip() and _web_enabled:
                web_query = (state.get("question") or q or "").strip()
                web_num = int(state.get("web_search_num") or _env_get("WEB_SEARCH_NUM", "5"))
                web_timeout = float(state.get("web_search_timeout_s") or _env_get("WEB_SEARCH_TIMEOUT_S", "15"))
                if web_query:
                    emitter.emit({"type": "trace", "kind": "tool", "name": "web.search", "payload": {"query": web_query, "provider": web_provider, "num": web_num}})
                web_hits = web_search_sync(web_query, provider=web_provider, api_key=web_key, num=web_num, timeout_s=web_timeout)
                if web_hits:
                    responses.append({"hits": web_hits})
                    hits = merge_hits(responses, cap=max(16, top_k + len(web_hits)))
                    context_chunks = context_from_hits(hits, context_chunks)
                    emitter.emit({"type": "trace", "kind": "action", "content": f"Web search: {len(web_hits)} results merged"})

            if not context_chunks:
                context_chunks = context_from_hits(hits, context_chunks)

            sources = primary.get("sources") or sources_from_context(context_chunks)
            docs = _documents_from_context(context_chunks, q, max_docs=max_docs, max_chars=max_chars)

            new_context = _merge_context(new_context, context_chunks, cap=max_docs)
            new_sources = _merge_sources(new_sources, sources, cap=max_docs)
            gathered_docs.extend(docs)

            emitter.emit(
                {
                    "type": "trace",
                    "kind": "thought",
                    "label": "Retrieval stats",
                    "content": f"hits={len(hits)} context={len(context_chunks)} partial={partial} degraded={len(degraded)}",
                }
            )

            emitter.emit(
                {
                    "type": "retrieval",
                    "mode": mode,
                    "partial": partial,
                    "degraded": list(degraded),
                    "context": context_chunks,
                    "sources": sources,
                }
            )

        joined = "\n\n".join(
            f"QUERY: {d['query']}\nTITLE: {d['title']}\nURI: {d.get('uri') or '-'}\nCONTENT:\n{d['content']}"
            for d in gathered_docs
        )
        context_text = build_context(new_context, limit=8, max_chars=4000)
        prompt = (
            "You are doing deep research using retrieved context. Return JSON only.\n"
            f"Question: {state.get('question') or ''}\n\n"
            "Plan:\n"
            + "\n".join(f"- {p}" for p in state.get("plan") or [])
            + "\n\n"
            "Context:\n"
            + context_text
            + "\n\nNew sources:\n"
            + joined
            + "\n\nProvide JSON with:\n"
            "distilled_notes: 8-16 short, verifiable notes based on sources.\n"
            "next_queries: 4-8 follow-up queries to fill gaps and validate facts.\n"
        )

        try:
            step_out = llm.with_structured_output(ResearchStepOutput).invoke(prompt)
            if step_out is None:
                raw = llm.invoke(prompt).content
                logger.warning("research.raw_fallback", extra={"iteration": iteration, "chars": len(str(raw))})
                data = _safe_json(str(raw))
                notes = list(data.get("distilled_notes") or [])
                next_queries = list(data.get("next_queries") or [])
            else:
                notes = list(step_out.distilled_notes or [])
                next_queries = list(step_out.next_queries or [])
        except Exception:
            logger.exception("research.summarize_failed", extra={"iteration": iteration})
            notes = []
            next_queries = []

        merged_sources = _merge_sources(state.get("sources", []), new_sources, cap=50)
        merged_context = _merge_context(state.get("context", []), new_context, cap=50)
        merged_queries = _dedupe_queries(remaining + next_queries, cap=20)

        logger.info(
            "research.done",
            extra={
                "iteration": iteration,
                "notes": len(notes),
                "next_queries": len(next_queries),
                "sources": len(merged_sources),
            },
        )
        notes_added = len(notes)
        sources_added = len(new_sources)
        return {
            "iteration": iteration,
            "sources": merged_sources,
            "context": merged_context,
            "notes": state.get("notes", []) + notes,
            "queries": merged_queries,
            "notes_added": notes_added,
            "sources_added": sources_added,
        }

    def should_continue(state: dict[str, Any]) -> str:
        if int(state.get("iteration") or 0) >= int(state.get("max_iterations") or 0):
            return "write"
        if not state.get("queries"):
            return "write"
        min_gain = int(_env_get("DEEP_EARLY_STOP_MIN_GAIN", "2"))
        notes_added = int(state.get("notes_added") or 0)
        sources_added = int(state.get("sources_added") or 0)
        if notes_added < min_gain and sources_added < min_gain and int(state.get("iteration") or 0) > 0:
            logger.info("early_stop", extra={"notes_added": notes_added, "sources_added": sources_added})
            return "write"
        return "research"

    def write_node(state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration") or 0)
        max_iters = int(state.get("max_iterations") or 0)
        emit_progress("write", iteration, max_iters, "Write: drafting report")
        logger.info("write.start", extra={"notes": len(state.get("notes", [])), "sources": len(state.get("sources", []))})
        emitter.emit(
            {
                "type": "trace",
                "kind": "thought",
                "label": "Write",
                "content": "Drafting the report.",
            }
        )
        notes = list(state.get("notes") or [])
        sources = list(state.get("sources") or [])
        question = state.get("question") or ""
        if not notes and not sources:
            report = (
                f"Research request: {question}\n\n"
                "No sources were found from the gate retrieval for this query. "
                "Upload or index documents that mention the topic, then retry deep research.\n\n"
                "Sources:\n- (none)\n"
            )
            logger.warning("write.no_sources")
            return {"report": report}
        source_lines = []
        for src in state.get("sources", []):
            uri = src.get("uri")
            doc_id = src.get("doc_id")
            if uri:
                source_lines.append(uri)
            elif doc_id:
                source_lines.append(str(doc_id))
        prompt = (
            "Write a structured report that follows the template.\n"
            f"Question: {state.get('question') or ''}\n\n"
            "Plan:\n"
            + "\n".join(f"- {p}" for p in state.get("plan") or [])
            + "\n\n"
            "Notes:\n"
            + "\n".join(f"- {n}" for n in state.get("notes", []))
            + "\n\n"
            "Sources (for citation):\n"
            + "\n".join(f"- {s}" for s in source_lines[:50])
            + "\n\n"
            "Requirements:\n"
            "- Use only the notes as facts.\n"
            "- Call out uncertainties explicitly.\n"
            f"- Follow this template:\n{template}\n"
        )
        report_parts: list[str] = []
        for chunk in llm.stream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                emitter.emit({"type": "token", "content": content})
                report_parts.append(content)
        report = "".join(report_parts).strip()
        if source_lines:
            report += "\n\nSources:\n" + "\n".join(f"- {s}" for s in source_lines[:50])
        logger.info("write.done", extra={"report_chars": len(report)})
        return {"report": report}

    graph = StateGraph(dict)
    logger.info("graph.build.nodes")
    logger.info("graph.add_node plan start")
    graph.add_node("plan", plan_node)
    logger.info("graph.add_node plan done")
    logger.info("graph.add_node scope start")
    graph.add_node("scope", scope_node)
    logger.info("graph.add_node scope done")
    logger.info("graph.add_node research start")
    graph.add_node("research", research_node)
    logger.info("graph.add_node research done")
    logger.info("graph.add_node write start")
    graph.add_node("write", write_node)
    logger.info("graph.add_node write done")

    logger.info("graph.build.edges")
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "scope")
    graph.add_edge("scope", "research")
    graph.add_conditional_edges("research", should_continue, {"research": "research", "write": "write"})
    graph.add_edge("write", END)

    logger.info("graph.compile.start")
    compiled = graph.compile()
    logger.info("graph.compile.done")
    return compiled


def _run_graph(payload: DeepResearchRequest, event_queue: queue.Queue[object]) -> ResearchState:
    emitter = EventEmitter(event_queue)
    template = os.environ.get("DEEP_RESEARCH_TEMPLATE", DEFAULT_REPORT_TEMPLATE)
    llm = _build_llm()

    mcp_config = _load_mcp_config()
    gate_timeout = float(_env_get("DEEP_GATE_TIMEOUT_S", "60"))
    gate = SyncGateClient(mcp_config.base_url, timeout_s=gate_timeout)
    logger.info(
        "graph.init gate_url=%s max_iterations=%s include_sources=%s question=%s",
        mcp_config.base_url,
        payload.max_iterations,
        payload.include_sources,
        payload.query,
    )
    emitter.emit(
        {
            "type": "progress",
            "stage": "init",
            "iteration": 0,
            "max_iterations": payload.max_iterations,
            "percent": 0.01,
            "message": "Init",
        }
    )

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None
    max_iterations = payload.max_iterations or int(_env_get("DEEP_MAX_ITERATIONS", "2"))
    retrieval_mode = payload.retrieval_mode or _env_get("DEEP_RETRIEVAL_MODE", "hybrid")
    top_k = payload.top_k or int(_env_get("DEEP_TOP_K", "8"))
    rerank = payload.rerank if payload.rerank is not None else _env_get("DEEP_RERANK", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    web_num = payload.web_search_num if payload.web_search_num is not None else int(_env_get("WEB_SEARCH_NUM", "5"))
    web_timeout = payload.web_search_timeout_s if payload.web_search_timeout_s is not None else float(_env_get("WEB_SEARCH_TIMEOUT_S", "15"))
    use_web_search = (
        payload.use_web_search
        if payload.use_web_search is not None
        else _env_get("DEEP_USE_WEB_SEARCH", "false").lower() in {"1", "true", "yes"}
    )

    initial_state: dict[str, Any] = {
        "question": payload.query,
        "initial_question": payload.query,  # never overwritten, fallback when question is lost
        "plan": [],
        "queries": [payload.query],
        "notes": [],
        "sources": [],
        "context": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "report": "",
        "filters": filters,
        "include_sources": payload.include_sources,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "rerank": rerank,
        "use_hyde": False,
        "web_search_num": web_num,
        "web_search_timeout_s": web_timeout,
        "use_web_search": use_web_search,
        "plan_reason": "",
        "history": list(payload.history) if payload.history else [],
    }

    logger.info("graph.build.start initial_queries=%s", initial_state.get("queries"))
    graph = _build_graph(llm, gate, emitter, template)
    logger.info("graph.build.done")
    emitter.emit(
        {
            "type": "progress",
            "stage": "compiled",
            "iteration": 0,
            "max_iterations": payload.max_iterations,
            "percent": 0.03,
            "message": "Graph compiled",
        }
    )
    try:
        logger.info("graph.invoke.start")
        result = graph.invoke(initial_state)
        logger.info("graph.invoke.done")
        logger.info("graph.done", extra={"iteration": result.get("iteration", 0)})
        return result
    finally:
        event_queue.put(SENTINEL)


app = FastAPI(title="Deep Research", version="0.1.0")


@app.get("/v1/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.post("/v1/deep-research/stream")
async def deep_research_stream(payload: DeepResearchRequest) -> StreamingResponse:
    if not (payload.query or "").strip():
        raise HTTPException(status_code=400, detail="query is required")
    async def event_stream() -> AsyncIterator[str]:
        event_queue: queue.Queue[object] = queue.Queue()
        logger.info("stream.open")
        task = asyncio.create_task(asyncio.to_thread(_run_graph, payload, event_queue))

        while True:
            event = await asyncio.to_thread(event_queue.get)
            if event is SENTINEL:
                break
            yield f"data: {json.dumps(event)}\n\n"
        logger.info("stream.queue.done")

        try:
            state = await task
        except Exception as exc:
            logger.exception("stream.failed", extra={"error": str(exc)})
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            return

        report = state.get("report") or ""
        yield f"data: {json.dumps({'type': 'progress', 'stage': 'done', 'iteration': state.get('iteration', 0), 'max_iterations': state.get('max_iterations', 0), 'percent': 1.0, 'message': 'Done'})}\n\n"

        done_event = {
            "type": "done",
            "answer": report,
            "sources": state.get("sources") if payload.include_sources else [],
            "context": state.get("context"),
            "partial": False,
            "degraded": [],
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
