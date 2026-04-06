"""
Agent-search service: LLM-driven retrieval with plan, HyDE, fact queries, retry.

Flow: plan → retrieval.search → quality check → optional fact queries → answer → assess → retry if incomplete.
Endpoints: POST /v1/agent/stream (SSE), POST /v1/agent (JSON).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from typing import Any, AsyncIterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from agent_common.retrieval_client import AsyncRetrievalClient
from agent_common import (
    ANSWER_SYSTEM,
    ANSWER_SYSTEM_FACTOID,
    ANSWER_SYSTEM_WITH_TOOLS,
    ANSWER_SYSTEM_WITH_TOOLS_FACTOID,
    ANSWER_USER,
    ANSWER_USER_FACTOID,
    strip_thinking,
    FACT_QUERIES_SYSTEM,
    FACT_QUERIES_USER,
    FACTOID_REWRITE_SYSTEM,
    FACTOID_REWRITE_USER,
    HYDE_SYSTEM,
    HYDE_USER,
    run_calculator,
    run_execute_code,
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
# Answer stream: cap total time for LLM answer phase; floor so short llm_timeout does not cut too early
ANSWER_STREAM_CAP_S = 120
ANSWER_STREAM_MIN_S = 90
ANSWER_STREAM_LLM_MULTIPLIER = 2

# Prometheus metrics
AGENT_REQS = Counter("agent_requests_total", "Agent requests", ["endpoint", "status"])
AGENT_LAT = Histogram(
    "agent_request_duration_seconds",
    "Request duration (seconds)",
    ["endpoint"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 90, 120),
)
AGENT_LLM_CALLS = Counter("agent_llm_calls_total", "LLM calls per request", ["stage"])
AGENT_RETRIEVAL_CALLS = Counter("agent_retrieval_calls_total", "Retrieval calls per request")
AGENT_RETRY = Counter("agent_retry_total", "Retry triggered (incomplete answer)")

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_FACTOID_LEAD_RE = re.compile(
    r"^\s*(who|what|which|where|when|how many|how much|кто|что|какой|какая|какие|где|когда|сколько)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[\d+\]")
_LEADING_ANSWER_RE = re.compile(r"^\s*(answer|final answer)\s*:\s*", re.IGNORECASE)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9А-ЯЁ\"\[])")


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


class RetrievalFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    project_ids: list[str] | None = None


class AgentRequest(BaseModel):
    """Request body for agent search (plan → retrieval → quality check → fact queries → answer)."""

    query: str = Field(..., description="User question")
    history: list[dict[str, str]] = Field(default_factory=list, description="Conversation history (role, content)")
    filters: RetrievalFilters | None = Field(None, description="Retrieval filters")
    include_sources: bool = Field(True, description="Include sources in response")
    top_k: int | None = Field(None, description="Override retrieval top_k (5..24), else use plan")
    use_adaptive_k: bool | None = Field(None, description="Cut at steepest score drop (adaptive-k)")
    max_llm_calls: int | None = Field(None, description="Max LLM calls per request")
    max_fact_queries: int | None = Field(None, description="Max fact queries per request")
    use_hyde: bool | None = Field(None, description="Enable HyDE (override env)")
    hyde_num: int | None = Field(None, description="Number of HyDE variants to merge (1=single, 3=multi)")
    use_fact_queries: bool | None = Field(None, description="Enable fact queries (override env)")
    use_retry: bool | None = Field(None, description="Enable retry on incomplete (override env)")
    use_tools: bool | None = Field(None, description="Enable calculator & code execution tools")
    mode: str | None = Field(None, description="Preset: minimal | conservative | aggressive")
    answer_style: str | None = Field(None, description="Answer style: default | factoid | auto")


AGENT_MODE_PRESETS = {
    "minimal": {"use_hyde": False, "use_fact_queries": False, "use_retry": False, "max_llm_calls": 4, "max_fact_queries": 0},
    "conservative": {"use_hyde": False, "use_fact_queries": False, "use_retry": False, "max_llm_calls": 6, "max_fact_queries": 0},
    "aggressive": {"use_hyde": True, "use_fact_queries": True, "use_retry": True, "max_llm_calls": 16, "max_fact_queries": 4},
}


app = FastAPI(
    title="Agent Search",
    version="0.1.0",
    description="LLM-driven retrieval: plan → retrieval.search → quality check → fact queries → answer. Supports feature flags and mode presets.",
    openapi_url="/v1/openapi.json",
    docs_url="/v1/docs",
)


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


async def _llm_chat_with_tools(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    max_tool_rounds: int = 5,
) -> str:
    """Call LLM with tools; execute tool calls and loop until final answer."""
    url = base_url.rstrip("/") + "/chat/completions"
    msgs = list(messages)
    for _ in range(max_tool_rounds):
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": "auto",
        }
        resp = await client.post(url, json=payload, headers=_llm_headers(api_key), timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        msgs.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls or []})
        if not tool_calls:
            return (content or "").strip()
        for tc in tool_calls:
            fid = tc.get("id", "")
            fn = (tc.get("function") or {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            if name == "calculator":
                out = run_calculator(args.get("expression", ""))
                result = str(out.get("result")) if out.get("ok") else f"Error: {out.get('error', 'unknown')}"
            elif name == "execute_code":
                out = run_execute_code(args.get("code", ""))
                result = out.get("stdout", "") or (f"stderr: {out.get('stderr', '')}" if not out.get("ok") else "ok")
            else:
                result = "Unknown tool"
            msgs.append({
                "role": "tool",
                "tool_call_id": fid,
                "content": result,
            })
    return ""


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
    # Many models (e.g. reasoning) wrap JSON in <think>...</think>; strip it before parse
    raw_clean = strip_thinking(raw or "").strip()
    try:
        return json.loads(raw_clean)
    except Exception:
        # Fallback if no valid JSON (e.g. model returned plain text)
        return {"retrieval_mode": "hybrid", "top_k": 10, "rerank": True, "use_hyde": False, "reason": "fallback"}


# Tool definitions for OpenAI-compatible API
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, log, sin, cos, pi, e.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression, e.g. 2+2*3 or sqrt(16)"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python code in sandbox. No imports, no file I/O. Use for list comprehensions, data transforms.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to run, e.g. print(sum(range(10)))"}},
                "required": ["code"],
            },
        },
    },
]


async def _make_hyde(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    timeout_s: float,
    lang_hint: str = "English",
    temperature: float = 0.2,
) -> str:
    system = {"role": "system", "content": HYDE_SYSTEM.format(lang=lang_hint)}
    user = {"role": "user", "content": HYDE_USER.format(query=query, lang=lang_hint)}
    return await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [system, user],
        temperature=temperature,
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
    return {**event, "trace_id": trace_id}


def _is_factoid_like_question(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    if len(_TOKEN_RE.findall(text)) > 14:
        return False
    if "?" in text:
        return True
    return bool(_FACTOID_LEAD_RE.search(text))


def _resolve_answer_style(payload: AgentRequest) -> str:
    style = (payload.answer_style or "").strip().lower()
    if style in {"default", "factoid"}:
        return style
    if style not in {"", "auto"}:
        return "default"
    if not payload.history and _is_factoid_like_question(payload.query):
        return "factoid"
    return "default"


def _postprocess_factoid_answer(text: str) -> str:
    answer = strip_thinking(text or "").strip()
    if not answer:
        return ""
    citations = _CITATION_RE.findall(answer)
    answer = _LEADING_ANSWER_RE.sub("", answer)
    answer = answer.splitlines()[0].strip()
    parts = _SENTENCE_BREAK_RE.split(answer, maxsplit=1)
    if parts:
        answer = parts[0].strip()
    if citations and not _CITATION_RE.search(answer):
        answer = answer.rstrip(" .,:;") + f" {citations[0]}"
    return answer.strip()


def _build_answer_messages(
    *,
    answer_style: str,
    use_tools: bool,
    answer_lang: str,
    history_text: str,
    query: str,
    context_text: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if answer_style == "factoid":
        system_content = (
            ANSWER_SYSTEM_WITH_TOOLS_FACTOID.format(lang=answer_lang)
            if use_tools
            else ANSWER_SYSTEM_FACTOID.format(lang=answer_lang)
        )
        user_content = ANSWER_USER_FACTOID.format(history=history_text, query=query, context=context_text)
    else:
        system_content = ANSWER_SYSTEM_WITH_TOOLS.format(lang=answer_lang) if use_tools else ANSWER_SYSTEM.format(lang=answer_lang)
        user_content = ANSWER_USER.format(history=history_text, query=query, context=context_text)
    return {"role": "system", "content": system_content}, {"role": "user", "content": user_content}


async def _rewrite_factoid_answer(
    client: httpx.AsyncClient,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    draft_answer: str,
    context_text: str,
    timeout_s: float,
) -> str:
    rewritten = await _llm_chat(
        client,
        llm_base,
        llm_model,
        llm_key,
        [
            {"role": "system", "content": FACTOID_REWRITE_SYSTEM},
            {
                "role": "user",
                "content": FACTOID_REWRITE_USER.format(query=query, draft=draft_answer, context=context_text),
            },
        ],
        temperature=0.0,
        timeout_s=timeout_s,
    )
    return _postprocess_factoid_answer(rewritten or draft_answer)


async def _retrieval_search_once(
    retrieval: "AsyncRetrievalClient",
    client: httpx.AsyncClient,
    search_query: str,
    *,
    mode: str,
    top_k: int,
    rerank: bool,
    use_adaptive_k: bool | None,
    filters: dict | None,
    include_sources: bool,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Returns (result, None) or (None, error_event)."""
    try:
        result = await asyncio.wait_for(
            retrieval.search(
                client,
                search_query,
                retrieval_mode=mode,
                top_k=top_k,
                rerank=rerank,
                use_adaptive_k=use_adaptive_k,
                filters=filters,
                include_sources=include_sources,
            ),
            timeout=timeout_s,
        )
        return (result, None)
    except asyncio.TimeoutError:
        return (None, {"type": "error", "error": "Retrieval timeout. Check AGENT_RETRIEVAL_URL and retrieval service."})
    except Exception as e:
        return (None, {"type": "error", "error": f"Retrieval error: {e!s}"})


async def _run_agent(payload: AgentRequest, client: httpx.AsyncClient) -> AsyncIterator[dict[str, Any]]:
    # Apply mode preset (conservative/aggressive/minimal) — explicit payload fields override
    preset = AGENT_MODE_PRESETS.get((payload.mode or "").lower()) if payload.mode else {}
    max_llm_calls = payload.max_llm_calls if payload.max_llm_calls is not None else preset.get("max_llm_calls")
    max_llm_calls = max_llm_calls if max_llm_calls is not None else int(_env_get("AGENT_MAX_LLM_CALLS", "12"))
    use_hyde_payload = payload.use_hyde if payload.use_hyde is not None else preset.get("use_hyde")
    use_fact_queries_payload = payload.use_fact_queries if payload.use_fact_queries is not None else preset.get("use_fact_queries")
    use_retry_payload = payload.use_retry if payload.use_retry is not None else preset.get("use_retry")
    use_tools = payload.use_tools if payload.use_tools is not None else _env_get("AGENT_USE_TOOLS", "false").lower() in ("1", "true", "yes")
    max_fact_queries = payload.max_fact_queries if payload.max_fact_queries is not None else preset.get("max_fact_queries")
    max_fact_queries = max_fact_queries if max_fact_queries is not None else int(_env_get("AGENT_MAX_FACT_QUERIES", "2"))

    retrieval_url = _env_get(
        "AGENT_RETRIEVAL_URL",
        _env_get("GATE_RETRIEVAL_URL", _env_get("PROCESSOR_RETRIEVAL_URL", "http://retrieval:8080")),
    )
    llm_base = _env_get("AGENT_LLM_BASE_URL", _env_get("GATE_LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model = _env_get("AGENT_LLM_MODEL", _env_get("GATE_LLM_MODEL", "gpt-4o-mini"))
    llm_key = os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("GATE_LLM_API_KEY")
    llm_timeout = float(_env_get("AGENT_LLM_TIMEOUT_S", "60"))
    retrieval_timeout = float(_env_get("AGENT_RETRIEVAL_TIMEOUT_S", "60"))
    llm_calls = [0]

    def _can_llm() -> bool:
        llm_calls[0] += 1
        return llm_calls[0] <= max_llm_calls

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None
    retrieval = AsyncRetrievalClient(retrieval_url, timeout_s=retrieval_timeout)

    hist = [m for m in payload.history] if payload.history else None
    answer_style = _resolve_answer_style(payload)
    logger.info("agent yielding mood then plan tool")
    yield {"type": "trace", "kind": "mood", "content": random.choice(MEME_GRUMPS)}
    yield {"type": "trace", "kind": "tool", "name": "llm.plan", "payload": {"model": llm_model, "query": payload.query}}
    if _can_llm():
        AGENT_LLM_CALLS.labels(stage="plan").inc()
        try:
            logger.info("agent calling _plan_retrieval (LLM)")
            plan = await asyncio.wait_for(
                _plan_retrieval(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist),
                timeout=min(40, llm_timeout + 10),
            )
            logger.info("agent plan received mode=%s", plan.get("retrieval_mode"))
        except asyncio.TimeoutError:
            yield {"type": "error", "error": "LLM plan timeout. Check AGENT_LLM_BASE_URL and API availability."}
            return
        except Exception as e:
            yield {"type": "error", "error": f"Plan error: {e!s}"}
            return
    else:
        plan = {"retrieval_mode": "hybrid", "top_k": 10, "rerank": True, "use_hyde": False, "reason": "max_llm_calls"}

    mode = str(plan.get("retrieval_mode") or "hybrid")
    top_k_raw = int(payload.top_k) if payload.top_k is not None else int(plan.get("top_k") or 10)
    top_k_min = int(_env_get("AGENT_TOP_K_MIN", "5"))
    top_k_max = int(_env_get("AGENT_TOP_K_MAX", "24"))
    top_k = max(top_k_min, min(top_k_max, top_k_raw))
    rerank = bool(plan.get("rerank") if plan.get("rerank") is not None else True)
    use_adaptive_k = payload.use_adaptive_k
    use_hyde_plan = bool(plan.get("use_hyde") or False)
    if use_hyde_payload is not None:
        use_hyde = use_hyde_payload
    elif _env_get("AGENT_USE_HYDE", "").lower() in ("1", "true", "yes"):
        use_hyde = True
    elif _env_get("AGENT_USE_HYDE", "").lower() in ("0", "false", "no"):
        use_hyde = False
    else:
        use_hyde = use_hyde_plan
    reason = str(plan.get("reason") or "no_reason")

    yield {
        "type": "trace",
        "kind": "thought",
        "label": "Plan",
        "content": f"mode={mode}, top_k={top_k}, rerank={rerank}, hyde={use_hyde}, answer_style={answer_style}. Reason: {reason}",
    }

    if _can_llm():
        AGENT_LLM_CALLS.labels(stage="detect_lang").inc()
        answer_lang = await _detect_language(
            client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout
        )
    else:
        answer_lang = "English"

    hyde_num = payload.hyde_num if payload.hyde_num is not None else int(_env_get("AGENT_HYDE_NUM", "1"))
    hyde_num = max(1, min(7, hyde_num))

    if use_hyde and _can_llm():
        if hyde_num > 1:
            yield {"type": "trace", "kind": "thought", "label": "HyDE", "content": f"Generating {hyde_num} hypothetical passages in {answer_lang}."}
            temps = [0.2 + 0.15 * i for i in range(hyde_num)][:hyde_num]
            hyde_tasks = [
                _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, lang_hint=answer_lang, temperature=t)
                for t in temps
            ]
            for _ in hyde_tasks:
                AGENT_LLM_CALLS.labels(stage="hyde").inc()
            hyde_passages = await asyncio.gather(*hyde_tasks)
            search_queries = [h.strip() or payload.query for h in hyde_passages if h.strip()]
            if not search_queries:
                search_queries = [payload.query]
            yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"model": llm_model, "query": payload.query, "lang": answer_lang, "num": len(search_queries)}}
            retrieval_tasks = [
                retrieval.search(
                    client,
                    q,
                    retrieval_mode=mode,
                    top_k=top_k,
                    rerank=rerank,
                    use_adaptive_k=use_adaptive_k,
                    filters=filters,
                    include_sources=payload.include_sources,
                )
                for q in search_queries
            ]
            hyde_responses = await asyncio.gather(*retrieval_tasks)
            AGENT_RETRIEVAL_CALLS.inc(len(hyde_responses))
            merged_hits = merge_hits(hyde_responses, cap=max(12, top_k))
            all_ctx = []
            for r in hyde_responses:
                all_ctx.extend(r.get("context") or [])
            primary = {
                "hits": merged_hits,
                "context": context_from_hits(merged_hits, all_ctx),
                "partial": any(r.get("partial") for r in hyde_responses),
                "degraded": [],
                "sources": hyde_responses[0].get("sources") or [] if hyde_responses else [],
            }
            for r in hyde_responses:
                primary["degraded"].extend(r.get("degraded") or [])
        else:
            yield {"type": "trace", "kind": "thought", "label": "HyDE", "content": f"Generating hypothetical passage in {answer_lang}."}
            yield {"type": "trace", "kind": "tool", "name": "llm.hyde", "payload": {"model": llm_model, "query": payload.query, "lang": answer_lang}}
            hyde = await _make_hyde(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, lang_hint=answer_lang)
            search_query = hyde.strip() or payload.query
            yield {"type": "trace", "kind": "tool", "name": "retrieval.search", "payload": {"query": search_query, "retrieval_mode": mode, "top_k": top_k, "rerank": rerank}}
            AGENT_RETRIEVAL_CALLS.inc()
            primary, err = await _retrieval_search_once(
                retrieval, client, search_query,
                mode=mode, top_k=top_k, rerank=rerank, use_adaptive_k=use_adaptive_k,
                filters=filters, include_sources=payload.include_sources,
                timeout_s=retrieval_timeout + 15,
            )
            if err:
                yield err
                return
    else:
        search_query = payload.query
        yield {"type": "trace", "kind": "tool", "name": "retrieval.search", "payload": {"query": search_query, "retrieval_mode": mode, "top_k": top_k, "rerank": rerank}}
        AGENT_RETRIEVAL_CALLS.inc()
        primary, err = await _retrieval_search_once(
            retrieval, client, search_query,
            mode=mode, top_k=top_k, rerank=rerank, use_adaptive_k=use_adaptive_k,
            filters=filters, include_sources=payload.include_sources,
            timeout_s=retrieval_timeout + 15,
        )
        if err:
            yield err
            return

    min_hits = int(_env_get("AGENT_QUALITY_MIN_HITS", "3"))
    min_score = float(_env_get("AGENT_QUALITY_MIN_SCORE", "0.15"))
    always_fact = _env_get("AGENT_ALWAYS_FACT_QUERIES", "false").lower() in ("1", "true", "yes")
    use_fact_env = _env_get("AGENT_USE_FACT_QUERIES", "true").lower() not in ("0", "false", "no")
    if use_fact_queries_payload is False:
        run_fact = False
    elif use_fact_queries_payload is True:
        run_fact = use_fact_env and _can_llm()
    else:
        run_fact = use_fact_env and (quality_is_poor(primary, min_hits=min_hits, min_score=min_score) or always_fact) and _can_llm()
    responses = [primary]
    hits = list(primary.get("hits") or [])
    context_chunks = list(primary.get("context") or [])
    degraded = set(primary.get("degraded") or [])
    partial = bool(primary.get("partial"))
    if run_fact:
        AGENT_LLM_CALLS.labels(stage="fact_split").inc()
        msg = "Search looks weak. Splitting into fact queries." if quality_is_poor(primary, min_hits=min_hits, min_score=min_score) else "Multi-query fusion: adding fact queries."
        yield {"type": "trace", "kind": "thought", "label": "Quality", "content": msg}
        yield {"type": "trace", "kind": "tool", "name": "llm.fact_split", "payload": {"model": llm_model, "query": payload.query}}
        fact_qs = await _fact_queries(client, llm_base, llm_model, llm_key, payload.query, timeout_s=llm_timeout, history=hist)
        fact_qs = fact_qs[:max_fact_queries] if fact_qs else []
        if fact_qs:
            for fq in fact_qs:
                yield {"type": "trace", "kind": "action", "content": f"Fact query: {fq}"}
                yield {"type": "trace", "kind": "tool", "name": "retrieval.search", "payload": {"query": fq, "retrieval_mode": mode, "top_k": max(4, top_k // 2), "rerank": rerank}}
            fact_tasks = [
                retrieval.search(
                    client,
                    fq,
                    retrieval_mode=mode,
                    top_k=max(4, top_k // 2),
                    rerank=rerank,
                    use_adaptive_k=use_adaptive_k,
                    filters=filters,
                    include_sources=payload.include_sources,
                )
                for fq in fact_qs
            ]
            fact_responses = await asyncio.gather(*fact_tasks)
            AGENT_RETRIEVAL_CALLS.inc(len(fact_responses))
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
        if use_tools:
            context_text = "[No relevant documents. Use calculator for math, execute_code for code.]"
        else:
            yield {"type": "error", "error": "no_context"}
            return

    hist_ctx = _history_summary(hist) if hist else ""
    system, user = _build_answer_messages(
        answer_style=answer_style,
        use_tools=use_tools,
        answer_lang=answer_lang,
        history_text=hist_ctx,
        query=payload.query,
        context_text=context_text,
    )
    answer_temperature = 0.0 if answer_style == "factoid" else 0.2

    AGENT_LLM_CALLS.labels(stage="answer").inc()
    if use_tools:
        yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": False, "tools": True}}
        full_answer = await _llm_chat_with_tools(
            client,
            llm_base,
            llm_model,
            llm_key,
            [system, user],
            AGENT_TOOLS,
            temperature=answer_temperature,
            timeout_s=llm_timeout,
        )
        full_answer = strip_thinking(full_answer)
        for ch in full_answer:
            yield {"type": "token", "content": ch}
    else:
        stream_answer = answer_style != "factoid"
        yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": stream_answer}}
        if stream_answer:
            answer_parts: list[str] = []
            answer_stream_deadline = time.monotonic() + min(
                ANSWER_STREAM_CAP_S,
                max(llm_timeout * ANSWER_STREAM_LLM_MULTIPLIER, ANSWER_STREAM_MIN_S),
            )
            try:
                async for chunk in _llm_chat_stream(
                    client,
                    llm_base,
                    llm_model,
                    llm_key,
                    [system, user],
                    temperature=answer_temperature,
                    timeout_s=llm_timeout,
                ):
                    if time.monotonic() > answer_stream_deadline:
                        break
                    answer_parts.append(chunk)
                    yield {"type": "token", "content": chunk}
            except (asyncio.TimeoutError, httpx.TimeoutException):
                pass
            full_answer = strip_thinking("".join(answer_parts))
        else:
            full_answer = await _llm_chat(
                client,
                llm_base,
                llm_model,
                llm_key,
                [system, user],
                temperature=answer_temperature,
                timeout_s=llm_timeout,
            )
        if not full_answer.strip():
            full_answer = "[Answer generation timed out or produced no text. Try a shorter query or check the LLM endpoint.]"
    if answer_style == "factoid":
        full_answer = _postprocess_factoid_answer(full_answer)
        if full_answer and full_answer != "Insufficient context." and _can_llm():
            AGENT_LLM_CALLS.labels(stage="answer_rewrite").inc()
            full_answer = await _rewrite_factoid_answer(
                client,
                llm_base,
                llm_model,
                llm_key,
                payload.query,
                full_answer,
                context_text,
                timeout_s=llm_timeout,
            )
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

    use_retry = use_retry_payload if use_retry_payload is not None else _env_get("AGENT_USE_RETRY", "true").lower() in ("1", "true", "yes")
    if assessment.get("incomplete") and _can_llm() and use_retry:
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
            yield {"type": "trace", "kind": "tool", "name": "retrieval.search", "payload": {"query": q, "retrieval_mode": retry_mode, "top_k": retry_top_k, "rerank": retry_rerank}}
        retry_tasks = [
            retrieval.search(
                client,
                q,
                retrieval_mode=retry_mode,
                top_k=retry_top_k,
                rerank=retry_rerank,
                use_adaptive_k=use_adaptive_k,
                filters=filters,
                include_sources=payload.include_sources,
            )
            for q in queries
        ]
        retry_responses = await asyncio.gather(*retry_tasks)
        AGENT_RETRIEVAL_CALLS.inc(len(retry_responses))

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
            fact_qs = fact_qs[:max_fact_queries] if fact_qs else []
        if fact_qs:
            for fq in fact_qs:
                yield {"type": "trace", "kind": "action", "content": f"Fact query: {fq}"}
                yield {"type": "trace", "kind": "tool", "name": "retrieval.search", "payload": {"query": fq, "retrieval_mode": retry_mode, "top_k": max(4, retry_top_k // 2), "rerank": retry_rerank}}
            fact_retry_tasks = [
                retrieval.search(
                    client,
                    fq,
                    retrieval_mode=retry_mode,
                    top_k=max(4, retry_top_k // 2),
                    rerank=retry_rerank,
                    use_adaptive_k=use_adaptive_k,
                    filters=filters,
                    include_sources=payload.include_sources,
                )
                for fq in fact_qs
            ]
            fact_retry_responses = await asyncio.gather(*fact_retry_tasks)
            AGENT_RETRIEVAL_CALLS.inc(len(fact_retry_responses))
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
            system, user = _build_answer_messages(
                answer_style=answer_style,
                use_tools=use_tools,
                answer_lang=answer_lang,
                history_text=hist_ctx,
                query=payload.query,
                context_text=retry_context_text,
            )
            answer_temperature = 0.0 if answer_style == "factoid" else 0.2
            AGENT_LLM_CALLS.labels(stage="answer_retry").inc()
            if use_tools:
                yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": False, "tools": True}}
                full_answer = await _llm_chat_with_tools(
                    client,
                    llm_base,
                    llm_model,
                    llm_key,
                    [system, user],
                    AGENT_TOOLS,
                    temperature=answer_temperature,
                    timeout_s=llm_timeout,
                )
            else:
                yield {"type": "trace", "kind": "tool", "name": "llm.answer", "payload": {"model": llm_model, "stream": False}}
                full_answer = await _llm_chat(
                    client,
                    llm_base,
                    llm_model,
                    llm_key,
                    [system, user],
                    temperature=answer_temperature,
                    timeout_s=llm_timeout,
                )
            if answer_style == "factoid":
                full_answer = _postprocess_factoid_answer(full_answer)
                if full_answer and full_answer != "Insufficient context." and _can_llm():
                    AGENT_LLM_CALLS.labels(stage="answer_rewrite_retry").inc()
                    full_answer = await _rewrite_factoid_answer(
                        client,
                        llm_base,
                        llm_model,
                        llm_key,
                        payload.query,
                        full_answer,
                        retry_context_text,
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


@app.get("/v1/metrics", tags=["metrics"])
async def metrics():
    """Prometheus metrics (requests, latency, LLM/retrieval calls)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/agent", tags=["agent"])
async def agent_non_streaming(payload: AgentRequest):
    """Non-streaming: collect events and return full response as JSON. Use mode=minimal|conservative|aggressive for presets."""
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


@app.post("/v1/agent/stream", tags=["agent"])
async def agent_stream(request: Request, payload: AgentRequest):
    t0 = time.perf_counter()
    trace_id = uuid.uuid4().hex
    timeout_s = float(_env_get("AGENT_REQUEST_TIMEOUT_S", "120"))

    async def event_stream() -> AsyncIterator[str]:
        deadline = asyncio.get_event_loop().time() + timeout_s
        stream_error: str | None = None
        try:
            yield f"data: {json.dumps(_with_trace_id({'type': 'init', 'trace_id': trace_id}, trace_id))}\n\n"
            await asyncio.sleep(0)
            yield f"data: {json.dumps(_with_trace_id({'type': 'trace', 'kind': 'thought', 'label': 'Start', 'content': 'Preparing plan...'}, trace_id))}\n\n"
            await asyncio.sleep(0)
            async with httpx.AsyncClient() as client:
                try:
                    agent_it = _run_agent(payload, client)
                    next_timeout_s = float(_env_get("AGENT_STREAM_NEXT_TIMEOUT_S", "65"))
                    while True:
                        try:
                            event = await asyncio.wait_for(
                                agent_it.__anext__(),
                                timeout=min(next_timeout_s, max(5, deadline - asyncio.get_event_loop().time())),
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            stream_error = "request_timeout"
                            yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': 'Timeout: no response from LLM/services. Check AGENT_LLM_BASE_URL and network.'}, trace_id))}\n\n"
                            return
                        if await request.is_disconnected():
                            stream_error = "client_disconnected"
                            return
                        if asyncio.get_event_loop().time() > deadline:
                            stream_error = "request_timeout"
                            yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': 'request_timeout'}, trace_id))}\n\n"
                            return
                        yield f"data: {json.dumps(_with_trace_id(event, trace_id))}\n\n"
                        await asyncio.sleep(0)  # flush chunk to client
                except asyncio.CancelledError:
                    stream_error = "client_disconnected"
                    raise
                except asyncio.TimeoutError:
                    stream_error = "request_timeout"
                    yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': 'request_timeout'}, trace_id))}\n\n"
                except Exception as exc:
                    stream_error = str(exc)
                    logger.exception("agent stream error")
                    yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': str(exc)}, trace_id))}\n\n"
        except asyncio.CancelledError:
            stream_error = "client_disconnected"
        except Exception as exc:
            # Catch-all: ensure stream doesn't silently close (UI would spin forever)
            stream_error = str(exc)
            logger.exception("agent stream fatal error")
            try:
                yield f"data: {json.dumps(_with_trace_id({'type': 'error', 'error': str(exc)}, trace_id))}\n\n"
            except Exception:
                # If we can't write to the client (already disconnected), just exit.
                pass
        finally:
            AGENT_LAT.labels(endpoint="/v1/agent/stream").observe(time.perf_counter() - t0)
            AGENT_REQS.labels(endpoint="/v1/agent/stream", status="ok" if not stream_error else "error").inc()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
