"""FastAPI routers for /chat, /chat/stream, /agent, /agent/stream, /execute.

The pipeline returns a PipelineResult. Endpoints own:
- retry logic (calling run_pipeline again with a new plan)
- streaming the final answer to the client
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import tiktoken

from api.presets import AGENT_PRESET_BUILDERS, agent, retry_round, simple
from config import Settings
from engine.brain_pipeline import run_pipeline
from clients.llm import LLMClient
from models import (
    AgentRequest,
    AgentResponse,
    BrainRound,
    ChatRequest,
    ChatResponse,
    ExecuteRequest,
    PipelineResult,
)
from models.events import DoneEvent, ErrorEvent, Event, InitEvent, TokenEvent, TraceEvent

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])
agent_router = APIRouter(tags=["agent"])
execute_router = APIRouter(tags=["execute"])


# ── SSE helpers ──────────────────────────────────────────

def _event_to_dict(event: Event) -> dict[str, Any]:
    return dataclasses.asdict(event)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _pipeline_kwargs(request: Request) -> dict[str, Any]:
    return dict(
        llm=request.app.state.llm,
        http_client=request.app.state.http_client,
        settings=request.app.state.settings,
    )


def _done_event(result: PipelineResult, *, trace_id: str = "") -> DoneEvent:
    return DoneEvent(
        answer=result.answer,
        mode=result.mode,
        partial=False,
        degraded=[],
        sources=result.sources,
        context=[
            {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
            for c in result.chunks[:10]
        ],
    )


def _stream_answer(
    result: PipelineResult, *, trace_id: str, include_traces: bool = False,
) -> AsyncIterator[str]:
    """Stream traces, then the answer as token events, then a done event."""
    async def _generate() -> AsyncIterator[str]:
        rng = random.Random(trace_id)
        encoding = tiktoken.get_encoding("o200k_base")

        yield _sse(_event_to_dict(InitEvent(trace_id=trace_id)))

        if include_traces:
            for trace in result.traces:
                ev = TraceEvent(**trace)
                d = _event_to_dict(ev)
                d["trace_id"] = trace_id
                yield _sse(d)

        token_ids = encoding.encode(result.answer)
        i = 0
        chunk_index = 0
        while i < len(token_ids):
            token_count = rng.choices([1, 2, 3, 4], weights=[0.2, 0.45, 0.25, 0.1], k=1)[0]
            chunk = encoding.decode(token_ids[i : i + token_count])
            i += token_count
            chunk_index += 1

            # Smooth token cadence with subtle punctuation pauses.
            delay = rng.uniform(0.010, 0.026)
            if chunk.rstrip().endswith((".", "!", "?")):
                delay += rng.uniform(0.018, 0.055)
            await asyncio.sleep(delay)

            d = _event_to_dict(TokenEvent(content=chunk))
            d["trace_id"] = trace_id
            yield _sse(d)

        d = _event_to_dict(_done_event(result))
        d["trace_id"] = trace_id
        yield _sse(d)

    return _generate()


# ── /chat ────────────────────────────────────────────────

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    settings: Settings = request.app.state.settings
    plan = simple(
        preset=body.preset,
        top_k=body.top_k,
        rerank=body.rerank,
        reflection_enabled=body.reflection_enabled and settings.reflection_enabled,
    )
    kwargs = _pipeline_kwargs(request)

    result: PipelineResult | None = None
    retries_used = 0

    try:
        for attempt in range(1 + body.max_retries):
            result = await run_pipeline(
                plan,
                project_id=body.project_id, query=body.query,
                include_sources=True, **kwargs,
            )
            if not result.needs_retry or attempt >= body.max_retries:
                break
            retries_used += 1
    except Exception:
        logger.exception("Chat pipeline failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Generation pipeline error")

    if result is None:
        raise HTTPException(status_code=502, detail="No pipeline result")

    return ChatResponse(
        answer=result.answer,
        sources=[s.get("source_id", "") for s in result.sources if isinstance(s, dict)],
        chunks_used=len(result.chunks),
        retries_used=retries_used,
        query=body.query,
    )


# ── /chat/stream (SSE) ───────────────────────────────────

@chat_router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    settings: Settings = request.app.state.settings
    trace_id = uuid.uuid4().hex
    plan = simple(
        preset=body.preset,
        top_k=body.top_k,
        rerank=body.rerank,
        reflection_enabled=body.reflection_enabled and settings.reflection_enabled,
    )
    kwargs = _pipeline_kwargs(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            result: PipelineResult | None = None
            retries_used = 0

            for attempt in range(1 + body.max_retries):
                result = await asyncio.wait_for(
                    run_pipeline(
                        plan,
                        project_id=body.project_id, query=body.query,
                        include_sources=True, **kwargs,
                    ),
                    timeout=120.0,
                )
                if not result.needs_retry or attempt >= body.max_retries:
                    break
                retries_used += 1

            if result is None:
                yield _sse({"type": "error", "error": "no_pipeline_result", "trace_id": trace_id})
                return

            async for chunk in _stream_answer(result, trace_id=trace_id, include_traces=False):
                if await request.is_disconnected():
                    return
                yield chunk

        except asyncio.TimeoutError:
            yield _sse({"type": "error", "error": "request_timeout", "trace_id": trace_id})
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Chat stream error")
            try:
                yield _sse({"type": "error", "error": str(exc), "trace_id": trace_id})
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /agent (non-streaming) ───────────────────────────────

def _build_agent_plan(body: AgentRequest, settings: Settings) -> BrainRound:
    mode = (body.mode or "").lower()
    if mode in AGENT_PRESET_BUILDERS:
        return AGENT_PRESET_BUILDERS[mode]()

    return agent(
        use_hyde=body.use_hyde if body.use_hyde is not None else settings.agent_use_hyde,
        use_fact_queries=body.use_fact_queries if body.use_fact_queries is not None else settings.agent_use_fact_queries,
        use_retry=body.use_retry if body.use_retry is not None else settings.agent_use_retry,
        use_tools=body.use_tools if body.use_tools is not None else settings.agent_use_tools,
        max_llm_calls=body.max_llm_calls or settings.agent_max_llm_calls,
        max_fact_queries=body.max_fact_queries or settings.agent_max_fact_queries,
        top_k=body.top_k or 10,
        rerank=True,
    )


@agent_router.post("/agent", response_model=AgentResponse)
async def agent_non_streaming(body: AgentRequest, request: Request):
    settings: Settings = request.app.state.settings
    trace_id = uuid.uuid4().hex
    plan = _build_agent_plan(body, settings)
    kwargs = _pipeline_kwargs(request)

    try:
        result = await asyncio.wait_for(
            run_pipeline(
                plan,
                project_id=body.project_id, query=body.query,
                history=body.history, include_sources=body.include_sources,
                **kwargs,
            ),
            timeout=120.0,
        )

        if result.needs_retry and settings.agent_use_retry:
            retry_query = result.requery or body.query
            if result.missing_terms:
                suffix = " ".join(t for t in result.missing_terms if t.lower() not in retry_query.lower())
                if suffix:
                    retry_query = f"{retry_query} {suffix}"
            rplan = retry_round(top_k=body.top_k or 10)
            result = await asyncio.wait_for(
                run_pipeline(
                    rplan,
                    project_id=body.project_id,
                    query=retry_query,
                    history=body.history, include_sources=body.include_sources,
                    **kwargs,
                ),
                timeout=60.0,
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="request_timeout")
    except Exception as exc:
        logger.exception("Agent pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return AgentResponse(
        trace_id=trace_id,
        answer=result.answer,
        sources=result.sources if body.include_sources else [],
        context=[
            {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
            for c in result.chunks[:10]
        ],
        mode=result.mode,
        partial=False,
        degraded=[],
    )


# ── /agent/stream (SSE) ─────────────────────────────────

@agent_router.post("/agent/stream")
async def agent_stream(body: AgentRequest, request: Request):
    settings: Settings = request.app.state.settings
    trace_id = uuid.uuid4().hex
    plan = _build_agent_plan(body, settings)
    kwargs = _pipeline_kwargs(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            result = await asyncio.wait_for(
                run_pipeline(
                    plan,
                    project_id=body.project_id, query=body.query,
                    history=body.history, include_sources=body.include_sources,
                    **kwargs,
                ),
                timeout=120.0,
            )

            if result.needs_retry and settings.agent_use_retry:
                retry_query = result.requery or body.query
                if result.missing_terms:
                    suffix = " ".join(t for t in result.missing_terms if t.lower() not in retry_query.lower())
                    if suffix:
                        retry_query = f"{retry_query} {suffix}"
                rplan = retry_round(top_k=body.top_k or 10)
                result = await asyncio.wait_for(
                    run_pipeline(
                        rplan,
                        project_id=body.project_id,
                        query=retry_query,
                        history=body.history, include_sources=body.include_sources,
                        **kwargs,
                    ),
                    timeout=60.0,
                )

            async for chunk in _stream_answer(result, trace_id=trace_id, include_traces=False):
                if await request.is_disconnected():
                    return
                yield chunk

        except asyncio.TimeoutError:
            yield _sse({"type": "error", "error": "request_timeout", "trace_id": trace_id})
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Stream error")
            try:
                yield _sse({"type": "error", "error": str(exc), "trace_id": trace_id})
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /execute (raw BrainPlan) ─────────────────────────────

@execute_router.post("/execute")
async def execute_plan(body: ExecuteRequest, request: Request):
    trace_id = uuid.uuid4().hex
    kwargs = _pipeline_kwargs(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            result = await asyncio.wait_for(
                run_pipeline(
                    body.brain_plan,
                    project_id=body.project_id, query=body.query,
                    history=body.history, include_sources=body.include_sources,
                    **kwargs,
                ),
                timeout=120.0,
            )
            async for chunk in _stream_answer(result, trace_id=trace_id, include_traces=True):
                yield chunk
        except asyncio.TimeoutError:
            yield _sse({"type": "error", "error": "request_timeout", "trace_id": trace_id})
        except Exception as exc:
            logger.exception("Execute error")
            yield _sse({"type": "error", "error": str(exc), "trace_id": trace_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
