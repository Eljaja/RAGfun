"""FastAPI routers for /chat, /agent, /agent/stream, /execute."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from brain_presets import AGENT_PRESET_BUILDERS, agent, simple
from config import Settings
from engine import execute
from llm import LLMClient
from models import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    ExecuteRequest,
)
from models.events import DoneEvent, ErrorEvent, Event, InitEvent, TokenEvent

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])
agent_router = APIRouter(tags=["agent"])
execute_router = APIRouter(tags=["execute"])


# ── SSE helpers ──────────────────────────────────────────

def _event_to_dict(event: Event) -> dict[str, Any]:
    return dataclasses.asdict(event)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_execute(
    plan: Any,
    *,
    trace_id: str,
    request: Request,
    timeout_s: float = 120.0,
    **execute_kwargs: Any,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        deadline = asyncio.get_event_loop().time() + timeout_s
        try:
            yield _sse(_event_to_dict(InitEvent(trace_id=trace_id)))

            engine = execute(plan, **execute_kwargs)
            while True:
                remaining = max(5.0, deadline - asyncio.get_event_loop().time())
                try:
                    event = await asyncio.wait_for(engine.__anext__(), timeout=min(65.0, remaining))
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield _sse({"type": "error", "error": "Timeout waiting for response", "trace_id": trace_id})
                    return

                if await request.is_disconnected():
                    return
                if asyncio.get_event_loop().time() > deadline:
                    yield _sse({"type": "error", "error": "request_timeout", "trace_id": trace_id})
                    return

                d = _event_to_dict(event)
                d["trace_id"] = trace_id
                yield _sse(d)

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


# ── /chat ────────────────────────────────────────────────

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    settings: Settings = request.app.state.settings
    llm: LLMClient = request.app.state.llm

    plan = simple(
        preset=body.preset,
        top_k=body.top_k,
        rerank=body.rerank,
        max_retries=body.max_retries,
        reflection_enabled=body.reflection_enabled and settings.reflection_enabled,
    )

    answer = ""
    chunks_used = 0
    sources: list[str] = []
    retries_used = 0

    try:
        async for event in execute(
            plan,
            project_id=body.project_id,
            query=body.query,
            include_sources=True,
            llm=llm,
            http_client=request.app.state.http_client,
            settings=settings,
        ):
            if isinstance(event, TokenEvent):
                answer += event.content
            elif isinstance(event, DoneEvent):
                answer = event.answer
                sources = [s.get("source_id", "") for s in event.sources if isinstance(s, dict)]
                chunks_used = len(event.context)
            elif isinstance(event, ErrorEvent):
                raise HTTPException(status_code=502, detail=event.error)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat pipeline failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Generation pipeline error")

    return ChatResponse(
        answer=answer,
        sources=sources,
        chunks_used=chunks_used,
        retries_used=retries_used,
        query=body.query,
    )


# ── /agent (non-streaming) ───────────────────────────────

def _build_agent_plan(body: AgentRequest, settings: Settings):
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
    llm: LLMClient = request.app.state.llm
    trace_id = uuid.uuid4().hex
    plan = _build_agent_plan(body, settings)

    answer = ""
    sources: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    mode = "hybrid"
    partial = False
    degraded: list[str] = []
    error: str | None = None

    async def _collect():
        nonlocal answer, sources, context, mode, partial, degraded, error
        async for event in execute(
            plan,
            project_id=body.project_id,
            query=body.query,
            history=body.history,
            include_sources=body.include_sources,
            llm=llm,
            http_client=request.app.state.http_client,
            settings=settings,
        ):
            if isinstance(event, TokenEvent):
                answer += event.content
            elif isinstance(event, DoneEvent):
                answer = event.answer
                sources = event.sources
                context = event.context
                mode = event.mode
                partial = event.partial
                degraded = event.degraded
            elif isinstance(event, ErrorEvent):
                error = event.error

    try:
        await asyncio.wait_for(_collect(), timeout=120.0)
    except asyncio.TimeoutError:
        error = "request_timeout"
    except Exception as exc:
        error = str(exc)

    if error:
        raise HTTPException(status_code=500, detail=error)

    return AgentResponse(
        trace_id=trace_id,
        answer=answer,
        sources=sources if body.include_sources else [],
        context=context,
        mode=mode,
        partial=partial,
        degraded=degraded,
    )


# ── /agent/stream (SSE) ─────────────────────────────────

@agent_router.post("/agent/stream")
async def agent_stream(body: AgentRequest, request: Request):
    settings: Settings = request.app.state.settings
    llm: LLMClient = request.app.state.llm
    trace_id = uuid.uuid4().hex
    plan = _build_agent_plan(body, settings)

    return await _stream_execute(
        plan,
        trace_id=trace_id,
        request=request,
        project_id=body.project_id,
        query=body.query,
        history=body.history,
        include_sources=body.include_sources,
        llm=llm,
        http_client=request.app.state.http_client,
        settings=settings,
    )


# ── /execute (raw BrainPlan) ─────────────────────────────

@execute_router.post("/execute")
async def execute_plan(body: ExecuteRequest, request: Request):
    settings: Settings = request.app.state.settings
    llm: LLMClient = request.app.state.llm
    trace_id = uuid.uuid4().hex

    return await _stream_execute(
        body.brain_plan,
        trace_id=trace_id,
        request=request,
        project_id=body.project_id,
        query=body.query,
        history=body.history,
        include_sources=body.include_sources,
        llm=llm,
        http_client=request.app.state.http_client,
        settings=settings,
    )
