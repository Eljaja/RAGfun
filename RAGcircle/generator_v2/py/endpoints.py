"""FastAPI routers for /chat, /agent, /agent/stream."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent_pipeline import agent_pipeline
from config import Settings
from llm import LLMClient
from models import AgentRequest, AgentResponse, ChatRequest, ChatResponse
from simple_pipeline import simple_pipeline

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])
agent_router = APIRouter(tags=["agent"])


# ── /chat ────────────────────────────────────────────────


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    settings: Settings = request.app.state.settings
    llm: LLMClient = request.app.state.llm

    try:
        return await simple_pipeline(
            body,
            llm=llm,
            http_client=request.app.state.http_client,
            retrieval_url=settings.retrieval_url,
            gen_model=settings.llm_model,
            reflection_model=settings.reflection_model or settings.llm_model,
            reflection_enabled=settings.reflection_enabled,
            max_context_chars=settings.max_context_chars,
            max_chunk_chars=settings.max_chunk_chars,
        )
    except Exception:
        logger.exception("Chat pipeline failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Generation pipeline error")


# ── /agent (non-streaming) ───────────────────────────────


@agent_router.post("/agent", response_model=AgentResponse)
async def agent_non_streaming(body: AgentRequest, request: Request):
    settings: Settings = request.app.state.settings
    llm: LLMClient = request.app.state.llm
    trace_id = uuid.uuid4().hex

    answer = ""
    sources: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    mode = "hybrid"
    partial = False
    degraded: list[str] = []
    error: str | None = None

    async def _collect():
        nonlocal answer, sources, context, mode, partial, degraded, error
        async for event in agent_pipeline(
            body, llm=llm, http_client=request.app.state.http_client, settings=settings,
        ):
            etype = event.get("type")
            if etype == "retrieval":
                context = list(event.get("context") or [])
                mode = event.get("mode", mode)
            elif etype == "token":
                answer += event.get("content", "")
            elif etype == "done":
                answer = event.get("answer", answer)
                sources = list(event.get("sources") or [])
                context = list(event.get("context") or context)
                mode = event.get("mode", mode)
                partial = event.get("partial", partial)
                degraded = list(event.get("degraded") or [])
            elif etype == "error":
                error = event.get("error", "unknown")

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
    timeout_s = 120.0

    async def event_stream() -> AsyncIterator[str]:
        deadline = asyncio.get_event_loop().time() + timeout_s
        try:
            yield _sse({"type": "init", "trace_id": trace_id})
            yield _sse({"type": "trace", "kind": "thought", "label": "Start", "content": "Preparing plan..."})

            agent_it = agent_pipeline(
                body, llm=llm, http_client=request.app.state.http_client, settings=settings,
            )
            while True:
                remaining = max(5.0, deadline - asyncio.get_event_loop().time())
                try:
                    event = await asyncio.wait_for(agent_it.__anext__(), timeout=min(65.0, remaining))
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

                event["trace_id"] = trace_id
                yield _sse(event)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Agent stream error")
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


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"
