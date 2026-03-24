"""FastAPI router for /plan/run and /plan/run/stream.

Dedicated endpoint for running explicit BrainPlans — separate from the
production /chat, /agent, and /execute endpoints.  Accepts a fully typed
BrainRound so Pydantic validates the plan tree at parse time.
"""

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

from engine.brain_pipeline import run_pipeline
from models import PipelineResult, PlanRunRequest, PlanRunResponse
from models.events import DoneEvent, InitEvent, TokenEvent, TraceEvent

logger = logging.getLogger(__name__)

plan_router = APIRouter(prefix="/plan", tags=["plan"])


# ── helpers ───────────────────────────────────────────────


def _pipeline_kwargs(request: Request) -> dict[str, Any]:
    return dict(
        llm=request.app.state.llm,
        http_client=request.app.state.http_client,
        settings=request.app.state.settings,
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _event_dict(event: object) -> dict[str, Any]:
    return dataclasses.asdict(event)


def _build_response(
    result: PipelineResult, body: PlanRunRequest,
) -> PlanRunResponse:
    return PlanRunResponse(
        brain_plan=body.brain_plan,
        answer=result.answer,
        sources=result.sources if body.include_sources else [],
        context=[
            {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
            for c in result.chunks[:10]
        ],
        traces=result.traces if body.include_traces else [],
        mode=result.mode,
        lang=result.lang,
        is_factoid=result.is_factoid,
        needs_retry=result.needs_retry,
        missing_terms=result.missing_terms,
        requery=result.requery,
    )


async def _run(body: PlanRunRequest, request: Request) -> PipelineResult:
    kwargs = _pipeline_kwargs(request)
    return await asyncio.wait_for(
        run_pipeline(
            body.brain_plan,
            project_id=body.project_id,
            query=body.query,
            history=body.history,
            include_sources=body.include_sources,
            **kwargs,
        ),
        timeout=120.0,
    )


# ── POST /plan/run  (sync JSON) ──────────────────────────


@plan_router.post("/run", response_model=PlanRunResponse)
async def plan_run(body: PlanRunRequest, request: Request):
    try:
        result = await _run(body, request)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="request_timeout")
    except Exception as exc:
        logger.exception("Plan run failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return _build_response(result, body)


# ── POST /plan/run/stream  (SSE) ─────────────────────────


@plan_router.post("/run/stream")
async def plan_run_stream(body: PlanRunRequest, request: Request):
    trace_id = uuid.uuid4().hex

    async def event_stream() -> AsyncIterator[str]:
        try:
            result = await _run(body, request)

            yield _sse(_event_dict(InitEvent(trace_id=trace_id)))

            if body.include_traces:
                for trace in result.traces:
                    ev = TraceEvent(**trace)
                    d = _event_dict(ev)
                    d["trace_id"] = trace_id
                    yield _sse(d)

            text = result.answer
            for i in range(0, len(text), 40):
                d = _event_dict(TokenEvent(content=text[i : i + 40]))
                d["trace_id"] = trace_id
                yield _sse(d)

            done = DoneEvent(
                answer=result.answer,
                mode=result.mode,
                partial=False,
                degraded=[],
                sources=result.sources if body.include_sources else [],
                context=[
                    {"source_id": c.source_id, "text": c.text[:300], "score": c.score}
                    for c in result.chunks[:10]
                ],
                needs_retry=result.needs_retry,
                missing_terms=result.missing_terms,
            )
            d = _event_dict(done)
            d["trace_id"] = trace_id
            d["brain_plan"] = body.brain_plan.model_dump(mode="json")
            yield _sse(d)

        except asyncio.TimeoutError:
            yield _sse({"type": "error", "error": "request_timeout", "trace_id": trace_id})
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Plan stream error")
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
