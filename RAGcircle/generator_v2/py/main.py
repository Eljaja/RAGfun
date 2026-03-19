from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from config import Settings
from llm import LLMClient
from pipeline import rag_agent_pipeline, rag_pipeline
from models import AgentRequest, AgentResponse, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger.info("Starting generator service on port %d", settings.port)

    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
    http_client = httpx.AsyncClient(timeout=30.0)

    app.state.llm = llm
    app.state.settings = settings
    app.state.http_client = http_client

    yield

    await http_client.aclose()
    await llm.close()


app = FastAPI(lifespan=lifespan, title="Generator Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    settings: Settings = app.state.settings

    try:
        result = await rag_pipeline(
            request=body,
            retrieval_url=settings.retrieval_url,
            http_client=app.state.http_client,
            llm=app.state.llm,
            gen_model=settings.llm_model,
            reflection_model=settings.reflection_model or settings.llm_model,
            reflection_enabled=settings.reflection_enabled,
        )
    except Exception:
        logger.exception("RAG pipeline failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Generation pipeline error")

    return result


@app.post("/agent", response_model=AgentResponse)
async def agent(body: AgentRequest):
    settings: Settings = app.state.settings
    trace_id = uuid.uuid4().hex

    try:
        result = await rag_agent_pipeline(
            request=body,
            retrieval_url=settings.retrieval_url,
            http_client=app.state.http_client,
            llm=app.state.llm,
            gen_model=settings.llm_model,
        )
    except Exception:
        logger.exception("Agent pipeline failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Agent pipeline error")

    return AgentResponse(
        trace_id=trace_id,
        answer=result["answer"],
        sources=result["sources"],
        context=result["context"],
        mode=result["mode"],
        partial=result["partial"],
        degraded=result["degraded"],
    )


@app.post("/agent/stream")
async def agent_stream(body: AgentRequest):
    settings: Settings = app.state.settings
    trace_id = uuid.uuid4().hex

    async def _event_stream() -> AsyncIterator[str]:
        try:
            yield _sse({"type": "init", "trace_id": trace_id})

            result = await rag_agent_pipeline(
                request=body,
                retrieval_url=settings.retrieval_url,
                http_client=app.state.http_client,
                llm=app.state.llm,
                gen_model=settings.llm_model,
            )
            yield _sse(
                {
                    "type": "trace",
                    "kind": "tool",
                    "name": "retrieval.plan",
                    "payload": {
                        "query": body.query,
                        "strategy": body.strategy,
                        "top_k": body.top_k,
                        "rerank": body.rerank,
                        "mode": body.mode,
                        "plan": result.get("plan"),
                    },
                    "trace_id": trace_id,
                }
            )
            yield _sse(
                {
                    "type": "retrieval",
                    "mode": result["mode"],
                    "partial": result["partial"],
                    "degraded": result["degraded"],
                    "context": result["context"],
                    "trace_id": trace_id,
                }
            )

            answer = result["answer"]
            if answer:
                # Stream words to keep UI responsive while preserving simple generation path.
                for token in answer.split(" "):
                    yield _sse({"type": "token", "content": f"{token} ", "trace_id": trace_id})

            yield _sse(
                {
                    "type": "done",
                    "answer": answer,
                    "sources": result["sources"],
                    "context": result["context"],
                    "mode": result["mode"],
                    "partial": result["partial"],
                    "degraded": result["degraded"],
                    "trace_id": trace_id,
                }
            )
        except Exception as exc:
            logger.exception("Agent stream failed for project %s", body.project_id)
            yield _sse({"type": "error", "error": str(exc), "trace_id": trace_id})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="localhost")
