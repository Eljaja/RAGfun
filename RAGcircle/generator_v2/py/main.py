from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from config import Settings
from llm import LLMClient
from pipeline import rag_pipeline
from models import ChatRequest, ChatResponse

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


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="localhost")
