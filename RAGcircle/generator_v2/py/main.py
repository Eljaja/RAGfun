from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from config import Settings
from endpoints import agent_router, chat_router, execute_router
from llm import LLMClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting brain service on port %d", settings.port)

    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
    http_client = httpx.AsyncClient(timeout=max(settings.agent_gate_timeout, 30.0))

    app.state.llm = llm
    app.state.settings = settings
    app.state.http_client = http_client

    yield

    await http_client.aclose()
    await llm.close()


app = FastAPI(lifespan=lifespan, title="Brain Service")

app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(execute_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="0.0.0.0")
