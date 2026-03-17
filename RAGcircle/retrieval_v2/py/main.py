from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opensearchpy import AsyncOpenSearch
from qdrant_client import AsyncQdrantClient

from api import legacy_router, plan_router
from config import Settings
from retriever import HybridRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting retrieval service on port %d", settings.port)

    async with AsyncExitStack() as stack:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        opensearch = AsyncOpenSearch(hosts=[settings.opensearch_url], use_ssl=False)
        embed_http = httpx.AsyncClient(
            base_url=settings.embedder_url.rstrip("/"),
            timeout=httpx.Timeout(settings.embedder_timeout, connect=10.0, read=settings.embedder_timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )
        rerank_http = httpx.AsyncClient(
            base_url=settings.reranker_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0, read=30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )

        stack.push_async_callback(qdrant.close)
        stack.push_async_callback(opensearch.close)
        stack.push_async_callback(embed_http.aclose)
        stack.push_async_callback(rerank_http.aclose)

        app.state.retriever = HybridRetriever(
            qdrant=qdrant,
            opensearch=opensearch,
            embed_http=embed_http,
            embed_model=settings.embedder_model,
            rerank_http=rerank_http,
            rerank_model=settings.reranker_model,
        )
        app.state.settings = settings

        yield


app = FastAPI(lifespan=lifespan, title="Retrieval Service")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=502, content={"detail": "Retrieval backend error"})


app.include_router(legacy_router)
app.include_router(plan_router)


async def _check(name: str, fn) -> tuple[str, str]:
    try:
        result = await asyncio.wait_for(fn(), timeout=10.0)
        if result is False:
            return name, "unavailable"
        return name, "ok"
    except (Exception, asyncio.CancelledError):
        return name, "unavailable"


@app.get("/health")
async def health(request: Request):
    r: HybridRetriever = request.app.state.retriever

    results = await asyncio.gather(
        _check("qdrant", r.qdrant.get_collections),
        _check("opensearch", r.opensearch.ping),
        _check("embedder", lambda: r.embed_http.get("/health")),
        _check("reranker", lambda: r.rerank_http.get("/health")),
    )

    deps = dict(results)
    ok = all(v == "ok" for v in deps.values())
    return {"status": "ok" if ok else "degraded", "dependencies": deps}


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="localhost")
