from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from opensearchpy import AsyncOpenSearch
from qdrant_client import AsyncQdrantClient

from config import Settings
from models import ExecutionPlan, RetrieveRequest, RetrieveResponse
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest):
    retriever: HybridRetriever = app.state.retriever

    try:
        if body.plan is not None:
            plan_used = body.plan
            strategy = "plan"
        else:
            plan_used = ExecutionPlan.from_legacy(
                strategy=body.strategy,
                top_k=body.top_k,
                rerank=body.rerank,
                rerank_top_n=body.rerank_top_n,
            )
            strategy = body.strategy

        chunks = await retriever.execute_plan(
            query=body.query,
            collection=body.project_id,
            plan=plan_used,
        )
    except Exception:
        logger.exception("Retrieval failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Retrieval backend error")

    return RetrieveResponse(
        chunks=chunks,
        strategy=strategy,
        query=body.query,
        plan_used=plan_used,
        rounds_executed=len(plan_used.rounds),
    )


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="localhost")
