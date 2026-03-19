from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from config import Settings
from store import QdrantStore, BM25Store
from embedder import Embedder
from reranker import Reranker
from retriever import HybridRetriever
from models import ExecuteRequest, ExecuteResponse, ExecutionPlan, RetrieveRequest, RetrieveResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger.info("Starting retrieval service on port %d", settings.port)

    qdrant = QdrantStore(settings.qdrant_url)
    bm25 = BM25Store(settings.opensearch_url)
    embedder = Embedder(
        settings.embedder_url,
        settings.embedder_model,
        timeout=settings.embedder_timeout,
    )
    reranker = Reranker(settings.reranker_url, settings.reranker_model)

    app.state.retriever = HybridRetriever(qdrant, bm25, embedder, reranker)
    app.state.settings = settings

    yield

    await embedder.close()
    await reranker.close()
    await bm25.close()
    await qdrant.close()


app = FastAPI(lifespan=lifespan, title="Retrieval Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest):
    retriever: HybridRetriever = app.state.retriever

    try:
        plan = ExecutionPlan.from_legacy(
            strategy=body.strategy,
            top_k=body.top_k,
            rerank=body.rerank,
            rerank_top_n=body.rerank_top_n,
        )
        chunks = await retriever.execute_plan(
            query=body.query,
            collection=body.project_id,
            plan=plan,
        )
    except Exception:
        logger.exception("Retrieval failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Retrieval backend error")

    return RetrieveResponse(chunks=chunks, strategy=body.strategy, query=body.query, plan_used=plan)


@app.post("/plan/retrieve", response_model=ExecuteResponse)
async def plan_retrieve(body: ExecuteRequest):
    retriever: HybridRetriever = app.state.retriever
    try:
        chunks = await retriever.execute_plan(
            query=body.query,
            collection=body.project_id,
            plan=body.plan,
        )
    except Exception:
        logger.exception("Plan retrieval failed for project %s", body.project_id)
        raise HTTPException(status_code=502, detail="Retrieval backend error")

    return ExecuteResponse(
        chunks=chunks,
        query=body.query,
        plan=body.plan,
    )


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app=app, port=settings.port, host="localhost")
