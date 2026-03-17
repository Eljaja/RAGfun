from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from models import (
    ExecuteRequest,
    ExecuteResponse,
    ExecutionPlan,
    RetrieveRequest,
    RetrieveResponse,
)
from retriever import HybridRetriever

legacy_router = APIRouter(tags=["legacy"])
plan_router = APIRouter(prefix="/plan", tags=["plan"])


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


@legacy_router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest, retriever: HybridRetriever = Depends(get_retriever)):
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

    return RetrieveResponse(
        chunks=chunks,
        strategy=body.strategy,
        query=body.query,
        plan_used=plan,
    )


@plan_router.post("/retrieve", response_model=ExecuteResponse)
async def plan_retrieve(body: ExecuteRequest, retriever: HybridRetriever = Depends(get_retriever)):
    chunks = await retriever.execute_plan(
        query=body.query,
        collection=body.project_id,
        plan=body.plan,
    )

    return ExecuteResponse(
        chunks=chunks,
        query=body.query,
        plan=body.plan,
    )
