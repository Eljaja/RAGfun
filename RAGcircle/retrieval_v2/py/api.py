from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Request, HTTPException

from models import (
    ExecuteRequest,
    ExecuteResponse,
    ExecutionPlan,
    RetrieveRequest,
    RetrieveResponse,
)
from project_deps import ProjectDeps, extract_project_deps
from retriever import HybridRetriever

legacy_router = APIRouter(tags=["legacy"])
plan_router = APIRouter(prefix="/plan", tags=["plan"])


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_gate_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.gate_http


async def fetch_project_deps(
    client: httpx.AsyncClient,
    project_id: str,
) -> ProjectDeps:
    """Fetch project from gate and extract deps."""
    url = f"/public/v1/internal/projects/{project_id}"
    response = await client.get(url, timeout=10.0)

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"project_not_found:{project_id}")

    response.raise_for_status()
    payload = response.json()
    project = payload.get("project")

    if not project:
        raise HTTPException(status_code=502, detail="invalid_project_response")

    return extract_project_deps(project)


@legacy_router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    gate_client: httpx.AsyncClient = Depends(get_gate_client),
):
    project_deps = await fetch_project_deps(gate_client, body.project_id)

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
        project_deps=project_deps,
    )

    return RetrieveResponse(
        chunks=chunks,
        strategy=body.strategy,
        query=body.query,
        plan_used=plan,
    )


@plan_router.post("/retrieve", response_model=ExecuteResponse)
async def plan_retrieve(
    body: ExecuteRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    gate_client: httpx.AsyncClient = Depends(get_gate_client),
):
    project_deps = await fetch_project_deps(gate_client, body.project_id)

    chunks = await retriever.execute_plan(
        query=body.query,
        collection=body.project_id,
        plan=body.plan,
        project_deps=project_deps,
    )

    return ExecuteResponse(
        chunks=chunks,
        query=body.query,
        plan=body.plan,
    )
