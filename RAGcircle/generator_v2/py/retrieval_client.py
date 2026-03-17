"""HTTP client for the plan-based retrieval service. Pure transport."""

from __future__ import annotations

import httpx

from models import ChunkResult, ExecutionPlan


async def retrieve(
    client: httpx.AsyncClient,
    retrieval_url: str,
    *,
    project_id: str,
    query: str,
    plan: ExecutionPlan,
) -> list[ChunkResult]:
    resp = await client.post(
        f"{retrieval_url.rstrip('/')}/plan/retrieve",
        json={
            "project_id": project_id,
            "query": query,
            "plan": plan.model_dump(),
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [ChunkResult(**c) for c in data.get("chunks", [])]
