"""HTTP client for the plan-based retrieval service. Pure transport."""

from __future__ import annotations

from typing import Any

import httpx

from models import ChunkResult, ExecutionPlan


def _strip_empty_queries(obj: Any) -> Any:
    """Remove empty 'query' fields from plan dicts before sending.

    The retrieval_v2 service rejects query="" with a 422; omitting the
    field makes it fall back to the request-level query instead.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_empty_queries(v)
            for k, v in obj.items()
            if not (k == "query" and v == "")
        }
    if isinstance(obj, list):
        return [_strip_empty_queries(item) for item in obj]
    return obj


async def retrieve(
    client: httpx.AsyncClient,
    retrieval_url: str,
    *,
    project_id: str,
    query: str,
    plan: ExecutionPlan,
) -> list[ChunkResult]:
    plan_data = _strip_empty_queries(plan.model_dump())
    resp = await client.post(
        f"{retrieval_url.rstrip('/')}/plan/retrieve",
        json={
            "project_id": project_id,
            "query": query,
            "plan": plan_data,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [ChunkResult(**c) for c in data.get("chunks", [])]
