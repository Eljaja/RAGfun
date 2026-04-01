"""Retrieve phase: requests in, chunks out.

Single implementation for fan-out retrieval.  Both the engine pipeline
and any external callers go through ``fetch_all`` /  ``safe_retrieve``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from config import Settings
from lib.context import merge_chunks
from models.plan import RetrievalRequest
from retrieval_contract import ChunkResult, ExecutionPlan
from retrieval_contract import from_preset
from clients.retrieval import RetrievalTransportError, retrieve as retrieval_call

logger = logging.getLogger(__name__)


async def safe_retrieve(
    query: str, plan: ExecutionPlan, *,
    project_id: str, http_client: httpx.AsyncClient, settings: Settings,
) -> list[ChunkResult]:
    try:
        return await retrieval_call(
            http_client, settings.retrieval_url,
            project_id=project_id, query=query, plan=plan,
        )
    except RetrievalTransportError as exc:
        logger.error("Retrieval failed (transport): %s", exc)
        return []


async def fetch_all(
    requests: list[RetrievalRequest],
    *,
    default_plan: ExecutionPlan,
    project_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> list[ChunkResult]:
    """Fan out retrieval requests in parallel, merge results."""
    if not requests:
        return []

    if len(requests) == 1:
        req = requests[0]
        return await safe_retrieve(
            req.query, req.plan_override or default_plan,
            project_id=project_id, http_client=http_client, settings=settings,
        )

    results = await asyncio.gather(*(
        safe_retrieve(
            req.query, req.plan_override or default_plan,
            project_id=project_id, http_client=http_client, settings=settings,
        )
        for req in requests
    ))
    return merge_chunks(list(results))


async def retrieve_queries(
    queries: list[str],
    *,
    plan: ExecutionPlan | None = None,
    preset: str = "hybrid",
    top_k: int = 10,
    rerank: bool = True,
    project_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> list[ChunkResult]:
    """Convenience wrapper: plain query strings -> chunks."""
    resolved = plan or from_preset(preset, top_k=top_k, rerank=rerank)
    return await fetch_all(
        [RetrievalRequest(query=q) for q in queries],
        default_plan=resolved,
        project_id=project_id, http_client=http_client, settings=settings,
    )
