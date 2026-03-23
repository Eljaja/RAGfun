"""Retrieve phase: queries in, chunks out.

Pure function: (queries, plan, project_id, deps) -> list[ChunkResult].
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from config import Settings
from lib.context import merge_chunks
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


async def retrieve_all(
    queries: list[str],
    *,
    plan: ExecutionPlan | None,
    preset: str = "hybrid",
    top_k: int = 10,
    rerank: bool = True,
    project_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> list[ChunkResult]:
    """Run retrieval for all queries, merge and return chunks."""
    resolved_plan = plan or from_preset(preset, top_k=top_k, rerank=rerank)

    if len(queries) <= 1:
        query = queries[0] if queries else ""
        return await safe_retrieve(
            query, resolved_plan,
            project_id=project_id, http_client=http_client, settings=settings,
        )

    results = await asyncio.gather(*(
        safe_retrieve(
            sq, resolved_plan,
            project_id=project_id, http_client=http_client, settings=settings,
        )
        for sq in queries
    ))
    return merge_chunks(list(results))
