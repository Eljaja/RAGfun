"""HTTP client for the plan-based retrieval service. Pure transport."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import ChunkResult, ExecutionPlan

logger = logging.getLogger(__name__)

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


class RetrievalTransportError(Exception):
    """Network / timeout / HTTP error from the retrieval service."""


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


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def retrieve(
    client: httpx.AsyncClient,
    retrieval_url: str,
    *,
    project_id: str,
    query: str,
    plan: ExecutionPlan,
) -> list[ChunkResult]:
    plan_data = _strip_empty_queries(plan.model_dump())
    try:
        resp = await client.post(
            f"{retrieval_url.rstrip('/')}/plan/retrieve",
            json={
                "project_id": project_id,
                "query": query,
                "plan": plan_data,
            },
        )
        resp.raise_for_status()
    except _RETRYABLE:
        raise
    except httpx.HTTPStatusError as exc:
        raise RetrievalTransportError(
            f"{exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc

    data = resp.json()
    return [ChunkResult(**c) for c in data.get("chunks", [])]
