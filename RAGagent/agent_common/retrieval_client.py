"""Retrieval client for agent-search (async)."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agent_common.retrieval import context_from_hits, sources_from_context


def _normalize_retrieval_response(data: dict[str, Any], mode: str) -> dict[str, Any]:
    """Normalize retrieval response: ensure hits/context/sources are always present."""
    hits = list(data.get("hits") or [])
    context_chunks = context_from_hits(hits, [])
    sources = data.get("sources")
    if not sources:
        sources = sources_from_context(context_chunks)

    return {
        "ok": bool(data.get("ok", True)),
        "mode": data.get("mode", mode),
        "partial": bool(data.get("partial")),
        "degraded": data.get("degraded") or [],
        "hits": hits,
        "context": context_chunks,
        "sources": sources or [],
        "retrieval": data,
    }


class AsyncRetrievalClient:
    """Async HTTP client for retrieval /v1/search with retry and timeout."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        retry_attempts: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        retrieval_mode: str = "hybrid",
        top_k: int = 8,
        rerank: bool = True,
        use_adaptive_k: bool | None = None,
        filters: dict[str, Any] | None = None,
        include_sources: bool = True,
        acl: list[str] | None = None,
        max_chunks_per_doc: int | None = None,
    ) -> dict[str, Any]:
        """Call retrieval /v1/search and return normalized hits/context/sources."""
        url = f"{self._base_url}/v1/search"
        payload: dict[str, Any] = {
            "query": query,
            "mode": retrieval_mode,
            "top_k": int(top_k),
            "rerank": bool(rerank),
            "include_sources": bool(include_sources),
            "sources_level": "basic",
            "group_by_doc": True,
            "filters": filters,
            "acl": acl or [],
        }
        if use_adaptive_k is not None:
            payload["use_adaptive_k"] = use_adaptive_k
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = int(max_chunks_per_doc)

        @retry(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            reraise=True,
        )
        async def _call() -> dict[str, Any]:
            resp = await client.post(url, json=payload, timeout=self._timeout_s)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await _call()
            return _normalize_retrieval_response(data, retrieval_mode)
        except Exception:
            return {
                "ok": False,
                "mode": retrieval_mode,
                "partial": True,
                "degraded": ["retrieval_error"],
                "hits": [],
                "context": [],
                "sources": [],
                "retrieval": {"ok": False, "partial": True, "degraded": ["retrieval_error"]},
            }
