"""Gate client for agent-search (async)."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


def _normalize_gate_response(data: dict[str, Any], mode: str) -> dict[str, Any]:
    """Normalize gate response: fill hits from context, ensure structure."""
    retrieval = data.get("retrieval") or {}
    hits = list(retrieval.get("hits") or [])
    context_chunks = list(data.get("context") or [])

    if context_chunks:
        by_cid = {str(c.get("chunk_id")): c for c in context_chunks if c.get("chunk_id")}
        for h in hits:
            if h.get("text"):
                continue
            cid = str(h.get("chunk_id") or "")
            c = by_cid.get(cid)
            if c:
                h["text"] = c.get("text")
                h["source"] = c.get("source")
                if h.get("score") is None:
                    h["score"] = c.get("score")

    if not hits and context_chunks:
        hits = [
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "score": c.get("score"),
                "text": c.get("text"),
                "source": c.get("source"),
            }
            for c in context_chunks
        ]

    return {
        "ok": bool(retrieval.get("ok", True)),
        "mode": retrieval.get("mode", mode),
        "partial": bool(retrieval.get("partial")),
        "degraded": retrieval.get("degraded") or [],
        "hits": hits,
        "context": context_chunks,
        "sources": data.get("sources") or [],
        "retrieval": retrieval,
    }


class AsyncGateClient:
    """Async HTTP client for rag-gate /v1/chat with retry and timeout."""

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

    async def chat(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        history: list[dict[str, str]] | None = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 8,
        rerank: bool = True,
        use_adaptive_k: bool | None = None,
        filters: dict[str, Any] | None = None,
        include_sources: bool = True,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call gate /v1/chat and return normalized response with hits, context, sources."""
        url = f"{self._base_url}/v1/chat"
        payload: dict[str, Any] = {
            "query": query,
            "history": history or [],
            "retrieval_mode": retrieval_mode,
            "top_k": int(top_k),
            "rerank": bool(rerank),
            "include_sources": bool(include_sources),
        }
        if use_adaptive_k is not None:
            payload["use_adaptive_k"] = use_adaptive_k
        if filters:
            payload["filters"] = filters

        @retry(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            reraise=True,
        )
        async def _call() -> dict[str, Any]:
            resp = await client.post(url, json=payload, timeout=self._timeout_s, headers=headers)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await _call()
            return _normalize_gate_response(data, retrieval_mode)
        except Exception:
            return {
                "ok": False,
                "mode": retrieval_mode,
                "partial": True,
                "degraded": ["gate_error"],
                "hits": [],
                "context": [],
                "sources": [],
                "retrieval": {"ok": False, "partial": True, "degraded": ["gate_error"]},
            }
