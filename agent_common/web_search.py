<<<<<<< HEAD
"""
Web search client for agent-search.

Supports Serper (Google Search API) and Tavily. Set WEB_SEARCH_PROVIDER=serper|tavily
and the corresponding API key. When unset or empty key, web search is disabled.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"
TAVILY_URL = "https://api.tavily.com/search"


def _serper_search_sync(
    client: httpx.Client,
    query: str,
    api_key: str,
    num: int = 5,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Serper: returns list of hits in our format."""
    r = client.post(
        SERPER_URL,
        json={"q": query, "num": num},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    organic = data.get("organic") or []
    hits = []
    for i, o in enumerate(organic[:num]):
        title = (o.get("title") or "").strip()
        link = (o.get("link") or "").strip()
        snippet = (o.get("snippet") or "").strip()
        pos = o.get("position", i + 1)
        text = f"{title}\n{snippet}" if title else snippet
        if not text:
            continue
        hits.append({
            "doc_id": link or f"web:{i}",
            "chunk_id": f"web:{link}:{i}" if link else f"web:{i}",
            "text": text[:2000],
            "score": 1.0 / max(1, pos),
            "source": {"doc_id": link or f"web:{i}", "title": title, "uri": link},
            "metadata": {"url": link, "title": title, "source": "web"},
        })
    return hits


def _tavily_search_sync(
    client: httpx.Client,
    query: str,
    api_key: str,
    num: int = 5,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Tavily: returns list of hits in our format."""
    r = client.post(
        TAVILY_URL,
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": num,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    hits = []
    for i, res in enumerate(results[:num]):
        title = (res.get("title") or "").strip()
        url = (res.get("url") or "").strip()
        content = (res.get("content") or "").strip()
        score = float(res.get("score", 1.0 / (i + 1)))
        text = f"{title}\n{content}" if title else content
        if not text:
            continue
        hits.append({
            "doc_id": url or f"web:{i}",
            "chunk_id": f"web:{url}:{i}" if url else f"web:{i}",
            "text": text[:2000],
            "score": score,
            "source": {"doc_id": url or f"web:{i}", "title": title, "uri": url},
            "metadata": {"url": url, "title": title, "source": "web"},
        })
    return hits


def web_search_sync(
    query: str,
    *,
    provider: str = "serper",
    api_key: str | None = None,
    num: int = 5,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """
    Synchronous web search. Returns hits in format compatible with merge_hits.
    When provider is empty or api_key is missing, returns [].
    """
    provider = (provider or "").lower().strip()
    if not provider or not (api_key or "").strip():
        return []
    with httpx.Client() as client:
        try:
            if provider == "serper":
                return _serper_search_sync(client, query, api_key.strip(), num=num, timeout_s=timeout_s)
            if provider == "tavily":
                return _tavily_search_sync(client, query, api_key.strip(), num=num, timeout_s=timeout_s)
            logger.warning("web_search unknown provider=%s", provider)
            return []
        except Exception as e:
            logger.warning("web_search failed: %s", e)
            return []


async def web_search_async(
    query: str,
    *,
    provider: str = "serper",
    api_key: str | None = None,
    num: int = 5,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Async web search. Runs sync in thread to avoid blocking."""
    import asyncio
    return await asyncio.to_thread(
        web_search_sync,
        query,
        provider=provider,
        api_key=api_key,
        num=num,
        timeout_s=timeout_s,
    )
