from __future__ import annotations

import httpx

from models import ChunkResult


class Reranker:
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.url = f"{base_url.rstrip('/')}/rerank"
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def rerank(
        self,
        query: str,
        chunks: list[ChunkResult],
        top_n: int = 5,
    ) -> list[ChunkResult]:
        resp = await self.client.post(self.url, json={
            "model": self.model,
            "query": query,
            "documents": [c.text for c in chunks],
        })
        resp.raise_for_status()

        ranked = sorted(
            resp.json()["results"],
            key=lambda x: x["relevance_score"],
            reverse=True,
        )
        return [chunks[r["index"]] for r in ranked[:top_n]]

    async def close(self):
        await self.client.aclose()
