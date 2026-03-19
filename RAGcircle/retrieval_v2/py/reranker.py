from __future__ import annotations

import httpx

from models import ChunkResult, ScoreSource


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
        reranked: list[ChunkResult] = []
        for r in ranked[:top_n]:
            i = int(r["index"])
            c = chunks[i]
            reranked.append(
                ChunkResult(
                    text=c.text,
                    source_id=c.source_id,
                    chunk_index=c.chunk_index,
                    score=float(r["relevance_score"]),
                    score_source=ScoreSource.RERANK,
                )
            )
        return reranked

    async def close(self):
        await self.client.aclose()
