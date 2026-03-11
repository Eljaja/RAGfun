from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from opensearchpy import AsyncOpenSearch


class QdrantStore:
    def __init__(self, url: str):
        self.client = AsyncQdrantClient(url=url)

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
    ) -> list[dict]:
        results = await self.client.query_points(
            collection, query=vector, limit=limit, with_payload=True,
        )
        return [
            {
                "text": r.payload["text"],
                "source_id": r.payload.get("source_id") or r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "score": r.score,
            }
            for r in results.points
        ]

    async def close(self):
        await self.client.close()


class BM25Store:
    def __init__(self, url: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)

    async def search(
        self,
        query: str,
        index: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        resp = await self.client.search(
            index=index,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    async def get(self, doc_id: str, index: str) -> dict | None:
        try:
            resp = await self.client.get(index=index, id=doc_id)
            return resp["_source"]
        except Exception:
            return None

    async def close(self):
        await self.client.close()
