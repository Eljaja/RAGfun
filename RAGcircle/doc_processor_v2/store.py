# services/store.py
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def _chunk_to_point_id(chunk_id: str) -> int:
    """Convert chunk_id to a positive int for Qdrant."""
    return hash(chunk_id) & 0x7FFFFFFFFFFFFFFF


class QdrantStore:
    def __init__(self, url: str, collection: str, dimension: int):
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection
        self.dimension = dimension

    async def ensure_collection(self):
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )

    async def upsert(self, chunks: list, vectors: list[list[float]]):
        """Upsert ChunkMeta objects with their vectors."""
        from hash_strategy import sha256_hex
        points = [
            PointStruct(
                id=_chunk_to_point_id(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "source": c.source,
                    "uri": c.uri,
                    "page": c.locator.page if c.locator else None,
                    "content_hash": sha256_hex(c.text),  # For idempotency
                }
            )
            for c, v in zip(chunks, vectors)
        ]
        await self.client.upsert(self.collection, points)

    async def get_existing_chunks(self, chunk_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve existing chunk metadata from Qdrant.
        Returns {chunk_id: {"exists": True, "content_hash": ...}} for found chunks.
        """
        if not chunk_ids:
            return {}

        point_ids = [_chunk_to_point_id(cid) for cid in chunk_ids]
        try:
            points = await self.client.retrieve(
                self.collection,
                ids=point_ids,
                with_payload=["chunk_id", "content_hash"],
            )
            result = {}

            for p in points:
                if p.payload:
                    cid = p.payload.get("chunk_id")
                    if cid:
                        result[cid] = {
                            "exists": True,
                            "content_hash": p.payload.get("content_hash"),
                        }
            return result
        except Exception:
            return {}

    async def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        await self.client.delete(
            self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )

    async def close(self):
        await self.client.close()



# services/bm25_store.py
from opensearchpy import AsyncOpenSearch


class BM25Store:
    def __init__(self, url: str, index: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)
        self.index = index

    async def ensure_index(self):
        exists = await self.client.indices.exists(index=self.index)
        if not exists:
            await self.client.indices.create(index=self.index, body={
                "settings": {
                    "analysis": {
                        "filter": {
                            "russian_stemmer": {"type": "stemmer", "language": "russian"}
                        },
                        "analyzer": {
                            "russian": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "russian_stemmer"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "text": {"type": "text", "analyzer": "russian"},
                        "source": {"type": "keyword"},
                        "uri": {"type": "keyword"},
                        "page": {"type": "integer"},
                    }
                }
            })

    async def upsert(self, chunks: list):
        """Bulk upsert ChunkMeta objects."""
        if not chunks:
            return

        # Use bulk API for efficiency
        actions = []
        for c in chunks:
            actions.append({"index": {"_index": self.index, "_id": c.chunk_id}})
            actions.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "source": c.source,
                "uri": c.uri,
                "page": c.locator.page if c.locator else None,
            })

        if actions:
            await self.client.bulk(body=actions)

    async def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a document."""
        await self.client.delete_by_query(
            index=self.index,
            body={"query": {"term": {"doc_id": doc_id}}}
        )

    async def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        resp = await self.client.search(
            index=self.index,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    async def close(self):
        await self.client.close()

    async def get(self, doc_id: str) -> dict | None:
        resp = await self.client.get(index=self.index, id=doc_id)
        print("THIS IS RESP", resp)
        return resp["_source"]
