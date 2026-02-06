# services/store.py
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import hashlib

def _chunk_to_point_id(chunk_id: str) -> int:
    """Convert chunk_id to a deterministic positive int for Qdrant."""
    hash_bytes = hashlib.md5(chunk_id.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big') & 0x7FFFFFFFFFFFFFFF


class QdrantStore:
    def __init__(self, url: str, dimension: int):
        self.client = AsyncQdrantClient(url=url)
        self.dimension = dimension

    async def ensure_collection(self, collection: str, dimension: int | None = None):
        """
        Ensure a collection exists. If dimension is not provided, uses the default dimension.
        """
        dim = dimension if dimension is not None else self.dimension
        exists = await self.client.collection_exists(collection)
        if not exists:
            await self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    async def upsert(self, chunks: list, vectors: list[list[float]], collection: str):
        """Upsert ChunkMeta objects with their vectors."""
        from hash_strategy import sha256_hex

        # print(collection)

        #print(chunks[0])

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
        await self.client.upsert(collection, points)

    async def get_existing_chunks(self, chunk_ids: list[str], collection: str) -> dict[str, dict]:
        """
        Retrieve existing chunk metadata from Qdrant.
        Returns {chunk_id: {"exists": True, "content_hash": ...}} for found chunks.
        """
        if not chunk_ids:
            return {}

        point_ids = [_chunk_to_point_id(cid) for cid in chunk_ids]
        try:
            points = await self.client.retrieve(
                collection,
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

    async def delete_by_doc_id(self, doc_id: str, collection: str):
        """Delete all chunks for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        if not await self.client.collection_exists(collection):
            return

        await self.client.delete(
            collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )

    async def close(self):
        await self.client.close()

    
    # In QdrantStore
    async def delete_collection(self, collection: str):
        """Delete an entire collection."""
        if await self.client.collection_exists(collection):
            await self.client.delete_collection(collection)



# services/bm25_store.py
from opensearchpy import AsyncOpenSearch


class BM25Store:
    def __init__(self, url: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)

    async def ensure_index(self, index: str):
        """Ensure an index exists."""
        exists = await self.client.indices.exists(index=index)
        if not exists:
            await self.client.indices.create(index=index, body={
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

    async def upsert(self, chunks: list, index: str):
        """Bulk upsert ChunkMeta objects."""
        if not chunks:
            return

        # Use bulk API for efficiency
        actions = []
        for c in chunks:
            actions.append({"index": {"_index": index, "_id": c.chunk_id}})
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

    async def delete_by_doc_id(self, doc_id: str, index: str):

        """Delete all chunks for a document."""
        if not await self.client.indices.exists(index=index):
            return
        
        await self.client.delete_by_query(
            index=index,
            body={"query": {"term": {"doc_id": doc_id}}}
        )

    async def search(self, query: str, index: str, top_k: int = 20) -> list[tuple[str, float]]:
        resp = await self.client.search(
            index=index,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    async def close(self):
        await self.client.close()

    async def get(self, doc_id: str, index: str) -> dict | None:
        resp = await self.client.get(index=index, id=doc_id)
        print("THIS IS RESP", resp)
        return resp["_source"]


    # In BM25Store  
    async def delete_index(self, index: str):
        """Delete an entire index."""
        if await self.client.indices.exists(index=index):
            await self.client.indices.delete(index=index)
