# services/store.py
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def _chunk_to_point_id(chunk_id: str) -> int:
    """Convert chunk_id to a positive int for Qdrant."""
    return hash(chunk_id) & 0x7FFFFFFFFFFFFFFF


class QdrantStore:
    def __init__(self, url: str, dimension: int):
        self.client = AsyncQdrantClient(url=url)
        self.dimension = dimension

    async def ensure_collection(self, collection: str, dimension: int | None = None):
        """Ensure a collection exists with proper indexing."""
        dim = dimension if dimension is not None else self.dimension
        exists = await self.client.collection_exists(collection)
        if not exists:
            await self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            
            # Create payload index on doc_id for faster filtering
            await self.client.create_payload_index(
                collection,
                field_name="doc_id",
                field_schema="keyword"
            )

    async def upsert(self, chunks: list, vectors: list[list[float]], collection: str):
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
                    "content_hash": sha256_hex(c.text),
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

    async def get_document_chunks(self, doc_id: str, collection: str) -> list[dict]:
        """Get all chunks for a document, sorted by chunk_index."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        result = await self.client.scroll(
            collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=10000  # Adjust based on max chunks per doc
        )
        
        points = result[0] if result else []
        chunks = [p.payload for p in points if p.payload]
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks

    async def delete_by_doc_id(self, doc_id: str, collection: str):
        """Delete all chunks for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        await self.client.delete(
            collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )

    async def search(self, vector: list[float], collection: str, top_k: int = 20, 
                     doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """
        Search for similar chunks, optionally filtered by document IDs.
        Returns list of (chunk_id, score) tuples.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        query_filter = None
        if doc_ids:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
            )
        
        results = await self.client.search(
            collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=["chunk_id"]
        )
        
        return [(hit.payload["chunk_id"], hit.score) for hit in results]

    async def close(self):
        await self.client.close()



# services/bm25_store.py
from opensearchpy import AsyncOpenSearch


class BM25Store:
    def __init__(self, url: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)

    async def ensure_index(self, index: str):
        """Ensure an index exists with nested chunk structure."""
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
                        "doc_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "uri": {"type": "keyword"},
                        "chunks": {
                            "type": "nested",
                            "properties": {
                                "chunk_id": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "text": {"type": "text", "analyzer": "russian"},
                                "page": {"type": "integer"},
                            }
                        }
                    }
                }
            })

    async def upsert(self, chunks: list, index: str):
        """Bulk upsert ChunkMeta objects grouped by document."""
        if not chunks:
            return

        # Group chunks by doc_id
        from collections import defaultdict
        docs_by_id = defaultdict(list)
        for chunk in chunks:
            docs_by_id[chunk.doc_id].append(chunk)

        # Build bulk actions with nested chunks
        actions = []
        for doc_id, doc_chunks in docs_by_id.items():
            # Sort chunks by index for consistency
            doc_chunks.sort(key=lambda c: c.chunk_index)
            
            # Take document-level metadata from first chunk
            first_chunk = doc_chunks[0]
            
            actions.append({"index": {"_index": index, "_id": doc_id}})
            actions.append({
                "doc_id": doc_id,
                "source": first_chunk.source,
                "uri": first_chunk.uri,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "page": c.locator.page if c.locator else None,
                    }
                    for c in doc_chunks
                ]
            })

        if actions:
            await self.client.bulk(body=actions)

    async def delete_by_doc_id(self, doc_id: str, index: str):
        """Delete a document (which contains all its chunks)."""
        await self.client.delete(index=index, id=doc_id, ignore=[404])

    async def search(self, query: str, index: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search within nested chunks and return chunk_ids with scores."""
        resp = await self.client.search(
            index=index,
            body={
                "query": {
                    "nested": {
                        "path": "chunks",
                        "query": {
                            "match": {"chunks.text": query}
                        },
                        "inner_hits": {
                            "size": top_k,
                            "_source": ["chunks.chunk_id"],
                        }
                    }
                },
                "size": top_k  # Return top_k documents
            }
        )
        
        # Extract chunk_ids and scores from nested inner_hits
        results = []
        for hit in resp["hits"]["hits"]:
            if "inner_hits" in hit and "chunks" in hit["inner_hits"]:
                for inner_hit in hit["inner_hits"]["chunks"]["hits"]["hits"]:
                    chunk_id = inner_hit["_source"]["chunk_id"]
                    score = inner_hit["_score"]
                    results.append((chunk_id, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get(self, doc_id: str, index: str) -> dict | None:
        """Get a document by doc_id."""
        try:
            resp = await self.client.get(index=index, id=doc_id)
            print("THIS IS RESP", resp)
            return resp["_source"]
        except Exception:
            return None

    async def close(self):
        await self.client.close()
