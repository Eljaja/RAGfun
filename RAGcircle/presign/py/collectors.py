# services/store.py
from __future__ import annotations
from opensearchpy import AsyncOpenSearch

from typing import AsyncIterator
from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import hashlib


class QdrantStore:
    def __init__(self, url: str):
        self.client = AsyncQdrantClient(url=url)

    async def ensure_collection(self, collection: str, dimension: int):
        """
        Ensure a collection exists. If dimension is not provided, uses the default dimension.
        """
        exists = await self.client.collection_exists(collection)
        if not exists:
            await self.client.create_collection(
                collection,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE)
            )

    async def delete_collection(self, collection: str):
        """Delete an entire collection."""
        if await self.client.collection_exists(collection):
            await self.client.delete_collection(collection)

    async def close(self):
        await self.client.close()


# services/bm25_store.py


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

    # In BM25Store
    async def delete_index(self, index: str):
        """Delete an entire index."""
        if await self.client.indices.exists(index=index):
            await self.client.indices.delete(index=index)

    async def close(self):
        await self.client.close()


@asynccontextmanager
async def create_qdrant_store(
    url: str,
) -> AsyncIterator[QdrantStore]:
    """Create and initialize a QdrantStore."""
    store = QdrantStore(url=url)
    try:
        yield store
    finally:
        await store.close()


@asynccontextmanager
async def create_bm25_store(
    url: str,
) -> AsyncIterator[BM25Store]:
    """Create and initialize a BM25Store."""
    store = BM25Store(url=url)
    try:
        yield store
    finally:
        await store.close()
