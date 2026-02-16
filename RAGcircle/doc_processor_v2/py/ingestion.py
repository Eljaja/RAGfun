"""
Ingestion orchestration for Qdrant (vector) and OpenSearch (BM25).

Exceptions propagate naturally — the caller decides retry policy.
"""
from __future__ import annotations

import asyncio

from models import ChunkMeta
from embed_caller import Embedder
from store import QdrantStore, BM25Store


async def ingest_chunks(
    chunks: list[ChunkMeta],
    embedder: Embedder,
    qdrant: QdrantStore,
    opensearch: BM25Store,
    qdrant_collection: str,
    opensearch_index: str,
    *,
    embed_batch_size: int = 32,
    model: str | None = None,
) -> None:
    """Ingest chunks into both stores in parallel. Raises on failure."""
    if not chunks:
        return
    # If one fails both are cancelled and propagated
    await asyncio.gather(
        qdrant.ingest_chunks(chunks, embedder, qdrant_collection, batch_size=embed_batch_size, model=model),
        opensearch.upsert(chunks, opensearch_index),
    )

