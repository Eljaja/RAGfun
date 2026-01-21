"""
Simple ingestion module for Qdrant (vector) and OpenSearch (BM25).

TODO: Consider splitting into:
  - qdrant_ingestion.py - just Qdrant logic
  - opensearch_ingestion.py - just OpenSearch logic  
  - ingestion.py - orchestration only
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from models import ChunkMeta
from embed_caller import Embedder
from store import QdrantStore, BM25Store
from hash_strategy import ExistingChunk, ContentHashStrategy, filter_chunks_needing_embedding


# TODO: Consider using Result[IngestionResult, IngestionError] pattern
@dataclass
class IngestionResult:
    ok: bool
    chunks_count: int
    embedded: int = 0
    skipped: int = 0
    qdrant_ok: bool = True
    opensearch_ok: bool = True
    error: str | None = None  # TODO: Use structured error type instead of string


async def ingest_to_qdrant(
    chunks: list[ChunkMeta],
    embedder: Embedder,
    store: QdrantStore,
    *,
    batch_size: int = 32,
) -> tuple[bool, int, int, str | None]:
    """
    Embed chunks and upsert to Qdrant (with idempotency).
    Returns (success, embedded_count, skipped_count, error_message).
    """
    if not chunks:
        return True, 0, 0, None

    try:
        # Check which chunks already exist with same content
        chunk_ids = [c.chunk_id for c in chunks]
        existing_raw = await store.get_existing_chunks(chunk_ids)
        
        existing_by_id = {
            cid: ExistingChunk(
                exists=data.get("exists", False),
                content_hash=data.get("content_hash"),
            )
            for cid, data in existing_raw.items()
        }

        # Filter to only chunks that need embedding
        strategy = ContentHashStrategy()
        need_embed, skipped = filter_chunks_needing_embedding(chunks, existing_by_id, strategy)

        if not need_embed:
            return True, 0, skipped, None

        # Embed in batches
        all_vectors: list[list[float]] = []
        for i in range(0, len(need_embed), batch_size):
            batch = need_embed[i:i + batch_size]
            texts = [c.text for c in batch]
            vectors = await embedder.embed(texts)
            all_vectors.extend(vectors)

        # Upsert to Qdrant
        await store.upsert(need_embed, all_vectors)
        return True, len(need_embed), skipped, None

    except Exception as e:
        return False, 0, 0, str(e)


async def ingest_to_opensearch(
    chunks: list[ChunkMeta],
    store: BM25Store,
) -> tuple[bool, str | None]:
    """
    Upsert chunks to OpenSearch for BM25 search.
    Returns (success, error_message).
    """
    if not chunks:
        return True, None

    try:
        await store.upsert(chunks)
        return True, None

    except Exception as e:
        return False, str(e)


async def ingest_chunks(
    chunks: list[ChunkMeta],
    embedder: Embedder,
    qdrant: QdrantStore,
    opensearch: BM25Store,
    *,
    embed_batch_size: int = 32,
) -> IngestionResult:
    """
    Ingest chunks into both Qdrant and OpenSearch in parallel.
    Qdrant uses idempotency check to skip unchanged chunks.
    """
    if not chunks:
        return IngestionResult(ok=True, chunks_count=0)

    # Run both in parallel
    qdrant_task = ingest_to_qdrant(chunks, embedder, qdrant, batch_size=embed_batch_size)
    opensearch_task = ingest_to_opensearch(chunks, opensearch)

    (qdrant_ok, embedded, skipped, qdrant_err), (os_ok, os_err) = await asyncio.gather(
        qdrant_task, opensearch_task
    )

    errors = []
    if not qdrant_ok:
        errors.append(f"qdrant: {qdrant_err}")
    if not os_ok:
        errors.append(f"opensearch: {os_err}")

    return IngestionResult(
        ok=qdrant_ok and os_ok,
        chunks_count=len(chunks),
        embedded=embedded,
        skipped=skipped,
        qdrant_ok=qdrant_ok,
        opensearch_ok=os_ok,
        error="; ".join(errors) if errors else None,
    )
