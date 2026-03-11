# services/store.py
from __future__ import annotations

import hashlib
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from embed_caller import Embedder
from errors import NonRetryableError
from hash_strategy import ExistingChunk, ContentHashStrategy, filter_chunks_needing_embedding

logger = logging.getLogger("data.processing.store")

# ─────────────────────────────────────────────────────────────
# Qdrant error classification
# ─────────────────────────────────────────────────────────────

# Status codes that will never succeed on retry
_QDRANT_NON_RETRYABLE_CODES = {400, 401, 403, 409, 422}


def _classify_qdrant_error(e: Exception, *, operation: str) -> Exception:
    """Return NonRetryableError when retrying is pointless, otherwise return *e* unchanged."""
    if isinstance(e, UnexpectedResponse) and e.status_code in _QDRANT_NON_RETRYABLE_CODES:
        return NonRetryableError(
            f"qdrant_{operation}:{e.status_code}:{e.reason_phrase}",
            cause=e,
        )
    # Timeouts, 5xx, connection refused → retryable (return as-is)
    return e


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
        try:
            exists = await self.client.collection_exists(collection)
            if not exists:
                await self.client.create_collection(
                    collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
        except Exception as e:
            raise _classify_qdrant_error(e, operation="ensure_collection") from e

    async def upsert(self, chunks: list, vectors: list[list[float]], collection: str):
        """Upsert ChunkMeta objects with their vectors."""
        from hash_strategy import sha256_hex

        points = [
            PointStruct(
                id=_chunk_to_point_id(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "db_id": c.db_id,
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
        try:
            await self.client.upsert(collection, points)
        except Exception as e:
            raise _classify_qdrant_error(e, operation="upsert") from e

    async def get_existing_chunks(self, chunk_ids: list[str], collection: str) -> dict[str, dict]:
        """
        Retrieve existing chunk metadata from Qdrant.
        Returns {chunk_id: {"exists": True, "content_hash": ...}} for found chunks.

        On failure, logs a warning and returns {} (the caller will re-embed everything,
        which is wasteful but not data-losing).
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
            logger.warning(
                "Failed to retrieve existing chunks from qdrant "
                "(collection=%s, %d chunk_ids) — will re-embed all",
                collection, len(chunk_ids),
                exc_info=True,
            )
            return {}

    async def ingest_chunks(
        self,
        chunks: list,
        embedder: Embedder,
        collection: str,
        *,
        batch_size: int = 32,
        model: str | None = None,
    ) -> None:
        """Embed and upsert chunks, skipping unchanged ones via content hash."""
        if not chunks:
            return

        # Check which chunks already exist with same content
        chunk_ids = [c.chunk_id for c in chunks]
        existing_raw = await self.get_existing_chunks(chunk_ids, collection)

        existing_by_id = {
            cid: ExistingChunk(
                exists=data.get("exists", False),
                content_hash=data.get("content_hash"),
            )
            for cid, data in existing_raw.items()
        }

        need_embed, _skipped = filter_chunks_needing_embedding(
            chunks, existing_by_id, ContentHashStrategy()
        )

        if not need_embed:
            return

        # Embed in batches
        all_vectors: list[list[float]] = []
        for i in range(0, len(need_embed), batch_size):
            texts = [c.text for c in need_embed[i : i + batch_size]]
            vectors = await embedder.embed(texts, model=model)
            all_vectors.extend(vectors)

        await self.upsert(need_embed, all_vectors, collection)

    async def delete_by_doc_id(self, doc_id: str, collection: str):
        """Delete all chunks for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        try:
            if not await self.client.collection_exists(collection):
                return

            await self.client.delete(
                collection,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            )
        except Exception as e:
            raise _classify_qdrant_error(e, operation="delete_by_doc_id") from e

    async def close(self):
        await self.client.close()

    async def delete_collection(self, collection: str):
        """Delete an entire collection."""
        if await self.client.collection_exists(collection):
            await self.client.delete_collection(collection)



# services/bm25_store.py
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import (
    AuthorizationException,
    AuthenticationException,
    RequestError,
    NotFoundError,
    ConflictError,
)

# ─────────────────────────────────────────────────────────────
# OpenSearch error classification
# ─────────────────────────────────────────────────────────────

_OS_NON_RETRYABLE = (AuthorizationException, AuthenticationException, RequestError, ConflictError)


def _classify_opensearch_error(e: Exception, *, operation: str) -> Exception:
    """Return NonRetryableError when retrying is pointless, otherwise return *e* unchanged."""
    if isinstance(e, _OS_NON_RETRYABLE):
        return NonRetryableError(
            f"opensearch_{operation}:{type(e).__name__}:{e}",
            cause=e,
        )
    if isinstance(e, NotFoundError) and operation != "delete":
        # Missing index on delete is fine; on upsert/search it's a config problem
        return NonRetryableError(
            f"opensearch_{operation}:index_not_found", cause=e,
        )
    # Timeouts, connection errors, 5xx → retryable
    return e


class BM25Store:
    def __init__(self, url: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)

    async def ensure_index(self, index: str):
        """Ensure an index exists."""
        try:
            exists = await self.client.indices.exists(index=index)
            logger.debug("ensure_index '%s': exists=%s", index, exists)
            if not exists:
                resp = await self.client.indices.create(index=index, body={
                    "settings": {
                        "number_of_replicas": 0,
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
                            "db_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "text": {"type": "text", "analyzer": "russian"},
                            "source": {"type": "keyword"},
                            "uri": {"type": "keyword"},
                            "page": {"type": "integer"},
                        }
                    }
                })
                logger.info("Created opensearch index '%s': %s", index, resp)
        except Exception as e:
            raise _classify_opensearch_error(e, operation="ensure_index") from e

    async def upsert(self, chunks: list, index: str):
        """Bulk upsert ChunkMeta objects."""
        if not chunks:
            logger.debug("upsert called with empty chunks, skipping")
            return

        # Use bulk API for efficiency
        actions = []
        for c in chunks:
            actions.append({"index": {"_index": index, "_id": c.chunk_id}})
            actions.append({
                "chunk_id": c.chunk_id,
                "db_id": c.db_id,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "source": c.source,
                "uri": c.uri,
                "page": c.locator.page if c.locator else None,
            })

        try:
            resp = await self.client.bulk(body=actions)
        except Exception as e:
            raise _classify_opensearch_error(e, operation="upsert") from e

        # Check for partial bulk failures (OpenSearch returns 200 with errors in body)
        if resp.get("errors"):
            failed_items = []
            non_retryable_detected = False
            for item in resp.get("items", []):
                act = item.get("index", {})
                err = act.get("error")
                if err:
                    err_type = err.get("type", "unknown") if isinstance(err, dict) else str(err)
                    err_reason = err.get("reason", "") if isinstance(err, dict) else ""
                    failed_items.append(f"id={act.get('_id')} type={err_type} reason={err_reason}")
                    # Mapping errors are non-retryable (schema mismatch)
                    if isinstance(err, dict) and err.get("type") in (
                        "mapper_parsing_exception",
                        "strict_dynamic_mapping_exception",
                        "illegal_argument_exception",
                    ):
                        non_retryable_detected = True
            logger.error(
                "OpenSearch bulk partial failure: %d/%d items failed in index '%s': %s",
                len(failed_items), len(chunks), index, "; ".join(failed_items[:10]),
            )
            if non_retryable_detected:
                raise NonRetryableError(
                    f"opensearch_upsert:bulk_mapping_error:{len(failed_items)}_items_failed",
                )

        await self.client.indices.refresh(index=index)

    async def delete_by_doc_id(self, doc_id: str, index: str):
        """Delete all chunks for a document."""
        try:
            if not await self.client.indices.exists(index=index):
                return

            await self.client.delete_by_query(
                index=index,
                body={"query": {"term": {"doc_id": doc_id}}}
            )
        except NotFoundError:
            # Index or doc already gone — idempotent, nothing to do
            return
        except Exception as e:
            raise _classify_opensearch_error(e, operation="delete") from e

    async def search(self, query: str, index: str, top_k: int = 20) -> list[tuple[str, float]]:
        try:
            resp = await self.client.search(
                index=index,
                body={"query": {"match": {"text": query}}, "size": top_k}
            )
            return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]
        except Exception as e:
            raise _classify_opensearch_error(e, operation="search") from e

    async def close(self):
        await self.client.close()

    async def get(self, doc_id: str, index: str) -> dict | None:
        resp = await self.client.get(index=index, id=doc_id)
        return resp["_source"]

    async def delete_index(self, index: str):
        """Delete an entire index."""
        if await self.client.indices.exists(index=index):
            await self.client.indices.delete(index=index)
