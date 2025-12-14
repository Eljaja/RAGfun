from __future__ import annotations

import asyncio
import logging
from typing import Any

from opentelemetry import trace
from qdrant_client.http import models as qm

from app.chunking import chunk_text
from app.clients.embeddings import EmbeddingsClient
from app.clients.opensearch import OpenSearchClient
from app.clients.qdrant import QdrantFacade
from app.metrics import ERRS, LAT
from app.models import ChunkMeta, DocumentMeta, IndexDeleteRequest, IndexDeleteResponse, IndexUpsertRequest, IndexUpsertResponse
from app.qdrant_ids import point_id_for_chunk_id
from app.utils import sha256_hex

logger = logging.getLogger("rag.index")
tracer = trace.get_tracer("rag.index")


def _normalize_chunk(chunk: ChunkMeta, doc: DocumentMeta | None) -> ChunkMeta:
    if doc is None:
        return chunk
    # inherit doc fields if missing
    if chunk.uri is None:
        chunk.uri = doc.uri
    if chunk.title is None:
        chunk.title = doc.title
    if chunk.lang is None:
        chunk.lang = doc.lang
    if not chunk.tags:
        chunk.tags = list(doc.tags)
    if not chunk.acl:
        chunk.acl = list(doc.acl)
    if chunk.source is None:
        chunk.source = doc.source
    if chunk.tenant_id is None:
        chunk.tenant_id = doc.tenant_id
    if chunk.project_id is None:
        chunk.project_id = doc.project_id
    if chunk.created_at is None:
        chunk.created_at = doc.created_at
    if chunk.updated_at is None:
        chunk.updated_at = doc.updated_at
    return chunk


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"


async def upsert(
    *,
    req: IndexUpsertRequest,
    os_client: OpenSearchClient | None,
    qdrant: QdrantFacade | None,
    embedder: EmbeddingsClient,
    max_tokens: int,
    overlap_tokens: int,
) -> IndexUpsertResponse:
    if req.mode == "document":
        if not req.document or req.text is None:
            return IndexUpsertResponse(ok=False, partial=True, errors=[{"error": "document mode requires document+text"}])
        chs = []
        for idx, ctext, tcnt in chunk_text(req.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
            chs.append(
                ChunkMeta(
                    chunk_id=_chunk_id(req.document.doc_id, idx),
                    doc_id=req.document.doc_id,
                    chunk_index=idx,
                    text=ctext,
                    token_count=tcnt,
                )
            )
        req = IndexUpsertRequest(mode="chunks", document=req.document, chunks=chs, refresh=req.refresh)

    if not req.chunks:
        return IndexUpsertResponse(ok=False, partial=True, errors=[{"error": "no_chunks"}])

    # normalize + compute content_hash
    chunks: list[ChunkMeta] = []
    for c in req.chunks:
        c = _normalize_chunk(c, req.document)
        c.content_hash = c.content_hash or sha256_hex(c.text)
        chunks.append(c)

    upserted = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    partial = False

    # Idempotency:
    # - If Qdrant is enabled, it is the source of truth for "already embedded" (content_hash in payload).
    #   If point is missing in Qdrant we MUST embed, even if OpenSearch already has the same content_hash.
    # - If Qdrant is disabled, we fallback to OpenSearch only to avoid pointless work.
    need_embed: list[ChunkMeta] = []
    for c in chunks:
        existing_hash = None
        qdrant_has_point = False
        if qdrant is not None:
            try:
                ex = await asyncio.to_thread(qdrant.retrieve, c.chunk_id)
                if ex and isinstance(ex.get("payload"), dict):
                    qdrant_has_point = True
                    existing_hash = ex["payload"].get("content_hash")
            except Exception:
                existing_hash = None
        if qdrant is None and existing_hash is None and os_client is not None:
            try:
                exs = await asyncio.to_thread(os_client.get_by_id, c.chunk_id)
                if exs:
                    existing_hash = exs.get("content_hash")
            except Exception:
                existing_hash = None
        if qdrant is not None and qdrant_has_point and existing_hash == c.content_hash:
            skipped += 1
        elif qdrant is None and existing_hash == c.content_hash:
            skipped += 1
        else:
            need_embed.append(c)

    # embed batch
    vectors: dict[str, list[float]] = {}
    if need_embed:
        with tracer.start_as_current_span("embed"):
            with LAT.labels("embed").time():
                try:
                    embs = await embedder.embed([c.text for c in need_embed])
                    for c, v in zip(need_embed, embs, strict=False):
                        vectors[c.chunk_id] = v
                except Exception as e:
                    ERRS.labels("embed", type(e).__name__).inc()
                    partial = True
                    errors.append({"stage": "embed", "error": str(e)})

    # upsert OpenSearch
    if os_client is not None:
        with tracer.start_as_current_span("os_upsert"):
            with LAT.labels("os_upsert").time():
                try:
                    docs = []
                    for c in chunks:
                        docs.append(
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "chunk_index": c.chunk_index,
                                "text": c.text,
                                "title": c.title,
                                "source": c.source,
                                "tags": c.tags,
                                "lang": c.lang,
                                "uri": c.uri,
                                "acl": c.acl,
                                "tenant_id": c.tenant_id,
                                "project_id": c.project_id,
                                "created_at": c.created_at.isoformat() if c.created_at else None,
                                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                                "content_hash": c.content_hash,
                                "token_count": c.token_count,
                                "locator": c.locator.model_dump() if c.locator else None,
                            }
                        )
                    bulk = await asyncio.to_thread(os_client.bulk_upsert, docs, req.refresh)
                    if bulk.get("errors"):
                        partial = True
                        errors.append({"stage": "opensearch", "bulk_errors": True})
                except Exception as e:
                    ERRS.labels("os_upsert", type(e).__name__).inc()
                    partial = True
                    errors.append({"stage": "opensearch", "error": str(e)})
    else:
        partial = True
        errors.append({"stage": "opensearch", "error": "not_configured"})

    # upsert Qdrant
    if qdrant is not None and vectors:
        with tracer.start_as_current_span("qdrant_upsert"):
            with LAT.labels("qdrant_upsert").time():
                try:
                    pts: list[qm.PointStruct] = []
                    for c in need_embed:
                        v = vectors.get(c.chunk_id)
                        if v is None:
                            continue
                        payload = {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "chunk_index": c.chunk_index,
                            "text": c.text,
                            "title": c.title,
                            "source": c.source,
                            "tags": c.tags,
                            "lang": c.lang,
                            "uri": c.uri,
                            "acl": c.acl,
                            "tenant_id": c.tenant_id,
                            "project_id": c.project_id,
                            "created_at": c.created_at.isoformat() if c.created_at else None,
                            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                            "content_hash": c.content_hash,
                            "token_count": c.token_count,
                            "locator": c.locator.model_dump() if c.locator else None,
                        }
                        pts.append(qm.PointStruct(id=point_id_for_chunk_id(c.chunk_id), vector=v, payload=payload))
                    if pts:
                        await asyncio.to_thread(qdrant.upsert_points, pts)
                except Exception as e:
                    ERRS.labels("qdrant_upsert", type(e).__name__).inc()
                    partial = True
                    errors.append({"stage": "qdrant", "error": str(e)})
    elif qdrant is None:
        partial = True
        errors.append({"stage": "qdrant", "error": "not_configured"})

    upserted = len(chunks) - skipped
    return IndexUpsertResponse(ok=not partial, upserted=upserted, skipped_unchanged=skipped, partial=partial, errors=errors)


async def delete(
    *,
    req: IndexDeleteRequest,
    os_client: OpenSearchClient | None,
    qdrant: QdrantFacade | None,
) -> IndexDeleteResponse:
    errors: list[dict[str, Any]] = []
    partial = False
    deleted = 0

    if not req.doc_id and not req.chunk_id:
        return IndexDeleteResponse(ok=False, partial=True, errors=[{"error": "doc_id_or_chunk_id_required"}])

    if req.doc_id:
        # cascade
        if os_client is not None:
            try:
                r = await asyncio.to_thread(os_client.delete_by_doc_id, req.doc_id, req.refresh)
                deleted += int(r.get("deleted") or 0)
            except Exception as e:
                partial = True
                errors.append({"stage": "opensearch", "error": str(e)})
        else:
            partial = True
            errors.append({"stage": "opensearch", "error": "not_configured"})

        if qdrant is not None:
            try:
                await asyncio.to_thread(qdrant.delete_by_doc_id, req.doc_id)
            except Exception as e:
                partial = True
                errors.append({"stage": "qdrant", "error": str(e)})
        else:
            partial = True
            errors.append({"stage": "qdrant", "error": "not_configured"})

    if req.chunk_id:
        if os_client is not None:
            try:
                await asyncio.to_thread(os_client.delete_by_chunk_id, req.chunk_id, req.refresh)
                deleted += 1
            except Exception as e:
                partial = True
                errors.append({"stage": "opensearch", "error": str(e)})
        else:
            partial = True
            errors.append({"stage": "opensearch", "error": "not_configured"})

        if qdrant is not None:
            try:
                await asyncio.to_thread(qdrant.delete_by_chunk_id, req.chunk_id)
            except Exception as e:
                partial = True
                errors.append({"stage": "qdrant", "error": str(e)})
        else:
            partial = True
            errors.append({"stage": "qdrant", "error": "not_configured"})

    return IndexDeleteResponse(ok=not partial, deleted=deleted, partial=partial, errors=errors)


