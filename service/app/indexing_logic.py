from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import TypeVar

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

T = TypeVar("T")


def _iter_batches(items: list[T], batch_size: int) -> list[list[T]]:
    if not items:
        return []
    bs = max(1, int(batch_size))
    return [items[i : i + bs] for i in range(0, len(items), bs)]


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


def _build_embedding_input(
    *,
    chunk: ChunkMeta,
    enabled: bool,
    max_header_chars: int,
) -> str:
    """
    Build text to embed. When enabled, prepend a compact contextual header (doc metadata + locator)
    to improve retrieval, while keeping stored chunk text unchanged.
    """
    txt = (chunk.text or "").strip()
    if not enabled:
        return txt

    header_lines: list[str] = []
    if chunk.title:
        header_lines.append(f"Title: {chunk.title}")
    if chunk.uri:
        header_lines.append(f"URI: {chunk.uri}")
    if chunk.source:
        header_lines.append(f"Source: {chunk.source}")
    if chunk.lang:
        header_lines.append(f"Lang: {chunk.lang}")
    if chunk.tenant_id:
        header_lines.append(f"Tenant: {chunk.tenant_id}")
    if chunk.project_id:
        header_lines.append(f"Project: {chunk.project_id}")

    if chunk.locator and chunk.locator.page is not None:
        header_lines.append(f"Page: {chunk.locator.page}")
    if chunk.locator and chunk.locator.anchor:
        header_lines.append(f"Anchor: {chunk.locator.anchor}")
    if chunk.locator and chunk.locator.heading_path:
        hp = " / ".join([h for h in (chunk.locator.heading_path or []) if h])
        if hp:
            header_lines.append(f"Heading: {hp}")

    header = "\n".join(header_lines).strip()
    if max_header_chars and max_header_chars > 0 and len(header) > int(max_header_chars):
        header = header[: int(max_header_chars)].rstrip() + "…"
    if not header:
        return txt
    return (header + "\n\n" + txt).strip()


async def upsert(
    *,
    req: IndexUpsertRequest,
    os_client: OpenSearchClient | None,
    qdrant: QdrantFacade | None,
    embedder: EmbeddingsClient,
    max_tokens: int,
    overlap_tokens: int,
    embedding_batch_size: int = 32,
    embedding_contextual_headers_enabled: bool = False,
    embedding_contextual_headers_max_chars: int = 400,
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

    # normalize + compute hashes
    chunks: list[ChunkMeta] = []
    for c in req.chunks:
        c = _normalize_chunk(c, req.document)
        c.content_hash = c.content_hash or sha256_hex(c.text)
        chunks.append(c)

    # embedding inputs + embed hashes (may include contextual headers)
    embed_input_by_id: dict[str, str] = {}
    embed_hash_by_id: dict[str, str] = {}
    for c in chunks:
        emb_in = _build_embedding_input(
            chunk=c,
            enabled=bool(embedding_contextual_headers_enabled),
            max_header_chars=int(embedding_contextual_headers_max_chars),
        )
        embed_input_by_id[c.chunk_id] = emb_in
        embed_hash_by_id[c.chunk_id] = sha256_hex(emb_in)

    upserted = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    partial = False

    # Idempotency:
    # - If Qdrant is enabled, it is the source of truth for "already embedded" (content_hash in payload).
    #   If point is missing in Qdrant we MUST embed, even if OpenSearch already has the same content_hash.
    # - If Qdrant is disabled, we fallback to OpenSearch only to avoid pointless work.
    need_embed: list[ChunkMeta] = []

    # Preload existing points from Qdrant in batches to avoid N network calls.
    # We only need payload fields: content_hash/embed_hash.
    qdrant_existing_by_chunk_id: dict[str, dict[str, Any]] = {}
    if qdrant is not None:
        try:
            for batch in _iter_batches([c.chunk_id for c in chunks], batch_size=256):
                ex_map = await asyncio.to_thread(qdrant.retrieve_many, batch)
                if ex_map:
                    qdrant_existing_by_chunk_id.update(ex_map)
        except Exception:
            qdrant_existing_by_chunk_id = {}

    for c in chunks:
        existing_hash = None
        existing_embed_hash = None
        qdrant_has_point = False
        if qdrant is not None:
            try:
                ex = qdrant_existing_by_chunk_id.get(c.chunk_id)
                if ex and isinstance(ex.get("payload"), dict):
                    qdrant_has_point = True
                    existing_hash = ex["payload"].get("content_hash")
                    existing_embed_hash = ex["payload"].get("embed_hash")
            except Exception:
                existing_hash = None
                existing_embed_hash = None
        if qdrant is None and existing_hash is None and os_client is not None:
            try:
                exs = await asyncio.to_thread(os_client.get_by_id, c.chunk_id)
                if exs:
                    existing_hash = exs.get("content_hash")
                    existing_embed_hash = exs.get("embed_hash")
            except Exception:
                existing_hash = None
                existing_embed_hash = None
        desired_embed_hash = embed_hash_by_id.get(c.chunk_id)
        if embedding_contextual_headers_enabled:
            # When embedding input changes, we must re-embed even if raw chunk text is unchanged.
            if qdrant is not None and qdrant_has_point and existing_embed_hash == desired_embed_hash:
                skipped += 1
            elif qdrant is None and existing_embed_hash == desired_embed_hash:
                skipped += 1
            else:
                need_embed.append(c)
        else:
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
                    # Avoid huge embedding requests: split into batches.
                    bs = max(1, int(embedding_batch_size))
                    for batch in _iter_batches(need_embed, batch_size=bs):
                        texts = [embed_input_by_id.get(c.chunk_id, c.text) for c in batch]
                        embs = await embedder.embed(texts)
                        for c, v in zip(batch, embs, strict=False):
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
                                "embed_hash": embed_hash_by_id.get(c.chunk_id),
                                "token_count": c.token_count,
                                "locator": c.locator.model_dump() if c.locator else None,
                            }
                        )
                    bulk = await asyncio.to_thread(os_client.bulk_upsert, docs, req.refresh)
                    if bulk.get("errors"):
                        partial = True
                        bulk_errors = []
                        for item in bulk.get("items", []):
                            for op_type, op_data in item.items():
                                if "error" in op_data:
                                    bulk_errors.append({"op": op_type, "error": op_data["error"]})
                        logger.error("opensearch_bulk_errors", extra={"errors": bulk_errors, "doc_ids": list(set(c.doc_id for c in chunks))})
                        errors.append({"stage": "opensearch", "bulk_errors": bulk_errors})
                    else:
                        logger.info("opensearch_bulk_success", extra={"docs_count": len(docs), "doc_ids": list(set(c.doc_id for c in chunks))})
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
                            "embed_hash": embed_hash_by_id.get(c.chunk_id),
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


