"""Document management endpoints."""

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.html_text import html_to_text
from app.metrics import ING_PUB, ING_PUB_LAT, LAT, REQS
from app.state import state

logger = logging.getLogger("gate")

router = APIRouter()


def _inc_count(d: dict[str, int], key: str, *, n: int = 1) -> None:
    """Increment counter in dict."""
    if not key:
        key = "unknown"
    try:
        d[key] = int(d.get(key) or 0) + int(n)
    except Exception:
        d[key] = int(n)


def _ingestion_state_from_doc(doc: dict[str, Any]) -> str:
    """Extract ingestion state from document metadata."""
    ing = (doc or {}).get("extra") or {}
    ing = ing.get("ingestion") if isinstance(ing, dict) else None
    state_val = (ing or {}).get("state")
    return str(state_val).strip().lower() if state_val else "unknown"


@router.post("/v1/documents/upload")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    title: str | None = Form(None),
    uri: str | None = Form(None),
    source: str | None = Form(None),
    lang: str | None = Form(None),
    tags: str | None = Form(None),  # comma-separated
    acl: str | None = Form(None),  # comma-separated
    tenant_id: str | None = Form(None),
    project_id: str | None = Form(None),
    refresh: bool = Form(False),
):
    """
    Uploads document to storage service first, then indexes it via retrieval service.
    """
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/upload", status="503").inc()
        logger.error("upload_config_error", extra={"extra": {"error": state.config_error}})
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    
    if not state.retrieval:
        REQS.labels(endpoint="/v1/documents/upload", status="503").inc()
        logger.error("upload_retrieval_unavailable")
        return {"ok": False, "error": "retrieval_unavailable", "detail": "Retrieval service is not available"}

    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    acl_list = [a.strip() for a in (acl or "").split(",") if a.strip()]
    file_title = title or file.filename or "untitled"

    # Step 1: Store in document-storage if available
    storage_result = None
    storage_ok = False
    stored_bytes: int | None = None
    duplicate: bool = False
    duplicate_of: str | None = None
    if state.storage:
        try:
            with LAT.labels("storage_store").time():
                storage_result = await state.storage.store_document(
                    # Stream file to document-storage (avoid buffering in gate).
                    file_content=file.file,
                    filename=file.filename or "unknown",
                    content_type=file.content_type,
                    doc_id=doc_id,
                    title=file_title,
                    uri=uri,
                    source=source,
                    lang=lang,
                    tags=tags_list,
                    acl=acl_list,
                    tenant_id=tenant_id,
                    project_id=project_id,
                )
            try:
                stored_bytes = int(storage_result.get("size")) if storage_result and storage_result.get("size") is not None else None
            except Exception:
                stored_bytes = None
            try:
                duplicate_of = (storage_result or {}).get("duplicate_of")
                duplicate = bool((storage_result or {}).get("duplicate")) or bool(duplicate_of)
            except Exception:
                duplicate_of = None
                duplicate = False
            logger.info(
                "document_stored",
                extra={
                    "extra": {
                        "doc_id": doc_id,
                        "storage_id": storage_result.get("storage_id") if storage_result else None,
                        "size": stored_bytes,
                        "duplicate": duplicate,
                        "duplicate_of": duplicate_of,
                    }
                },
            )
            storage_ok = True
        except httpx.TimeoutException as e:
            logger.warning("storage_store_timeout", extra={"extra": {"doc_id": doc_id, "error": str(e)}})
            # Continue with indexing even if storage fails
        except httpx.HTTPStatusError as e:
            logger.warning("storage_store_http_error", extra={"extra": {"doc_id": doc_id, "status": e.response.status_code, "error": e.response.text[:200]}})
            # Continue with indexing even if storage fails
        except Exception as e:
            logger.warning("storage_store_failed", extra={"extra": {"doc_id": doc_id, "error": str(e), "error_type": type(e).__name__}})
            # Continue with indexing even if storage fails

    # Step 2: Convert-to-text + index
    # Preferred: doc-processor (fetches from storage, uses Granite-Docling via vLLM, then indexes via retrieval).
    # Fallback: best-effort UTF-8 decode and direct indexing in retrieval.
    doc_meta = {
        "doc_id": doc_id,
        "source": source,
        "title": file_title,
        "uri": uri,
        "lang": lang,
        "tags": tags_list,
        "acl": acl_list,
        "tenant_id": tenant_id,
        "project_id": project_id,
    }

    def _legacy_extract_text(*, raw_bytes: bytes) -> str:
        decoded = raw_bytes.decode("utf-8", errors="replace")
        ct = (file.content_type or "").split(";")[0].strip().lower()
        name = (file.filename or "").lower()
        if ct in {"text/html", "application/xhtml+xml"} or name.endswith(".html") or name.endswith(".htm") or name.endswith(".xhtml"):
            text = html_to_text(decoded)
        else:
            text = decoded
        return text.strip()

    try:
        # NOTE about exact (bytes-level) deduplication:
        # document-storage may report this doc_id as a byte-identical duplicate of another doc_id.
        # We MUST still index by *this* doc_id, because doc_id is part of the retrieval contract
        # (filters, gold labels in benchmarks, user-facing identifiers, deletes, etc.).
        # Keep the dedup info in metadata for observability, but do not skip indexing.
        if storage_ok and state.storage and (duplicate or duplicate_of):
            now = time.time()
            try:
                await state.storage.patch_extra(
                    doc_id=doc_id,
                    patch={
                        "ingestion": {
                            "state": "processing",
                            "type": "index",
                            "doc_id": doc_id,
                            "updated_at": now,
                            "stage": "duplicate_detected",
                            "result": {"duplicate_of": duplicate_of},
                        }
                    },
                )
            except Exception:
                pass

        # Async ingestion path: store first, enqueue, return immediately.
        if state.publisher and state.storage and storage_ok:
            task_id = str(uuid.uuid4())
            now = time.time()
            try:
                # Mark queued in storage metadata (best-effort)
                await state.storage.patch_extra(
                    doc_id=doc_id,
                    patch={
                        "ingestion": {
                            "state": "queued",
                            "type": "index",
                            "task_id": task_id,
                            "doc_id": doc_id,
                            "queued_at": now,
                            "updated_at": now,
                            "attempt": 0,
                        }
                    },
                )
            except Exception as e:
                logger.warning("ingestion_patch_queued_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

            payload = {
                "task_id": task_id,
                "type": "index",
                "doc_id": doc_id,
                "document": doc_meta,
                "refresh": bool(refresh),
                "attempt": 0,
                "queued_at": now,
            }
            with ING_PUB_LAT.labels("index").time():
                try:
                    await state.publisher.publish(payload=payload)
                    ING_PUB.labels(type="index", status="ok").inc()
                except Exception as e:
                    ING_PUB.labels(type="index", status="error").inc()
                    logger.error("ingestion_publish_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})
                    REQS.labels(endpoint="/v1/documents/upload", status="502").inc()
                    return {"ok": False, "error": "ingestion_enqueue_failed", "detail": str(e), "storage": storage_result}

            REQS.labels(endpoint="/v1/documents/upload", status="202").inc()
            response.status_code = 202
            return {
                "ok": True,
                "accepted": True,
                "task_id": task_id,
                "doc_id": doc_id,
                "storage": storage_result,
                "filename": file.filename,
                "bytes": stored_bytes,
            }
        else:
            # Legacy path: decode bytes as UTF-8 and index as one document text.
            # If doc-processor is configured but storage didn't confirm the upload, we cannot use doc-processor
            # (it pulls bytes from storage). Fall back to direct indexing.
            try:
                # If we streamed the file to storage above, we need to rewind before reading for legacy indexing.
                await file.seek(0)
            except Exception:
                pass
            try:
                raw = await file.read()
            except Exception as e:
                REQS.labels(endpoint="/v1/documents/upload", status="400").inc()
                logger.error("upload_read_file_error", extra={"extra": {"doc_id": doc_id, "error": str(e)}})
                return {"ok": False, "error": "file_read_error", "detail": f"Failed to read file: {str(e)}"}

            if not raw:
                REQS.labels(endpoint="/v1/documents/upload", status="400").inc()
                logger.warning("upload_empty_file", extra={"extra": {"doc_id": doc_id, "filename": file.filename}})
                return {"ok": False, "error": "empty_file", "detail": "File is empty"}

            text = _legacy_extract_text(raw_bytes=raw)
            if not text:
                REQS.labels(endpoint="/v1/documents/upload", status="400").inc()
                logger.warning(
                    "empty_text_after_decode",
                    extra={"extra": {"doc_id": doc_id, "filename": file.filename, "size": len(raw)}},
                )
                return {"ok": False, "error": "empty_text", "detail": "File contains no text content (or is completely binary)"}
            payload = {"mode": "document", "document": doc_meta, "text": text, "refresh": bool(refresh)}
            with LAT.labels("index_upsert").time():
                r = await state.retrieval.index_upsert(payload=payload)
    except httpx.TimeoutException:
        REQS.labels(endpoint="/v1/documents/upload", status="504").inc()
        logger.error("upload_timeout", extra={"extra": {"doc_id": doc_id, "filename": file.filename}})
        return {"ok": False, "error": "retrieval_timeout", "detail": "Retrieval service timeout", "storage": storage_result}
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_detail = None
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = {"error": e.response.text[:500] if e.response.text else str(e)}
        REQS.labels(endpoint="/v1/documents/upload", status=str(status_code)).inc()
        logger.error(
            "upload_http_error",
            extra={
                "extra": {
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "status_code": status_code,
                    "detail": error_detail,
                }
            },
        )
        return {
            "ok": False,
            "error": "retrieval_http_error",
            "detail": error_detail,
            "status_code": status_code,
            "storage": storage_result,
        }
    except Exception as e:
        REQS.labels(endpoint="/v1/documents/upload", status="502").inc()
        logger.error(
            "upload_error",
            extra={
                "extra": {
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            },
        )
        return {"ok": False, "error": "retrieval_error", "detail": f"{type(e).__name__}: {str(e)}", "storage": storage_result}

    REQS.labels(endpoint="/v1/documents/upload", status="200").inc()
    return {"ok": True, "result": r, "storage": storage_result, "filename": file.filename, "bytes": len(raw)}


@router.get("/v1/documents")
async def list_documents(
    source: str | None = None,
    tags: str | None = None,
    lang: str | None = None,
    collections: str | None = None,  # comma-separated project_ids ("collections")
    limit: int = 100,
    offset: int = 0,
):
    """List documents from storage service."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    if not state.storage:
        REQS.labels(endpoint="/v1/documents", status="503").inc()
        return {"ok": False, "error": "storage_unavailable", "documents": [], "total": 0}

    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else None
    collection_ids = [c.strip() for c in (collections or "").split(",") if c.strip()] if collections else None

    try:
        result = await state.storage.search_documents(
            source=source,
            tags=tags_list,
            lang=lang,
            project_ids=collection_ids,
            limit=limit,
            offset=offset,
        )
        # Check indexing status (batch) for the current page
        docs = result.get("documents", []) or []
        indexed_set: set[str] = set()
        if state.retrieval and docs:
            doc_ids = [d.get("doc_id") for d in docs if d.get("doc_id")]
            try:
                exists = await state.retrieval.index_exists(doc_ids=doc_ids)
                indexed_set = set(exists.get("indexed_doc_ids") or [])
            except Exception:
                indexed_set = set()

        for doc in docs:
            doc_id = doc.get("doc_id")
            doc["indexed"] = bool(doc_id and doc_id in indexed_set)

        REQS.labels(endpoint="/v1/documents", status="200").inc()
        return result
    except Exception as e:
        REQS.labels(endpoint="/v1/documents", status="500").inc()
        logger.error("list_documents_error", extra={"extra": {"error": str(e)}})
        return {"ok": False, "error": str(e), "documents": [], "total": 0}


@router.get("/v1/documents/stats")
async def documents_stats(
    source: str | None = None,
    tags: str | None = None,
    lang: str | None = None,
    collections: str | None = None,  # comma-separated project_ids ("collections")
    page_size: int = 500,
    max_docs: int = 200_000,
):
    """
    Aggregate document stats server-side to avoid loading all docs in the UI.
    """
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/stats", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    if not state.storage:
        REQS.labels(endpoint="/v1/documents/stats", status="503").inc()
        return {"ok": False, "error": "storage_unavailable"}

    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else None
    collection_ids = [c.strip() for c in (collections or "").split(",") if c.strip()] if collections else None
    page_size = max(1, min(int(page_size), 2000))
    max_docs = max(1, min(int(max_docs), 1_000_000))

    stats: dict[str, Any] = {
        "ok": True,
        "total": 0,
        "docs_seen": 0,
        "bytes": 0,
        "partial": False,
        "indexed": 0,
        "not_indexed": 0,
        "ingestion": {
            "queued": 0,
            "processing": 0,
            "retrying": 0,
            "failed": 0,
            "completed": 0,
            "unknown": 0,
        },
        "by_content_type": {},
        "by_source": {},
        "by_lang": {},
        "by_collection": {},
    }

    offset = 0
    indexed_available = bool(state.retrieval)
    while stats["docs_seen"] < max_docs:
        result = await state.storage.search_documents(
            source=source,
            tags=tags_list,
            lang=lang,
            project_ids=collection_ids,
            limit=page_size,
            offset=offset,
        )
        docs = list(result.get("documents") or [])
        if "total" in result and isinstance(result.get("total"), int):
            stats["total"] = int(result.get("total") or 0)

        if not docs:
            break

        indexed_set: set[str] = set()
        if state.retrieval:
            doc_ids = [str(d.get("doc_id")) for d in docs if d.get("doc_id")]
            if doc_ids:
                try:
                    exists = await state.retrieval.index_exists(doc_ids=doc_ids)
                    indexed_set = set(str(x) for x in (exists.get("indexed_doc_ids") or []))
                except Exception:
                    indexed_set = set()
                    indexed_available = False

        for doc in docs:
            if stats["docs_seen"] >= max_docs:
                stats["partial"] = True
                break
            stats["docs_seen"] += 1
            try:
                stats["bytes"] += int(doc.get("size") or 0)
            except Exception:
                pass

            content_type = (doc.get("content_type") or "unknown").strip() or "unknown"
            _inc_count(stats["by_content_type"], content_type)

            source_val = (doc.get("source") or "unknown").strip() or "unknown"
            _inc_count(stats["by_source"], source_val)

            lang_val = (doc.get("lang") or "unknown").strip() or "unknown"
            _inc_count(stats["by_lang"], lang_val)

            collection_val = (doc.get("project_id") or "unassigned").strip() or "unassigned"
            _inc_count(stats["by_collection"], collection_val)

            ing_state = _ingestion_state_from_doc(doc)
            if ing_state in stats["ingestion"]:
                stats["ingestion"][ing_state] += 1
            else:
                stats["ingestion"]["unknown"] += 1

            doc_id = str(doc.get("doc_id") or "")
            if indexed_available and doc_id:
                if doc_id in indexed_set:
                    stats["indexed"] += 1
                else:
                    stats["not_indexed"] += 1

        offset += len(docs)
        if len(docs) < page_size:
            break

    if stats["total"] and stats["docs_seen"] < stats["total"] and stats["docs_seen"] >= max_docs:
        stats["partial"] = True

    stats["indexed_available"] = indexed_available
    REQS.labels(endpoint="/v1/documents/stats", status="200").inc()
    return stats


@router.get("/v1/collections")
async def collections(tenant_id: str | None = None, limit: int = 1000):
    """Proxy: list distinct collections (project_id) from document-storage."""
    if state.config_error:
        REQS.labels(endpoint="/v1/collections", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error, "collections": []}
    if not state.storage:
        REQS.labels(endpoint="/v1/collections", status="503").inc()
        return {"ok": False, "error": "storage_unavailable", "collections": []}
    try:
        r = await state.storage.list_collections(tenant_id=tenant_id, limit=limit)
        REQS.labels(endpoint="/v1/collections", status="200").inc()
        return r
    except Exception as e:
        REQS.labels(endpoint="/v1/collections", status="500").inc()
        logger.error("collections_error", extra={"extra": {"error": str(e)}})
        return {"ok": False, "error": str(e), "collections": []}


@router.delete("/v1/documents/{doc_id:path}")
async def delete_document(doc_id: str, response: Response):
    """
    Deletes document from storage (if configured) and removes its chunks from retrieval index.
    Best-effort: tries both, returns partial=true if one side fails.
    """
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    if not state.retrieval:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        return {"ok": False, "error": "retrieval_unavailable"}

    # Async delete: enqueue and return immediately.
    if state.publisher and state.storage:
        task_id = str(uuid.uuid4())
        now = time.time()
        try:
            await state.storage.patch_extra(
                doc_id=doc_id,
                patch={
                    "ingestion": {
                        "state": "queued",
                        "type": "delete",
                        "task_id": task_id,
                        "doc_id": doc_id,
                        "queued_at": now,
                        "updated_at": now,
                        "attempt": 0,
                    }
                },
            )
        except Exception:
            # might be already deleted or not in storage; still enqueue to clear retrieval
            pass

        payload = {"task_id": task_id, "type": "delete", "doc_id": doc_id, "attempt": 0, "queued_at": now}
        with ING_PUB_LAT.labels("delete").time():
            try:
                await state.publisher.publish(payload=payload)
                ING_PUB.labels(type="delete", status="ok").inc()
            except Exception as e:
                ING_PUB.labels(type="delete", status="error").inc()
                REQS.labels(endpoint="/v1/documents/{doc_id}", status="502").inc()
                return {"ok": False, "error": "ingestion_enqueue_failed", "detail": str(e)}

        REQS.labels(endpoint="/v1/documents/{doc_id}", status="202").inc()
        response.status_code = 202
        return {"ok": True, "accepted": True, "task_id": task_id, "doc_id": doc_id}

    storage_resp = None
    retrieval_resp = None
    partial = False
    degraded: list[str] = []

    # 1) Delete from storage (optional)
    if state.storage:
        try:
            with LAT.labels("storage_delete").time():
                storage_resp = await state.storage.delete_document(doc_id=doc_id)
        except httpx.HTTPStatusError as e:
            partial = True
            degraded.append("storage_delete_failed")
            storage_resp = {"ok": False, "status_code": e.response.status_code, "detail": e.response.text[:500]}
        except Exception as e:
            partial = True
            degraded.append("storage_delete_failed")
            storage_resp = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

    # 2) Delete from retrieval index
    try:
        with LAT.labels("retrieval_index_delete").time():
            retrieval_resp = await state.retrieval.index_delete(payload={"doc_id": doc_id, "refresh": True})
    except httpx.HTTPStatusError as e:
        partial = True
        degraded.append("retrieval_delete_failed")
        retrieval_resp = {"ok": False, "status_code": e.response.status_code, "detail": e.response.text[:500]}
    except Exception as e:
        partial = True
        degraded.append("retrieval_delete_failed")
        retrieval_resp = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

    status = "200" if not partial else "207"
    REQS.labels(endpoint="/v1/documents/{doc_id}", status=status).inc()
    return {
        "ok": retrieval_resp is not None and bool(retrieval_resp.get("ok")),
        "doc_id": doc_id,
        "partial": partial,
        "degraded": degraded,
        "storage": storage_resp,
        "retrieval": retrieval_resp,
    }


@router.delete("/v1/documents")
async def delete_all_documents(
    confirm: bool = False,
    batch_size: int = 200,
    concurrency: int = 10,
    max_batches: int = 10_000,
):
    """
    Deletes ALL documents from storage (if configured) and removes their chunks from retrieval index.
    Safety: requires `confirm=true` query param.

    Implementation note: document-storage doesn't expose "delete all", so we iterate pages and delete by doc_id.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required")

    if state.config_error:
        REQS.labels(endpoint="/v1/documents", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    if not state.storage:
        REQS.labels(endpoint="/v1/documents", status="503").inc()
        return {"ok": False, "error": "storage_unavailable"}
    if not state.retrieval:
        REQS.labels(endpoint="/v1/documents", status="503").inc()
        return {"ok": False, "error": "retrieval_unavailable"}

    # Clamp params to keep the API safe.
    batch_size = max(1, min(int(batch_size), 1000))
    concurrency = max(1, min(int(concurrency), 50))
    max_batches = max(1, min(int(max_batches), 1_000_000))

    sem = asyncio.Semaphore(concurrency)

    deleted = 0
    partial_count = 0
    degraded: set[str] = set()
    errors: list[dict[str, str]] = []

    async def _delete_one(*, doc_id: str, refresh: bool) -> dict[str, object]:
        async with sem:
            storage_resp = None
            retrieval_resp = None
            partial = False

            # 1) Delete from storage
            try:
                with LAT.labels("storage_delete_all").time():
                    storage_resp = await state.storage.delete_document(doc_id=doc_id)
            except httpx.HTTPStatusError as e:
                partial = True
                degraded.add("storage_delete_failed")
                storage_resp = {"ok": False, "status_code": e.response.status_code, "detail": (e.response.text or "")[:500]}
            except Exception as e:
                partial = True
                degraded.add("storage_delete_failed")
                storage_resp = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

            # 2) Delete from retrieval index
            try:
                with LAT.labels("retrieval_delete_all").time():
                    retrieval_resp = await state.retrieval.index_delete(payload={"doc_id": doc_id, "refresh": bool(refresh)})
            except httpx.HTTPStatusError as e:
                partial = True
                degraded.add("retrieval_delete_failed")
                retrieval_resp = {"ok": False, "status_code": e.response.status_code, "detail": (e.response.text or "")[:500]}
            except Exception as e:
                partial = True
                degraded.add("retrieval_delete_failed")
                retrieval_resp = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

            ok = bool((retrieval_resp or {}).get("ok"))
            return {"ok": ok, "doc_id": doc_id, "partial": partial, "storage": storage_resp, "retrieval": retrieval_resp}

    batches = 0
    try:
        while True:
            batches += 1
            if batches > max_batches:
                degraded.add("max_batches_reached")
                errors.append({"error": "max_batches_reached", "detail": f"max_batches={max_batches}"})
                break

            # Always fetch from offset=0 because deletion changes pagination.
            with LAT.labels("storage_list_for_delete_all").time():
                page = await state.storage.search_documents(limit=batch_size, offset=0)

            docs = page.get("documents", []) or []
            if not docs:
                break

            total = int(page.get("total") or len(docs))
            refresh_last_in_batch = total <= len(docs)

            doc_ids: list[str] = [d.get("doc_id") for d in docs if d.get("doc_id")]
            if not doc_ids:
                # Unexpected but prevents infinite loops.
                degraded.add("no_doc_ids_in_page")
                errors.append({"error": "no_doc_ids_in_page"})
                break

            tasks = []
            for i, doc_id in enumerate(doc_ids):
                refresh = bool(refresh_last_in_batch and i == len(doc_ids) - 1)
                tasks.append(_delete_one(doc_id=doc_id, refresh=refresh))

            results = await asyncio.gather(*tasks)
            deleted += len(results)
            for r in results:
                if r.get("partial"):
                    partial_count += 1
                if r.get("ok") is not True:
                    # Keep only a small sample to avoid huge responses.
                    if len(errors) < 50:
                        errors.append({"doc_id": str(r.get("doc_id")), "error": "delete_failed_or_partial"})
    except Exception as e:
        degraded.add("delete_all_failed")
        errors.append({"error": f"{type(e).__name__}: {str(e)}"})

    partial = bool(partial_count) or bool(degraded) or bool(errors)
    status = "200" if not partial else "207"
    REQS.labels(endpoint="/v1/documents", status=status).inc()
    return {
        "ok": True,
        "deleted": deleted,
        "partial": partial,
        "partial_count": partial_count,
        "degraded": sorted(degraded),
        "errors": errors,
    }


@router.get("/v1/documents/{doc_id:path}/status")
async def get_document_status(doc_id: str):
    """Get document status including storage and indexing status."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}/status", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}

    result = {"doc_id": doc_id, "stored": False, "indexed": False, "metadata": None, "ingestion": None}

    # Check storage
    if state.storage:
        try:
            meta = await state.storage.get_metadata(doc_id)
            if meta:
                result["stored"] = True
                result["metadata"] = meta
                result["ingestion"] = (meta.get("extra") or {}).get("ingestion")
        except Exception as e:
            logger.warning("get_metadata_error", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

    # Check indexing
    if state.retrieval:
        try:
            # Fast check: avoid embeddings/rerank/search by using index_exists endpoint.
            ex = await state.retrieval.index_exists(doc_ids=[doc_id])
            indexed_doc_ids = ex.get("indexed_doc_ids") or []
            result["indexed"] = str(doc_id) in set([str(x) for x in indexed_doc_ids])
        except Exception as e:
            logger.warning("check_indexed_error", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

    REQS.labels(endpoint="/v1/documents/{doc_id}/status", status="200").inc()
    return {"ok": True, **result}

