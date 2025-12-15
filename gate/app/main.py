from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import json
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.clients import DocProcessorClient, DocumentStorageClient, LLMClient, RetrievalClient
from app.config import Settings, load_settings
from app.html_text import html_to_text
from app.logging_setup import setup_json_logging
from app.models import ChatRequest, ChatResponse, ContextChunk, Source
from app.queue import RabbitPublisher
from app.rag import build_context_blocks, build_messages

logger = logging.getLogger("gate")

REQS = Counter("gate_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("gate_latency_seconds", "Latency", ["stage"])

ING_PUB = Counter("gate_ingestion_tasks_published_total", "Ingestion tasks published", ["type", "status"])
ING_PUB_LAT = Histogram("gate_ingestion_publish_latency_seconds", "Publish latency", ["type"])


_CIT_RE = re.compile(r"\[\d+\]")


def _ensure_inline_citations(answer: str, refs: list[int]) -> str:
    """
    Best-effort fallback if the LLM forgets to cite.
    - If citations already present: keep as-is.
    - Else: append refs to each sentence end (over-citation is better than none for UI traceability).
    """
    if not answer:
        return answer
    if not refs:
        return answer
    if _CIT_RE.search(answer):
        return answer

    suffix = "".join([f"[{r}]" for r in refs])
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    parts = [p + ("" if p.endswith(suffix) else suffix) for p in parts if p]
    return " ".join(parts)


class AppState:
    settings: Settings | None = None
    config_error: str | None = None
    retrieval: RetrievalClient | None = None
    llm: LLMClient | None = None
    storage: DocumentStorageClient | None = None
    doc_processor: DocProcessorClient | None = None
    publisher: RabbitPublisher | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.settings = load_settings()
    except Exception as e:
        state.config_error = str(e)
        setup_json_logging("INFO")
        logger.error("config_error", extra={"extra": {"error": state.config_error}})
        yield
        return

    setup_json_logging(state.settings.log_level)
    state.retrieval = RetrievalClient(base_url=str(state.settings.retrieval_url), timeout_s=state.settings.retrieval_timeout_s)
    state.llm = LLMClient(
        provider=state.settings.llm_provider,
        base_url=str(state.settings.llm_base_url) if state.settings.llm_base_url else None,
        api_key=state.settings.llm_api_key.get_secret_value() if state.settings.llm_api_key else None,
        model=state.settings.llm_model,
        timeout_s=state.settings.llm_timeout_s,
    )
    if state.settings.storage_url:
        state.storage = DocumentStorageClient(base_url=str(state.settings.storage_url), timeout_s=state.settings.storage_timeout_s)
    if state.settings.doc_processor_url:
        state.doc_processor = DocProcessorClient(
            base_url=str(state.settings.doc_processor_url),
            timeout_s=state.settings.doc_processor_timeout_s,
        )
    if state.settings.rabbit_url:
        state.publisher = RabbitPublisher(url=str(state.settings.rabbit_url), queue_name=state.settings.rabbit_queue)
        try:
            await state.publisher.start()
        except Exception as e:
            # Degrade gracefully: still allow sync indexing path.
            logger.error("rabbit_publisher_init_failed", extra={"extra": {"error": str(e)}})
            state.publisher = None
    yield
    if state.publisher:
        await state.publisher.close()


app = FastAPI(title="RAG Gate", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configured in runtime after settings load; kept permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/healthz")
async def healthz():
    return {"ok": True}


@app.get("/v1/readyz")
async def readyz(response: Response):
    if state.config_error:
        response.status_code = 503
        return {"ready": False, "config_error": state.config_error}
    assert state.retrieval is not None
    r = await state.retrieval.readyz()
    ready = bool(r.get("ready"))
    if not ready:
        response.status_code = 503
    return {"ready": ready, "retrieval": r}


@app.get("/v1/version")
async def version():
    if state.settings is None:
        return {"service": {"name": "rag-gate"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@app.get("/v1/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/chat", status="503").inc()
        return ChatResponse(ok=False, answer=f"config_error: {state.config_error}", used_mode="none")

    assert state.settings is not None
    assert state.retrieval is not None
    assert state.llm is not None

    mode = payload.retrieval_mode or state.settings.retrieval_mode
    top_k = payload.top_k or state.settings.top_k

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    try:
        with LAT.labels("retrieval").time():
            retrieval_json = await state.retrieval.search(
                query=payload.query,
                mode=mode,
                top_k=top_k,
                filters=filters,
                acl=payload.acl,
                include_sources=payload.include_sources,
            )
    except httpx.TimeoutException:
        REQS.labels(endpoint="/v1/chat", status="504").inc()
        return ChatResponse(ok=False, answer="retrieval_timeout", used_mode=mode)
    except Exception as e:
        REQS.labels(endpoint="/v1/chat", status="502").inc()
        return ChatResponse(ok=False, answer=f"retrieval_error: {type(e).__name__}", used_mode=mode)

    hits = retrieval_json.get("hits") or []
    degraded = retrieval_json.get("degraded") or []
    partial = bool(retrieval_json.get("partial"))

    kept, context_text, sources = build_context_blocks(hits=hits, max_chars=state.settings.max_context_chars)
    messages = build_messages(query=payload.query, history=[m.model_dump() for m in payload.history], context_text=context_text)

    with LAT.labels("llm").time():
        answer = await state.llm.chat(messages=messages)
        if payload.include_sources:
            refs = [int(s.get("ref")) for s in (sources or []) if s.get("ref") is not None]
            answer = _ensure_inline_citations(answer, refs)

    ref_by_key = {(str(s.get("doc_id")), s.get("uri")): s.get("ref") for s in (sources or [])}

    ctx = []
    for h in kept:
        src = h.get("source") or {}
        key = (str(src.get("doc_id") or h.get("doc_id")), src.get("uri"))
        ctx.append(
            ContextChunk(
                chunk_id=str(h.get("chunk_id")),
                doc_id=str(h.get("doc_id")),
                text=h.get("text"),
                score=float(h.get("score") or 0.0),
                source=Source(
                    ref=ref_by_key.get(key),
                    doc_id=str(src.get("doc_id") or h.get("doc_id")),
                    title=src.get("title"),
                    uri=src.get("uri"),
                    locator=src.get("locator"),
                )
                if payload.include_sources
                else None,
            )
        )

    REQS.labels(endpoint="/v1/chat", status="200").inc()
    return ChatResponse(
        ok=True,
        answer=answer,
        used_mode=mode,
        degraded=list(degraded),
        partial=partial,
        context=ctx,
        sources=[Source(**s) for s in sources] if payload.include_sources else [],
        retrieval=retrieval_json,
    )


@app.post("/v1/documents/upload")
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

    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    acl_list = [a.strip() for a in (acl or "").split(",") if a.strip()]
    file_title = title or file.filename or "untitled"

    # Step 1: Store in document-storage if available
    storage_result = None
    storage_ok = False
    if state.storage:
        try:
            with LAT.labels("storage_store").time():
                storage_result = await state.storage.store_document(
                    file_content=raw,
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
            logger.info("document_stored", extra={"extra": {"doc_id": doc_id, "storage_id": storage_result.get("storage_id")}})
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
                "bytes": len(raw),
            }
        else:
            # Legacy path: decode bytes as UTF-8 and index as one document text.
            # If doc-processor is configured but storage didn't confirm the upload, we cannot use doc-processor
            # (it pulls bytes from storage). Fall back to direct indexing.
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


@app.get("/v1/documents")
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


@app.get("/v1/collections")
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


@app.delete("/v1/documents/{doc_id:path}")
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


@app.delete("/v1/documents")
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


@app.get("/v1/documents/{doc_id:path}/status")
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
            search_result = await state.retrieval.search(
                query="document",  # simple query to check if doc is indexed
                mode="hybrid",
                top_k=1,
                filters={"doc_ids": [doc_id]},
                acl=[],
                include_sources=False,
            )
            result["indexed"] = len(search_result.get("hits") or []) > 0
        except Exception as e:
            logger.warning("check_indexed_error", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

    REQS.labels(endpoint="/v1/documents/{doc_id}/status", status="200").inc()
    return {"ok": True, **result}


@app.post("/v1/chat/stream")
async def chat_stream(payload: ChatRequest):
    """Streaming chat endpoint using SSE."""
    if state.config_error:
        REQS.labels(endpoint="/v1/chat/stream", status="503").inc()
        async def error_stream():
            yield f"data: {json.dumps({'error': f'config_error: {state.config_error}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    assert state.settings is not None
    assert state.retrieval is not None
    assert state.llm is not None

    mode = payload.retrieval_mode or state.settings.retrieval_mode
    top_k = payload.top_k or state.settings.top_k

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    async def generate_stream():
        try:
            # Step 1: Retrieval
            try:
                with LAT.labels("retrieval").time():
                    retrieval_json = await state.retrieval.search(
                        query=payload.query,
                        mode=mode,
                        top_k=top_k,
                        filters=filters,
                        acl=payload.acl,
                        include_sources=payload.include_sources,
                    )
            except httpx.TimeoutException:
                yield f"data: {json.dumps({'error': 'retrieval_timeout'})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'error': f'retrieval_error: {type(e).__name__}'})}\n\n"
                return

            hits = retrieval_json.get("hits") or []
            degraded = retrieval_json.get("degraded") or []
            partial = bool(retrieval_json.get("partial"))

            kept, context_text, sources = build_context_blocks(hits=hits, max_chars=state.settings.max_context_chars)
            messages = build_messages(query=payload.query, history=[m.model_dump() for m in payload.history], context_text=context_text)

            # Build context chunks (same as /v1/chat response) and send retrieval payload for UI.
            ref_by_key = {(str(s.get("doc_id")), s.get("uri")): s.get("ref") for s in (sources or [])}
            ctx = []
            for h in kept:
                src = h.get("source") or {}
                key = (str(src.get("doc_id") or h.get("doc_id")), src.get("uri"))
                ctx.append(
                    ContextChunk(
                        chunk_id=str(h.get("chunk_id")),
                        doc_id=str(h.get("doc_id")),
                        text=h.get("text"),
                        score=float(h.get("score") or 0.0),
                        source=Source(
                            ref=ref_by_key.get(key),
                            doc_id=str(src.get("doc_id") or h.get("doc_id")),
                            title=src.get("title"),
                            uri=src.get("uri"),
                            locator=src.get("locator"),
                        )
                        if payload.include_sources
                        else None,
                    )
                )

            yield f"data: {json.dumps({'type': 'retrieval', 'mode': mode, 'partial': partial, 'degraded': list(degraded), 'context': [c.model_dump() for c in ctx], 'retrieval': retrieval_json})}\n\n"

            # Step 2: Stream LLM response
            answer_parts = []
            async for line in state.llm.chat_stream(messages=messages):
                # Parse OpenAI-compatible SSE format
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                answer_parts.append(content)
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue

            # Send final answer and metadata
            full_answer = "".join(answer_parts)
            if payload.include_sources:
                refs = [int(s.get("ref")) for s in (sources or []) if s.get("ref") is not None]
                full_answer = _ensure_inline_citations(full_answer, refs)
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'mode': mode, 'degraded': list(degraded), 'partial': partial, 'sources': sources if payload.include_sources else [], 'context': [c.model_dump() for c in ctx]})}\n\n"

            REQS.labels(endpoint="/v1/chat/stream", status="200").inc()
        except Exception as e:
            logger.error("chat_stream_error", extra={"extra": {"error": str(e)}})
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            REQS.labels(endpoint="/v1/chat/stream", status="500").inc()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


