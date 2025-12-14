from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.chunking import chunk_text_chars
from app.clients import RetrievalClient, StorageClient, VLMClient
from app.config import Settings, load_settings
from app.extraction import extract_text_non_vlm, normalize_to_pdf, pdf_to_page_pngs
from app.logging_setup import setup_json_logging
from app.models import ChunkMeta, Locator, ProcessRequest, ProcessResponse

logger = logging.getLogger("processor")

REQS = Counter("processor_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("processor_latency_seconds", "Latency", ["stage"])


class AppState:
    settings: Settings | None = None
    config_error: str | None = None
    storage: StorageClient | None = None
    retrieval: RetrievalClient | None = None
    vlm: VLMClient | None = None


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
    state.storage = StorageClient(base_url=str(state.settings.storage_url), timeout_s=state.settings.storage_timeout_s)
    state.retrieval = RetrievalClient(base_url=str(state.settings.retrieval_url), timeout_s=state.settings.retrieval_timeout_s)
    state.vlm = VLMClient(
        base_url=str(state.settings.vlm_base_url),
        api_key=state.settings.vlm_api_key.get_secret_value() if state.settings.vlm_api_key else None,
        model=state.settings.vlm_model,
        timeout_s=state.settings.vlm_timeout_s,
    )
    yield


app = FastAPI(title="Doc Processor", version="0.1.0", lifespan=lifespan)


@app.get("/v1/healthz")
async def healthz():
    return {"ok": True}


@app.get("/v1/readyz")
async def readyz(response: Response):
    if state.config_error:
        response.status_code = 503
        return {"ready": False, "config_error": state.config_error}
    ready = state.storage is not None and state.retrieval is not None and state.vlm is not None
    if not ready:
        response.status_code = 503
    return {"ready": ready}


@app.get("/v1/version")
async def version():
    if state.settings is None:
        return {"service": {"name": "doc-processor"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@app.get("/v1/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/process", status="503").inc()
        return ProcessResponse(ok=False, doc_id=req.document.doc_id, error="config_error", detail=state.config_error)
    assert state.settings is not None
    assert state.storage is not None
    assert state.retrieval is not None
    assert state.vlm is not None

    doc_id = req.document.doc_id

    degraded: list[str] = []
    partial = False

    # Fetch metadata + bytes from storage
    with LAT.labels("storage_meta").time():
        meta = await state.storage.get_metadata(doc_id=doc_id)
    if not meta:
        REQS.labels(endpoint="/v1/process", status="404").inc()
        return ProcessResponse(ok=False, doc_id=doc_id, error="not_found", detail="doc_id not found in storage")

    filename = meta.get("title") or meta.get("doc_id") or doc_id
    content_type = meta.get("content_type")

    with LAT.labels("storage_get").time():
        raw, header_ct = await state.storage.get_file(doc_id=doc_id)
    content_type = content_type or header_ct

    if not raw:
        REQS.labels(endpoint="/v1/process", status="400").inc()
        return ProcessResponse(ok=False, doc_id=doc_id, error="empty_file", detail="empty content from storage")

    # Decide path: PDF/DOC/DOCX -> VLM; XML -> parse; else -> utf-8 fallback.
    pdf_bytes, norm_ct = None, None
    try:
        with LAT.labels("normalize").time():
            pdf_bytes, norm_ct = normalize_to_pdf(raw, content_type, filename)
    except Exception as e:
        degraded.append("office_convert_failed")
        partial = True
        pdf_bytes = None
        logger.warning("normalize_to_pdf_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

    pages_text: list[str] = []
    pages = None

    if pdf_bytes is None:
        # Non-VLM fallback (XML/plain)
        with LAT.labels("extract_non_vlm").time():
            ed = extract_text_non_vlm(raw, content_type, filename)
        pages_text = ed.pages_text
        pages = len(pages_text)
        content_type = ed.content_type or content_type
        degraded.append("vlm_skipped")
    else:
        # VLM path: render pages -> ask VLM per page.
        with LAT.labels("pdf_render").time():
            pngs = pdf_to_page_pngs(
                pdf_bytes,
                max_pages=state.settings.max_pages,
                max_side_px=state.settings.max_image_side_px,
            )
        pages = len(pngs)
        if pages == 0:
            REQS.labels(endpoint="/v1/process", status="400").inc()
            return ProcessResponse(ok=False, doc_id=doc_id, content_type=norm_ct, error="no_pages", detail="no pages rendered")

        async def one(i: int, b: bytes) -> tuple[int, str]:
            t = await state.vlm.page_to_text(png_bytes=b)
            return (i, t)

        with LAT.labels("vlm").time():
            # bounded parallelism
            sem = asyncio.Semaphore(4)

            async def run_one(i: int, b: bytes):
                async with sem:
                    return await one(i, b)

            results = await asyncio.gather(*(run_one(i, b) for i, b in enumerate(pngs)), return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                degraded.append("vlm_page_failed")
                partial = True
                continue
            i, t = r
            if t:
                # 1-based page number in locator
                while len(pages_text) < i + 1:
                    pages_text.append("")
                pages_text[i] = t

        # Ensure list length
        if len(pages_text) < pages:
            pages_text.extend([""] * (pages - len(pages_text)))

        if all(not t.strip() for t in pages_text):
            REQS.labels(endpoint="/v1/process", status="400").inc()
            return ProcessResponse(
                ok=False,
                doc_id=doc_id,
                content_type=norm_ct,
                pages=pages,
                error="empty_text",
                detail="VLM returned empty text for all pages",
                partial=partial,
                degraded=degraded,
            )

        content_type = norm_ct

    # Build chunks (preserve page in locator when we have pages)
    chunks: list[ChunkMeta] = []
    global_idx = 0
    for pi, page_text in enumerate(pages_text):
        for part in chunk_text_chars(
            page_text,
            chunk_size=state.settings.chunk_size_chars,
            overlap=state.settings.chunk_overlap_chars,
        ):
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{doc_id}:{global_idx}",
                    doc_id=doc_id,
                    chunk_index=global_idx,
                    text=part,
                    locator=Locator(page=pi + 1) if pages and pages > 1 else None,
                    title=req.document.title,
                    uri=req.document.uri,
                    lang=req.document.lang,
                    tags=req.document.tags,
                    acl=req.document.acl,
                    source=req.document.source,
                    tenant_id=req.document.tenant_id,
                    project_id=req.document.project_id,
                )
            )
            global_idx += 1

    if not chunks:
        REQS.labels(endpoint="/v1/process", status="400").inc()
        return ProcessResponse(ok=False, doc_id=doc_id, error="no_chunks", detail="no text extracted/chunked")

    upsert_payload = {
        "mode": "chunks",
        "document": req.document.model_dump(exclude_none=True),
        "chunks": [c.model_dump(exclude_none=True) for c in chunks],
        "refresh": bool(req.refresh),
    }

    with LAT.labels("retrieval_upsert").time():
        retrieval_resp = await state.retrieval.index_upsert(payload=upsert_payload)

    REQS.labels(endpoint="/v1/process", status="200").inc()
    extracted_chars = sum(len(t) for t in pages_text if t)
    return ProcessResponse(
        ok=True,
        doc_id=doc_id,
        content_type=content_type,
        pages=pages,
        extracted_chars=extracted_chars,
        chunks=len(chunks),
        retrieval=retrieval_resp,
        partial=partial,
        degraded=degraded,
    )




