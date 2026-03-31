from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import unquote

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_not_exception_type, before_sleep_log

from chunker import chunk_text_chars
from embed_caller import Embedder
from errors import NonRetryableError
from s3_events import S3EventInfo
from ingestion import ingest_chunks
from models import ChunkMeta, Locator
from processing import Settings, VLMClient, document_from_bytes, is_windowed
from store import QdrantStore, BM25Store
from db_ops import DocumentEventDB, DocumentEventType

logger = logging.getLogger("data.processing.pipeline")

_INGEST_RETRY = dict(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_not_exception_type(NonRetryableError),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def generate_doc_id(bucket: str, key: str) -> str:
    """Generate a stable document ID from bucket and key."""
    import hashlib
    return hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:16]


def create_chunks_from_texts(
    db_id: str,
    doc_id: str,
    texts: list[str],
    settings: Settings,
    meta: S3ObjectMeta,
    *,
    page_offset: int = 0,
    start_chunk_idx: int = 0,
    has_pages: bool | None = None,
) -> list[ChunkMeta]:
    """Turn page texts into indexable chunks.

    ``page_offset``
        0-based offset of the first page in *texts* within the full document.
        Used so that windowed calls produce correct global page numbers.
    ``start_chunk_idx``
        Chunk counter carried across windows so chunk_id / chunk_index stay
        unique and monotonically increasing for the whole document.
    ``has_pages``
        Explicit override; when ``None`` we infer from ``len(texts) > 1``.
    """
    doc_meta = meta.doc
    project = meta.project

    chunk_size = int(project.get("chunk_size", settings.chunk_size_chars))
    chunk_overlap = int(project.get("chunk_overlap", settings.chunk_overlap_chars))

    tags = [t.strip() for t in doc_meta.get("tags", "").split(",") if t.strip()]
    acl = [a.strip() for a in doc_meta.get("acl", "").split(",") if a.strip()]

    if has_pages is None:
        has_pages = len(texts) > 1

    chunks: list[ChunkMeta] = []
    global_idx = start_chunk_idx

    for local_idx, page_text in enumerate(texts):
        if not page_text or not page_text.strip():
            continue

        page_num = page_offset + local_idx + 1      # 1-based for display

        for part in chunk_text_chars(
            page_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        ):
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{doc_id}:{global_idx}",
                    db_id=db_id,
                    doc_id=doc_id,
                    chunk_index=global_idx,
                    text=part,
                    locator=Locator(page=page_num) if has_pages else None,
                    title=doc_meta.get("title"),
                    source=doc_meta.get("source"),
                    uri=doc_meta.get("uri"),
                    lang=doc_meta.get("lang") or project.get("language"),
                    tags=tags,
                    acl=acl,
                    project_id=project.get("project_id"),
                )
            )
            global_idx += 1

    if chunks:
        logger.info(f"Created {len(chunks)} chunks (first: {chunks[0].chunk_id})")
    return chunks


@dataclass(frozen=True)
class PipelineDeps:
    vlm: VLMClient
    settings: Settings
    embedder: Embedder
    qdrant: QdrantStore
    opensearch: BM25Store
    event_db_docs: DocumentEventDB
    qdrant_collection: str
    opensearch_index: str
    embed_batch_size: int = 32
    


@dataclass(frozen=True)
class S3ObjectMeta:
    """Parsed S3 user-defined metadata, split by prefix."""
    doc: dict[str, str]
    project: dict[str, str]
    extra: dict[str, str]


def parse_s3_metadata(raw: dict[str, str]) -> S3ObjectMeta:
    """Split flat S3 metadata into doc/project/extra buckets by prefix."""
    doc, project, extra = {}, {}, {}
    for k, v in raw.items():
        match k.split("__", 1):
            case ["doc", rest]:
                doc[rest] = v
            case ["project", rest]:
                project[rest] = v
            case _:
                extra[k] = v
    return S3ObjectMeta(doc=doc, project=project, extra=extra)


@dataclass(frozen=True)
class S3Download:
    file_bytes: bytes
    content_type: str | None
    filename: str
    meta: S3ObjectMeta


@dataclass(frozen=True)
class UserFacingError:
    stage: str
    error_code: str
    user_message: str
    retryable: bool


def _classify_user_error(*, stage: str, exc: Exception) -> UserFacingError:
    """Classify exceptions into stable, user-facing error metadata."""
    retryable = not isinstance(exc, NonRetryableError)

    by_stage: dict[str, tuple[str, str]] = {
        "download": (
            "source_read_failed",
            "Could not access the uploaded file from storage.",
        ),
        "extract": (
            "text_extraction_failed",
            "Could not extract text from the document.",
        ),
        "chunk": (
            "chunking_failed",
            "Could not prepare document content for indexing.",
        ),
        "index": (
            "indexing_failed",
            "Could not index the document for search.",
        ),
        "delete_index": (
            "delete_index_failed",
            "Could not remove indexed document data.",
        ),
    }
    error_code, user_message = by_stage.get(
        stage,
        ("processing_failed", "Document processing failed."),
    )
    if not retryable:
        error_code = f"{error_code}_non_retryable"

    return UserFacingError(
        stage=stage,
        error_code=error_code,
        user_message=user_message,
        retryable=retryable,
    )


async def _log_user_facing_error(
    *,
    deps: PipelineDeps,
    doc_id: str,
    project_id: str,
    stage: str,
    exc: Exception,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    err = _classify_user_error(stage=stage, exc=exc)
    metadata = {
        "stage": err.stage,
        "error_code": err.error_code,
        "user_message": err.user_message,
        "retryable": err.retryable,
        # Keep diagnostics compact and safe for user-facing views.
        "exception_type": type(exc).__name__,
        "internal_reason": str(exc)[:300],
    }
    if attempt is not None:
        metadata["attempt"] = attempt
    if max_attempts is not None:
        metadata["max_attempts"] = max_attempts

    await deps.event_db_docs.log_event(
        doc_id=doc_id,
        project_id=project_id,
        event_type=DocumentEventType.ERROR_PROCESSING,
        service_name="doc_processor_v2",
        metadata=metadata,
    )


async def download_from_s3(*, bucket: str, key: str, s3_client) -> S3Download:
    """
    Download object from S3 and return file bytes + parsed metadata.

    NOTE: key in events may be URL-encoded; we decode for actual S3 lookup.
    """
    decoded_key = unquote(key or "")
    response = await s3_client.get_object(Bucket=bucket, Key=decoded_key)
    file_bytes = await response["Body"].read()
    content_type = response.get("ContentType")
    meta = parse_s3_metadata(response.get("Metadata") or {})
    # print(meta)
    filename = decoded_key.split("/")[-1] if "/" in decoded_key else decoded_key
    return S3Download(file_bytes=file_bytes, content_type=content_type, filename=filename, meta=meta)


def _parse_filename(key: str) -> tuple[str, str]:
    """Extract (project_id, doc_id) from an S3 key like 'prefix/projectId_docId.ext'.

    Raises NonRetryableError when the filename doesn't match the expected format —
    retrying the same message will never fix a structural naming issue.
    """
    filename = key.split("/")[-1]
    try:
        project_id, doc_id = filename.split("_", 1)
    except ValueError as e:
        raise NonRetryableError(
            f"malformed_filename:{filename} (expected 'projectId_docId')",
            cause=e,
        ) from e
    if not project_id or not doc_id:
        raise NonRetryableError(
            f"malformed_filename:{filename} (empty project_id or doc_id)",
        )
    return project_id, doc_id


async def _extract_chunk_ingest_simple(
    *,
    doc: object,
    dl: S3Download,
    doc_id: str,
    project_id: str,
    deps: PipelineDeps,
    attempt: int | None,
    max_attempts: int | None,
) -> int:
    """Non-windowed path (HTML, XML, XLSX, plain text).
    Returns total chunks indexed."""
    try:
        texts = await doc.to_text()
    except Exception as e:
        await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                     stage="extract", exc=e, attempt=attempt, max_attempts=max_attempts)
        raise

    try:
        chunks = create_chunks_from_texts(
            db_id=dl.meta.extra.get("doc-id"), doc_id=doc_id,
            texts=texts, settings=deps.settings, meta=dl.meta,
        )
    except Exception as e:
        await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                     stage="chunk", exc=e, attempt=attempt, max_attempts=max_attempts)
        raise

    try:
        async for retry in AsyncRetrying(**_INGEST_RETRY):
            with retry:
                await ingest_chunks(
                    chunks=chunks, embedder=deps.embedder,
                    qdrant=deps.qdrant, opensearch=deps.opensearch,
                    qdrant_collection=project_id, opensearch_index=project_id,
                    embed_batch_size=deps.embed_batch_size,
                    model=dl.meta.project.get("embedding_model"),
                )
    except Exception as e:
        await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                     stage="index", exc=e, attempt=attempt, max_attempts=max_attempts)
        raise

    return len(chunks)


async def _extract_chunk_ingest_windowed(
    *,
    doc: object,
    dl: S3Download,
    doc_id: str,
    project_id: str,
    deps: PipelineDeps,
    attempt: int | None,
    max_attempts: int | None,
) -> int:
    """Windowed path for PDFs / Office docs.

    Each window: VLM extract -> chunk -> embed+ingest, keeping memory and
    indexer load bounded by ``settings.page_window`` pages at a time.
    Returns total chunks indexed.
    """
    total_chunks = 0
    chunk_idx = 0
    logged_stage: str | None = None

    try:
        async for page_offset, texts in doc.to_text_windowed():
            try:
                chunks = create_chunks_from_texts(
                    db_id=dl.meta.extra.get("doc-id"), doc_id=doc_id,
                    texts=texts, settings=deps.settings, meta=dl.meta,
                    page_offset=page_offset, start_chunk_idx=chunk_idx,
                    has_pages=True,
                )
            except Exception as e:
                logged_stage = "chunk"
                await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                             stage="chunk", exc=e, attempt=attempt, max_attempts=max_attempts)
                raise

            if not chunks:
                continue

            try:
                async for retry in AsyncRetrying(**_INGEST_RETRY):
                    with retry:
                        await ingest_chunks(
                            chunks=chunks, embedder=deps.embedder,
                            qdrant=deps.qdrant, opensearch=deps.opensearch,
                            qdrant_collection=project_id, opensearch_index=project_id,
                            embed_batch_size=deps.embed_batch_size,
                            model=dl.meta.project.get("embedding_model"),
                        )
            except Exception as e:
                logged_stage = "index"
                await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                             stage="index", exc=e, attempt=attempt, max_attempts=max_attempts)
                raise

            chunk_idx += len(chunks)
            total_chunks += len(chunks)
            logger.info(
                "window_done page_offset=%d chunks=%d total_so_far=%d",
                page_offset, len(chunks), total_chunks,
            )
    except Exception as e:
        if logged_stage is None:
            await _log_user_facing_error(deps=deps, doc_id=doc_id, project_id=project_id,
                                         stage="extract", exc=e, attempt=attempt, max_attempts=max_attempts)
        raise

    return total_chunks


async def handle_object_created(
    *,
    info: S3EventInfo,
    s3_client,
    deps: PipelineDeps,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:

    project_id, doc_id = _parse_filename(info.key)
    started_at = asyncio.get_running_loop().time()
    await deps.event_db_docs.log_ingested(
        doc_id=doc_id,
        project_id=project_id,
        processing_time_ms=0,
        source_bucket=info.bucket,
    )

    # Download
    try:
        dl = await download_from_s3(bucket=info.bucket, key=info.key, s3_client=s3_client)
    except Exception as e:
        await _log_user_facing_error(
            deps=deps, doc_id=doc_id, project_id=project_id,
            stage="download", exc=e, attempt=attempt, max_attempts=max_attempts,
        )
        raise

    doc = document_from_bytes(
        dl.file_bytes, dl.content_type, dl.filename,
        deps.vlm, deps.settings,
    )

    if is_windowed(doc):
        total_chunks = await _extract_chunk_ingest_windowed(
            doc=doc, dl=dl, doc_id=doc_id, project_id=project_id,
            deps=deps, attempt=attempt, max_attempts=max_attempts,
        )
    else:
        total_chunks = await _extract_chunk_ingest_simple(
            doc=doc, dl=dl, doc_id=doc_id, project_id=project_id,
            deps=deps, attempt=attempt, max_attempts=max_attempts,
        )

    await deps.event_db_docs.log_event(
        doc_id=doc_id,
        project_id=project_id,
        event_type=DocumentEventType.INDEXED,
        service_name="doc_processor_v2",
        metadata={"chunks_indexed": total_chunks},
    )

    elapsed_ms = int((asyncio.get_running_loop().time() - started_at) * 1000)
    await deps.event_db_docs.log_processed(
        doc_id=doc_id,
        project_id=project_id,
        processing_time_ms=elapsed_ms,
        chunks_count=total_chunks,
    )


async def handle_object_removed(
    *,
    info: S3EventInfo,
    deps: PipelineDeps,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    """Cleanup indexed data for a removed S3 object."""
    project_id, doc_id = _parse_filename(info.key)

    tasks = []
    if deps.qdrant is not None:
        tasks.append(deps.qdrant.delete_by_doc_id(doc_id, project_id))
    if deps.opensearch is not None:
        tasks.append(deps.opensearch.delete_by_doc_id(doc_id, project_id))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errs = [r for r in results if isinstance(r, Exception)]
        if errs:
            await _log_user_facing_error(
                deps=deps,
                doc_id=doc_id,
                project_id=project_id,
                stage="delete_index",
                exc=errs[0],
                attempt=attempt,
                max_attempts=max_attempts,
            )
            raise errs[0]

    await deps.event_db_docs.log_deleted(
        doc_id=doc_id, project_id=project_id,
    )

async def handle_s3_event(
    *,
    info: S3EventInfo,
    s3_client,
    deps: PipelineDeps,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    if "_temp" in info.key:
        return

    match info.event_name.split(":"):
        case [_, "ObjectCreated", *_] if info.key:
            await handle_object_created(
                info=info,
                s3_client=s3_client,
                deps=deps,
                attempt=attempt,
                max_attempts=max_attempts,
            )
        case [_, "ObjectRemoved", *_]:
            await handle_object_removed(
                info=info,
                deps=deps,
                attempt=attempt,
                max_attempts=max_attempts,
            )

