from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import unquote

from chunker import chunk_text_chars
from embed_caller import Embedder
from errors import NonRetryableError
from s3_events import S3EventInfo
from ingestion import ingest_chunks
from models import ChunkMeta, Locator
from processing import Settings, file_to_texts, VLMClient
from store import QdrantStore, BM25Store
from db_ops import DocumentEventDB

logger = logging.getLogger("data.processing.pipeline")


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
) -> list[ChunkMeta]:
    doc = meta.doc
    project = meta.project

    # Prefer project-level chunk settings, fall back to pipeline defaults
    chunk_size = int(project.get("chunk_size", settings.chunk_size_chars))
    chunk_overlap = int(project.get("chunk_overlap", settings.chunk_overlap_chars))

    # Parse comma-separated fields into lists
    tags = [t.strip() for t in doc.get("tags", "").split(",") if t.strip()]
    acl = [a.strip() for a in doc.get("acl", "").split(",") if a.strip()]

    chunks: list[ChunkMeta] = []
    global_idx = 0
    has_pages = len(texts) > 1

    for page_idx, page_text in enumerate(texts):
        if not page_text or not page_text.strip():
            continue

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
                    locator=Locator(page=page_idx + 1) if has_pages else None,
                    title=doc.get("title"),
                    source=doc.get("source"),
                    uri=doc.get("uri"),
                    lang=doc.get("lang") or project.get("language"),
                    tags=tags,
                    acl=acl,
                    project_id=project.get("project_id"),
                )
            )
            global_idx += 1
    logger.info(f"Created {chunks[0]}")
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


async def handle_object_created(*, info: S3EventInfo, s3_client, deps: PipelineDeps) -> None:

    project_id, doc_id = _parse_filename(info.key)
    await deps.event_db_docs.log_ingested(doc_id=doc_id, project_id=project_id, processing_time_ms=1000)

    # Download + extract
    dl = await download_from_s3(bucket=info.bucket, key=info.key, s3_client=s3_client)
    texts = await file_to_texts(
        raw=dl.file_bytes,
        content_type=dl.content_type,
        filename=dl.filename,
        vlm=deps.vlm,
        settings=deps.settings,
    )

    # Chunk + ingest
    project_id, doc_id = _parse_filename(info.key)
    chunks = create_chunks_from_texts(
        db_id=dl.meta.extra.get("doc-id"),
        doc_id=doc_id,
        texts=texts,
        settings=deps.settings,
        meta=dl.meta,
    )

    await ingest_chunks(
        chunks=chunks,
        embedder=deps.embedder,
        qdrant=deps.qdrant,
        opensearch=deps.opensearch,
        qdrant_collection=project_id,
        opensearch_index=project_id,
        embed_batch_size=deps.embed_batch_size,
        model=dl.meta.project.get("embedding_model"),
    )

    await deps.event_db_docs.log_processed(doc_id=doc_id, project_id=project_id, processing_time_ms=1000)


async def handle_object_removed(*, info: S3EventInfo, deps: PipelineDeps) -> None:
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
            raise errs[0]

    await deps.event_db_docs.log_deleted(
        doc_id=doc_id, project_id=project_id,
    )

async def handle_s3_event(*, info: S3EventInfo, s3_client, deps: PipelineDeps) -> None:
    if "_temp" in info.key:
        return

    match info.event_name.split(":"):
        case [_, "ObjectCreated", *_] if info.key:
            await handle_object_created(info=info, s3_client=s3_client, deps=deps)
        case [_, "ObjectRemoved", *_]:
            await handle_object_removed(info=info, deps=deps)

