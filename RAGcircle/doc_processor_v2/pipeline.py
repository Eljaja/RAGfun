from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import unquote

from chunker import chunk_text_chars
from embed_caller import Embedder
from errors import NonRetryableError
from s3_events import S3EventInfo, is_supported_file, is_object_created, is_object_removed
from ingestion import ingest_chunks
from models import ChunkMeta, Locator
from processing import Settings, file_to_texts, VLMClient
from store import QdrantStore, BM25Store


def generate_doc_id(bucket: str, key: str) -> str:
    """Generate a stable document ID from bucket and key."""
    import hashlib
    return hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:16]


def create_chunks_from_texts(
    doc_id: str,
    texts: list[str],
    settings: Settings,
    *,
    source: str | None = None,
    uri: str | None = None,
) -> list[ChunkMeta]:
    chunks: list[ChunkMeta] = []
    global_idx = 0
    has_pages = len(texts) > 1

    for page_idx, page_text in enumerate(texts):
        if not page_text or not page_text.strip():
            continue

        for part in chunk_text_chars(
            page_text,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        ):
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{doc_id}:{global_idx}",
                    doc_id=doc_id,
                    chunk_index=global_idx,
                    text=part,
                    locator=Locator(page=page_idx + 1) if has_pages else None,
                    source=source,
                    uri=uri,
                )
            )
            global_idx += 1

    return chunks


@dataclass(frozen=True)
class PipelineDeps:
    vlm: VLMClient
    settings: Settings
    embedder: Embedder
    qdrant: QdrantStore
    opensearch: BM25Store
    qdrant_collection: str
    opensearch_index: str
    embed_batch_size: int = 32


async def download_from_s3(*, bucket: str, key: str, s3_client) -> tuple[bytes, str | None, str]:
    """
    Download object from S3 and return (bytes, content_type, filename).

    NOTE: key in events may be URL-encoded; we decode for actual S3 lookup.
    """
    decoded_key = unquote(key or "")
    response = await s3_client.get_object(Bucket=bucket, Key=decoded_key)
    file_bytes = await response["Body"].read()
    content_type = response.get("ContentType")
    filename = decoded_key.split("/")[-1] if "/" in decoded_key else decoded_key
    return file_bytes, content_type, filename


async def handle_object_created(*, info: S3EventInfo, s3_client, deps: PipelineDeps) -> None:
    
    # print(info)
    # if not is_supported_file(info.key):
    #     return

    print("HELLO")
    print(info)
    # Download + extract
    file_bytes, content_type, filename = await download_from_s3(bucket=info.bucket, key=info.key, s3_client=s3_client)
    texts = await file_to_texts(
        raw=file_bytes,
        content_type=content_type,
        filename=filename,
        vlm=deps.vlm,
        settings=deps.settings,
    )

    if not texts:
        # Treat as retryable: could be VLM outage or parsing issues.
        raise RuntimeError("extracted_no_text")

    # Chunk + ingest
    project_id, doc_id = filename.split("_")
    source = f"s3://{info.bucket}/{unquote(info.key or '')}"
    chunks = create_chunks_from_texts(
        doc_id=doc_id,
        texts=texts,
        settings=deps.settings,
        source=source,
        uri=source,
    )

    print("THIS IS IMPORTANT")



    #print("COLLECTIONS CREATED")

    result = await ingest_chunks(
        chunks=chunks,
        embedder=deps.embedder,
        qdrant=deps.qdrant,
        opensearch=deps.opensearch,
        qdrant_collection=project_id,
        opensearch_index=project_id,
        embed_batch_size=deps.embed_batch_size,
    )
    print("WELL")
    if not result.ok:
        # Make it retryable so we either succeed or end up in DLQ after max retries.
        raise RuntimeError(f"ingestion_failed:{result.error}")


async def handle_object_removed(*, info: S3EventInfo, deps: PipelineDeps) -> None:
    # Cleanup indexed data for this object.
    #
    # Important: depending on the event source/transport, `key` may be URL-encoded
    # (e.g. spaces as %20). Historically doc_id used event-provided key, but the
    # same object might later arrive with a decoded key. To make deletes robust,
    # we attempt both possibilities when they differ.
    doc_ids: list[str] = []
    doc_ids.append(info.key)
    #decoded_key = unquote(info.key or "")
    #if decoded_key != info.key:
    #    doc_ids.append(generate_doc_id(info.bucket, decoded_key))

    # De-dupe while preserving order
    seen: set[str] = set()
    doc_ids = [d for d in doc_ids if not (d in seen or seen.add(d))]

    async def _delete_one(doc_id: str) -> None:
        tasks = []

        project_id, document_id = doc_id.split("_")
        if deps.qdrant is not None:
            tasks.append(deps.qdrant.delete_by_doc_id(document_id, project_id))
        if deps.opensearch is not None:
            tasks.append(deps.opensearch.delete_by_doc_id(document_id, project_id))
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errs = [r for r in results if isinstance(r, Exception)]
        if errs:
            raise errs[0]

    for did in doc_ids:
        await _delete_one(did)


async def handle_s3_event(*, info: S3EventInfo, s3_client, deps: PipelineDeps) -> None:
    print(info)
    if is_object_created(info.event_name):
        if info.key and not "_temp" in info.key:
            await handle_object_created(info=info, s3_client=s3_client, deps=deps)
        # return
    if is_object_removed(info.event_name):
        # print(info)
        if not "_temp" in info.key:
            await handle_object_removed(info=info, deps=deps)
        # else:
        #     print("TEMP FOUND")
        #pass
    # Unknown events: ignore (non-error)
    return

