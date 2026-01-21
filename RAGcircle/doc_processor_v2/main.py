"""
Document Processor Service v2
- Consumes events from RabbitMQ (async)
- Downloads PDFs from S3 when uploaded
- Processes documents (extract text, metadata, etc.)
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from urllib.parse import unquote

import aio_pika
from aiobotocore.session import get_session
from fastapi import FastAPI

from processing import file_to_texts, VLMClient, Settings
from chunker import chunk_text_chars
from models import ChunkMeta, Locator
from embed_caller import Embedder
from store import QdrantStore, BM25Store
from ingestion import ingest_chunks
# ----------------------------
# Configuration
# ----------------------------

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5676"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}{RABBITMQ_VHOST}"

AMQP_EXCHANGE = os.getenv("AMQP_EXCHANGE", "amq.topic")
AMQP_BINDING_KEY = os.getenv("AMQP_BINDING_KEY", "rustfs.events")
AMQP_QUEUE = os.getenv("AMQP_QUEUE", "rustfs_events")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9004")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "rustfs")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "password")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

# VLM Configuration
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8123")
VLM_API_KEY = os.getenv("VLM_API_KEY", None)
VLM_MODEL = os.getenv("VLM_MODEL", "ibm-granite/granite-docling-258M")
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "120.0"))

# Processing Settings
PROC_MAX_PAGES = int(os.getenv("PROC_MAX_PAGES", "50"))
PROC_MAX_PX = int(os.getenv("PROC_MAX_PX", "2048"))
PROC_VLM_CONCURRENCY = int(os.getenv("PROC_VLM_CONCURRENCY", "4"))

# Chunking Settings
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# Embedding Service
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://localhost:8902")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "sentence-transformers/e5-base-v2")
EMBEDDER_DIM = int(os.getenv("EMBEDDER_DIM", "768"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:8903")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# OpenSearch
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:8905")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")


# ----------------------------
# Shared Clients (initialized at startup)
# ----------------------------
# TODO: REMOVE global mutable state - use dependency injection via Context dataclass
# TODO: Pass context explicitly to all functions instead of relying on globals

vlm_client: VLMClient | None = None
proc_settings: Settings | None = None
embedder: Embedder | None = None
qdrant_store: QdrantStore | None = None
opensearch_store: BM25Store | None = None


# ----------------------------
# Processing Logic
# ----------------------------


# TODO: Replace dict return with proper Result[ProcessedDocument, ProcessingError] type
# TODO: Split into: fetch_from_s3() -> FetchedDoc, then extract_texts() -> ExtractedDoc
# TODO: Remove print() side effects - return data, let caller decide logging
# TODO: Accept vlm_client and settings as parameters, not globals
async def process_document(bucket: str, key: str, s3_client) -> dict:
    """
    Download file from S3 and extract text using the processing pipeline.

    Returns:
        dict with processing results including extracted texts
    """
    print(f"[DOC] Downloading {bucket}/{key}...")  # TODO: Remove - side effect

    # Download from S3
    decoded_key = unquote(key)
    response = await s3_client.get_object(Bucket=bucket, Key=decoded_key)
    file_bytes = await response['Body'].read()
    file_size = len(file_bytes)

    print(f"[DOC] Downloaded {decoded_key} - Size: {file_size:,} bytes")  # TODO: Remove

    # Determine content type from S3 response or filename
    content_type = response.get("ContentType")
    filename = decoded_key.split(
        "/")[-1] if "/" in decoded_key else decoded_key

    # Extract text using the processing pipeline
    try:
        texts = await file_to_texts(
            raw=file_bytes,
            content_type=content_type,
            filename=filename,
            vlm=vlm_client,  # TODO: Pass as parameter
            settings=proc_settings,  # TODO: Pass as parameter
        )

        # TODO: Replace dict with ProcessedDocument dataclass
        result = {
            "bucket": bucket,
            "key": decoded_key,
            "size": file_size,
            "content_type": content_type,
            "status": "processed",  # TODO: Use enum or Result type
            "pages": len(texts),
            "texts": texts,
            "text_preview": texts[0][:500] if texts and texts[0] else None,
        }

        print(
            f"[DOC] ✓ Processed {decoded_key} - {len(texts)} text segment(s)")  # TODO: Remove
        return result

    except Exception as e:
        print(f"[DOC] ✗ Failed to process {bucket}/{decoded_key}: {e}")  # TODO: Remove
        # TODO: Return Result.err(ProcessingError(...)) instead
        return {
            "bucket": bucket,
            "key": decoded_key,
            "size": file_size,
            "status": "failed",
            "error": str(e),
        }


# ----------------------------
# Event Handlers
# ----------------------------

# TODO: Return Result[S3EventInfo, ParseError] instead of tuple with Nones
# TODO: Move to separate module (e.g., events.py or s3_events.py)
# TODO: Create S3EventInfo dataclass with bucket, key, event_type fields
def extract_s3_event_info(event: dict) -> tuple[str | None, str | None, str | None]:
    """
    Extract bucket, key, and event type from S3 event format.
    Supports both AWS S3 format and RustFS format.
    """
    try:
        # Try RustFS format first
        records = event.get("Records", [])
        if records:
            record = records[0]

            # RustFS format: bucket_name, object_name, event_name at record level
            bucket = record.get("bucket_name")
            key = record.get("object_name")
            event_name = record.get("event_name")

            if bucket and key:
                return bucket, key, event_name

            # AWS S3 format: nested in s3 object
            s3_info = record.get("s3", {})
            if s3_info:
                event_name = record.get("eventName", "")
                bucket = s3_info.get("bucket", {}).get("name")
                key = s3_info.get("object", {}).get("key")

                if bucket and key:
                    return bucket, key, event_name

        # Try top-level EventName and Key (some RustFS variants)
        event_name = event.get("EventName") or event.get("eventName")
        key_path = event.get("Key")
        if key_path and "/" in key_path:
            # Key format: "bucket/object"
            parts = key_path.split("/", 1)
            return parts[0], parts[1], event_name

        return None, None, None
    except Exception as e:
        print(f"[ERROR] Failed to parse event structure: {e}")  # TODO: Remove - return error
        return None, None, None


# Supported file extensions for processing
SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx",  # Documents
    ".txt", ".md",            # Plain text
    ".html", ".htm", ".xhtml",  # HTML
    ".xml",                    # XML
    ".xlsx",                   # Spreadsheets
}


def is_supported_file(filename: str) -> bool:
    """Check if a file is supported for processing."""
    name = filename.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


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
    """
    Convert extracted texts to ChunkMeta objects for storage.
    
    Args:
        doc_id: Document identifier
        texts: List of text segments (one per page/section)
        settings: Processing settings with chunk_size_chars and chunk_overlap_chars
        source: Optional source identifier (e.g., "s3://bucket/key")
        uri: Optional URI for the document
    
    Returns:
        List of ChunkMeta objects ready for Qdrant/OpenSearch ingestion
    """
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


# TODO: This function does WAY too much - split into:
#   1. parse_event() -> S3EventInfo
#   2. route_event() -> determines handler
#   3. handle_object_created() -> processes uploads
#   4. handle_object_removed() -> cleanup
# TODO: Replace with pipeline: parse |> route |> handle |> log_result
# TODO: Accept Context with all clients instead of globals + scattered params
# TODO: Return Result[ProcessingResult, Error] instead of None
async def handle_event(event: dict, s3_client, settings: Settings):
    """Process incoming event from RabbitMQ."""
    bucket, key, event_name = extract_s3_event_info(event)

    if not bucket or not key:
        print("[EVENT] ⚠️  Could not extract bucket/key from event")  # TODO: Remove
        print(f"[EVENT] Event keys: {list(event.keys())}")  # TODO: Remove
        if "Records" in event:
            print(f"[EVENT] Record keys: {list(event['Records'][0].keys())}")  # TODO: Remove
        return  # TODO: Return Result.err(ParseError(...))

    print(f"[EVENT] ✓ Parsed: {event_name}")  # TODO: Remove
    print(f"[EVENT]   Bucket: {bucket}")  # TODO: Remove
    print(f"[EVENT]   Key: {key}")  # TODO: Remove

    # Handle ObjectCreated events for supported file types
    # TODO: Use match/case or event type enum instead of string matching
    if "ObjectCreated" in event_name:
        if is_supported_file(key):
            print(f"[DOC] New document detected: {bucket}/{key}")  # TODO: Remove
            result = await process_document(bucket, key, s3_client)

            # TODO: Use pattern matching on Result type instead of dict.get()
            if result.get("status") == "processed":
                # Generate doc_id and create chunks
                doc_id = generate_doc_id(bucket, key)
                texts = result.get("texts", [])
                source = f"s3://{bucket}/{result.get('key', key)}"

                chunks = create_chunks_from_texts(
                    doc_id=doc_id,
                    texts=texts,
                    settings=settings,
                    source=source,
                    uri=source,
                )

                print(f"[CHUNK] ✓ Created {len(chunks)} chunks from {len(texts)} page(s)")  # TODO: Remove

                # Ingest to Qdrant + OpenSearch
                # TODO: Use globals embedder, qdrant_store, opensearch_store - pass via Context
                result = await ingest_chunks(
                    chunks=chunks,
                    embedder=embedder,  # TODO: From context
                    qdrant=qdrant_store,  # TODO: From context
                    opensearch=opensearch_store,  # TODO: From context
                    embed_batch_size=EMBED_BATCH_SIZE,  # TODO: From settings
                )

                if result.ok:
                    print(f"[INGEST] ✓ doc_id={doc_id}: {result.embedded} embedded, {result.skipped} skipped (unchanged)")  # TODO: Remove
                else:
                    print(f"[INGEST] ⚠️ Partial failure for doc_id={doc_id}: {result.error}")  # TODO: Remove

            else:
                print(f"[DOC] ✗ Processing failed: {result.get('error', 'unknown')}")  # TODO: Remove

        else:
            print(f"[EVENT] Ignoring unsupported file type: {key}")  # TODO: Remove

    elif "ObjectRemoved" in event_name:
        print(f"[EVENT] Object removed: {bucket}/{key}")  # TODO: Remove
        # TODO: Implement cleanup - delete from Qdrant/OpenSearch by doc_id

    else:
        print(f"[EVENT] Ignoring event: {event_name}")  # TODO: Remove


# TODO: Consider dead-letter queue for failed messages
# TODO: Add structured logging instead of print
# TODO: Pass full Context instead of individual params
async def on_message(message: aio_pika.IncomingMessage, s3_client, settings: Settings):
    """Handle incoming RabbitMQ message."""
    async with message.process():
        try:
            event = json.loads(message.body.decode("ascii"))
            print(f"[AMQP] Received event: {message.body.decode('ascii')[:200]}...")  # TODO: Remove

            if isinstance(event, dict):
                await handle_event(event, s3_client, settings)
            else:
                print("[AMQP] Skipping non-dict event")  # TODO: Remove

        except json.JSONDecodeError:
            print("[AMQP] Failed to parse JSON message")  # TODO: Log properly, send to DLQ
        except Exception as e:
            print(f"[ERROR] Failed to process message: {e}")  # TODO: Log properly, send to DLQ


# ----------------------------
# RabbitMQ Consumer
# ----------------------------
# TODO: Move to separate module (e.g., consumer.py or amqp.py)
# TODO: Accept Context instead of individual params
# TODO: Add backoff strategy for reconnection (exponential backoff)

async def consume_rabbitmq(s3_client, settings: Settings):
    """Async RabbitMQ consumer with reconnection logic."""
    while True:
        try:
            print(f"[AMQP] Connecting to {RABBITMQ_HOST}:{RABBITMQ_PORT}...")  # TODO: Remove

            connection = await aio_pika.connect_robust(
                RABBITMQ_URL,
                heartbeat=30,
            )

            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)

            queue = await channel.declare_queue(AMQP_QUEUE, durable=True)
            await queue.bind(exchange=AMQP_EXCHANGE, routing_key=AMQP_BINDING_KEY)

            print(f"[AMQP] ✓ Connected - Listening on queue={AMQP_QUEUE}")  # TODO: Remove
            print(f"[AMQP]   Exchange: {AMQP_EXCHANGE}")  # TODO: Remove
            print(f"[AMQP]   Routing Key: {AMQP_BINDING_KEY}")  # TODO: Remove

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    await on_message(message, s3_client, settings)

        except asyncio.CancelledError:
            print("[AMQP] Consumer cancelled, shutting down...")  # TODO: Remove
            break
        except Exception as e:
            print(f"[AMQP] Connection error: {e}")  # TODO: Log properly
            print("[AMQP] Reconnecting in 3 seconds...")  # TODO: Remove
            await asyncio.sleep(3)  # TODO: Exponential backoff


# ----------------------------
# FastAPI Application
# ----------------------------
# TODO: Create Context dataclass that holds all clients and settings
# TODO: Build context once, pass to consumer instead of using globals
# TODO: Consider factory functions for creating clients from config

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup clients and start RabbitMQ consumer."""
    # TODO: REMOVE all global mutations - build Context and pass explicitly
    global vlm_client, proc_settings, embedder, qdrant_store, opensearch_store

    print("[STARTUP] Initializing document processor v2...")  # TODO: Use logger

    # TODO: Extract to create_vlm_client(config) -> VLMClient
    vlm_client = VLMClient(
        base_url=VLM_BASE_URL,
        api_key=VLM_API_KEY,
        model=VLM_MODEL,
        timeout_s=VLM_TIMEOUT,
    )
    print(f"[STARTUP] ✓ VLM client: {VLM_BASE_URL} (model={VLM_MODEL})")  # TODO: Use logger

    # TODO: Load settings from config/env in one place
    proc_settings = Settings(
        max_pages=PROC_MAX_PAGES,
        max_px=PROC_MAX_PX,
        vlm_concurrency=PROC_VLM_CONCURRENCY,
        chunk_size_chars=CHUNK_SIZE_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
    )
    print(f"[STARTUP] ✓ Settings: chunk_size={CHUNK_SIZE_CHARS}, overlap={CHUNK_OVERLAP_CHARS}")  # TODO: Use logger

    # TODO: Extract to create_embedder(config) -> Embedder
    embedder = Embedder(base_url=EMBEDDER_URL, model=EMBEDDER_MODEL)
    print(f"[STARTUP] ✓ Embedder: {EMBEDDER_URL} (model={EMBEDDER_MODEL})")  # TODO: Use logger

    # TODO: Extract to create_qdrant(config) -> QdrantStore
    qdrant_store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION, dimension=EMBEDDER_DIM)
    await qdrant_store.ensure_collection()
    print(f"[STARTUP] ✓ Qdrant: {QDRANT_URL} (collection={QDRANT_COLLECTION})")  # TODO: Use logger

    # TODO: Extract to create_opensearch(config) -> BM25Store
    opensearch_store = BM25Store(url=OPENSEARCH_URL, index=OPENSEARCH_INDEX)
    await opensearch_store.ensure_index()
    print(f"[STARTUP] ✓ OpenSearch: {OPENSEARCH_URL} (index={OPENSEARCH_INDEX})")  # TODO: Use logger

    # TODO: Extract to create_s3_client(config) -> S3Client
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    ) as s3_client:
        print(f"[STARTUP] ✓ S3 client: {S3_ENDPOINT}")  # TODO: Use logger

        # TODO: Build Context here and pass to consumer:
        # ctx = Context(vlm=vlm_client, embedder=embedder, qdrant=qdrant_store, ...)
        # consumer_task = asyncio.create_task(consume_rabbitmq(ctx))
        consumer_task = asyncio.create_task(consume_rabbitmq(s3_client, proc_settings))
        print("[STARTUP] ✓ RabbitMQ consumer started")  # TODO: Use logger
        print("[STARTUP] Ready to process documents!")  # TODO: Use logger

        yield

        # Cleanup
        print("[SHUTDOWN] Stopping services...")  # TODO: Use logger
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        await embedder.close()
        await qdrant_store.close()
        await opensearch_store.close()
        print("[SHUTDOWN] ✓ Complete")  # TODO: Use logger


app = FastAPI(
    lifespan=lifespan,
    title="Document Processor v2",
    description="Async document processing service - consumes S3 events from RabbitMQ"
)


@app.get("/")
async def root():
    return {
        "service": "doc-processor-v2",
        "status": "running",
        "description": "Async document processing from RabbitMQ events"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "doc-processor-v2"
    }


# Run with:
# uvicorn main:app --host 0.0.0.0 --port 9998
