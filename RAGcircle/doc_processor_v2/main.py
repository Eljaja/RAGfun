"""
Document Processor Service v2
- Consumes events from RabbitMQ (async)
- Downloads PDFs from S3 when uploaded
- Processes documents (extract text, metadata, etc.)
"""

import asyncio
import json
from multiprocessing import process
import os
from contextlib import asynccontextmanager

import aio_pika
from aiobotocore.session import get_session
from fastapi import FastAPI

import tiktoken
import semchunk
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


# ----------------------------
# PDF Processing Logic
# ----------------------------
from dataclasses import dataclass







@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int


from embed_caller import Embedder
from store import QdrantStore , BM25Store

em = Embedder("http://localhost:8902", model="sentence-transformers/e5-base-v2")

bm25 = BM25Store("http://localhost:8905", "basa2")

store = QdrantStore("http://localhost:8903", "basa4", 768)
# await store.ensure_collection()

async def process_pdf(bucket: str, key: str, s3_client) -> dict:
    from urllib.parse import unquote
    """
    Download PDF from S3 and process it.
    Simple processing: count bytes, could add text extraction, etc.
    """
    #try:
    print(f"[PDF] Downloading {bucket}/{key}...")
    
    # Download from S3
    key = unquote(key)
    response = await s3_client.get_object(Bucket=bucket, Key=key)
    pdf_data = await response['Body'].read()
    
    file_size = len(pdf_data)
    print(f"[PDF] Downloaded {key} - Size: {file_size:,} bytes")
    
    # TODO: Add real PDF processing here
    # - Extract text with pypdf/pdfplumber
    # - Extract metadata
    # - Generate thumbnails
    # - OCR if needed
    # - Store results in database

    encoding = "cl100k_base"
    enc = tiktoken.get_encoding(encoding)
    chunk_size = 512
    chunker = semchunk.chunkerify(lambda t: len(enc.encode(t)), chunk_size)

    # print(pdf_data)
    chunks = chunker(pdf_data.decode('utf-8'))

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print("****")
        print(chunk)
        print("****")
        ch = Chunk(text=chunk, source_id="1234", chunk_index=i)
        processed_chunks.append(ch)
        vectors = await em.embed(texts=[chunk])
        await store.upsert([ch], vectors)
    #print(chunks)

    await bm25.upsert(processed_chunks)

    # for chunk in chunks:
    #     print("----")
    #     print(chunk)

    # print([pdf_data.decode('utf-8')[offset[0]:offset[1]] for offset in offsets])
    #print(list(chunks))
    
    result = {
        "bucket": bucket,
        "key": key,
        "size": file_size,
        "status": "processed",
        "pages": None,  # TODO: extract page count
        "text_preview": None,  # TODO: extract first N chars
    }
    
    print(f"[PDF] ✓ Processed {key}")
    return result
        
    # except Exception as e:
    #     print(f"[PDF] ✗ Failed to process {bucket}/{key}: {e}")
    #     return {
    #         "bucket": bucket,
    #         "key": key,
    #         "status": "failed",
    #         "error": str(e)
    #     }


# ----------------------------
# Event Handlers
# ----------------------------

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
        print(f"[ERROR] Failed to parse event structure: {e}")
        return None, None, None


async def handle_event(event: dict, s3_client):

    from models import parse_rustfs_event

    parse_rustfs_event(event)
    """Process incoming event from RabbitMQ"""
    bucket, key, event_name = extract_s3_event_info(event)
    
    if not bucket or not key:
        print("[EVENT] ⚠️  Could not extract bucket/key from event")
        print(f"[EVENT] Event keys: {list(event.keys())}")
        if "Records" in event:
            print(f"[EVENT] Record keys: {list(event['Records'][0].keys())}")
        return
    
    print(f"[EVENT] ✓ Parsed: {event_name}")
    print(f"[EVENT]   Bucket: {bucket}")
    print(f"[EVENT]   Key: {key}")
    
    # Handle ObjectCreated events for PDFs
    if "ObjectCreated" in event_name and key.lower().endswith(".txt"):
        print(f"[TXT] New TXT detected: {bucket}/{key}")
        _result = await process_pdf(bucket, key, s3_client)
        
        # TODO: Store processing result in database
        # TODO: Publish processing result to another queue
        # TODO: Send notification if needed
        
    elif "ObjectRemoved" in event_name:
        print(f"[EVENT] Object removed: {bucket}/{key}")
        # TODO: Clean up associated data
        
    else:
        print("[EVENT] Ignoring non-PDF or non-create event")


async def on_message(message: aio_pika.IncomingMessage, s3_client):
    """Handle incoming RabbitMQ message"""
    async with message.process():
        try:
            #raw = message.body.decode("utf-8", errors="replace")
            event = json.loads(message.body.decode("ascii")) # if raw.strip().startswith(("{", "[")) else {"raw": raw}
            
            # print(f"[AMQP] Message received (routing_key={message.routing_key})")
            # print(f"[AMQP] Event preview: {json.dumps(event, indent=4)}")

            print(message.body.decode("ascii"))

            
            if isinstance(event, dict):
                await handle_event(event, s3_client)
            else:
                print("[AMQP] Skipping non-dict event")
                
        except json.JSONDecodeError:
            print("[AMQP] Failed to parse JSON message")
        except Exception as e:
           print(f"[ERROR] Failed to process message: {e}")


# ----------------------------
# RabbitMQ Consumer
# ----------------------------

async def consume_rabbitmq(s3_client):
    """Async RabbitMQ consumer with reconnection logic"""
    while True:
        try:
            print(f"[AMQP] Connecting to {RABBITMQ_HOST}:{RABBITMQ_PORT}...")
            
            # Connect with automatic reconnection
            connection = await aio_pika.connect_robust(
                RABBITMQ_URL,
                heartbeat=30,
            )
            
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)
            
            # Declare queue and bind to exchange
            queue = await channel.declare_queue(AMQP_QUEUE, durable=True)
            await queue.bind(exchange=AMQP_EXCHANGE, routing_key=AMQP_BINDING_KEY)
            
            print(f"[AMQP] ✓ Connected - Listening on queue={AMQP_QUEUE}")
            print(f"[AMQP]   Exchange: {AMQP_EXCHANGE}")
            print(f"[AMQP]   Routing Key: {AMQP_BINDING_KEY}")
            
            # Consume messages
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    await on_message(message, s3_client)
                    
        except asyncio.CancelledError:
            print("[AMQP] Consumer cancelled, shutting down...")
            break
        # except Exception as e:
        #     print(f"[AMQP] Connection error: {e}")
            
        #     print("[AMQP] Reconnecting in 3 seconds...")
        #     await asyncio.sleep(3)


# ----------------------------
# FastAPI Application
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup S3 client and start RabbitMQ consumer"""
    print("[STARTUP] Initializing document processor v2...")
    
    #awful but hanging there
    await store.ensure_collection()
    
    
    # Create S3 client
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    ) as s3_client:
        print(f"[STARTUP] ✓ S3 client connected to {S3_ENDPOINT}")
        
        # Start RabbitMQ consumer in background
        consumer_task = asyncio.create_task(consume_rabbitmq(s3_client))
        print("[STARTUP] ✓ RabbitMQ consumer started")
        
        yield
        
        # Cleanup
        print("[SHUTDOWN] Stopping RabbitMQ consumer...")
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        print("[SHUTDOWN] ✓ Shutdown complete")


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
