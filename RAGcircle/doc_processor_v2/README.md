# Document Processor v2

Async microservice that:
- Listens to RabbitMQ for S3 events
- Downloads PDFs when uploaded
- Processes documents (extract text, metadata, etc.)

## Architecture

```
S3 Upload → RustFS → RabbitMQ → Doc Processor v2
                                      ↓
                                  Process PDF
                                      ↓
                                  Store Results
```

## Features

- ✅ Async RabbitMQ consumer (aio-pika)
- ✅ Async S3 operations (aiobotocore)
- ✅ Auto-reconnection on failure
- ✅ Event-driven architecture
- ✅ Loosely coupled from S3 service

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv (recommended)
uv pip install -r requirements.txt
```

## Configuration

Environment variables:

```bash
# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5676
RABBITMQ_USER=admin
RABBITMQ_PASS=admin
RABBITMQ_VHOST=/
AMQP_QUEUE=rustfs_events
AMQP_EXCHANGE=amq.topic
AMQP_BINDING_KEY=rustfs.events

# S3 / RustFS
S3_ENDPOINT=http://localhost:9004
S3_ACCESS_KEY=rustfs
S3_SECRET_KEY=password
S3_REGION=us-east-1
```

## Run

```bash
# Development
uvicorn main:app --reload --port 9998

# Production
uvicorn main:app --host 0.0.0.0 --port 9998 --workers 1
```

## How It Works

1. **Consumer Starts**: On startup, connects to RabbitMQ and starts listening
2. **Event Received**: S3 ObjectCreated events arrive via RabbitMQ
3. **Filter PDFs**: Only processes `.pdf` files
4. **Download**: Async download from S3
5. **Process**: Extract text, metadata (TODO: enhance)
6. **Store**: Save results to database (TODO: add)

## Event Flow Example

```json
{
  "Records": [{
    "eventName": "ObjectCreated:Put",
    "s3": {
      "bucket": {"name": "my-bucket"},
      "object": {"key": "document.pdf", "size": 123456}
    }
  }]
}
```

↓ Doc Processor detects PDF ↓

```
[PDF] Downloading my-bucket/document.pdf...
[PDF] Downloaded document.pdf - Size: 123,456 bytes
[PDF] ✓ Processed document.pdf
```

## Extending

### Add PDF Text Extraction

```python
import pypdf

async def process_pdf(bucket, key, s3_client):
    # ... download ...
    
    # Extract text
    reader = pypdf.PdfReader(io.BytesIO(pdf_data))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Store in database, search index, etc.
```

### Add Database Storage

```python
# Store processing results
await db.execute("""
    INSERT INTO processed_documents (bucket, key, text, metadata)
    VALUES ($1, $2, $3, $4)
""", bucket, key, text, metadata)
```

## Health Check

```bash
curl http://localhost:9998/health
```

## Microservices Integration

This service is **loosely coupled** via events:
- ✅ S3 service doesn't know about doc processor
- ✅ Doc processor can be down, events queue up
- ✅ Easy to add more processors (images, videos, etc.)
- ✅ Scales independently
