from fastapi import FastAPI, HTTPException, Request
import json
import os
import threading
import time
import pika
import boto3
from pydantic import BaseModel
from botocore.client import Config


app = FastAPI()

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5676"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")

# If your MQTT plugin publishes to amq.topic (default), use:
AMQP_EXCHANGE = os.getenv("AMQP_EXCHANGE", "amq.topic")
# MQTT topic rustfs/events becomes routing key rustfs.events
AMQP_BINDING_KEY = os.getenv("AMQP_BINDING_KEY", "rustfs.events")
AMQP_QUEUE = os.getenv("AMQP_QUEUE", "rustfs_events")
AMQP_PREFETCH = int(os.getenv("AMQP_PREFETCH", "50"))

# ---- simple message-only checks (adjust freely) ----
SUSPICIOUS_EXTS = {".exe", ".dll", ".js", ".vbs", ".ps1", ".bat", ".cmd", ".scr", ".jar", ".msi", ".lnk", ".iso"}
def is_suspicious(event: dict) -> tuple[bool, list[str]]:
    reasons = []
    # try a few common places a filename/key might appear
    candidates = []

    # S3-ish
    try:
        rec = event.get("Records", [])[0]
        key = rec.get("s3", {}).get("object", {}).get("key")
        if isinstance(key, str):
            candidates.append(key)
    except Exception:
        pass

    # flat-ish
    for k in ("key", "object", "objectKey", "name", "filename", "path"):
        v = event.get(k)
        if isinstance(v, str):
            candidates.append(v)

    for c in candidates:
        lc = c.lower()
        for ext in SUSPICIOUS_EXTS:
            if lc.endswith(ext):
                reasons.append(f"extension {ext} ({c})")
                break
        if ".." in lc or lc.startswith(("/", "\\")):
            reasons.append(f"path traversal/absolute path ({c})")
        if "/." in lc or lc.startswith("."):
            reasons.append(f"hidden file pattern ({c})")

    return (len(reasons) > 0), reasons

def consume_loop():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

    while True:
        try:
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                virtual_host=RABBITMQ_VHOST,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=30,
            )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=AMQP_PREFETCH)

            # Declare queue and bind to exchange/routing key
            channel.queue_declare(queue=AMQP_QUEUE, durable=True)
            channel.queue_bind(queue=AMQP_QUEUE, exchange=AMQP_EXCHANGE, routing_key=AMQP_BINDING_KEY)

            print(f"[AMQP] consuming queue={AMQP_QUEUE} exchange={AMQP_EXCHANGE} key={AMQP_BINDING_KEY}")

            def on_message(ch, method, properties, body: bytes):
                try:
                    raw = body.decode("utf-8", errors="replace")
                    event = json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}

                    suspicious, reasons = is_suspicious(event if isinstance(event, dict) else {"event": event})
                    label = "SUSPICIOUS" if suspicious else "OK"
                    print(f"[EVENT:{label}] rk={method.routing_key}")
                    # keep output readable
                    print(json.dumps(event, indent=2)[:4000])
                    if reasons:
                        print("  reasons:", ", ".join(reasons))
                except Exception as e:
                    print(f"[ERROR] failed to process message: {e}")
                finally:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(queue=AMQP_QUEUE, on_message_callback=on_message)
            channel.start_consuming()

        except Exception as e:
            print(f"[AMQP] connection/consume error: {e}. Reconnecting in 3s...")
            print(e)
            print(type(e))
            time.sleep(3)

@app.on_event("startup")
def startup():
    threading.Thread(target=consume_loop, daemon=True).start()
    print("[START] AMQP consumer thread started")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/webhook")
async def webhook_endpoint(request: Request):
    """
    Webhook endpoint to receive notifications from rustfs.
    This endpoint handles file system events and can forward them to RabbitMQ
    or process them according to your business logic.
    """
    try:
        request_data = await request.json()
        print(request.headers)
        # Log the incoming webhook
        print(f"[WEBHOOK] Received notification: {json.dumps(request_data, indent=2)}")
        
        # Check if the event is suspicious
        suspicious, reasons = is_suspicious(request_data)
        label = "SUSPICIOUS" if suspicious else "OK"
        print(f"[WEBHOOK:{label}] Processing webhook event")
        
        if reasons:
            print(f"[WEBHOOK] Security concerns: {', '.join(reasons)}")
        
        # Here you can add your business logic:
        # - Forward to RabbitMQ for further processing
        # - Store in database
        # - Send notifications
        # - Trigger workflows, etc.
        
        # Example: Forward to RabbitMQ (uncomment if needed)
        # try:
        #     credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        #     params = pika.ConnectionParameters(
        #         host=RABBITMQ_HOST,
        #         port=RABBITMQ_PORT,
        #         virtual_host=RABBITMQ_VHOST,
        #         credentials=credentials,
        #     )
        #     connection = pika.BlockingConnection(params)
        #     channel = connection.channel()
        #     channel.basic_publish(
        #         exchange=AMQP_EXCHANGE,
        #         routing_key="rustfs.webhook.events",
        #         body=json.dumps(request_data),
        #         properties=pika.BasicProperties(delivery_mode=2)
        #     )
        #     connection.close()
        #     print("[WEBHOOK] Event forwarded to RabbitMQ")
        # except Exception as e:
        #     print(f"[WEBHOOK] Failed to forward to RabbitMQ: {e}")
        
        return {
            "status": "received",
            "processed": True,
            "suspicious": suspicious,
            "reasons": reasons
        }
        
    except Exception as e:
        print(f"[WEBHOOK] Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {e}")


# --- RustFS / S3 config ---
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9004")  # change if needed
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "rustfs")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "password")
S3_REGION = os.getenv("S3_REGION", "eu-central-1")

# Important for S3-compatible servers in Docker: path-style addressing is often safest
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

class PresignUploadRequest(BaseModel):
    bucket: str
    key: str
    expires_seconds: int = 600  # 10 minutes
    content_type: str | None = None  # optional, but if provided client must send it

class PresignDownloadRequest(BaseModel):
    bucket: str
    key: str
    expires_seconds: int = 600

@app.post("/presign/upload")
def presign_upload(req: PresignUploadRequest):
    try:
        params = {"Bucket": req.bucket, "Key": req.key}
        # If you include ContentType in the signature, the uploader MUST use the same Content-Type header.
        if req.content_type:
            params["ContentType"] = req.content_type

        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=req.expires_seconds,
        )
        return {
            "method": "PUT",
            "url": url,
            "headers": {"Content-Type": req.content_type} if req.content_type else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to presign upload: {e}")

@app.post("/presign/download")
def presign_download(req: PresignDownloadRequest):
    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": req.bucket, "Key": req.key},
            ExpiresIn=req.expires_seconds,
        )
        return {"method": "GET", "url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to presign download: {e}")



# Run:
# uvicorn main:app --host 0.0.0.0 --port 9999
