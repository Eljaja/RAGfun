from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # RabbitMQ
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_pass: str
    rabbitmq_vhost: str

    amqp_exchange: str
    amqp_binding_key: str
    amqp_queue: str

    # Dead-letter “parking lot” (manual publish)
    amqp_dlx_exchange: str
    amqp_dlq_queue: str
    amqp_dlq_routing_key: str

    # Delayed retry (TTL queue that dead-letters back to main exchange/routing key)
    amqp_retry_exchange: str
    amqp_retry_queue: str
    amqp_retry_routing_key: str
    amqp_retry_ttl_ms: int
    amqp_max_retries: int

    # S3
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str

    # VLM
    vlm_base_url: str
    vlm_api_key: str | None
    vlm_model: str
    vlm_timeout: float

    # Processing
    proc_max_pages: int
    proc_max_px: int
    proc_vlm_concurrency: int
    chunk_size_chars: int
    chunk_overlap_chars: int

    # Embedding
    embedder_url: str
    embedder_model: str
    embedder_dim: int
    embed_batch_size: int

    # Stores
    qdrant_url: str
    qdrant_collection: str
    opensearch_url: str
    opensearch_index: str

    @property
    def rabbitmq_url(self) -> str:
        # NOTE: RabbitMQ vhost in URI is path-ish. We accept env var as "/" or "/myvhost" or "myvhost".
        vhost = self.rabbitmq_vhost or "/"
        if not vhost.startswith("/"):
            vhost = "/" + vhost
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_pass}@{self.rabbitmq_host}:{self.rabbitmq_port}{vhost}"

    @classmethod
    def from_env(cls) -> "AppConfig":
        rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5676"))
        rabbitmq_user = os.getenv("RABBITMQ_USER", "admin")
        rabbitmq_pass = os.getenv("RABBITMQ_PASS", "admin")
        rabbitmq_vhost = os.getenv("RABBITMQ_VHOST", "/")

        amqp_exchange = os.getenv("AMQP_EXCHANGE", "amq.topic")
        amqp_binding_key = os.getenv("AMQP_BINDING_KEY", "rustfs.events")
        amqp_queue = os.getenv("AMQP_QUEUE", "rustfs_events")

        amqp_dlx_exchange = os.getenv("AMQP_DLX_EXCHANGE", "rustfs.events.dlx")
        amqp_dlq_queue = os.getenv("AMQP_DLQ_QUEUE", f"{amqp_queue}.dlq")
        amqp_dlq_routing_key = os.getenv("AMQP_DLQ_ROUTING_KEY", f"{amqp_queue}.dlq")

        amqp_retry_exchange = os.getenv("AMQP_RETRY_EXCHANGE", "rustfs.events.retry")
        amqp_retry_queue = os.getenv("AMQP_RETRY_QUEUE", f"{amqp_queue}.retry")
        amqp_retry_routing_key = os.getenv("AMQP_RETRY_ROUTING_KEY", f"{amqp_queue}.retry")
        amqp_retry_ttl_ms = int(os.getenv("AMQP_RETRY_TTL_MS", "30000"))
        amqp_max_retries = int(os.getenv("AMQP_MAX_RETRIES", "5"))

        s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9004")
        s3_access_key = os.getenv("S3_ACCESS_KEY", "rustfs")
        s3_secret_key = os.getenv("S3_SECRET_KEY", "password")
        s3_region = os.getenv("S3_REGION", "us-east-1")

        vlm_base_url = os.getenv("VLM_BASE_URL", "http://localhost:8123")
        vlm_api_key = os.getenv("VLM_API_KEY", None)
        vlm_model = os.getenv("VLM_MODEL", "ibm-granite/granite-docling-258M")
        vlm_timeout = float(os.getenv("VLM_TIMEOUT", "120.0"))

        proc_max_pages = int(os.getenv("PROC_MAX_PAGES", "50"))
        proc_max_px = int(os.getenv("PROC_MAX_PX", "2048"))
        proc_vlm_concurrency = int(os.getenv("PROC_VLM_CONCURRENCY", "4"))
        chunk_size_chars = int(os.getenv("CHUNK_SIZE_CHARS", "1500"))
        chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

        embedder_url = os.getenv("EMBEDDER_URL", "http://localhost:8902")
        embedder_model = os.getenv("EMBEDDER_MODEL", "sentence-transformers/e5-base-v2")
        embedder_dim = int(os.getenv("EMBEDDER_DIM", "768"))
        embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:8903")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
        opensearch_url = os.getenv("OPENSEARCH_URL", "http://localhost:8905")
        opensearch_index = os.getenv("OPENSEARCH_INDEX", "documents")

        return cls(
            rabbitmq_host=rabbitmq_host,
            rabbitmq_port=rabbitmq_port,
            rabbitmq_user=rabbitmq_user,
            rabbitmq_pass=rabbitmq_pass,
            rabbitmq_vhost=rabbitmq_vhost,
            amqp_exchange=amqp_exchange,
            amqp_binding_key=amqp_binding_key,
            amqp_queue=amqp_queue,
            amqp_dlx_exchange=amqp_dlx_exchange,
            amqp_dlq_queue=amqp_dlq_queue,
            amqp_dlq_routing_key=amqp_dlq_routing_key,
            amqp_retry_exchange=amqp_retry_exchange,
            amqp_retry_queue=amqp_retry_queue,
            amqp_retry_routing_key=amqp_retry_routing_key,
            amqp_retry_ttl_ms=amqp_retry_ttl_ms,
            amqp_max_retries=amqp_max_retries,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=s3_region,
            vlm_base_url=vlm_base_url,
            vlm_api_key=vlm_api_key,
            vlm_model=vlm_model,
            vlm_timeout=vlm_timeout,
            proc_max_pages=proc_max_pages,
            proc_max_px=proc_max_px,
            proc_vlm_concurrency=proc_vlm_concurrency,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            embedder_url=embedder_url,
            embedder_model=embedder_model,
            embedder_dim=embedder_dim,
            embed_batch_size=embed_batch_size,
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            opensearch_url=opensearch_url,
            opensearch_index=opensearch_index,
        )

