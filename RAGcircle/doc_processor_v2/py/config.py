from __future__ import annotations

from pydantic import AliasChoices, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    # ── RabbitMQ connection ──────────────────────────────────────────────
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5676
    rabbitmq_user: str = "admin"
    rabbitmq_pass: str = "admin"
    rabbitmq_vhost: str = "/"

    # ── AMQP topology ────────────────────────────────────────────────────
    amqp_exchange: str = "amq.topic"
    amqp_binding_key: str = "rustfs.events"
    amqp_queue: str = "rustfs_events"

    # Dead-letter "parking lot" (manual publish)
    amqp_dlx_exchange: str = "rustfs.events.dlx"
    amqp_dlq_queue: str = ""
    amqp_dlq_routing_key: str = ""

    # Retry ladder: one queue per level with increasing TTL
    amqp_retry_exchange: str = "rustfs.events.retry"
    amqp_retry_queue: str = ""
    amqp_retry_routing_key: str = ""
    amqp_retry_ttls: str = "50,150,450,1200,3000"

    # ── S3 ───────────────────────────────────────────────────────────────
    s3_endpoint: str = "http://localhost:9004"
    s3_access_key: str = "rustfs"
    s3_secret_key: str = "password"
    s3_region: str = "us-east-1"

    # ── VLM ──────────────────────────────────────────────────────────────
    vlm_base_url: str = "http://localhost:8123"
    vlm_api_key: str | None = None
    vlm_model: str = "ibm-granite/granite-docling-258M"
    vlm_timeout: float = 120.0

    # ── Processing ───────────────────────────────────────────────────────
    proc_page_window: int = 50
    proc_max_px: int = 2048
    proc_vlm_concurrency: int = 4
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 200

    # ── Embedding ────────────────────────────────────────────────────────
    embedder_url: str = "http://localhost:8902"
    embedder_model: str = "BAAI/bge-m3"
    embedder_dim: int = 1024
    embed_batch_size: int = 32

    # ── Stores ───────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:8903"
    qdrant_collection: str = "documents"
    opensearch_url: str = "http://localhost:8905"
    opensearch_index: str = "documents"

    db_addr: str = Field(
        default="postgresql://user:pass@localhost:5439/db",
        validation_alias=AliasChoices("db_addr", "postgre_url"),
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "DEBUG"

    # ── Validators ───────────────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def _fill_derived_defaults(cls, values: dict) -> dict:
        """DLQ / retry queue names default to {amqp_queue}.dlq / .retry."""
        queue = values.get("amqp_queue", "rustfs_events")
        for key, suffix in (
            ("amqp_dlq_queue", ".dlq"),
            ("amqp_dlq_routing_key", ".dlq"),
            ("amqp_retry_queue", ".retry"),
            ("amqp_retry_routing_key", ".retry"),
        ):
            if not values.get(key):
                values[key] = f"{queue}{suffix}"
        return values

    # ── Computed ─────────────────────────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rabbitmq_url(self) -> str:
        vhost = self.rabbitmq_vhost or "/"
        if not vhost.startswith("/"):
            vhost = "/" + vhost
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_pass}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{vhost}"
        )

    @property
    def amqp_retry_ttls_ms(self) -> tuple[int, ...]:
        """Parsed retry TTLs as millisecond values."""
        return tuple(int(x.strip()) for x in self.amqp_retry_ttls.split(","))
