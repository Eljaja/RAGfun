from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    # ── Embedding service (Infinity) ─────────────────────────
    embedder_url: str = "http://localhost:8902"
    embedder_model: str = "BAAI/bge-m3"
    embedder_timeout: float = 30.0

    # ── Qdrant (vector search) ───────────────────────────────
    qdrant_url: str = "http://localhost:8903"

    # ── OpenSearch (BM25 search) ─────────────────────────────
    opensearch_url: str = "http://localhost:8905"

    # ── Reranker ─────────────────────────────────────────────
    reranker_url: str = "http://172.18.0.7:7998"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ── Server ───────────────────────────────────────────────
    port: int = 8921
    log_level: str = "INFO"
