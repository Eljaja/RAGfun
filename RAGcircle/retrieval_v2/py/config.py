from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    # ── Gate (project CRUD) ──────────────────────────────────
    gate_url: str = "http://localhost:8918"

    # ── Embedding service (infra only — model comes from project config) ─
    embedder_url: str = "http://localhost:11008"
    embedder_timeout: float = 30.0

    # ── Qdrant (vector search) ───────────────────────────────
    qdrant_url: str = "http://localhost:11007"

    # ── OpenSearch (BM25 search) ─────────────────────────────
    opensearch_url: str = "http://localhost:11011"

    # ── Reranker (infra only — model comes from project config) ─
    reranker_url: str = "http://localhost:11009"

    # ── Server ───────────────────────────────────────────────
    port: int = 8921
    log_level: str = "INFO"
