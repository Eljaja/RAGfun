from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATE_",
        env_file=".env",           # Load from .env file in cwd
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Auth check 
    auth_server: str
    admin_secret_token: str

    # Service
    service_name: str = "rag-gate"
    environment: str = "dev"
    log_level: str = "INFO"

    # Retrieval backend (existing microservice in ./service)
    retrieval_url: AnyHttpUrl = Field(default="http://retrieval:8080")
    retrieval_timeout_s: float = 10.0

    # Document Storage (optional, for uploads/listing)
    storage_url: AnyHttpUrl | None = None
    storage_timeout_s: float = 30.0

    # Document Processor (optional, used by async ingestion pipeline)
    doc_processor_url: AnyHttpUrl | None = None
    doc_processor_timeout_s: float = 300.0

    # Async ingestion queue (optional)
    rabbit_url: str | None = None
    rabbit_queue: str = "ingestion.tasks"

    # LLM
    llm_provider: Literal["mock", "openai_compat"] = "mock"
    llm_base_url: AnyHttpUrl | None = Field(default="https://api.openai.com/v1")
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0

    # RAG behavior
    retrieval_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    top_k: int = 8
    max_context_chars: int = 18_000

    # Multi-query retrieval (RRF fusion of query variants)
    multi_query_enabled: bool = False
    multi_query_max_queries: int = 4
    multi_query_top_k_multiplier: int = 8
    multi_query_rrf_k: int = 60

    # Two-pass retrieval (optional)
    two_pass_enabled: bool = False
    two_pass_hint_max_terms: int = 8
    two_pass_min_unique_docs: int = 3

    # BM25 anchor pass (optional)
    bm25_anchor_enabled: bool = False
    bm25_anchor_top_k: int = 30
    bm25_anchor_rrf_k: int = 60

    # Context packing (segment stitching)
    segment_stitching_enabled: bool = False
    segment_stitching_max_chunks: int = 4
    segment_stitching_group_by_page: bool = True

    # HTTP / UI
    cors_allow_origins: str = "*"  # for dev, can be narrowed

    def safe_summary(self) -> dict:
        return {
            "service": {"name": self.service_name, "environment": self.environment},
            "retrieval": {
                "url": str(self.retrieval_url),
                "timeout_s": self.retrieval_timeout_s,
                "mode": self.retrieval_mode,
                "top_k": self.top_k,
            },
            "llm": {
                "provider": self.llm_provider,
                "base_url": str(self.llm_base_url) if self.llm_base_url else None,
                "model": self.llm_model,
                "api_key_set": self.llm_api_key is not None,
                "timeout_s": self.llm_timeout_s,
            },
            "rag": {
                "max_context_chars": self.max_context_chars,
                "multi_query_enabled": bool(self.multi_query_enabled),
                "two_pass_enabled": bool(self.two_pass_enabled),
                "bm25_anchor_enabled": bool(self.bm25_anchor_enabled),
                "segment_stitching_enabled": bool(self.segment_stitching_enabled),
            },
            "storage": {
                "url": str(self.storage_url) if self.storage_url else None,
                "timeout_s": self.storage_timeout_s,
            },
            "doc_processor": {
                "url": str(self.doc_processor_url) if self.doc_processor_url else None,
                "timeout_s": self.doc_processor_timeout_s,
            },
            "queue": {"rabbit_url_set": bool(self.rabbit_url), "rabbit_queue": self.rabbit_queue},
        }


def load_settings() -> Settings:
    s = Settings()
    # Normalize empty secrets from env (e.g. VAR="") to None
    if s.llm_api_key is not None and s.llm_api_key.get_secret_value().strip() == "":
        s.llm_api_key = None
    if s.top_k <= 0:
        raise ValueError("GATE_TOP_K must be > 0")
    if s.max_context_chars <= 0:
        raise ValueError("GATE_MAX_CONTEXT_CHARS must be > 0")
    if s.multi_query_max_queries <= 0:
        raise ValueError("GATE_MULTI_QUERY_MAX_QUERIES must be > 0")
    if s.multi_query_top_k_multiplier <= 0:
        raise ValueError("GATE_MULTI_QUERY_TOP_K_MULTIPLIER must be > 0")
    if s.segment_stitching_max_chunks <= 0:
        raise ValueError("GATE_SEGMENT_STITCHING_MAX_CHUNKS must be > 0")
    if s.bm25_anchor_top_k <= 0:
        raise ValueError("GATE_BM25_ANCHOR_TOP_K must be > 0")
    return s

