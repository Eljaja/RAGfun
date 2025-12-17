from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATE_", extra="ignore")

    # Service
    service_name: str = "rag-gate"
    environment: str = "dev"
    log_level: str = "INFO"

    # Retrieval backend (existing microservice in ./service)
    retrieval_url: AnyHttpUrl = Field(default="http://retrieval:8080")
    retrieval_timeout_s: float = 60.0

    # Document Storage backend
    storage_url: AnyHttpUrl | None = Field(default=None)
    storage_timeout_s: float = 60.0

    # Document Processor (extracts text from stored docs and indexes via retrieval)
    doc_processor_url: AnyHttpUrl | None = Field(default=None)
    doc_processor_timeout_s: float = 300.0

    # Async ingestion (RabbitMQ)
    rabbit_url: str | None = Field(default=None)
    rabbit_queue: str = "ingestion.tasks"

    # LLM
    llm_provider: Literal["mock", "openai_compat"] = "openai_compat"
    llm_base_url: AnyHttpUrl | None = Field(default="https://api.openai.com/v1")
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0

    # RAG behavior
    retrieval_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    top_k: int = 8
    max_context_chars: int = 18_000

    # Multi-hop retrieval improvements (opt-in)
    # Multi-query: generate a few query variants and fuse their results.
    multi_query_enabled: bool = False
    multi_query_max_queries: int = 4
    multi_query_top_k_multiplier: int = 12  # raw_top_k = max(top_k, 1) * multiplier
    multi_query_rrf_k: int = 60
    # Two-pass: run an initial retrieval, extract hint terms from the top hits, then retrieve again.
    two_pass_enabled: bool = False
    two_pass_hint_max_terms: int = 8
    two_pass_min_unique_docs: int = 3

    # HTTP / UI
    cors_allow_origins: str = "*"  # for dev, can be narrowed

    @field_validator("storage_url", "doc_processor_url", "llm_base_url", "rabbit_url", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v):
        # Common in docker-compose env: VAR= (empty) should behave like unset.
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

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
                "multi_query": {
                    "enabled": self.multi_query_enabled,
                    "max_queries": self.multi_query_max_queries,
                    "top_k_multiplier": self.multi_query_top_k_multiplier,
                    "rrf_k": self.multi_query_rrf_k,
                },
                "two_pass": {
                    "enabled": self.two_pass_enabled,
                    "hint_max_terms": self.two_pass_hint_max_terms,
                    "min_unique_docs": self.two_pass_min_unique_docs,
                },
            },
            "storage": {
                "url": str(self.storage_url) if self.storage_url else None,
                "timeout_s": self.storage_timeout_s,
            },
            "doc_processor": {
                "url": str(self.doc_processor_url) if self.doc_processor_url else None,
                "timeout_s": self.doc_processor_timeout_s,
            },
            "async_ingestion": {
                "enabled": bool(self.rabbit_url),
                "queue": self.rabbit_queue,
            },
        }


def load_settings() -> Settings:
    s = Settings()
    # Normalize empty secrets from env (e.g. VAR="") to None
    if s.llm_api_key is not None and s.llm_api_key.get_secret_value().strip() == "":
        s.llm_api_key = None
    if s.llm_provider == "openai_compat" and s.llm_api_key is None:
        raise ValueError("GATE_LLM_PROVIDER=openai_compat requires GATE_LLM_API_KEY")
    if s.top_k <= 0:
        raise ValueError("GATE_TOP_K must be > 0")
    if s.max_context_chars <= 0:
        raise ValueError("GATE_MAX_CONTEXT_CHARS must be > 0")
    return s


