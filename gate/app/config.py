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

    # Query router (optional): uses a small LLM to pick retrieval knobs per request.
    # This is intentionally separate from the main answering LLM to keep routing cheap and controllable.
    router_enabled: bool = False
    # If enabled, we still compute the route, but do not apply it (logs only).
    router_dry_run: bool = False
    router_llm_provider: Literal["mock", "openai_compat"] = "openai_compat"
    router_llm_base_url: AnyHttpUrl | None = Field(default=None)
    router_llm_api_key: SecretStr | None = None
    router_llm_model: str = "ministral-3b-2512"
    router_llm_timeout_s: float = 10.0
    router_llm_temperature: float = 0.0
    router_llm_max_tokens: int = 220
    # If true, router is allowed to override explicit per-request fields like retrieval_mode/top_k/rerank.
    # Keep false by default to preserve API contract for callers that set these intentionally.
    router_override_request_params: bool = False

    # RAG behavior
    retrieval_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    top_k: int = 8
    max_context_chars: int = 18_000

    # Context packing (recommended): stitch multiple retrieved chunks from the same page/doc into
    # a coherent segment before sending to the LLM. Helps multi-hop and narrative questions.
    segment_stitching_enabled: bool = False
    # Max distinct chunks to stitch into one segment (per page/doc group).
    segment_stitching_max_chunks: int = 4
    # If page is available, prefer grouping by (doc_id, uri, page); else fall back to (doc_id, uri).
    segment_stitching_group_by_page: bool = True

    # BM25 anchor pass (recommended): run an additional BM25 lookup on a keyword-only query
    # and union/fuse candidates, so exact-match entities don't get lost in hybrid/rerank.
    bm25_anchor_enabled: bool = True
    bm25_anchor_top_k: int = 30
    bm25_anchor_rrf_k: int = 60

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
            "router": {
                "enabled": self.router_enabled,
                "dry_run": self.router_dry_run,
                "provider": self.router_llm_provider,
                "base_url": str(self.router_llm_base_url) if self.router_llm_base_url else None,
                "model": self.router_llm_model,
                "api_key_set": self.router_llm_api_key is not None,
                "timeout_s": self.router_llm_timeout_s,
                "temperature": self.router_llm_temperature,
                "max_tokens": self.router_llm_max_tokens,
                "override_request_params": self.router_override_request_params,
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
                "segment_stitching": {
                    "enabled": self.segment_stitching_enabled,
                    "max_chunks": self.segment_stitching_max_chunks,
                    "group_by_page": self.segment_stitching_group_by_page,
                },
                "bm25_anchor": {
                    "enabled": self.bm25_anchor_enabled,
                    "top_k": self.bm25_anchor_top_k,
                    "rrf_k": self.bm25_anchor_rrf_k,
                },
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
    if s.router_llm_api_key is not None and s.router_llm_api_key.get_secret_value().strip() == "":
        s.router_llm_api_key = None
    if s.llm_provider == "openai_compat" and s.llm_api_key is None:
        raise ValueError("GATE_LLM_PROVIDER=openai_compat requires GATE_LLM_API_KEY")
    if s.router_enabled and s.router_llm_provider == "openai_compat" and s.router_llm_base_url is None:
        raise ValueError("GATE_ROUTER_ENABLED=true requires GATE_ROUTER_LLM_BASE_URL for router_llm_provider=openai_compat")
    if s.top_k <= 0:
        raise ValueError("GATE_TOP_K must be > 0")
    if s.max_context_chars <= 0:
        raise ValueError("GATE_MAX_CONTEXT_CHARS must be > 0")
    if s.router_llm_timeout_s <= 0:
        raise ValueError("GATE_ROUTER_LLM_TIMEOUT_S must be > 0")
    if s.router_llm_max_tokens <= 0:
        raise ValueError("GATE_ROUTER_LLM_MAX_TOKENS must be > 0")
    return s


