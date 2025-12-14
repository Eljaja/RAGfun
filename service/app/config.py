from __future__ import annotations

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # Service
    service_name: str = "hybrid-retrieval"
    environment: str = "dev"
    log_level: str = "INFO"

    # OpenSearch
    os_url: AnyHttpUrl = Field(default="http://localhost:9200")
    os_username: str | None = None
    os_password: SecretStr | None = None
    os_index_alias: str = "rag_chunks"
    os_index_prefix: str = "rag_chunks_v"

    # Qdrant
    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 768
    vector_distance: str = "cosine"  # cosine|dot|euclid

    # Embeddings
    embedding_provider: str = "http"  # http|mock
    embedding_url: AnyHttpUrl | None = None
    embedding_model: str | None = None  # optional (e.g. OpenAI-compatible backends like Infinity)
    embedding_api_key: SecretStr | None = None
    embedding_timeout_s: float = 10.0
    embedding_batch_size: int = 32

    # Chunking
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 50

    # Search
    default_top_k: int = 10
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    weight_bm25: float = 1.0
    weight_vector: float = 1.0
    max_chunks_per_doc: int = 3

    # Rerank
    rerank_mode: str = "disabled"  # disabled|always|auto
    rerank_url: AnyHttpUrl | None = None
    rerank_model: str | None = None  # optional (e.g. Infinity rerank model id)
    rerank_api_key: SecretStr | None = None
    rerank_timeout_s: float = 6.0
    rerank_max_candidates: int = 50
    rerank_auto_min_query_tokens: int = 6
    rerank_auto_min_intersection: int = 2

    # Sources / URI redaction
    redact_uri_mode: str = "none"  # none|strip_query|strip_all

    # Page retrieval improvements
    enable_page_deduplication: bool = False  # Deduplicate chunks by (doc_id, page)
    enable_parent_page_retrieval: bool = False  # Return full pages instead of chunks

    # Observability
    otel_enabled: bool = False
    otel_service_name: str = "hybrid-retrieval"

    def safe_summary(self) -> dict:
        """Configuration summary without secrets."""
        return {
            "service": {
                "name": self.service_name,
                "environment": self.environment,
            },
            "opensearch": {
                "url": str(self.os_url),
                "index_alias": self.os_index_alias,
                "index_prefix": self.os_index_prefix,
                "username_set": self.os_username is not None,
            },
            "qdrant": {
                "url": str(self.qdrant_url),
                "collection": self.qdrant_collection,
                "vector_size": self.vector_size,
                "vector_distance": self.vector_distance,
                "api_key_set": self.qdrant_api_key is not None,
            },
            "embeddings": {
                "provider": self.embedding_provider,
                "url": str(self.embedding_url) if self.embedding_url else None,
                "model": self.embedding_model,
                "api_key_set": self.embedding_api_key is not None,
                "timeout_s": self.embedding_timeout_s,
                "batch_size": self.embedding_batch_size,
            },
            "chunking": {
                "max_tokens": self.chunk_max_tokens,
                "overlap_tokens": self.chunk_overlap_tokens,
            },
            "search": {
                "default_top_k": self.default_top_k,
                "bm25_top_k": self.bm25_top_k,
                "vector_top_k": self.vector_top_k,
                "rrf_k": self.rrf_k,
                "weights": {"bm25": self.weight_bm25, "vector": self.weight_vector},
                "max_chunks_per_doc": self.max_chunks_per_doc,
            },
            "rerank": {
                "mode": self.rerank_mode,
                "url": str(self.rerank_url) if self.rerank_url else None,
                "model": self.rerank_model,
                "api_key_set": self.rerank_api_key is not None,
                "timeout_s": self.rerank_timeout_s,
                "max_candidates": self.rerank_max_candidates,
            },
            "sources": {
                "redact_uri_mode": self.redact_uri_mode,
            },
            "page_retrieval": {
                "enable_page_deduplication": self.enable_page_deduplication,
                "enable_parent_page_retrieval": self.enable_parent_page_retrieval,
            },
            "otel": {
                "enabled": self.otel_enabled,
                "service_name": self.otel_service_name,
            },
        }


def load_settings() -> Settings:
    s = Settings()
    if s.embedding_provider == "http" and s.embedding_url is None:
        raise ValueError("RAG_EMBEDDING_PROVIDER=http requires RAG_EMBEDDING_URL")
    if s.rerank_mode != "disabled" and s.rerank_url is None:
        raise ValueError("RAG_RERANK_MODE requires RAG_RERANK_URL (unless disabled)")
    if s.vector_size <= 0:
        raise ValueError("RAG_VECTOR_SIZE must be > 0")
    if s.chunk_max_tokens <= 0:
        raise ValueError("RAG_CHUNK_MAX_TOKENS must be > 0")
    if s.chunk_overlap_tokens < 0 or s.chunk_overlap_tokens >= s.chunk_max_tokens:
        raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be >=0 and < CHUNK_MAX_TOKENS")
    return s


