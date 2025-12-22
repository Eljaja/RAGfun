from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


REQS = Counter("rag_requests_total", "Total requests", ["endpoint", "status"])
DEP_DEGRADED = Counter("rag_degraded_total", "Degradation events", ["kind"])
ERRS = Counter("rag_errors_total", "Errors", ["stage", "kind"])

LAT = Histogram(
    "rag_stage_latency_seconds",
    "Latency by stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

CAND = Histogram(
    "rag_candidates",
    "Candidate set sizes",
    ["stage"],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
)

# Standard HTTP server metrics (shared names across services; Prometheus "job" label disambiguates)
HTTP_INFLIGHT = Gauge("http_server_inflight_requests", "In-flight HTTP requests", ["method", "route"])
HTTP_REQS = Counter("http_server_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LAT = Histogram(
    "http_server_request_duration_seconds",
    "HTTP request duration (seconds)",
    ["method", "route", "status"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60),
)
HTTP_REQ_SIZE = Histogram(
    "http_server_request_size_bytes",
    "HTTP request size (bytes), from Content-Length when available",
    ["method", "route"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)
HTTP_RESP_SIZE = Histogram(
    "http_server_response_size_bytes",
    "HTTP response size (bytes), from Content-Length when available",
    ["method", "route", "status"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)

# Business-quality metrics (retrieval/search)
RAG_PARTIAL = Counter("rag_partial_total", "Partial responses", ["endpoint", "kind"])  # kind=partial|partial_rerank
RAG_RERANK = Counter("rag_rerank_total", "Rerank outcomes", ["outcome"])  # applied|skipped|failed|unavailable
RAG_PAGE_DEDUP_SAVED = Histogram(
    "rag_page_dedup_saved",
    "Chunks removed by page-level deduplication",
    buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
)
RAG_PARENT_PAGES = Histogram(
    "rag_parent_pages_fetched",
    "Parent pages fetched (page-level retrieval)",
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)


