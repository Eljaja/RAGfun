from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


REQS = Counter("storage_requests_total", "Total requests", ["endpoint", "status"])
ERRS = Counter("storage_errors_total", "Errors", ["stage", "kind"])

LAT = Histogram(
    "storage_latency_seconds",
    "Latency by stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
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

# Business / storage usage metrics
STORAGE_DEDUP = Counter("storage_dedup_total", "Deduplication outcomes", ["outcome"])  # duplicate|unique
STORAGE_UPLOAD_BYTES = Counter("storage_upload_bytes_total", "Uploaded bytes stored", ["outcome"])  # duplicate|unique
STORAGE_DOCS = Gauge("storage_docs_total", "Total documents in metadata DB")
STORAGE_BYTES = Gauge("storage_bytes_total", "Total bytes in metadata DB (sum(size))")







