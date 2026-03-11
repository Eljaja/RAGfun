"""Prometheus metrics definitions for the RAG Gate service."""

from prometheus_client import Counter, Gauge, Histogram

# Request tracking
REQS = Counter("gate_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("gate_latency_seconds", "Latency", ["stage"])

# Ingestion metrics
ING_PUB = Counter("gate_ingestion_tasks_published_total", "Ingestion tasks published", ["type", "status"])
ING_PUB_LAT = Histogram("gate_ingestion_publish_latency_seconds", "Publish latency", ["type"])

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

# Business-quality metrics
GATE_REFUSALS = Counter("gate_refusals_total", "Refusals (answer == 'I don't know')", ["endpoint"])
GATE_DEGRADED = Counter("gate_degraded_total", "Degradation events", ["kind"])
GATE_PARTIAL = Counter("gate_partial_total", "Partial responses", ["endpoint"])

# Pre-create common label series so Grafana panels show 0 instead of "No data" right after startup.
GATE_REFUSALS.labels(endpoint="/v1/chat").inc(0)
GATE_PARTIAL.labels(endpoint="/v1/chat").inc(0)

