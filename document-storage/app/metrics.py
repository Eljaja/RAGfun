from __future__ import annotations

from prometheus_client import Counter, Histogram


REQS = Counter("storage_requests_total", "Total requests", ["endpoint", "status"])
ERRS = Counter("storage_errors_total", "Errors", ["stage", "kind"])

LAT = Histogram(
    "storage_latency_seconds",
    "Latency by stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)



