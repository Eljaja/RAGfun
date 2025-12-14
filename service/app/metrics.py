from __future__ import annotations

from prometheus_client import Counter, Histogram


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


