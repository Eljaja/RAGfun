from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

import httpx

from bench.prom_metrics import (
    diff_samples,
    extract_histogram,
    histogram_quantile_from_cumulative,
    parse_prometheus_text,
)


GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")
STORAGE_BASE_URL = os.getenv("STORAGE_BASE_URL", "http://document-storage:8081").rstrip("/")
RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")


@dataclass(frozen=True)
class RunConfig:
    docs: int = int(os.getenv("BENCH_DOCS", "3"))
    concurrency: int = int(os.getenv("BENCH_CONCURRENCY", "10"))
    duration_s: float = float(os.getenv("BENCH_DURATION_S", "15"))
    top_k: int = int(os.getenv("BENCH_TOP_K", "5"))
    include_sources: bool = os.getenv("BENCH_INCLUDE_SOURCES", "true").lower() != "false"


async def _fetch_metrics(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url)
    r.raise_for_status()
    return r.text


async def _upload_docs(doc_ids: list[str]) -> None:
    async with httpx.AsyncClient(timeout=120.0) as c:
        for doc_id in doc_ids:
            text = (
                f"Doc {doc_id}\n"
                "Acme Corp 2024 revenue was 10 million USD.\n"
                "Operating profit was 2 million USD.\n"
            )
            files = {"file": ("doc.txt", text.encode("utf-8"), "text/plain")}
            data = {
                "doc_id": doc_id,
                "title": f"Bench {doc_id}",
                "uri": f"https://example.test/bench/{doc_id}",
                "source": "bench",
                "lang": "en",
                "tags": "bench",
                "acl": "group:testers",
                "refresh": "true",
            }
            r = await c.post(f"{GATE_BASE_URL}/v1/documents/upload", files=files, data=data)
            r.raise_for_status()
            j = r.json()
            if not j.get("ok"):
                raise RuntimeError(f"upload failed: {j}")


async def _run_load(cfg: RunConfig, doc_ids: list[str]) -> dict[str, float]:
    """
    Returns latency summary (p50/p90/p95) in milliseconds for /v1/chat.
    """
    lat_ms: list[float] = []
    errors = 0

    async def worker(worker_id: int):
        nonlocal errors
        async with httpx.AsyncClient(timeout=60.0) as c:
            end_at = time.time() + cfg.duration_s
            while time.time() < end_at:
                doc_id = random.choice(doc_ids)
                payload = {
                    "query": "What was Acme Corp revenue in 2024?",
                    "history": [],
                    "retrieval_mode": "hybrid",
                    "top_k": cfg.top_k,
                    "filters": {"doc_ids": [doc_id]},
                    "acl": ["group:testers"],
                    "include_sources": cfg.include_sources,
                }
                t0 = time.time()
                try:
                    r = await c.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
                    if r.status_code != 200:
                        errors += 1
                        continue
                    j = r.json()
                    if not j.get("ok"):
                        errors += 1
                        continue
                    lat_ms.append((time.time() - t0) * 1000.0)
                except Exception:
                    errors += 1

    await asyncio.gather(*(worker(i) for i in range(cfg.concurrency)))

    lat_ms.sort()

    def q(p: float) -> float:
        if not lat_ms:
            return 0.0
        idx = int((len(lat_ms) - 1) * p)
        return float(lat_ms[idx])

    return {
        "requests": float(len(lat_ms)),
        "errors": float(errors),
        "p50_ms": q(0.50),
        "p90_ms": q(0.90),
        "p95_ms": q(0.95),
    }


def _fmt_seconds(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}s"


async def main():
    cfg = RunConfig()
    doc_ids = [f"bench-{int(time.time())}-{i}" for i in range(cfg.docs)]

    print(f"Target gate: {GATE_BASE_URL}")
    print(f"Docs={cfg.docs} concurrency={cfg.concurrency} duration_s={cfg.duration_s} top_k={cfg.top_k}")

    # Baseline metrics
    async with httpx.AsyncClient(timeout=10.0) as c:
        gate_before = parse_prometheus_text(await _fetch_metrics(c, f"{GATE_BASE_URL}/v1/metrics"))
        retrieval_before = parse_prometheus_text(await _fetch_metrics(c, f"{RETRIEVAL_BASE_URL}/v1/metrics"))
        storage_before = parse_prometheus_text(await _fetch_metrics(c, f"{STORAGE_BASE_URL}/v1/metrics"))

    await _upload_docs(doc_ids)

    # Warm-up
    await _run_load(RunConfig(docs=cfg.docs, concurrency=1, duration_s=2, top_k=cfg.top_k, include_sources=cfg.include_sources), doc_ids)

    load_summary = await _run_load(cfg, doc_ids)

    # Post metrics
    async with httpx.AsyncClient(timeout=10.0) as c:
        gate_after = parse_prometheus_text(await _fetch_metrics(c, f"{GATE_BASE_URL}/v1/metrics"))
        retrieval_after = parse_prometheus_text(await _fetch_metrics(c, f"{RETRIEVAL_BASE_URL}/v1/metrics"))
        storage_after = parse_prometheus_text(await _fetch_metrics(c, f"{STORAGE_BASE_URL}/v1/metrics"))

    # Compute deltas (drop pid-specific labels if present in future)
    gate_d = diff_samples(gate_before, gate_after)
    retrieval_d = diff_samples(retrieval_before, retrieval_after)
    storage_d = diff_samples(storage_before, storage_after)

    gate_hist = extract_histogram(gate_d, metric_prefix="gate_latency_seconds", match_labels={}, group_by=["stage"])
    retrieval_hist = extract_histogram(retrieval_d, metric_prefix="rag_stage_latency_seconds", match_labels={}, group_by=["stage"])
    storage_hist = extract_histogram(storage_d, metric_prefix="storage_latency_seconds", match_labels={}, group_by=["stage"])

    print("\n== Client-side /v1/chat latency ==")
    print(load_summary)

    def render_hist(title: str, hist: dict):
        print(f"\n== {title} (histogram deltas) ==")
        keys = sorted(hist.keys())
        for k in keys:
            slot = hist[k]
            stage = k[0]
            cnt = float(slot.get("count") or 0.0)
            sm = float(slot.get("sum") or 0.0)
            mean = (sm / cnt) if cnt > 0 else 0.0
            p50 = histogram_quantile_from_cumulative(buckets=slot.get("buckets") or [], quantile=0.50)
            p95 = histogram_quantile_from_cumulative(buckets=slot.get("buckets") or [], quantile=0.95)
            p99 = histogram_quantile_from_cumulative(buckets=slot.get("buckets") or [], quantile=0.99)
            print(
                f"- {stage:24s} count={int(cnt):5d} mean={mean:7.3f}s p50={_fmt_seconds(p50)} p95={_fmt_seconds(p95)} p99={_fmt_seconds(p99)}"
            )

    render_hist("gate_latency_seconds", gate_hist)
    render_hist("retrieval rag_stage_latency_seconds", retrieval_hist)
    render_hist("storage_latency_seconds", storage_hist)


if __name__ == "__main__":
    asyncio.run(main())

















