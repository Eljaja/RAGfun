#!/usr/bin/env python3
"""
Upload a file to rag-gate multiple times with fixed concurrency.

Requires: pip install requests
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
import uuid
from pathlib import Path

import requests


def _upload_one(
    *,
    url: str,
    file_path: Path,
    title: str,
    source: str,
    lang: str,
    timeout_s: float,
) -> tuple[bool, float]:
    doc_id = f"speed-{uuid.uuid4().hex}"
    start = time.perf_counter()
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "text/plain")}
        data = {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "lang": lang,
        }
        r = requests.post(url, files=files, data=data, timeout=timeout_s)
    ok = r.status_code >= 200 and r.status_code < 300
    return ok, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload file to rag-gate multiple times.")
    parser.add_argument("--url", default="http://localhost:8090/v1/documents/upload")
    parser.add_argument("--file", required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--title", default=None)
    parser.add_argument("--source", default="speedtest")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"file not found: {file_path}")

    title = args.title or file_path.name
    total = max(1, int(args.total))
    workers = max(1, int(args.workers))

    start = time.perf_counter()
    ok = 0
    fail = 0
    latencies: list[float] = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _upload_one,
                url=args.url,
                file_path=file_path,
                title=title,
                source=args.source,
                lang=args.lang,
                timeout_s=float(args.timeout),
            )
            for _ in range(total)
        ]
        for f in cf.as_completed(futs):
            try:
                success, dur = f.result()
                latencies.append(dur)
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1

    elapsed = time.perf_counter() - start
    rps = total / elapsed if elapsed > 0 else 0.0
    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))] if latencies else 0.0
    p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0.0

    print(f"total={total} ok={ok} fail={fail} duration_s={elapsed:.2f} rps={rps:.2f}")
    print(f"latency_p50_s={p50:.3f} latency_p95_s={p95:.3f}")


if __name__ == "__main__":
    main()
