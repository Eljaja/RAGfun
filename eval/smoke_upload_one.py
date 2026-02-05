#!/usr/bin/env python3
"""
Проверка полного пайплайна: загрузка одного .txt документа через Gate и ожидание indexed=true.

Убеждаемся, что цепочка работает: Gate → storage → RabbitMQ → ingestion-worker → doc-processor → retrieval.

Пример:
  python eval/smoke_upload_one.py --gate-url http://localhost:8090
  python eval/smoke_upload_one.py --gate-url http://localhost:8090 --wait-timeout 120
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from urllib.parse import quote

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload one .txt via Gate and wait for indexed")
    parser.add_argument("--gate-url", default="http://localhost:8090", help="Gate base URL")
    parser.add_argument("--doc-id", default="smoke-test-doc-1", help="doc_id for the test document")
    parser.add_argument("--wait-timeout", type=float, default=90.0, help="Seconds to wait for indexed")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Status poll interval (sec)")
    args = parser.parse_args()

    gate = args.gate_url.rstrip("/")
    upload_url = f"{gate}/v1/documents/upload"
    status_url = f"{gate}/v1/documents/{quote(args.doc_id, safe='')}/status"

    body = b"Smoke test document. This is a single paragraph for pipeline check."
    files = {"file": ("smoke.txt", io.BytesIO(body), "text/plain; charset=utf-8")}
    data = {"doc_id": args.doc_id}

    print(f"Uploading doc_id={args.doc_id} to {upload_url}...", file=sys.stderr)
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(upload_url, files=files, data=data)
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return 1

    if r.status_code not in (200, 202):
        print(f"Upload HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return 1

    if r.status_code == 202:
        print("202 Accepted — task queued. Waiting for indexed...", file=sys.stderr)
    else:
        print("200 OK — legacy path (direct index). Checking status...", file=sys.stderr)

    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                s = client.get(status_url)
        except Exception as e:
            print(f"Status check failed: {e}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue
        if s.status_code != 200:
            time.sleep(args.poll_interval)
            continue
        data = s.json()
        if data.get("indexed") is True:
            print("indexed=true. Pipeline OK.", file=sys.stderr)
            return 0
        time.sleep(args.poll_interval)

    print("Timeout: indexed still false. Check ingestion-worker and doc-processor.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
