#!/usr/bin/env python3
"""
Диагностика пайплайна индексации: Gate → storage → RabbitMQ → ingestion-worker → doc-processor → retrieval.

Проверяет доступность сервисов и (опционально) глубину очереди RabbitMQ.
Если очередь растёт, а документы не индексируются — ingestion-worker или doc-processor не работают.

Пример:
  python eval/check_ingestion_pipeline.py --gate-url http://localhost:8090 --retrieval-url http://localhost:8080
  python eval/check_ingestion_pipeline.py --rabbit-url http://localhost:15672 --rabbit-user guest --rabbit-pass guest
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import quote

import httpx


def check(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code == 200:
            return True, f"OK ({r.status_code})"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка цепочки индексации (Gate, retrieval, RabbitMQ).")
    parser.add_argument("--gate-url", default="http://localhost:8090", help="URL Gate")
    parser.add_argument("--retrieval-url", default="http://localhost:8080", help="URL retrieval")
    parser.add_argument("--storage-url", default="http://localhost:8081", help="URL document-storage")
    parser.add_argument(
        "--rabbit-url",
        default=None,
        help="URL RabbitMQ Management (e.g. http://localhost:15672) для проверки очереди",
    )
    parser.add_argument("--rabbit-user", default="guest", help="RabbitMQ user")
    parser.add_argument("--rabbit-pass", default="guest", help="RabbitMQ password")
    parser.add_argument("--queue", default="ingestion.tasks", help="Имя очереди")
    parser.add_argument("--json", action="store_true", help="Вывести JSON")
    args = parser.parse_args()

    gate_base = args.gate_url.rstrip("/")
    retrieval_base = args.retrieval_url.rstrip("/")
    storage_base = args.storage_url.rstrip("/")

    results: dict[str, dict[str, str | bool]] = {}

    # Gate
    ok, msg = check(f"{gate_base}/v1/healthz")
    results["gate"] = {"ok": ok, "url": f"{gate_base}/v1/healthz", "message": msg}

    # Retrieval
    ok, msg = check(f"{retrieval_base}/v1/healthz")
    results["retrieval"] = {"ok": ok, "url": f"{retrieval_base}/v1/healthz", "message": msg}

    # Document-storage: readyz показывает готовность к приёму документов (db + storage backend)
    try:
        r = httpx.get(f"{storage_base}/v1/readyz", timeout=5.0)
        if r.status_code == 200:
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ready = data.get("ready", True)
            results["document_storage"] = {
                "ok": ready,
                "url": f"{storage_base}/v1/readyz",
                "message": f"ready={ready}, db={data.get('db', '?')}, storage={data.get('storage', '?')}" + (
                    f", config_error={data.get('config_error', '')}" if data.get("config_error") else ""
                ),
            }
        else:
            results["document_storage"] = {"ok": False, "url": f"{storage_base}/v1/readyz", "message": f"HTTP {r.status_code}"}
    except Exception as e:
        results["document_storage"] = {"ok": False, "url": f"{storage_base}/v1/readyz", "message": str(e)}

    # RabbitMQ queue depth (if Management API available)
    rabbit_queue_depth: int | None = None
    rabbit_ready: bool | None = None
    if args.rabbit_url:
        # GET /api/queues/%2F/<queue_name> (vhost / = %2F)
        vhost = quote("/", safe="")
        queue_url = f"{args.rabbit_url.rstrip('/')}/api/queues/{vhost}/{quote(args.queue, safe='')}"
        try:
            r = httpx.get(queue_url, auth=(args.rabbit_user, args.rabbit_pass), timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                rabbit_ready = True
                rabbit_queue_depth = int(data.get("messages", 0)) + int(data.get("messages_ready", 0))
            else:
                rabbit_ready = False
                rabbit_queue_depth = None
        except Exception as e:
            rabbit_ready = False
            rabbit_queue_depth = None
        results["rabbitmq_queue"] = {
            "ok": rabbit_ready is True,
            "url": queue_url,
            "message": f"messages={rabbit_queue_depth}" if rabbit_queue_depth is not None else str(e) if rabbit_ready is False else "?",
            "messages": rabbit_queue_depth,
        }

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    # Human output
    all_ok = all(r.get("ok") for r in results.values())
    for name, r in results.items():
        status = "OK" if r.get("ok") else "FAIL"
        print(f"  {name}: {status} — {r.get('message', '')}", file=sys.stderr)
    if rabbit_queue_depth is not None:
        print(f"\n  Очередь {args.queue}: сообщений = {rabbit_queue_depth}", file=sys.stderr)
        if rabbit_queue_depth > 0:
            print(
                "  Если число не уменьшается — ingestion-worker или doc-processor не обрабатывают очередь. "
                "Проверьте: docker ps (ingestion-worker, doc-processor), docker logs ingestion-worker.",
                file=sys.stderr,
            )

    print("\nЦепочка для полного пайплайна (202 Accepted → indexed):", file=sys.stderr)
    print("  Gate → document-storage (сохранение) → RabbitMQ (очередь ingestion.tasks)", file=sys.stderr)
    print("  → ingestion-worker (consumes) → doc-processor (/v1/process) → retrieval (index)", file=sys.stderr)
    print("  Нужны: Gate, document-storage, RabbitMQ, ingestion-worker, doc-processor, retrieval.", file=sys.stderr)
    print("  Для .txt файлов VLM (vllm-docling) не обязателен.", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
