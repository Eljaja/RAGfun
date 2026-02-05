#!/usr/bin/env python3
"""
Индексация BEIR-корпуса через полный пайплайн: Gate → storage → RabbitMQ → worker → doc-processor → retrieval.

Каждый документ из corpus.jsonl загружается как файл через POST /v1/documents/upload.
Doc-processor достаёт файл из storage, режет текст своей стратегией (semantic/fixed) и шлёт чанки в retrieval.
Требует: Gate, document-storage, RabbitMQ, ingestion-worker, doc-processor, retrieval.

Ожидается ответ 202 Accepted. Если 200 — сработал legacy-путь (без doc-processor).

Пример:
  python eval/prepare_beir_data.py --dataset scifact --out data/beir/scifact
  python eval/index_beir_via_gate.py --corpus data/beir/scifact/corpus.jsonl --gate-url http://localhost:8090 --limit 100
  # дождаться индексации, затем:
  python eval/run_rag_baseline.py --dataset data/beir/scifact/queries.jsonl --format beir --output data/beir/scifact/results.json
  python eval/run_rag_ir_metrics.py --results data/beir/scifact/results.json --qrels data/beir/scifact/qrels_test.tsv
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _iter_corpus_rows(path: Path, *, limit: int | None) -> "iter[dict[str, str]]":
    """
    Stream corpus.jsonl line-by-line to avoid loading the full corpus into memory.
    Yields dicts: {"doc_id": str, "text": str}.
    """
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("doc_id") or obj.get("_id")
            text = (obj.get("text") or "").strip()
            if not doc_id or not text:
                continue
            yield {"doc_id": str(doc_id), "text": text}
            n += 1
            if limit is not None and limit > 0 and n >= limit:
                return


def _count_corpus_rows(path: Path, *, limit: int | None) -> int:
    """
    Count eligible rows for progress bars. This does a lightweight pass.
    """
    n = 0
    for _ in _iter_corpus_rows(path, limit=limit):
        n += 1
    return n


def _fetch_stored_doc_ids(*, storage_url: str, page_size: int = 1000, timeout_s: float = 30.0) -> set[str]:
    """
    Load ALL doc_ids currently present in document-storage (best-effort).
    This is used to resume large uploads without re-enqueueing duplicates.
    """
    storage_url = storage_url.rstrip("/")
    out: set[str] = set()
    offset = 0
    total = None
    with httpx.Client(timeout=timeout_s) as client:
        while True:
            payload = {"source": None, "tags": [], "lang": None, "tenant_id": None, "project_ids": [], "limit": int(page_size), "offset": int(offset)}
            r = client.post(f"{storage_url}/v1/documents/search", json=payload)
            r.raise_for_status()
            data = r.json()
            docs = list((data or {}).get("documents") or [])
            if total is None:
                try:
                    total = int((data or {}).get("total") or 0)
                except Exception:
                    total = None
            if not docs:
                break
            for d in docs:
                doc_id = d.get("doc_id")
                if doc_id:
                    out.add(str(doc_id))
            offset += len(docs)
            if total is not None and offset >= total:
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Индексация BEIR corpus через Gate (полный пайплайн: doc-processor semantic chunking).",
    )
    parser.add_argument("--corpus", required=True, help="Путь к corpus.jsonl")
    parser.add_argument(
        "--gate-url",
        default="http://localhost:8090",
        help="Base URL Gate (default: http://localhost:8090)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Максимум документов")
    parser.add_argument(
        "--wait-indexed",
        action="store_true",
        help="После загрузки опрашивать статус до появления indexed=true по всем doc_id",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=1800.0,
        help="Таймаут ожидания индексации, сек (default: 1800)",
    )
    parser.add_argument(
        "--wait-interval",
        type=float,
        default=5.0,
        help="Интервал опроса статуса, сек (default: 5)",
    )
    parser.add_argument(
        "--no-fail-on-wait-timeout",
        action="store_true",
        help="При таймауте ожидания индексации не выходить с кодом 1 (продолжить; baseline может увидеть пустой индекс)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout на один upload")
    parser.add_argument("--no-progress", action="store_true", help="Отключить progress bar")
    parser.add_argument(
        "--storage-url",
        default=None,
        help="Base URL document-storage (optional; enables --skip-stored). Example: http://localhost:8083",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="Skip doc_ids already present in document-storage (requires --storage-url). Useful to resume after interruption.",
    )
    parser.add_argument("--storage-page-size", type=int, default=1000, help="Pagination size for document-storage scan (default: 1000)")
    args = parser.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    stored: set[str] | None = None
    if args.skip_stored:
        if not args.storage_url:
            print("--skip-stored requires --storage-url", file=sys.stderr)
            return 2
        print("Сканирую document-storage для resume/skip...", file=sys.stderr)
        try:
            stored = _fetch_stored_doc_ids(
                storage_url=str(args.storage_url),
                page_size=max(1, min(int(args.storage_page_size), 2000)),
                timeout_s=30.0,
            )
            print(f"Найдено doc_ids в storage: {len(stored)}", file=sys.stderr)
        except Exception as e:
            print(f"Не удалось прочитать storage для --skip-stored: {e}", file=sys.stderr)
            return 1

    total = None
    if not args.no_progress and tqdm is not None:
        try:
            if stored is None:
                total = _count_corpus_rows(path, limit=args.limit)
            else:
                # Count only missing docs (for a meaningful progress bar when resuming).
                n = 0
                for row in _iter_corpus_rows(path, limit=args.limit):
                    if row["doc_id"] not in stored:
                        n += 1
                total = n
        except Exception:
            total = None

    iterator = _iter_corpus_rows(path, limit=args.limit)
    if stored is not None:
        iterator = (row for row in iterator if row["doc_id"] not in stored)
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="upload")

    gate_url = args.gate_url.rstrip("/")
    upload_url = f"{gate_url}/v1/documents/upload"
    status_url_tpl = f"{gate_url}/v1/documents/{{doc_id}}/status"

    accepted = 0
    legacy = 0
    failed = 0
    doc_ids: list[str] = []

    total_seen = 0
    with httpx.Client(timeout=args.timeout) as client:
        for row in iterator:
            doc_id = row["doc_id"]
            text = row["text"]
            filename = f"{doc_id}.txt"
            body = text.encode("utf-8")
            total_seen += 1
            try:
                r = client.post(
                    upload_url,
                    files={"file": (filename, io.BytesIO(body), "text/plain; charset=utf-8")},
                    data={"doc_id": doc_id},
                )
                if r.status_code == 202:
                    accepted += 1
                    doc_ids.append(doc_id)
                elif r.status_code == 200:
                    legacy += 1
                    doc_ids.append(doc_id)
                    if legacy == 1:
                        print(
                            "\nВнимание: Gate вернул 200 (legacy), не 202. Полный пайплайн не использован (doc-processor не вызывался).",
                            file=sys.stderr,
                        )
                else:
                    failed += 1
                    print(f"\n{doc_id}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
            except Exception as e:
                failed += 1
                print(f"\n{doc_id}: {e}", file=sys.stderr)

    if total_seen == 0:
        print("Нет документов для загрузки.", file=sys.stderr)
        return 1

    print(f"\nЗагружено: 202 Accepted={accepted}, 200 (legacy)={legacy}, ошибок={failed}, всего={total_seen}", file=sys.stderr)
    if legacy and not args.no_progress:
        print("Полный пайплайн не использован: нужны Gate + storage + RabbitMQ для 202.", file=sys.stderr)

    if not doc_ids:
        return 1

    if args.wait_indexed:
        print("Ожидание индексации...", file=sys.stderr)
        deadline = time.monotonic() + args.wait_timeout
        remaining = set(doc_ids)
        with httpx.Client(timeout=30.0) as client:
            while remaining and time.monotonic() < deadline:
                for doc_id in list(remaining):
                    try:
                        r = client.get(status_url_tpl.format(doc_id=quote(str(doc_id), safe="")))
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("indexed") is True:
                                remaining.discard(doc_id)
                    except Exception:
                        pass
                if remaining:
                    time.sleep(args.wait_interval)
        if remaining:
            print(f"Таймаут: не проиндексированы {len(remaining)} из {len(doc_ids)}", file=sys.stderr)
            print(
                "Проверьте: ingestion-worker и doc-processor запущены и подключены к той же RabbitMQ и retrieval? "
                "Gate отдаёт 202 — документы в очереди; индекс обновляется только после обработки воркером.",
                file=sys.stderr,
            )
            if not args.no_fail_on_wait_timeout:
                return 1
        else:
            print("Все документы проиндексированы.", file=sys.stderr)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
