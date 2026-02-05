#!/usr/bin/env python3
"""
Индексация корпуса BEIR (corpus.jsonl) в retrieval.

Читает corpus.jsonl из prepare_beir_data.py (поля doc_id, text) и для каждого
документа вызывает POST /v1/index/upsert с mode=document, чтобы doc_id в индексе
совпадал с BEIR — тогда run_rag_ir_metrics.py сможет сравнить выдачу с qrels.

Требует: retrieval сервис доступен (OpenSearch, Qdrant, эмбеддинги).

Пример:
  python eval/prepare_beir_data.py --dataset scifact --out data/beir/scifact
  python eval/index_beir_corpus.py --corpus data/beir/scifact/corpus.jsonl --retrieval-url http://localhost:8080
  python eval/index_beir_corpus.py --corpus data/beir/fiqa/corpus.jsonl --retrieval-url http://localhost:8080 --concurrency 10
  python eval/run_rag_baseline.py --dataset data/beir/scifact/queries.jsonl --format beir --output data/beir/scifact/results.json
  python eval/run_rag_ir_metrics.py --results data/beir/scifact/results.json --qrels data/beir/scifact/qrels_test.tsv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _iter_corpus_rows(path: Path, *, limit: int | None) -> "iter[dict[str, str]]":
    """
    Stream corpus.jsonl line-by-line to avoid loading full corpus into memory.
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
    n = 0
    for _ in _iter_corpus_rows(path, limit=limit):
        n += 1
    return n


async def _upsert_one(
    client: httpx.AsyncClient,
    url: str,
    row: dict[str, str],
    *,
    refresh: bool,
    sem: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Index one document. Returns (ok, doc_id)."""
    doc_id = row["doc_id"]
    text = row["text"]
    payload = {
        "mode": "document",
        "document": {"doc_id": doc_id},
        "text": text,
        "refresh": refresh,
    }
    async with sem:
        try:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                data = r.json()
                return (bool(data.get("ok")), doc_id)
            return (False, doc_id)
        except Exception:
            return (False, doc_id)


async def _run_parallel(
    rows: list[dict[str, str]],
    url: str,
    timeout_s: float,
    concurrency: int,
    refresh_last: bool,
    show_progress: bool,
) -> tuple[int, int]:
    """Parallel indexing via asyncio. Returns (ok_count, err_count)."""
    sem = asyncio.Semaphore(concurrency)
    ok_count = 0
    err_count = 0

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        tasks = []
        for i, row in enumerate(rows):
            is_last = refresh_last and (i == len(rows) - 1)
            tasks.append(asyncio.create_task(_upsert_one(client, url, row, refresh=is_last, sem=sem)))

        iterator = asyncio.as_completed(tasks)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(tasks), desc="index", unit="doc")

        for done in iterator:
            ok, doc_id = await done
            if ok:
                ok_count += 1
            else:
                err_count += 1

    return ok_count, err_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация corpus.jsonl в retrieval (BEIR doc_id сохраняется).")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Путь к corpus.jsonl (строки: {\"doc_id\": \"...\", \"text\": \"...\"})",
    )
    parser.add_argument(
        "--retrieval-url",
        default="http://localhost:8080",
        help="Base URL retrieval сервиса (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимум документов (default: все)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Передавать refresh=true при последнем документе",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout на один документ (сек)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Отключить progress bar",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Параллелизм запросов (default: 1; 5-20 для ускорения, осторожно с нагрузкой)",
    )
    args = parser.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    url = args.retrieval_url.rstrip("/") + "/v1/index/upsert"
    concurrency = max(1, int(args.concurrency))

    if concurrency > 1:
        rows = list(_iter_corpus_rows(path, limit=args.limit))
        if not rows:
            print("Нет документов для индексации.", file=sys.stderr)
            return 1
        ok_count, err_count = asyncio.run(
            _run_parallel(
                rows=rows,
                url=url,
                timeout_s=float(args.timeout),
                concurrency=concurrency,
                refresh_last=args.refresh,
                show_progress=not args.no_progress,
            )
        )
    else:
        ok_count = 0
        err_count = 0
        total = None
        if not args.no_progress and tqdm is not None:
            try:
                total = _count_corpus_rows(path, limit=args.limit)
            except Exception:
                total = None
        iterator = _iter_corpus_rows(path, limit=args.limit)
        if not args.no_progress and tqdm is not None:
            iterator = tqdm(iterator, total=total, desc="index")

        with httpx.Client(timeout=args.timeout) as client:
            seen = 0
            for row in iterator:
                seen += 1
                doc_id = row["doc_id"]
                text = row["text"]
                is_last = bool(total) and seen == int(total)
                payload = {
                    "mode": "document",
                    "document": {"doc_id": doc_id},
                    "text": text,
                    "refresh": args.refresh and is_last,
                }
                try:
                    r = client.post(url, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("ok") is True:
                            ok_count += 1
                        else:
                            err_count += 1
                            if err_count <= 3:
                                print(f"\n{doc_id}: ok=false {data.get('errors', data)}", file=sys.stderr)
                    else:
                        err_count += 1
                        if err_count <= 3:
                            print(f"\n{doc_id}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
                except Exception as e:
                    err_count += 1
                    if err_count <= 3:
                        print(f"\n{doc_id}: {e}", file=sys.stderr)

    if ok_count == 0 and err_count == 0:
        print("Нет документов для индексации.", file=sys.stderr)
        return 1
    total_seen = ok_count + err_count
    print(f"\nИндексировано: {ok_count} ok, {err_count} ошибок, всего {total_seen}", file=sys.stderr)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
