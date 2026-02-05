#!/usr/bin/env python3
"""
Очистка индекса retrieval по списку doc_id из corpus.jsonl.

Поддерживает два режима:
- batch (по умолчанию): POST /v1/index/delete-batch — удаляет пачками по batch-size doc_id.
  Гораздо быстрее (57k doc_id = ~115 запросов вместо 57k).
- legacy: POST /v1/index/delete для каждого doc_id (fallback, если retrieval без batch API).

Пример:
  python eval/clear_retrieval_by_corpus.py --corpus data/beir/fiqa/corpus.jsonl --retrieval-url http://localhost:8080 --limit 500
  python eval/clear_retrieval_by_corpus.py --corpus data/beir/fiqa/corpus.jsonl --retrieval-url http://localhost:8080 --batch-size 1000
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

BATCH_SIZE_DEFAULT = 500


def _iter_doc_ids(path: Path, *, limit: int | None) -> "iter[str]":
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
            did = obj.get("doc_id") or obj.get("_id")
            if did:
                yield str(did)
                n += 1
                if limit is not None and limit > 0 and n >= limit:
                    return


async def _delete_one(client: httpx.AsyncClient, url: str, doc_id: str) -> bool:
    r = await client.post(url, json={"doc_id": doc_id, "refresh": False})
    if r.status_code != 200:
        return False
    try:
        return bool(r.json().get("ok"))
    except Exception:
        return False


async def _run_parallel(
    *,
    doc_ids: list[str],
    url: str,
    timeout_s: float,
    concurrency: int,
    show_progress: bool,
) -> tuple[int, int]:
    concurrency = max(1, int(concurrency))
    sem = asyncio.Semaphore(concurrency)
    failed = 0
    ok = 0

    iterator = doc_ids
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=len(doc_ids), desc="delete", unit="doc")

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async def _task(did: str) -> None:
            nonlocal failed, ok
            async with sem:
                try:
                    good = await _delete_one(client, url, did)
                except Exception:
                    good = False
                if good:
                    ok += 1
                else:
                    failed += 1
                if pbar is not None:
                    pbar.update(1)

        tasks: list[asyncio.Task[None]] = []
        for did in iterator:
            tasks.append(asyncio.create_task(_task(did)))
        await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()
    return ok, failed


def _run_batch(
    *,
    doc_ids: list[str],
    base_url: str,
    timeout_s: float,
    batch_size: int,
    show_progress: bool,
) -> tuple[int, int]:
    """Use /v1/index/delete-batch for bulk deletion. Returns (ok_count, failed_count)."""
    url = base_url.rstrip("/") + "/v1/index/delete-batch"
    failed = 0
    total_deleted = 0
    batches = [doc_ids[i : i + batch_size] for i in range(0, len(doc_ids), batch_size)]
    iterator = batches if not show_progress or tqdm is None else tqdm(batches, desc="delete-batch", unit="batch")
    with httpx.Client(timeout=timeout_s) as client:
        for batch in iterator:
            try:
                r = client.post(url, json={"doc_ids": batch, "refresh": False, "batch_size": batch_size})
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok"):
                        total_deleted += int(data.get("deleted") or 0)
                    else:
                        failed += len(batch)
                elif r.status_code in (404, 405):
                    raise RuntimeError("delete-batch not available")
                else:
                    failed += len(batch)
            except RuntimeError:
                raise
            except Exception:
                failed += len(batch)
    return total_deleted, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Удалить из retrieval все doc_id из corpus.jsonl")
    parser.add_argument("--corpus", required=True, help="Путь к corpus.jsonl")
    parser.add_argument(
        "--retrieval-url",
        default="http://localhost:8080",
        help="Base URL retrieval (default: http://localhost:8080)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Максимум doc_id для удаления")
    parser.add_argument("--no-progress", action="store_true", help="Отключить progress bar")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout на один delete")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Параллелизм запросов delete в legacy режиме (default: 1; для ускорения можно 10-80)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Не использовать delete-batch, всегда делать по одному doc_id (legacy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Размер пачки для delete-batch (default: 500)",
    )
    args = parser.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    doc_ids = list(_iter_doc_ids(path, limit=args.limit))
    if not doc_ids:
        print("Нет doc_id в корпусе.", file=sys.stderr)
        return 1

    base_url = args.retrieval_url.rstrip("/")
    url = base_url + "/v1/index/delete"
    failed = 0

    use_batch = not args.no_batch
    if use_batch:
        try:
            total_deleted, failed = _run_batch(
                doc_ids=doc_ids,
                base_url=base_url,
                timeout_s=max(float(args.timeout), 120.0),
                batch_size=max(1, int(args.batch_size)),
                show_progress=not args.no_progress,
            )
        except RuntimeError as e:
            if "not available" in str(e):
                print("delete-batch недоступен, переключаюсь на legacy (по одному doc_id)...", file=sys.stderr)
                use_batch = False
            else:
                raise
    if not use_batch:
        if int(args.concurrency) <= 1:
            iterator = doc_ids if args.no_progress else (tqdm(doc_ids, desc="delete") if tqdm else doc_ids)
            with httpx.Client(timeout=args.timeout) as client:
                for doc_id in iterator:
                    r = client.post(url, json={"doc_id": doc_id, "refresh": False})
                    if not (r.status_code == 200 and r.json().get("ok")):
                        failed += 1
                        if failed <= 3:
                            print(f"\n{doc_id}: {r.status_code} {r.text[:150]}", file=sys.stderr)
        else:
            ok, failed = asyncio.run(
                _run_parallel(
                    doc_ids=doc_ids,
                    url=url,
                    timeout_s=float(args.timeout),
                    concurrency=int(args.concurrency),
                    show_progress=not args.no_progress,
                )
            )

    # один refresh в конце
    try:
        with httpx.Client(timeout=args.timeout) as client:
            client.post(url, json={"doc_id": doc_ids[-1], "refresh": True})
    except Exception:
        pass

    print(f"\nУдалено doc_id: {len(doc_ids)}, ошибок: {failed}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
