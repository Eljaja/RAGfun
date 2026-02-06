#!/usr/bin/env python3
"""
Index BRIGHT corpus into RAGfun retrieval service.

Loads documents from HuggingFace, indexes via POST /v1/index/upsert (mode=document).
Use project_id=bright for filtering during eval.

Usage:
  python scripts/index_bright.py --retrieval-url http://localhost:8080 \\
    [--splits biology,economics,...] [--limit 1000] [--concurrency 10]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

BRIGHT_SPLITS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]


async def _index_one(
    client: httpx.AsyncClient,
    url: str,
    doc_id: str,
    content: str,
    project_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, bool, str | None]:
    """Index one document. Returns (doc_id, ok, error)."""
    if not content or not content.strip():
        return (doc_id, True, None)

    payload = {
        "mode": "document",
        "document": {"doc_id": doc_id, "project_id": project_id, "source": "bright"},
        "text": content,
        "refresh": False,
    }

    async with semaphore:
        try:
            r = await client.post(f"{url.rstrip('/')}/v1/index/upsert", json=payload)
            r.raise_for_status()
            return (doc_id, True, None)
        except Exception as e:
            return (doc_id, False, str(e))


def _get_gold_doc_ids(split: str, limit: int) -> set[str]:
    """Load examples, collect gold doc ids, return first `limit` unique."""
    from datasets import load_dataset

    ds = load_dataset("xlangai/BRIGHT", "examples", split=split)
    seen: set[str] = set()
    ids: list[str] = []
    for i in range(len(ds)):
        gold = ds[i].get("gold_ids") or ds[i].get("gold_ids_long") or []
        if isinstance(gold, str):
            gold = [gold]
        for g in gold[:20]:
            if g and g not in seen:
                seen.add(g)
                ids.append(g)
                if len(ids) >= limit:
                    return set(ids)
    return set(ids)


async def index_bright(
    retrieval_url: str,
    splits: list[str],
    project_id: str = "bright",
    limit: int | None = None,
    docs_from_gold: int | None = None,
    concurrency: int = 10,
    timeout_s: float = 120.0,
) -> dict:
    """Load BRIGHT documents and index into RAGfun."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        sys.exit(1)

    url = retrieval_url.rstrip("/")
    semaphore = asyncio.Semaphore(concurrency)
    indexed = 0
    failed = 0

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for split in splits:
            if docs_from_gold:
                print(f"Getting gold doc ids from examples/{split} (limit={docs_from_gold})...", file=sys.stderr)
                ids_to_index = _get_gold_doc_ids(split, docs_from_gold)
                print(f"  Will index {len(ids_to_index)} docs referenced by questions", file=sys.stderr)
            else:
                ids_to_index = None

            print(f"Loading documents/{split}...", file=sys.stderr)
            try:
                # Load full split (biology ~57k rows). Streaming scan of 57k to find 372 docs is too slow.
                ds = load_dataset("xlangai/BRIGHT", "documents", split=split)
            except Exception as e:
                print(f"  Skip {split}: {e}", file=sys.stderr)
                continue

            async def _flush(batch_tasks: list[asyncio.Task]) -> None:
                nonlocal indexed, failed
                if not batch_tasks:
                    return
                results = await asyncio.gather(*batch_tasks)
                for doc_id, ok, err in results:
                    if ok:
                        indexed += 1
                    else:
                        failed += 1
                        if failed <= 5:
                            print(f"  Failed {doc_id}: {err}", file=sys.stderr)

            if ids_to_index is not None:
                # Fast path: Arrow filter (like ifedotov/rag_fun) — O(1) vs streaming O(57k).
                need = set(ids_to_index)
                rows_to_index: list[tuple[str, str]] = []

                try:
                    import pyarrow as pa
                    import pyarrow.compute as pc

                    # HuggingFace Dataset: data is in .data (DatasetDict) or ._data
                    table = None
                    if hasattr(ds, "data") and ds.data is not None:
                        tbl = getattr(ds.data, "table", None)
                        if tbl is None and hasattr(ds.data, "__getitem__"):
                            tbl = ds.data[ds._data_files[0].filename] if getattr(ds, "_data_files", None) else None
                        table = tbl
                    if table is None and hasattr(ds, "_data"):
                        table = getattr(ds._data, "table", None)

                    if table is not None and "id" in table.column_names and "content" in table.column_names:
                        mask = pc.is_in(table["id"], value_set=pa.array(list(need)))
                        filtered = table.filter(mask)
                        ids_col = filtered["id"].to_pylist()
                        contents_col = filtered["content"].to_pylist()
                        rows_to_index = [(str(i).strip(), str(c or "")) for i, c in zip(ids_col, contents_col, strict=False) if str(i).strip()]
                except Exception:
                    pass

                if not rows_to_index:
                    # Fallback: batched scan (early-exit)
                    bs = 2048
                    for off in range(0, len(ds), bs):
                        batch = ds[off : off + bs]
                        ids_b = batch.get("id") or []
                        contents_b = batch.get("content") or []
                        for did, content in zip(ids_b, contents_b, strict=False):
                            sdid = (did or "").strip() if did else ""
                            if sdid and sdid in need:
                                rows_to_index.append((sdid, str(content or "")))
                                need.discard(sdid)
                        if not need:
                            break

                print(f"  Indexing {len(rows_to_index)} docs...", file=sys.stderr)
                batch: list[asyncio.Task] = []
                for doc_id, content in rows_to_index:
                    batch.append(asyncio.create_task(_index_one(client, url, doc_id, content, project_id, semaphore)))
                    if len(batch) >= max(50, concurrency * 8):
                        await _flush(batch)
                        batch = []
                await _flush(batch)
                print(f"  Done split={split}: indexed {len(rows_to_index)} docs", file=sys.stderr)
            else:
                n = len(ds)
                if limit:
                    n = min(n, limit)

                print(f"  Indexing {n} docs from {split}...", file=sys.stderr)
                batch: list[asyncio.Task] = []
                for i in range(n):
                    row = ds[i]
                    doc_id = row.get("id", "")
                    content = row.get("content", "")
                    batch.append(asyncio.create_task(_index_one(client, url, doc_id, content, project_id, semaphore)))
                    if len(batch) >= max(50, concurrency * 8):
                        await _flush(batch)
                        batch = []
                        if (i + 1) % 200 == 0:
                            print(f"  progress split={split}: {i+1}/{n} (indexed={indexed} failed={failed})", file=sys.stderr)
                await _flush(batch)

                if limit and indexed >= limit:
                    break

    return {"indexed": indexed, "failed": failed, "splits": splits, "project_id": project_id}


def main() -> None:
    p = argparse.ArgumentParser(description="Index BRIGHT corpus into RAGfun")
    p.add_argument("--retrieval-url", default="http://localhost:8080")
    p.add_argument("--project-id", default="bright")
    p.add_argument("--splits", default=",".join(BRIGHT_SPLITS), help="Comma-separated splits")
    p.add_argument("--limit", type=int, default=None, help="Max docs per split (for testing)")
    p.add_argument("--docs-from-gold", type=int, default=None, help="Index only N docs that appear in questions' gold_ids (best for small eval)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = BRIGHT_SPLITS

    # Connectivity check (retry: retrieval may be slow under load from other containers)
    url = args.retrieval_url.rstrip("/")
    for attempt in range(1, 6):
        try:
            r = httpx.get(f"{url}/v1/healthz", timeout=60.0)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt < 5:
                print(f"Retrieval check attempt {attempt}/5 failed: {e}, retrying in 5s...", file=sys.stderr)
                time.sleep(5)
            else:
                print(f"Retrieval unreachable after 5 attempts: {e}", file=sys.stderr)
                sys.exit(1)

    print(f"Indexing BRIGHT into {args.retrieval_url} (project_id={args.project_id})", file=sys.stderr)
    result = asyncio.run(
        index_bright(
            retrieval_url=args.retrieval_url,
            splits=splits,
            project_id=args.project_id,
            limit=args.limit,
            docs_from_gold=args.docs_from_gold,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        )
    )
    print(f"Done: indexed={result['indexed']}, failed={result['failed']}", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
