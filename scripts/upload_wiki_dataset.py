#!/usr/bin/env python3
"""
Upload HuggingFace dataset (saved with datasets.save_to_disk) into gate.

Example:
  python3 scripts/upload_wiki_dataset.py \
    --dataset-dir wiki_dataset_en \
    --collection wiki-ru \
    --gate-url http://localhost:8090
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any, Optional

import httpx
import json
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from datasets import load_from_disk
except Exception as exc:  # pragma: no cover - runtime dependency guard
    print(
        "Missing dependency: datasets. Install with `pip install datasets`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _SimpleProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.count = 0

    def update(self, n: int = 1) -> None:
        self.count += n
        if self.total <= 0:
            return
        if self.count == self.total or self.count % 25 == 0:
            pct = (self.count / self.total) * 100
            msg = f"\rProgress: {self.count}/{self.total} ({pct:.1f}%)"
            sys.stderr.write(msg)
            sys.stderr.flush()
            if self.count == self.total:
                sys.stderr.write("\n")

    def close(self) -> None:
        if self.count < self.total:
            sys.stderr.write("\n")


def _create_progress(total: int):
    if total <= 0:
        return None
    if tqdm:
        return tqdm(total=total, unit="doc")
    return _SimpleProgress(total)


def _sanitize_id(value: str) -> str:
    value = value.strip()
    if not value:
        return "unknown"
    # Keep ascii-ish separators for safe doc_id usage.
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unknown"


def _pick_field(column_names: list[str], preferred: str, fallbacks: list[str]) -> Optional[str]:
    if preferred in column_names:
        return preferred
    for name in fallbacks:
        if name in column_names:
            return name
    return None


def _make_doc_id(collection: str, record: dict[str, Any], idx: int, id_field: Optional[str], title: str) -> str:
    if id_field and record.get(id_field) is not None:
        base = str(record[id_field])
    elif title:
        base = title
    else:
        base = str(idx)
    return f"{collection}:{_sanitize_id(base)}"


async def _upload_record(
    client: httpx.AsyncClient,
    gate_url: str,
    collection: str,
    record: dict[str, Any],
    idx: int,
    title_field: str,
    text_field: str,
    id_field: Optional[str],
    lang: Optional[str],
    source: Optional[str],
):
    title = str(record.get(title_field) or "").strip()
    text = str(record.get(text_field) or "")
    if not text.strip():
        return {"ok": False, "error": "empty_text", "idx": idx}

    doc_id = _make_doc_id(collection, record, idx, id_field, title)
    filename = f"{_sanitize_id(title or doc_id)}.txt"

    files = {"file": (filename, text.encode("utf-8"), "text/plain")}
    data = {
        "doc_id": doc_id,
        "title": title or doc_id,
        "project_id": collection,
    }
    if lang:
        data["lang"] = lang
    if source:
        data["source"] = source

    response = await client.post(f"{gate_url}/v1/documents/upload", files=files, data=data, timeout=300.0)
    response.raise_for_status()
    result = response.json()

    downstream_ok = True
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        if "ok" in result["result"] and result["result"].get("ok") is not True:
            downstream_ok = False

    if result.get("ok") and downstream_ok:
        return {"ok": True, "doc_id": doc_id, "idx": idx}
    err = result.get("error", "unknown")
    return {"ok": False, "error": err, "idx": idx, "doc_id": doc_id}


async def main_async(args: argparse.Namespace) -> int:
    ds = load_from_disk(args.dataset_dir)
    if isinstance(ds, dict) or hasattr(ds, "keys"):
        ds = ds["train"]

    column_names = list(ds.column_names)
    title_field = _pick_field(column_names, args.title_field, ["title", "page_title", "heading"])
    text_field = _pick_field(column_names, args.text_field, ["text", "content", "article"])

    if not title_field or not text_field:
        logger.error("Column names: %s", ", ".join(column_names))
        logger.error(
            "Cannot find title/text fields. Use --title-field/--text-field to specify."
        )
        return 1

    total = len(ds)
    start = max(0, args.offset)
    end = total if args.limit is None else min(total, start + args.limit)
    progress_path = Path(args.progress_file)
    completed: set[int] = set()
    if args.resume and progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if "idx" in payload:
                            completed.add(int(payload["idx"]))
                    except Exception:
                        continue
        except Exception as exc:
            logger.error("Failed to read progress file: %s", exc)

    logger.info("Dataset size: %s, uploading range: [%s, %s)", total, start, end)
    if completed:
        logger.info("Resume enabled: %s already uploaded entries", len(completed))
    logger.info("Using fields: title=%s, text=%s, id=%s", title_field, text_field, args.id_field)

    stats = {"ok": 0, "failed": 0}

    limits = httpx.Limits(
        max_connections=args.max_concurrent * 2,
        max_keepalive_connections=args.max_concurrent,
    )
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(start, end):
            if i in completed:
                continue
            queue.put_nowait(i)

        progress_total = queue.qsize()
        progress = _create_progress(progress_total)
        progress_lock = asyncio.Lock()
        file_lock = asyncio.Lock()

        async def worker() -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = ds[i]
                    result = await _upload_record(
                        client=client,
                        gate_url=args.gate_url,
                        collection=args.collection,
                        record=record,
                        idx=i,
                        title_field=title_field,
                        text_field=text_field,
                        id_field=args.id_field,
                        lang=args.lang,
                        source=args.source,
                    )
                except Exception as exc:
                    stats["failed"] += 1
                    logger.error("Upload failed: %s", exc)
                else:
                    if result.get("ok"):
                        stats["ok"] += 1
                        if args.progress_file:
                            async with file_lock:
                                with progress_path.open("a", encoding="utf-8") as fh:
                                    fh.write(json.dumps({"idx": i, "doc_id": result.get("doc_id")}) + "\n")
                    else:
                        stats["failed"] += 1
                        logger.error("Upload failed idx=%s error=%s", result.get("idx"), result.get("error"))
                finally:
                    if progress:
                        async with progress_lock:
                            progress.update(1)
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(args.max_concurrent)]
        await queue.join()
        for task in workers:
            task.cancel()
        if progress:
            progress.close()

    logger.info("Done. ok=%s failed=%s", stats["ok"], stats["failed"])
    return 0 if stats["failed"] == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload wiki dataset into gate collection.")
    parser.add_argument("--dataset-dir", default="wiki_dataset_en", help="Path to dataset saved by datasets.save_to_disk")
    parser.add_argument("--collection", default="wiki-ru", help="project_id/collection name in gate")
    parser.add_argument("--gate-url", default="http://localhost:8090", help="Gate base URL")
    parser.add_argument("--title-field", default="title", help="Dataset column for title")
    parser.add_argument("--text-field", default="text", help="Dataset column for text")
    parser.add_argument("--id-field", default=None, help="Dataset column for doc_id (optional)")
    parser.add_argument("--lang", default="ru", help="lang metadata for documents")
    parser.add_argument("--source", default="wiki_dataset_en", help="source metadata")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to upload")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent uploads")
    parser.add_argument(
        "--progress-file",
        default=None,
        help="JSONL file to track uploaded indices (default: <dataset-dir>/.upload_progress.jsonl)",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_file is None:
        args.progress_file = str(Path(args.dataset_dir) / ".upload_progress.jsonl")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
