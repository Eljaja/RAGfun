from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from bench.t2_hf_io import T2Row, iter_t2_rows
from bench.t2_metrics import number_match, retrieval_stats


RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")
GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")


def _wait_ready(url: str, *, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=5.0) as c:
        while time.time() < deadline:
            try:
                r = c.get(url)
                if r.status_code == 200 and r.json().get("ready"):
                    return
            except Exception:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"not ready: {url}")


def _index_contexts(
    *,
    rows: list[T2Row],
    mode: str,
    batch_docs: int,
    refresh: bool,
) -> int:
    """
    Index unique contexts into retrieval as documents.
    doc_id == context_id.
    """
    ctx_meta: dict[str, dict[str, Any]] = {}
    ctx_text: dict[str, str] = {}
    for r in rows:
        if not r.context_id or not r.context:
            continue
        if r.context_id not in ctx_meta:
            ctx_meta[r.context_id] = {
                "doc_id": r.context_id,
                "title": r.file_name or r.context_id,
                "uri": r.file_name,
                "source": "t2-ragbench",
                "tags": [],
            }
            ctx_text[r.context_id] = r.context

    items = list(ctx_meta.items())
    total = 0
    failed = []
    # Increased timeout for indexing large documents
    # Disable refresh during indexing for performance, refresh only at the end
    with httpx.Client(timeout=900.0) as c:  # Increased to 15 minutes for very large documents
        for i in tqdm(range(0, len(items), batch_docs), desc="index_docs"):
            batch = items[i : i + batch_docs]
            # send one-by-one to keep payload sizes moderate (contexts can be large)
            for doc_id, meta in batch:
                text = ctx_text.get(doc_id, "")
                text_size_mb = len(text.encode('utf-8')) / (1024 * 1024)
                
                # Log large documents
                if text_size_mb > 1.0:
                    print(f"  Large document {doc_id}: {text_size_mb:.2f} MB, {len(text)} chars")
                
                # Only refresh on last document if refresh is requested
                should_refresh = bool(refresh) and (i + batch_docs >= len(items)) and (doc_id == batch[-1][0])
                payload = {
                    "mode": "document",
                    "document": meta,
                    "text": text,
                    "refresh": should_refresh,
                }
                # Retry on timeout or connection errors with longer delays
                last_error = None
                for attempt in range(5):  # More retries for large documents
                    try:
                        r = c.post(f"{RETRIEVAL_BASE_URL}/v1/index/upsert", json=payload)
                        r.raise_for_status()
                        break
                    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
                        last_error = e
                        if attempt == 4:  # Last attempt
                            print(f"  ERROR: Failed to index {doc_id} after 5 attempts: {type(e).__name__}")
                            failed.append((doc_id, str(e)))
                            break
                        # Longer backoff for large documents
                        backoff = min(10.0 * (2 ** attempt), 60.0)  # Max 60 seconds
                        print(f"  Retry {attempt + 1}/5 for {doc_id} after {backoff:.1f}s...")
                        time.sleep(backoff)
                    except httpx.HTTPStatusError as e:
                        print(f"  ERROR: HTTP {e.response.status_code} for {doc_id}: {e.response.text[:200]}")
                        failed.append((doc_id, f"HTTP {e.response.status_code}"))
                        break
                    except Exception as e:
                        print(f"  ERROR: Unexpected error for {doc_id}: {type(e).__name__}: {str(e)}")
                        failed.append((doc_id, str(e)))
                        break
                else:
                    # Success
                    total += 1
    
    if failed:
        print(f"\nWARNING: Failed to index {len(failed)} documents:")
        for doc_id, error in failed[:10]:  # Show first 10
            print(f"  - {doc_id}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return total


@retry(
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _retrieval_rank(*, query: str, gold_context_id: str, k: int, mode: str) -> int | None:
    payload = {
        "query": query,
        "mode": mode,
        "top_k": k,
        "include_sources": False,
        "sources_level": "none",
        "group_by_doc": True,
        "max_chunks_per_doc": 1,
        "filters": None,
        "acl": [],
    }
    # Increased timeout for search operations
    with httpx.Client(timeout=180.0) as c:
        r = c.post(f"{RETRIEVAL_BASE_URL}/v1/search", json=payload)
        r.raise_for_status()
        j = r.json()
    hits = j.get("hits") or []
    for idx, h in enumerate(hits, start=1):
        if str(h.get("doc_id")) == str(gold_context_id):
            return idx
    return None


@retry(
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _generate_answer(*, question: str, top_k: int, mode: str) -> str:
    payload = {
        "query": question,
        "history": [],
        "retrieval_mode": mode,
        "top_k": top_k,
        "filters": None,
        "acl": [],
        "include_sources": False,
    }
    # Increased timeout for generation (includes retrieval + LLM)
    with httpx.Client(timeout=300.0) as c:
        r = c.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
        r.raise_for_status()
        j = r.json()
    return str(j.get("answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", required=True, help="One or more metadata.jsonl paths")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--index", action="store_true", help="Index unique contexts before evaluation")
    ap.add_argument("--index_batch_docs", type=int, default=50)
    ap.add_argument("--refresh", action="store_true", help="Refresh index after each upsert")
    ap.add_argument("--max_qas", type=int, default=0, help="0 = all")
    ap.add_argument("--gen", action="store_true", help="Also run generation and Number Match")
    ap.add_argument("--gen_max", type=int, default=200, help="Max QA to run generation on (cost control)")
    ap.add_argument("--out", required=True, help="Output JSON file path")
    args = ap.parse_args()

    _wait_ready(f"{RETRIEVAL_BASE_URL}/v1/readyz")
    _wait_ready(f"{GATE_BASE_URL}/v1/readyz")

    rows = list(iter_t2_rows([Path(p) for p in args.jsonl]))
    if args.max_qas and args.max_qas > 0:
        rows = rows[: args.max_qas]

    if args.index:
        indexed = _index_contexts(rows=rows, mode=args.mode, batch_docs=args.index_batch_docs, refresh=args.refresh)
    else:
        indexed = 0

    ranks: list[int | None] = []
    gen_nm_correct = 0
    gen_n = 0
    errors = 0

    for r in tqdm(rows, desc="eval_qas"):
        if not r.question or not r.context_id:
            continue
        try:
            rank = _retrieval_rank(query=r.question, gold_context_id=r.context_id, k=args.k, mode=args.mode)
            ranks.append(rank)
        except Exception:
            errors += 1
            ranks.append(None)
            continue

        if args.gen and gen_n < args.gen_max:
            try:
                ans = _generate_answer(question=r.question, top_k=args.k, mode=args.mode)
                if number_match(ans, r.program_answer, eps=1e-2):
                    gen_nm_correct += 1
                gen_n += 1
            except Exception:
                errors += 1

    ret = retrieval_stats(ranks, k=args.k)
    out = {
        "input_files": args.jsonl,
        "k": args.k,
        "mode": args.mode,
        "qas_evaluated": ret.count,
        "indexed_contexts": indexed,
        "retrieval": {"mrr_at_k": ret.mrr_at_k, "recall_at_k": ret.recall_at_k},
        "generation": {"enabled": bool(args.gen), "n": gen_n, "number_match": (gen_nm_correct / gen_n) if gen_n else None},
        "errors": errors,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

















