#!/usr/bin/env python3
"""
BRIGHT retrieval-only evaluation.

Expects docs to be pre-indexed via scripts/index_bright.py.
Runs search against retrieval, computes nDCG@10, hit@k, recall@10.

Usage:
  python scripts/bright_eval.py --retrieval-url http://retrieval:8080 \\
    --bright-split biology --bright-limit 30 --project-id bright --top-k 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import httpx
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from tqdm import tqdm

RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")


def _dcg_binary(rels: list[int]) -> float:
    s = 0.0
    for i, rel in enumerate(rels):
        if rel <= 0:
            continue
        s += 1.0 / math.log2(i + 2)
    return s


def _ndcg_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    k = max(1, int(k))
    gold = set(str(x) for x in (gold_ids or []) if str(x))
    if not gold:
        return 0.0
    top = ranked_doc_ids[:k]
    rels = [1 if d in gold else 0 for d in top]
    dcg = _dcg_binary(rels)
    ideal_rels = [1] * min(k, len(gold))
    idcg = _dcg_binary(ideal_rels)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _hit_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> int:
    k = max(1, int(k))
    gold = set(str(x) for x in (gold_ids or []) if str(x))
    if not gold:
        return 0
    return 1 if any(d in gold for d in ranked_doc_ids[:k]) else 0


def _recall_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    k = max(1, int(k))
    gold = [str(x) for x in (gold_ids or []) if str(x)]
    if not gold:
        return 0.0
    gold_set = set(gold)
    got = sum(1 for d in set(ranked_doc_ids[:k]) if d in gold_set)
    return float(got / max(1, len(gold_set)))


def _dedupe_keep_order(xs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def _wait_retrieval_ready() -> None:
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{RETRIEVAL_BASE_URL}/v1/readyz")
        if r.status_code != 200:
            raise RuntimeError(f"retrieval readyz status={r.status_code}")
        j = r.json()
        if not j.get("ready"):
            raise RuntimeError(f"retrieval not ready: {j}")


def _search_retrieval(
    *,
    client: httpx.Client,
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    project_id: str,
    rerank: bool | None = None,
    max_chunks_per_doc: int | None = 1,
) -> list[str]:
    payload = {
        "query": query,
        "mode": mode,
        "top_k": int(top_k),
        "include_sources": False,
        "sources_level": "basic",
        "group_by_doc": True,
        "rerank": rerank,
        "max_chunks_per_doc": max_chunks_per_doc,
        "filters": {"project_id": project_id},
        "acl": [],
    }
    r = client.post(f"{RETRIEVAL_BASE_URL}/v1/search", json=payload)
    r.raise_for_status()
    j = r.json()
    hits = j.get("hits") or []
    doc_ids = [str((h or {}).get("doc_id") or "").strip() for h in hits]
    doc_ids = [d for d in doc_ids if d]
    return _dedupe_keep_order(doc_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="BRIGHT retrieval-only eval (pre-indexed docs)")
    ap.add_argument("--retrieval-url", default=RETRIEVAL_BASE_URL, help="Retrieval service URL")
    ap.add_argument("--bright", action="store_true", help="Ignored (compat)")
    ap.add_argument("--bright-split", default="biology", help="BRIGHT domain (biology, economics, ...)")
    ap.add_argument("--bright-limit", type=int, default=0, help="Limit examples (0=all)")
    ap.add_argument("--bright-docs-from-gold", type=int, default=0, help="Ignored (compat)")
    ap.add_argument("--project-id", default="bright", help="project_id filter")
    ap.add_argument("--concurrency", type=int, default=1, help="Ignored (compat)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="/out/bright_eval.jsonl")
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    args = ap.parse_args()

    global RETRIEVAL_BASE_URL
    RETRIEVAL_BASE_URL = (args.retrieval_url or RETRIEVAL_BASE_URL).rstrip("/")

    limit = None if args.bright_limit is None or args.bright_limit <= 0 else int(args.bright_limit)
    split = f"{args.bright_split}[:{limit}]" if limit else args.bright_split

    print(f"Loading BRIGHT examples/{split}...", file=sys.stderr)
    ds = load_dataset("xlangai/BRIGHT", "examples", split=split)
    examples = []
    for idx, row in enumerate(ds):
        q = str(row.get("query") or "").strip()
        if not q:
            continue
        gold_ids = [str(x) for x in (row.get("gold_ids") or []) if str(x)]
        excluded_ids = [str(x) for x in (row.get("excluded_ids") or []) if str(x)]
        examples.append({"i": idx, "domain": args.bright_split, "query": q, "gold_ids": gold_ids, "excluded_ids": excluded_ids})

    if not examples:
        print("No examples loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Waiting for retrieval at {RETRIEVAL_BASE_URL}...", file=sys.stderr)
    _wait_retrieval_ready()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {
        "started_at": time.time(),
        "retrieval_base_url": RETRIEVAL_BASE_URL,
        "domain": args.bright_split,
        "project_id": args.project_id,
        "top_k": args.top_k,
        "mode": args.mode,
        "count": len(examples),
    }

    ndcg_sum = 0.0
    hit1_sum = 0
    hit3_sum = 0
    hit10_sum = 0
    recall_sum = 0.0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

        with httpx.Client(timeout=args.timeout) as client:
            for ex in tqdm(examples, desc="Evaluating"):
                ranked = _search_retrieval(
                    client=client,
                    query=ex["query"],
                    mode=args.mode,
                    top_k=args.top_k,
                    project_id=args.project_id,
                    rerank=None,
                    max_chunks_per_doc=1,
                )

                if ex.get("excluded_ids"):
                    excluded = set(ex["excluded_ids"])
                    ranked = [d for d in ranked if d not in excluded]

                ndcg10 = _ndcg_at_k(ranked_doc_ids=ranked, gold_ids=ex["gold_ids"], k=10)
                hit1 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex["gold_ids"], k=1)
                hit3 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex["gold_ids"], k=3)
                hit10 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex["gold_ids"], k=10)
                recall10 = _recall_at_k(ranked_doc_ids=ranked, gold_ids=ex["gold_ids"], k=10)

                ndcg_sum += ndcg10
                hit1_sum += hit1
                hit3_sum += hit3
                hit10_sum += hit10
                recall_sum += recall10

                row = {
                    "i": ex["i"],
                    "domain": ex["domain"],
                    "query": ex["query"],
                    "gold_ids": ex["gold_ids"],
                    "excluded_ids": ex.get("excluded_ids", []),
                    "retrieved_doc_ids": ranked[: args.top_k],
                    "metrics": {"ndcg@10": ndcg10, "hit@1": hit1, "hit@3": hit3, "hit@10": hit10, "recall@10": recall10},
                    "project_id": args.project_id,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

        n = len(examples)
        agg = {
            "aggregate": {
                "count": n,
                "ndcg@10": ndcg_sum / n,
                "hit@1": hit1_sum / n,
                "hit@3": hit3_sum / n,
                "hit@10": hit10_sum / n,
                "recall@10": recall_sum / n,
            }
        }
        f.write(json.dumps(agg, ensure_ascii=False) + "\n")

    print(f"Done. Results: {args.out}", file=sys.stderr)
    print(f"  nDCG@10: {agg['aggregate']['ndcg@10']:.4f}", file=sys.stderr)
    print(f"  Recall@10: {agg['aggregate']['recall@10']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
