from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from bench.t2_hf_io import iter_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to jsonl")
    ap.add_argument("--limit", type=int, default=2000, help="Rows to sample (0 = all)")
    args = ap.parse_args()

    required = {"id", "context_id", "split", "question", "program_answer", "context"}
    cnt = 0
    missing = Counter()
    splits = Counter()
    ctx_ids = set()
    bad_json = 0

    p = Path(args.file)
    for row in iter_jsonl(p):
        cnt += 1
        if args.limit and args.limit > 0 and cnt > args.limit:
            break
        for k in required:
            if k not in row or row.get(k) in (None, ""):
                missing[k] += 1
        splits[str(row.get("split", ""))] += 1
        cid = row.get("context_id")
        if cid:
            ctx_ids.add(str(cid))

    print(
        {
            "file": str(p),
            "sampled_rows": cnt,
            "unique_context_ids": len(ctx_ids),
            "splits": dict(splits),
            "missing_required_fields": dict(missing),
        }
    )


if __name__ == "__main__":
    main()

















