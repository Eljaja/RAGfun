#!/usr/bin/env python3
"""
RAGfun v0.2 baseline evaluation script.

Runs the current RAGfun retrieval + generator pipeline on an external dataset
(MS MARCO / BEIR / Natural Questions–style or generic JSONL), records
retrieved chunks and final answers, and writes results to JSON.

Usage:
  # Gate must be running (e.g. docker-compose up gate retrieval ...).
  python run_rag_baseline.py \\
    --gate-url http://localhost:8090 \\
    --dataset path/to/queries.jsonl \\
    --format jsonl \\
    --output results.json \\
    [--limit 100] [--example-size 5]

  # Small example (5 queries) to verify correctness:
  python run_rag_baseline.py --gate-url http://localhost:8090 \\
    --dataset path/to/queries.jsonl --format jsonl \\
    --output results.json --limit 5 --example-size 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---- Dataset loaders (reusable for any external dataset) ----


def _normalize_notes(
    *,
    answer: str,
    unanswerable: bool,
    partial: bool,
    chunk_count: int,
) -> str:
    """
    Classify result into: relevant | hallucinated | partial | correct_refusal | incorrect_refusal.
    """
    ans = (answer or "").strip().lower()
    refusal_phrases = (
        "i don't know",
        "i do not know",
        "не знаю",
        "not enough information",
        "insufficient information",
        "no relevant",
        "cannot answer",
        "cannot be determined",
        "context does not",
        "provided context",
    )
    looks_refusal = any(p in ans for p in refusal_phrases) or ans in ("i don't know", "не знаю.")

    if unanswerable:
        return "correct_refusal" if looks_refusal else "incorrect_refusal"
    if partial or chunk_count == 0:
        return "partial"
    if looks_refusal:
        return "partial"  # system refused despite having chunks
    return "relevant"  # placeholder; "hallucinated" would need NLI/gold check


def load_queries(
    path: str | Path,
    format: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load a list of queries from a file.

    Supported formats:
      - jsonl: one JSON object per line; each must have "query" (str).
               Optional: "qid", "unanswerable" (bool).
      - ms_marco: TSV with header, columns (qid, query) or (query_id, query).
                  Query text may be in 2nd column or a "query" column.
      - beir: JSONL with "query" and optionally "query_id", "_id", "unanswerable".
      - nq: JSONL with "question" (or "query") and optionally "unanswerable".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    rows: list[dict[str, Any]] = []

    if format == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("query") or obj.get("question") or ""
                    if not q:
                        continue
                    row = {
                        "query": q,
                        "qid": obj.get("qid") or obj.get("query_id") or obj.get("_id"),
                        "unanswerable": bool(obj.get("unanswerable", False)),
                    }
                    if "expected_doc_ids" in obj:
                        row["expected_doc_ids"] = list(obj["expected_doc_ids"]) if isinstance(obj["expected_doc_ids"], (list, tuple)) else [obj["expected_doc_ids"]]
                    if "expected_text" in obj:
                        row["expected_text"] = str(obj["expected_text"]).strip()
                    rows.append(row)
                except json.JSONDecodeError:
                    continue

    elif format == "ms_marco":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip() for ln in f if ln.rstrip()]
        if not lines:
            return []
        parts0 = lines[0].split("\t")
        header = [c.strip().lower() for c in parts0]
        idx_qid = 0
        idx_query = 1
        start = 0
        # Official MS MARCO files have no header: first line is "qid\tquery"
        if len(header) >= 2 and header[0].isdigit() and "query" not in header:
            start = 0  # headerless: use all lines, col0=qid, col1=query
        else:
            for i, h in enumerate(header):
                if h in ("qid", "query_id", "queryid"):
                    idx_qid = i
                if h == "query":
                    idx_query = i
            if idx_query is None and len(header) >= 2:
                idx_query = 1
            start = 1
        for line in lines[start:]:
            parts = line.split("\t")
            if len(parts) > idx_query:
                q = parts[idx_query].strip()
                if q:
                    rows.append({
                        "query": q,
                        "qid": parts[idx_qid] if idx_qid < len(parts) else None,
                        "unanswerable": False,
                    })
            if limit and len(rows) >= limit:
                break
        if limit is not None and limit > 0:
            rows = rows[:limit]

    elif format == "beir":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("query") or obj.get("question") or ""
                    if not q:
                        continue
                    rows.append({
                        "query": q,
                        "qid": obj.get("query_id") or obj.get("_id") or obj.get("id") or obj.get("qid"),
                        "unanswerable": bool(obj.get("unanswerable", False)),
                    })
                except json.JSONDecodeError:
                    continue

    elif format == "nq":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question") or obj.get("query") or ""
                    if not q:
                        continue
                    rows.append({
                        "query": q,
                        "qid": obj.get("example_id") or obj.get("id"),
                        "unanswerable": bool(obj.get("unanswerable", False)),
                    })
                except json.JSONDecodeError:
                    continue

    else:
        raise ValueError(f"Unknown format: {format}. Use one of: jsonl, ms_marco, beir, nq")

    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


# ---- RAG evaluation (calls Gate; keeps RAGfun code unchanged) ----


def run_single(
    gate_url: str,
    query: str,
    top_k: int | None = None,
    include_sources: bool = True,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """
    Call Gate POST /v1/chat for one query. Returns parsed fields for the eval schema.
    """
    payload: dict[str, Any] = {"query": query, "include_sources": include_sources}
    if top_k is not None:
        payload["top_k"] = top_k

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(
            f"{gate_url.rstrip('/')}/v1/chat",
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    ok = data.get("ok", True)
    answer = data.get("answer") or ""
    context = data.get("context") or []
    partial = bool(data.get("partial", False))

    retrieved_chunks = [
        {"doc_id": str(c.get("doc_id") or ""), "text": (c.get("text") or "").strip()}
        for c in context
    ]

    return {
        "ok": ok,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "partial": partial,
        "raw_context_count": len(context),
    }


def run_eval(
    gate_url: str,
    queries: list[dict[str, Any]],
    top_k: int | None = None,
    include_sources: bool = True,
    timeout_s: float = 120.0,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run RAG pipeline for each query. Returns (results, errors).
    Each result has: query, retrieved_chunks, final_answer, notes, (qid, unanswerable if present).
    """
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    it = enumerate(queries)
    if progress and tqdm is not None:
        it = enumerate(tqdm(queries, desc="eval", unit="q"))

    for i, row in it:
        query = (row.get("query") or "").strip()
        qid = row.get("qid")
        unanswerable = bool(row.get("unanswerable", False))

        if not query:
            errors.append({"index": i, "qid": qid, "error": "empty_query"})
            continue

        try:
            out = run_single(
                gate_url=gate_url,
                query=query,
                top_k=top_k,
                include_sources=include_sources,
                timeout_s=timeout_s,
            )
        except httpx.HTTPStatusError as e:
            errors.append({
                "index": i,
                "qid": qid,
                "query": query[:200],
                "error": f"http_{e.response.status_code}",
                "detail": (e.response.text or "")[:500],
            })
            continue
        except httpx.TimeoutException:
            errors.append({"index": i, "qid": qid, "query": query[:200], "error": "timeout"})
            continue
        except Exception as e:
            errors.append({
                "index": i,
                "qid": qid,
                "query": query[:200],
                "error": type(e).__name__,
                "detail": str(e),
            })
            continue

        notes = _normalize_notes(
            answer=out["answer"],
            unanswerable=unanswerable,
            partial=out["partial"],
            chunk_count=len(out["retrieved_chunks"]),
        )

        rec: dict[str, Any] = {
            "query": query,
            "retrieved_chunks": out["retrieved_chunks"],
            "final_answer": out["answer"],
            "notes": notes,
        }
        if qid is not None:
            rec["qid"] = qid
        if unanswerable:
            rec["unanswerable"] = True
        if row.get("expected_doc_ids"):
            rec["expected_doc_ids"] = row["expected_doc_ids"]
        if row.get("expected_text"):
            rec["expected_text"] = row["expected_text"]
        results.append(rec)

    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAGfun v0.2 baseline evaluation on an external dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gate-url",
        default="http://localhost:8090",
        help="Base URL of the RAG Gate (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the query dataset (file)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "ms_marco", "beir", "nq"],
        default="jsonl",
        help="Dataset format (default: jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of queries to evaluate (default: all)",
    )
    parser.add_argument(
        "--example-size",
        type=int,
        default=5,
        help="Number of results to duplicate into 'example' key for quick checks (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k per request (default: use Gate config)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Set include_sources=False when calling /v1/chat",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    args = parser.parse_args()

    # Load queries
    try:
        queries = load_queries(args.dataset, args.format, limit=args.limit)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    if not queries:
        print("No queries loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {len(queries)} queries from {args.dataset} (format={args.format})", file=sys.stderr)
    print(f"Gate URL: {args.gate_url}", file=sys.stderr)

    # Optional readiness check
    try:
        r = httpx.get(f"{args.gate_url.rstrip('/')}/v1/healthz", timeout=5.0)
        if r.status_code != 200:
            print(f"Warning: gate health check returned {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: could not reach gate: {e}", file=sys.stderr)

    # Run evaluation
    results, errors = run_eval(
        gate_url=args.gate_url,
        queries=queries,
        top_k=args.top_k,
        include_sources=not args.no_sources,
        timeout_s=args.timeout,
        progress=not args.no_progress,
    )

    # Build output
    n = min(max(0, args.example_size), len(results))
    example = results[:n] if n else []

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Primary output: requested format — list of {query, retrieved_chunks, final_answer, notes}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Small example (first N queries) as part of the output for quick correctness checks
    example_path = out_path.parent / (out_path.stem + ".example.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False, indent=2)

    # Optional report with metadata and errors
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report = {
        "metadata": {
            "gate_url": args.gate_url,
            "dataset": str(args.dataset),
            "format": args.format,
            "num_queries": len(queries),
            "num_results": len(results),
            "num_errors": len(errors),
        },
        "example_path": str(example_path),
        "errors": errors,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} results to {out_path}", file=sys.stderr)
    print(f"Wrote example ({len(example)} queries) to {example_path}", file=sys.stderr)
    print(f"Wrote report (metadata + errors) to {report_path}", file=sys.stderr)
    if errors:
        print(f"Encountered {len(errors)} errors (see report).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
