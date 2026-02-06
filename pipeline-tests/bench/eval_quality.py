from __future__ import annotations

import argparse
import json
import os
import re
import time

import httpx
from tenacity import retry, stop_after_delay, wait_fixed

from bench.t2_xlsx import read_xlsx_examples


GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_number(s: str) -> str | None:
    """
    Best-effort numeric extraction for financial QA.
    Returns a normalized number string (e.g., '-7922', '13928', '3.14').
    """
    if not s:
        return None
    # remove common formatting
    t = s.replace(",", "")
    # pick first number-like token
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    if not m:
        return None
    return m.group(0)


def score_answer(pred: str, gold: str) -> dict:
    pred_n = _extract_number(pred)
    gold_n = _extract_number(gold)
    if gold_n is not None:
        return {
            "match_type": "numeric" if pred_n is not None else "missing_numeric",
            "correct": pred_n == gold_n if pred_n is not None else False,
            "pred_n": pred_n,
            "gold_n": gold_n,
        }
    # fallback string match
    return {"match_type": "text", "correct": _norm_text(pred) == _norm_text(gold)}

def score_retrieval_against_oracle(*, retrieval_json: dict, oracle_context: str) -> dict:
    """
    If the dataset doesn't include a gold answer, we can still score retrieval by comparing
    returned hit texts against the provided oracle context.
    """
    hits = retrieval_json.get("hits") or []
    oracle = oracle_context or ""
    oracle_n = _norm_text(oracle)
    best = 0.0
    matched = False
    for h in hits:
        t = str(h.get("text") or "")
        tn = _norm_text(t)
        if not tn:
            continue
        # Very cheap overlap proxy: fraction of hit tokens that appear in oracle
        toks = [x for x in re.split(r"[^a-z0-9]+", tn) if x]
        if not toks:
            continue
        inter = sum(1 for x in toks if x in oracle_n)
        score = inter / max(1, len(toks))
        if score > best:
            best = score
        if score >= 0.6:
            matched = True
    return {"match_type": "oracle_overlap", "matched": matched, "best_overlap": best, "hits": len(hits)}

def _upload_oracle_contexts(client: httpx.Client, examples) -> None:
    # Upload each oracle context as a separate document so retrieval can be evaluated end-to-end.
    for ex in examples:
        if not ex.oracle_context:
            continue
        doc_id = f"oracle-{ex.qid}"
        files = {"file": ("context.txt", ex.oracle_context.encode("utf-8", errors="replace"), "text/plain")}
        data = {
            "doc_id": doc_id,
            "title": f"OracleContext {ex.qid}",
            "source": "t2_oracle",
            "lang": "en",
            "tags": "t2,oracle",
            "refresh": "true",
        }
        r = client.post(f"{GATE_BASE_URL}/v1/documents/upload", files=files, data=data)
        if r.status_code != 200:
            raise RuntimeError(f"upload oracle context failed {doc_id}: {r.status_code} {r.text[:200]}")

@retry(wait=wait_fixed(1), stop=stop_after_delay(180))
def _wait_gate_ready() -> None:
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{GATE_BASE_URL}/v1/readyz")
        if r.status_code != 200:
            raise RuntimeError(f"gate readyz status={r.status_code}")
        j = r.json()
        if not j.get("ready"):
            raise RuntimeError(f"gate not ready: {j}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to xlsx (mounted inside container)")
    ap.add_argument("--limit", type=int, default=50, help="<=0 means no limit (all rows)")
    ap.add_argument("--out", default="/work/bench/quality_results.jsonl")
    ap.add_argument("--use_doc_id_filter", action="store_true", help="If xlsx has doc_id, filter retrieval by it.")
    ap.add_argument(
        "--index_oracle_contexts",
        action="store_true",
        help="Upload xlsx 'context' into the pipeline before evaluation (oracle-mode retrieval/generation).",
    )
    args = ap.parse_args()

    lim = None if args.limit is None or args.limit <= 0 else int(args.limit)
    examples = read_xlsx_examples(args.xlsx, limit=lim)
    if not examples:
        raise SystemExit("No examples read from xlsx (schema mismatch?)")

    rows = []
    correct = 0
    total = 0
    started = time.time()

    with httpx.Client(timeout=120.0) as c:
        _wait_gate_ready()
        if args.index_oracle_contexts:
            _upload_oracle_contexts(c, examples)

        for ex in examples:
            filters = None
            if args.use_doc_id_filter and ex.doc_id:
                filters = {"doc_ids": [ex.doc_id]}
            elif args.index_oracle_contexts and ex.oracle_context:
                # If we uploaded oracle contexts, don't filter: test whether retrieval finds the right context among all indexed contexts.
                filters = None
            payload = {
                "query": ex.question,
                "history": [],
                "retrieval_mode": "hybrid",
                "top_k": 8,
                "filters": filters,
                "acl": [],
                "include_sources": True,
            }
            try:
                r = c.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
            except Exception as e:
                rows.append(
                    {
                        "qid": ex.qid,
                        "question": ex.question,
                        "gold": ex.answer,
                        "error": f"request_error:{type(e).__name__}",
                    }
                )
                continue
            if r.status_code != 200:
                rows.append(
                    {
                        "qid": ex.qid,
                        "question": ex.question,
                        "gold": ex.answer,
                        "status_code": r.status_code,
                        "error": r.text[:500],
                    }
                )
                continue
            j = r.json()
            pred = str(j.get("answer") or "")
            rec = {
                "qid": ex.qid,
                "question": ex.question,
                "gold": ex.answer,
                "doc_id": ex.doc_id,
                "oracle_context_present": bool(ex.oracle_context),
                "oracle_context_indexed": bool(args.index_oracle_contexts and ex.oracle_context),
                "pred": pred,
                "used_mode": j.get("used_mode"),
                "partial": bool(j.get("partial")),
                "degraded": j.get("degraded") or [],
                "sources": j.get("sources") or [],
                "retrieval": j.get("retrieval"),
            }
            if ex.answer:
                total += 1
                s = score_answer(pred, ex.answer)
                rec["score"] = s
                if s.get("correct"):
                    correct += 1
            elif ex.oracle_context and isinstance(j.get("retrieval"), dict):
                # Retrieval-only evaluation (oracle overlap)
                total += 1
                s = score_retrieval_against_oracle(retrieval_json=j["retrieval"], oracle_context=ex.oracle_context)
                rec["score"] = s
                if s.get("matched"):
                    correct += 1
            rows.append(rec)

    elapsed = time.time() - started
    summary = {
        "xlsx": args.xlsx,
        "limit": lim,
        "answered": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "seconds": elapsed,
        "gate_base_url": GATE_BASE_URL,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

















