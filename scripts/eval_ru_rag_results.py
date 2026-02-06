from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Row:
    i: int
    question: str | None
    gold: str | None
    pred: str | None
    expected_doc_id: str | None
    retrieved_doc_ids: list[str]
    status_code: int | None
    error: str | None
    judge: dict[str, Any] | None
    judge_error: str | None
    file_hit_1: bool | None
    file_hit_3: bool | None
    file_hit_5: bool | None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s


def _safe_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes"):
            return True
        if t in ("false", "0", "no"):
            return False
    return None


def _as_str_list(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            out.append(str(x))
        return out
    return [str(v)]


def _compute_hits(expected_doc_id: str | None, retrieved_doc_ids: list[str]) -> tuple[bool | None, bool | None, bool | None]:
    if not expected_doc_id:
        return None, None, None
    hit1 = bool(retrieved_doc_ids[:1] and retrieved_doc_ids[0] == expected_doc_id)
    hit3 = expected_doc_id in retrieved_doc_ids[:3]
    hit5 = expected_doc_id in retrieved_doc_ids[:5]
    return hit1, hit3, hit5


def _read_jsonl(path: Path) -> tuple[dict[str, Any] | None, list[Row]]:
    summary: dict[str, Any] | None = None
    rows: list[Row] = []

    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if first.strip():
            try:
                obj = json.loads(first)
                if isinstance(obj, dict) and "summary" in obj and isinstance(obj["summary"], dict):
                    summary = obj["summary"]
            except Exception:
                summary = None

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not isinstance(r, dict):
                continue

            idx = _safe_int(r.get("i"))
            if idx is None:
                # Not a per-example row
                continue

            retrieved = _as_str_list(r.get("retrieved_doc_ids"))
            expected = _safe_str(r.get("expected_doc_id"))
            hit1 = _safe_bool(r.get("file_hit@1"))
            hit3 = _safe_bool(r.get("file_hit@3"))
            hit5 = _safe_bool(r.get("file_hit@5"))
            if hit1 is None and hit3 is None and hit5 is None:
                hit1, hit3, hit5 = _compute_hits(expected, retrieved)

            rows.append(
                Row(
                    i=idx,
                    question=_safe_str(r.get("question")),
                    gold=_safe_str(r.get("gold")),
                    pred=_safe_str(r.get("pred")),
                    expected_doc_id=expected,
                    retrieved_doc_ids=retrieved,
                    status_code=_safe_int(r.get("status_code")),
                    error=_safe_str(r.get("error")),
                    judge=r.get("judge") if isinstance(r.get("judge"), dict) else None,
                    judge_error=_safe_str(r.get("judge_error")),
                    file_hit_1=hit1,
                    file_hit_3=hit3,
                    file_hit_5=hit5,
                )
            )

    return summary, rows


def _mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x*100:.2f}%"


def _score_0_5_to_pct(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        s = float(score)
    except Exception:
        return None
    if math.isnan(s) or math.isinf(s):
        return None
    return (s / 5.0) * 100.0


def _top_failures(rows: list[Row], n: int) -> list[dict[str, Any]]:
    # Prefer examples where: retrieval missed doc_id OR judge says incorrect OR request errored
    scored: list[tuple[int, Row]] = []
    for r in rows:
        s = 0
        if r.status_code and r.status_code != 200:
            s += 100
        if r.file_hit_1 is False:
            s += 10
        if r.judge and r.judge.get("is_correct") is False:
            s += 20
        if r.judge_error:
            s += 5
        scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: list[dict[str, Any]] = []
    for s, r in scored[: max(0, n)]:
        out.append(
            {
                "i": r.i,
                "severity": s,
                "status_code": r.status_code,
                "expected_doc_id": r.expected_doc_id,
                "retrieved_doc_ids_top5": r.retrieved_doc_ids[:5],
                "file_hit@1": r.file_hit_1,
                "judge_is_correct": (r.judge or {}).get("is_correct") if r.judge else None,
                "judge_score_0_5": (r.judge or {}).get("score_0_5") if r.judge else None,
                "judge_error": r.judge_error,
                "question": (r.question or "")[:300],
                "gold": (r.gold or "")[:200],
                "pred": (r.pred or "")[:300],
                "error": (r.error or "")[:300],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze ru_eval JSONL results (no execution, only metrics).")
    ap.add_argument("--in", dest="inp", required=True, help="Path to ru_eval JSONL (e.g. out/ru_rag_eval_maincompose_full.jsonl)")
    ap.add_argument("--top_failures", type=int, default=20, help="How many worst examples to print into report")
    ap.add_argument("--out_json", default="", help="Optional path to write full report as JSON")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    summary, rows = _read_jsonl(in_path)

    # De-dup by i (resume can append, but should not; handle anyway).
    by_i: dict[int, Row] = {}
    for r in rows:
        by_i[r.i] = r
    rows = [by_i[i] for i in sorted(by_i.keys())]

    total = len(rows)
    status_counts = Counter()
    for r in rows:
        if r.status_code is not None:
            status_counts[r.status_code] += 1
        else:
            status_counts[200] += 1

    ok_rows = [r for r in rows if (r.status_code is None or r.status_code == 200)]
    err_rows = [r for r in rows if (r.status_code is not None and r.status_code != 200)]

    def hit_rate(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in ok_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    hit1 = hit_rate("file_hit_1")
    hit3 = hit_rate("file_hit_3")
    hit5 = hit_rate("file_hit_5")

    judged = [r for r in ok_rows if r.judge is not None]
    judge_errors = [r for r in ok_rows if r.judge_error]
    judge_scores: list[float] = []
    judge_correct: list[bool] = []
    for r in judged:
        j = r.judge or {}
        if "score_0_5" in j:
            try:
                judge_scores.append(float(j["score_0_5"]))
            except Exception:
                pass
        if "is_correct" in j:
            ic = _safe_bool(j.get("is_correct"))
            if ic is not None:
                judge_correct.append(ic)

    report: dict[str, Any] = {
        "input": str(in_path),
        "rows_total": total,
        "rows_ok": len(ok_rows),
        "rows_error": len(err_rows),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
        "file_hit@1": hit1,
        "file_hit@3": hit3,
        "file_hit@5": hit5,
        "judge_enabled": bool(judged or judge_errors),
        "judge_rows": len(judged),
        "judge_error_rows": len(judge_errors),
        "judge_accuracy": (sum(1 for x in judge_correct if x) / len(judge_correct)) if judge_correct else None,
        "judge_avg_score_0_5": _mean(judge_scores),
        "judge_avg_score_pct": _score_0_5_to_pct(_mean(judge_scores)),
        "judge_median_score_0_5": statistics.median(judge_scores) if judge_scores else None,
        "top_failures": _top_failures(rows, n=int(args.top_failures)),
        "summary_in_file": summary,
    }

    # Human-readable console output
    print(f"Input: {in_path}")
    print(f"Rows: {total} (ok={len(ok_rows)}, error={len(err_rows)})")
    print(f"Status counts: {dict(sorted(status_counts.items()))}")
    print(f"File hit@1: {_fmt_pct(hit1)}  hit@3: {_fmt_pct(hit3)}  hit@5: {_fmt_pct(hit5)}")
    avg_pct = report.get("judge_avg_score_pct")
    avg_pct_str = "n/a" if avg_pct is None else f"{float(avg_pct):.2f}%"
    print(
        "Judge: "
        f"rows={len(judged)}, errors={len(judge_errors)}, "
        f"acc={_fmt_pct(report['judge_accuracy'])}, "
        f"avg_score={report['judge_avg_score_0_5']} ({avg_pct_str})"
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")


if __name__ == "__main__":
    main()















