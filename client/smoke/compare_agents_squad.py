from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
DEFAULT_DATASET_PATH = Path("/tmp/squad-dev-v1.1.json")

DEFAULT_OLD_RETRIEVAL = "http://127.0.0.1:8085"
DEFAULT_OLD_AGENT = "http://127.0.0.1:8093"
DEFAULT_NEW_RETRIEVAL = "http://127.0.0.1:18085"
DEFAULT_NEW_AGENT = "http://127.0.0.1:18093"

_CITATION_RE = re.compile(r"\[\d+\]")


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _http_text(method: str, url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _ensure_dataset(dataset_url: str, dataset_path: Path) -> dict[str, Any]:
    if dataset_path.exists():
        return json.loads(dataset_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(dataset_url, timeout=120) as resp:
        raw = resp.read()
    dataset_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _CITATION_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _exact_match(prediction: str, golds: list[str]) -> bool:
    npred = _normalize_text(prediction)
    return any(npred == _normalize_text(gold) for gold in golds)


def _contains_gold(prediction: str, golds: list[str]) -> bool:
    npred = _normalize_text(prediction)
    return any((_normalize_text(gold) in npred) or (npred in _normalize_text(gold)) for gold in golds if gold.strip())


def _token_f1(prediction: str, gold: str) -> float:
    ptoks = _normalize_text(prediction).split()
    gtoks = _normalize_text(gold).split()
    if not ptoks and not gtoks:
        return 1.0
    if not ptoks or not gtoks:
        return 0.0
    common = 0
    remaining = list(gtoks)
    for tok in ptoks:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(ptoks)
    recall = common / len(gtoks)
    return 2 * precision * recall / (precision + recall)


def _best_f1(prediction: str, golds: list[str]) -> float:
    return max(_token_f1(prediction, gold) for gold in golds)


def _parse_prometheus_counters(text: str) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not _:
            continue
        metric_name = name.split("{", 1)[0]
        try:
            counters[metric_name] = counters.get(metric_name, 0.0) + float(value)
        except ValueError:
            continue
    return counters


def _agent_metrics(agent_base: str) -> dict[str, float]:
    raw = _http_text("GET", f"{agent_base}/v1/metrics", timeout=15.0)
    parsed = _parse_prometheus_counters(raw)
    return {
        "llm_calls": parsed.get("agent_llm_calls_total", 0.0),
        "retrieval_calls": parsed.get("agent_retrieval_calls_total", parsed.get("agent_gate_calls_total", 0.0)),
        "retries": parsed.get("agent_retry_total", 0.0),
        "errors": parsed.get("agent_requests_total", 0.0),  # informative only; delta not used directly
    }


def _sample_examples(dataset: dict[str, Any], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for article in dataset["data"]:
        title = article["title"]
        for para_idx, para in enumerate(article["paragraphs"]):
            context = para["context"]
            for qa in para["qas"]:
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                if not answers:
                    continue
                rows.append(
                    {
                        "qid": qa["id"],
                        "title": title,
                        "para_idx": para_idx,
                        "question": qa["question"],
                        "answers": answers,
                        "context": context,
                    }
                )
    random.Random(seed).shuffle(rows)
    picked: list[dict[str, Any]] = []
    seen_contexts: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["title"], row["para_idx"])
        if key in seen_contexts:
            continue
        seen_contexts.add(key)
        picked.append(row)
        if len(picked) >= sample_size:
            break
    return picked


def _build_docs(examples: list[dict[str, Any]], prefix: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    docs: OrderedDict[tuple[str, int], dict[str, str]] = OrderedDict()
    out_examples: list[dict[str, Any]] = []
    for row in examples:
        copied = dict(row)
        key = (copied["title"], copied["para_idx"])
        if key not in docs:
            docs[key] = {
                "doc_id": f"{prefix}-{len(docs) + 1}",
                "title": f"{copied['title']} #{copied['para_idx']}",
                "text": copied["context"],
            }
        copied["doc_id"] = docs[key]["doc_id"]
        out_examples.append(copied)
    return list(docs.values()), out_examples


def _index_docs(retrieval_base: str, docs: list[dict[str, str]], timeout_s: float) -> None:
    for doc in docs:
        payload = {
            "mode": "document",
            "document": {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": "squad-eval",
            },
            "text": doc["text"],
            "refresh": True,
        }
        response = _http_json("POST", f"{retrieval_base}/v1/index/upsert", payload, timeout=timeout_s)
        if not response.get("ok"):
            raise RuntimeError(f"index failed for {doc['doc_id']}: {response}")


def _evaluate_agent(agent_base: str, examples: list[dict[str, Any]], all_doc_ids: list[str], timeout_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in examples:
        before = _agent_metrics(agent_base)
        started = time.time()
        try:
            response = _http_json(
                "POST",
                f"{agent_base}/v1/agent",
                {
                    "query": row["question"],
                    "include_sources": True,
                    "mode": "conservative",
                    "filters": {"doc_ids": all_doc_ids},
                },
                timeout=timeout_s,
            )
            latency_s = time.time() - started
            after = _agent_metrics(agent_base)
            answer = response.get("answer", "")
            context_text = "\n".join((chunk.get("text") or "") for chunk in (response.get("context") or []))
            rows.append(
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    "gold_answers": row["answers"],
                    "answer": answer,
                    "latency_s": latency_s,
                    "exact_match": _exact_match(answer, row["answers"]),
                    "contains_gold": _contains_gold(answer, row["answers"]),
                    "f1": _best_f1(answer, row["answers"]),
                    "citations": bool(_CITATION_RE.search(answer)),
                    "gold_in_context": _contains_gold(context_text, row["answers"]),
                    "partial": bool(response.get("partial")),
                    "degraded": list(response.get("degraded") or []),
                    "sources": len(response.get("sources") or []),
                    "llm_calls": after["llm_calls"] - before["llm_calls"],
                    "retrieval_calls": after["retrieval_calls"] - before["retrieval_calls"],
                    "retry_count": after["retries"] - before["retries"],
                }
            )
        except Exception as exc:
            after = _agent_metrics(agent_base)
            rows.append(
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    "gold_answers": row["answers"],
                    "answer": "",
                    "latency_s": None,
                    "exact_match": False,
                    "contains_gold": False,
                    "f1": 0.0,
                    "citations": False,
                    "gold_in_context": False,
                    "partial": True,
                    "degraded": [f"error:{type(exc).__name__}"],
                    "sources": 0,
                    "llm_calls": after["llm_calls"] - before["llm_calls"],
                    "retrieval_calls": after["retrieval_calls"] - before["retrieval_calls"],
                    "retry_count": after["retries"] - before["retries"],
                    "error": str(exc),
                }
            )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None]
    llm_calls = [row["llm_calls"] for row in rows]
    retrieval_calls = [row["retrieval_calls"] for row in rows]
    retries = [row["retry_count"] for row in rows]
    return {
        "count": count,
        "exact_match_rate": sum(1 for row in rows if row["exact_match"]) / count,
        "contains_gold_rate": sum(1 for row in rows if row["contains_gold"]) / count,
        "gold_in_context_rate": sum(1 for row in rows if row["gold_in_context"]) / count,
        "mean_f1": sum(row["f1"] for row in rows) / count,
        "citation_rate": sum(1 for row in rows if row["citations"]) / count,
        "partial_rate": sum(1 for row in rows if row["partial"]) / count,
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "mean_llm_calls": sum(llm_calls) / count,
        "mean_retrieval_calls": sum(retrieval_calls) / count,
        "retry_rate": sum(1 for value in retries if value > 0) / count,
        "error_count": sum(1 for row in rows if row.get("error")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old and new agent-search stacks on a sampled SQuAD v1.1 benchmark.")
    parser.add_argument("--sample-size", type=int, default=40, help="Number of unique-context QA examples to benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL, help="SQuAD-like dataset URL.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="Cached dataset path.")
    parser.add_argument("--old-retrieval", default=DEFAULT_OLD_RETRIEVAL, help="Old retrieval base URL.")
    parser.add_argument("--old-agent", default=DEFAULT_OLD_AGENT, help="Old agent base URL.")
    parser.add_argument("--new-retrieval", default=DEFAULT_NEW_RETRIEVAL, help="New retrieval base URL.")
    parser.add_argument("--new-agent", default=DEFAULT_NEW_AGENT, help="New agent base URL.")
    parser.add_argument("--timeout-s", type=float, default=25.0, help="Per-request timeout for agent and indexing requests.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = _ensure_dataset(args.dataset_url, Path(args.dataset_path))
    sampled = _sample_examples(dataset, args.sample_size, args.seed)

    old_docs, old_examples = _build_docs(sampled, "squad-old")
    new_docs, new_examples = _build_docs(sampled, "squad-new")

    print(f"Downloaded dataset and sampled {len(sampled)} unique contexts.", file=sys.stderr)
    print("Indexing docs into old retrieval...", file=sys.stderr)
    _index_docs(args.old_retrieval, old_docs, args.timeout_s)
    print("Indexing docs into new retrieval...", file=sys.stderr)
    _index_docs(args.new_retrieval, new_docs, args.timeout_s)

    old_doc_ids = [doc["doc_id"] for doc in old_docs]
    new_doc_ids = [doc["doc_id"] for doc in new_docs]

    print("Evaluating old agent...", file=sys.stderr)
    old_rows = _evaluate_agent(args.old_agent, old_examples, old_doc_ids, args.timeout_s)
    print("Evaluating new agent...", file=sys.stderr)
    new_rows = _evaluate_agent(args.new_agent, new_examples, new_doc_ids, args.timeout_s)

    result = {
        "dataset": {
            "name": "SQuAD v1.1 dev",
            "url": args.dataset_url,
            "sample_size": len(sampled),
            "seed": args.seed,
        },
        "old": _summarize(old_rows),
        "new": _summarize(new_rows),
        "old_rows": old_rows,
        "new_rows": new_rows,
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
