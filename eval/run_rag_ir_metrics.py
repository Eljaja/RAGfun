#!/usr/bin/env python3
"""
IR-метрики по qrels (золотые релевантности): nDCG@k, MAP, MRR.

Используется для «серьёзной» оценки retrieval на бенчмарках (BEIR, MS MARCO):
результаты run_rag_baseline.py + файл qrels (query_id -> doc_id -> relevance).

Формат qrels (TREC): строки "qid doc_id relevance" (tab или пробел).
  qid и doc_id должны совпадать с теми, что в results (qid в каждой записи,
  doc_id в retrieved_chunks[].doc_id).

Пример:
  python eval/run_rag_baseline.py --dataset data/beir/queries.jsonl --format beir --output results.json
  python eval/run_rag_ir_metrics.py --results results.json --qrels data/beir/qrels_test.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _norm_id(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    """qid -> { doc_id: relevance } (relevance >= 0, обычно 0/1 или 1–3)."""
    qrels: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) < 3:
                continue
            qid, doc_id, rel = _norm_id(parts[0]), _norm_id(parts[1]), float(parts[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel
    return qrels


def get_retrieved_doc_ids(result: dict[str, Any]) -> list[str]:
    """Порядок doc_id в выдаче (как в retrieved_chunks)."""
    chunks = result.get("retrieved_chunks") or []
    return [_norm_id(c.get("doc_id")) for c in chunks]


def mrr_from_qrels(
    results: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
) -> tuple[float, int]:
    """MRR по qrels. Возвращает (mrr, num_queries_with_qrels)."""
    total = 0.0
    count = 0
    for r in results:
        qid = _norm_id(r.get("qid") or r.get("query_id"))
        if not qid or qid not in qrels:
            continue
        rel_docs = qrels[qid]
        doc_ids = get_retrieved_doc_ids(r)
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id in rel_docs and rel_docs[doc_id] > 0:
                total += 1.0 / rank
                break
        count += 1
    return (total / count if count else 0.0, count)


def dcg_at_k(relevances: list[float], k: int) -> float:
    """DCG@k: sum (2^rel - 1) / log2(rank+1)."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / __import__("math").log2(i + 1)
    return total


def ndcg_at_k(
    results: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    k: int,
) -> tuple[float, int]:
    """nDCG@k: DCG@k / IDCG@k в среднем по запросам с qrels."""
    ndcg_sum = 0.0
    count = 0
    for r in results:
        qid = _norm_id(r.get("qid") or r.get("query_id"))
        if not qid or qid not in qrels:
            continue
        rel_docs = qrels[qid]
        doc_ids = get_retrieved_doc_ids(r)
        # Relevances в порядке выдачи (до k)
        rels = [rel_docs.get(did, 0.0) for did in doc_ids[:k]]
        ideal = sorted([r for r in rel_docs.values() if r > 0], reverse=True)[:k]
        dcg = dcg_at_k(rels, k)
        idcg = dcg_at_k(ideal, k)
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
        count += 1
    return (ndcg_sum / count if count else 0.0, count)


def map_from_qrels(
    results: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
) -> tuple[float, int]:
    """MAP: среднее по запросам от (sum prec@k * rel(k)) / num_relevant."""
    map_sum = 0.0
    count = 0
    for r in results:
        qid = _norm_id(r.get("qid") or r.get("query_id"))
        if not qid or qid not in qrels:
            continue
        rel_docs = qrels[qid]
        num_relevant = sum(1 for rel in rel_docs.values() if rel > 0)
        if num_relevant == 0:
            continue
        doc_ids = get_retrieved_doc_ids(r)
        prec_sum = 0.0
        seen_relevant = 0
        for rank, doc_id in enumerate(doc_ids, start=1):
            rel = rel_docs.get(doc_id, 0.0)
            if rel > 0:
                seen_relevant += 1
                prec_sum += (seen_relevant / rank) * rel
        # MAP for query: (sum P@k * rel(k)) / num_relevant (binary) or average precision
        ap = prec_sum / num_relevant if num_relevant else 0.0
        map_sum += ap
        count += 1
    return (map_sum / count if count else 0.0, count)


def recall_at_k_from_qrels(
    results: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    k: int,
) -> tuple[float, int]:
    """Recall@k: доля запросов, у которых хотя бы один релевантный в топ-k."""
    hits = 0
    count = 0
    for r in results:
        qid = _norm_id(r.get("qid") or r.get("query_id"))
        if not qid or qid not in qrels:
            continue
        rel_docs = {did for did, rel in qrels[qid].items() if rel > 0}
        if not rel_docs:
            continue
        doc_ids = get_retrieved_doc_ids(r)[:k]
        if any(did in rel_docs for did in doc_ids):
            hits += 1
        count += 1
    return (hits / count if count else 0.0, count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="nDCG@k, MAP, MRR по results.json и qrels (TREC-формат).",
    )
    parser.add_argument("--results", required=True, help="Путь к results.json")
    parser.add_argument("--qrels", required=True, help="Путь к qrels (qid doc_id relevance)")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="k для nDCG@k и Recall@k (default: 5 10)",
    )
    parser.add_argument("--output", default=None, help="Сохранить метрики в JSON")
    args = parser.parse_args()

    res_path = Path(args.results)
    qrels_path = Path(args.qrels)
    if not res_path.exists():
        print(f"Файл не найден: {res_path}", file=sys.stderr)
        return 1
    if not qrels_path.exists():
        print(f"Файл не найден: {qrels_path}", file=sys.stderr)
        return 1

    with open(res_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        print("Ожидается JSON-массив результатов.", file=sys.stderr)
        return 1

    qrels = load_qrels(qrels_path)

    mrr, n_qrels = mrr_from_qrels(results, qrels)
    if n_qrels == 0:
        print("Нет запросов с qid из qrels. Проверьте совпадение qid в results и в qrels.", file=sys.stderr)
        out = {
            "error": "no_matching_qrels",
            "num_results": len(results),
            "qrels_queries": len(qrels),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    ndcg_at: dict[int, float] = {}
    for k in args.k:
        ndcg_at[k], _ = ndcg_at_k(results, qrels, k)

    recall_at: dict[int, float] = {}
    for k in args.k:
        recall_at[k], _ = recall_at_k_from_qrels(results, qrels, k)

    map_score, _ = map_from_qrels(results, qrels)

    metrics = {
        "num_queries_with_qrels": n_qrels,
        "num_queries_total": len(results),
        "mrr": mrr,
        "map": map_score,
        "ndcg_at_k": ndcg_at,
        "recall_at_k": recall_at,
        "k_values": args.k,
    }

    print("IR-метрики (по qrels)", file=sys.stderr)
    print(f"  Запросов с qrels: {n_qrels} / {len(results)}", file=sys.stderr)
    print(f"  MRR: {mrr:.4f}", file=sys.stderr)
    print(f"  MAP: {map_score:.4f}", file=sys.stderr)
    for k in args.k:
        print(f"  nDCG@{k}: {ndcg_at[k]:.4f}  Recall@{k}: {recall_at[k]:.4f}", file=sys.stderr)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Сохранено: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
