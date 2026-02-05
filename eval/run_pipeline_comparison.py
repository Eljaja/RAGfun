#!/usr/bin/env python3
"""
Тест сравнения: неполный пайплайн (прямая индексация в retrieval) vs полный (Gate → doc-processor → retrieval).

Последовательность:
  1) Индексация напрямую (index_beir_corpus) → baseline → метрики → metrics_direct.json
  2) Очистка retrieval по doc_id из корпуса
  3) Индексация через Gate (index_beir_via_gate, wait-indexed) → baseline → метрики → metrics_full.json
  4) Сравнение и вывод, где разница заметнее.

На коротких документах (SciFact) разница обычно в шуме. Чтобы увидеть отличия, используйте
датасет с более длинными документами, например FiQA:

  python eval/prepare_beir_data.py --dataset fiqa --out data/beir/fiqa
  python eval/corpus_stats.py data/beir/fiqa/corpus.jsonl --chars   # проверить длину
  python eval/run_pipeline_comparison.py --data-dir data/beir/fiqa --limit-corpus 500 --limit-queries 100

Требует: Gate, retrieval, для полного пайплайна — storage, RabbitMQ, ingestion-worker, doc-processor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> int:
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнение полный vs неполный пайплайн на BEIR; выбор данных по длине документов.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/beir/fiqa",
        help="Папка с corpus.jsonl, queries.jsonl, qrels_test.tsv (default: data/beir/fiqa)",
    )
    parser.add_argument(
        "--dataset",
        default="fiqa",
        help="Имя BEIR датасета для --prepare (default: fiqa)",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Сначала скачать и подготовить данные (prepare_beir_data.py)",
    )
    parser.add_argument(
        "--limit-corpus",
        type=int,
        default=None,
        help="Максимум документов (для быстрого теста; без лимита — полный корпус)",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Максимум запросов для baseline",
    )
    parser.add_argument(
        "--retrieval-url",
        default="http://localhost:8080",
        help="URL retrieval (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--gate-url",
        default="http://localhost:8090",
        help="URL Gate (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="k для nDCG@k (default: 5 10)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    if args.prepare:
        print("Подготовка данных BEIR...", file=sys.stderr)
        code = run(
            [
                sys.executable,
                str(root / "eval" / "prepare_beir_data.py"),
                "--dataset",
                args.dataset,
                "--out",
                str(data_dir),
            ],
            cwd=root,
        )
        if code != 0:
            return code

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels_test.tsv"
    for p in (corpus_path, queries_path, qrels_path):
        if not p.exists():
            print(f"Не найден: {p}. Запустите с --prepare или укажите папку с данными.", file=sys.stderr)
            return 1

    results_direct = data_dir / "results_direct.json"
    metrics_direct_path = data_dir / "metrics_direct.json"
    results_full = data_dir / "results_full.json"
    metrics_full_path = data_dir / "metrics_full.json"

    # ----- 1) Direct pipeline -----
    print("\n=== 1) Неполный пайплайн (прямая индексация в retrieval) ===", file=sys.stderr)
    index_cmd = [
        sys.executable,
        str(root / "eval" / "index_beir_corpus.py"),
        "--corpus",
        str(corpus_path),
        "--retrieval-url",
        args.retrieval_url,
    ]
    if args.limit_corpus is not None:
        index_cmd += ["--limit", str(args.limit_corpus)]
    if run(index_cmd, cwd=root) != 0:
        return 1

    baseline_cmd = [
        sys.executable,
        str(root / "eval" / "run_rag_baseline.py"),
        "--gate-url",
        args.gate_url,
        "--dataset",
        str(queries_path),
        "--format",
        "beir",
        "--output",
        str(results_direct),
    ]
    if args.limit_queries is not None:
        baseline_cmd += ["--limit", str(args.limit_queries)]
    if run(baseline_cmd, cwd=root) != 0:
        return 1

    metrics_cmd = [
        sys.executable,
        str(root / "eval" / "run_rag_ir_metrics.py"),
        "--results",
        str(results_direct),
        "--qrels",
        str(qrels_path),
        "--output",
        str(metrics_direct_path),
        "--k",
    ] + [str(k) for k in args.k]
    if run(metrics_cmd, cwd=root) != 0:
        return 1

    # ----- 2) Clear retrieval -----
    print("\n=== 2) Очистка индекса retrieval ===", file=sys.stderr)
    clear_cmd = [
        sys.executable,
        str(root / "eval" / "clear_retrieval_by_corpus.py"),
        "--corpus",
        str(corpus_path),
        "--retrieval-url",
        args.retrieval_url,
    ]
    if args.limit_corpus is not None:
        clear_cmd += ["--limit", str(args.limit_corpus)]
    if run(clear_cmd, cwd=root) != 0:
        return 1

    # ----- 3) Full pipeline -----
    print("\n=== 3) Полный пайплайн (Gate → doc-processor → retrieval) ===", file=sys.stderr)
    index_full_cmd = [
        sys.executable,
        str(root / "eval" / "index_beir_via_gate.py"),
        "--corpus",
        str(corpus_path),
        "--gate-url",
        args.gate_url,
        "--wait-indexed",
        "--no-fail-on-wait-timeout",
    ]
    if args.limit_corpus is not None:
        index_full_cmd += ["--limit", str(args.limit_corpus)]
    if run(index_full_cmd, cwd=root) != 0:
        return 1

    baseline_full_cmd = [
        sys.executable,
        str(root / "eval" / "run_rag_baseline.py"),
        "--gate-url",
        args.gate_url,
        "--dataset",
        str(queries_path),
        "--format",
        "beir",
        "--output",
        str(results_full),
    ]
    if args.limit_queries is not None:
        baseline_full_cmd += ["--limit", str(args.limit_queries)]
    if run(baseline_full_cmd, cwd=root) != 0:
        return 1

    metrics_full_cmd = [
        sys.executable,
        str(root / "eval" / "run_rag_ir_metrics.py"),
        "--results",
        str(results_full),
        "--qrels",
        str(qrels_path),
        "--output",
        str(metrics_full_path),
        "--k",
    ] + [str(k) for k in args.k]
    if run(metrics_full_cmd, cwd=root) != 0:
        return 1

    # ----- 4) Compare -----
    with open(metrics_direct_path, "r", encoding="utf-8") as f:
        m_direct = json.load(f)
    with open(metrics_full_path, "r", encoding="utf-8") as f:
        m_full = json.load(f)

    print("\n" + "=" * 60, file=sys.stderr)
    print("СРАВНЕНИЕ: неполный (прямая индексация) vs полный (doc-processor)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    def row(name: str, key: str, fmt: str = ".4f") -> None:
        a = m_direct.get(key)
        b = m_full.get(key)
        if a is None and b is None:
            return
        a = a if a is not None else 0
        b = b if b is not None else 0
        diff = b - a
        sign = "+" if diff >= 0 else ""
        print(f"  {name:12}  direct={a:{fmt}}  full={b:{fmt}}  diff={sign}{diff:{fmt}}", file=sys.stderr)

    row("MRR", "mrr")
    row("MAP", "map")
    for k in args.k:
        ndcg_d = (m_direct.get("ndcg_at_k") or {}).get(k)
        ndcg_f = (m_full.get("ndcg_at_k") or {}).get(k)
        rec_d = (m_direct.get("recall_at_k") or {}).get(k)
        rec_f = (m_full.get("recall_at_k") or {}).get(k)
        if ndcg_d is not None and ndcg_f is not None:
            print(f"  nDCG@{k:<2}       direct={ndcg_d:.4f}  full={ndcg_f:.4f}  diff={ndcg_f - ndcg_d:+.4f}", file=sys.stderr)
        if rec_d is not None and rec_f is not None:
            print(f"  Recall@{k:<2}     direct={rec_d:.4f}  full={rec_f:.4f}  diff={rec_f - rec_d:+.4f}", file=sys.stderr)

    mrr_d = m_direct.get("mrr", 0)
    mrr_f = m_full.get("mrr", 0)
    if mrr_f > mrr_d + 0.01:
        print("\nПолный пайплайн лучше по MRR (разница >0.01). Чанкинг doc-processor даёт выигрыш на этих данных.", file=sys.stderr)
    elif mrr_d > mrr_f + 0.01:
        print("\nНеполный пайплайн лучше по MRR. На этих данных прямая индексация выгоднее.", file=sys.stderr)
    else:
        print("\nРазница в пределах шума (~0.01). Для отличий попробуйте датасет с длинными документами (fiqa, nfcorpus).", file=sys.stderr)

    print(f"\nМетрики: {metrics_direct_path} (direct), {metrics_full_path} (full)", file=sys.stderr)

    # Machine-readable summary
    summary = {
        "direct": {"mrr": m_direct.get("mrr"), "map": m_direct.get("map"), "ndcg_at_k": m_direct.get("ndcg_at_k"), "recall_at_k": m_direct.get("recall_at_k")},
        "full": {"mrr": m_full.get("mrr"), "map": m_full.get("map"), "ndcg_at_k": m_full.get("ndcg_at_k"), "recall_at_k": m_full.get("recall_at_k")},
        "data_dir": str(data_dir),
        "limit_corpus": args.limit_corpus,
        "limit_queries": args.limit_queries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
