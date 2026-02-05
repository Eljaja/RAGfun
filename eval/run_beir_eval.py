#!/usr/bin/env python3
"""
Полный цикл оценки на датасете BEIR: подготовка данных → индексация → baseline → метрики.

Использует датасет с готовыми тестовыми вопросами и qrels (золотые релевантности).
По умолчанию: SciFact (небольшой датасет). Можно указать nfcorpus, fiqa и др.

Требует:
  - pip install beir httpx tqdm  (beir только для --prepare)
  - Запущены retrieval (и OpenSearch, Qdrant, эмбеддинги) и Gate (для baseline).

Пример (быстрый прогон, 50 док, 20 запросов):
  python eval/run_beir_eval.py --prepare --limit-corpus 50 --limit-queries 20

Пример (полный SciFact):
  python eval/run_beir_eval.py --prepare
  python eval/run_beir_eval.py  # без --prepare: только index + baseline + metrics
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
        description="Полный цикл BEIR: подготовка → индексация → baseline → IR-метрики.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/beir/scifact",
        help="Папка с corpus.jsonl, queries.jsonl, qrels_test.tsv (default: data/beir/scifact)",
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="Имя датасета BEIR для prepare (default: scifact)",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Скачать BEIR датасет и экспорт (prepare_beir_data.py); иначе ожидать готовые файлы в --data-dir",
    )
    parser.add_argument(
        "--limit-corpus",
        type=int,
        default=None,
        help="Максимум документов для индексации (для быстрого теста)",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Максимум запросов для baseline (для быстрого теста)",
    )
    parser.add_argument(
        "--retrieval-url",
        default="http://localhost:8080",
        help="URL retrieval сервиса (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--gate-url",
        default="http://localhost:8090",
        help="URL Gate для baseline (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Не индексировать корпус (уже проиндексирован)",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Индексировать через Gate (upload → doc-processor → retrieval), а не напрямую в retrieval. Нужны storage, RabbitMQ, ingestion-worker, doc-processor.",
    )
    parser.add_argument(
        "--wait-indexed",
        action="store_true",
        help="После загрузки ждать индексации всех документов (только для --full-pipeline)",
    )
    parser.add_argument(
        "--no-fail-on-wait-timeout",
        action="store_true",
        help="При таймауте ожидания индексации не падать (только для --full-pipeline --wait-indexed)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Не запускать baseline (уже есть results.json)",
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

    # 1) Prepare (optional)
    if args.prepare:
        print("Шаг 1: подготовка данных BEIR...", file=sys.stderr)
        code = run(
            [sys.executable, str(root / "eval" / "prepare_beir_data.py"), "--dataset", args.dataset, "--out", str(data_dir)],
            cwd=root,
        )
        if code != 0:
            return code
    else:
        corpus_path = data_dir / "corpus.jsonl"
        if not corpus_path.exists():
            print(f"Нет {corpus_path}. Запустите с --prepare или положите corpus.jsonl, queries.jsonl, qrels_test.tsv в {data_dir}", file=sys.stderr)
            return 1

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels_test.tsv"
    results_path = data_dir / "results.json"
    metrics_path = data_dir / "metrics.json"

    for p in (corpus_path, queries_path, qrels_path):
        if not p.exists():
            print(f"Не найден файл: {p}", file=sys.stderr)
            return 1

    # 2) Index corpus
    if not args.skip_index:
        if args.full_pipeline:
            print("Шаг 2: индексация через Gate (полный пайплайн: doc-processor)...", file=sys.stderr)
            index_cmd = [
                sys.executable, str(root / "eval" / "index_beir_via_gate.py"),
                "--corpus", str(corpus_path),
                "--gate-url", args.gate_url,
            ]
            if args.limit_corpus is not None:
                index_cmd += ["--limit", str(args.limit_corpus)]
            if args.wait_indexed:
                index_cmd += ["--wait-indexed"]
            if getattr(args, "no_fail_on_wait_timeout", False):
                index_cmd += ["--no-fail-on-wait-timeout"]
            code = run(index_cmd, cwd=root)
        else:
            print("Шаг 2: индексация корпуса в retrieval (напрямую)...", file=sys.stderr)
            index_cmd = [
                sys.executable, str(root / "eval" / "index_beir_corpus.py"),
                "--corpus", str(corpus_path),
                "--retrieval-url", args.retrieval_url,
            ]
            if args.limit_corpus is not None:
                index_cmd += ["--limit", str(args.limit_corpus)]
            code = run(index_cmd, cwd=root)
        if code != 0:
            return code
    else:
        print("Шаг 2: пропуск индексации (--skip-index)", file=sys.stderr)

    # 3) Baseline
    if not args.skip_baseline:
        print("Шаг 3: прогон baseline (Gate)...", file=sys.stderr)
        baseline_cmd = [
            sys.executable, str(root / "eval" / "run_rag_baseline.py"),
            "--gate-url", args.gate_url,
            "--dataset", str(queries_path),
            "--format", "beir",
            "--output", str(results_path),
        ]
        if args.limit_queries is not None:
            baseline_cmd += ["--limit", str(args.limit_queries)]
        code = run(baseline_cmd, cwd=root)
        if code != 0:
            return code
    else:
        print("Шаг 3: пропуск baseline (--skip-baseline)", file=sys.stderr)

    if not results_path.exists():
        print(f"Нет {results_path}. Запустите без --skip-baseline.", file=sys.stderr)
        return 1

    # 4) IR metrics
    print("Шаг 4: расчёт IR-метрик...", file=sys.stderr)
    metrics_cmd = [
        sys.executable, str(root / "eval" / "run_rag_ir_metrics.py"),
        "--results", str(results_path),
        "--qrels", str(qrels_path),
        "--output", str(metrics_path),
    ]
    if args.k:
        metrics_cmd += ["--k"] + [str(k) for k in args.k]
    code = run(metrics_cmd, cwd=root)
    if code != 0:
        return code

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        print("\n--- Метрики ---", file=sys.stderr)
        print(f"  Запросов с qrels: {m.get('num_queries_with_qrels')} / {m.get('num_queries_total')}", file=sys.stderr)
        print(f"  MRR:    {m.get('mrr', 0):.4f}", file=sys.stderr)
        print(f"  MAP:    {m.get('map', 0):.4f}", file=sys.stderr)
        for k in args.k:
            ndcg = (m.get("ndcg_at_k") or {}).get(k)
            rec = (m.get("recall_at_k") or {}).get(k)
            if ndcg is not None and rec is not None:
                print(f"  nDCG@{k}: {ndcg:.4f}  Recall@{k}: {rec:.4f}", file=sys.stderr)
        print(f"\nМетрики сохранены: {metrics_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
