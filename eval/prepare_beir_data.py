#!/usr/bin/env python3
"""
Подготовка данных BEIR для «серьёзной» оценки retrieval.

Скачивает датасет BEIR (например SciFact или NFCorpus), экспортирует:
  - corpus.jsonl — документы для индекса (doc_id и text; можно загружать в RAG);
  - queries.jsonl — запросы с qid для run_rag_baseline.py;
  - qrels_test.tsv — qrels в TREC-формате для run_rag_ir_metrics.py.

Требует: pip install beir

Пример:
  python eval/prepare_beir_data.py --dataset scifact --out data/beir/scifact
  # Индексируйте corpus (каждый документ с doc_id из corpus), затем:
  python eval/run_rag_baseline.py --dataset data/beir/scifact/queries.jsonl --format beir --output results.json
  python eval/run_rag_ir_metrics.py --results results.json --qrels data/beir/scifact/qrels_test.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print("Установите BEIR: pip install beir", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Скачать BEIR датасет и экспорт corpus/queries/qrels.")
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="Имя датасета BEIR: scifact, nfcorpus, fiqa, arguana, ... (default: scifact)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Папка вывода (default: data/beir/<dataset>)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Сплит (default: test)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out or f"data/beir/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    print(f"Скачивание {args.dataset}...", file=sys.stderr)
    util.download_and_unzip(url, str(data_path))

    # BEIR кладёт данные в data_path/<dataset>/ или в data_path
    possible = [data_path / args.dataset, data_path]
    load_path = None
    for p in possible:
        if (p / "corpus.jsonl").exists() or (p / "qrels").exists():
            load_path = p
            break
    if load_path is None:
        # Попробуем GenericDataLoader с data_path
        load_path = data_path
        for sub in load_path.iterdir():
            if sub.is_dir() and (sub / "corpus.jsonl").exists():
                load_path = sub
                break

    print(f"Загрузка из {load_path}...", file=sys.stderr)
    corpus, queries, qrels = GenericDataLoader(str(load_path)).load(split=args.split)

    # corpus: doc_id -> {"title": "...", "text": "..."}
    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, doc in corpus.items():
            text = (doc.get("text") or "").strip()
            if doc.get("title"):
                text = (doc["title"] + " " + text).strip()
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    print(f"  {corpus_path}: {len(corpus)} документов", file=sys.stderr)

    # queries: qid -> query text
    queries_path = out_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"query": text, "qid": qid}, ensure_ascii=False) + "\n")
    print(f"  {queries_path}: {len(queries)} запросов", file=sys.stderr)

    # qrels: qid -> {doc_id: relevance}
    qrels_path = out_dir / "qrels_test.tsv"
    n_qrel = 0
    with open(qrels_path, "w", encoding="utf-8") as f:
        for qid, doc_rels in qrels.items():
            for doc_id, rel in doc_rels.items():
                f.write(f"{qid}\t{doc_id}\t{rel}\n")
                n_qrel += 1
    print(f"  {qrels_path}: {n_qrel} пар (qid, doc_id, relevance)", file=sys.stderr)

    print(f"\nДальше:", file=sys.stderr)
    print(f"  1) Проиндексируйте corpus (doc_id из corpus.jsonl должен совпадать с doc_id в RAG).", file=sys.stderr)
    print(f"  2) python eval/run_rag_baseline.py --dataset {queries_path} --format beir --output {out_dir / 'results.json'}", file=sys.stderr)
    print(f"  3) python eval/run_rag_ir_metrics.py --results {out_dir / 'results.json'} --qrels {qrels_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
