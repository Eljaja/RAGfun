#!/usr/bin/env python3
"""
Статистика по corpus.jsonl (BEIR): длина документов, распределение.

Помогает выбрать датасет, на котором разница между полным пайплайном
(doc-processor semantic chunking) и прямой индексацией будет заметна:
нужны документы с разумной длиной (не один абзац), где чанкинг по смыслу даёт выигрыш.

Пример:
  python eval/corpus_stats.py data/beir/scifact/corpus.jsonl
  python eval/corpus_stats.py data/beir/fiqa/corpus.jsonl --limit 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Статистика длины документов в corpus.jsonl")
    parser.add_argument("corpus", help="Путь к corpus.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Максимум строк для анализа")
    parser.add_argument("--chars", action="store_true", help="Показывать в символах (по умолчанию — слова)")
    args = parser.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    lengths: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if args.chars:
                    lengths.append(len(text))
                else:
                    lengths.append(len(text.split()))
            except json.JSONDecodeError:
                continue

    if not lengths:
        print("Нет документов.", file=sys.stderr)
        return 1

    n = len(lengths)
    lengths_sorted = sorted(lengths)
    unit = "символов" if args.chars else "слов"
    p50 = lengths_sorted[n // 2] if n else 0
    p90 = lengths_sorted[int(n * 0.9)] if n else 0
    p99 = lengths_sorted[int(n * 0.99)] if n else 0

    print(f"Документов: {n}", file=sys.stderr)
    print(f"Длина ({unit}): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/n:.0f}, median={p50}, p90={p90}, p99={p99}", file=sys.stderr)

    # Рекомендация: где чанкинг может дать отличия
    if args.chars:
        if p50 < 500 and p90 < 1500:
            print("\nКороткие документы (медиана <500 символов). Разница полный/неполный пайплайн скорее мала (как на SciFact).", file=sys.stderr)
        elif p50 >= 500 or p90 >= 2000:
            print("\nЕсть документы средней/большой длины. Хороший кандидат для сравнения полный vs неполный пайплайн.", file=sys.stderr)
    else:
        if p50 < 80 and p90 < 200:
            print("\nКороткие документы (медиана <80 слов). Разница полный/неполный пайплайн скорее мала.", file=sys.stderr)
        elif p50 >= 80 or p90 >= 300:
            print("\nЕсть документы средней/большой длины. Хороший кандидат для сравнения полный vs неполный пайплайн.", file=sys.stderr)

    # JSON для скриптов
    out = {
        "num_docs": n,
        "unit": unit,
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / n, 1),
        "median": p50,
        "p90": p90,
        "p99": p99,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
