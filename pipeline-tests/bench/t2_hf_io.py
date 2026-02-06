from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class T2Row:
    id: str
    context_id: str
    split: str
    question: str
    program_answer: str
    original_answer: Any
    context: str
    file_name: str | None
    raw: dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _context_to_text(value: Any) -> str:
    """
    Dataset `context` field:
    - FinQA/ConvFinQA/TAT-DQA: plain string
    - VQAonBD: often stringified Python list of markdown table strings
    """
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            # Try to parse python list literal, then join.
            try:
                lst = ast.literal_eval(s)
                if isinstance(lst, list):
                    parts = [str(x) for x in lst if x is not None]
                    return "\n\n".join([p.strip() for p in parts if p.strip()])
            except Exception:
                return s
        return s
    if isinstance(value, list):
        parts = [str(x) for x in value if x is not None]
        return "\n\n".join([p.strip() for p in parts if p.strip()])
    return str(value).strip()


def iter_t2_rows(paths: Iterable[str | Path]) -> Iterator[T2Row]:
    for path in paths:
        for raw in iter_jsonl(path):
            yield T2Row(
                id=str(raw.get("id", "")),
                context_id=str(raw.get("context_id", "")),
                split=str(raw.get("split", "")),
                question=str(raw.get("question", "")),
                program_answer=str(raw.get("program_answer", "")),
                original_answer=raw.get("original_answer"),
                context=_context_to_text(raw.get("context")),
                file_name=raw.get("file_name"),
                raw=raw,
            )

















