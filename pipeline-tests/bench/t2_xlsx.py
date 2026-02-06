from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openpyxl


@dataclass(frozen=True)
class QAExample:
    qid: str
    question: str
    generated_question: str | None
    answer: str | None
    doc_id: str | None
    oracle_context: str | None
    # Raw row for debugging
    raw: dict[str, Any]


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def read_xlsx_examples(path: str, *, sheet_name: str | None = None, limit: int | None = None) -> list[QAExample]:
    """
    Best-effort reader for the xlsx files in test/annotations/data.
    The exact schema differs across datasets, so we:
    - read first sheet (or sheet_name)
    - detect columns by name heuristics
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.worksheets[0]

    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    cols = [_norm(h) for h in header]

    # column detection (common variants)
    def find_col(candidates: list[str]) -> int | None:
        lower = [c.lower() for c in cols]
        for cand in candidates:
            if cand.lower() in lower:
                return lower.index(cand.lower())
        return None

    i_qid = find_col(["id", "qid", "question_id", "query_id"])
    i_q = find_col(["question", "query"])
    i_gen_q = find_col(["generated_question", "generated query", "generated_query"])
    i_a = find_col(["answer", "gold_answer", "final_answer"])
    i_doc = find_col(["doc_id", "document_id", "report_id"])
    i_ctx = find_col(["context", "oracle_context"])

    out: list[QAExample] = []
    for n, row in enumerate(rows, start=1):
        if limit is not None and len(out) >= limit:
            break
        if not row:
            continue
        question = _norm(row[i_q]) if i_q is not None and i_q < len(row) else ""
        if not question:
            continue
        qid = _norm(row[i_qid]) if i_qid is not None and i_qid < len(row) else str(n)
        gen_q = _norm(row[i_gen_q]) if i_gen_q is not None and i_gen_q < len(row) else ""
        ans = _norm(row[i_a]) if i_a is not None and i_a < len(row) else ""
        doc_id = _norm(row[i_doc]) if i_doc is not None and i_doc < len(row) else ""
        ctx = _norm(row[i_ctx]) if i_ctx is not None and i_ctx < len(row) else ""
        raw = {cols[i]: row[i] for i in range(min(len(cols), len(row)))}
        out.append(
            QAExample(
                qid=qid,
                question=question,
                generated_question=gen_q if gen_q else None,
                answer=ans if ans else None,
                doc_id=doc_id if doc_id else None,
                oracle_context=ctx if ctx else None,
                raw=raw,
            )
        )
    return out

















