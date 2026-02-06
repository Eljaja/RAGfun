from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalStats:
    mrr_at_k: float
    recall_at_k: float
    count: int


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    out: list[float] = []
    for m in _NUM_RE.finditer(text.replace("\u2212", "-")):  # minus
        s = m.group(0).replace(",", "")
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def number_match(pred_text: str, gold: str, *, eps: float = 1e-2) -> bool:
    """
    Number Match from T2-RAGBench paper:
    - Let A* and A be predicted and ground truth answers.
    - Use absolute values a* = |A*| and a = |A|.
    - Correct if (a* < eps and a < eps) OR |q - 1| < eps where
        q = (a* / a) * 10^{-round(log10(a* / a))}
    We implement this by trying all numeric candidates extracted from pred_text and taking best.
    """
    try:
        a = abs(float(str(gold).strip().replace(",", "")))
    except Exception:
        return False

    nums = _extract_numbers(pred_text)
    if not nums:
        return False

    for cand in nums:
        a_star = abs(float(cand))
        if a_star < eps and a < eps:
            return True
        if a <= 0.0:
            # gold is ~0 but not caught by eps rule
            continue
        ratio = a_star / a
        if ratio <= 0.0:
            continue
        try:
            q = ratio * (10.0 ** (-round(math.log10(ratio))))
        except Exception:
            continue
        if abs(q - 1.0) < eps:
            return True

    return False


def retrieval_stats(ranks: list[int | None], *, k: int) -> RetrievalStats:
    """
    ranks: 1-based rank of the correct context within retrieved list, or None if not retrieved.
    """
    if not ranks:
        return RetrievalStats(mrr_at_k=0.0, recall_at_k=0.0, count=0)
    mrr = 0.0
    rec = 0.0
    for r in ranks:
        if r is None or r > k:
            continue
        rec += 1.0
        mrr += 1.0 / float(r)
    n = float(len(ranks))
    return RetrievalStats(mrr_at_k=mrr / n, recall_at_k=rec / n, count=int(n))

















