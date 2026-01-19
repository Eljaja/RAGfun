from __future__ import annotations

from typing import Iterable


def rrf_fusion(
    ranked_lists: dict[str, list[str]],
    weights: dict[str, float],
    rrf_k: int,
) -> dict[str, float]:
    """
    ranked_lists: source_name -> [chunk_id in rank order]
    returns chunk_id -> fused_score
    """
    scores: dict[str, float] = {}
    for src, ids in ranked_lists.items():
        w = float(weights.get(src, 1.0))
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w * (1.0 / (rrf_k + rank))
    return scores


def score_fusion(
    *,
    scored_lists: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Normalize per-source scores to [0..1] (by max) and compute weighted sum.

    This helps hybrid fusion when each retriever has its own score scale
    (e.g. BM25 vs cosine similarity).
    """
    out: dict[str, float] = {}
    for src, cid2score in scored_lists.items():
        if not cid2score:
            continue
        w = float(weights.get(src, 1.0))
        max_s = max(cid2score.values()) if cid2score else 0.0
        if max_s <= 0:
            continue
        for cid, s in cid2score.items():
            out[cid] = out.get(cid, 0.0) + w * (float(s) / max_s)
    return out
def hybrid_fusion(
    *,
    ranked_lists: dict[str, list[str]],
    scored_lists: dict[str, dict[str, float]],
    weights: dict[str, float],
    rrf_k: int,
    alpha: float,
) -> dict[str, float]:
    """
    Combine rank-based RRF and normalized-score fusion:
      fused = alpha * RRF + (1-alpha) * normalized_score_sum

    alpha in [0..1]. Larger alpha relies more on rank agreement.
    """
    a = max(0.0, min(1.0, float(alpha)))
    rrf = rrf_fusion(ranked_lists, weights, rrf_k) if ranked_lists else {}
    sc = score_fusion(scored_lists=scored_lists, weights=weights) if scored_lists else {}
    keys: set[str] = set(rrf.keys()) | set(sc.keys())
    out: dict[str, float] = {}
    for cid in keys:
        out[cid] = a * float(rrf.get(cid, 0.0)) + (1.0 - a) * float(sc.get(cid, 0.0))
    return out