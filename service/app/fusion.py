from __future__ import annotations


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


