from __future__ import annotations

from collections import Counter

from models import ChunkResult, FuseStep, ScoreSource, TrimStep


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists.

    Returns a mapping of doc_id -> fused score, where each score is
    the sum of 1/(k + rank + 1) across all rankings that contain the doc.
    """
    contributions = (
        (doc_id, 1.0 / (k + rank + 1))
        for ranking in rankings
        for rank, doc_id in enumerate(ranking)
    )
    scores: Counter[str] = Counter()
    for doc_id, value in contributions:
        scores[doc_id] += value
    return dict(scores)


def chunk_key(chunk: ChunkResult) -> str:
    return f"{chunk.source_id}:{chunk.chunk_index}"


def dedupe_keep_best(chunks: list[ChunkResult]) -> list[ChunkResult]:
    best: dict[str, ChunkResult] = {}
    for chunk in chunks:
        key = chunk_key(chunk)
        existing = best.get(key)
        if existing is None or chunk.score > existing.score:
            best[key] = chunk

    result = list(best.values())
    result.sort(key=lambda c: c.score, reverse=True)
    return result


def adaptive_k_cutoff(
    chunks: list[ChunkResult],
    *,
    min_k: int = 3,
    max_k: int = 24,
) -> list[ChunkResult]:
    if len(chunks) <= 1:
        return chunks

    scores = [float(c.score) for c in chunks]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    if not gaps:
        return chunks

    largest_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    cutoff = max(min_k, min(max_k, largest_gap_idx + 1))
    return chunks[:cutoff]


def _build_chunk_lookup(
    source_lists: list[list[ChunkResult]],
) -> dict[str, ChunkResult]:
    """Build a key -> chunk lookup, keeping the first occurrence per key."""
    return {
        chunk_key(chunk): chunk
        for chunks in reversed(source_lists)
        for chunk in reversed(chunks)
    }


def fuse_sources(
    source_lists: list[list[ChunkResult]],
    fuse_step: FuseStep,
) -> list[ChunkResult]:
    if not source_lists:
        return []
    if len(source_lists) == 1:
        return dedupe_keep_best(source_lists[0])

    rankings = [[chunk_key(c) for c in chunks] for chunks in source_lists]
    fused_scores = rrf(rankings, k=fuse_step.rrf_k)
    lookup = _build_chunk_lookup(source_lists)

    result: list[ChunkResult] = []
    for key, fused_score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = lookup.get(key)
        if chunk is not None:
            result.append(chunk.model_copy(update={
                "score": fused_score,
                "score_source": ScoreSource.RRF,
            }))
    return result


def combine_sources(
    source_lists: list[list[ChunkResult]],
    combine: FuseStep | None,
) -> list[ChunkResult]:
    if combine is not None:
        return fuse_sources(source_lists, combine)

    all_chunks = [c for chunks in source_lists for c in chunks]
    return dedupe_keep_best(all_chunks)


def apply_finalize(
    chunks: list[ChunkResult],
    steps: list[TrimStep],
) -> list[ChunkResult]:
    # Maybe in the future we will add some more steps here
    # like summarization, etc.
    # Now we just trim the chunks to the top_k
    out = chunks
    for step in steps:
        out = out[: step.top_k]
    return out
