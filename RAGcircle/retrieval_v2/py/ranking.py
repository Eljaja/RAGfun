from __future__ import annotations

import re
from collections import Counter as CollCounter

from models import ChunkResult, FuseStep, TrimStep

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def keyword_query(q: str) -> str:
    toks = [t.lower() for t in _TOKEN_RE.findall(q or "")]
    if not toks:
        return ""
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
        "are", "was", "were", "be", "been", "with", "what", "which", "who", "when",
        "where", "why", "how", "many", "much", "did", "does", "do", "have", "has",
        "что", "какой", "какая", "какие", "кто", "где", "когда", "как", "сколько",
        "это", "эта", "эти", "этот", "для", "или", "а", "и", "в", "на", "по", "с",
    }
    toks = [t for t in toks if len(t) >= 3 and t not in stop]
    if not toks:
        return ""
    c = CollCounter(toks)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    keep = [t for t, _ in ranked[:10]]
    return " ".join(keep)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def chunk_key(chunk: ChunkResult) -> str:
    return f"{chunk.source_id}:{chunk.chunk_index}"


def dedupe_keep_best(chunks: list[ChunkResult]) -> list[ChunkResult]:
    best: dict[str, ChunkResult] = {}
    for chunk in chunks:
        key = chunk_key(chunk)
        current = best.get(key)
        if current is None or chunk.score > current.score:
            best[key] = chunk
    out = list(best.values())
    out.sort(key=lambda c: c.score, reverse=True)
    return out


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
    best_i = max(range(len(gaps)), key=lambda i: gaps[i])
    effective_k = max(min_k, min(max_k, best_i + 1))
    return chunks[:effective_k]


def fuse_sources(
    source_lists: list[list[ChunkResult]],
    fuse_step: FuseStep,
) -> list[ChunkResult]:
    if not source_lists:
        return []
    if len(source_lists) == 1:
        return dedupe_keep_best(source_lists[0])

    rankings = [[chunk_key(c) for c in chunks] for chunks in source_lists]
    keyed_chunks = [
        (chunk_key(c), c)
        for chunks in source_lists
        for c in chunks
    ]
    payloads = {cid: chunk for cid, chunk in reversed(keyed_chunks)}

    return [
        base.model_copy(update={"score": float(fused_score)})
        for cid, fused_score in rrf(rankings, k=fuse_step.rrf_k)
        if (base := payloads.get(cid)) is not None
    ]


def combine_sources(
    source_lists: list[list[ChunkResult]],
    combine: FuseStep | None,
) -> list[ChunkResult]:
    if combine is not None:
        return fuse_sources(source_lists, combine)
    flat = [c for chunks in source_lists for c in chunks]
    return dedupe_keep_best(flat)


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
