"""
Shared retrieval helpers for agent-search and deep-research.

- quality_is_poor: check if retrieval needs expansion (fact queries, retry)
- merge_hits: RRF-like merge with dedupe (chunk_id or doc_id:chunk_index/hash fallback)
- build_context: format hits for LLM prompt with [1], [2] numbering
"""
from __future__ import annotations

import hashlib
from typing import Any


def quality_is_poor(
    resp: dict[str, Any],
    *,
    min_hits: int = 3,
    min_score: float = 0.15,
) -> bool:
    """Return True if retrieval quality is insufficient for answering."""
    hits = list(resp.get("hits") or [])
    if not hits:
        return True
    if len(hits) < min_hits:
        return True
    if resp.get("partial") or (resp.get("degraded") or []):
        return True
    top = hits[0]
    score = top.get("rerank_score") if top.get("rerank_score") is not None else top.get("score")
    if isinstance(score, (int, float)) and score < min_score:
        return True
    return False


def _hit_key(h: dict[str, Any], fallback_idx: int) -> str:
    """Dedupe key: chunk_id, or doc_id:chunk_index, or doc_id:hash(text) as fallback."""
    cid = str(h.get("chunk_id") or "")
    if cid:
        return cid
    doc_id = str(h.get("doc_id") or "unk")
    chunk_index = h.get("chunk_index")
    if chunk_index is not None:
        return f"{doc_id}:{chunk_index}"
    text = str(h.get("text") or "")[:200]
    if text:
        hsh = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"{doc_id}:{hsh}"
    return f"{doc_id}:{fallback_idx}"


def merge_hits(responses: list[dict[str, Any]], cap: int = 20) -> list[dict[str, Any]]:
    """Merge hits from multiple retrieval responses, dedupe by chunk_id (or fallback), keep best score."""
    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    fallback_idx = 0
    for r in responses:
        for h in r.get("hits") or []:
            key = _hit_key(h, fallback_idx)
            fallback_idx += 1
            score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else 0.0
            if key not in merged or score_f > scores.get(key, -1.0):
                merged[key] = h
                scores[key] = score_f
    out = sorted(merged.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    return [h for _, h in out[:cap]]


def build_context(
    hits: list[dict[str, Any]],
    *,
    limit: int = 8,
    max_chars: int = 4000,
) -> str:
    """Build a formatted context string from hits for LLM prompt."""
    blocks: list[str] = []
    total = 0
    for i, h in enumerate(hits[:limit], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(h.get("doc_id") or "-")
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        max_block = max(300, min(1200, max_chars - total))
        if len(text) > max_block:
            text = text[: max_block - 3].rstrip() + "..."
        block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        if total + len(block) > max_chars:
            remaining = max(0, max_chars - total - 60)
            if remaining <= 0:
                break
            text = text[:remaining].rstrip() + "..."
            block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def context_from_hits(
    hits: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build context chunks from hits, filling from existing contexts where available."""
    by_cid = {str(c.get("chunk_id")): c for c in contexts if c.get("chunk_id")}
    out: list[dict[str, Any]] = []
    for h in hits:
        cid = str(h.get("chunk_id") or "")
        if cid and cid in by_cid:
            out.append(by_cid[cid])
            continue
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        out.append(
            {
                "chunk_id": cid,
                "doc_id": str(h.get("doc_id") or ""),
                "text": h.get("text"),
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "source": h.get("source"),
            }
        )
    return out


def sources_from_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract sources from context chunks with ref = 1-based index for citation [1], [2]."""
    out: list[dict[str, Any]] = []
    for i, c in enumerate(context):
        src = c.get("source") or {}
        doc_id = src.get("doc_id") or c.get("doc_id")
        if not doc_id:
            continue
        out.append(
            {
                "ref": i + 1,
                "doc_id": doc_id,
                "title": src.get("title"),
                "uri": src.get("uri"),
                "locator": src.get("locator"),
            }
        )
    return out
