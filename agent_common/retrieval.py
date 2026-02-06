"""Shared retrieval helpers for agent-search and deep-research."""

from __future__ import annotations

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


def merge_hits(responses: list[dict[str, Any]], cap: int = 20) -> list[dict[str, Any]]:
    """Merge hits from multiple retrieval responses, dedupe by chunk_id, keep best score."""
    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for r in responses:
        for h in r.get("hits") or []:
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else 0.0
            if cid not in merged or score_f > scores.get(cid, -1.0):
                merged[cid] = h
                scores[cid] = score_f
    out = sorted(merged.values(), key=lambda h: scores.get(str(h.get("chunk_id") or ""), 0.0), reverse=True)
    return out[:cap]


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
    """Extract unique sources from context chunks."""
    seen: set[tuple[str, Any]] = set()
    out: list[dict[str, Any]] = []
    for c in context:
        src = c.get("source") or {}
        doc_id = src.get("doc_id") or c.get("doc_id")
        if not doc_id:
            continue
        key = (doc_id, src.get("uri"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "ref": src.get("ref"),
                "doc_id": doc_id,
                "title": src.get("title"),
                "uri": src.get("uri"),
                "locator": src.get("locator"),
            }
        )
    return out
