"""Chunk processing for LLM context: merging, stitching, formatting, citations, sources."""

from __future__ import annotations

import re
from itertools import groupby
from typing import Any

from retrieval_contract import ChunkResult

_CIT_RE = re.compile(r"\[\d+\]")


# ── Merging ──────────────────────────────────────────────


def merge_chunks(
    chunk_lists: list[list[ChunkResult]],
    *,
    cap: int = 30,
) -> list[ChunkResult]:
    """Merge multiple retrieval results, dedup by (source_id, chunk_index), keep best score."""
    best: dict[str, ChunkResult] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            key = f"{chunk.source_id}:{chunk.chunk_index}"
            existing = best.get(key)
            if existing is None or chunk.score > existing.score:
                best[key] = chunk
    result = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return result[:cap]


# ── Segment stitching ────────────────────────────────────


def stitch_segments(
    chunks: list[ChunkResult],
    *,
    max_per_segment: int = 4,
) -> list[ChunkResult]:
    """Combine adjacent chunks from the same document into larger segments.

    Groups by source_id, sorts by chunk_index within each group,
    joins text with '\\n...\\n', and returns synthetic chunks using
    the best score and lowest chunk_index per group.
    """
    if not chunks:
        return []

    keyfn = lambda c: c.source_id
    sorted_chunks = sorted(chunks, key=lambda c: (c.source_id, c.chunk_index))
    stitched: list[ChunkResult] = []

    for _src_id, group_iter in groupby(sorted_chunks, key=keyfn):
        group = list(group_iter)[:max_per_segment]
        if len(group) == 1:
            stitched.append(group[0])
            continue
        combined_text = "\n...\n".join(c.text.strip() for c in group if c.text.strip())
        if not combined_text:
            continue
        best = max(group, key=lambda c: c.score)
        stitched.append(ChunkResult(
            text=combined_text,
            source_id=group[0].source_id,
            chunk_index=group[0].chunk_index,
            score=best.score,
            score_source=best.score_source,
        ))

    stitched.sort(key=lambda c: c.score, reverse=True)
    return stitched


# ── Context building ─────────────────────────────────────


def build_context(
    chunks: list[ChunkResult],
    *,
    max_chars: int = 6000,
    max_chunk_chars: int = 1200,
    # This one is left because it might have use in the future plus code already references it
    source_meta: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format chunks into a numbered context string for the LLM."""
    blocks: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if not text:
            continue
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars - 3].rstrip() + "..."

        header = f"[{i}] source={chunk.source_id}"
        header += f" score={chunk.score:.4f}"

        block = f"{header}\n{text}"
        if total + len(block) > max_chars:
            remaining = max_chars - total - len(header) - 10
            if remaining <= 0:
                break
            text = text[:remaining].rstrip() + "..."
            block = f"{header}\n{text}"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


# ── Source extraction ────────────────────────────────────


def extract_sources(chunks: list[ChunkResult]) -> list[str]:
    """Deduplicate source IDs from chunks."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.source_id not in seen:
            seen.add(chunk.source_id)
            sources.append(chunk.source_id)
    return sources


def extract_source_details(
    chunks: list[ChunkResult],
) -> list[dict]:
    """Build source list with ref numbers."""
    seen: dict[str, int] = {}
    sources: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.source_id not in seen:
            seen[chunk.source_id] = i
            sources.append({"ref": i, "source_id": chunk.source_id})
    return sources


# ── Citations ────────────────────────────────────────────


def has_citations(text: str) -> bool:
    return bool(text) and bool(_CIT_RE.search(text))


# ── Conversation history ─────────────────────────────────


def history_as_messages(
    history: list[dict[str, str]],
    *,
    max_turns: int | None = None,
) -> list[dict[str, str]]:
    """Return full chat history as validated message dicts.

    Suitable for direct insertion into the LLM messages list.
    """
    valid: list[dict[str, str]] = []
    for m in history:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant", "system") and content:
            valid.append({"role": role, "content": content})
    if max_turns is not None and max_turns > 0:
        valid = valid[-(max_turns * 2):]
    return valid


def history_summary(history: list[dict[str, str]], max_turns: int = 3) -> str:
    """Format last N conversation turns for prompt context (legacy)."""
    if not history:
        return ""
    turns = history[-(max_turns * 2):]
    parts = []
    for m in turns:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content[:200]}{'...' if len(content) > 200 else ''}")
    if not parts:
        return ""
    return "Recent conversation:\n" + "\n".join(parts) + "\n\n"
