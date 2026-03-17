"""Chunk processing for LLM context: merging, formatting, citations, sources."""

from __future__ import annotations

import re

from models import ChunkResult

_CIT_RE = re.compile(r"\[\d+\]")


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


def build_context(
    chunks: list[ChunkResult],
    *,
    max_chars: int = 6000,
    max_chunk_chars: int = 1200,
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
        header = f"[{i}] source={chunk.source_id} score={chunk.score:.4f}"
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


def extract_sources(chunks: list[ChunkResult]) -> list[str]:
    """Deduplicate source IDs from chunks."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.source_id not in seen:
            seen.add(chunk.source_id)
            sources.append(chunk.source_id)
    return sources


def extract_source_details(chunks: list[ChunkResult]) -> list[dict]:
    """Build source list with ref numbers for citation (1-based)."""
    seen: dict[str, int] = {}
    sources: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.source_id not in seen:
            seen[chunk.source_id] = i
            sources.append({"ref": i, "source_id": chunk.source_id})
    return sources


def has_citations(text: str) -> bool:
    return bool(text) and bool(_CIT_RE.search(text))


def history_summary(history: list[dict[str, str]], max_turns: int = 3) -> str:
    """Format last N conversation turns for prompt context."""
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
