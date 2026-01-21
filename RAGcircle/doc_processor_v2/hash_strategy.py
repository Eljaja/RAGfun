"""
Hash strategies for idempotent chunk embedding.
Avoids re-vectorizing chunks that haven't changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


def sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class ExistingChunk:
    """Represents a chunk already in Qdrant."""
    exists: bool = False
    content_hash: str | None = None
    embed_hash: str | None = None


class HashStrategy(Protocol):
    """Protocol for hash comparison strategies."""
    def get_desired_hash(self, text: str) -> str: ...
    def get_existing_hash(self, existing: ExistingChunk) -> str | None: ...


class ContentHashStrategy:
    """Compare by content hash (raw chunk text)."""
    def get_desired_hash(self, text: str) -> str:
        return sha256_hex(text)

    def get_existing_hash(self, existing: ExistingChunk) -> str | None:
        return existing.content_hash


def needs_embedding(
    text: str,
    existing: ExistingChunk,
    strategy: HashStrategy,
) -> bool:
    """Check if a chunk needs to be embedded."""
    if not existing.exists:
        return True
    desired = strategy.get_desired_hash(text)
    current = strategy.get_existing_hash(existing)
    return current != desired


def filter_chunks_needing_embedding(
    chunks: list,
    existing_by_id: dict[str, ExistingChunk],
    strategy: HashStrategy | None = None,
) -> tuple[list, int]:
    """
    Filter chunks that need embedding.
    
    Returns:
        (chunks_to_embed, skipped_count)
    """
    if strategy is None:
        strategy = ContentHashStrategy()

    need_embed = []
    for c in chunks:
        existing = existing_by_id.get(c.chunk_id, ExistingChunk(exists=False))
        if needs_embedding(c.text, existing, strategy):
            need_embed.append(c)

    skipped = len(chunks) - len(need_embed)
    return need_embed, skipped
