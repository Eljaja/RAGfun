from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Default encoding used by GPT-4, GPT-3.5-turbo, and text-embedding-3-large
_DEFAULT_ENCODING = "cl100k_base"

# Lazy-loaded encoding instance
if TYPE_CHECKING:
    _encoding_cache: tiktoken.Encoding | None = None
else:
    _encoding_cache = None


def _get_encoding():
    """Get tiktoken encoding, fallback to None if not available."""
    global _encoding_cache
    if _encoding_cache is not None:
        return _encoding_cache
    
    if tiktoken is None:
        return None
    
    try:
        _encoding_cache = tiktoken.get_encoding(_DEFAULT_ENCODING)
        return _encoding_cache
    except Exception:
        return None


def token_count(text: str, encoding=None) -> int:
    """
    Accurate token count using tiktoken.
    Falls back to approximation if tiktoken is not available.
    """
    if encoding is None:
        encoding = _get_encoding()
    
    if encoding is not None:
        try:
            return len(encoding.encode(text, allowed_special="all"))
        except Exception:
            pass
    
    # Fallback: approximation using word count
    return max(1, len(re.findall(r"\S+", text)))


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, str, int]]:
    """
    Returns list of (chunk_index, chunk_text, token_count).
    Chunking strategy: paragraph-aware + sliding window on tokens.
    Uses tiktoken for accurate token counting (cl100k_base encoding).
    """
    text = text.strip()
    if not text:
        return []

    encoding = _get_encoding()
    
    if encoding is None:
        # Fallback to simple regex-based approach if tiktoken unavailable
        return _chunk_text_fallback(text, max_tokens, overlap_tokens)
    
    # Split into blocks (paragraphs) but preserve code/table-ish blocks
    blocks = re.split(r"\n{2,}", text)
    blocks = [b.strip() for b in blocks if b.strip()]
    
    if not blocks:
        return []
    
    # Encode all blocks to tokens and track block boundaries
    all_tokens: list[int] = []
    block_boundaries: list[int] = []  # token indices where blocks end
    
    for i, block in enumerate(blocks):
        try:
            block_tokens = encoding.encode(block, allowed_special="all")
            all_tokens.extend(block_tokens)
            # Add separator tokens between blocks
            if i < len(blocks) - 1:  # Don't add separator after last block
                separator_tokens = encoding.encode("\n\n", allowed_special="all")
                all_tokens.extend(separator_tokens)
                # Block boundary is after the separator
                block_boundaries.append(len(all_tokens))
            else:
                # Last block: boundary is at the end
                block_boundaries.append(len(all_tokens))
        except Exception:
            # If encoding fails for a block, skip it
            continue
    
    if not all_tokens:
        return []
    
    chunks: list[tuple[int, str, int]] = []
    start = 0
    idx = 0
    n = len(all_tokens)
    
    while start < n:
        end = min(n, start + max_tokens)
        
        # Try to cut on a block boundary if close (within 20 tokens)
        for boundary in block_boundaries:
            if start < boundary <= end and (end - boundary) <= 20:
                end = boundary
                break
        
        # Extract tokens for this chunk
        chunk_tokens = all_tokens[start:end]
        
        if not chunk_tokens:
            break
        
        # Decode tokens back to text
        try:
            ctext = encoding.decode(chunk_tokens)
        except Exception:
            # If decoding fails, try to extract text from original blocks
            # Find blocks that overlap with this token range
            token_start_in_block = 0
            token_end_in_block = 0
            block_start_idx = 0
            
            # Find which blocks are included
            for i, boundary in enumerate(block_boundaries):
                if boundary > start:
                    block_start_idx = i
                    if i > 0:
                        token_start_in_block = start - block_boundaries[i - 1]
                    else:
                        token_start_in_block = start
                    break
            
            # Reconstruct text from overlapping blocks (simplified)
            ctext = "\n\n".join(blocks[max(0, block_start_idx - 1):block_start_idx + 2])
        
        if ctext:
            # Count tokens accurately
            token_cnt = token_count(ctext, encoding)
            chunks.append((idx, ctext, token_cnt))
            idx += 1
        
        if end >= n:
            break
        
        # Apply overlap: move start back by overlap_tokens
        start = max(0, end - overlap_tokens)
        
        # If we haven't moved forward (stuck), break to avoid infinite loop
        if start >= end:
            break
    
    return chunks


# --- Semantic / section-based chunking (mode=document) ---
# Section boundaries: Markdown headings, numbered sections. RU/EN agnostic.
_RE_HEADING = re.compile(r"^(#{1,6})\s+.+$")
_RE_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*\.)\s+\S")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading_line, body) sections. Heading may be empty for first section."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_body
        body = "\n".join(current_body).strip()
        if body or current_heading:
            sections.append((current_heading, body))
        current_heading = ""
        current_body = []

    for line in lines:
        stripped = line.strip()
        is_heading = bool(_RE_HEADING.match(line)) or bool(_RE_NUMBERED.match(line))
        if is_heading:
            flush()
            current_heading = line.rstrip()
            current_body = []
        else:
            current_body.append(line)
    flush()
    return sections


def chunk_text_semantic(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    encoding=None,
) -> list[tuple[int, str, int]]:
    """
    Section-based chunking for mode=document. Splits on headings/sections, fits within max_tokens.
    Minimal overlap (only when a section exceeds max_tokens). Returns (chunk_index, chunk_text, token_count).
    """
    text = text.strip()
    if not text:
        return []
    enc = encoding or _get_encoding()
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1)) if max_tokens > 1 else 0

    sections = _split_sections(text)
    chunks: list[tuple[int, str, int]] = []
    idx = 0
    cur_parts: list[str] = []
    cur_tokens = 0

    def token_len(s: str) -> int:
        return token_count(s, enc)

    def flush() -> None:
        nonlocal cur_parts, cur_tokens, idx
        if cur_parts:
            ctext = "\n\n".join(cur_parts).strip()
            if ctext:
                cnt = token_len(ctext)
                chunks.append((idx, ctext, cnt))
                idx += 1
        cur_parts = []
        cur_tokens = 0

    for heading, body in sections:
        if not body:
            if heading:
                t = token_len(heading)
                sep = 2 if cur_parts else 0
                if cur_parts and cur_tokens + sep + t > max_tokens:
                    flush()
                cur_parts.append(heading)
                cur_tokens += (sep if cur_tokens else 0) + t
            continue
        section_text = (heading + "\n\n" + body).strip() if heading else body
        t = token_len(section_text)

        if t > max_tokens:
            flush()
            # Fallback: token-based split with overlap for oversized section
            for _, ctext, cnt in _chunk_text_fallback(section_text, max_tokens, overlap_tokens):
                chunks.append((idx, ctext, cnt))
                idx += 1
            continue

        sep = 2 if cur_parts else 0
        if cur_parts and cur_tokens + sep + t > max_tokens:
            flush()
        cur_parts.append(section_text)
        cur_tokens += (sep if cur_tokens else 0) + t

    flush()
    return chunks


def chunk_by_strategy(
    text: str,
    *,
    strategy: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, str, int]]:
    """
    Single entry point. strategy: "semantic" (section-based) or "token" (paragraph + sliding window).
    Returns list of (chunk_index, chunk_text, token_count).
    """
    if strategy == "semantic":
        return chunk_text_semantic(text, max_tokens, overlap_tokens)
    return chunk_text(text, max_tokens, overlap_tokens)


def _chunk_text_fallback(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, str, int]]:
    """
    Fallback chunking using regex-based token approximation.
    Used when tiktoken is not available.
    """
    text = text.strip()
    if not text:
        return []

    # Split into blocks
    blocks = re.split(r"\n{2,}", text)
    tokens: list[str] = []
    block_boundaries: list[int] = []

    for b in blocks:
        bt = re.findall(r"\S+|\n", b.strip())
        tokens.extend(bt)
        block_boundaries.append(len(tokens))
        tokens.append("\n\n")

    # Remove trailing separators
    while tokens and tokens[-1] in ("\n", "\n\n"):
        tokens.pop()

    def tokens_to_text(ts: list[str]) -> str:
        s = " ".join([t for t in ts if t not in ("\n", "\n\n")])
        s = re.sub(r"\s+\n\s+", "\n", s)
        return s.strip()

    chunks: list[tuple[int, str, int]] = []
    start = 0
    idx = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + max_tokens)

        # Try to cut on a block boundary if close
        for b in block_boundaries:
            if start < b <= end and (end - b) <= 20:
                end = b
                break

        ctoks = tokens[start:end]
        ctext = tokens_to_text(ctoks)
        if ctext:
            chunks.append((idx, ctext, len(ctoks)))
            idx += 1

        if end >= n:
            break
        start = max(0, end - overlap_tokens)

    return chunks


