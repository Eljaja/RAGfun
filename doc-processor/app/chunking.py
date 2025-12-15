from __future__ import annotations


def chunk_text_chars(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1)) if chunk_size > 1 else 0

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return [c for c in out if c]





