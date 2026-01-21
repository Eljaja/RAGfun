from __future__ import annotations

import re
from dataclasses import dataclass


def chunk_text_chars(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """
    Chunk text by character budget, but try hard to preserve Markdown structure,
    especially tables and fenced code blocks.

    Why: VLM extraction can return Markdown tables. Naive char slicing can split
    table rows/columns and drastically reduce retrieval quality.
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1)) if chunk_size > 1 else 0

    blocks = _split_markdown_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    cur: list[_Block] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        joined = _join_blocks(cur).strip()
        if joined:
            chunks.append(joined)
        cur = []
        cur_len = 0

    for b in blocks:
        btxt = b.text.strip()
        if not btxt:
            continue

        # If a single block doesn't fit: split it safely.
        if len(btxt) > chunk_size:
            flush()
            if b.kind == "table":
                chunks.extend(_split_markdown_table_block(btxt, chunk_size=chunk_size, overlap=overlap))
            else:
                chunks.extend(_split_text_fallback(btxt, chunk_size=chunk_size, overlap=overlap))
            continue

        # Try append to current chunk.
        add_sep = 2 if cur else 0  # we'll join with \n\n between blocks
        if cur and (cur_len + add_sep + len(btxt) > chunk_size):
            # finalize current, start new with best-effort overlap using whole blocks
            prev = cur[:]
            flush()
            if overlap > 0:
                carry: list[_Block] = []
                carry_len = 0
                for bb in reversed(prev):
                    t = bb.text.strip()
                    if not t:
                        continue
                    # approximate join cost: add \n\n between blocks
                    extra = len(t) + (2 if carry else 0)
                    if carry_len + extra > overlap:
                        break
                    carry.insert(0, bb)
                    carry_len += extra
                if carry:
                    cur = carry
                    cur_len = len(_join_blocks(cur))

        if not cur:
            cur = [b]
            cur_len = len(btxt)
        else:
            cur.append(b)
            cur_len = len(_join_blocks(cur))

    flush()
    return [c for c in chunks if c]


@dataclass(frozen=True)
class _Block:
    kind: str  # "text" | "code" | "table"
    text: str


_RE_FENCE = re.compile(r"^```")
_RE_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_markdown_blocks(text: str) -> list[_Block]:
    """
    Split Markdown-ish text into blocks, keeping fenced code and Markdown tables intact.
    """
    lines = text.splitlines()
    out: list[_Block] = []
    buf: list[str] = []

    def flush_text() -> None:
        nonlocal buf
        if not buf:
            return
        s = "\n".join(buf).strip()
        buf = []
        if s:
            out.append(_Block(kind="text", text=s))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # fenced code block
        if _RE_FENCE.match(line.strip()):
            flush_text()
            fence = line.strip()[:3]
            code_lines = [line]
            i += 1
            while i < n:
                code_lines.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            out.append(_Block(kind="code", text="\n".join(code_lines).strip()))
            continue

        # Markdown table block: header line + separator line + rows
        if i + 1 < n and "|" in line and _RE_MD_TABLE_SEP.match(lines[i + 1]):
            flush_text()
            tbl = [line, lines[i + 1]]
            i += 2
            while i < n:
                l = lines[i]
                if not l.strip():
                    break
                # stop if we clearly left table-like lines
                if "|" not in l:
                    break
                tbl.append(l)
                i += 1
            out.append(_Block(kind="table", text="\n".join(tbl).strip()))
            # consume any blank lines after table as separator
            while i < n and not lines[i].strip():
                i += 1
            continue

        # paragraph split on blank lines
        if not line.strip():
            flush_text()
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            continue

        buf.append(line)
        i += 1

    flush_text()
    return out


def _join_blocks(blocks: list[_Block]) -> str:
    return "\n\n".join(b.text.strip() for b in blocks if b.text and b.text.strip()).strip()


def _split_text_fallback(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    overlap = max(0, min(overlap, chunk_size - 1)) if chunk_size > 1 else 0
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = max(0, j - overlap)
        if i >= j:
            break
    return [c for c in out if c]


def _split_markdown_table_block(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a large Markdown table by rows, preserving header + separator in every chunk.
    """
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return _split_text_fallback(text, chunk_size=chunk_size, overlap=overlap)

    header = lines[0]
    sep = lines[1]
    rows = lines[2:]
    prefix = header + "\n" + sep + "\n"

    # Heuristic: overlap by 1 row if overlap is enabled.
    overlap_rows = 1 if overlap > 0 else 0

    chunks: list[str] = []
    i = 0
    while i < len(rows):
        cur_rows: list[str] = []
        cur_len = len(prefix)
        j = i
        while j < len(rows):
            r = rows[j]
            extra = len(r) + 1  # + newline
            if cur_rows and cur_len + extra > chunk_size:
                break
            if not cur_rows and cur_len + extra > chunk_size:
                # single row too big: fall back to text slicing for this row
                chunks.extend(_split_text_fallback(prefix + r, chunk_size=chunk_size, overlap=overlap))
                j += 1
                break
            cur_rows.append(r)
            cur_len += extra
            j += 1

        if cur_rows:
            chunks.append((prefix + "\n".join(cur_rows)).strip())

        if j >= len(rows):
            break

        i = max(i + 1, j - overlap_rows)

    return [c for c in chunks if c]





