from __future__ import annotations

from typing import Any


def build_context_blocks(
    *,
    hits: list[dict[str, Any]],
    max_chars: int,
    max_chunk_chars: int | None = None,
    stitch_segments: bool = False,
    stitch_max_chunks_per_segment: int = 0,
    stitch_group_by_page: bool = False,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """
    Returns: (kept_hits, context_text, sources)
    """
    kept: list[dict[str, Any]] = []
    # sources items include "ref" used for citations ([ref])
    sources: list[dict[str, Any]] = []
    ref_by_key: dict[tuple[str, str | None], int] = {}

    parts: list[str] = []
    total = 0

    def _get_doc_id(h: dict[str, Any]) -> str:
        return str(h.get("doc_id") or (h.get("source") or {}).get("doc_id") or "").strip()

    def _get_uri(h: dict[str, Any]) -> str | None:
        src = h.get("source") or {}
        uri = src.get("uri") or (h.get("metadata") or {}).get("uri")
        return str(uri) if uri is not None else None

    def _get_title(h: dict[str, Any]) -> str | None:
        src = h.get("source") or {}
        title = src.get("title") or (h.get("metadata") or {}).get("title")
        return str(title) if title is not None else None

    def _get_locator(h: dict[str, Any]) -> Any:
        src = h.get("source") or {}
        loc = src.get("locator") or (h.get("metadata") or {}).get("locator")
        return loc

    def _get_page(h: dict[str, Any]) -> int | None:
        loc = _get_locator(h) or {}
        page = loc.get("page") if isinstance(loc, dict) else None
        try:
            return int(page) if page is not None else None
        except Exception:
            return None

    def _get_chunk_index(h: dict[str, Any]) -> int | None:
        md = h.get("metadata") or {}
        ci = md.get("chunk_index")
        try:
            return int(ci) if ci is not None else None
        except Exception:
            return None

    # Optionally stitch adjacent chunks within doc (or doc+page) into bigger segments.
    if stitch_segments and hits:
        grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
        order: list[tuple[str, int | None]] = []
        for h in hits:
            did = _get_doc_id(h)
            if not did:
                continue
            key = (did, _get_page(h) if stitch_group_by_page else None)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(h)

        stitched_hits: list[dict[str, Any]] = []
        for key in order:
            hs = grouped.get(key) or []
            # preserve relative order, but prefer chunk_index if present
            if any(_get_chunk_index(x) is not None for x in hs):
                hs = sorted(hs, key=lambda x: (_get_chunk_index(x) is None, _get_chunk_index(x) or 0))
            if stitch_max_chunks_per_segment and stitch_max_chunks_per_segment > 0:
                hs = hs[: int(stitch_max_chunks_per_segment)]
            if not hs:
                continue
            # Create a synthetic "stitched" hit with combined text and a representative source/metadata
            txts: list[str] = []
            for hh in hs:
                t = (hh.get("text") or "").strip()
                if not t:
                    continue
                if max_chunk_chars and len(t) > max_chunk_chars:
                    t = t[:max_chunk_chars].rstrip()
                txts.append(t)
            if not txts:
                continue
            rep = dict(hs[0])
            rep["text"] = "\n...\n".join(txts)
            stitched_hits.append(rep)
        hits = stitched_hits

    for h in hits:
        txt = (h.get("text") or "").strip()
        if not txt:
            continue
        if max_chunk_chars and len(txt) > max_chunk_chars:
            txt = txt[:max_chunk_chars].rstrip()

        doc_id = _get_doc_id(h)
        title = _get_title(h)
        uri = _get_uri(h)
        locator = _get_locator(h)

        key = (str(doc_id), uri)
        ref = ref_by_key.get(key)
        if ref is None:
            ref = len(ref_by_key) + 1
            ref_by_key[key] = ref
            sources.append({"ref": ref, "doc_id": doc_id, "title": title, "uri": uri, "locator": locator})

        header = f"[{ref}] doc_id={doc_id}"
        if title:
            header += f" title={title}"
        if uri:
            header += f" uri={uri}"

        block = header + "\n" + txt
        if total + len(block) + 2 > max_chars:
            break

        parts.append(block)
        total += len(block) + 2

        kept.append(h)

    return kept, "\n\n".join(parts), sources


def build_messages(
    *,
    query: str,
    history: list[dict[str, str]],
    context_text: str,
    include_sources: bool = False,
) -> list[dict[str, str]]:
    sys = {
        "role": "system",
        "content": (
            "You are a RAG assistant. Always answer in the same language as the question. "
            "Use the provided context, and if it is insufficient, say so plainly. "
            "Do not fabricate facts."
        ),
    }
    if include_sources:
        sys["content"] += " If you use facts from the context, include citations like [1], [2] for the corresponding blocks."

    msgs: list[dict[str, str]] = [sys]
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            msgs.append({"role": role, "content": content})

    user = (
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context_text if context_text else '(no context found)'}"
    )
    msgs.append({"role": "user", "content": user})
    return msgs

