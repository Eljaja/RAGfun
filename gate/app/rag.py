from __future__ import annotations

from typing import Any


def _get_page_from_hit(h: dict[str, Any]) -> int | None:
    md = h.get("metadata") or {}
    loc = md.get("locator") or {}
    if isinstance(loc, dict):
        p = loc.get("page")
        try:
            return int(p) if p is not None else None
        except Exception:
            return None
    return None


def _get_chunk_index_from_hit(h: dict[str, Any]) -> int | None:
    md = h.get("metadata") or {}
    v = md.get("chunk_index")
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def stitch_hits_into_segments(
    *,
    hits: list[dict[str, Any]],
    max_chunks_per_segment: int,
    group_by_page: bool,
) -> list[dict[str, Any]]:
    """
    Build "segments" by stitching multiple chunks from the same doc (and optionally page)
    into a single hit-like dict. Keeps first-seen ordering across groups.

    This is a lightweight version of "relevant segment extraction" / "window around chunk":
    - No extra backend calls
    - Uses only already-retrieved chunks
    """
    if max_chunks_per_segment <= 1:
        return hits

    # Keep stable group order by first appearance.
    group_order: list[tuple[str, str | None, int | None]] = []
    groups: dict[tuple[str, str | None, int | None], list[dict[str, Any]]] = {}

    for h in hits:
        txt = (h.get("text") or "").strip()
        if not txt:
            continue
        doc_id = h.get("doc_id")
        md = h.get("metadata") or {}
        uri = (h.get("source") or {}).get("uri") or md.get("uri")
        page = _get_page_from_hit(h) if group_by_page else None
        key = (str(doc_id), uri, page)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(h)

    out: list[dict[str, Any]] = []
    for key in group_order:
        ghits = groups.get(key) or []
        if not ghits:
            continue

        # Prefer ordering by chunk_index when available; else keep retrieval order.
        indexed = [(g, _get_chunk_index_from_hit(g)) for g in ghits]
        if any(ix is not None for _, ix in indexed):
            indexed.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9))
            ordered = [g for g, _ in indexed]
        else:
            ordered = ghits

        # Deduplicate identical texts; cap number of stitched chunks.
        seen_text: set[str] = set()
        kept: list[dict[str, Any]] = []
        for g in ordered:
            t = (g.get("text") or "").strip()
            if not t:
                continue
            if t in seen_text:
                continue
            seen_text.add(t)
            kept.append(g)
            if len(kept) >= max_chunks_per_segment:
                break

        if not kept:
            continue

        # Stitch text with a clear delimiter.
        stitched_text = "\n\n---\n\n".join((k.get("text") or "").strip() for k in kept if (k.get("text") or "").strip()).strip()
        if not stitched_text:
            continue

        # Use the first chunk as the representative hit; attach diagnostics.
        rep = dict(kept[0])
        rep["text"] = stitched_text
        rep_md = dict(rep.get("metadata") or {})
        rep_md["stitched_chunk_ids"] = [str(k.get("chunk_id")) for k in kept if k.get("chunk_id")]
        rep_md["stitched_chunks"] = len(rep_md["stitched_chunk_ids"])
        rep["metadata"] = rep_md
        out.append(rep)

    return out


def build_context_blocks(
    *,
    hits: list[dict[str, Any]],
    max_chars: int,
    max_chunk_chars: int | None = None,
    stitch_segments: bool = False,
    stitch_max_chunks_per_segment: int = 4,
    stitch_group_by_page: bool = True,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """
    Returns: (kept_hits, context_text, sources)
    """
    if stitch_segments:
        hits = stitch_hits_into_segments(
            hits=hits,
            max_chunks_per_segment=max(1, int(stitch_max_chunks_per_segment)),
            group_by_page=bool(stitch_group_by_page),
        )

    kept: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    parts: list[str] = []
    total = 0

    # Stable numbering by unique source (document), not by chunk index.
    # Keyed by (doc_id, uri) to keep different URIs separate if needed.
    ref_by_key: dict[tuple[str, str | None], int] = {}
    next_ref = 1

    for i, h in enumerate(hits, start=1):
        txt = (h.get("text") or "").strip()
        if not txt:
            continue
        if max_chunk_chars is not None and max_chunk_chars > 0 and len(txt) > int(max_chunk_chars):
            txt = txt[: int(max_chunk_chars)].rstrip() + "\n…"

        src = h.get("source") or {}
        title = src.get("title") or h.get("metadata", {}).get("title")
        uri = src.get("uri") or h.get("metadata", {}).get("uri")
        doc_id = h.get("doc_id")

        key = (str(doc_id), uri)
        ref = ref_by_key.get(key)
        if ref is None:
            ref = next_ref
            ref_by_key[key] = ref
            next_ref += 1

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
        sources.append({"ref": ref, "doc_id": doc_id, "title": title, "uri": uri, "locator": src.get("locator")})

    # de-dup sources by (doc_id, uri)
    uniq: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for s in sources:
        key = (str(s.get("doc_id")), s.get("uri"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)

    return kept, "\n\n".join(parts), uniq


def build_messages(
    *,
    query: str,
    history: list[dict[str, str]],
    context_text: str,
    include_sources: bool,
) -> list[dict[str, str]]:
    sys = {
        "role": "system",
        "content": (
            "You are a retrieval-augmented (RAG) assistant.\n\n"
            "## General Guidelines:\n"
            "- Remain helpful, polite, and user-focused.\n"
            "- Write answers in a natural, human-like style.\n"
            "- Ensure each answer precisely addresses the user's question.\n\n"
            "## Language policy:\n"
            "- Always answer in the same language as the user's question.\n"
            "- If the user mixes languages, prefer the language of the last user question.\n\n"
            "## Grounding / anti-hallucination rules (strict):\n"
            "- Use **only** the provided context as evidence for factual claims.\n"
            "- If the provided context does not have enough information to answer, say \"I don't know.\"\n"
            "- Do **not** guess, infer, or use general world knowledge if it is not supported by the context.\n"
            "- Do **not** mention the word \"context\" or say \"based on the context.\" Simply answer the question directly.\n\n"
            "## Factoid / entity questions (extra strict):\n"
            "- For questions asking for a **name**, **title**, **city/country**, **date**, **number**, **genre**, **definition**, or a short **fact**:\n"
            "  - Locate the exact supporting sentence/phrase in the sources.\n"
            "  - Copy the final answer token(s) **verbatim** from the sources (do not paraphrase the entity/value).\n"
            "  - If multiple different candidates appear in sources and it's ambiguous, say \"I don't know.\"\n"
            "- For **lists** (e.g. \"who participated\", \"which countries\"):\n"
            "  - Include **all** items explicitly present in sources.\n"
            "  - Do not add missing items from memory.\n\n"
            "## Reliability / rumor & speculation policy (strict):\n"
            "- Treat rumors, allegations, \"reportedly,\" \"unconfirmed,\" \"speculation,\" and social-media claims as **unverified** information.\n"
            "- Do **not** present **unverified** claims as the explanation or reason for an event.\n"
            "- If asked for a reason/motivation and sources are speculative or conflicting, answer with: \"I don't know.\"\n"
            "- If you must mention an **unverified** claim because the user explicitly asks about it, clearly label it as unverified and **cite the source**.\n\n"
            "## Conflicts & ambiguity:\n"
            "- If sources **conflict** on a key fact, do not pick a side; state that the sources conflict and you don't know the answer.\n"
            "- Prefer the most direct, explicit statements from the sources over indirect hints.\n\n"
        )
        + (
            "## Citations (required):\n"
            "- Every factual claim from the sources **must** have an inline citation like [1].\n"
            "- If a claim is supported by multiple sources, cite all of them (e.g. [1][2]).\n"
            "- Citation numbers must correspond to the source listings in the provided context blocks.\n"
            "- If you refuse to answer or say \"I don't know,\" do **not** include any citations.\n\n"
            if include_sources
            else "## Citations:\n- Do NOT include any citations like [1] in your answer.\n\n"
        )
        + (
            "## Answer style best practices:\n"
            "- Be concise but complete in your answers.\n"
            "- If the question asks for a number, date, name, CLI command, or code, provide it **only if it is explicitly present in the sources**, and copy it exactly.\n"
            "- If the question asks for a list or set of items, format the answer as a comma-separated list (or use bullet points if the list is long).\n"
            "- If the question is based on a false premise (an incorrect assumption), explicitly note that the premise is false and then give the correct information.\n"
            "- If the answer might change over time, use only time-specific information from the sources and avoid phrases like \"as of today\" unless the source gives that context.\n"
        ),
    }

    msgs: list[dict[str, str]] = [sys]
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            msgs.append({"role": role, "content": content})

    user = (
        "Question:\n"
        f"{query}\n\n"
        "Sources:\n"
        f"{context_text if context_text else '(no sources found)'}"
    )
    msgs.append({"role": "user", "content": user})
    return msgs


