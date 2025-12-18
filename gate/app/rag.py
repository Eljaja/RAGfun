from __future__ import annotations

from typing import Any


def build_context_blocks(*, hits: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """
    Returns: (kept_hits, context_text, sources)
    """
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


def build_messages(*, query: str, history: list[dict[str, str]], context_text: str) -> list[dict[str, str]]:
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
            "## Reliability / rumor & speculation policy (strict):\n"
            "- Treat rumors, allegations, \"reportedly,\" \"unconfirmed,\" \"speculation,\" and social-media claims as **unverified** information.\n"
            "- Do **not** present **unverified** claims as the explanation or reason for an event.\n"
            "- If asked for a reason/motivation and sources are speculative or conflicting, answer with: \"I don't know.\"\n"
            "- If you must mention an **unverified** claim because the user explicitly asks about it, clearly label it as unverified and **cite the source**.\n\n"
            "## Conflicts & ambiguity:\n"
            "- If sources **conflict** on a key fact, do not pick a side; state that the sources conflict and you don't know the answer.\n"
            "- Prefer the most direct, explicit statements from the sources over indirect hints.\n\n"
            "## Citations (required):\n"
            "- Every factual claim from the sources **must** have an inline citation like [1].\n"
            "- If a claim is supported by multiple sources, cite all of them (e.g. [1][2]).\n"
            "- Citation numbers must correspond to the source listings in the provided context blocks.\n"
            "- If you refuse to answer or say \"I don't know,\" do **not** include any citations.\n\n"
            "## Answer style best practices:\n"
            "- Be concise but complete in your answers.\n"
            "- If the question asks for a number, date, or name, provide the exact value in the answer.\n"
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


