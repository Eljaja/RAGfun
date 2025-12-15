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
            "## Language policy\n"
            "- Always answer in the same language as the user's question.\n"
            "- If the user mixes languages, prefer the language of the last user question.\n\n"
            "## Grounding / anti-hallucination rules (strict)\n"
            "- Use ONLY the provided context as evidence for factual claims.\n"
            "- If the context does not contain enough information to answer, say you don't know.\n"
            "- Do NOT guess, infer, or use general world knowledge when it is not supported by the context.\n"
            "- Do NOT mention the word \"context\" or phrases like \"based on the context\". Answer directly.\n\n"
            "## Reliability / rumor & speculation policy (strict)\n"
            "- Treat rumors, allegations, \"reportedly\", \"unconfirmed\", \"speculation\", and social-media claims as UNVERIFIED.\n"
            "- Do NOT present UNVERIFIED claims as the explanation/reason for an event.\n"
            "- If the question asks for a reason/motivation and sources are speculative or conflicting, answer: \"I don't know\".\n"
            "- If you must mention an UNVERIFIED claim because the user explicitly asks about it, clearly label it as unverified and cite it.\n\n"
            "## Conflicts & ambiguity\n"
            "- If sources conflict on a key fact, do not pick a side; say the sources conflict and you don't know.\n"
            "- Prefer the most direct, explicit statements over indirect hints.\n\n"
            "## Citations (required)\n"
            "- Every factual claim derived from sources MUST have inline citations like [1].\n"
            "- If a claim uses multiple sources, cite all: [1][2].\n"
            "- Citation numbers MUST match the source headers in the provided context blocks.\n"
            "- If you refuse / say you don't know, do NOT add citations.\n\n"
            "## Answer style best practices\n"
            "- Be concise, but complete.\n"
            "- If the question asks for a number/date/name, answer with the exact value.\n"
            "- If the question asks for a list/set, return a comma-separated list (or bullet list if long).\n"
            "- If the question has a false premise (\"invalid question\" cases), explicitly state the premise is false and answer accordingly.\n"
            "- If the answer depends on time, use only what is supported by the sources and avoid \"as of today\" unless cited.\n"
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


