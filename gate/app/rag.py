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
            "Ты RAG-ассистент. Всегда отвечай на том же языке, на котором задан вопрос пользователя (определи язык по вопросу). "
            "Используй предоставленный контекст, а если его недостаточно — честно скажи. "
            "Не выдумывай факты. "
            "Не используй вводные фразы вроде «судя по контексту»/«по контексту» — отвечай сразу по сути.\n\n"
            "Правило цитирования:\n"
            "- Каждый тезис, который опирается на контекст, помечай ссылкой в квадратных скобках, например: [1].\n"
            "- Номер ссылки должен соответствовать номеру источника в заголовках контекста.\n"
            "- Если тезис опирается на несколько источников, укажи несколько ссылок: [1][2].\n"
            "- Если в контексте нет ответа — скажи, что не знаешь, и НЕ ставь ссылки."
        ),
    }

    msgs: list[dict[str, str]] = [sys]
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            msgs.append({"role": role, "content": content})

    user = (
        "Вопрос:\n"
        f"{query}\n\n"
        "Контекст:\n"
        f"{context_text if context_text else '(контекст не найден)'}"
    )
    msgs.append({"role": "user", "content": user})
    return msgs


