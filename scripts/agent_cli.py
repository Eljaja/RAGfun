#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
    "blue": "\033[34m",
}


MEME_GRUMPS = [
    "Sigh. Fine. I will do science.",
    "This better be worth the tokens.",
    "I am not mad. I am just... disappointed in entropy.",
    "Okay, okay, I will carry this search. Again.",
    "One more query and I start charging by the sigh.",
]


def _c(text: str, color: str) -> str:
    return f"{ANSI.get(color, '')}{text}{ANSI['reset']}"


def _banner() -> None:
    logo = r"""
   ___                    _       ____ _     ___ 
  / _ \__ _ _ __ ___  __ _| |_    / ___| |   |_ _|
 / /_\/ _` | '__/ _ \/ _` | __|  | |   | |    | | 
/ /_\\ (_| | | |  __/ (_| | |_   | |___| |___ | | 
\____/\__,_|_|  \___|\__,_|\__|   \____|_____|___|
"""
    print(_c(logo.rstrip("\n"), "magenta"))
    print(_c("Agentic Retrieval CLI", "bold") + _c("  (SSE ready, mildly grumpy)", "dim"))
    print(_c("-" * 62, "dim"))


def _load_env(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            data[key] = val
    return data


def _env_get(env: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or env.get(key) or default


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _stream_openai(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 60.0) -> Iterable[str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            yield data


def _tool_call(name: str, payload: dict[str, Any]) -> None:
    print(_c("[tool]", "cyan"), _c(name, "bold"))
    print(_c(textwrap.indent(json.dumps(payload, ensure_ascii=True, indent=2), "  "), "dim"))


def _thought(label: str, text: str) -> None:
    print(_c("[thought]", "yellow"), _c(label + ":", "bold"), text)


def _grump() -> None:
    print(_c("[mood]", "magenta"), random.choice(MEME_GRUMPS))


def _pretty_hits(hits: list[dict[str, Any]], limit: int = 6) -> None:
    if not hits:
        print(_c("  (no hits)", "dim"))
        return
    for i, h in enumerate(hits[:limit], start=1):
        doc_id = str(h.get("doc_id") or "-")
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        text = (h.get("text") or "").replace("\n", " ").strip()
        snippet = text[:160] + ("..." if len(text) > 160 else "")
        print(_c(f"  {i:>2}.", "dim"), _c(doc_id, "green"), _c(f"score={score_s}", "dim"))
        if snippet:
            print(_c("      " + snippet, "dim"))


def _build_context(hits: list[dict[str, Any]], limit: int = 8, max_chars: int = 4000) -> str:
    blocks: list[str] = []
    total = 0
    for i, h in enumerate(hits[:limit], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(h.get("doc_id") or "-")
        score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        # Keep each block small so we never drop the whole context due to one huge chunk.
        max_block = max(300, min(1200, max_chars - total))
        if len(text) > max_block:
            text = text[: max_block - 3].rstrip() + "..."
        block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        if total + len(block) > max_chars:
            # Trim again to fit remaining space.
            remaining = max(0, max_chars - total - 60)
            if remaining <= 0:
                break
            text = text[:remaining].rstrip() + "..."
            block = f"[{i}] doc_id={doc_id} score={score_s}\n{text}"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _quality_is_poor(resp: dict[str, Any]) -> bool:
    hits = list(resp.get("hits") or [])
    if not hits:
        return True
    if len(hits) < 3:
        return True
    if resp.get("partial") or (resp.get("degraded") or []):
        return True
    top = hits[0]
    score = top.get("rerank_score") if top.get("rerank_score") is not None else top.get("score")
    if isinstance(score, (int, float)) and score < 0.15:
        return True
    return False


def _merge_hits(responses: list[dict[str, Any]], cap: int = 20) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for r in responses:
        for h in r.get("hits") or []:
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            score = h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else 0.0
            if cid not in merged or score_f > scores.get(cid, -1.0):
                merged[cid] = h
                scores[cid] = score_f
    out = sorted(merged.values(), key=lambda h: scores.get(str(h.get("chunk_id") or ""), 0.0), reverse=True)
    return out[:cap]


def _llm_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _llm_chat(
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    data = _post_json(url, payload, headers=_llm_headers(api_key), timeout=timeout)
    return str(data["choices"][0]["message"]["content"])


def _llm_chat_stream(
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> Iterable[str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
    for line in _stream_openai(url, payload, headers=_llm_headers(api_key), timeout=timeout):
        try:
            data = json.loads(line)
        except Exception:
            continue
        try:
            delta = data["choices"][0]["delta"].get("content")
        except Exception:
            delta = None
        if delta:
            yield str(delta)


def _plan_retrieval(llm_base: str, llm_model: str, llm_key: str | None, query: str) -> dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You are a retrieval strategist for a RAG system. "
            "Return a single JSON object only. Keep 'reason' short."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "Decide per-request retrieval knobs.\n"
            "JSON fields: retrieval_mode (bm25|vector|hybrid), top_k (1..40), "
            "rerank (true/false), use_hyde (true/false), reason.\n"
            f"Query: {query}"
        ),
    }
    _tool_call("llm.plan", {"model": llm_model, "query": query})
    raw = _llm_chat(llm_base, llm_model, llm_key, [system, user], temperature=0.0)
    try:
        return json.loads(raw)
    except Exception:
        return {"retrieval_mode": "hybrid", "top_k": 8, "rerank": True, "use_hyde": False, "reason": "fallback"}


def _make_hyde(llm_base: str, llm_model: str, llm_key: str | None, query: str) -> str:
    system = {"role": "system", "content": "Write a short hypothetical answer passage for retrieval. English only."}
    user = {"role": "user", "content": f"Query: {query}\nReturn a 3-5 sentence passage."}
    _tool_call("llm.hyde", {"model": llm_model, "query": query})
    return _llm_chat(llm_base, llm_model, llm_key, [system, user], temperature=0.2)


def _fact_queries(llm_base: str, llm_model: str, llm_key: str | None, query: str) -> list[str]:
    system = {"role": "system", "content": "Extract fact-oriented sub-queries from the user request."}
    user = {
        "role": "user",
        "content": (
            "Return JSON: {\"fact_queries\": [..]} with 2-3 short queries.\n"
            f"Query: {query}"
        ),
    }
    _tool_call("llm.fact_split", {"model": llm_model, "query": query})
    raw = _llm_chat(llm_base, llm_model, llm_key, [system, user], temperature=0.2)
    try:
        data = json.loads(raw)
        out = data.get("fact_queries") or []
        return [str(q).strip() for q in out if str(q).strip()]
    except Exception:
        return []


def _gate_search(base_url: str, query: str, mode: str, top_k: int, rerank: bool) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat"
    payload = {
        "query": query,
        "history": [],
        "retrieval_mode": mode,
        "top_k": int(top_k),
        "rerank": bool(rerank),
        "include_sources": True,
    }
    _tool_call("gate.chat", payload)
    try:
        data = _post_json(url, payload, timeout=60.0)
    except urllib.error.URLError as e:
        raise RuntimeError(f"gate_unreachable: {url}") from e

    retrieval = data.get("retrieval") or {}
    hits = list(retrieval.get("hits") or [])
    context_chunks = list(data.get("context") or [])
    if context_chunks:
        by_cid = {str(c.get("chunk_id")): c for c in context_chunks if c.get("chunk_id")}
        for h in hits:
            if h.get("text"):
                continue
            cid = str(h.get("chunk_id") or "")
            c = by_cid.get(cid)
            if c:
                h["text"] = c.get("text")
                h["source"] = c.get("source")
                h["score"] = h.get("score") if h.get("score") is not None else c.get("score")
    if not hits and context_chunks:
        # Fallback: map context chunks to search-like hits.
        hits = [
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "score": c.get("score"),
                "text": c.get("text"),
                "source": c.get("source"),
            }
            for c in context_chunks
        ]
    return {
        "ok": bool(retrieval.get("ok", True)),
        "mode": retrieval.get("mode", mode),
        "partial": bool(retrieval.get("partial")),
        "degraded": retrieval.get("degraded") or [],
        "hits": hits,
        "context": context_chunks,
    }


def _answer_stream(
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    query: str,
    context: str,
    stream: bool = True,
) -> None:
    system = {
        "role": "system",
        "content": (
            "You answer using the provided context only. "
            "If the context is insufficient, say what is missing."
        ),
    }
    user = {
        "role": "user",
        "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer in English.",
    }
    _tool_call("llm.answer", {"model": llm_model, "stream": stream})
    print(_c("\nAnswer:", "bold"))
    if not stream:
        text = _llm_chat(llm_base, llm_model, llm_key, [system, user], temperature=0.2)
        print(text)
        return
    for chunk in _llm_chat_stream(llm_base, llm_model, llm_key, [system, user], temperature=0.2):
        print(chunk, end="", flush=True)
    print()


def _interactive_loop(
    gate_url: str,
    llm_base: str,
    llm_model: str,
    llm_key: str | None,
    stream: bool,
) -> None:
    print(_c("Type your question. Commands: /exit, /help, /clear", "dim"))
    while True:
        try:
            query = input(_c("\nrag-agent> ", "blue")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in ("/exit", "/quit"):
            break
        if query == "/help":
            print(_c("Commands:", "bold"))
            print("  /exit  - quit")
            print("  /clear - clear screen")
            continue
        if query == "/clear":
            os.system("clear" if os.name != "nt" else "cls")
            _banner()
            continue

        _grump()
        plan = _plan_retrieval(llm_base, llm_model, llm_key, query)
        mode = str(plan.get("retrieval_mode") or "hybrid")
        top_k = int(plan.get("top_k") or 8)
        rerank = bool(plan.get("rerank") if plan.get("rerank") is not None else True)
        use_hyde = bool(plan.get("use_hyde") or False)
        reason = str(plan.get("reason") or "no_reason")
        _thought("Plan", f"mode={mode}, top_k={top_k}, rerank={rerank}, hyde={use_hyde}. Reason: {reason}")

        search_query = query
        if use_hyde:
            _thought("HyDE", "Generating a hypothetical passage for better recall.")
            hyde = _make_hyde(llm_base, llm_model, llm_key, query)
            search_query = hyde.strip() or query

        try:
            resp = _gate_search(gate_url, search_query, mode, top_k, rerank)
        except RuntimeError as e:
            print(_c(f"\nGate error: {e}", "red"))
            print(_c("Hint: ensure rag-gate is reachable (default: http://localhost:8090).", "dim"))
            continue
        hits = list(resp.get("hits") or [])
        print(_c("\nTop hits:", "bold"))
        _pretty_hits(hits, limit=6)

        if _quality_is_poor(resp):
            _thought("Quality", "Search looks weak. Splitting into fact queries.")
            fact_qs = _fact_queries(llm_base, llm_model, llm_key, query)
            if fact_qs:
                responses: list[dict[str, Any]] = [resp]
                for fq in fact_qs:
                    r2 = _gate_search(gate_url, fq, mode, max(4, top_k // 2), rerank)
                    responses.append(r2)
                hits = _merge_hits(responses, cap=max(12, top_k))
                print(_c("\nMerged hits (fact queries):", "bold"))
                _pretty_hits(hits, limit=6)
            else:
                _thought("Quality", "No useful fact queries found. Carrying on.")

        context = _build_context(hits, limit=8, max_chars=4000)
        if not context and resp.get("context"):
            context = _build_context(list(resp.get("context") or []), limit=8, max_chars=4000)
        if not context:
            print(_c("\nNo context available to answer.", "red"))
            continue

        try:
            _answer_stream(llm_base, llm_model, llm_key, query, context, stream=stream)
        except urllib.error.HTTPError as e:
            print(_c(f"\nLLM error: {e}", "red"))
        except Exception as e:
            print(_c(f"\nLLM error: {type(e).__name__}: {e}", "red"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic Retrieval CLI")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--gate-url", default=None, help="Gate base URL (e.g. http://localhost:8090)")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL (ends with /v1)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    parser.add_argument("--no-stream", action="store_true", help="Disable SSE streaming")
    args = parser.parse_args()

    env = _load_env(args.env)
    gate_url = (
        args.gate_url
        or _env_get(env, "GATE_URL")
        or _env_get(env, "GATE_BASE_URL")
        or "http://localhost:8090"
    )
    llm_base = (
        args.llm_base_url
        or _env_get(env, "GATE_LLM_BASE_URL")
        or _env_get(env, "OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    llm_model = (
        args.llm_model
        or _env_get(env, "GATE_LLM_MODEL")
        or _env_get(env, "OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    llm_key = args.llm_api_key or _env_get(env, "GATE_LLM_API_KEY") or _env_get(env, "OPENAI_API_KEY")

    _banner()
    print(_c("Config:", "bold"))
    print(f"  gate_url      = {gate_url}")
    print(f"  llm_base_url  = {llm_base}")
    print(f"  llm_model     = {llm_model}")
    print(f"  llm_api_key   = {'set' if llm_key else 'missing'}")
    print(_c("-" * 62, "dim"))

    _interactive_loop(gate_url, llm_base, llm_model, llm_key, stream=not args.no_stream)
    print(_c("\nBye. I will be in my server rack, judging.", "dim"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
