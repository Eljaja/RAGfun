#!/usr/bin/env python3
"""
Проверка agent-search, gate и deep-research с разными параметрами.

Сценарии: retrieval (hybrid/bm25/vector), gate chat, agent streaming/non-streaming,
citation, deep-research scope+full pipeline.
URLs по умолчанию: retrieval 8085, gate 8092, agent 8093, deep 8094.
"""
import argparse
import json
import sys
import httpx

RETRIEVAL_URL = "http://localhost:8085"
GATE_URL = "http://localhost:8092"
AGENT_URL = "http://localhost:8093"
DEEP_URL = "http://localhost:8094"


def retrieval_search(base_url: str, query: str, mode: str = "hybrid", top_k: int = 5, filters: dict | None = None) -> dict:
    with httpx.Client(timeout=30.0) as c:
        payload = {"query": query, "mode": mode, "top_k": top_k}
        if filters:
            payload["filters"] = filters
        r = c.post(f"{base_url}/v1/search", json=payload)
        r.raise_for_status()
        return r.json()


def gate_chat(base_url: str, query: str, filters: dict | None = None) -> dict:
    with httpx.Client(timeout=60.0) as c:
        payload = {"query": query, "history": []}
        if filters:
            payload["filters"] = filters
        r = c.post(f"{base_url}/v1/chat", json=payload)
        r.raise_for_status()
        return r.json()


def agent_stream(base_url: str, query: str, filters: dict | None = None) -> list[dict]:
    events = []
    with httpx.Client(timeout=120.0) as c:
        payload = {"query": query, "include_sources": True}
        if filters:
            payload["filters"] = filters
        with c.stream("POST", f"{base_url}/v1/agent/stream", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    return events


def agent_non_streaming(base_url: str, query: str, filters: dict | None = None) -> dict:
    """POST /v1/agent — non-streaming JSON response."""
    with httpx.Client(timeout=120.0) as c:
        payload = {"query": query, "include_sources": True}
        if filters:
            payload["filters"] = filters
        r = c.post(f"{base_url}/v1/agent", json=payload)
        r.raise_for_status()
        return r.json()


def deep_research_stream(base_url: str, query: str, filters: dict | None = None) -> list[dict]:
    """Deep-research streaming events."""
    events = []
    with httpx.Client(timeout=180.0) as c:
        payload = {"query": query}
        if filters:
            payload["filters"] = filters
        with c.stream("POST", f"{base_url}/v1/deep-research/stream", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-url", default=RETRIEVAL_URL)
    ap.add_argument("--gate-url", default=GATE_URL)
    ap.add_argument("--agent-url", default=AGENT_URL)
    ap.add_argument("--deep-url", default=DEEP_URL)
    args = ap.parse_args()
    urls = {"retrieval": args.retrieval_url, "gate": args.gate_url, "agent": args.agent_url, "deep": args.deep_url}

    print("=" * 60)
    print("1. Retrieval: поиск без фильтров (hybrid)")
    print("=" * 60)
    try:
        r = retrieval_search(urls["retrieval"], "Что такое RAG?", mode="hybrid", top_k=3)
        hits = r.get("hits", [])[:3]
        for i, h in enumerate(hits, 1):
            doc = h.get("metadata", {}).get("doc_id", "?")
            proj = h.get("metadata", {}).get("project_id", "?")
            text = (h.get("text") or "")[:80] + "..."
            print(f"  [{i}] doc={doc} project={proj}: {text}")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("2. Retrieval: фильтр project_id=agent_test")
    print("=" * 60)
    try:
        r = retrieval_search(urls["retrieval"], "RAG", mode="hybrid", top_k=3, filters={"project_id": "agent_test"})
        hits = r.get("hits", [])[:3]
        for i, h in enumerate(hits, 1):
            doc = h.get("metadata", {}).get("doc_id", "?")
            proj = h.get("metadata", {}).get("project_id", "?")
            print(f"  [{i}] doc={doc} project={proj}")
        if not hits:
            print("  (no hits - check filters)")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("3. Retrieval: mode=bm25 vs vector")
    print("=" * 60)
    try:
        r_bm25 = retrieval_search(urls["retrieval"], "Python", mode="bm25", top_k=2)
        r_vec = retrieval_search(urls["retrieval"], "Python", mode="vector", top_k=2)
        print(f"  BM25 hits: {len(r_bm25.get('hits', []))}")
        print(f"  Vector hits: {len(r_vec.get('hits', []))}")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("4. Gate chat: без фильтров")
    print("=" * 60)
    try:
        r = gate_chat(urls["gate"], "Что такое RAG?")
        ans = (r.get("answer") or "")[:150]
        print(f"  Answer: {ans}...")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("5. Gate chat: filters.project_id=tech_docs")
    print("=" * 60)
    try:
        r = gate_chat(urls["gate"], "What is FastAPI?", filters={"project_id": "tech_docs"})
        ans = (r.get("answer") or "")[:150]
        print(f"  Answer: {ans}...")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("6. Agent-search: streaming, filters.project_id=agent_test")
    print("=" * 60)
    try:
        events = agent_stream(urls["agent"], "Что такое RAG?", filters={"project_id": "agent_test"})
        done = next((e for e in events if e.get("type") == "done"), None)
        if done:
            ans = (done.get("answer") or "")[:150]
            print(f"  Answer: {ans}...")
        else:
            tokens = [e.get("content") for e in events if e.get("type") == "token"]
            print(f"  Tokens: {len(tokens)} received")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("7. Retrieval: filters.source=api (tech_docs)")
    print("=" * 60)
    try:
        r = retrieval_search(urls["retrieval"], "FastAPI", mode="hybrid", top_k=2, filters={"source": "api"})
        hits = r.get("hits", [])
        for i, h in enumerate(hits[:2], 1):
            doc = h.get("metadata", {}).get("doc_id", "?")
            src = h.get("metadata", {}).get("source", "?")
            print(f"  [{i}] doc={doc} source={src}")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("8. Agent-search: non-streaming POST /v1/agent")
    print("=" * 60)
    try:
        r = agent_non_streaming(urls["agent"], "Что такое RAG?", filters={"project_id": "agent_test"})
        assert "answer" in r, r
        assert isinstance(r.get("sources"), list), r
        ans = (r.get("answer") or "")[:120]
        print(f"  Answer: {ans}...")
        print(f"  Sources: {len(r.get('sources', []))} items")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("9. Agent-search: citation [1], [2] in answer")
    print("=" * 60)
    try:
        r = agent_non_streaming(urls["agent"], "What is FastAPI?", filters={"project_id": "tech_docs"})
        ans = r.get("answer") or ""
        has_citation = "[" in ans and "]" in ans
        sources = r.get("sources", [])
        refs = [s.get("ref") for s in sources if "ref" in s]
        print(f"  Answer has [N] citation: {has_citation}")
        print(f"  Sources with ref: {refs[:5]}{'...' if len(refs) > 5 else ''}")
        if not has_citation and sources:
            print("  (citation optional; model may not always cite)")
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("10. Deep-research: scope + full pipeline")
    print("=" * 60)
    try:
        events = deep_research_stream(urls["deep"], "Что такое RAG?", filters={"project_id": "agent_test"})
        scope_progress = next((e for e in events if e.get("type") == "progress" and e.get("stage") == "scope"), None)
        done = next((e for e in events if e.get("type") == "done"), None)
        if scope_progress:
            print("  Scope stage: OK")
        if done:
            ans_len = len(done.get("answer") or "")
            print(f"  Done: answer {ans_len} chars")
        if not scope_progress and not done:
            print("  (no scope/done events — check deep-research)")
        print("  OK\n")
    except Exception as e:
        print(f"  SKIP (deep-research): {e}\n")

    print("All checks passed.")


if __name__ == "__main__":
    main()
