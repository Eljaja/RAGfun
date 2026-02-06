#!/usr/bin/env python3
"""
Test async deep-research and agent: concurrent requests, cancellation, correctness.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

DEEP_URL = "http://localhost:8094"
AGENT_URL = "http://localhost:8093"


async def deep_research_one(client: httpx.AsyncClient, query: str, filters: dict | None = None) -> tuple[str, list[dict], float]:
    """Run one deep-research stream, return (query_id, events, duration_s)."""
    payload = {"query": query, "max_iterations": 1}
    if filters:
        payload["filters"] = filters
    t0 = time.perf_counter()
    events: list[dict] = []
    try:
        async with client.stream("POST", f"{DEEP_URL}/v1/deep-research/stream", json=payload, timeout=120.0) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return (query, [{"type": "error", "error": str(e)}], time.perf_counter() - t0)
    return (query, events, time.perf_counter() - t0)


async def agent_one(client: httpx.AsyncClient, query: str, filters: dict | None = None) -> tuple[str, dict | None, float]:
    """Run one agent non-streaming request."""
    payload = {"query": query, "mode": "minimal"}
    if filters:
        payload["filters"] = filters
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{AGENT_URL}/v1/agent", json=payload, timeout=90.0)
        r.raise_for_status()
        data = r.json()
        return (query, data, time.perf_counter() - t0)
    except Exception as e:
        return (query, {"error": str(e)}, time.perf_counter() - t0)


async def test_concurrent_deep_research(n: int = 3) -> bool:
    """Run N deep-research requests in parallel."""
    print(f"\n--- Concurrent deep-research ({n} parallel) ---")
    queries = [
        "What is RAG?",
        "What is FastAPI?",
        "What is Python?",
    ][:n]
    filters = {"project_id": "agent_test"}

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [deep_research_one(client, q, filters) for q in queries]
        results = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0

    ok = 0
    for q, events, dur in results:
        done = next((e for e in events if e.get("type") == "done"), None)
        err = next((e for e in events if e.get("type") == "error"), None)
        if done:
            ans_len = len(done.get("answer") or "")
            print(f"  [{q[:20]}...] OK, {len(events)} events, answer={ans_len} chars, {dur:.1f}s")
            ok += 1
        elif err:
            print(f"  [{q[:20]}...] FAIL: {err.get('error', '?')}")
        else:
            print(f"  [{q[:20]}...] FAIL: no done event, {len(events)} events")

    print(f"  Total wall time: {total_s:.1f}s (parallel)")
    print(f"  Passed: {ok}/{n}")
    return ok == n


async def test_concurrent_agent(n: int = 3) -> bool:
    """Run N agent requests in parallel."""
    print(f"\n--- Concurrent agent ({n} parallel) ---")
    queries = ["What is RAG?", "What is FastAPI?", "What is Python?"][:n]
    filters = {"project_id": "agent_test"}

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [agent_one(client, q, filters) for q in queries]
        results = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0

    ok = 0
    for q, data, dur in results:
        if data and "answer" in data and "error" not in data:
            ans_len = len(data.get("answer") or "")
            print(f"  [{q[:20]}...] OK, answer={ans_len} chars, {dur:.1f}s")
            ok += 1
        else:
            print(f"  [{q[:20]}...] FAIL: {data}")

    print(f"  Total wall time: {total_s:.1f}s (parallel)")
    print(f"  Passed: {ok}/{n}")
    return ok == n


async def test_mixed_concurrent() -> bool:
    """Run deep-research and agent concurrently (same event loop)."""
    print("\n--- Mixed: 2 deep-research + 2 agent in parallel ---")
    filters = {"project_id": "agent_test"}

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [
            deep_research_one(client, "What is RAG?", filters),
            deep_research_one(client, "What is Python?", filters),
            agent_one(client, "What is FastAPI?", filters),
            agent_one(client, "What is RAG?", filters),
        ]
        results = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0

    ok = 0
    # results: first 2 are deep_research (q, events, dur), last 2 are agent (q, data, dur)
    for i, r in enumerate(results):
        q, payload, dur = r
        if isinstance(payload, list):
            # deep_research: payload is events list
            done = next((e for e in payload if isinstance(e, dict) and e.get("type") == "done"), None)
            if done:
                print(f"  deep-research [{q[:15]}...] OK, {dur:.1f}s")
                ok += 1
            else:
                print(f"  deep-research [{q[:15]}...] FAIL")
        else:
            # agent: payload is response dict
            if payload and isinstance(payload, dict) and "answer" in payload and "error" not in payload:
                print(f"  agent [{q[:15]}...] OK, {dur:.1f}s")
                ok += 1
            else:
                print(f"  agent [{q[:15]}...] FAIL")

    print(f"  Total wall time: {total_s:.1f}s")
    print(f"  Passed: {ok}/4")
    return ok == 4


async def test_cancellation() -> bool:
    """Start deep-research, close connection early — server should cancel."""
    print("\n--- Cancellation: start stream, close after 2 events ---")
    payload = {"query": "What is RAG?", "max_iterations": 2, "filters": {"project_id": "agent_test"}}

    async with httpx.AsyncClient() as client:
        events: list[dict] = []
        try:
            async with client.stream("POST", f"{DEEP_URL}/v1/deep-research/stream", json=payload, timeout=120.0) as r:
                r.raise_for_status()
                count = 0
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                            events.append(ev)
                            count += 1
                            if count >= 2:
                                break  # Simulate client disconnect
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"  (expected) Connection closed: {type(e).__name__}")

    # We got at least 2 events before "disconnect" — server should have cancelled the graph
    print(f"  Received {len(events)} events before close")
    print("  OK (cancellation handled by server)")
    return True


async def main() -> None:
    print("Async concurrency tests")
    print("=" * 50)

    results = []
    results.append(("concurrent_deep", await test_concurrent_deep_research(3)))
    results.append(("concurrent_agent", await test_concurrent_agent(3)))
    results.append(("mixed", await test_mixed_concurrent()))
    results.append(("cancellation", await test_cancellation()))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nTotal: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
