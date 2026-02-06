#!/usr/bin/env python3
"""Быстрый smoke-тест RAG: 1 док → индекс → 1 поиск. ~15 сек."""
import sys
import httpx

URL = "http://localhost:8085"

def main():
    print("1. Health check...")
    r = httpx.get(f"{URL}/v1/healthz", timeout=10)
    r.raise_for_status()
    print("   OK")

    print("2. Index 1 doc...")
    r = httpx.post(
        f"{URL}/v1/index/upsert",
        json={
            "mode": "document",
            "document": {"doc_id": "smoke_test_1", "project_id": "smoke", "source": "test"},
            "text": "Python is a programming language. It is used for web development and data science.",
            "refresh": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    print("   OK")

    print("3. Search...")
    r = httpx.post(
        f"{URL}/v1/search",
        json={
            "query": "What is Python?",
            "mode": "hybrid",
            "top_k": 5,
            "filters": {"project_id": "smoke"},
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    hits = j.get("hits") or []
    print(f"   OK, {len(hits)} hits")
    if hits:
        print(f"   Top: {hits[0].get('doc_id')} (score={hits[0].get('score', 0):.3f})")

    print("\nSmoke test PASSED")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
