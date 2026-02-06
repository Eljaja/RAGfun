import os

import httpx


RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")


def test_retrieval_readyz_reports_deps():
    """
    Contract test: /v1/readyz returns deps visibility.
    This is useful for bottleneck triage (which backend is down / degraded).
    """
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{RETRIEVAL_BASE_URL}/v1/readyz")
        # readiness may be 503 if both backends are down, but JSON contract must hold.
        assert r.status_code in (200, 503), r.text
        j = r.json()
        assert "ready" in j, j
        assert "deps" in j and isinstance(j["deps"], dict), j

















