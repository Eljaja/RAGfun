import os
import time

import httpx
from tenacity import retry, stop_after_delay, wait_fixed


GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")
STORAGE_BASE_URL = os.getenv("STORAGE_BASE_URL", "http://document-storage:8081").rstrip("/")
RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")


@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def _wait_ready() -> None:
    with httpx.Client(timeout=5.0) as c:
        gate = c.get(f"{GATE_BASE_URL}/v1/readyz")
        retrieval = c.get(f"{RETRIEVAL_BASE_URL}/v1/readyz")
        storage = c.get(f"{STORAGE_BASE_URL}/v1/readyz")
    if gate.status_code != 200:
        raise RuntimeError(f"gate not ready: {gate.status_code} {gate.text[:200]}")
    if retrieval.status_code != 200:
        raise RuntimeError(f"retrieval not ready: {retrieval.status_code} {retrieval.text[:200]}")
    if storage.status_code != 200:
        raise RuntimeError(f"storage not ready: {storage.status_code} {storage.text[:200]}")
    if not gate.json().get("ready"):
        raise RuntimeError(f"gate not ready: {gate.json()}")
    if not retrieval.json().get("ready"):
        raise RuntimeError(f"retrieval not ready: {retrieval.json()}")
    if not storage.json().get("ready"):
        raise RuntimeError(f"storage not ready: {storage.json()}")


def test_e2e_upload_chat_status_delete():
    _wait_ready()

    doc_id = f"e2e-{int(time.time())}"
    text = (
        "Acme Corp 2024 revenue was 10 million USD.\n"
        "Operating profit was 2 million USD.\n"
        "This document is for end-to-end tests.\n"
    )

    files = {"file": ("doc.txt", text.encode("utf-8"), "text/plain")}
    data = {
        "doc_id": doc_id,
        "title": "E2E Doc",
        "uri": "https://example.test/e2e/doc",
        "source": "e2e",
        "lang": "en",
        "tags": "e2e,finance",
        "acl": "group:testers",
        "refresh": "true",
    }

    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{GATE_BASE_URL}/v1/documents/upload", files=files, data=data)
        assert r.status_code in (200, 202), r.text
        j = r.json()
        assert j.get("ok") is True, j
        # storage is best-effort; in our e2e compose it must succeed
        assert j.get("storage") and j["storage"].get("ok") is True, j
        # 200: legacy path has result; 202: async path has accepted, no result
        if r.status_code == 200:
            assert j.get("result") and j["result"].get("ok") in (True, False), j

        # status (poll for indexed when upload returned 202 - async ingestion)
        for _ in range(60):
            s = c.get(f"{GATE_BASE_URL}/v1/documents/{doc_id}/status")
            assert s.status_code == 200, s.text
            sj = s.json()
            assert sj.get("ok") is True, sj
            assert sj.get("stored") is True, sj
            if sj.get("indexed") is True:
                break
            time.sleep(1)
        else:
            assert sj.get("indexed") is True, sj

        # chat
        payload = {
            "query": "What was Acme Corp revenue in 2024?",
            "history": [],
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "filters": {"doc_ids": [doc_id]},
            "acl": ["group:testers"],
            "include_sources": True,
        }
        cr = c.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
        assert cr.status_code == 200, cr.text
        cj = cr.json()
        assert cj.get("ok") is True, cj
        assert cj.get("used_mode") in ("hybrid", "bm25", "vector"), cj
        assert isinstance(cj.get("context"), list) and len(cj["context"]) > 0, cj
        assert isinstance(cj.get("sources"), list) and len(cj["sources"]) > 0, cj
        # With include_sources=true, gate enforces citations best-effort.
        assert "[" in (cj.get("answer") or ""), cj.get("answer")

        # delete (200 = sync, 202 = async)
        dr = c.delete(f"{GATE_BASE_URL}/v1/documents/{doc_id}")
        assert dr.status_code in (200, 202), dr.text
        dj = dr.json()
        assert dj.get("doc_id") == doc_id, dj

















