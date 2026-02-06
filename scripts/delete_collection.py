#!/usr/bin/env python3
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OS_URL = "http://localhost:9200"
QDRANT_URL = "http://localhost:6333"
STORAGE_URL = "http://localhost:8081"

COLLECTION = "wiki-ru"
QDRANT_COLLECTION = "rag_chunks_bge_m3_1024"
OS_INDEX = "rag_chunks"


def http_json(method: str, url: str, data=None, timeout: int = 30):
    req = urllib.request.Request(url, method=method)
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=body, timeout=timeout) as resp:
        raw = resp.read()
        text = raw.decode("utf-8") if raw else "{}"
        return resp.status, json.loads(text)


def delete_qdrant() -> None:
    print("[1/3] Deleting from Qdrant by project_id...")
    payload = {"filter": {"must": [{"key": "project_id", "match": {"value": COLLECTION}}]}}
    try:
        status, resp = http_json(
            "POST",
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/delete",
            data=payload,
            timeout=30,
        )
        result = (resp or {}).get("result", {}) if isinstance(resp, dict) else {}
        print("Qdrant delete status:", status, "result:", result.get("status"))
    except Exception as exc:
        print("Qdrant delete failed:", exc)


def _resolve_project_field() -> str:
    try:
        status, mapping = http_json("GET", f"{OS_URL}/{OS_INDEX}/_mapping", timeout=30)
        if status >= 400:
            return "project_id"
        index_name = next(iter(mapping.keys()))
        props = (mapping.get(index_name, {}) or {}).get("mappings", {}).get("properties", {}) or {}
        proj_def = props.get("project_id") or {}
        if isinstance(proj_def, dict) and isinstance(proj_def.get("fields"), dict):
            if "keyword" in proj_def.get("fields"):
                return "project_id.keyword"
        if proj_def.get("type") == "keyword":
            return "project_id"
    except Exception:
        pass
    return "project_id"


def delete_opensearch() -> None:
    print("[2/3] Deleting from OpenSearch by project_id...")
    project_field = _resolve_project_field()
    payload = {"query": {"term": {project_field: COLLECTION}}}
    try:
        status, resp = http_json(
            "POST",
            f"{OS_URL}/{OS_INDEX}/_delete_by_query?conflicts=proceed&refresh=true",
            data=payload,
            timeout=60,
        )
        print("OpenSearch delete status:", status, "deleted:", (resp or {}).get("deleted"))
    except Exception as exc:
        print("OpenSearch delete failed:", exc)


def delete_storage() -> None:
    print("[3/3] Deleting from document-storage by project_id (metadata + blobs)...")
    total_deleted = 0
    page = 0

    def delete_doc(doc_id: str):
        try:
            params = urllib.parse.urlencode({"doc_id": doc_id})
            status, _ = http_json(
                "DELETE",
                f"{STORAGE_URL}/v1/documents/by-id?{params}",
                timeout=30,
            )
            return (doc_id, status)
        except Exception as exc:
            return (doc_id, f"error:{exc}")

    while True:
        page += 1
        payload = {"project_id": COLLECTION, "limit": 500, "offset": 0}
        try:
            status, resp = http_json(
                "POST",
                f"{STORAGE_URL}/v1/documents/search",
                data=payload,
                timeout=60,
            )
        except Exception as exc:
            print("Search failed:", exc)
            break
        if status >= 400:
            print("Search failed with status:", status)
            break

        docs = resp.get("documents") or []
        if not docs:
            break

        doc_ids = [d.get("doc_id") for d in docs if d.get("doc_id")]
        if not doc_ids:
            break

        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = [ex.submit(delete_doc, doc_id) for doc_id in doc_ids]
            for fut in as_completed(futures):
                _, status = fut.result()
                if status == 200:
                    total_deleted += 1
        if page % 10 == 0:
            print(f"deleted so far: {total_deleted} (pages: {page})")
        time.sleep(0.05)

    print("Done. Total deleted from storage:", total_deleted)


def main() -> None:
    delete_qdrant()
    delete_opensearch()
    delete_storage()


if __name__ == "__main__":
    main()
