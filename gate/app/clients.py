from __future__ import annotations

import json
import hashlib
import logging
from typing import Any, Literal

import httpx

logger = logging.getLogger("gate.clients")


class RetrievalClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def search(
        self,
        *,
        query: str,
        mode: Literal["bm25", "vector", "hybrid"],
        top_k: int,
        rerank: bool | None = None,
        max_chunks_per_doc: int | None = None,
        filters: dict[str, Any] | None,
        acl: list[str],
        include_sources: bool,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            # Let gate override service-level rerank behavior (critical for multi-query/anchor passes).
            "rerank": rerank,
            "max_chunks_per_doc": max_chunks_per_doc,
            "include_sources": include_sources,
            "sources_level": "basic",
            "filters": filters,
            "acl": acl,
            "group_by_doc": True,
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/search", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_upsert(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/upsert", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_delete(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/delete", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_exists(self, *, doc_ids: list[str]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/exists", json={"doc_ids": doc_ids})
            r.raise_for_status()
            return r.json()

    async def healthz(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(3.0, self._timeout_s)) as client:
                r = await client.get(f"{self._base_url}/v1/healthz")
                return r.status_code == 200
        except Exception:
            return False

    async def readyz(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=min(5.0, self._timeout_s)) as client:
            r = await client.get(f"{self._base_url}/v1/readyz")
            # readiness is informative; don't raise
            try:
                data = r.json()
            except Exception:
                data = {"ready": False, "error": "invalid_json", "status_code": r.status_code}
            data["_status_code"] = r.status_code
            return data


class DocumentStorageClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def store_document(
        self,
        *,
        doc_id: str,
        filename: str,
        # NOTE: gate streams UploadFile.file here to avoid buffering in memory.
        file_content: Any,
        content_type: str | None,
        title: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: list[str] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        acl: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        files = {"file": (filename, file_content, content_type or "application/octet-stream")}
        data: dict[str, str] = {"doc_id": doc_id}
        if title is not None:
            data["title"] = title
        if uri is not None:
            data["uri"] = uri
        if source is not None:
            data["source"] = source
        if lang is not None:
            data["lang"] = lang
        if tags:
            data["tags"] = ",".join([t for t in tags if t])
        if tenant_id is not None:
            data["tenant_id"] = tenant_id
        if project_id is not None:
            data["project_id"] = project_id
        if acl:
            data["acl"] = ",".join([a for a in acl if a])
        if metadata:
            data["metadata"] = json.dumps(metadata, ensure_ascii=False)

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/store", files=files, data=data)
            r.raise_for_status()
            return r.json()

    async def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            # Use query-param endpoint to avoid path parsing issues with slashy/colon doc_ids.
            r = await client.get(f"{self._base_url}/v1/documents/by-id/metadata", params={"doc_id": doc_id})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    async def patch_extra(self, *, doc_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/by-id/extra", json={"doc_id": doc_id, "patch": patch})
            r.raise_for_status()
            return r.json()

    async def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.delete(f"{self._base_url}/v1/documents/by-id", params={"doc_id": doc_id})
            r.raise_for_status()
            return r.json()

    async def search_documents(
        self,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        lang: str | None = None,
        tenant_id: str | None = None,
        project_ids: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "source": source,
            "tags": tags or [],
            "lang": lang,
            "tenant_id": tenant_id,
            "project_ids": project_ids or [],
            "limit": int(limit),
            "offset": int(offset),
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/search", json=payload)
            r.raise_for_status()
            return r.json()

    async def list_collections(self, *, tenant_id: str | None = None, limit: int = 1000) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.get(f"{self._base_url}/v1/collections", params={"tenant_id": tenant_id, "limit": int(limit)})
            r.raise_for_status()
            return r.json()


class DocProcessorClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def healthz(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(3.0, self._timeout_s)) as client:
                r = await client.get(f"{self._base_url}/v1/healthz")
                return r.status_code == 200
        except Exception:
            return False


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str | None,
        api_key: str | None,
        model: str,
        timeout_s: float,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def chat(self, *, messages: list[dict[str, str]]) -> str:
        if self._provider == "mock":
            return self._mock(messages)
        if self._provider != "openai_compat":
            raise ValueError(f"unknown_llm_provider:{self._provider}")
        assert self._base_url is not None
        assert self._api_key is not None
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": self._model, "messages": messages, "temperature": 0.2}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            logger.error("llm_unexpected_response", extra={"extra": {"keys": list(data.keys())}})
            raise RuntimeError("llm_unexpected_response")

    def _mock(self, messages: list[dict[str, str]]) -> str:
        # Deterministic-ish placeholder: summarizes question and number of context blocks.
        joined = "\n".join([m.get("content", "") for m in messages if m.get("role") != "system"]).strip()
        h = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]
        user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        return (
            "MOCK_ANSWER\n"
            f"- request_hash: {h}\n"
            f"- question: {user[:500]}\n"
            "\n"
            "Настрой `GATE_LLM_PROVIDER=openai_compat` и `GATE_LLM_API_KEY`, чтобы получать реальный ответ модели."
        )


