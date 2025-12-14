from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, AsyncIterator, Literal

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
        filters: dict[str, Any] | None,
        acl: list[str],
        include_sources: bool,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "include_sources": include_sources,
            "sources_level": "basic",
            "filters": filters,
            "acl": acl,
            "group_by_doc": True,
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/search", json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                except Exception:
                    error_detail = {"error": e.response.text[:500] if e.response.text else str(e)}
                logger.error(
                    "retrieval_search_error",
                    extra={
                        "extra": {
                            "status_code": e.response.status_code,
                            "url": str(e.request.url),
                            "detail": error_detail,
                        }
                    },
                )
                raise
            return r.json()

    async def index_upsert(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/upsert", json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                except Exception:
                    error_detail = {"error": e.response.text[:500] if e.response.text else str(e)}
                logger.error(
                    "retrieval_index_upsert_error",
                    extra={
                        "extra": {
                            "status_code": e.response.status_code,
                            "url": str(e.request.url),
                            "detail": error_detail,
                        }
                    },
                )
                raise
            return r.json()

    async def index_exists(self, *, doc_ids: list[str]) -> dict[str, Any]:
        payload = {"doc_ids": doc_ids}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/exists", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_delete(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/delete", json=payload)
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
        file_content: bytes,
        filename: str,
        content_type: str | None = None,
        doc_id: str,
        title: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: list[str] | None = None,
        acl: list[str] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        # Provide content_type explicitly; otherwise some clients fall back to application/octet-stream
        # which breaks downstream format detection (pdf/docx/etc).
        files = {"file": (filename, file_content, content_type or "application/octet-stream")}
        data = {"doc_id": doc_id}
        if title:
            data["title"] = title
        if uri:
            data["uri"] = uri
        if source:
            data["source"] = source
        if lang:
            data["lang"] = lang
        if tags:
            data["tags"] = ",".join(tags)
        if acl:
            data["acl"] = ",".join(acl)
        if tenant_id:
            data["tenant_id"] = tenant_id
        if project_id:
            data["project_id"] = project_id

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/store", files=files, data=data)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                except Exception:
                    error_detail = {"error": e.response.text[:500] if e.response.text else str(e)}
                logger.error(
                    "storage_store_error",
                    extra={
                        "extra": {
                            "status_code": e.response.status_code,
                            "doc_id": doc_id,
                            "detail": error_detail,
                        }
                    },
                )
                raise
            return r.json()

    async def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.get(f"{self._base_url}/v1/documents/{doc_id}/metadata")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    async def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.delete(f"{self._base_url}/v1/documents/{doc_id}")
            r.raise_for_status()
            return r.json()

    async def search_documents(
        self,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        lang: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"limit": limit, "offset": offset}
        if source:
            payload["source"] = source
        if tags:
            payload["tags"] = tags
        if lang:
            payload["lang"] = lang
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if project_id:
            payload["project_id"] = project_id

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/search", json=payload)
            r.raise_for_status()
            return r.json()


class DocProcessorClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def process(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/process", json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                except Exception:
                    error_detail = {"error": e.response.text[:500] if e.response.text else str(e)}
                logger.error(
                    "doc_processor_error",
                    extra={
                        "extra": {
                            "status_code": e.response.status_code,
                            "url": str(e.request.url),
                            "detail": error_detail,
                        }
                    },
                )
                raise
            return r.json()



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

    async def chat_stream(self, *, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream chat completions as SSE events."""
        if self._provider == "mock":
            # For mock streaming, return a simple async generator
            answer = self._mock(messages)
            for chunk in answer.split():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk + ' '}}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if self._provider != "openai_compat":
            raise ValueError(f"unknown_llm_provider:{self._provider}")
        assert self._base_url is not None
        assert self._api_key is not None
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": self._model, "messages": messages, "temperature": 0.2, "stream": True}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream("POST", f"{self._base_url}/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line

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


