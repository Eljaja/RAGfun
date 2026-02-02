from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import quote

import httpx

from .errors import GateConnectionError, GateHTTPError, GateTimeoutError
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    GateFilters,
)


def _normalize_base_url(base_url: str) -> str:
    base = base_url.strip()
    if not base:
        raise ValueError("base_url must not be empty")
    return base.rstrip("/")


def _to_csv(value: str | Iterable[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    items = [v.strip() for v in value if v and v.strip()]
    return ",".join(items) if items else None


def _parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    event_type: str | None = None
    data_lines: list[str] = []

    for line in lines:
        if line == "":
            if data_lines:
                yield event_type, "\n".join(data_lines)
            event_type = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].lstrip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

        data_lines.append(line)

    if data_lines:
        yield event_type, "\n".join(data_lines)


class GateClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout
        self._headers = headers or {}
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GateClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_base_url(self, base_url: str) -> None:
        self.base_url = _normalize_base_url(base_url)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        accept: str = "application/json",
    ) -> Any:
        url = f"{self.base_url}{path}"
        merged_headers = {"Accept": accept, **self._headers}
        if headers:
            merged_headers.update(headers)
        try:
            response = self._client.request(
                method,
                url,
                json=json_body,
                params=params,
                files=files,
                data=data,
                headers=merged_headers,
                timeout=timeout or self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise GateTimeoutError("Request timed out") from exc
        except httpx.RequestError as exc:
            raise GateConnectionError(str(exc)) from exc

        if response.status_code >= 400:
            self._raise_http_error(response)

        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json() if response.content else None
        return response.text

    def _raise_http_error(self, response: httpx.Response) -> None:
        message = response.text
        payload: Any | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, dict) and "message" in detail:
                    message = str(detail["message"])
                elif detail is not None:
                    message = str(detail)
                elif "message" in payload:
                    message = str(payload["message"])
        except Exception:
            payload = None
        raise GateHTTPError(
            status_code=response.status_code,
            message=message,
            response_text=response.text,
            response_json=payload,
        )

    def healthz(self) -> dict[str, Any]:
        return self._request("GET", "/v1/healthz")

    def readyz(self) -> dict[str, Any]:
        return self._request("GET", "/v1/readyz")

    def version(self) -> dict[str, Any]:
        return self._request("GET", "/v1/version")

    def metrics(self) -> str:
        return self._request("GET", "/v1/metrics", accept="text/plain")

    def chat(
        self,
        *,
        query: str,
        history: list[ChatMessage] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: GateFilters | dict[str, Any] | None = None,
        acl: Iterable[str] | None = None,
        include_sources: bool = True,
        timeout: float | None = None,
    ) -> ChatResponse:
        req = ChatRequest(
            query=query,
            history=history or [],
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=top_k,
            rerank=rerank,
            filters=filters if isinstance(filters, GateFilters) else None,
            acl=list(acl) if acl is not None else [],
            include_sources=include_sources,
        )
        payload = req.to_dict()
        if filters and not isinstance(filters, GateFilters):
            payload["filters"] = dict(filters)
        result = self._request("POST", "/v1/chat", json_body=payload, timeout=timeout)
        return ChatResponse.from_dict(result or {})

    def chat_stream(
        self,
        *,
        query: str,
        history: list[ChatMessage] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: GateFilters | dict[str, Any] | None = None,
        acl: Iterable[str] | None = None,
        include_sources: bool = True,
        timeout: float | None = 300.0,
    ) -> Iterator[ChatStreamEvent]:
        req = ChatRequest(
            query=query,
            history=history or [],
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=top_k,
            rerank=rerank,
            filters=filters if isinstance(filters, GateFilters) else None,
            acl=list(acl) if acl is not None else [],
            include_sources=include_sources,
        )
        payload = req.to_dict()
        if filters and not isinstance(filters, GateFilters):
            payload["filters"] = dict(filters)

        url = f"{self.base_url}/v1/chat/stream"
        try:
            with self._client.stream(
                "POST",
                url,
                json=payload,
                headers={"Accept": "text/event-stream", **self._headers},
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    self._raise_http_error(response)

                for event_type, raw in _parse_sse_lines(response.iter_lines()):
                    if not raw:
                        continue
                    data: dict[str, Any] | None = None
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "type" in data:
                            yield ChatStreamEvent(type=str(data["type"]), data=data, raw=raw)
                            continue
                    except Exception:
                        data = None
                    yield ChatStreamEvent(type=event_type or "message", data=data, raw=raw)
        except httpx.TimeoutException as exc:
            raise GateTimeoutError("Request timed out") from exc
        except httpx.RequestError as exc:
            raise GateConnectionError(str(exc)) from exc

    def upload(
        self,
        *,
        file_path: str,
        doc_id: str,
        title: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: str | Iterable[str] | None = None,
        acl: str | Iterable[str] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        refresh: bool = False,
        timeout: float | None = 120.0,
    ) -> dict[str, Any]:
        data = {
            "doc_id": doc_id,
            "refresh": "true" if refresh else "false",
        }
        if title:
            data["title"] = title
        if uri:
            data["uri"] = uri
        if source:
            data["source"] = source
        if lang:
            data["lang"] = lang
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                data["tags"] = csv
        if acl is not None:
            csv = _to_csv(acl)
            if csv:
                data["acl"] = csv
        if tenant_id:
            data["tenant_id"] = tenant_id
        if project_id:
            data["project_id"] = project_id

        with open(file_path, "rb") as handle:
            files = {"file": (file_path, handle)}
            return self._request(
                "POST",
                "/v1/documents/upload",
                data=data,
                files=files,
                timeout=timeout,
            )

    def list_documents(
        self,
        *,
        source: str | None = None,
        tags: str | Iterable[str] | None = None,
        lang: str | None = None,
        collections: str | Iterable[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if source:
            params["source"] = source
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                params["tags"] = csv
        if lang:
            params["lang"] = lang
        if collections is not None:
            csv = _to_csv(collections)
            if csv:
                params["collections"] = csv
        return self._request("GET", "/v1/documents", params=params)

    def document_status(self, *, doc_id: str) -> dict[str, Any]:
        enc = quote(doc_id, safe="")
        return self._request("GET", f"/v1/documents/{enc}/status")

    def documents_stats(
        self,
        *,
        source: str | None = None,
        tags: str | Iterable[str] | None = None,
        lang: str | None = None,
        collections: str | Iterable[str] | None = None,
        page_size: int = 500,
        max_docs: int = 200_000,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page_size": page_size, "max_docs": max_docs}
        if source:
            params["source"] = source
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                params["tags"] = csv
        if lang:
            params["lang"] = lang
        if collections is not None:
            csv = _to_csv(collections)
            if csv:
                params["collections"] = csv
        return self._request("GET", "/v1/documents/stats", params=params)

    def collections(self, *, tenant_id: str | None = None, limit: int = 1000) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id
        return self._request("GET", "/v1/collections", params=params)

    def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        enc = quote(doc_id, safe="")
        return self._request("DELETE", f"/v1/documents/{enc}")

    def delete_all_documents(
        self,
        *,
        confirm: bool = False,
        batch_size: int = 200,
        concurrency: int = 10,
        max_batches: int = 10_000,
    ) -> dict[str, Any]:
        if not confirm:
            raise ValueError("Refusing to delete all documents without confirm=True")
        params = {
            "confirm": "true",
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_batches": max_batches,
        }
        return self._request("DELETE", "/v1/documents", params=params)


class GateAsyncClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout
        self._headers = headers or {}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GateAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def set_base_url(self, base_url: str) -> None:
        self.base_url = _normalize_base_url(base_url)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        accept: str = "application/json",
    ) -> Any:
        url = f"{self.base_url}{path}"
        merged_headers = {"Accept": accept, **self._headers}
        if headers:
            merged_headers.update(headers)
        try:
            response = await self._client.request(
                method,
                url,
                json=json_body,
                params=params,
                files=files,
                data=data,
                headers=merged_headers,
                timeout=timeout or self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise GateTimeoutError("Request timed out") from exc
        except httpx.RequestError as exc:
            raise GateConnectionError(str(exc)) from exc

        if response.status_code >= 400:
            self._raise_http_error(response)

        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json() if response.content else None
        return response.text

    def _raise_http_error(self, response: httpx.Response) -> None:
        message = response.text
        payload: Any | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, dict) and "message" in detail:
                    message = str(detail["message"])
                elif detail is not None:
                    message = str(detail)
                elif "message" in payload:
                    message = str(payload["message"])
        except Exception:
            payload = None
        raise GateHTTPError(
            status_code=response.status_code,
            message=message,
            response_text=response.text,
            response_json=payload,
        )

    async def healthz(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/healthz")

    async def readyz(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/readyz")

    async def version(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/version")

    async def metrics(self) -> str:
        return await self._request("GET", "/v1/metrics", accept="text/plain")

    async def chat(
        self,
        *,
        query: str,
        history: list[ChatMessage] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: GateFilters | dict[str, Any] | None = None,
        acl: Iterable[str] | None = None,
        include_sources: bool = True,
        timeout: float | None = None,
    ) -> ChatResponse:
        req = ChatRequest(
            query=query,
            history=history or [],
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=top_k,
            rerank=rerank,
            filters=filters if isinstance(filters, GateFilters) else None,
            acl=list(acl) if acl is not None else [],
            include_sources=include_sources,
        )
        payload = req.to_dict()
        if filters and not isinstance(filters, GateFilters):
            payload["filters"] = dict(filters)
        result = await self._request("POST", "/v1/chat", json_body=payload, timeout=timeout)
        return ChatResponse.from_dict(result or {})

    async def chat_stream(
        self,
        *,
        query: str,
        history: list[ChatMessage] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: GateFilters | dict[str, Any] | None = None,
        acl: Iterable[str] | None = None,
        include_sources: bool = True,
        timeout: float | None = 300.0,
    ) -> Iterator[ChatStreamEvent]:
        req = ChatRequest(
            query=query,
            history=history or [],
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=top_k,
            rerank=rerank,
            filters=filters if isinstance(filters, GateFilters) else None,
            acl=list(acl) if acl is not None else [],
            include_sources=include_sources,
        )
        payload = req.to_dict()
        if filters and not isinstance(filters, GateFilters):
            payload["filters"] = dict(filters)

        url = f"{self.base_url}/v1/chat/stream"
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers={"Accept": "text/event-stream", **self._headers},
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    self._raise_http_error(response)

                event_type: str | None = None
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if line == "":
                        if data_lines:
                            raw = "\n".join(data_lines)
                            data_lines = []
                            data: dict[str, Any] | None = None
                            try:
                                data = json.loads(raw)
                                if isinstance(data, dict) and "type" in data:
                                    yield ChatStreamEvent(type=str(data["type"]), data=data, raw=raw)
                                    event_type = None
                                    continue
                            except Exception:
                                data = None
                            yield ChatStreamEvent(type=event_type or "message", data=data, raw=raw)
                        event_type = None
                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        event_type = line[len("event:") :].lstrip()
                        continue

                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
                        continue

                    data_lines.append(line)

                if data_lines:
                    raw = "\n".join(data_lines)
                    data: dict[str, Any] | None = None
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "type" in data:
                            yield ChatStreamEvent(type=str(data["type"]), data=data, raw=raw)
                            return
                    except Exception:
                        data = None
                    yield ChatStreamEvent(type=event_type or "message", data=data, raw=raw)
        except httpx.TimeoutException as exc:
            raise GateTimeoutError("Request timed out") from exc
        except httpx.RequestError as exc:
            raise GateConnectionError(str(exc)) from exc

    async def upload(
        self,
        *,
        file_path: str,
        doc_id: str,
        title: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: str | Iterable[str] | None = None,
        acl: str | Iterable[str] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        refresh: bool = False,
        timeout: float | None = 120.0,
    ) -> dict[str, Any]:
        data = {
            "doc_id": doc_id,
            "refresh": "true" if refresh else "false",
        }
        if title:
            data["title"] = title
        if uri:
            data["uri"] = uri
        if source:
            data["source"] = source
        if lang:
            data["lang"] = lang
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                data["tags"] = csv
        if acl is not None:
            csv = _to_csv(acl)
            if csv:
                data["acl"] = csv
        if tenant_id:
            data["tenant_id"] = tenant_id
        if project_id:
            data["project_id"] = project_id

        with open(file_path, "rb") as handle:
            files = {"file": (file_path, handle)}
            return await self._request(
                "POST",
                "/v1/documents/upload",
                data=data,
                files=files,
                timeout=timeout,
            )

    async def list_documents(
        self,
        *,
        source: str | None = None,
        tags: str | Iterable[str] | None = None,
        lang: str | None = None,
        collections: str | Iterable[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if source:
            params["source"] = source
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                params["tags"] = csv
        if lang:
            params["lang"] = lang
        if collections is not None:
            csv = _to_csv(collections)
            if csv:
                params["collections"] = csv
        return await self._request("GET", "/v1/documents", params=params)

    async def document_status(self, *, doc_id: str) -> dict[str, Any]:
        enc = quote(doc_id, safe="")
        return await self._request("GET", f"/v1/documents/{enc}/status")

    async def documents_stats(
        self,
        *,
        source: str | None = None,
        tags: str | Iterable[str] | None = None,
        lang: str | None = None,
        collections: str | Iterable[str] | None = None,
        page_size: int = 500,
        max_docs: int = 200_000,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page_size": page_size, "max_docs": max_docs}
        if source:
            params["source"] = source
        if tags is not None:
            csv = _to_csv(tags)
            if csv:
                params["tags"] = csv
        if lang:
            params["lang"] = lang
        if collections is not None:
            csv = _to_csv(collections)
            if csv:
                params["collections"] = csv
        return await self._request("GET", "/v1/documents/stats", params=params)

    async def collections(self, *, tenant_id: str | None = None, limit: int = 1000) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id
        return await self._request("GET", "/v1/collections", params=params)

    async def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        enc = quote(doc_id, safe="")
        return await self._request("DELETE", f"/v1/documents/{enc}")

    async def delete_all_documents(
        self,
        *,
        confirm: bool = False,
        batch_size: int = 200,
        concurrency: int = 10,
        max_batches: int = 10_000,
    ) -> dict[str, Any]:
        if not confirm:
            raise ValueError("Refusing to delete all documents without confirm=True")
        params = {
            "confirm": "true",
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_batches": max_batches,
        }
        return await self._request("DELETE", "/v1/documents", params=params)
