from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import httpx
from pydantic import BaseModel, ConfigDict, Field


class SDKError(RuntimeError):
    """Base exception for the gateway SDK."""


class APIError(SDKError):
    """Raised when an HTTP request returns non-2xx status."""

    def __init__(self, status_code: int, message: str, payload: Any | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload


class UnexpectedResponse(SDKError):
    """Raised when response body cannot be parsed as expected."""


class ChatMessage(BaseModel):
    role: str
    content: str


class GateFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    project_ids: list[str] | None = None


class ChatStreamRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    filters: GateFilters | None = None
    include_sources: bool = True


class AgentStreamRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    filters: GateFilters | None = None
    include_sources: bool = True
    mode: str | None = None


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    retrieval_mode: str | None = None
    top_k: int | None = None
    rerank: bool | None = None
    use_adaptive_k: bool | None = None
    adaptive_k_multi_query: str | None = None
    filters: GateFilters | None = None
    acl: list[str] = Field(default_factory=list)
    include_sources: bool = True


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = None
    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = 512
    chunk_overlap: int = 64
    language: str = "ru"
    llm_model: str = "gemma-3-12b"


class FlexibleResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ProjectsResponse(FlexibleResponse):
    projects: list[dict[str, Any]] = Field(default_factory=list)


class ProjectResponse(FlexibleResponse):
    project: dict[str, Any]


class DocumentsResponse(FlexibleResponse):
    documents: list[dict[str, Any]] = Field(default_factory=list)
    total: int | None = None
    limit: int | None = None
    offset: int | None = None


class UploadResponse(FlexibleResponse):
    doc_id: str
    project_id: str
    size: int | None = None


class ChatResponse(FlexibleResponse):
    answer: str
    used_mode: str | None = None
    partial: bool | None = None
    degraded: list[str] | None = None
    context: list[dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None
    retrieval: dict[str, Any] | None = None


@dataclass(slots=True)
class ClientAuth:
    bearer_token: str | None = None


class RagGatewayClient:
    """
    Thin typed SDK for the single nginx gateway (default port 8916).

    This wrapper intentionally mirrors only the allowlisted API surface:
    - rag-gate chat (/api/v1/chat, /api/v1/chat/stream)
    - agent-search stream (/agent-api/v1/agent/stream)
    - gate_v2 storage/ingestion endpoints under /storage-api/api/v1/...
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth: ClientAuth | None = None,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth or ClientAuth()
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RagGatewayClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{path}"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        out: dict[str, str] = {"accept": "application/json"}
        if self._auth.bearer_token:
            tok = self._auth.bearer_token.strip()
            out["Authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
        if extra:
            out.update(extra)
        return out

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        payload: Any | None = None
        detail = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                detail = str(payload.get("detail") or payload.get("error") or payload.get("message") or "")
        except Exception:
            detail = (resp.text or "").strip()
        if not detail:
            detail = "request_failed"
        raise APIError(resp.status_code, detail, payload)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        resp = self._client.request(
            method=method,
            url=self._url(path),
            params=params,
            json=json_body,
            data=data,
            files=files,
            headers=self._headers(headers),
        )
        self._raise_for_status(resp)
        try:
            payload = resp.json()
        except Exception as exc:
            raise UnexpectedResponse(f"Expected JSON response for {method} {path}") from exc
        if not isinstance(payload, dict):
            raise UnexpectedResponse(f"Expected JSON object response for {method} {path}")
        return payload

    @staticmethod
    def _iter_sse_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
        data_lines: list[str] = []
        for line in lines:
            if line == "":
                if not data_lines:
                    continue
                payload = "\n".join(data_lines).strip()
                data_lines = []
                if payload == "[DONE]":
                    break
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = {"type": "raw", "data": payload}
                if isinstance(parsed, dict):
                    yield parsed
                else:
                    yield {"type": "raw", "data": parsed}
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if data_lines:
            payload = "\n".join(data_lines).strip()
            if payload and payload != "[DONE]":
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = {"type": "raw", "data": payload}
                if isinstance(parsed, dict):
                    yield parsed
                else:
                    yield {"type": "raw", "data": parsed}

    def _stream_sse(self, path: str, *, payload: BaseModel) -> Iterator[dict[str, Any]]:
        with self._client.stream(
            "POST",
            self._url(path),
            json=payload.model_dump(exclude_none=True),
            headers=self._headers({"content-type": "application/json"}),
        ) as resp:
            self._raise_for_status(resp)
            for event in self._iter_sse_lines(resp.iter_lines()):
                yield event

    # -------------------------
    # Chat / agent methods
    # -------------------------
    def chat(self, payload: ChatRequest) -> ChatResponse:
        chat_payload = payload.model_dump(exclude_none=True)
        # Keep agent filters intact, but avoid chat filter over-restriction on /api path.
        chat_payload.pop("filters", None)
        data = self._request_json(
            "POST",
            "/api/v1/chat",
            json_body=chat_payload,
        )
        return ChatResponse.model_validate(data)

    def chat_stream(self, payload: ChatStreamRequest) -> Iterator[dict[str, Any]]:
        stream_payload = payload.model_copy(update={"filters": None})
        return self._stream_sse("/api/v1/chat/stream", payload=stream_payload)

    def agent_stream(self, payload: AgentStreamRequest) -> Iterator[dict[str, Any]]:
        return self._stream_sse("/agent-api/v1/agent/stream", payload=payload)

    # -------------------------
    # Storage / ingestion (gate_v2)
    # -------------------------
    def list_projects(self) -> ProjectsResponse:
        data = self._request_json("GET", "/storage-api/api/v1/projects")
        return ProjectsResponse.model_validate(data)

    def create_project(self, payload: ProjectCreateRequest) -> ProjectResponse:
        data = self._request_json("POST", "/storage-api/api/v1/projects", json_body=payload.model_dump(exclude_none=True))
        return ProjectResponse.model_validate(data)

    def get_project(self, project_id: str) -> ProjectResponse:
        data = self._request_json("GET", f"/storage-api/api/v1/projects/{project_id}")
        return ProjectResponse.model_validate(data)

    def delete_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/storage-api/api/v1/projects/{project_id}")

    def list_project_documents(self, project_id: str, *, limit: int = 50, offset: int = 0) -> DocumentsResponse:
        data = self._request_json(
            "GET",
            f"/storage-api/api/v1/projects/{project_id}/documents",
            params={"limit": limit, "offset": offset},
        )
        return DocumentsResponse.model_validate(data)

    def upload_document(
        self,
        project_id: str,
        file_path: str | Path,
        *,
        title: str | None = None,
        description: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: str | None = None,
        acl: str | None = None,
        refresh: bool = False,
    ) -> UploadResponse:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        form_data: dict[str, Any] = {
            "title": title or path.name,
            "refresh": "true" if refresh else "false",
        }
        optional = {
            "description": description,
            "uri": uri,
            "source": source,
            "lang": lang,
            "tags": tags,
            "acl": acl,
        }
        for key, value in optional.items():
            if value is not None:
                form_data[key] = value

        with path.open("rb") as fh:
            data = self._request_json(
                "POST",
                f"/storage-api/api/v1/projects/{project_id}/upload",
                data=form_data,
                files={"file": (path.name, fh, "application/octet-stream")},
            )
        return UploadResponse.model_validate(data)

    def get_document(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/storage-api/api/v1/documents/{doc_id}")

    def get_document_status(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/storage-api/api/v1/documents/{doc_id}/status")

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/storage-api/api/v1/documents/{doc_id}")

    def download_document(self, doc_id: str) -> bytes:
        resp = self._client.get(self._url(f"/storage-api/api/v1/documents/{doc_id}/download"), headers=self._headers())
        self._raise_for_status(resp)
        return resp.content

