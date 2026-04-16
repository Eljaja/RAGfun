from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Request / response models — mirrors generator_contract + gate_v2 models
# ---------------------------------------------------------------------------

class SearchFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None


class AgentRequest(BaseModel):
    """Maps to POST /v1/chat and /v1/chat/stream (gate proxies to generator /agent)."""
    project_id: str
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: SearchFilters | None = None
    include_sources: bool = True
    mode: Literal["minimal", "conservative", "aggressive"] | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    max_llm_calls: int | None = Field(None, ge=1, le=100)
    max_fact_queries: int | None = Field(None, ge=0, le=10)
    use_hyde: bool | None = None
    use_fact_queries: bool | None = None
    use_retry: bool | None = None
    use_tools: bool | None = None


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    trace_id: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "hybrid"
    partial: bool = False
    degraded: list[str] = Field(default_factory=list)


class SimpleChatRequest(BaseModel):
    """Maps to POST /v1/simple-chat and /v1/simple-chat/stream."""
    project_id: str
    query: str
    preset: Literal["fast", "hybrid", "thorough", "budget"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    max_retries: int = Field(default=1, ge=0, le=5)
    reflection_enabled: bool = True


class SimpleChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    answer: str
    sources: list[str] = Field(default_factory=list)
    chunks_used: int = 0
    retries_used: int = 0
    query: str = ""


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = None
    embedding_model: str = "BAAI/bge-m3"
    chunk_size: int = 512
    chunk_overlap: int = 64
    language: str = "ru"
    llm_model: str = "openai/gpt-oss-120b"


class ProjectResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    project: dict[str, Any]


class ProjectsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    projects: list[dict[str, Any]] = Field(default_factory=list)


class DocumentsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    documents: list[dict[str, Any]] = Field(default_factory=list)
    total: int | None = None
    limit: int | None = None
    offset: int | None = None


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    doc_id: str
    project_id: str
    size: int | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClientAuth:
    bearer_token: str | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RagGatewayClient:
    """Typed SDK for gate_v2.

    Talks directly to the gate on ``/api/v1/...`` by default.
    Set *path_prefix* if the gate sits behind a reverse proxy
    (e.g. ``path_prefix="/storage-api"`` for nginx setups).
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth: ClientAuth | None = None,
        timeout_s: float = 60.0,
        path_prefix: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth or ClientAuth()
        self._prefix = path_prefix.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RagGatewayClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    # -- internals ----------------------------------------------------------

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{self._prefix}{path}"

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
                detail = str(
                    payload.get("detail")
                    or payload.get("error")
                    or payload.get("message")
                    or ""
                )
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
            body = resp.json()
        except Exception as exc:
            raise UnexpectedResponse(f"Expected JSON response for {method} {path}") from exc
        if not isinstance(body, dict):
            raise UnexpectedResponse(f"Expected JSON object for {method} {path}")
        return body

    # -- SSE streaming ------------------------------------------------------

    @staticmethod
    def _iter_sse_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
        data_buf: list[str] = []
        for line in lines:
            if line == "":
                if not data_buf:
                    continue
                raw = "\n".join(data_buf).strip()
                data_buf = []
                if raw == "[DONE]":
                    break
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"type": "raw", "data": raw}
                yield parsed if isinstance(parsed, dict) else {"type": "raw", "data": parsed}
                continue
            if line.startswith("data:"):
                data_buf.append(line[5:].strip())

        if data_buf:
            raw = "\n".join(data_buf).strip()
            if raw and raw != "[DONE]":
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"type": "raw", "data": raw}
                yield parsed if isinstance(parsed, dict) else {"type": "raw", "data": parsed}

    def _stream_sse(self, path: str, *, json_body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._client.stream(
            "POST",
            self._url(path),
            json=json_body,
            headers=self._headers({"content-type": "application/json"}),
        ) as resp:
            self._raise_for_status(resp)
            yield from self._iter_sse_lines(resp.iter_lines())

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = self._client.get(
            f"{self._base_url}/public/health",
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # -----------------------------------------------------------------------
    # Projects
    # -----------------------------------------------------------------------

    def create_project(self, payload: ProjectCreateRequest) -> ProjectResponse:
        data = self._request_json(
            "POST", "/api/v1/projects",
            json_body=payload.model_dump(exclude_none=True),
        )
        return ProjectResponse.model_validate(data)

    def list_projects(self) -> ProjectsResponse:
        data = self._request_json("GET", "/api/v1/projects")
        return ProjectsResponse.model_validate(data)

    def get_project(self, project_id: str) -> ProjectResponse:
        data = self._request_json("GET", f"/api/v1/projects/{project_id}")
        return ProjectResponse.model_validate(data)

    def delete_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/api/v1/projects/{project_id}")

    # -----------------------------------------------------------------------
    # Documents
    # -----------------------------------------------------------------------

    def list_project_documents(
        self,
        project_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentsResponse:
        data = self._request_json(
            "GET",
            f"/api/v1/projects/{project_id}/documents",
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

        form: dict[str, Any] = {
            "title": title or path.name,
            "refresh": "true" if refresh else "false",
        }
        for key, val in [
            ("description", description),
            ("uri", uri),
            ("source", source),
            ("lang", lang),
            ("tags", tags),
            ("acl", acl),
        ]:
            if val is not None:
                form[key] = val

        with path.open("rb") as fh:
            data = self._request_json(
                "POST",
                f"/api/v1/projects/{project_id}/upload",
                data=form,
                files={"file": (path.name, fh, "application/octet-stream")},
            )
        return UploadResponse.model_validate(data)

    def get_document(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/documents/{doc_id}")

    def get_document_status(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/documents/{doc_id}/status")

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/api/v1/documents/{doc_id}")

    def download_document(self, doc_id: str) -> bytes:
        resp = self._client.get(
            self._url(f"/api/v1/documents/{doc_id}/download"),
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.content

    # -----------------------------------------------------------------------
    # Agent chat  (POST /v1/chat, /v1/chat/stream)
    # -----------------------------------------------------------------------

    def agent_chat(self, payload: AgentRequest) -> AgentResponse:
        data = self._request_json(
            "POST", "/api/v1/chat",
            json_body=payload.model_dump(exclude_none=True),
        )
        return AgentResponse.model_validate(data)

    def agent_chat_stream(self, payload: AgentRequest) -> Iterator[dict[str, Any]]:
        return self._stream_sse(
            "/api/v1/chat/stream",
            json_body=payload.model_dump(exclude_none=True),
        )

    # -----------------------------------------------------------------------
    # Simple chat  (POST /v1/simple-chat, /v1/simple-chat/stream)
    # -----------------------------------------------------------------------

    def simple_chat(self, payload: SimpleChatRequest) -> SimpleChatResponse:
        data = self._request_json(
            "POST", "/api/v1/simple-chat",
            json_body=payload.model_dump(exclude_none=True),
        )
        return SimpleChatResponse.model_validate(data)

    def simple_chat_stream(self, payload: SimpleChatRequest) -> Iterator[dict[str, Any]]:
        return self._stream_sse(
            "/api/v1/simple-chat/stream",
            json_body=payload.model_dump(exclude_none=True),
        )
