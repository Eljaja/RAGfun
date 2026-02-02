from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(role=data.get("role", "user"), content=data.get("content", ""))


@dataclass(frozen=True)
class GateFilters:
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    project_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.source is not None:
            payload["source"] = self.source
        if self.tags is not None:
            payload["tags"] = list(self.tags)
        if self.lang is not None:
            payload["lang"] = self.lang
        if self.doc_ids is not None:
            payload["doc_ids"] = list(self.doc_ids)
        if self.tenant_id is not None:
            payload["tenant_id"] = self.tenant_id
        if self.project_id is not None:
            payload["project_id"] = self.project_id
        if self.project_ids is not None:
            payload["project_ids"] = list(self.project_ids)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateFilters":
        return cls(
            source=data.get("source"),
            tags=data.get("tags"),
            lang=data.get("lang"),
            doc_ids=data.get("doc_ids"),
            tenant_id=data.get("tenant_id"),
            project_id=data.get("project_id"),
            project_ids=data.get("project_ids"),
        )


@dataclass(frozen=True)
class ChatRequest:
    query: str
    history: list[ChatMessage] = field(default_factory=list)
    retrieval_mode: Literal["bm25", "vector", "hybrid"] | None = None
    top_k: int | None = None
    rerank: bool | None = None
    filters: GateFilters | None = None
    acl: list[str] = field(default_factory=list)
    include_sources: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "history": [m.to_dict() for m in self.history],
            "include_sources": self.include_sources,
            "acl": list(self.acl),
        }
        if self.retrieval_mode is not None:
            payload["retrieval_mode"] = self.retrieval_mode
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.rerank is not None:
            payload["rerank"] = self.rerank
        if self.filters is not None:
            payload["filters"] = self.filters.to_dict()
        return payload


@dataclass(frozen=True)
class Source:
    doc_id: str
    ref: int | None = None
    title: str | None = None
    uri: str | None = None
    locator: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        return cls(
            ref=data.get("ref"),
            doc_id=data.get("doc_id", ""),
            title=data.get("title"),
            uri=data.get("uri"),
            locator=data.get("locator"),
            raw=data,
        )


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    doc_id: str
    text: str | None
    score: float
    source: Source | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextChunk":
        source = data.get("source")
        return cls(
            chunk_id=data.get("chunk_id", ""),
            doc_id=data.get("doc_id", ""),
            text=data.get("text"),
            score=float(data.get("score", 0.0)),
            source=Source.from_dict(source) if isinstance(source, dict) else None,
            raw=data,
        )


@dataclass(frozen=True)
class ChatResponse:
    ok: bool
    answer: str
    used_mode: str
    degraded: list[str]
    partial: bool
    context: list[ContextChunk]
    sources: list[Source]
    retrieval: dict[str, Any] | None
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        return cls(
            ok=bool(data.get("ok", True)),
            answer=data.get("answer", ""),
            used_mode=data.get("used_mode", ""),
            degraded=list(data.get("degraded", []) or []),
            partial=bool(data.get("partial", False)),
            context=[
                ContextChunk.from_dict(c)
                for c in (data.get("context", []) or [])
                if isinstance(c, dict)
            ],
            sources=[
                Source.from_dict(s)
                for s in (data.get("sources", []) or [])
                if isinstance(s, dict)
            ],
            retrieval=data.get("retrieval"),
            raw=data,
        )


@dataclass(frozen=True)
class ChatStreamEvent:
    type: str
    data: dict[str, Any] | None
    raw: str
