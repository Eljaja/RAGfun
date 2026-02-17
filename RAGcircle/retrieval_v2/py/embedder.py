from __future__ import annotations

import httpx
from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbeddingResponseData(BaseModel):
    embedding: list[float]
    index: int
    object: Literal["embedding"] = Field(...)


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingResponseData]
    model: str
    object: Literal["list"] = Field(...)
    usage: dict[str, Any] | None = None


class Embedder:
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=10.0, read=timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )
        self.model = model

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        resp = await self.client.post(
            "/embeddings",
            json={"model": model or self.model, "input": texts},
        )
        resp.raise_for_status()

        parsed = EmbeddingResponse.model_validate(resp.json())
        sorted_data = sorted(parsed.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def close(self) -> None:
        await self.client.aclose()
