from __future__ import annotations
import httpx
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

class EmbeddingResponseData(BaseModel):
    embedding: list[float]
    index: int
    object: Literal["embedding"] = Field(...)

class EmbeddingResponse(BaseModel):
    data: list[EmbeddingResponseData]
    model: str
    object: Literal["list"] = Field(...)
    usage: dict[str, Any] | None = None  # we don't care but at least we won't crash

class Embedder:
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=10.0, read=timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(t, str) for t in texts):
            raise ValueError("All inputs must be strings")

        try:
            resp = await self.client.post(
                "/embeddings",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()

            parsed = EmbeddingResponse.model_validate(resp.json())
            # Sort just in case the API returns them out of order (rare but happens)
            sorted_data = sorted(parsed.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Embedding API failed: {e.response.status_code} {e.response.text}") from e
        except (ValidationError, KeyError, TypeError) as e:
            raise ValueError(f"Unexpected response format from embedding API: {e}") from e

    async def close(self) -> None:
        await self.client.aclose()

    def __await__(self):
        return self.close().__await__()