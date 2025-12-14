from __future__ import annotations

import random
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential


class EmbeddingsClient:
    def __init__(
        self,
        provider: str,
        vector_size: int,
        url: str | None,
        model: str | None,
        api_key: str | None,
        timeout_s: float,
    ):
        self.provider = provider
        self.vector_size = vector_size
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _mock_embed_one(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding: stable per text; good for local dev/tests.
        seed = int.from_bytes(text.encode("utf-8")[:16].ljust(16, b"\0"), "little", signed=False)
        rnd = random.Random(seed)
        return [rnd.uniform(-1.0, 1.0) for _ in range(self.vector_size)]

    @retry(wait=wait_random_exponential(multiplier=0.2, max=2.0), stop=stop_after_attempt(3))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "mock":
            return [self._mock_embed_one(t) for t in texts]

        if not self.url:
            raise RuntimeError("Embeddings URL is not configured")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {"input": texts}
        # Support OpenAI-compatible embedding backends (e.g. Infinity) that require a model id.
        if self.model:
            payload["model"] = self.model
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()

        # Supported response formats:
        # 1) Simple contract: { "vectors": [[...], ...] }
        vectors = data.get("vectors")
        if isinstance(vectors, list):
            return vectors

        # 2) OpenAI embeddings: { "data": [{"embedding":[...]}...], ... }
        oai_data = data.get("data")
        if isinstance(oai_data, list):
            out: list[list[float]] = []
            for it in oai_data:
                if not isinstance(it, dict):
                    continue
                emb = it.get("embedding")
                if isinstance(emb, list):
                    out.append(emb)
            if out:
                return out

        raise RuntimeError("Bad embeddings response: expected 'vectors' or OpenAI-style 'data[].embedding'")


