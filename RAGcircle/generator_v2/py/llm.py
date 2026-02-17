from __future__ import annotations

import json
import logging

import httpx
from pydantic import BaseModel

from models import ChunkResult

logger = logging.getLogger(__name__)


# ── Shared LLM client ───────────────────────────────────────


class LLMClient:
    """Single async client for all chat/completions calls."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        response_format: dict | None = None,
    ) -> str:
        payload: dict = {"model": model, "messages": messages}
        if response_format:
            payload["response_format"] = response_format

        resp = await self.client.post(self.url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self):
        await self.client.aclose()


# ── One-step agents ─────────────────────────────────────────


async def generate_answer(
    client: LLMClient,
    model: str,
    query: str,
    chunks: list[ChunkResult],
) -> str:
    context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))

    return await client.complete(model, [
        {
            "role": "system",
            "content": (
                "Answer based on the provided context. "
                "Cite ONLY PROVIDED sources using square brackets "
                "and the name of the source in the brackets."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ])


class ReflectionResult(BaseModel):
    complete: bool
    missing_context: str | None = None
    requery: str | None = None


async def reflect(
    client: LLMClient,
    model: str,
    query: str,
    chunks: list[ChunkResult],
    answer: str,
) -> ReflectionResult:
    context = "\n".join(c.text for c in chunks)

    raw = await client.complete(
        model,
        [
            {
                "role": "system",
                "content": (
                    "Evaluate if the answer fully addresses the question "
                    "based on the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "reflection-result",
                "schema": ReflectionResult.model_json_schema(),
            },
        },
    )

    data = json.loads(raw)
    return ReflectionResult(
        complete=data.get("complete", True),
        missing_context=data.get("missing_context"),
        requery=data.get("requery"),
    )
