"""Pure OpenAI-compatible async LLM transport. No prompts, no domain logic."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from tools import execute_tool

logger = logging.getLogger(__name__)


class ThinkStripper:
    """Streaming-safe removal of <think>...</think> blocks.

    Handles tags split across SSE chunks by buffering partial matches.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []

        while True:
            low = self._buf.lower()
            if not self._in_think:
                idx = low.find(self._OPEN)
                if idx == -1:
                    keep = len(self._OPEN) - 1
                    if len(self._buf) <= keep:
                        return "".join(out_parts)
                    out_parts.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                    return "".join(out_parts)
                out_parts.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True
            else:
                idx = low.find(self._CLOSE)
                if idx == -1:
                    return "".join(out_parts)
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False

    def finalize(self) -> str:
        if self._in_think:
            tail = ""
        else:
            tail = self._buf
        self._buf = ""
        return tail


class LLMClient:
    """Async client for OpenAI-compatible chat/completions endpoints."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = await self._client.post(self._url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        strip_thinking: bool = True,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        stripper = ThinkStripper() if strip_thinking else None
        async with self._client.stream(
            "POST", self._url, json=payload, headers=self._headers,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    if stripper:
                        safe = stripper.feed(content)
                        if safe:
                            yield safe
                    else:
                        yield content
            if stripper:
                tail = stripper.finalize()
                if tail:
                    yield tail

    async def complete_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_rounds: int = 5,
    ) -> str:
        msgs = list(messages)
        for _ in range(max_rounds):
            payload: dict[str, Any] = {
                "model": model,
                "messages": msgs,
                "temperature": temperature,
                "tools": tool_defs,
                "tool_choice": "auto",
            }
            resp = await self._client.post(self._url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            msgs.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls or [],
            })

            if not tool_calls:
                return (content or "").strip()

            for tc in tool_calls:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(fn.get("name", ""), args)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })
        return ""

    async def close(self):
        await self._client.aclose()
