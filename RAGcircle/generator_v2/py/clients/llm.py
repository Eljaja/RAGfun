"""OpenAI-compatible async LLM client.

Architecture: pure functions for all data transforms, class only for IO.

Pure (no IO, no self, testable with plain assert):
    build_payload, extract_content, extract_tool_state,
    clean_response, parse_json, validate_model, parse_sse_line

IO boundary (the only thing that touches the network):
    LLMClient._post

Composition (IO >> pure):
    complete      = _post  >> extract_content
    complete_json = complete >> parse_json
    complete_model= complete_json >> validate_model

Exceptions
----------
LLMTransportError – network / timeout / HTTP status errors.  Retryable.
LLMParseError     – LLM returned a response that couldn't be parsed.
                    Not retryable.  Carries `.raw` for debugging.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from clients.tools import execute_tool

logger = logging.getLogger(__name__)

_THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:\w*)\n?(.*?)```", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    if not text or not text.strip():
        return text
    return _THINKING_RE.sub("", text).strip()


def strip_fences(text: str) -> str:
    """Extract content from markdown code fences if present."""
    text = text.strip()
    matches = _FENCE_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    return text

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)

T = TypeVar("T", bound=BaseModel)


# ── Exceptions ────────────────────────────────────────────


class LLMTransportError(Exception):
    """Network / timeout / HTTP status error from the LLM provider."""


class LLMParseError(Exception):
    """LLM returned a response that couldn't be parsed as expected."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


# ── Pure functions ────────────────────────────────────────


def build_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    response_format: dict[str, Any] | None = None,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble an OpenAI-compatible request payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    if stream:
        payload["stream"] = True
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def extract_content(data: dict[str, Any]) -> str:
    """Pull message content from an API response."""
    return data["choices"][0]["message"]["content"]


def extract_tool_state(
    data: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Pull content + tool_calls from an API response."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return msg.get("content"), msg.get("tool_calls")


def clean_response(raw: str) -> str:
    """Strip <think> tags and markdown code fences."""
    return strip_fences(strip_thinking(raw))


def parse_json(raw: str) -> dict[str, Any]:
    """Clean an LLM string and parse it as a JSON object.

    Raises LLMParseError if the cleaned string is not valid JSON
    or is not a dict.
    """
    cleaned = clean_response(raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMParseError(str(exc), raw=raw) from exc
    if not isinstance(result, dict):
        raise LLMParseError(
            f"Expected JSON object, got {type(result).__name__}", raw=raw,
        )
    return result


def validate_model(data: dict[str, Any], result_type: type[T]) -> T:
    """Validate a dict into a Pydantic model.

    Raises LLMParseError if validation fails.
    """
    try:
        return result_type.model_validate(data)
    except ValidationError as exc:
        raise LLMParseError(str(exc), raw=json.dumps(data)) from exc


def parse_sse_line(line: str) -> str | None:
    """Extract delta content from a single SSE line.

    Returns None for non-content lines (comments, [DONE], bad JSON).
    """
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    return chunk.get("choices", [{}])[0].get("delta", {}).get("content")


# ── Streaming helper (stateful — not a pure function) ────


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


# ── Client (IO boundary) ─────────────────────────────────


class LLMClient:
    """Async HTTP transport for OpenAI-compatible endpoints.

    Only holds connection state. All parsing / cleaning lives in the
    module-level pure functions above — testable without a network.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── single IO boundary ───────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.3, max=2),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the completions endpoint. Retries on transport errors."""
        try:
            resp = await self._client.post(
                self._url, headers=self._headers, json=payload,
            )
            resp.raise_for_status()
            return resp.json()
        except _RETRYABLE:
            raise
        except httpx.HTTPStatusError as exc:
            raise LLMTransportError(
                f"{exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

    # ── compositions: IO >> pure ─────────────────────────

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        """_post >> extract_content"""
        payload = build_payload(
            model, messages,
            temperature=temperature, response_format=response_format,
        )
        try:
            data = await self._post(payload)
        except _RETRYABLE as exc:
            raise LLMTransportError(str(exc)) from exc
        return extract_content(data)

    async def complete_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """complete >> parse_json"""
        raw = await self.complete(
            model, messages,
            temperature=temperature, response_format=response_format,
        )
        
        return parse_json(raw)

    async def complete_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        result_type: type[T],
        *,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> T:
        """complete_json >> validate_model"""
        data = await self.complete_json(
            model, messages,
            temperature=temperature, response_format=response_format,
        )
        return validate_model(data, result_type)

    # ── streaming (IO + stateful ThinkStripper) ──────────

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        strip_thinking: bool = True,
    ) -> AsyncIterator[str]:
        payload = build_payload(
            model, messages, temperature=temperature, stream=True,
        )
        stripper = ThinkStripper() if strip_thinking else None
        try:
            async with self._client.stream(
                "POST", self._url, json=payload, headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    content = parse_sse_line(line)
                    if content is None:
                        if line and line.strip() == "data: [DONE]":
                            break
                        continue
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
        except (*_RETRYABLE, httpx.HTTPStatusError) as exc:
            raise LLMTransportError(str(exc)) from exc

    # ── tool-calling loop ────────────────────────────────

    async def complete_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_rounds: int = 5,
    ) -> str:
        """Multi-round tool-calling loop.

        Each round: _post >> extract_tool_state, then local tool execution.
        """
        msgs = list(messages)
        for _ in range(max_rounds):
            payload = build_payload(
                model, msgs, temperature=temperature, tools=tool_defs,
            )
            try:
                data = await self._post(payload)
            except _RETRYABLE as exc:
                raise LLMTransportError(str(exc)) from exc

            content, tool_calls = extract_tool_state(data)

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
