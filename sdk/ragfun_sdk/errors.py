from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GateError(Exception):
    """Base SDK error."""


@dataclass
class GateHTTPError(GateError):
    status_code: int
    message: str
    response_text: str | None = None
    response_json: Any | None = None

    def __str__(self) -> str:
        return f"HTTP {self.status_code}: {self.message}"


class GateConnectionError(GateError):
    """Network error or connection failure."""


class GateTimeoutError(GateError):
    """Request timed out."""
