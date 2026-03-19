"""StepEnv — uniform environment passed to every step handler."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from config import Settings
from engine.context import RunContext
from llm import LLMClient


@dataclass(slots=True)
class StepEnv:
    ctx: RunContext
    llm: LLMClient
    model: str
    http_client: httpx.AsyncClient
    settings: Settings
