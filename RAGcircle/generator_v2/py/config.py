from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    # ── Retrieval service ────────────────────────────────────
    retrieval_url: str = "http://localhost:8920"

    # ── LLM ──────────────────────────────────────────────────
    llm_base_url: str = "https://llm.c.singularitynet.io/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_timeout: float = 60.0

    # ── Reflection ───────────────────────────────────────────
    reflection_enabled: bool = True
    reflection_model: str = ""  # defaults to llm_model if empty
    max_retries: int = 1

    # ── Server ───────────────────────────────────────────────
    port: int = 8930
    log_level: str = "INFO"
