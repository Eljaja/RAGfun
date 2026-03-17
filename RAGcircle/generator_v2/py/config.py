from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    # ── Retrieval service ────────────────────────────────────
    retrieval_url: str = "http://localhost:8920"

    # ── LLM (generation) ────────────────────────────────────
    llm_base_url: str = "https://llm.c.singularitynet.io/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_timeout: float = 60.0

    # ── Reflection (simple pipeline) ─────────────────────────
    reflection_enabled: bool = True
    reflection_model: str = ""
    max_retries: int = 1

    # ── Agent pipeline ───────────────────────────────────────
    agent_llm_model: str = ""
    agent_llm_timeout: float = 60.0
    agent_max_llm_calls: int = 12
    agent_max_fact_queries: int = 2
    agent_use_hyde: bool = False
    agent_use_fact_queries: bool = True
    agent_use_retry: bool = True
    agent_use_tools: bool = False
    agent_hyde_num: int = 1
    agent_top_k_min: int = 5
    agent_top_k_max: int = 24
    agent_gate_timeout: float = 60.0

    # ── Context building ─────────────────────────────────────
    max_context_chars: int = 6000
    max_chunk_chars: int = 1200

    # ── Server ───────────────────────────────────────────────
    port: int = 8930
    log_level: str = "INFO"
