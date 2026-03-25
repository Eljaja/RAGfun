from __future__ import annotations

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCESSOR_", extra="ignore")

    # Service
    service_name: str = "doc-processor"
    environment: str = "dev"
    log_level: str = "INFO"

    # Downstream services
    storage_url: AnyHttpUrl = Field(default="http://document-storage:8081")
    storage_timeout_s: float = 60.0

    retrieval_url: AnyHttpUrl = Field(default="http://retrieval:8080")
    retrieval_timeout_s: float = 60.0

    # VLM (kept for compatibility while OCR-first ingestion is rolled out)
    vlm_base_url: AnyHttpUrl = Field(default="http://vllm-docling:8123/v1")
    vlm_api_key: SecretStr | None = None
    vlm_model: str = "ibm-granite/granite-docling-258M"
    vlm_timeout_s: float = 120.0

    # OCR
    ocr_enabled: bool = True
    ocr_lang: str = "en"
    ocr_device: str = "cpu"
    pdf_text_min_chars: int = 80

    # Limits
    max_pages: int = 25
    max_image_side_px: int = 1600

    # Chunking: strategy "semantic" (section-based, minimal overlap) or "fixed" (char-based with overlap)
    chunk_strategy: str = "semantic"  # semantic|fixed
    chunk_size_chars: int = 4000
    chunk_overlap_chars: int = 300

    def safe_summary(self) -> dict:
        return {
            "service": {"name": self.service_name, "environment": self.environment},
            "storage": {"url": str(self.storage_url), "timeout_s": self.storage_timeout_s},
            "retrieval": {"url": str(self.retrieval_url), "timeout_s": self.retrieval_timeout_s},
            "vlm": {
                "base_url": str(self.vlm_base_url),
                "model": self.vlm_model,
                "api_key_set": self.vlm_api_key is not None,
                "timeout_s": self.vlm_timeout_s,
            },
            "ocr": {
                "enabled": self.ocr_enabled,
                "lang": self.ocr_lang,
                "device": self.ocr_device,
                "pdf_text_min_chars": self.pdf_text_min_chars,
            },
            "limits": {
                "max_pages": self.max_pages,
                "max_image_side_px": self.max_image_side_px,
                "chunk_strategy": self.chunk_strategy,
                "chunk_size_chars": self.chunk_size_chars,
                "chunk_overlap_chars": self.chunk_overlap_chars,
            },
        }


def load_settings() -> Settings:
    s = Settings()
    if s.chunk_strategy not in ("semantic", "fixed"):
        raise ValueError("PROCESSOR_CHUNK_STRATEGY must be 'semantic' or 'fixed'")
    return s


