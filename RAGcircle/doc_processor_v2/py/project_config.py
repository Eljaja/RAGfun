from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DocProcessorConfig:
    """Project configuration used by doc_processor (chunking & embedding)."""

    # Chunking & embedding
    project_id: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    # VLM / processing
    vlm_model: str
    vlm_concurrency: int
    page_window: int
    max_px: int
    ocr_mode: str | None

    # Optional
    language: str | None
    status: str | None


def extract_doc_processor_config(
    project: Mapping[str, Any],
    *,
    default_vlm_model: str,
    default_vlm_concurrency: int,
    default_page_window: int,
    default_max_px: int,
    default_ocr_mode: str,
) -> DocProcessorConfig:
    """Extract doc_processor config from gate project. Fails if required fields missing.

    Args:
        project: Raw project dict from gate CRUD endpoint.

    Returns:
        DocProcessorConfig with all fields.

    Raises:
        KeyError: If a required field is missing from the project dict.
    """
    raw_project = dict(project)

    return DocProcessorConfig(
        project_id=raw_project.get("project_id"),
        vlm_model=str(raw_project.get("vlm_model", default_vlm_model)),
        vlm_concurrency=int(raw_project.get("vlm_concurrency", default_vlm_concurrency)),
        page_window=int(raw_project.get("page_window", default_page_window)),
        max_px=int(raw_project.get("max_px", default_max_px)),
        ocr_mode=raw_project.get("ocr_mode", default_ocr_mode),
        chunk_size=int(raw_project.get("chunk_size", 512)),
        chunk_overlap=int(raw_project.get("chunk_overlap", 64)),
        embedding_model=str(raw_project.get("embedding_model", "BAAI/bge-m3")),
        language=raw_project.get("language"),
        status=raw_project.get("status"),
    )