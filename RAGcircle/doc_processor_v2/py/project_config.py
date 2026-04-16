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

    # Optional
    language: str | None
    status: str | None


def extract_doc_processor_config(project: Mapping[str, Any]) -> DocProcessorConfig:
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
        vlm_model=raw_project["vlm_model"],
        vlm_concurrency=raw_project["vlm_concurrency"],
        page_window=raw_project["page_window"],
        max_px=raw_project["max_px"],
        chunk_size=raw_project["chunk_size"],
        chunk_overlap=raw_project["chunk_overlap"],
        embedding_model=raw_project["embedding_model"],
        language=raw_project.get("language"),
        status=raw_project.get("status"),
    )