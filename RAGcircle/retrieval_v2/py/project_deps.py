from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectDeps:
    """Project-specific dependencies for retrieval (embedding & reranking models).

    Extracted from gate project data before each retriever call.
    """

    project_id: str
    embedding_model: str
    reranker_model: str


def extract_project_deps(project: Mapping[str, Any]) -> ProjectDeps:
    """Extract retrieval deps from gate project data.

    Follows the same pattern as extract_doc_processor_config in doc_processor_v2.
    """
    return ProjectDeps(
        project_id=project["project_id"],
        embedding_model=project["embedding_model"],
        reranker_model=project["reranker_model"],
    )