from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx
from fastapi import HTTPException


@dataclass(frozen=True)
class ProjectDeps:
    """Project-specific dependencies for generator (LLM & language).

    Extracted from gate project data before each pipeline run.
    """

    project_id: str
    llm_model: str
    language: str


def extract_project_deps(project: Mapping[str, Any]) -> ProjectDeps:
    """Extract generator deps from gate project data."""
    return ProjectDeps(
        project_id=project["project_id"],
        llm_model=project["llm_model"],
        language=project.get("language", "ru"),
    )


async def fetch_project_deps(
    client: httpx.AsyncClient,
    project_id: str,
) -> ProjectDeps:
    """Fetch project from gate and extract deps."""
    url = f"/public/v1/internal/projects/{project_id}"
    response = await client.get(url, timeout=10.0)

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"project_not_found:{project_id}")

    response.raise_for_status()
    payload = response.json()
    project = payload.get("project")

    if not project:
        raise HTTPException(status_code=502, detail="invalid_project_response")

    return extract_project_deps(project)