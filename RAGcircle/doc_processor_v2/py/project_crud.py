from __future__ import annotations

from typing import Any

import httpx

from errors import NonRetryableError


def build_project_url(*, base_url: str, path_template: str, project_id: str) -> str:
    """Build full URL for project CRUD endpoint."""
    base = base_url.rstrip("/")
    path = path_template.format(project_id)
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


async def fetch_project(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    path_template: str,
    project_id: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Fetch project from gate CRUD endpoint.

    Args:
        client: HTTP client to use.
        base_url: Base URL for the CRUD endpoint.
        path_template: Path template with {} placeholder for project_id.
        project_id: ID of the project to fetch.
        timeout_s: Request timeout in seconds.

    Returns:
        The project dict from the {"project": ...} response.

    Raises:
        NonRetryableError: Project not found (404) or malformed response.
        httpx.HTTPError: Transport or HTTP errors (typically retryable).
    """
    url = build_project_url(
        base_url=base_url,
        path_template=path_template,
        project_id=project_id,
    )
    response = await client.get(url, timeout=timeout_s)

    if response.status_code == 404:
        raise NonRetryableError(f"project_not_found:{project_id}")

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as e:
        raise NonRetryableError("project_crud_invalid_json", cause=e) from e

    project = payload.get("project")
    if not isinstance(project, dict):
        raise NonRetryableError(f"project_crud_missing_project_key:{url!r}")

    return project