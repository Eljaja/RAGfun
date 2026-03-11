from __future__ import annotations

from fastapi import HTTPException

from auth import UserCreds
from database_ops import Project, ProjectDB, ProjectStatus


async def authorize_project(
    user: UserCreds,
    project_id: str,
    project_db: ProjectDB
) -> Project:
    """
    Verify user owns the project.
    Raises 404 if not found, 403 if not owned.
    """
    project = await project_db.get(project_id)
    if not project or project.status != ProjectStatus.ACTIVE:
        raise HTTPException(status_code=404, detail="project_not_found")
    if project.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="project_access_denied")
    return project
