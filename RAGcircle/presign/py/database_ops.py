from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import asyncpg
from fastapi import HTTPException

from storage import UploadResult, UploadMeta


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"


@dataclass
class Project:
    project_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: float = field(default_factory=float)
    updated_at: float = field(default_factory=float)

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "Project":
        return cls(
            project_id=row["project_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            status=ProjectStatus(row["status"]),
            created_at=row["created_at"].timestamp(),
            updated_at=row["updated_at"].timestamp(),
        )

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProjectDB:
    """PostgreSQL-backed project storage."""

    def __init__(self, pool: asyncpg.Pool, max_projects_per_user: int = 5):
        self.pool = pool
        self.max_projects_per_user = max_projects_per_user

    async def ensure_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_user_id 
                ON projects(user_id) WHERE status = 'active'
            """)

    async def create(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None
    ) -> Project:
        """Create a new project. Raises 400 if limit reached."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM projects 
                WHERE user_id = $1 AND status = 'active'
            """, user_id)

            if count >= self.max_projects_per_user:
                raise HTTPException(
                    status_code=400,
                    detail=f"project_limit_reached: max {self.max_projects_per_user} projects allowed"
                )

            project_id = str(uuid.uuid4())
            row = await conn.fetchrow("""
                INSERT INTO projects (project_id, user_id, name, description, status)
                VALUES ($1, $2, $3, $4, 'active')
                RETURNING *
            """, project_id, user_id, name, description)

            return Project.from_row(row)

    async def get(self, project_id: str) -> Optional[Project]:
        """Get a project by ID. Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM projects WHERE project_id = $1
            """, project_id)
            if not row:
                return None
            return Project.from_row(row)

    async def list_for_user(self, user_id: str) -> list[Project]:
        """List all active projects for a user."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM projects 
                WHERE user_id = $1 AND status = 'active'
                ORDER BY created_at DESC
            """, user_id)
            return [Project.from_row(row) for row in rows]

    async def update(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Project]:
        """Update project fields. Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE projects SET
                    name = COALESCE($2, name),
                    description = COALESCE($3, description),
                    updated_at = NOW()
                WHERE project_id = $1 AND status = 'active'
                RETURNING *
            """, project_id, name, description)
            if not row:
                return None
            return Project.from_row(row)

    async def delete(self, project_id: str, user_id: str) -> bool:
        """
        Soft-delete a project.
        Returns True if deleted, False if not found.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE projects SET 
                    status = 'deleted',
                    updated_at = NOW()
                WHERE project_id = $1 AND user_id = $2 AND status = 'active'
            """, project_id, user_id)
            return result == "UPDATE 1"

    async def user_owns_project(self, user_id: str, project_id: str) -> bool:
        """Check if user owns the project."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 1 FROM projects 
                WHERE project_id = $1 AND user_id = $2 AND status = 'active'
            """, project_id, user_id)
            return row is not None


class DocumentDB:
    """PostgreSQL-backed document storage."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id),
                    storage_id TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    size BIGINT,
                    sha256 TEXT,
                    duplicate BOOLEAN DEFAULT FALSE,
                    extra JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_project_id 
                ON documents(project_id)
            """)

    async def persist_document(
        self,
        doc_id: str,
        project_id: str,
        upload: UploadResult,
        meta: UploadMeta,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents (doc_id, project_id, storage_id, title, description, size, sha256, duplicate, extra)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (doc_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    storage_id = EXCLUDED.storage_id,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    size = EXCLUDED.size,
                    sha256 = EXCLUDED.sha256,
                    duplicate = EXCLUDED.duplicate,
                    extra = EXCLUDED.extra
            """, doc_id, project_id, upload.storage_id, meta.title, meta.description,
                upload.size, upload.sha256, upload.duplicate, str(meta.extra))

    async def get(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID. Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM documents WHERE doc_id = $1
            """, doc_id)
            if not row:
                return None
            return dict(row)

    async def list_by_project(self, project_id: str) -> list[dict]:
        """List all documents for a project."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM documents 
                WHERE project_id = $1 
                ORDER BY created_at DESC
            """, project_id)
            return [dict(row) for row in rows]

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM documents WHERE doc_id = $1
            """, doc_id)
            return result == "DELETE 1"

    async def delete_by_project(self, project_id: str) -> int:
        """Delete all documents for a project. Returns count deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM documents WHERE project_id = $1
            """, project_id)
            return int(result.split()[-1])

    async def count_by_project(self, project_id: str) -> int:
        """Count documents in a project."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM documents WHERE project_id = $1
            """, project_id)


@asynccontextmanager
async def create_db(dsn: str, max_projects_per_user: int = 5, **kwargs):
    """Create both DBs with a shared connection pool."""
    pool = await asyncpg.create_pool(dsn, **kwargs)

    project_db = ProjectDB(pool, max_projects_per_user)
    document_db = DocumentDB(pool)

    # Projects table must be created first (documents references it)
    await project_db.ensure_schema()
    await document_db.ensure_schema()

    try:
        yield project_db, document_db
    finally:
        await pool.close()
