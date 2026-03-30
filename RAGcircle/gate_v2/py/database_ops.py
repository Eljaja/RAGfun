from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json 

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
                    embedding_model TEXT NOT NULL DEFAULT 'intfloat/multilingual-e5-base',
                    chunk_size INT NOT NULL DEFAULT 512,
                    chunk_overlap INT NOT NULL DEFAULT 64,
                    language TEXT NOT NULL DEFAULT 'ru',
                    llm_model TEXT NOT NULL DEFAULT 'gemma-3-12b',
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
        description: Optional[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        language: str = "ru",
        llm_model: str = "gemma-3-12b",
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
                INSERT INTO projects (
                    project_id, user_id, name, description,
                    embedding_model, chunk_size, chunk_overlap, language, llm_model,
                    status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'active')
                RETURNING *
            """, project_id, user_id, name, description,
                embedding_model, chunk_size, chunk_overlap, language, llm_model)

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



    # async def update_extra(self, doc_id: str, extra: dict) -> bool:
    #     """Replace the entire extra field with a new dict."""
    #     async with self.pool.acquire() as conn:
    #         result = await conn.execute("""
    #             UPDATE documents 
    #             SET extra = $2 
    #             WHERE doc_id = $1
    #         """, doc_id, json.dumps(extra))
    #         return result == "UPDATE 1"

    async def get(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID. Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM documents WHERE doc_id = $1
            """, doc_id)
            if not row:
                return None
            return dict(row)

    async def list_by_project(
        self, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """List documents for a project with pagination. Returns (docs, total)."""
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE project_id = $1",
                project_id,
            )
            # probably deterministic 
            rows = await conn.fetch("""
                SELECT * FROM documents 
                WHERE project_id = $1 
                ORDER BY created_at DESC, doc_id
                LIMIT $2 OFFSET $3
            """, project_id, limit, offset)
            return [dict(row) for row in rows], total

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




from datetime import datetime


class DocumentEventType(str, Enum):
    UPLOADED = "uploaded"
    INGESTED = "ingested"
    DELETED = "deleted" 
    PROCESSED = "processed"
    ERROR_PROCESSING = "error_processing"
    EMBEDDINGS_CREATED = "embeddings_created"
    INDEXED = "indexed"

@dataclass
class DocumentEvent:
    event_id: str
    doc_id: str
    project_id: str
    event_type: DocumentEventType
    service_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "DocumentEvent":
        return cls(
            event_id=row["event_id"],
            doc_id=row["doc_id"],
            project_id=row["project_id"],
            event_type=DocumentEventType(row["event_type"]),
            service_name=row["service_name"],
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=row["created_at"].timestamp(),
        )

class DocumentEventDB:
    """PostgreSQL-backed document event log (append-only)."""
    
import uuid
import json
from datetime import datetime

class DocumentEventDB:
    """PostgreSQL-backed document storage."""
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_schema(self):
        """Ensure documents and document_events tables exist."""
        async with self.pool.acquire() as conn:
            # Documents table
            # await conn.execute("""
            #     CREATE TABLE IF NOT EXISTS documents (...)
            # """)
            
            # Document events table (append-only)
            # REFERENCES documents(doc_id) ON DELETE CASCADE,
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_events (
                    event_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL, 
                    project_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    service_name TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_events_doc_id 
                ON document_events(doc_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_events_created_at 
                ON document_events(created_at DESC)
            """)

    # ... existing methods (persist_document, get, etc.)

    # === Event Logging Methods ===

    async def log_event(
        self,
        doc_id: str,
        project_id: str,
        event_type: DocumentEventType,
        service_name: str | None = None,
        metadata: dict | None = None,
    ) -> DocumentEvent:
        """Append a new event to the document event log."""
        async with self.pool.acquire() as conn:
            event_id = str(uuid.uuid4())
            row = await conn.fetchrow("""
                INSERT INTO document_events 
                (event_id, doc_id, project_id, event_type, service_name, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
            """, event_id, doc_id, project_id, event_type.value, service_name, json.dumps(metadata or {}))
            return DocumentEvent.from_row(row)

    async def log_ingested(
        self,
        doc_id: str,
        project_id: str,
        service_name: str = "presign_service",
        **extra_metadata
    ) -> DocumentEvent:
        """Log that a document was ingested."""
        return await self.log_event(
            doc_id=doc_id,
            project_id=project_id,
            event_type=DocumentEventType.INGESTED,
            service_name=service_name,
            metadata=extra_metadata,
        )

    async def log_deleted(
        self,
        doc_id: str,
        project_id: str,
        deleted_by: str | None = None,
        reason: str | None = None,
    ) -> DocumentEvent:
        """Log that a document was deleted."""
        return await self.log_event(
            doc_id=doc_id,
            project_id=project_id,
            event_type=DocumentEventType.DELETED,
            service_name="presign_service",
            metadata={"deleted_by": deleted_by, "reason": reason} if deleted_by or reason else {},
        )

    async def log_processed(
        self,
        doc_id: str,
        project_id: str,
        service_name: str = "doc_processor v2",
        processing_time_ms: int | None = None,
        **extra
    ) -> DocumentEvent:
        """Log that a document was processed successfully."""
        metadata = extra
        if processing_time_ms is not None:
            metadata["processing_time_ms"] = processing_time_ms
        return await self.log_event(
            doc_id=doc_id,
            project_id=project_id,
            event_type=DocumentEventType.PROCESSED,
            service_name=service_name,
            metadata=metadata,
        )

    async def log_error(
        self,
        doc_id: str,
        project_id: str,
        error_message: str,
        error_type: str | None = None,
        service_name: str = "doc_processor v2",
    ) -> DocumentEvent:
        """Log a processing error."""
        return await self.log_event(
            doc_id=doc_id,
            project_id=project_id,
            event_type=DocumentEventType.ERROR_PROCESSING,
            service_name=service_name,
            metadata={
                "error_message": error_message,
                "error_type": error_type,
            },
        )

    # === Query Methods ===

    async def get_events(
        self,
        doc_id: str,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DocumentEvent]:
        """Get all events for a document (ordered by time descending)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM document_events 
                WHERE doc_id = $1 AND project_id = $2
                ORDER BY created_at DESC
                LIMIT $3 OFFSET $4
            """, doc_id, project_id, limit, offset)
            return [DocumentEvent.from_row(row) for row in rows]

    async def get_latest_event(
        self,
        doc_id: str,
        project_id: str,
        event_type: DocumentEventType | None = None,
    ) -> DocumentEvent | None:
        """Get the most recent event for a document."""
        async with self.pool.acquire() as conn:
            if event_type:
                row = await conn.fetchrow("""
                    SELECT * FROM document_events 
                    WHERE doc_id = $1 AND project_id = $2 AND event_type = $3
                    ORDER BY created_at DESC
                    LIMIT 1
                """, doc_id, project_id, event_type.value)
            else:
                row = await conn.fetchrow("""
                    SELECT * FROM document_events 
                    WHERE doc_id = $1 AND project_id = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                """, doc_id, project_id)
            return DocumentEvent.from_row(row) if row else None

    async def get_events_by_type(
        self,
        event_type: DocumentEventType,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[DocumentEvent]:
        """Get all events of a specific type (optionally scoped to a project)."""
        async with self.pool.acquire() as conn:
            if project_id is not None:
                rows = await conn.fetch("""
                    SELECT * FROM document_events 
                    WHERE event_type = $1 AND project_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                """, event_type.value, project_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM document_events 
                    WHERE event_type = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, event_type.value, limit)
            return [DocumentEvent.from_row(row) for row in rows]

    async def get_documents_with_error(
        self,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[tuple[str, DocumentEvent]]:
        """Get documents that have error events (optionally scoped to a project)."""
        async with self.pool.acquire() as conn:
            if project_id is not None:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (doc_id) * FROM document_events 
                    WHERE event_type = 'error_processing' AND project_id = $1
                    ORDER BY doc_id, created_at DESC
                    LIMIT $2
                """, project_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (doc_id) * FROM document_events 
                    WHERE event_type = 'error_processing'
                    ORDER BY doc_id, created_at DESC
                    LIMIT $1
                """, limit)
            return [(row["doc_id"], DocumentEvent.from_row(row)) for row in rows]

    async def count_events(self, doc_id: str, project_id: str) -> int:
        """Count total events for a document."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM document_events WHERE doc_id = $1 AND project_id = $2
            """, doc_id, project_id)



@asynccontextmanager
async def create_db(dsn: str, max_projects_per_user: int = 5, **kwargs):
    """Create both DBs with a shared connection pool."""
    pool = await asyncpg.create_pool(dsn, **kwargs)

    project_db = ProjectDB(pool, max_projects_per_user)
    document_db = DocumentDB(pool)
    event_db = DocumentEventDB(pool)

    # Projects table must be created first (documents references it)
    await project_db.ensure_schema()
    await document_db.ensure_schema()
    await event_db.ensure_schema()

    try:
        yield project_db, document_db, event_db
    finally:
        await pool.close()
