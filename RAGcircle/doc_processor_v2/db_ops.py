from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid
import json

import asyncpg

class DocumentEventType(str, Enum):
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
    event_type: DocumentEventType
    service_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "DocumentEvent":
        return cls(
            event_id=row["event_id"],
            doc_id=row["doc_id"],
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (...)
            """)
            
            # Document events table (append-only)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_events (
                    event_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
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
        event_type: DocumentEventType,
        service_name: str | None = None,
        metadata: dict | None = None,
    ) -> DocumentEvent:
        """Append a new event to the document event log."""
        async with self.pool.acquire() as conn:
            event_id = str(uuid.uuid4())
            row = await conn.fetchrow("""
                INSERT INTO document_events 
                (event_id, doc_id, event_type, service_name, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """, event_id, doc_id, event_type.value, service_name, json.dumps(metadata or {}))
            return DocumentEvent.from_row(row)

    async def log_ingested(
        self,
        doc_id: str,
        service_name: str = "presign_service",
        **extra_metadata
    ) -> DocumentEvent:
        """Log that a document was ingested."""
        return await self.log_event(
            doc_id=doc_id,
            event_type=DocumentEventType.INGESTED,
            service_name=service_name,
            metadata=extra_metadata,
        )

    async def log_deleted(
        self,
        doc_id: str,
        deleted_by: str | None = None,
        reason: str | None = None,
    ) -> DocumentEvent:
        """Log that a document was deleted."""
        return await self.log_event(
            doc_id=doc_id,
            event_type=DocumentEventType.DELETED,
            service_name="presign_service",
            metadata={"deleted_by": deleted_by, "reason": reason} if deleted_by or reason else {},
        )

    async def log_processed(
        self,
        doc_id: str,
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
            event_type=DocumentEventType.PROCESSED,
            service_name=service_name,
            metadata=metadata,
        )

    async def log_error(
        self,
        doc_id: str,
        error_message: str,
        error_type: str | None = None,
        service_name: str = "doc_processor v2",
    ) -> DocumentEvent:
        """Log a processing error."""
        return await self.log_event(
            doc_id=doc_id,
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
        limit: int = 50,
        offset: int = 0,
    ) -> list[DocumentEvent]:
        """Get all events for a document (ordered by time descending)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM document_events 
                WHERE doc_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, doc_id, limit, offset)
            return [DocumentEvent.from_row(row) for row in rows]

    async def get_latest_event(
        self,
        doc_id: str,
        event_type: DocumentEventType | None = None,
    ) -> DocumentEvent | None:
        """Get the most recent event for a document."""
        async with self.pool.acquire() as conn:
            if event_type:
                row = await conn.fetchrow("""
                    SELECT * FROM document_events 
                    WHERE doc_id = $1 AND event_type = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                """, doc_id, event_type.value)
            else:
                row = await conn.fetchrow("""
                    SELECT * FROM document_events 
                    WHERE doc_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """, doc_id)
            return DocumentEvent.from_row(row) if row else None

    async def get_events_by_type(
        self,
        event_type: DocumentEventType,
        limit: int = 100,
    ) -> list[DocumentEvent]:
        """Get all events of a specific type across all documents."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM document_events 
                WHERE event_type = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, event_type.value, limit)
            return [DocumentEvent.from_row(row) for row in rows]

    async def get_documents_with_error(
        self,
        limit: int = 100,
    ) -> list[tuple[str, DocumentEvent]]:
        """Get documents that have error events (for retry processing)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT ON (doc_id) * FROM document_events 
                WHERE event_type = 'error_processing'
                ORDER BY doc_id, created_at DESC
                LIMIT $1
            """, limit)
            return [(row["doc_id"], DocumentEvent.from_row(row)) for row in rows]

    async def count_events(self, doc_id: str) -> int:
        """Count total events for a document."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM document_events WHERE doc_id = $1
            """, doc_id)

# Usage
# async with create_db(dsn) as (project_db, document_db, document_event_db):
#     await document_db.persist_document(...)
#     await document_event_db.log_event(doc_id, DocumentEventType.INGESTED, "presign_service")



@asynccontextmanager
async def create_db(dsn: str, **kwargs):
    """Create all DBs with a shared connection pool."""
    pool = await asyncpg.create_pool(dsn, **kwargs)


    document_event_db = DocumentEventDB(pool)


    await document_event_db.ensure_schema()

    try:
        yield document_event_db
    finally:
        await pool.close()