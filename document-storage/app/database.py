from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from app.models import DocumentMetadata, DocumentSearchRequest

logger = logging.getLogger("storage.db")


class DatabaseClient:
    def __init__(self, db_url: str, min_conn: int = 1, max_conn: int = 10):
        self.db_url = db_url
        self.pool: ThreadedConnectionPool | None = None
        self._init_pool(min_conn, max_conn)

    def _init_pool(self, min_conn: int, max_conn: int):
        try:
            # Thread-safe pool: required because document-storage runs DB operations
            # in a threadpool (to avoid blocking the async event loop).
            self.pool = ThreadedConnectionPool(
                min_conn, max_conn, self.db_url, cursor_factory=RealDictCursor
            )
            # Test connection
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                self.pool.putconn(conn)
                logger.info("database_pool_initialized", extra={"min": min_conn, "max": max_conn})
            except Exception as e:
                self.pool.putconn(conn)
                raise
        except Exception as e:
            logger.error("database_pool_init_failed", extra={"error": str(e)})
            raise

    def ensure_schema(self):
        """Create tables if they don't exist."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        storage_id VARCHAR(255) NOT NULL,
                        title VARCHAR(512),
                        uri TEXT,
                        source VARCHAR(255),
                        lang VARCHAR(10),
                        tags TEXT[],  -- PostgreSQL array
                        tenant_id VARCHAR(255),
                        project_id VARCHAR(255),
                        acl TEXT[],  -- PostgreSQL array
                        content_type VARCHAR(255),
                        size BIGINT,
                        stored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        extra JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_doc_source ON document_metadata(source);
                    CREATE INDEX IF NOT EXISTS idx_doc_lang ON document_metadata(lang);
                    CREATE INDEX IF NOT EXISTS idx_doc_tenant ON document_metadata(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_doc_project ON document_metadata(project_id);
                    CREATE INDEX IF NOT EXISTS idx_doc_stored_at ON document_metadata(stored_at);
                    CREATE INDEX IF NOT EXISTS idx_doc_tags ON document_metadata USING GIN(tags);
                """)
                conn.commit()
                logger.info("database_schema_ensured")
        except Exception as e:
            conn.rollback()
            logger.error("database_schema_error", extra={"error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def store_metadata(
        self,
        doc_id: str,
        storage_id: str,
        metadata: dict[str, Any],
        content_type: str | None = None,
        size: int | None = None,
    ) -> DocumentMetadata:
        """Store or update document metadata."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_metadata (
                        doc_id, storage_id, title, uri, source, lang, tags,
                        tenant_id, project_id, acl, content_type, size, extra
                    ) VALUES (
                        %(doc_id)s, %(storage_id)s, %(title)s, %(uri)s, %(source)s,
                        %(lang)s, %(tags)s, %(tenant_id)s, %(project_id)s,
                        %(acl)s, %(content_type)s, %(size)s, %(extra)s
                    )
                    ON CONFLICT (doc_id) DO UPDATE SET
                        storage_id = EXCLUDED.storage_id,
                        title = EXCLUDED.title,
                        uri = EXCLUDED.uri,
                        source = EXCLUDED.source,
                        lang = EXCLUDED.lang,
                        tags = EXCLUDED.tags,
                        tenant_id = EXCLUDED.tenant_id,
                        project_id = EXCLUDED.project_id,
                        acl = EXCLUDED.acl,
                        content_type = EXCLUDED.content_type,
                        size = EXCLUDED.size,
                        extra = EXCLUDED.extra,
                        updated_at = NOW()
                    RETURNING *
                """, {
                    "doc_id": doc_id,
                    "storage_id": storage_id,
                    "title": metadata.get("title"),
                    "uri": metadata.get("uri"),
                    "source": metadata.get("source"),
                    "lang": metadata.get("lang"),
                    "tags": metadata.get("tags", []),
                    "tenant_id": metadata.get("tenant_id"),
                    "project_id": metadata.get("project_id"),
                    "acl": metadata.get("acl", []),
                    "content_type": content_type,
                    "size": size,
                    "extra": Json(metadata.get("extra", {})),
                })
                row = cur.fetchone()
                conn.commit()
                return self._row_to_metadata(row)
        except Exception as e:
            conn.rollback()
            logger.error("database_store_metadata_error", extra={"doc_id": doc_id, "error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def get_metadata(self, doc_id: str) -> DocumentMetadata | None:
        """Get document metadata by doc_id."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM document_metadata WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                if row:
                    return self._row_to_metadata(row)
                return None
        except Exception as e:
            logger.error("database_get_metadata_error", extra={"doc_id": doc_id, "error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def find_any_doc_id_by_storage_id(self, *, storage_id: str, exclude_doc_id: str | None = None) -> str | None:
        """
        Find any doc_id that references the given storage_id.
        Used for exact deduplication (bytes-level) because storage_id is deterministic (sha256-based).
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                if exclude_doc_id:
                    cur.execute(
                        "SELECT doc_id FROM document_metadata WHERE storage_id = %s AND doc_id <> %s ORDER BY stored_at ASC LIMIT 1",
                        (storage_id, exclude_doc_id),
                    )
                else:
                    cur.execute(
                        "SELECT doc_id FROM document_metadata WHERE storage_id = %s ORDER BY stored_at ASC LIMIT 1",
                        (storage_id,),
                    )
                row = cur.fetchone()
                if not row:
                    return None
                return str(row.get("doc_id"))
        finally:
            self.pool.putconn(conn)

    def count_docs_by_storage_id(self, *, storage_id: str) -> int:
        """Count how many documents reference a given storage_id."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*)::bigint as c FROM document_metadata WHERE storage_id = %s", (storage_id,))
                row = cur.fetchone() or {}
                return int(row.get("c") or 0)
        finally:
            self.pool.putconn(conn)

    def delete_metadata(self, doc_id: str) -> bool:
        """Delete document metadata."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_metadata WHERE doc_id = %s", (doc_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            conn.rollback()
            logger.error("database_delete_metadata_error", extra={"doc_id": doc_id, "error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def patch_extra(self, *, doc_id: str, patch: dict[str, Any]) -> DocumentMetadata | None:
        """
        Merge-patch JSONB `extra` field for a document.
        Top-level keys in `patch` overwrite existing keys.
        Returns updated metadata or None if doc_id not found.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE document_metadata
                    SET
                        extra = COALESCE(extra, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE doc_id = %s
                    RETURNING *
                    """,
                    (json.dumps(patch), doc_id),
                )
                row = cur.fetchone()
                conn.commit()
                if not row:
                    return None
                return self._row_to_metadata(row)
        except Exception as e:
            conn.rollback()
            logger.error("database_patch_extra_error", extra={"doc_id": doc_id, "error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def search_metadata(self, req: DocumentSearchRequest) -> tuple[list[DocumentMetadata], int]:
        """Search documents by metadata filters."""
        conn = self.pool.getconn()
        try:
            conditions = []
            params: dict[str, Any] = {}

            if req.source:
                conditions.append("source = %(source)s")
                params["source"] = req.source

            if req.lang:
                conditions.append("lang = %(lang)s")
                params["lang"] = req.lang

            if req.tenant_id:
                conditions.append("tenant_id = %(tenant_id)s")
                params["tenant_id"] = req.tenant_id

            # Collections: support both project_id (single) and project_ids (multi).
            proj_ids: list[str] = []
            if req.project_id:
                proj_ids.append(req.project_id)
            if getattr(req, "project_ids", None):
                proj_ids.extend([x for x in (req.project_ids or []) if x])
            # de-dup while preserving order
            seen: set[str] = set()
            proj_ids_norm: list[str] = []
            for x in proj_ids:
                if x in seen:
                    continue
                seen.add(x)
                proj_ids_norm.append(x)
            if proj_ids_norm:
                if len(proj_ids_norm) == 1:
                    conditions.append("project_id = %(project_id)s")
                    params["project_id"] = proj_ids_norm[0]
                else:
                    conditions.append("project_id = ANY(%(project_ids)s)")
                    params["project_ids"] = proj_ids_norm

            if req.tags:
                conditions.append("tags && %(tags)s")
                params["tags"] = req.tags

            if req.date_range:
                if "from" in req.date_range:
                    conditions.append("stored_at >= %(date_from)s")
                    params["date_from"] = req.date_range["from"]
                if "to" in req.date_range:
                    conditions.append("stored_at <= %(date_to)s")
                    params["date_to"] = req.date_range["to"]

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Count total
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) as total FROM document_metadata WHERE {where_clause}", params)
                total = cur.fetchone()["total"]

            # Get results
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM document_metadata
                    WHERE {where_clause}
                    ORDER BY stored_at DESC
                    LIMIT %(limit)s OFFSET %(offset)s
                    """,
                    {**params, "limit": req.limit, "offset": req.offset}
                )
                rows = cur.fetchall()
                docs = [self._row_to_metadata(row) for row in rows]
                return docs, total

        except Exception as e:
            logger.error("database_search_metadata_error", extra={"error": str(e)})
            raise
        finally:
            self.pool.putconn(conn)

    def list_collections(self, *, tenant_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """
        Returns distinct project_id values with counts. Used by UI as "collections".
        """
        conn = self.pool.getconn()
        try:
            where = ["project_id IS NOT NULL", "project_id <> ''"]
            params: dict[str, Any] = {"limit": max(1, min(int(limit), 5000))}
            if tenant_id:
                where.append("tenant_id = %(tenant_id)s")
                params["tenant_id"] = tenant_id
            where_clause = " AND ".join(where)
            sql = f"""
                SELECT project_id as id, COUNT(*)::bigint as count
                FROM document_metadata
                WHERE {where_clause}
                GROUP BY project_id
                ORDER BY count DESC, project_id ASC
                LIMIT %(limit)s
            """
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() or []
            out: list[dict[str, Any]] = []
            for r in rows:
                try:
                    out.append({"id": str(r.get("id")), "count": int(r.get("count") or 0)})
                except Exception:
                    continue
            return out
        finally:
            self.pool.putconn(conn)

    def _row_to_metadata(self, row: dict) -> DocumentMetadata:
        """Convert database row to DocumentMetadata."""
        return DocumentMetadata(
            doc_id=row["doc_id"],
            storage_id=row.get("storage_id"),
            title=row.get("title"),
            uri=row.get("uri"),
            source=row.get("source"),
            lang=row.get("lang"),
            tags=row.get("tags", []) or [],
            tenant_id=row.get("tenant_id"),
            project_id=row.get("project_id"),
            acl=row.get("acl", []) or [],
            content_type=row.get("content_type"),
            size=row.get("size"),
            stored_at=row.get("stored_at"),
            extra=row.get("extra", {}) or {},
        )

    def health(self) -> bool:
        """Check database health."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False
        finally:
            self.pool.putconn(conn)

    def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()

