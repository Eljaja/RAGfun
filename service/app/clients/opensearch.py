from __future__ import annotations

import time
from typing import Any

from opensearchpy import OpenSearch


def _build_os_client(url: str, username: str | None, password: str | None) -> OpenSearch:
    http_auth = None
    if username is not None and password is not None:
        http_auth = (username, password)
    return OpenSearch(
        hosts=[url],
        http_auth=http_auth,
        use_ssl=url.startswith("https://"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=120,  # Increased timeout for large bulk operations
    )


class OpenSearchClient:
    def __init__(
        self,
        url: str,
        username: str | None,
        password: str | None,
        index_alias: str,
        index_prefix: str,
    ):
        self.url = url
        self.index_alias = index_alias
        self.index_prefix = index_prefix
        self.client = _build_os_client(url, username, password)
        self._doc_id_field: str | None = None

    def ping(self) -> bool:
        return bool(self.client.ping())

    def ensure_index_and_alias(self) -> str:
        """
        Ensures alias exists. If missing, creates a new index + assigns alias.
        Returns the concrete index name behind alias.
        """
        if self.client.indices.exists_alias(name=self.index_alias):
            ali = self.client.indices.get_alias(name=self.index_alias)
            # return first concrete index
            return next(iter(ali.keys()))

        index_name = f"{self.index_prefix}{int(time.time())}"
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "ru": {"type": "russian"},
                        "en": {"type": "english"},
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"ru": {"type": "text", "analyzer": "ru"}, "en": {"type": "text", "analyzer": "en"}},
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"ru": {"type": "text", "analyzer": "ru"}, "en": {"type": "text", "analyzer": "en"}},
                    },
                    "source": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "lang": {"type": "keyword"},
                    "uri": {"type": "keyword"},
                    "acl": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "project_id": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "content_hash": {"type": "keyword"},
                    "token_count": {"type": "integer"},
                    "locator": {"type": "object", "enabled": True},
                }
            },
        }
        self.client.indices.create(index=index_name, body=mapping)
        self.client.indices.put_alias(index=index_name, name=self.index_alias)
        return index_name

    def _resolve_doc_id_field(self) -> str:
        """
        Prefer keyword field for aggregations/terms queries.
        Falls back to doc_id if mapping is missing/unexpected.
        """
        if self._doc_id_field:
            return self._doc_id_field
        try:
            mapping = self.client.indices.get_mapping(index=self.index_alias)
            # alias points to a concrete index; grab first mapping
            index_name = next(iter(mapping.keys()))
            props = ((mapping.get(index_name) or {}).get("mappings") or {}).get("properties") or {}
            doc_id_def = props.get("doc_id") or {}
            keyword_def = (doc_id_def.get("fields") or {}).get("keyword")
            if keyword_def is not None:
                self._doc_id_field = "doc_id.keyword"
            elif doc_id_def.get("type") == "keyword":
                self._doc_id_field = "doc_id"
            else:
                self._doc_id_field = "doc_id"
        except Exception:
            self._doc_id_field = "doc_id"
        return self._doc_id_field

    def bulk_upsert(self, docs: list[dict[str, Any]], refresh: bool) -> dict[str, Any]:
        ops: list[dict[str, Any]] = []
        for d in docs:
            chunk_id = d["chunk_id"]
            ops.append({"index": {"_index": self.index_alias, "_id": chunk_id}})
            ops.append(d)
        return self.client.bulk(body=ops, refresh=refresh)

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        try:
            r = self.client.get(index=self.index_alias, id=chunk_id)
            return r.get("_source")
        except Exception:
            return None

    def delete_by_chunk_id(self, chunk_id: str, refresh: bool) -> None:
        self.client.delete(index=self.index_alias, id=chunk_id, refresh=refresh, ignore=[404])

    def delete_by_doc_id(self, doc_id: str, refresh: bool) -> dict[str, Any]:
        doc_id_field = self._resolve_doc_id_field()
        q = {"query": {"term": {doc_id_field: doc_id}}}
        return self.client.delete_by_query(index=self.index_alias, body=q, refresh=refresh, conflicts="proceed")

    def search(
        self,
        query: str,
        filters: list[dict[str, Any]],
        top_k: int,
    ) -> dict[str, Any]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^3", "text.ru^3", "text.en^3", "title^1.5", "title.ru^1.5", "title.en^1.5"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
            "highlight": {"fields": {"text": {}, "title": {}}, "pre_tags": ["<em>"], "post_tags": ["</em>"]},
        }
        return self.client.search(index=self.index_alias, body=body)

    def doc_counts_by_doc_id(self, doc_ids: list[str]) -> dict[str, int]:
        """
        Return document -> chunk count for the given doc_ids using OpenSearch.
        This is used to quickly check "is indexed" without relying on a textual query.
        """
        if not doc_ids:
            return {}
        doc_id_field = self._resolve_doc_id_field()
        body = {
            "size": 0,
            "query": {"terms": {doc_id_field: doc_ids}},
            "aggs": {"docs": {"terms": {"field": doc_id_field, "size": len(doc_ids)}}},
        }
        r = self.client.search(index=self.index_alias, body=body)
        buckets = (((r.get("aggregations") or {}).get("docs") or {}).get("buckets")) or []
        out: dict[str, int] = {}
        for b in buckets:
            try:
                out[str(b.get("key"))] = int(b.get("doc_count") or 0)
            except Exception:
                continue
        return out

    def get_chunks_by_page(self, doc_id: str, page: int) -> list[dict[str, Any]]:
        """Get all chunks for a specific (doc_id, page)."""
        doc_id_field = self._resolve_doc_id_field()
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {doc_id_field: doc_id}},
                        {"term": {"locator.page": page}},
                    ]
                }
            },
            "size": 1000,  # Should be enough for one page
            "sort": [{"chunk_index": {"order": "asc"}}],
        }
        try:
            r = self.client.search(index=self.index_alias, body=query)
            chunks = []
            for hit in r.get("hits", {}).get("hits", []):
                src = hit.get("_source", {}) or {}
                chunks.append(src)
            return chunks
        except Exception:
            return []

