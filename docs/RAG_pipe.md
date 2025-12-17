RAG pipeline (Gate → Doc Processor → Retrieval Service)
Ingestion (indexing)
Client → gate: upload via POST /v1/documents/upload with doc_id + metadata (tags/acl/tenant/project).
gate → document-storage (optional): streams the raw file to storage and stores metadata.
Async path (preferred): if Rabbit + storage are available, gate enqueues an ingestion task (type=index) and returns 202 Accepted. A worker updates storage.extra.ingestion state (queued → processing → done/failed).
Worker → doc-processor: calls POST /v1/process which:
Fetches bytes + metadata from storage.
Extraction:
PDF/DOC/DOCX → normalize to PDF → render pages → VLM per page → page texts.
XML/HTML/XLSX/plain → non‑VLM parsing/decoding (HTML/XLSX converted to Markdown-ish text).
Chunking: splits text into chunks (keeps Markdown tables/code blocks intact), adds page locator when available.
Index upsert: sends chunks to retrieval service (/v1/index/upsert).
Retrieval service indexing:
Stores text + metadata in OpenSearch (BM25).
Computes embeddings (with content-hash idempotency) and stores vectors + payload in Qdrant (vector search).
Retrieval + Answering
Client → gate: POST /v1/chat (or /v1/chat/stream).
gate → retrieval service: POST /v1/search with mode bm25 | vector | hybrid, filters (doc_ids/tags/lang/source/tenant/project), acl, top_k, optional rerank, and include_sources.
Retrieval service:
BM25 from OpenSearch and/or vector from Qdrant (in parallel for hybrid).
Fusion: hybrid score fusion (rank-based RRF + weighted score blending).
Optional rerank: cross-encoder reranker on a candidate pool.
Optional quality steps: group-by-doc limit, page deduplication, and “parent page” expansion (replace chunk text with full page text assembled from all chunks of that page).
Returns ranked chunks + optional source metadata.
gate:
Builds a context window (bounded by max_context_chars) from top hits and constructs the LLM prompt/messages.
Calls the LLM; if sources are requested, it enforces inline citations (one strict rewrite attempt, otherwise returns “I don’t know”).