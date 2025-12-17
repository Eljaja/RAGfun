## RAG pipeline (Gate → Doc Processor → Retrieval Service)

This repository implements a 3-service RAG pipeline:

- **`gate`**: API entrypoint for upload + chat, orchestration.
- **`doc-processor`**: file-to-text extraction + chunking.
- **`service`** (retrieval): indexing + hybrid search (BM25/vector) + optional rerank.

## Ingestion (indexing)

- **Client → `gate`**: upload document via `POST /v1/documents/upload`
  - Required: `doc_id`
  - Optional metadata: `tags`, `acl`, `tenant_id`, `project_id`, `lang`, `source`, `uri`, `title`
- **`gate` → `document-storage` (optional)**: streams raw bytes and stores metadata.
- **Async path (preferred)**: if RabbitMQ + storage are available:
  - `gate` publishes an ingestion task (`type=index`) and returns **`202 Accepted`**
  - ingestion status is tracked in `storage.extra.ingestion` (`queued → processing → done/failed`)
- **Worker → `doc-processor`**: calls `POST /v1/process`, which:
  - **Fetches** metadata + file bytes from storage
  - **Extracts text**
    - PDF/DOC/DOCX → normalize to PDF → render pages → VLM per page → page texts
    - XML/HTML/XLSX/plain → non‑VLM parsing/decoding (HTML/XLSX converted to Markdown-ish text)
  - **Chunks** text (preserves Markdown tables / fenced code blocks); adds page locator when available
  - **Upserts index** in retrieval service via `POST /v1/index/upsert` (`mode=chunks`)
- **Retrieval service indexing (`service`)**
  - **OpenSearch**: stores chunk text + metadata (BM25)
  - **Qdrant**: stores embeddings + payload (vector search), with content-hash based idempotency

## Retrieval + Answering

- **Client → `gate`**: ask a question via `POST /v1/chat` (or streaming `POST /v1/chat/stream`)
- **`gate` → retrieval service**: `POST /v1/search` with:
  - `mode`: `bm25 | vector | hybrid`
  - `filters`: `doc_ids`, `tags`, `lang`, `source`, `tenant_id`, `project_id(s)` (+ `acl`)
  - `top_k`, optional `rerank`, optional `include_sources`
- **Retrieval service (`service`)**
  - BM25 search (OpenSearch) and/or vector search (Qdrant), parallel for `hybrid`
  - Fusion: hybrid fusion (rank-based RRF + weighted score blending)
  - Optional rerank (cross-encoder) on a bounded candidate pool
  - Optional quality steps: per-doc grouping, page deduplication, “parent page” expansion (replace chunk text with full page text)
  - Returns ranked hits + optional source metadata
- **`gate`**
  - Builds the context window (bounded by `max_context_chars`) and constructs the prompt/messages
  - Calls the LLM; if `include_sources=true`, enforces inline citations (one strict rewrite attempt, otherwise returns `"I don't know"`)
