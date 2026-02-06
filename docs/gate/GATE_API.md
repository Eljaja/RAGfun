# Gate API

Default base URL: `http://localhost:8090`

Use a variable in examples:

```bash
export GATE_URL="http://localhost:8090"
```

## Health / Ready / Meta

### `GET /v1/healthz`
Liveness check.

**Response 200**

```json
{"ok": true}
```

### `GET /v1/readyz`
Readiness check. Verifies gate loaded config and retrieval is available.

- Returns **503** on failure.

**Response 200/503**

```json
{
  "ready": true,
  "retrieval": {"ready": true}
}
```

### `GET /v1/version`
Returns service name and safe config summary (no secrets).

### `GET /v1/metrics`
Prometheus metrics (content-type: `text/plain; version=0.0.4`).

## Chat

### `POST /v1/chat`
Non-streaming chat. Retrieval → build context → call LLM → return answer + (optional) sources and retrieval debug payload.

**JSON body (`ChatRequest`)**

- **query**: `string` — user query
- **history**: `[{role, content}]` — chat history (optional)
- **retrieval_mode**: `"bm25" | "vector" | "hybrid"` (optional; else config default)
- **top_k**: `int` (optional)
- **rerank**: `bool` (optional)
- **filters**: filter object (optional)
  - **source**: `string|null`
  - **tags**: `string[]|null`
  - **lang**: `string|null`
  - **doc_ids**: `string[]|null`
  - **tenant_id**: `string|null`
  - **project_id**: `string|null`
  - **project_ids**: `string[]|null` (multiple collections)
- **acl**: `string[]` (optional)
- **include_sources**: `bool` (default `true`)

**Example**

```bash
curl -sS "$GATE_URL/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the gate in this project?",
    "top_k": 8,
    "include_sources": true
  }' | jq .
```

**Response 200 (`ChatResponse`)**

- **ok**: `bool`
- **answer**: `string`
- **used_mode**: `string`
- **degraded**: `string[]`
- **partial**: `bool`
- **context**: context chunks (with `chunk_id/doc_id/text/score` and `source` if `include_sources=true`)
- **sources**: sources array (if `include_sources=true`)
- **retrieval**: raw retrieval debug payload (for UI/tracing)

**Errors**

- **503**/`config_error`: gate failed to load config
- `retrieval_timeout` / `retrieval_error:*`: retrieval issues

### `POST /v1/chat/stream`
Streaming chat via SSE (`text/event-stream`).

Server sends events:

- `{"type":"retrieval", ...}` — once after retrieval (includes `context` and `retrieval`)
- `{"type":"token","content":"..."}` — multiple times, answer tokens
- `{"type":"done","answer":"...","sources":[...],...}` — final
- `{"type":"error","error":"..."}` — error

**Example**

```bash
curl -N -sS "$GATE_URL/v1/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Summarize the project architecture.",
    "include_sources": true
  }'
```

## Documents

> Note: Gate works with documents via `document-storage` when configured. If `storage_url` is not set, many document endpoints return `storage_unavailable`.

> Note on `doc_id`: in URLs it must be **url-encoded** (e.g. `#`, spaces).

### `POST /v1/documents/upload`
Multipart upload.

**Form fields**

- **file** (required): file
- **doc_id** (required): string
- **title**: string
- **uri**: string
- **source**: string
- **lang**: string
- **tags**: comma-separated string (e.g. `tag1,tag2`)
- **acl**: comma-separated string
- **tenant_id**: string
- **project_id**: string (used as collection)
- **refresh**: bool (default `false`)

**Response behavior**

- If `document-storage` + RabbitMQ publisher are configured — upload is **queued**, response **202** (`accepted=true`) and `task_id`.
- Otherwise legacy path: gate reads file, UTF-8 decode (HTML → text via `html_to_text`), indexes directly via retrieval — response **200**.

### `GET /v1/documents`
List documents from `document-storage` + batch `indexed` check via `retrieval.index_exists` (if retrieval available).

Query params: `source`, `tags` (comma-separated), `lang`, `collections` (comma-separated project_ids), `limit` (default 100), `offset` (default 0).

### `GET /v1/documents/{doc_id}/status`
Returns: `stored`, `indexed`, `metadata`, `ingestion`.

### `GET /v1/documents/stats`
Server-side document statistics aggregation.

### `GET /v1/collections`
List collections (distinct `project_id`) from `document-storage`.

### `DELETE /v1/documents/{doc_id}`
Delete document. With document-storage + RabbitMQ: enqueue delete, **202**. Otherwise best-effort delete from storage and retrieval.

### `DELETE /v1/documents?confirm=true`
Delete **all documents** (requires `confirm=true`).

## Notes

- **`doc_id` with `#`/spaces**: use urlencode in URLs.
- **`/v1/documents` and `.../status` require storage**: else `storage_unavailable` or `stored=false`.
- **Upload may return 202**: normal when async ingestion pipeline is enabled.
