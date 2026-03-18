# Gateway SDK Outline

This folder contains a thin Python SDK scaffold for the single nginx gateway (default `http://202.181.159.221:8916`).

## Single-project policy

The SDK now enforces a single project/collection:

- default project ID: `default`
- project writes are disabled (`create_project`, `delete_project` raise `SDKError`)
- project-scoped methods reject non-`default` IDs
- `list_projects()` is filtered to only `default`
- chat endpoints strip filters (to avoid known `/api` filter mismatch)
- agent endpoint keeps filters but normalizes scope to `project_ids=["default"]`

## Exposed endpoints

### Chat and agent endpoints

- `POST /api/v1/chat` -> `RagGatewayClient.chat()`
  - SDK behavior: drops `filters` for this endpoint
- `POST /api/v1/chat/stream` -> `RagGatewayClient.chat_stream()`
  - SDK behavior: drops `filters` for this endpoint
- `POST /agent-api/v1/agent/stream` -> `RagGatewayClient.agent_stream()`
  - SDK behavior: enforces `filters.project_ids=["default"]`

### Storage and project endpoints

- `GET /storage-api/api/v1/projects` -> `RagGatewayClient.list_projects()`
  - SDK behavior: returns only the `default` project in response
- `GET /storage-api/api/v1/projects/{project_id}` -> `RagGatewayClient.get_project()`
  - SDK policy: only `project_id="default"`
- `GET /storage-api/api/v1/projects/{project_id}/documents?limit=&offset=`
  -> `RagGatewayClient.list_project_documents()`
  - SDK policy: only `project_id="default"`
- `POST /storage-api/api/v1/projects/{project_id}/upload`
  -> `RagGatewayClient.upload_document()`
  - SDK policy: only `project_id="default"`
  - `409 Conflict` is acceptable when the same document/content is uploaded again
    (duplicate upload attempt)
- `GET /storage-api/api/v1/documents/{doc_id}` -> `RagGatewayClient.get_document()`
- `GET /storage-api/api/v1/documents/{doc_id}/status` -> `RagGatewayClient.get_document_status()`
- `DELETE /storage-api/api/v1/documents/{doc_id}` -> `RagGatewayClient.delete_document()`
- `GET /storage-api/api/v1/documents/{doc_id}/download` -> `RagGatewayClient.download_document()`

### Health/readiness endpoints used by scripts

- `GET /public/health`
- `GET /storage-api/public/health`
- `GET /api/v1/readyz`
- `GET /agent-api/v1/readyz`

### Handling duplicate uploads (`409`)

```python
from gate_v2.client import APIError, ClientAuth, RagGatewayClient

client = RagGatewayClient(
    base_url="http://202.181.159.221:8916",
    auth=ClientAuth(bearer_token="stub"),
)

try:
    resp = client.upload_document("default", "pineapple_secret_message.txt")
    print("uploaded:", resp.doc_id)
except APIError as exc:
    if exc.status_code == 409:
        # Duplicate upload is expected in idempotent flows.
        print("duplicate upload detected, continuing")
    else:
        raise
finally:
    client.close()
```

## Quick usage

```python
from gate_v2.client import (
    ChatRequest,
    ChatStreamRequest,
    ClientAuth,
    RagGatewayClient,
)

client = RagGatewayClient(
    base_url="http://202.181.159.221:8916",
    auth=ClientAuth(bearer_token="stub"),
)

payload = ChatStreamRequest(query="What documents are available?")
for event in client.chat_stream(payload):
    print(event)
    if event.get("type") in {"done", "error"}:
        break

# Non-stream chat (actual chat method)
resp = client.chat(
    ChatRequest(
        query="Give me a short summary",
        include_sources=True,
    )
)
print(resp.answer)

client.close()
```

## `client.py` smoke test

`client.py` now runs a small end-to-end check:

1. writes/uses `pineapple_secret_message.txt`
2. uses the existing `default` project
3. uploads the file
4. waits briefly for status
5. sends a short chat query via rag-gate about the secret message

Optional env vars:

- `RAG_GATEWAY_URL` (default `http://202.181.159.221:8916`)
- `RAG_BEARER_TOKEN` (if auth is enabled)
- `RAG_PROJECT_ID` (ignored if not `default`; single-project mode)

## Full methods test script

Use `test_gateway_methods.py` to call the unified endpoint methods with a pass/fail summary:

```bash
python test_gateway_methods.py
```

Optional env vars:

- `RAG_GATEWAY_URL` (default `http://202.181.159.221:8916`)
- `RAG_BEARER_TOKEN` (if auth is enabled)
- `RAG_PROJECT_ID` (ignored if not `default`; single-project mode)
- `RAG_RUN_STREAM_CHECKS` (`true|false`, default `false`)
- `RAG_CLEANUP_DOC` (`true|false`, default `false`)

## Additional scripts

### Read-only example

```bash
python example_default_project_readonly.py
```

Demonstrates:
- list/get project (default only)
- list default project documents
- run a chat request

### Default-policy checks

```bash
python test_default_project_policy.py
```

Verifies:
- happy-path reads on `default`
- blocking of project writes
- blocking of non-default project IDs
- duplicate upload behavior can return `409` and is considered expected

### Streaming endpoint checks

```bash
python test_streaming_endpoints.py
```

Verifies:
- `chat_stream` completes with SSE `done`
- `agent_stream` completes with SSE `done`
- no `error` events are emitted
- final answer payload is non-empty

Optional env vars:
- `RAG_STREAM_MAX_EVENTS` (default `500`)

## Notes

- Models are intentionally permissive (`extra="allow"`) to make this a stable starting point while APIs evolve.
- Streaming returns parsed SSE event objects (`dict`).
- This is an outline scaffold; retries/backoff, richer error taxonomy, and async variants can be added next.

