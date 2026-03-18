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

## Project layout

```text
client/
  sdk.py
  __init__.py
  requirements.txt
  pineapple_secret_message.txt
  examples/
    basic_chat_stream.py
    basic_agent_stream.py
    upload_and_document_status.py
  tests/
    test_gateway_methods.py
    test_default_project_policy.py
    test_streaming_endpoints.py
    run_all.py
```

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
    AgentStreamRequest,
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

# Agent stream example
agent_payload = AgentStreamRequest(
    query="What is the pineapple secret message? Return quote only.",
    include_sources=True,
    filters={"project_ids": [client.default_project_id]},
)
for event in client.agent_stream(agent_payload):
    print(event)
    if event.get("type") in {"done", "error"}:
        break

client.close()
```

## Examples

### Basic chat stream

```bash
python examples/basic_chat_stream.py
```

### Basic agent stream

```bash
python examples/basic_agent_stream.py
```

### Upload + list documents + status checks

```bash
python examples/upload_and_document_status.py
```

Demonstrates:
- upload a file to `default`
- `409` duplicate handling on upload
- listing documents with `list_project_documents()`
- checking processing status with `get_document_status()`

## Tests

### Run all tests

```bash
python tests/run_all.py
```

### Run a single test

```bash
python tests/test_gateway_methods.py
python tests/test_default_project_policy.py
python tests/test_streaming_endpoints.py
```

What each test covers:
- `test_gateway_methods.py`: end-to-end methods/health/storage + optional stream checks
- `test_default_project_policy.py`: enforces default-only/read-only behavior
- `test_streaming_endpoints.py`: validates chat + agent SSE flows reach `done`

Common env vars for examples and tests:
- `RAG_GATEWAY_URL` (default `http://202.181.159.221:8916`)
- `RAG_BEARER_TOKEN` (if auth is enabled)
- `RAG_PROJECT_ID` (ignored if not `default`; single-project mode)
- `RAG_STREAM_MAX_EVENTS` (stream test cap, default `500`)
- `RAG_RUN_STREAM_CHECKS` (`test_gateway_methods.py`, default `false`)
- `RAG_CLEANUP_DOC` (`test_gateway_methods.py`, default `false`)

## Notes

- Models are intentionally permissive (`extra="allow"`) to make this a stable starting point while APIs evolve.
- Streaming returns parsed SSE event objects (`dict`).
- This is an outline scaffold; retries/backoff, richer error taxonomy, and async variants can be added next.

