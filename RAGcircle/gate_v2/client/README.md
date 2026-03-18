# Gateway SDK Outline

This folder contains a thin Python SDK scaffold for the single nginx gateway (default `http://localhost:8916`).

## Covered API surface

- Chat (rag-gate): `POST /api/v1/chat`, `POST /api/v1/chat/stream`
- Agent stream: `POST /agent-api/v1/agent/stream`
- Storage / ingestion (`gate_v2`) under `/storage-api/api/v1/...`:
  - projects: create/list/get/delete
  - documents: list/upload/get/status/delete/download

## Quick usage

```python
from gate_v2.client import (
    ChatRequest,
    ChatStreamRequest,
    ClientAuth,
    RagGatewayClient,
)

client = RagGatewayClient(
    base_url="http://localhost:8916",
    auth=ClientAuth(bearer_token="your-token"),
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
        filters={"project_ids": ["default"]},
    )
)
print(resp.answer)

client.close()
```

## `client.py` smoke test

`client.py` now runs a small end-to-end check:

1. writes/uses `pineapple_secret_message.txt`
2. uploads it to a project
3. waits briefly for status
4. sends a short chat query via rag-gate about the secret message

Optional env vars:

- `RAG_GATEWAY_URL` (default `http://localhost:8916`)
- `RAG_BEARER_TOKEN` (if auth is enabled)
- `RAG_PROJECT_ID` (default `default`)

## Full methods test script

Use `test_gateway_methods.py` to call the unified endpoint methods with a pass/fail summary:

```bash
python test_gateway_methods.py
```

Optional env vars:

- `RAG_GATEWAY_URL` (default `http://localhost:8916`)
- `RAG_BEARER_TOKEN` (if auth is enabled)
- `RAG_PROJECT_ID` (default `default`)
- `RAG_RUN_STREAM_CHECKS` (`true|false`, default `false`)
- `RAG_CLEANUP_DOC` (`true|false`, default `false`)

## Notes

- Models are intentionally permissive (`extra="allow"`) to make this a stable starting point while APIs evolve.
- Streaming returns parsed SSE event objects (`dict`).
- This is an outline scaffold; retries/backoff, richer error taxonomy, and async variants can be added next.

