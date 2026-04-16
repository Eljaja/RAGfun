# Gateway SDK

Typed Python client for **gate_v2** (`RAGcircle/gate_v2`).

Models mirror `generator_contract` — the shared schema used by both the gate and the generator service.

## Quick start

```python
from client.sdk import RagGatewayClient, ClientAuth, AgentRequest

client = RagGatewayClient(
    base_url="http://localhost:8918",
    auth=ClientAuth(bearer_token="sk-..."),
)

# list projects
projects = client.list_projects()

# upload a document
upload = client.upload_document("my-project", "report.pdf", title="Q1 Report")

# check processing status
status = client.get_document_status(upload.doc_id)

# agent chat (full pipeline)
answer = client.agent_chat(AgentRequest(
    project_id="my-project",
    query="Summarize the Q1 report",
))
print(answer.answer)

client.close()
```

## Endpoints covered

### Health

| Method | Gate path | SDK method |
|--------|-----------|------------|
| GET | `/public/health` | `health()` |

### Projects

| Method | Gate path | SDK method |
|--------|-----------|------------|
| POST | `/api/v1/projects` | `create_project(payload)` |
| GET | `/api/v1/projects` | `list_projects()` |
| GET | `/api/v1/projects/{id}` | `get_project(project_id)` |
| DELETE | `/api/v1/projects/{id}` | `delete_project(project_id)` |

### Documents

| Method | Gate path | SDK method |
|--------|-----------|------------|
| GET | `/api/v1/projects/{id}/documents` | `list_project_documents(project_id)` |
| POST | `/api/v1/projects/{id}/upload` | `upload_document(project_id, file_path)` |
| GET | `/api/v1/documents/{id}` | `get_document(doc_id)` |
| GET | `/api/v1/documents/{id}/status` | `get_document_status(doc_id)` |
| GET | `/api/v1/documents/{id}/download` | `download_document(doc_id)` |
| DELETE | `/api/v1/documents/{id}` | `delete_document(doc_id)` |

### Chat

| Method | Gate path | SDK method | Generator endpoint |
|--------|-----------|------------|--------------------|
| POST | `/api/v1/chat` | `agent_chat(payload)` | `/agent` |
| POST | `/api/v1/chat/stream` | `agent_chat_stream(payload)` | `/agent/stream` |
| POST | `/api/v1/simple-chat` | `simple_chat(payload)` | `/chat` |
| POST | `/api/v1/simple-chat/stream` | `simple_chat_stream(payload)` | `/chat/stream` |

## Reverse proxy support

By default the SDK talks directly to the gate (`/api/v1/...`).
If the gate sits behind nginx with a path prefix, pass `path_prefix`:

```python
client = RagGatewayClient(
    base_url="http://example.com:8916",
    path_prefix="/storage-api",
    auth=ClientAuth(bearer_token="sk-..."),
)
```

## Running tests (pytest)

The test suite is a set of **integration tests** that hit a live gate.
You only need to provide a token (or nothing at all in stub-auth mode):

```bash
pip install pytest

# with a real token
RAG_BEARER_TOKEN=sk-... pytest tests/ -v

# stub-auth mode (token defaults to "stub")
pytest tests/ -v
```

The suite auto-discovers an existing project (or creates one),
uploads a test document, waits for it to be indexed, then runs
chat tests against it. No manual setup needed.

### What each test file covers

| File | What it tests |
|------|---------------|
| `test_health.py` | Gate is reachable (`/public/health`) |
| `test_project_crud.py` | Create → get → list → delete project lifecycle |
| `test_upload_flow.py` | Upload txt/pdf → poll status → assert indexed, duplicate 409 |
| `test_chat.py` | `agent_chat`, `simple_chat`, streaming — all depend on `uploaded_doc` fixture |

### Useful pytest flags

```bash
# stop on first failure, show locals
pytest tests/ -x -l

# only upload tests
pytest tests/test_upload_flow.py -v

# only health smoke test
pytest tests/test_health.py -v

# generate HTML report (pip install pytest-html)
pytest tests/ --html=report.html
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_BEARER_TOKEN` | `stub` | Bearer token |
| `RAG_GATEWAY_URL` | `http://localhost:8918` | Gate base URL (optional override) |

## Project layout

```
client/
  sdk.py              # client + models
  __init__.py
  requirements.txt
  README.md
  examples/
    upload_and_document_status.py
  tests/
    conftest.py             # fixtures: client, project_id, sample files, poll helper
    test_health.py          # smoke test
    test_project_crud.py    # project lifecycle
    test_upload_flow.py     # upload + status polling
    test_chat.py            # agent_chat, simple_chat, streaming
```
