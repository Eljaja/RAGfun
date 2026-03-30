# RAG Gate v2 SDK

This SDK targets the `gate_v2` API.

Default target:
- `http://localhost:8917`

Auth:
- pass your token as `Bearer` via `ClientAuth(api_key="sk-...")`

## Install

```bash
pip install -r client/requirements.txt
```

## Quick Start

```python
from client import ClientAuth, RAGOpenAIClient

client = RAGOpenAIClient(
    base_url="http://localhost:8917",
    auth=ClientAuth(api_key="sk-..."),
)

project = client.projects.ensure(
    name="my-project",
    description="SDK-created project",
)

resp = client.chat.completions.create(
    project_id=project["project_id"],
    messages=[{"role": "user", "content": "Summarize project docs"}],
)
print(resp["choices"][0]["message"]["content"])
client.close()
```

## API Style

- `client.chat.completions.create(...)` uses chat-completions style
- input uses `messages=[{role, content}, ...]`
- output includes `choices[0].message.content`

Additionally, the response includes `rag` block with original gateway data:
- `sources`, `context`, `partial`, `degraded`, `raw`

## Streaming

```python
events = client.chat.completions.create(
    project_id=project["project_id"],
    messages=[{"role": "user", "content": "Give me key points"}],
    stream=True,
)
for chunk in events:
    delta = chunk["choices"][0]["delta"].get("content", "")
    if delta:
        print(delta, end="")
```

## Project APIs

- `client.projects.create(name, description=None)`
- `client.projects.ensure(name, description=None)` (reuse by name if project limit is reached)
- `client.projects.list()`
- `client.projects.get(project_id)`
- `client.projects.delete(project_id)`

## Upload helper

`client.upload_document(project_id=..., file_path=..., title=..., description=...)`

## Example script

```bash
export RAG_API_KEY="sk-..."
python client/examples/create_project_and_chat.py
```

## Diagrams

Mermaid diagrams for SDK flows:
- `client/MERMAID_DOCS.md`
