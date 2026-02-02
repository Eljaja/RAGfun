# rag_fun Gate Python SDK

Клиентский SDK для взаимодействия только с `gate` (FastAPI gateway).

## Установка зависимостей

```bash
pip install httpx
```

## Быстрый старт (sync)

```python
from ragfun_sdk import GateClient

with GateClient(base_url="http://localhost:8090") as client:
    ready = client.readyz()
    print(ready)

    resp = client.chat(query="Что такое gate в этом проекте?", top_k=5)
    print(resp.answer)
```

## Streaming chat (SSE)

```python
from ragfun_sdk import GateClient

client = GateClient()
for event in client.chat_stream(query="Сделай краткое резюме архитектуры проекта."):
    if event.type == "token" and event.data:
        print(event.data.get("content", ""), end="")
    if event.type == "done":
        print("\n--- done ---")
```

## Upload + статус документа

```python
from ragfun_sdk import GateClient

client = GateClient()
upload = client.upload(
    file_path="./README.md",
    doc_id="doc-1",
    title="Repo README",
    tags=["readme", "docs"],
)
print(upload)

status = client.document_status(doc_id="doc-1")
print(status)
```

## Async client

```python
import asyncio
from ragfun_sdk import GateAsyncClient

async def main() -> None:
    async with GateAsyncClient() as client:
        resp = await client.chat(query="Привет!")
        print(resp.answer)

asyncio.run(main())
```

## Методы

- `healthz()`, `readyz()`, `version()`, `metrics()`
- `chat(...)`, `chat_stream(...)`
- `upload(...)`
- `list_documents(...)`
- `document_status(...)`
- `documents_stats(...)`
- `collections(...)`
- `delete_document(...)`
- `delete_all_documents(confirm=True, ...)`
