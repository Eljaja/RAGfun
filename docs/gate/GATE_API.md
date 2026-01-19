# Gate API (локально)

Базовый URL по умолчанию: `http://localhost:8090`

Во всех примерах ниже можно использовать переменную:

```bash
export GATE_URL="http://localhost:8090"
```

## Health / Ready / Meta

### `GET /v1/healthz`
Liveness check.

**Ответ 200**

```json
{"ok": true}
```

### `GET /v1/readyz`
Readiness check. Проверяет, что `gate` успешно загрузил конфиг и что доступен `retrieval` (через `retrieval.readyz()`).

- При проблемах возвращает **503**.

**Ответ 200/503**

```json
{
  "ready": true,
  "retrieval": {"ready": true}
}
```

### `GET /v1/version`
Возвращает имя сервиса и “безопасное” резюме конфига (секреты не включаются).

### `GET /v1/metrics`
Prometheus metrics (content-type: `text/plain; version=0.0.4`).

## Chat

### `POST /v1/chat`
Нестримащий чат. Делает retrieval → собирает контекст → вызывает LLM → возвращает ответ + (опционально) источники и debug-пэйлоад retrieval.

**JSON body (`ChatRequest`)**

- **query**: `string` — пользовательский запрос
- **history**: `[{role, content}]` — история диалога (опционально)
- **retrieval_mode**: `"bm25" | "vector" | "hybrid"` (опционально; иначе берётся дефолт из конфига)
- **top_k**: `int` (опционально)
- **rerank**: `bool` (опционально)
- **filters**: объект фильтров (опционально)
  - **source**: `string|null`
  - **tags**: `string[]|null`
  - **lang**: `string|null`
  - **doc_ids**: `string[]|null`
  - **tenant_id**: `string|null`
  - **project_id**: `string|null`
  - **project_ids**: `string[]|null` (несколько “коллекций”)
- **acl**: `string[]` (опционально)
- **include_sources**: `bool` (по умолчанию `true`)

**Пример**

```bash
curl -sS "$GATE_URL/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Что такое gate в этом проекте?",
    "top_k": 8,
    "include_sources": true
  }' | jq .
```

**Ответ 200 (`ChatResponse`)**

- **ok**: `bool`
- **answer**: `string`
- **used_mode**: `string`
- **degraded**: `string[]`
- **partial**: `bool`
- **context**: массив чанков контекста (с `chunk_id/doc_id/text/score` и `source`, если `include_sources=true`)
- **sources**: массив источников (если `include_sources=true`)
- **retrieval**: сырой debug-пэйлоад retrieval (для UI/трассировки)

**Ошибки (типовые)**

- **503**/`config_error`: gate не смог загрузить конфиг
- `retrieval_timeout` / `retrieval_error:*`: проблемы при вызове retrieval

### `POST /v1/chat/stream`
Streaming чат через SSE (`text/event-stream`).

Сервер отправляет события вида:

- `{"type":"retrieval", ...}` — один раз после retrieval (включает `context` и `retrieval`)
- `{"type":"token","content":"..."}` — много раз, токены ответа
- `{"type":"done","answer":"...","sources":[...],...}` — финал
- `{"type":"error","error":"..."}` — ошибка

**Пример**

```bash
curl -N -sS "$GATE_URL/v1/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Сделай краткое резюме архитектуры проекта.",
    "include_sources": true
  }'
```

## Documents

> Важно: `gate` работает с документами через `document-storage`, если он настроен. Если `storage_url` не задан — многие document-эндпоинты вернут `storage_unavailable`.

> Важно про `doc_id`: в URL он должен быть **url-encoded** (например `#`, пробелы и т.п.).

### `POST /v1/documents/upload`
Multipart upload.

**Form fields**

- **file** (обязательно): файл
- **doc_id** (обязательно): строка
- **title**: строка
- **uri**: строка
- **source**: строка
- **lang**: строка
- **tags**: строка, comma-separated (например `tag1,tag2`)
- **acl**: строка, comma-separated
- **tenant_id**: строка
- **project_id**: строка (используется как “collection”)
- **refresh**: bool (по умолчанию `false`)

**Поведение ответа**

- Если настроены `document-storage` + RabbitMQ publisher — загрузка **ставится в очередь**, ответ **202** (`accepted=true`) и `task_id`.
- Иначе используется legacy path: gate читает файл, пытается UTF-8 decode (HTML → text через `html_to_text`), и индексирует напрямую через retrieval — ответ **200**.

### `GET /v1/documents`
Список документов из `document-storage` + batch-проверка `indexed` через `retrieval.index_exists` (если retrieval доступен).

Query params:

- `source`, `tags` (comma-separated), `lang`
- `collections` (comma-separated project_ids)
- `limit` (default 100), `offset` (default 0)

### `GET /v1/documents/{doc_id}/status`
Возвращает:

- `stored`: есть ли метаданные в `document-storage`
- `indexed`: проиндексирован ли документ в retrieval (через `index_exists`)
- `metadata`: метаданные из storage (если есть)
- `ingestion`: `metadata.extra.ingestion` (если есть)

### `GET /v1/documents/stats`
Серверная агрегация статистики по документам (чтобы UI не загружал всё целиком).

Полезно для проверки ingestion состояний: `queued/processing/retrying/failed/completed/unknown`.

### `GET /v1/collections`
Список “коллекций” (distinct `project_id`) из `document-storage`.

Query params:

- `tenant_id` (опционально)
- `limit` (default 1000)

### `DELETE /v1/documents/{doc_id}`
Удаление документа:

- Если настроены `document-storage` + RabbitMQ publisher: **enqueue delete**, ответ **202**.
- Иначе best-effort: удалить из storage (если настроен) и удалить из retrieval (обязательно).
  - Если одна из сторон упала — ответ будет `partial=true` и `degraded=[...]` (HTTP 207 не выставляется явно как статус, но внутри метрики используется).

### `DELETE /v1/documents?confirm=true`
Удаляет **все документы** (safety: нужен `confirm=true`).

Query params:

- `confirm=true` (обязательно)
- `batch_size` (<=1000), `concurrency` (<=50), `max_batches`

## Примечания / частые проблемы

- **`doc_id` с `#`/пробелами**: в URL используйте urlencode.
- **`/v1/documents` и `.../status` требуют storage**: иначе `storage_unavailable` либо поля `stored=false`.
- **upload может вернуть 202**: это нормально при включённой асинхронной ingestion-пайплайне.

