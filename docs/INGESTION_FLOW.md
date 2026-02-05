# Как в проекте добавляются документы (ingestion)

Процесс добавления документа в RAG: от загрузки файла до появления чанков в поисковом индексе (OpenSearch + Qdrant).

---

## Общая схема

```
Клиент  →  Gate (upload)  →  document-storage (сохранение)
                    ↓
              RabbitMQ (очередь)
                    ↓
         ingestion-worker  →  doc-processor (/v1/process)
                                    ↓
                    document-storage (чтение файла по doc_id)
                                    ↓
                         извлечение текста (VLM или без)
                                    ↓
                              чанкинг
                                    ↓
                    retrieval (POST /v1/index/upsert, mode=chunks)
                                    ↓
                    OpenSearch + Qdrant (индекс)
```

Есть **два пути** индексации после загрузки в Gate:

1. **Асинхронный (основной)** — Gate сохраняет файл в storage, кладёт задачу в RabbitMQ и сразу возвращает `202 Accepted`. Обработку делает **ingestion-worker** + **doc-processor**.
2. **Legacy (синхронный)** — если RabbitMQ или storage недоступны, Gate сам декодирует файл как UTF-8 и отправляет текст в **retrieval** одним куском (`mode=document`); retrieval сам режет на чанки и индексирует.

---

## Шаг 1: Загрузка в Gate

**Эндпоинт:** `POST /v1/documents/upload`

**Параметры (multipart/form-data):**

| Параметр   | Обязательный | Описание |
|-----------|--------------|----------|
| `file`    | да           | Файл (PDF, DOCX, HTML, TXT и т.д.) |
| `doc_id`  | да           | Уникальный идентификатор документа (используется в индексе и фильтрах) |
| `title`   | нет          | Заголовок |
| `uri`     | нет          | URI источника |
| `source`  | нет          | Источник |
| `lang`    | нет          | Язык |
| `tags`   | нет          | Теги (через запятую) |
| `acl`     | нет          | ACL (через запятую) |
| `tenant_id`  | нет       | Tenant |
| `project_id` | нет       | Project |
| `refresh` | нет (false)  | Сразу обновить индекс после индексации |

**Пример (curl):**

```bash
curl -X POST http://localhost:8092/v1/documents/upload \
  -F "file=@/path/to/doc.pdf" \
  -F "doc_id=doc-$(date +%s)" \
  -F "title=My Document"
```

**Ответ при успешной постановке в очередь (асинхронный путь):**

- Код: **202 Accepted**
- Тело: `{"ok": true, "accepted": true, "task_id": "...", "doc_id": "...", "storage": {...}, "filename": "...", "bytes": N}`

**Ответ при legacy-индексации (синхронно):**

- Код: **200 OK**
- Тело: `{"ok": true, "result": {...}, "storage": {...}, "filename": "...", "bytes": N}`

---

## Шаг 2: Сохранение в document-storage

Gate стримит загруженный файл в **document-storage** и сохраняет метаданные (doc_id, title, tags и т.д.).

- В стеке **rugfunsota** по умолчанию используется бэкенд **local** (файлы в томе), без внешнего RustFS/S3. Для S3 задайте `STORAGE_STORAGE_BACKEND=s3` и поднимите RustFS.
- Если storage недоступен, Gate всё равно попытается проиндексировать документ (legacy-путём), если это возможно.
- В storage позже пишется статус ingestion: `queued` → `processing` → `done` / `failed` (через `patch_extra`).

---

## Шаг 3: Очередь RabbitMQ (асинхронный путь)

Если настроены **RabbitMQ** и **document-storage**, Gate публикует в очередь задачу:

```json
{
  "task_id": "uuid",
  "type": "index",
  "doc_id": "…",
  "document": { "doc_id", "title", "uri", "source", "lang", "tags", "acl", "tenant_id", "project_id" },
  "refresh": false,
  "attempt": 0,
  "queued_at": 1234567890
}
```

Очередь по умолчанию: `ingestion.tasks` (конфиг Gate: `rabbit_queue`).

---

## Шаг 4: Ingestion-worker

**ingestion-worker** (часть doc-processor или отдельный сервис) подписан на очередь:

1. Получает задачу `type=index`.
2. Вызывает **doc-processor** по HTTP: `POST {DOC_PROCESSOR_URL}/v1/process` с телом:
   ```json
   {
     "document": { "doc_id", "title", … },
     "refresh": false
   }
   ```
3. По ответу doc-processor обновляет в storage статус ingestion (`processing` → `done` или `failed`).

Worker не передаёт байты файла: doc-processor сам достаёт файл из storage по `doc_id`.

---

## Шаг 5: Doc-processor (`POST /v1/process`)

Doc-processor получает только метаданные документа. Дальше:

1. **Чтение из storage**
   - `GET` метаданные: `storage.get_metadata(doc_id)`.
   - Скачивание файла: `storage.get_file(doc_id)` → байты + content-type.

2. **Извлечение текста**
   - PDF/DOC/DOCX: нормализация в PDF → рендер страниц в PNG → **VLM** (vLLM Docling) по страницам → текст по страницам.
   - XML/HTML/XLSX/plain: парсинг без VLM (HTML → Markdown-подобный текст и т.д.).

3. **Чанкинг**
   - Текст режется по выбранной стратегии (`PROCESSOR_CHUNK_STRATEGY`: `semantic` или `fixed`).
   - Для каждого чанка формируется `ChunkMeta`: `doc_id`, `chunk_index`, `text`, `locator` (page и т.д.), метаданные документа.

4. **Отправка в retrieval**
   - `POST {RETRIEVAL_URL}/v1/index/upsert` с телом:
   ```json
   {
     "mode": "chunks",
     "document": { "doc_id", "title", … },
     "chunks": [ { "chunk_id", "doc_id", "chunk_index", "text", "locator", … } ],
     "refresh": false
   }
   ```

---

## Шаг 6: Retrieval (индексация в OpenSearch и Qdrant)

**Эндпоинт:** `POST /v1/index/upsert`

**Режимы:**

| Режим        | Кто вызывает        | Описание |
|-------------|----------------------|----------|
| `chunks`    | doc-processor        | Готовый список чанков; retrieval эмбеддит текст, пишет в OpenSearch (BM25) и Qdrant (векторы). |
| `document`  | Gate (legacy)        | Один блок текста; retrieval сам режет на чанки (`chunk_by_strategy` из service) и индексирует. |

В обоих случаях:

- **OpenSearch:** индексируется текст чанков (BM25).
- **Qdrant:** для каждого чанка строится эмбеддинг (опционально с контекстным заголовком из метаданных), записывается точка с payload (doc_id, chunk_index, текст, теги и т.д.). Используется content-hash для идемпотентности.

Идентификатор чанка в индексе: `{doc_id}:{chunk_index}` (или производное для Qdrant).

---

## Legacy-путь (без RabbitMQ / без storage)

Если при upload **нет** publisher (RabbitMQ) или **нет** успешного сохранения в storage:

1. Gate читает файл в память.
2. Декодирует как UTF-8 (для HTML — конвертация в текст).
3. Вызывает **retrieval** один раз: `POST /v1/index/upsert` с `mode=document`, `document` (метаданные), `text` (весь текст).
4. Retrieval в `indexing_logic` режет текст через `chunk_by_strategy` (стратегия из `RAG_CHUNK_STRATEGY`: semantic/token), строит эмбеддинги и пишет в OpenSearch и Qdrant.

Ограничения: бинарные форматы (PDF, DOCX) по legacy-пути не обрабатываются — только текст/HTML.

---

## Проверка статуса документа

- **Список документов:** `GET /v1/documents` — список из document-storage, опционально с полем `indexed` (проверка через retrieval `index_exists`).
- **Статус одного документа:** `GET /v1/documents/{doc_id}/status` — хранится ли файл, проиндексирован ли (`indexed`), и при наличии — `metadata.extra.ingestion` (состояние очереди: queued / processing / done / failed).

---

## Удаление документа

- **Один документ:** `DELETE /v1/documents/{doc_id}` — удаление из storage и из индекса retrieval (все чанки с этим doc_id).
- При асинхронной конфигурации в очередь может класться задача `type=delete`; worker удаляет документ из retrieval и при необходимости из storage.

---

## Зависимости сервисов

| Путь              | Нужны сервисы |
|-------------------|----------------|
| Асинхронный       | Gate, document-storage, RabbitMQ, ingestion-worker, doc-processor, retrieval, (VLM для PDF и т.д.) |
| Legacy            | Gate, retrieval |
| Статус/список     | Gate, document-storage, (retrieval для `indexed`) |

Итого: **добавление документа** = upload в Gate → сохранение в storage → задача в RabbitMQ → worker вызывает doc-processor → doc-processor забирает файл из storage, извлекает текст, режет на чанки и отправляет их в retrieval → retrieval индексирует чанки в OpenSearch и Qdrant.

---

## Диагностика: документы не индексируются (202 Accepted, но indexed=false)

Если Gate возвращает **202 Accepted**, а `GET /v1/documents/{doc_id}/status` показывает **indexed: false** и не меняется со временем — задача лежит в очереди или пайплайн после очереди не работает.

### Цепочка и что проверить

| Шаг | Сервис | Что проверить |
|-----|--------|----------------|
| 1 | Gate | Отдаёт 202 — значит storage и RabbitMQ доступны, задача опубликована. |
| 2 | document-storage | Файл сохранён. Gate перед 202 уже записал в storage. |
| 3 | RabbitMQ | Очередь `ingestion.tasks`. Сообщения должны забираться **ingestion-worker**. |
| 4 | ingestion-worker | **Обязательно запущен.** Consumes из `ingestion.tasks`, вызывает doc-processor по HTTP. |
| 5 | doc-processor | **Обязательно запущен.** Отвечает на `POST /v1/process`, тянет файл из storage, режет на чанки, шлёт в retrieval. |
| 6 | retrieval | Тот же инстанс, что проверяет Gate в `/status` (по `index_exists`). |

Для **.txt** файлов VLM (vllm-docling) не используется — doc-processor идёт по ветке `extract_text_non_vlm`. Для PDF/DOCX нужен vllm-docling.

### Скрипт проверки

```bash
# Доступность Gate, retrieval, document-storage
python eval/check_ingestion_pipeline.py --gate-url http://localhost:8090 --retrieval-url http://localhost:8080 --storage-url http://localhost:8081

# + глубина очереди RabbitMQ (если Management API на 15672)
python eval/check_ingestion_pipeline.py --rabbit-url http://localhost:15672 --rabbit-user guest --rabbit-pass guest
```

Если **очередь растёт** (messages > 0 и не уменьшается) — ingestion-worker не забирает сообщения или не запущен.

### Docker

Убедитесь, что подняты все сервисы полного пайплайна:

```bash
docker compose ps
# Должны быть: rag-gate, document-storage, rabbitmq, ingestion-worker, doc-processor, retrieval
```

Логи воркера (ошибки при вызове doc-processor или retrieval):

```bash
docker compose logs ingestion-worker --tail 100
docker compose logs doc-processor --tail 100
```

Очередь в RabbitMQ: Management UI `http://localhost:15672` (guest/guest) → Queues → `ingestion.tasks` — смотреть Ready/Unacked.

### Конфиг очереди

- **Gate** публикует в очередь из `GATE_RABBIT_QUEUE` (по умолчанию `ingestion.tasks`).
- **ingestion-worker** подписан на `WORKER_QUEUE` (по умолчанию `ingestion.tasks`).

Имена должны совпадать. В `docker-compose.yml` оба заданы как `ingestion.tasks`.

### Один стек — один compose

Полный пайплайн (202 → indexed) должен работать в **одном** docker-compose: Gate, document-storage, RabbitMQ, ingestion-worker, doc-processor, retrieval из одного проекта. Если worker и doc-processor подняты из другого compose (например rugfunsota) и подключены к сети чужого стека (rag_fun), возможны несовпадения: документы, сохранённые одним Gate/storage, не видны другому storage по имени или API. Для проверки пайплайна поднимайте все сервисы из одного `docker compose up`.
