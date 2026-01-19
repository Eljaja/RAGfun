# Gate (RAG Gateway) — документация и тестовые скрипты

`gate` — FastAPI gateway, который принимает uploads, отдаёт чат (обычный и streaming), проксирует часть операций в `document-storage` и `retrieval`, а также может ставить ingestion-задачи в RabbitMQ.

## Быстрый старт

1) Поднимите инфраструктуру (пример — `docker-compose.yml` в корне репо).

2) Экспортируйте URL gate (по умолчанию):

```bash
export GATE_URL="http://localhost:8090"
```

3) Проверьте готовность:

```bash
python3 ./Docs/gate/scripts/gate.py readyz
```

4) Запустите smoke (upload → status → chat):

```bash
python3 ./Docs/gate/scripts/gate.py smoke --file "./README.md"
```

## Эндпоинты (кратко)

См. подробности: [`GATE_API.md`](./GATE_API.md).

- `GET /v1/healthz` — liveness.
- `GET /v1/readyz` — readiness + проверка `retrieval`.
- `GET /v1/version` — конфигурация (без секретов) и версия сервиса.
- `GET /v1/metrics` — Prometheus metrics.
- `POST /v1/chat` — чат (нестриминговый).
- `POST /v1/chat/stream` — чат (SSE streaming).
- `POST /v1/documents/upload` — загрузка документа (multipart/form-data).
- `GET /v1/documents` — список документов (из `document-storage`, если он настроен).
- `GET /v1/documents/{doc_id}/status` — статус хранения + индексации.
- `DELETE /v1/documents/{doc_id}` — удалить документ (best-effort: storage + retrieval).
- `DELETE /v1/documents?confirm=true` — удалить все документы (safety-гейт).
- `GET /v1/documents/stats` — агрегированная статистика по документам (из storage + индексация).
- `GET /v1/collections` — список “коллекций” (project_id) из storage.

## Тестовые скрипты

Скрипт лежит в [`scripts/gate.py`](./scripts/gate.py). Все команды используют переменную окружения:

- `GATE_URL` — base URL (например `http://localhost:8090`).

Полезные команды:

```bash
python3 ./Docs/gate/scripts/gate.py healthz
python3 ./Docs/gate/scripts/gate.py readyz
python3 ./Docs/gate/scripts/gate.py version
```

Примеры:

```bash
python3 ./Docs/gate/scripts/gate.py upload "./README.md" "doc-$(date +%s)" --title "Repo README"
python3 ./Docs/gate/scripts/gate.py doc-status "doc-$(date +%s)"
python3 ./Docs/gate/scripts/gate.py chat "Что такое gate в этом проекте?"
python3 ./Docs/gate/scripts/gate.py chat-stream "Сделай краткое резюме архитектуры проекта."
```

