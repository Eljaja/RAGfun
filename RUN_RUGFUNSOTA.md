# Запуск RAG (проект rugfunsota)

Проект в Docker Compose имеет имя **rugfunsota** — контейнеры и тома будут с префиксом `rugfunsota_`.

**Document-storage:** по умолчанию бэкенд **local** (файлы в томе `document-storage-data`) — стек работает без внешнего RustFS. Для S3: задать `STORAGE_STORAGE_BACKEND=s3` и поднять **presign-rustfs-1** (сеть `presign_rustfs-net`) или свой RustFS: `docker compose --profile local-rustfs up -d`.

## Что нужно

- **Docker** и **Docker Compose** (v2)
- **GPU** (NVIDIA + nvidia-container-toolkit) — для эмбеддингов (Infinity), реранкера (Infinity) и VLM (Granite-Docling). Без GPU сервисы `infinity-embed`, `infinity-rerank`, `vllm-docling` могут не стартовать или работать на CPU (медленно).
- По желанию: файл **`.env`** в корне (скопировать из `env.example`) — пароли, ключи LLM, переопределение портов.

## Запуск

```bash
cd /home/ubuntu/szavodnov/RAGfun

# Опционально: скопировать пример env (если ещё нет .env)
cp env.example .env

# Поднять весь стек (фоновый режим)
docker compose up -d

# Или с выводом логов в консоль
docker compose up
```

Имя проекта задаётся в `docker-compose.yml` (`name: rugfunsota`), поэтому отдельно `-p rugfunsota` указывать не нужно.

## Проверка (порты rugfunsota — альтернативные, чтобы не конфликтовать с другим стеком)

- **Gate (RAG API):** http://localhost:8092  
- **Document-storage:** http://localhost:8083  
- **Retrieval:** http://localhost:8085  
- **UI:** http://localhost:3300  
- **OpenSearch:** http://localhost:9201  
- **Qdrant:** http://localhost:6335  
- **Postgres:** localhost:5434  
- **RabbitMQ:** http://localhost:15673 (guest/guest), AMQP 5673  
- **Grafana:** http://localhost:3000 (admin/admin)

```bash
# Статус контейнеров
docker compose ps

# Логи конкретного сервиса
docker compose logs -f retrieval
docker compose logs -f rag-gate
```

## Остановка

```bash
docker compose down
# С удалением томов (полная очистка индексов и данных):
# docker compose down -v
```

## Режим без GPU (CPU-only)

Если GPU нет, можно:

1. **Реранкер:** в `.env` или в переменных для сервиса `retrieval` задать:
   - `RAG_RERANK_PROVIDER=local`
   - Убрать или не поднимать сервис `infinity-rerank`.
   В коде retrieval тогда будет использоваться локальный CPU-реранкер (sentence-transformers). В образ `retrieval` нужно добавить зависимость `sentence-transformers` (уже есть в `service/requirements.txt`).

2. **Эмбеддинги и VLM:** без GPU сервисы `infinity-embed` и `vllm-docling` либо не запустятся (если в compose зарезервирован GPU), либо будут очень медленными на CPU. Для полноценного CPU-режима нужна отдельная конфигурация (например, другой образ эмбеддингов или отключение VLM и загрузка только текстовых файлов).

Текущий `docker-compose.yml` рассчитан на хост с GPU; имя проекта **rugfunsota** уже задано в файле.
