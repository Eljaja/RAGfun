# Запуск RAG (проект rugfunsota)

Проект в Docker Compose имеет имя **rugfunsota** — контейнеры и тома будут с префиксом `rugfunsota_`.

**Document-storage:** по умолчанию бэкенд **local** (файлы в томе `document-storage-data`) — стек работает без внешнего RustFS. Для S3: задать `STORAGE_STORAGE_BACKEND=s3` и поднять **presign-rustfs-1** (сеть `presign_rustfs-net`) или свой RustFS: `docker compose --profile local-rustfs up -d`.

## Что нужно

- **Docker** и **Docker Compose** (v2)
- **GPU** (NVIDIA + nvidia-container-toolkit) — для эмбеддингов (Infinity), реранкера (Infinity) и VLM (Granite-Docling). Без GPU сервисы `infinity-embed`, `infinity-rerank`, `vllm-docling` могут не стартовать или работать на CPU (медленно).
- По желанию: файл **`.env`** в корне (скопировать из `env.example`) — пароли, ключи LLM, переопределение портов.

## GPU

По умолчанию все сервисы используют GPU 0. Для GPU 3 — в `.env` или при запуске:

```bash
NVIDIA_DEVICE_EMBED=3
NVIDIA_DEVICE_RERANK=3
NVIDIA_DEVICE_VLLM=3
```

Или одной строкой:
```bash
NVIDIA_DEVICE_EMBED=3 NVIDIA_DEVICE_RERANK=3 NVIDIA_DEVICE_VLLM=3 ./scripts/up_for_rag.sh
```

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

## Известная проблема: rugfunsota + rag_fun одновременно

**Симптомы:** retrieval (порт 8085) не отвечает на запросы — `curl http://localhost:8085/v1/healthz` зависает. TCP-соединение устанавливается, но ответа нет.

**Причина:** retrieval подключён к сети `rag_fun_rag-network` (для использования infinity из rag_fun при отсутствии GPU). Когда **оба стека** (rugfunsota и rag_fun) запущены одновременно, retrieval оказывается в двух сетях с одноимёнными сервисами (`infinity-embed`, `infinity-rerank`). Это приводит к конфликту — event loop блокируется (вероятно, из‑за DNS или соединений между сетями).

**Решения:**
1. **Запускать только один стек** — либо rugfunsota, либо rag_fun.
2. **Убрать rag_fun_rag-network** из retrieval в `docker-compose.yml`, если rag_fun не нужен и у rugfunsota есть свой GPU:
   ```yaml
   # retrieval, networks:
   - rag-network
   # - rag_fun_rag-network   # закомментировать
   ```
3. **Останавливать rag_fun** перед запуском rugfunsota: `cd ifedotov/rag_fun && docker compose down`
4. **Использовать ragfun retrieval** — если ragfun уже запущен и в нём есть документы, задать в `.env`:
   ```
   GATE_RETRIEVAL_URL=http://host.docker.internal:8080
   ```
   Тогда gate (и agent-search, и deep-research) будут ходить в ragfun retrieval на порту 8080.
