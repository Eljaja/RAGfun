# Agent-Search и Deep-Research API

Оба сервиса поднимаются профилем `agent-search`:  
`docker compose --profile agent-search up -d`

---

## Agent-Search (порт 8093)

LLM-поиск: **plan** → опционально HyDE → **Gate.chat** → проверка качества → опционально **fact queries** → **answer** с цитированием [1], [2].

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/v1/agent` | Ответ одним JSON: `answer`, `sources`, `context`, `mode`, `partial`, `degraded` |
| POST | `/v1/agent/stream` | SSE: события `init`, `trace`, `retrieval`, `token`, `done`, `error` |

### Тело запроса (AgentRequest)

| Поле | Тип | Описание |
|------|-----|----------|
| `query` | string | Вопрос пользователя (обязательно) |
| `history` | array | История диалога `[{role, content}]` |
| `filters` | object | Фильтры Gate (`project_id`, `source` и т.д.) |
| `include_sources` | bool | Включать источники (по умолчанию true) |
| `top_k` | int | Переопределить top_k (5..24) |
| `use_adaptive_k` | bool | Adaptive-k (переопределение env) |
| `max_llm_calls` | int | Макс. вызовов LLM на запрос |
| `max_fact_queries` | int | Макс. fact-запросов |
| `use_hyde` | bool | Включить HyDE |
| `use_fact_queries` | bool | Включить fact queries |
| `use_retry` | bool | Повтор при неполном ответе |
| `use_tools` | bool | Калькулятор и выполнение кода |
| `mode` | string | Пресет: `minimal` \| `conservative` \| `aggressive` |

Пресеты: **minimal** (без HyDE/fact/retry, 4 LLM), **conservative** (6 LLM), **aggressive** (HyDE, fact, retry, 16 LLM, 4 fact queries).

### Переменные окружения (agent-search)

- `AGENT_GATE_URL` — URL Gate (напр. http://rag-gate:8090)
- `AGENT_GATE_TIMEOUT_S` — таймаут вызовов Gate
- `AGENT_MAX_LLM_CALLS`, `AGENT_MAX_FACT_QUERIES`
- `AGENT_USE_HYDE`, `AGENT_USE_FACT_QUERIES`, `AGENT_USE_RETRY`
- `AGENT_LLM_BASE_URL`, `AGENT_LLM_MODEL`, `AGENT_LLM_API_KEY` (fallback: GATE_*)

### Пример (agent-search)

```bash
# Стриминг
curl -N -X POST "http://localhost:8093/v1/agent/stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","include_sources":true,"mode":"conservative"}'

# Без стриминга
curl -s -X POST "http://localhost:8093/v1/agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","include_sources":true}' | jq .
```

---

## Deep-Research (порт 8094)

Итеративное исследование (LangGraph): **plan** → **scope** (план + запросы) → цикл **research** (батч вызовов Gate, заметки, следующие запросы) → ранняя остановка по min gain → **write** (стриминг отчёта).

### Эндпоинт

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/v1/deep-research/stream` | Единственный эндпоинт; ответ — SSE-поток событий |

### Тело запроса (DeepResearchRequest)

| Поле | Тип | Описание |
|------|-----|----------|
| `query` | string | Исследовательский вопрос (обязательно) |
| `history` | array | История диалога `[{role, content}]` |
| `filters` | object | Фильтры Gate (`project_id`, `source` и т.д.) |
| `include_sources` | bool | Включать источники в ответе (по умолчанию true) |
| `max_iterations` | int | Макс. итераций цикла research (по умолчанию 2) |
| `retrieval_mode` | string | Режим retrieval: hybrid \| bm25 \| vector |
| `top_k` | int | Top-k на один запрос к Gate |
| `rerank` | bool | Включить rerank |
| `use_web_search` | bool | Включить веб-поиск |
| `web_search_num` | int | Макс. результатов веб-поиска |
| `web_search_timeout_s` | float | Таймаут веб-поиска (сек) |

### События SSE (deep-research)

Сервер шлёт строки `data: <json>`:

| type | Описание |
|------|----------|
| `progress` | Этап: `plan`, `scope`, `research`, `write`, `done`; может быть `stage`, `percent`, `message` |
| `retrieval` | Промежуточные хиты/контекст (если отдаётся) |
| `token` | Фрагмент отчёта (стриминг текста) |
| `done` | Финальное событие: `answer` (полный отчёт), `sources` |
| `error` | Ошибка: `error` (строка) |

### Переменные окружения (deep-research)

- **Gate / LLM:** `DEEP_GATE_URL`, `DEEP_GATE_TIMEOUT_S`; `DEEP_LLM_PROVIDER`, `DEEP_LLM_BASE_URL`, `DEEP_LLM_MODEL`, `DEEP_LLM_API_KEY`, `DEEP_LLM_TEMPERATURE` (fallback: GATE_*)
- **Итерации:** `DEEP_MAX_ITERATIONS` (макс. итераций цикла research)
- **Ранняя остановка:** `DEEP_EARLY_STOP_MIN_GAIN` (мин. прирост для продолжения)
- **Батч:** `DEEP_RESEARCH_BATCH` (размер батча запросов в research), `DEEP_RESEARCH_MAX_DOCS`, `DEEP_RESEARCH_MAX_CHARS`
- **Retrieval:** `DEEP_TOP_K`, `DEEP_RERANK`, `DEEP_RETRIEVAL_MODE` (hybrid \| bm25 \| vector)
- **Веб-поиск:** `DEEP_USE_WEB_SEARCH`; при включённом веб-поиске также `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_NUM`, `WEB_SEARCH_TIMEOUT_S`
- **Шаблон отчёта:** `DEEP_RESEARCH_TEMPLATE` (текст с инструкциями для этапа write)
- **Логирование:** `DEEP_LOG_LEVEL`
- **MCP (опционально):** `DEEP_MCP_CONFIG` — путь к конфигу MCP gate

### Пример (deep-research)

```bash
# Стриминг отчёта
curl -N -X POST "http://localhost:8094/v1/deep-research/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main approaches to RAG in 2024?",
    "include_sources": true,
    "max_iterations": 3,
    "filters": {"project_id": "demo"}
  }'
```

Разбор событий (jq):

```bash
curl -N -s -X POST "http://localhost:8094/v1/deep-research/stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","max_iterations":2}' \
  | while read -r line; do
      if [[ "$line" == data:* ]]; then
        echo "$line" | sed 's/^data: //' | jq -c 'select(.type) | {type, stage: .stage, answer: (.answer[:80] // "")}'
      fi
    done
```

---

## Общее

- Оба сервиса вызывают **Gate** (`/v1/chat`) для retrieval и контекста; LLM используется для плана, факт-запросов и ответа/отчёта.
- **agent-search** — один проход (план → Gate → ответ с опциональными fact queries и retry).
- **deep-research** — цикл: несколько итераций research с батч-запросами к Gate, накопление заметок и финальный стриминг отчёта.
- В `.env` / `env.example` для deep-research задаются переменные с префиксом `DEEP_*` (см. выше).
