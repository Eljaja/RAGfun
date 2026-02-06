# Планы улучшения agent-search и deep-research

## agent-search

### Функциональность

| Добавка | Описание |
|---------|----------|
| **Multi-turn** | Использовать `history` в gate и промптах — контекст диалога для plan, fact queries, answer |
| **Citation** | Вставлять ссылки `[1]`, `[2]` в текст ответа, маппинг на sources |
| **Non-streaming endpoint** | `POST /v1/agent` — возврат полного ответа без SSE |
| **Web search** | Интеграция Serper/Tavily/Bing — LLM решает: RAG, web или оба |
| **Tool use** | Калькулятор, code execution (sandbox) — LLM вызывает при необходимости |

### Retrieval

| Добавка | Описание |
|---------|----------|
| **Query routing** | LLM выбирает: BM25 / vector / hybrid / web / multi-source |
| **HyDE variants** | Несколько гипотетических ответов, агрегация результатов |
| **Multi-query fusion** | Параллельные запросы (оригинал + fact + keywords), RRF |

### Контроль и надёжность

| Добавка | Описание |
|---------|----------|
| **Max LLM calls** | Лимит вызовов LLM на запрос (plan, hyde, fact, keywords, answer, assess) |
| **Timeout** | Общий таймаут запроса, graceful stop |
| **Cancellation** | Отмена по `task_id` или connection close |
| **Rate limiting** | Защита от перегрузки LLM API |

### Observability

| Добавка | Описание |
|---------|----------|
| **Метрики** | Latency, LLM calls, gate calls, retry rate |
| **Structured logging** | Решения LLM (plan, fact_queries, retry trigger) |
| **Trace** | OpenTelemetry / LangSmith для отладки |

### Конфигурация

| Добавка | Описание |
|---------|----------|
| **Пороги quality** | `AGENT_QUALITY_MIN_HITS`, `AGENT_QUALITY_MIN_SCORE` через env |
| **Feature flags** | Включение HyDE, fact split, retry по env |
| **Режимы** | `conservative` / `aggressive` / `minimal` — разный баланс качества и скорости |

---

## deep-research

### Функциональность

| Добавка | Описание |
|---------|----------|
| **Web search** | Поиск в интернете для gaps — LLM решает, когда искать вне RAG |
| **Fact verification** | Проверка противоречий между источниками, confidence score |
| **Streaming отчёта** | Генерация отчёта по токенам, а не постфактум |
| **Шаблоны отчётов** | Выбор шаблона по типу запроса (technical, business, comparison) |
| **Export** | Markdown, PDF, DOCX на выходе |

### Research loop

| Добавка | Описание |
|---------|----------|
| **Больше итераций** | `max_iterations` до 4–5, early stop по насыщению |
| **Diversity queries** | Запросы на разные аспекты, избегание дублирования |
| **Source prioritization** | Приоритет авторитетных источников (если есть метаданные) |
| **Contradiction detection** | LLM помечает противоречия между источниками |

### Инфраструктура

| Добавка | Описание |
|---------|----------|
| **Async gate** | `httpx.AsyncClient` вместо синхронного — не блокировать event loop |
| **Cancellation** | Отмена долгого research по запросу |
| **Checkpointing** | Сохранение состояния графа для resume при сбое |
| **Parallel gate calls** | Параллельные запросы для batch queries |

### Observability

| Добавка | Описание |
|---------|----------|
| **Progress accuracy** | Более точный `percent` по этапам |
| **Метрики** | Итерации, gate calls, LLM calls, время по узлам графа |
| **Trace** | LangSmith / OpenTelemetry для графа |

### Конфигурация

| Добавка | Описание |
|---------|----------|
| **DEEP_RESEARCH_BATCH** | Уже есть — размер batch запросов за итерацию |
| **DEEP_RESEARCH_MAX_DOCS** | Уже есть — лимит документов в контексте |
| **Early stop** | Остановка при малом приросте notes/sources |
| **Template per domain** | Разные шаблоны для tech / business / legal |

---

## Общее (для обоих)

| Добавка | Описание |
|---------|----------|
| **Shared library** | Вынести `_quality_is_poor`, `_merge_hits`, `_build_context` в общий модуль |
| **Единый Gate client** | Async, retry, timeout, метрики |
| **Единый формат events** | Совместимые `trace`, `retrieval`, `token`, `done` для UI |
