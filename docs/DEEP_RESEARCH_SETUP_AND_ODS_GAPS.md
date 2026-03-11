# Deep-research: как устроен и чего не хватает до ODS

После мержа main в ветку `feature/deep-research-restore`: кратко, как устроен deep-research и что осталось до полной увязки с ODS (Shared ODS с tenant_id).

---

## 1. Как устроен deep-research

### Роль в стеке

- **Сервис**: `deep-research` (порт 8094, профиль `deep-research`).
- **Запуск**: `docker compose --profile deep-research up -d` (нужна папка `deep-research/` и образ rag-gate).
- **Один эндпоинт**: `POST /v1/deep-research/stream` — SSE-поток с отчётом.

### Поток данных

```
Клиент → deep-research (LangGraph) → Gate /v1/chat (многократно) → retrieval + LLM
                    ↓
              web_search (опционально: Serper/Tavily)
```

- **К Gate** идут только вызовы `/v1/chat`: план → scope (план + запросы) → цикл research (батч запросов к Gate, заметки, следующие запросы) → early stop → write (стриминг отчёта).
- **К document-storage** deep-research **не ходит** — только через Gate. То есть ODS-слой (document-storage + Postgres) используется только опосредованно: Gate сам дергает retrieval и при необходимости document-storage.

### Мультитенантность и фильтры

- В запросе есть **filters** (в т.ч. `tenant_id`, `project_id`, `source`, `tags`, `doc_ids`, `project_ids`).
- Эти фильтры пробрасываются в **state** и во **все** вызовы `gate.chat(...)` внутри графа (scope, research loop, batch Gate calls).
- То есть поиск и контекст уже ограничены тенантом/проектом, если клиент передал `filters.tenant_id` / `filters.project_id`.

### Что есть на ветке после мержа с main

- Каталог **deep-research/** (Dockerfile, app/main.py, app/mcp_gate.json, requirements.txt).
- В **docker-compose.yml** — сервис `deep-research` (profile `deep-research`), переменные `DEEP_*`, `WEB_SEARCH_*`.
- В **UI**: переключатель deep-research (`deep_research`), прокси в nginx на deep-research (например `/deep-api/`).
- В **agent_common**: общий Gate-клиент и `web_search_async`; deep-research использует те же вызовы Gate с `filters`.
- Документация: **docs/AGENT_SEARCH_AND_DEEP_RESEARCH.md** (API agent-search и deep-research).

---

## 2. Чего не хватает до ODS (без учёта auth)

Под «ODS» здесь — наш вариант **Shared ODS**: один слой документов/метаданных (document-storage + Postgres), единый Gate, везде фильтрация по `tenant_id`.

### Уже есть

- **Единая точка входа в ODS для deep-research** — Gate. Deep-research не обходит Gate и не лезет напрямую в document-storage/retrieval.
- **Тенантская область видимости** — через `filters` (tenant_id, project_id и др.), которые везде передаются в Gate.

### Чего не хватает (до «полного» ODS-сценария)

1. **Явная опора на список коллекций/проектов из ODS**  
   Deep-research не вызывает Gate `/v1/collections` (или `/v1/documents/stats`). Для UX можно добавить опциональный шаг «выбор project_id / коллекции» по данным из Gate (тогда deep-research остаётся потребителем ODS только через Gate, без прямого доступа к storage).

2. **Проверка/подстановка tenant_id со стороны Gate**  
   Сейчас `tenant_id` приходит с клиента (в body). По нашей ODS-логике «источник правды» по тенанту — auth + Gate: после появления auth Gate будет подставлять/проверять tenant_id по токену. Deep-research тогда достаточно передавать в Gate тот же токен или уже проверенный tenant_id с единой точки входа (Gate). Отдельный пункт про auth мы не разбираем здесь.

3. **Документация и дефолты**  
   В README и env.example явно указать, что для мультитенантного использования в deep-research нужно передавать `filters.tenant_id` / `filters.project_id` (и потом — что они будут браться из auth в Gate).

Итого: архитектурно deep-research уже «сидит» на ODS через Gate и использует tenant/project через filters. Не хватает: опционального использования коллекций/stats из Gate для UX и, в перспективе, переноса проверки tenant_id в Gate (auth).

---

## 3. Запуск после мержа

```bash
# Сборка и запуск с deep-research
docker compose --profile deep-research up -d --build

# Проверка
curl -N -X POST "http://localhost:8094/v1/deep-research/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "max_iterations": 2, "filters": {"project_id": "demo"}}'
```

Убедиться, что в `docker-compose.yml` у сервиса `deep-research` порт 8094 (или свой `RUGFUNSOTA_DEEP_PORT`) и что Gate доступен по `DEEP_GATE_URL`.
