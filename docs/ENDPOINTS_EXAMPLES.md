# Примеры вызовов RAG через эндпоинты

Retrieval: `http://localhost:8085`

## Три команды: добавить док и задать вопрос

```bash
# 1. Проверка
curl -s http://localhost:8085/v1/healthz

# 2. Добавить документ
curl -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{"mode":"document","document":{"doc_id":"doc1","project_id":"demo","source":"cli"},"text":"Python — язык программирования. Используется для веб-разработки и data science.","refresh":true}'

# 3. Задать вопрос по документу
curl -s -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Что такое Python и для чего используется?","mode":"hybrid","top_k":5,"filters":{"project_id":"demo"}}'
```

## 1. Health check

```bash
curl -s http://localhost:8085/v1/healthz
# {"ok":true}
```

## 2. Индексация документа

```bash
curl -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "document",
    "document": {"doc_id": "doc1", "project_id": "my_project", "source": "manual"},
    "text": "Python — язык программирования. Используется для веб-разработки и data science.",
    "refresh": true
  }'
```

## 3. Поиск

```bash
curl -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что такое Python?",
    "mode": "hybrid",
    "top_k": 5,
    "filters": {"project_id": "my_project"}
  }'
```

## 4. Полный цикл (bash)

```bash
# Индекс
curl -s -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{"mode":"document","document":{"doc_id":"test1","project_id":"demo","source":"cli"},"text":"Machine learning is a subset of artificial intelligence.","refresh":true}'

# Поиск
curl -s -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","mode":"hybrid","top_k":5,"filters":{"project_id":"demo"}}' | jq '.hits[:3]'
```

## Параметры search

| Параметр | Описание |
|----------|----------|
| `query` | Текст запроса |
| `mode` | `hybrid` \| `bm25` \| `vector` |
| `top_k` | Сколько результатов вернуть |
| `filters.project_id` | Фильтр по проекту/коллекции |
| `rerank` | `true` \| `false` \| `null` (auto) |
