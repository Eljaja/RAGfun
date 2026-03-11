# RAG API Examples

Retrieval: `http://localhost:8085`

## Quick Start: Add Document and Query

```bash
# 1. Health check
curl -s http://localhost:8085/v1/healthz

# 2. Add document
curl -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{"mode":"document","document":{"doc_id":"doc1","project_id":"demo","source":"cli"},"text":"Python is a programming language. Used for web development and data science.","refresh":true}'

# 3. Query the document
curl -s -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Python and what is it used for?","mode":"hybrid","top_k":5,"filters":{"project_id":"demo"}}'
```

## 1. Health Check

```bash
curl -s http://localhost:8085/v1/healthz
# {"ok":true}
```

## 2. Document Indexing

```bash
curl -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "document",
    "document": {"doc_id": "doc1", "project_id": "my_project", "source": "manual"},
    "text": "Python is a programming language. Used for web development and data science.",
    "refresh": true
  }'
```

## 3. Search

```bash
curl -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "mode": "hybrid",
    "top_k": 5,
    "filters": {"project_id": "my_project"}
  }'
```

## 4. Full Cycle (bash)

```bash
# Index
curl -s -X POST http://localhost:8085/v1/index/upsert \
  -H "Content-Type: application/json" \
  -d '{"mode":"document","document":{"doc_id":"test1","project_id":"demo","source":"cli"},"text":"Machine learning is a subset of artificial intelligence.","refresh":true}'

# Search
curl -s -X POST http://localhost:8085/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","mode":"hybrid","top_k":5,"filters":{"project_id":"demo"}}' | jq '.hits[:3]'
```

## Search Parameters

| Parameter | Description |
|-----------|-------------|
| `query` | Search query text |
| `mode` | `hybrid` \| `bm25` \| `vector` |
| `top_k` | Number of results to return |
| `filters.project_id` | Filter by project/collection |
| `rerank` | `true` \| `false` \| `null` (auto) |
