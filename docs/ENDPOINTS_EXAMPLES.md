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

---

## RAGcircle v2 (retrieval_v2 + generator_v2 + gate_v2)

Base URLs (default local compose):

- Retrieval v2: `http://localhost:8920`
- Generator v2: `http://localhost:8930`
- Gate v2: `http://localhost:8912`

```bash
export RETRIEVAL_V2_URL="http://localhost:8920"
export GENERATOR_V2_URL="http://localhost:8930"
export GATE_V2_URL="http://localhost:8912"
```

### Retrieval v2: legacy retrieve

```bash
curl -sS "$RETRIEVAL_V2_URL/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "default",
    "query": "что такое RAG?",
    "top_k": 8,
    "strategy": "hybrid",
    "rerank": true,
    "rerank_top_n": 8
  }' | jq .
```

### Retrieval v2: planning mode (`/plan/retrieve`)

```bash
curl -sS "$RETRIEVAL_V2_URL/plan/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "default",
    "query": "что такое RAG?",
    "plan": {
      "round": {
        "retrieve": [
          {"kind": "vector_search", "top_k": 16},
          {"kind": "bm25_search", "top_k": 16}
        ],
        "combine": {"kind": "fuse", "method": "rrf", "rrf_k": 60},
        "rank": [
          {"kind": "rerank", "top_n": 12},
          {"kind": "adaptive_k", "min_k": 3, "max_k": 24}
        ],
        "finalize": [{"kind": "trim", "top_k": 8}]
      }
    }
  }' | jq .
```

### Generator v2: agent (non-streaming)

```bash
curl -sS "$GENERATOR_V2_URL/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "default",
    "query": "Сделай короткое summary по архитектуре",
    "top_k": 8,
    "strategy": "hybrid",
    "mode": "aggressive",
    "include_sources": true
  }' | jq .
```

### Generator v2: agent stream (SSE)

```bash
curl -N -sS "$GENERATOR_V2_URL/agent/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "default",
    "query": "Сделай короткое summary по архитектуре",
    "top_k": 8,
    "strategy": "hybrid",
    "mode": "conservative",
    "include_sources": true
  }'
```

### Gate v2: project-scoped agent (proxy to generator v2)

```bash
# Replace X-ODS-API-KEY with your token when auth is enabled.
curl -sS "$GATE_V2_URL/api/v1/projects/default/agent" \
  -H "Content-Type: application/json" \
  -H "X-ODS-API-KEY: <TOKEN>" \
  -d '{
    "query": "Какие ключевые идеи в проекте?",
    "top_k": 8,
    "strategy": "hybrid",
    "mode": "aggressive",
    "include_sources": true
  }' | jq .
```

### Gate v2: project-scoped agent stream (SSE)

```bash
curl -N -sS "$GATE_V2_URL/api/v1/projects/default/agent/stream" \
  -H "Content-Type: application/json" \
  -H "X-ODS-API-KEY: <TOKEN>" \
  -d '{
    "query": "Какие ключевые идеи в проекте?",
    "top_k": 8,
    "strategy": "hybrid",
    "mode": "conservative",
    "include_sources": true
  }'
```
