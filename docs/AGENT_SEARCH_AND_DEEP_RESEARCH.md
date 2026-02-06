# Agent-Search and Deep-Research

LLM-driven retrieval services that extend the RAG stack with intelligent query planning, multi-source search, and iterative research.

## Overview

| Service        | Port | Description                                                                 |
|----------------|------|-----------------------------------------------------------------------------|
| **agent-search**  | 8093 | LLM-driven search: plan → Gate.chat → quality check → fact queries → answer |
| **deep-research** | 8094 | Iterative research: LangGraph (plan → scope → research loop → write)        |

Both services run on top of the Gate API and support **web search** (Serper/Tavily) for queries requiring current events or external knowledge.

## Starting the Services

```bash
docker compose --profile agent-search up -d
```

## Agent-Search

### Flow

1. **Plan** — LLM chooses retrieval mode (hybrid/bm25/vector), top_k, rerank, HyDE, and whether to use web search
2. **HyDE** (optional) — Generate hypothetical document for better semantic matching
3. **Gate.chat** — Primary retrieval via RAG Gate
4. **Quality check** — If results are weak, split into fact queries and fetch additional context
5. **Web search** (optional) — Merge Serper/Tavily results when enabled
6. **Answer** — Generate response with citations [1], [2]
7. **Assess** — Retry if answer is incomplete (up to `AGENT_MAX_LLM_CALLS`)

### API

**Streaming (SSE):**
```bash
curl -X POST http://localhost:8093/v1/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "include_sources": true}'
```

**Non-streaming (JSON):**
```bash
curl -X POST http://localhost:8093/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "include_sources": true}'
```

### Request Parameters

| Parameter            | Type   | Description                                                                 |
|----------------------|--------|-----------------------------------------------------------------------------|
| `query`              | string | User question (required)                                                    |
| `include_sources`    | bool   | Include sources in response (default: true)                                |
| `filters`            | object | Gate filters (project_id, doc_ids, etc.)                                    |
| `history`            | array  | Chat history for context                                                   |
| `use_web_search`     | bool   | `false` = never, `true` = force, `null` = use plan + env                   |
| `web_search_num`     | int    | Number of web results (default: 5)                                         |
| `web_search_timeout_s` | float | Web search timeout (default: 15)                                           |
| `max_llm_calls`      | int    | Max LLM calls per request (rate limit protection)                          |
| `max_fact_queries`   | int    | Max fact queries when quality is poor                                      |

### Configuration (env)

- `AGENT_MAX_LLM_CALLS` — Max LLM calls (default: 12)
- `AGENT_MAX_FACT_QUERIES` — Max fact queries (default: 2)
- `AGENT_ALWAYS_FACT_QUERIES` — Always run fact queries (default: false)
- `AGENT_ALWAYS_WEB_SEARCH` — Force web search for every request (default: false)
- `AGENT_REQUEST_TIMEOUT_S` — Request timeout (default: 120)

---

## Deep-Research

### Flow

1. **Plan** — LLM chooses retrieval settings (mode, top_k, rerank)
2. **Scope** — Draft research plan and initial queries
3. **Research loop** — Batch Gate calls + optional web search → distilled notes → next_queries
4. **Early stop** — Exit when min gain threshold is reached
5. **Write** — Stream structured report (Executive Summary, sections, sources)

### API

**Streaming (SSE):**
```bash
curl -X POST http://localhost:8094/v1/deep-research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "max_iterations": 2, "include_sources": true}'
```

### Request Parameters

| Parameter            | Type   | Description                                                                 |
|----------------------|--------|-----------------------------------------------------------------------------|
| `query`              | string | Research question (required)                                                |
| `max_iterations`      | int    | Max research iterations (default: 2)                                      |
| `include_sources`    | bool   | Include sources in report (default: true)                                  |
| `filters`            | object | Gate filters                                                               |
| `history`            | array  | Chat history                                                               |
| `retrieval_mode`     | string | Override: hybrid, bm25, vector                                             |
| `top_k`              | int    | Override retrieval top_k                                                   |
| `rerank`             | bool   | Override reranking                                                         |
| `use_web_search`     | bool   | `false` = never, `true` = force, `null` = use env                         |
| `web_search_num`     | int    | Number of web results (default: 5)                                         |
| `web_search_timeout_s` | float | Web search timeout (default: 15)                                         |

### Configuration (env)

- `DEEP_MAX_ITERATIONS` — Max research iterations (default: 6)
- `DEEP_USE_WEB_SEARCH` — Enable web search by default (default: false)
- `DEEP_RETRIEVAL_MODE` — hybrid, bm25, or vector
- `DEEP_TOP_K` — Retrieval top_k
- `DEEP_RERANK` — Enable reranking

---

## Web Search (Serper / Tavily)

Both services support web search for queries about current events, news, or facts outside the document corpus.

### Setup

1. Get an API key:
   - **Serper**: https://serper.dev
   - **Tavily**: https://tavily.com

2. Add to `.env`:
   ```bash
   WEB_SEARCH_PROVIDER=serper
   WEB_SEARCH_API_KEY=your_key
   # Or: SERPER_API_KEY=... or TAVILY_API_KEY=...
   ```

3. Enable per request:
   - Agent: `{"use_web_search": true}`
   - Deep-research: `{"use_web_search": true}` or `DEEP_USE_WEB_SEARCH=true`

### When to Use Web Search

- Current events, news, recent data
- Facts not in your document corpus
- Queries like "Who won the 2022 World Cup?" or "Latest AI news 2025"

### Configuration

- `WEB_SEARCH_NUM` — Results per query (default: 5)
- `WEB_SEARCH_TIMEOUT_S` — Timeout (default: 15)

---

## Verification

```bash
# Full verification: retrieval, gate, agent-search, deep-research
python scripts/verify_agent_queries.py
```

## Event Types (SSE)

| Type       | Description                    |
|------------|--------------------------------|
| `progress` | Stage, percent, message        |
| `trace`   | Thought, tool call, action     |
| `retrieval`| Context chunks and sources    |
| `token`   | Streaming report token         |
| `done`    | Final answer or report         |
