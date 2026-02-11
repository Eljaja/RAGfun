# Agent-Search and Deep-Research API

Both services start with profile `agent-search`:  
`docker compose --profile agent-search up -d`

---

## Agent-Search (port 8093)

LLM-driven search: **plan** → optional HyDE → **Gate.chat** → quality check → optional **fact queries** → **answer** with citations [1], [2].

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/agent` | Single JSON: `answer`, `sources`, `context`, `mode`, `partial`, `degraded` |
| POST | `/v1/agent/stream` | SSE: `init`, `trace`, `retrieval`, `token`, `done`, `error` |

### Request body (AgentRequest)

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | User question (required) |
| `history` | array | Chat history `[{role, content}]` |
| `filters` | object | Gate filters (`project_id`, `source`, etc.) |
| `include_sources` | bool | Include sources (default true) |
| `top_k` | int | Override top_k (5..24) |
| `use_adaptive_k` | bool | Adaptive-k (overrides env) |
| `max_llm_calls` | int | Max LLM calls per request |
| `max_fact_queries` | int | Max fact queries |
| `use_hyde` | bool | Enable HyDE |
| `use_fact_queries` | bool | Enable fact queries |
| `use_retry` | bool | Retry on incomplete answer |
| `use_tools` | bool | Calculator and code execution |
| `mode` | string | Preset: `minimal` \| `conservative` \| `aggressive` |

Presets: **minimal** (no HyDE/fact/retry, 4 LLM), **conservative** (6 LLM), **aggressive** (HyDE, fact, retry, 16 LLM, 4 fact queries).

### Environment (agent-search)

- `AGENT_GATE_URL` — Gate URL (e.g. http://rag-gate:8090)
- `AGENT_GATE_TIMEOUT_S` — Gate call timeout
- `AGENT_MAX_LLM_CALLS`, `AGENT_MAX_FACT_QUERIES`
- `AGENT_USE_HYDE`, `AGENT_USE_FACT_QUERIES`, `AGENT_USE_RETRY`
- `AGENT_LLM_BASE_URL`, `AGENT_LLM_MODEL`, `AGENT_LLM_API_KEY` (fallback: GATE_*)

### Example (agent-search)

```bash
# Streaming
curl -N -X POST "http://localhost:8093/v1/agent/stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","include_sources":true,"mode":"conservative"}'

# Non-streaming
curl -s -X POST "http://localhost:8093/v1/agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","include_sources":true}' | jq .
```

---

## Deep-Research (port 8094)

Iterative research (LangGraph): **plan** → **scope** (plan + queries) → **research** loop (batch Gate calls, notes, next queries) → early stop on min gain → **write** (streaming report).

### Endpoint

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/deep-research/stream` | Single endpoint; response is SSE stream |

### Request body (DeepResearchRequest)

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Research question (required) |
| `history` | array | Chat history `[{role, content}]` |
| `filters` | object | Gate filters (`project_id`, `source`, etc.) |
| `include_sources` | bool | Include sources in response (default true) |
| `max_iterations` | int | Max research loop iterations (default 2) |
| `retrieval_mode` | string | hybrid \| bm25 \| vector |
| `top_k` | int | Top-k per Gate request |
| `rerank` | bool | Enable rerank |
| `use_web_search` | bool | Enable web search |
| `web_search_num` | int | Max web search results |
| `web_search_timeout_s` | float | Web search timeout (s) |

### SSE events (deep-research)

Server sends `data: <json>` lines:

| type | Description |
|------|-------------|
| `progress` | Stage: `plan`, `scope`, `research`, `write`, `done`; may include `stage`, `percent`, `message` |
| `retrieval` | Intermediate hits/context (if requested) |
| `token` | Report fragment (streaming text) |
| `done` | Final: `answer` (full report), `sources` |
| `error` | Error: `error` (string) |

### Environment (deep-research)

- **Gate / LLM:** `DEEP_GATE_URL`, `DEEP_GATE_TIMEOUT_S`; `DEEP_LLM_PROVIDER`, `DEEP_LLM_BASE_URL`, `DEEP_LLM_MODEL`, `DEEP_LLM_API_KEY`, `DEEP_LLM_TEMPERATURE` (fallback: GATE_*)
- **Iterations:** `DEEP_MAX_ITERATIONS` (max research loop iterations)
- **Early stop:** `DEEP_EARLY_STOP_MIN_GAIN` (min gain to continue)
- **Batch:** `DEEP_RESEARCH_BATCH`, `DEEP_RESEARCH_MAX_DOCS`, `DEEP_RESEARCH_MAX_CHARS`
- **Retrieval:** `DEEP_TOP_K`, `DEEP_RERANK`, `DEEP_RETRIEVAL_MODE`
- **Web search:** `DEEP_USE_WEB_SEARCH`; when enabled: `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_NUM`, `WEB_SEARCH_TIMEOUT_S`
- **Report template:** `DEEP_RESEARCH_TEMPLATE`
- **Logging:** `DEEP_LOG_LEVEL`
- **MCP (optional):** `DEEP_MCP_CONFIG` — path to MCP gate config

### Example (deep-research)

```bash
# Streaming report
curl -N -X POST "http://localhost:8094/v1/deep-research/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main approaches to RAG in 2024?",
    "include_sources": true,
    "max_iterations": 3,
    "filters": {"project_id": "demo"}
  }'
```

Parse events (jq):

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

## Summary

- Both services call **Gate** (`/v1/chat`) for retrieval and context; LLM used for plan, fact queries, and answer/report.
- **agent-search:** single pass (plan → Gate → answer with optional fact queries and retry).
- **deep-research:** loop of research iterations (batch Gate calls, notes), then streaming report.
- `.env` / `env.example`: deep-research uses `DEEP_*` prefix (see above).
