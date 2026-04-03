# Agent-Search API

Service: agent-search (port 8093 by default). Profile: `agent-search`.

Flow: **plan** → optional HyDE → **retrieval.search** → quality check → optional **fact queries** → **answer** (with citation [1], [2]).

## Endpoints

### `POST /v1/agent`

Non-streaming: returns full JSON with `answer`, `sources`, `context`, `mode`, `partial`, `degraded`.

### `POST /v1/agent/stream`

Server-Sent Events (SSE). Event types: `init`, `trace`, `retrieval`, `token`, `done`, `error`.

## Request body (both endpoints)

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | User question (required) |
| `history` | array | Conversation history `[{role, content}]` (optional) |
| `filters` | object | Retrieval filters (e.g. `project_id`, `source`) |
| `include_sources` | bool | Include sources in response (default `true`) |
| `top_k` | int | Override retrieval top_k (5..24); else from plan |
| `use_adaptive_k` | bool | Adaptive-k cutoff (override env) |
| `max_llm_calls` | int | Max LLM calls per request |
| `max_fact_queries` | int | Max fact queries per request |
| `use_hyde` | bool | Enable HyDE (override env) |
| `hyde_num` | int | HyDE variants (1 or 3) |
| `use_fact_queries` | bool | Enable fact queries (override env) |
| `use_retry` | bool | Retry on incomplete answer (override env) |
| `use_tools` | bool | Enable calculator & code execution tools |
| `mode` | string | Preset: `minimal` \| `conservative` \| `aggressive` |

**Mode presets:**

- `minimal`: no HyDE, no fact queries, no retry, max_llm_calls=4
- `conservative`: no HyDE, no fact queries, no retry, max_llm_calls=6
- `aggressive`: HyDE, fact queries, retry, max_llm_calls=16, max_fact_queries=4

## Environment (agent-search service)

| Variable | Description |
|----------|-------------|
| `AGENT_RETRIEVAL_URL` | Retrieval base URL (e.g. http://retrieval:8080) |
| `AGENT_RETRIEVAL_TIMEOUT_S` | Timeout for retrieval calls |
| `AGENT_MAX_LLM_CALLS` | Default max LLM calls per request |
| `AGENT_MAX_FACT_QUERIES` | Default max fact queries |
| `AGENT_USE_HYDE` | Enable HyDE by default |
| `AGENT_USE_FACT_QUERIES` | Enable fact queries by default |
| `AGENT_USE_RETRY` | Enable retry on incomplete by default |
| `AGENT_LLM_BASE_URL`, `AGENT_LLM_MODEL`, `AGENT_LLM_API_KEY` | LLM for plan/answer (fallback: GATE_* ) |

## Example

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
