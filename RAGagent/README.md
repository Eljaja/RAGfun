# RAGagent

Services and shared code for **agent-search** and **deep-research**.

## Layout

- `agent-search/` — FastAPI agent: plan → `retrieval.search` → quality → fact queries → answer (optional **factoid** style). See [docs/AGENT_SEARCH.md](../docs/AGENT_SEARCH.md).
- `agent_common/` — Shared prompts, retrieval helpers, **`AsyncRetrievalClient`** (agent-search), **`AsyncGateClient`** (deep-research + legacy), tools.

## Docker (agent-search)

Build from this directory (`RAGagent`) so `COPY agent_common` / `COPY agent-search` paths match the Dockerfile:

```bash
cd RAGagent
docker build -f agent-search/Dockerfile -t ragagent-agent-search .
```

Configure **`AGENT_RETRIEVAL_URL`** (e.g. `http://retrieval:8080`) and LLM variables per [docs/AGENT_SEARCH.md](../docs/AGENT_SEARCH.md).
