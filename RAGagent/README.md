# RAGagent

Code and shared libraries for **agent-search**.

## Layout

- `agent-search/` — FastAPI agent: plan → `retrieval.search` → quality checks → fact queries → answer, with optional **factoid** answer style. See [docs/AGENT_SEARCH.md](../docs/AGENT_SEARCH.md).
- `agent_common/` — Shared prompts, retrieval helpers, **`AsyncRetrievalClient`**, tools, and other modules consumed by agent-search.

## Docker (agent-search)

Build from this directory (`RAGagent`) so `COPY agent_common` / `COPY agent-search` paths match the Dockerfile:

```bash
cd RAGagent
docker build -f agent-search/Dockerfile -t ragagent-agent-search .
```

Configure **`AGENT_RETRIEVAL_URL`** (for example `http://retrieval:8080`) and LLM variables as described in [docs/AGENT_SEARCH.md](../docs/AGENT_SEARCH.md).
