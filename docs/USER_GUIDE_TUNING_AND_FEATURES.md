# Tuning and features guide

Short reference for adaptive-k, chunking (semchunk), and recent defaults.

---

## Adaptive-k

**What it does:** Instead of returning a fixed number of chunks (e.g. always 10), the pipeline cuts the list at the **steepest score drop** between two consecutive hits. So the number of returned chunks varies per query (between `min_k` and `max_k`).

**Algorithm:** After fusion/rerank, scores are ordered. For each pair of consecutive scores, compute the gap `s[i] - s[i+1]`. The cut position is right after the index where this gap is largest. Final k is clamped to `[adaptive_k_min, adaptive_k_max]`.

**Where it runs:**
- **Retrieval service** (optional): applied to search results before response. Env: `RAG_ADAPTIVE_K_ENABLED`, `RAG_ADAPTIVE_K_MIN`, `RAG_ADAPTIVE_K_MAX`.
- **Gate** (optional): applied when calling retrieval from chat. Env: `GATE_ADAPTIVE_K_ENABLED`, `GATE_ADAPTIVE_K_MIN`, `GATE_ADAPTIVE_K_MAX`. Per request: body field `use_adaptive_k: true/false` overrides the env default.
- **Agent-search**: forwards `use_adaptive_k` from request to Gate; no extra env.

**How to enable:**

1. **Retrieval** (if you want adaptive-k inside retrieval):
   - In `.env` or `docker-compose` for the retrieval service:
   - `RAG_ADAPTIVE_K_ENABLED=true`
   - `RAG_ADAPTIVE_K_MIN=3` (default)
   - `RAG_ADAPTIVE_K_MAX=24` (default)

2. **Gate** (typical: adaptive-k when using chat/agent):
   - In `.env` or environment for the gate service:
   - `GATE_ADAPTIVE_K_ENABLED=true`
   - `GATE_ADAPTIVE_K_MIN=3`
   - `GATE_ADAPTIVE_K_MAX=24`
   - Optional: `GATE_ADAPTIVE_K_MULTI_QUERY=after_rrf` to apply adaptive-k after merging multi-query results (default `off`).

3. **Per request** (Gate chat, agent-search):
   - In the JSON body: `"use_adaptive_k": true` (or `false` to disable regardless of env).

**Metrics:** Prometheus histograms `rag_adaptive_k_chunks` (retrieval) and `gate_adaptive_k_chunks` (Gate) record the chosen k per request.

---

## Chunking: semantic, token, semchunk

Chunking is used when indexing in **document** mode (retrieval or doc-processor splits the text).

**Strategies:**

| Strategy   | Env value   | Description |
|-----------|-------------|-------------|
| **semantic** | `semantic` | Section-based: splits on Markdown headings and numbered sections, then fits segments within `max_tokens`. Overlap only when a section exceeds max_tokens. |
| **token**    | `token`    | Paragraph-aware + sliding window: splits on double newline, then sliding window with overlap. |
| **semchunk** | `semchunk` | Library-based: uses the `semchunk` package for semantically aware splits. Requires `pip install semchunk tiktoken`. |

**Env (retrieval service):**
- `RAG_CHUNK_STRATEGY` — `semantic` | `token` | `semchunk`
- `RAG_CHUNK_MAX_TOKENS` — max tokens per chunk (default 1024 after tuning)
- `RAG_CHUNK_OVERLAP_TOKENS` — overlap in tokens (default 50)
- `RAG_CHUNK_ENCODING` — tiktoken encoding for semchunk/token (default `cl100k_base`)

**Current defaults (from BRIGHT tuning):** `RAG_CHUNK_STRATEGY=semantic`, `RAG_CHUNK_MAX_TOKENS=1024`, `RAG_CHUNK_OVERLAP_TOKENS=50`. Tuned on BRIGHT benchmark (all 12 splits, 1384 queries). To use semchunk, set `RAG_CHUNK_STRATEGY=semchunk` and ensure the retrieval image has `semchunk` and `tiktoken` installed.

**Re-index:** Changing chunk strategy or parameters requires re-indexing documents; existing chunks are not updated automatically.

---

## Top-k: from plan vs override

**Gate chat:** `top_k` is taken from the request body if present, otherwise from env `GATE_TOP_K` (default 10). So every chat request can override how many chunks retrieval returns.

**Agent-search:** The LLM **plans** retrieval (mode, top_k, rerank, HyDE). The plan’s `top_k` is then clamped to `[AGENT_TOP_K_MIN, AGENT_TOP_K_MAX]` (env, default 5..24). The request body can **override** this: send `"top_k": 12` to force 12 chunks regardless of the plan. So: request `top_k` → else plan’s top_k → clamped by min/max. This lets the model choose retrieval size per query while keeping bounds.

---

## Agent-search: mode presets and tools

**Mode presets** (request body `mode`):
- `minimal` — no HyDE, no fact queries, no retry; max_llm_calls=4. Fast, few LLM calls.
- `conservative` — same but max_llm_calls=6.
- `aggressive` — HyDE, fact queries, retry; max_llm_calls=16, max_fact_queries=4. Best recall, more cost.

Explicit body fields (`use_hyde`, `use_fact_queries`, etc.) override the preset.

Agent-search uses the configured retrieval service for all context.

**Tools:** Request body `use_tools: true` enables calculator and sandboxed code execution for the LLM when it needs computation.

---

## Streaming and cancellation

**Agent-search** (`/v1/agent/stream`): SSE stream; client disconnect is detected and in-flight Gate/LLM work can be aborted depending on implementation.

---

## Important updates (summary)

- **Chunking defaults:** Tuned on full BRIGHT (12 splits, 1384 queries). Defaults: 1024 tokens, 50 overlap, strategy semantic. See `docker-compose.yml` and `env.example`.
- **BRIGHT tuning script:** `scripts/bright_tune_chunking.py` — iterates over chunk configs, re-indexes BRIGHT, runs eval, writes `results/bright_chunk_tune_summary.json`. Full benchmark: `--splits all --eval-splits all --docs-from-gold 100000`.
- **Adaptive-k:** Optional everywhere (retrieval and Gate). Off by default. Enable via env or per-request `use_adaptive_k`. Metrics: `rag_adaptive_k_chunks`, `gate_adaptive_k_chunks`.
- **Top-k from plan (agent):** LLM chooses retrieval top_k within AGENT_TOP_K_MIN/MAX; request can override with body `top_k`. Gate uses `payload.top_k` or `GATE_TOP_K`.
- **Agent-search:** Mode presets (minimal/conservative/aggressive), tools (calculator, code). See `docs/AGENT_SEARCH.md` for full API.
- **Streaming:** Agent-search supports cancellation on client disconnect.
- **Repo:** Comments and docs in English; no emojis in docs.

---

## What changed on this branch (by commit)

| Commit / area | What it does |
|---------------|--------------|
| **add semchunk, setup hyperparameters** | Chunking defaults 1024/50 (BRIGHT-tuned); README, env.example, docker-compose; repo English, no emojis; BRIGHT tuning script. |
| **add adaptive-k** | Adaptive-k support in retrieval and Gate (env + per-request); optional, off by default. |
| **improve LLM top-k preference, adaptive-k optional** | Agent-search: LLM plan chooses top_k; request can override; Gate uses payload.top_k or settings; mode presets (minimal/conservative/aggressive); tools (calculator, code); adaptive-k optional end-to-end. |
| **chore: gitignore** | bench, bright_eval, eval, scripts, results in .gitignore. |
| **feat: modes, cancellation, metrics, OpenAPI** | Agent-search mode presets; cancellation; Prometheus metrics; OpenAPI specs. |
| **feat(agent): agent-search improvements** | Agent-search behavior and API improvements. |
| **feat: agent-search, agent_common, docs, scripts** | Initial agent-search service; shared agent_common; docs and scripts. |
| **cleanup: remove extra docs, BEIR tests; keep SOTA, BRIGHT** | Trim repo; keep SOTA improvements and BRIGHT validation. |

---

## Quick reference: where to set what

| What            | Where to set |
|-----------------|--------------|
| Adaptive-k (retrieval) | Retrieval env: `RAG_ADAPTIVE_K_ENABLED`, `RAG_ADAPTIVE_K_MIN`, `RAG_ADAPTIVE_K_MAX` |
| Adaptive-k (chat/agent) | Gate env: `GATE_ADAPTIVE_K_ENABLED`, `GATE_ADAPTIVE_K_MIN`, `GATE_ADAPTIVE_K_MAX` |
| Adaptive-k per request | Request body: `use_adaptive_k: true \| false` |
| Chunk strategy/size   | Retrieval env: `RAG_CHUNK_STRATEGY`, `RAG_CHUNK_MAX_TOKENS`, `RAG_CHUNK_OVERLAP_TOKENS` |
| Chunk encoding (semchunk) | Retrieval env: `RAG_CHUNK_ENCODING` (e.g. `cl100k_base`) |
| top_k (Gate chat)     | Request body `top_k` or env `GATE_TOP_K` |
| top_k (agent-search)  | Request body `top_k` (5..24) or from LLM plan; bounds: `AGENT_TOP_K_MIN`, `AGENT_TOP_K_MAX` |
| Agent mode preset     | Request body `mode`: `minimal` \| `conservative` \| `aggressive` |
