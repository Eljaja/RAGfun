# SDK Smoke Tests

This folder contains live smoke checks for the SDK against a running gateway service.

## What it validates

- SDK project APIs: `create`, `ensure`, `list`, `get`, `delete`
- SDK upload helper: `upload_document`
- Chat API (preferred): `client.chat.create(...)`
- Chat API compatibility path: `client.chat.completions.create(...)`
- Service availability and end-to-end request handling through the gateway

## Prerequisites

- Running gateway service URL
- Valid API key

## Run

```bash
export RAG_GATEWAY_URL="https://your-gateway-host"
export RAG_API_KEY="sk-..."
python -m client.smoke.run_smoke
```

Optional flags:

```bash
# use existing file for upload check
python -m client.smoke.run_smoke --upload-file /path/to/doc.txt

# keep created projects for inspection
python -m client.smoke.run_smoke --keep-projects
```

## Agent Benchmark

Compare the old and new `agent-search` stacks on a sampled SQuAD v1.1 benchmark:

```bash
python -m client.smoke.compare_agents_squad --sample-size 40 --output /tmp/agent_compare.json
```

Default targets:

- old retrieval: `http://127.0.0.1:8085`
- old agent: `http://127.0.0.1:8093`
- new retrieval: `http://127.0.0.1:18085`
- new agent: `http://127.0.0.1:18093`

The script reports:

- `exact_match_rate`
- `contains_gold_rate`
- `gold_in_context_rate`
- `mean_f1`
- `citation_rate`
- `partial_rate`
- `latency`
- `mean_llm_calls`
- `mean_retrieval_calls`
- `retry_rate`
- `error_count`
