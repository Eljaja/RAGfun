# Local Deploy With Shared vLLM

This setup lets you run a personal `RAGfun` stack on the same machine without colliding with colleagues' containers, ports, or the already running shared vLLM.

## Files

- Copy `.env.local.example` to `.env.local`
- Use `docker-compose.shared-vllm.override.yml` together with the base compose file

## What this changes

- Uses a separate `COMPOSE_PROJECT_NAME`
- Shifts your host ports away from the defaults
- Points `rag-gate` and `agent-search` to a shared OpenAI-compatible vLLM endpoint
- Keeps embeddings local to your personal stack
- Points `doc-processor` to a shared VLM endpoint too
- Prevents accidental startup of the local `vllm-docling`
- Moves Prometheus/Grafana/exporters behind an explicit `observability` profile

## Default shared endpoint discovered on this host

- `SHARED_LLM_BASE_URL=http://host.docker.internal:8124/v1`
- `SHARED_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct`
- `NVIDIA_DEVICE_EMBED=4`
- `RAG_RERANK_MODE=disabled`

Adjust `.env.local` if your team wants a different shared endpoint or model.

## Recommended startup

Minimal claw-like agent stack:

```bash
cp .env.local.example .env.local
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.shared-vllm.override.yml up -d retrieval rag-gate
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.shared-vllm.override.yml --profile agent-search up -d agent-search
```

Include document ingestion services too:

```bash
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.shared-vllm.override.yml up -d document-storage doc-processor ingestion-worker
```

Bring up the UI:

```bash
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.shared-vllm.override.yml up -d ui
```

## Smoke checks

```bash
curl http://localhost:18085/v1/readyz
curl http://localhost:18092/v1/readyz
curl http://localhost:18093/v1/readyz
curl http://127.0.0.1:8124/v1/models
```

## Notes

- `host.docker.internal` is already injected into the relevant services via `extra_hosts`
- This avoids touching colleagues' compose projects, but shared vLLM capacity is still shared
- Start with low concurrency and short smoke tests before heavier eval runs
