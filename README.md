# RAG Experiments Playground

Some short and fun experiments with SOTA-style RAG pipelines, combined with hands-on learning and architectural exploration.

## Run the pipeline (Docker Compose)

### Prerequisites

- Docker + Docker Compose v2 (`docker compose ...`)
- (Optional but recommended) NVIDIA GPU for `vllm-docling` (document-to-text via VLM)
- (Optional) Embeddings service on the host at `http://localhost:7997/embeddings` (the default `docker-compose.yml` expects it)

### Start

```bash
cp env.example .env
# edit .env if you want a real LLM key/provider

docker compose up -d --build
```

### Open

- UI: `http://localhost:3300`
- Gateway API: `http://localhost:8090`

### Stop

```bash
docker compose down
```

### Reset state (delete volumes)

```bash
docker compose down -v
```

## CPU-only deterministic e2e stack

```bash
make e2e-up
make e2e-test
make e2e-down
```

## Documentation

See [`docs/README.md`](./docs/README.md).
