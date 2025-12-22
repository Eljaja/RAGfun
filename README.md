# RAG Experiments Playground

Some short and fun experiments with SOTA-style RAG pipelines, combined with hands-on learning and architectural exploration.
This repository is focused on reproducing large-scale RAG architectures in a simplified and experimental form. The main goal is to explore how modern retrieval techniques, stateless services, and composable pipelines behave when combined together.
The project is intentionally a work in progress and serves as a sandbox for testing ideas, patterns, and benchmarks rather than a production-ready system.

## Scope

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

## Current Direction

- Implement and test SOTA-style RAG pipelines
- Explore hybrid retrieval strategies combining dense and sparse methods
- Keep services stateless where possible to reflect scalable production patterns

## Planned Work

- Add and evaluate benchmarks such as T^2-RAGBench
- Introduce an optional continuous GraphRAG workflow inspired by tools like zepgraph
- Add deep research style pipelines and agentic search approaches
- Compare different pipeline compositions and retrieval strategies

## Status

- Work in progress
- Experimental by design
- Architecture and components may change frequently

## Disclaimer

This repository is primarily for research, learning, and experimentation. Expect incomplete features, refactors, and breaking changes as new ideas are tested.
