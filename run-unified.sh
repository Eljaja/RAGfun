#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local overrides/secrets (gitignored) if present.
if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env.local"
  set +a
fi

# Shared stub project across retrieval + ingestion
export STUB_PROJECT_ID="${STUB_PROJECT_ID:-default}"

# Expose retrieval stack's stores on the host ports gate_v2/doc_processor_v2 expect
export RUGFUNSOTA_QDRANT_HTTP="${RUGFUNSOTA_QDRANT_HTTP:-8903}"
export RUGFUNSOTA_OS_PORT="${RUGFUNSOTA_OS_PORT:-8905}"
export RUGFUNSOTA_PG_PORT="${RUGFUNSOTA_PG_PORT:-5438}"

# Keep RAGcircle's own postgres on a different host port to avoid collision.
export RAGCIRCLE_PG_PORT="${RAGCIRCLE_PG_PORT:-5439}"

# Optional for quick local testing (no tenant API key needed for chat)
export GATE_REQUIRE_TENANT_AUTH="${GATE_REQUIRE_TENANT_AUTH:-false}"

# LLM defaults (can be overridden via .env.local or env vars).
export GATE_LLM_PROVIDER="${GATE_LLM_PROVIDER:-openai_compat}"
export GATE_LLM_BASE_URL="${GATE_LLM_BASE_URL:-https://llm.c.singularitynet.io/v1}"
export GATE_LLM_MODEL="${GATE_LLM_MODEL:-minimax/minimax-m2.1}"
export GATE_LLM_API_KEY="${GATE_LLM_API_KEY:-sk-FQ0o8MR5WWeLgqCTP_cZSxtpL8xbpopVCTFxSs5GQk0}"

# Embedder model for RAGcircle's Infinity service (phase 2).
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"

# Required by RAGfun/docker-compose.yml (external network used by document-storage).
EXTERNAL_PRESIGN_NET="presign_rustfs-net"
if ! docker network inspect "${EXTERNAL_PRESIGN_NET}" >/dev/null 2>&1; then
  echo "[0/2] Creating missing external network: ${EXTERNAL_PRESIGN_NET}"
  docker network create "${EXTERNAL_PRESIGN_NET}" >/dev/null
fi

echo "[1/3] Starting retrieval/chat stack..."
docker compose -p ragfun-unified up -d \
  opensearch qdrant infinity-embed infinity-rerank retrieval \
  postgres pgbouncer document-storage rabbitmq rag-gate ui

echo "[1/3] Waiting for retrieval readiness on http://localhost:8085/v1/readyz ..."
until curl -fsS http://localhost:8085/v1/readyz >/dev/null; do
  sleep 2
done

echo "[2/3] Starting agent-search (deep-research disabled in unified mode)..."
docker compose -p ragfun-unified --profile agent-search up -d agent-search

echo "[2/3] Waiting for agent-search readiness on http://localhost:8093/v1/readyz ..."
until curl -fsS http://localhost:8093/v1/readyz >/dev/null; do
  sleep 2
done

echo "[3/3] Starting ingestion infra + embedder..."
docker compose -p ragcircle-ingest -f RAGcircle/docker-compose.yaml --profile apps up -d \
  postgres rabbitmq rustfs infinity

echo "[3/3] Waiting for embedder readiness on http://localhost:8902/models ..."
until curl -fsS http://localhost:8902/models >/dev/null; do
  sleep 2
done

echo "[3/3] Starting ingestion API/worker..."
docker compose -p ragcircle-ingest -f RAGcircle/docker-compose.yaml --profile apps up -d \
  gate doc-processor nginx-gate

echo "Done."
echo "ui:                  http://localhost:3301"
echo "rag-gate readyz:     http://localhost:8092/v1/readyz"
echo "retrieval readyz:    http://localhost:8085/v1/readyz"
echo "agent-search readyz: http://localhost:8093/v1/readyz"
echo "gate_v2 health:      http://localhost:8916/public/health"
