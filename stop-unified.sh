#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Stopping ingestion stack (ragcircle-ingest)..."
docker compose -p ragcircle-ingest -f RAGcircle/docker-compose.yaml --profile apps down \
  --remove-orphans \
  --timeout 30

echo "[2/3] Stopping agent-search (ragfun-unified)..."
docker compose -p ragfun-unified stop agent-search --timeout 30 || true

echo "[3/3] Stopping retrieval/chat/ui stack (ragfun-unified)..."
docker compose -p ragfun-unified down \
  --remove-orphans \
  --timeout 30

echo "Done."
