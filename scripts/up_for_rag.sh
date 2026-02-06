#!/usr/bin/env bash
# Поднять стек szavodnov/RAGfun (rugfunsota) для работы RAG и BRIGHT eval.
#
# Usage:
#   ./scripts/up_for_rag.sh           # полный стек
#   ./scripts/up_for_rag.sh --bright  # только для BRIGHT eval (retrieval + deps)
#
# GPU 3:
#   NVIDIA_DEVICE_EMBED=3 NVIDIA_DEVICE_RERANK=3 NVIDIA_DEVICE_VLLM=3 ./scripts/up_for_rag.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Внешние сети (document-storage, retrieval)
for net in presign_rustfs-net rag_fun_rag-network; do
  if ! docker network inspect "$net" &>/dev/null; then
    echo "Creating network: $net"
    docker network create "$net"
  fi
done

if [[ "${1:-}" == "--bright" ]]; then
  echo "Starting minimal stack for BRIGHT eval..."
  docker compose up -d --build opensearch qdrant postgres pgbouncer infinity-embed infinity-rerank retrieval
  echo ""
  echo "Retrieval: http://localhost:8085"
  echo "Run: ./bright_eval/run_bright_eval.sh --bright-limit 30 --top-k 10"
else
  echo "Starting full stack..."
  docker compose up -d --build
  echo ""
  echo "Gate: http://localhost:8092"
  echo "Retrieval: http://localhost:8085"
  echo "UI: http://localhost:3300"
fi
