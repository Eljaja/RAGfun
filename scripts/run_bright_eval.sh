#!/usr/bin/env bash
set -euo pipefail

# End-to-end BRIGHT eval runner inside the docker network (so it can reach `retrieval:8080`).
#
# Usage (from repo root):
#   ./scripts/run_bright_eval.sh
#
# Common overrides (env vars):
#   RAGFUN_DOCKER_NETWORK=rugfunsota_rag-network
#   BRIGHT_SPLIT=biology
#   BRIGHT_DOCS_FROM_GOLD=1000
#   BRIGHT_PROJECT_ID=bright
#   BRIGHT_CONCURRENCY=10
#   BRIGHT_TOP_K=10
#   BRIGHT_TIMEOUT=60
#
# Note: this will (1) index docs needed for the eval subset, then (2) run eval.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# cleanup-sota-repo: docker-compose name=rugfunsota -> rugfunsota_rag-network
NETWORK="${RAGFUN_DOCKER_NETWORK:-rugfunsota_rag-network}"
RETRIEVAL_URL="${RETRIEVAL_BASE_URL:-http://retrieval:8080}"

SPLIT="${BRIGHT_SPLIT:-biology}"
DOCS_FROM_GOLD="${BRIGHT_DOCS_FROM_GOLD:-1000}"
PROJECT_ID="${BRIGHT_PROJECT_ID:-bright}"
CONCURRENCY="${BRIGHT_CONCURRENCY:-10}"
TOP_K="${BRIGHT_TOP_K:-10}"
TIMEOUT="${BRIGHT_TIMEOUT:-60}"

HF_DIR="${HF_HOME_HOST:-$ROOT/data/hf}"
OUT_DIR="${OUT_DIR_HOST:-$ROOT/out}"

mkdir -p "${HF_DIR}" "${OUT_DIR}"

docker run --rm --network "${NETWORK}" \
  -e HF_HOME=/hf \
  -v "${ROOT}:/work" \
  -v "${HF_DIR}:/hf" \
  -v "${OUT_DIR}:/out" \
  -w /work \
  python:3.11-slim \
  bash -lc "
    python -m pip -q install --no-cache-dir httpx datasets pyarrow tenacity tqdm && \
    python scripts/index_bright.py \
      --retrieval-url '${RETRIEVAL_URL}' \
      --splits '${SPLIT}' \
      --docs-from-gold '${DOCS_FROM_GOLD}' \
      --project-id '${PROJECT_ID}' \
      --concurrency '${CONCURRENCY}' && \
    python scripts/bright_eval.py \
      --retrieval-url '${RETRIEVAL_URL}' \
      --bright --bright-split '${SPLIT}' \
      --bright-docs-from-gold '${DOCS_FROM_GOLD}' \
      --bright-limit 0 \
      --project-id '${PROJECT_ID}' \
      --concurrency '${CONCURRENCY}' \
      --top-k '${TOP_K}' \
      --timeout '${TIMEOUT}' \
      --out /out/bright_${SPLIT}_gold_only.jsonl \
      \"\$@\"
  " -- "$@"

