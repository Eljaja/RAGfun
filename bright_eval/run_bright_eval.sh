#!/usr/bin/env bash
# BRIGHT eval: direct indexing to retrieval (no gate/async — avoids hang).
# Usage: ./bright_eval/run_bright_eval.sh --bright-limit 30 --top-k 10
#
# Prereqs: docker compose up -d (retrieval required)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cleanup-sota-repo: docker-compose name=rugfunsota -> rugfunsota_rag-network
NETWORK="${RAGFUN_DOCKER_NETWORK:-rugfunsota_rag-network}"
RETRIEVAL_URL="${RETRIEVAL_BASE_URL:-http://retrieval:8080}"
SPLIT="${BRIGHT_SPLIT:-biology}"
DOCS_FROM_GOLD="${BRIGHT_DOCS_FROM_GOLD:-142}"
BRIGHT_LIMIT="${BRIGHT_LIMIT:-30}"
PROJECT_ID="${BRIGHT_PROJECT_ID:-bright}"
CONCURRENCY="${BRIGHT_CONCURRENCY:-10}"
TOP_K="${BRIGHT_TOP_K:-10}"
TIMEOUT="${BRIGHT_TIMEOUT:-60}"

# Use ifedotov BRIGHT cache if available
IFEDOTOV_HF="$(dirname "$(dirname "$ROOT")")/ifedotov/rag_fun/data/hf"
HF_DIR="${HF_HOME_HOST:-}"
if [[ -z "${HF_DIR}" ]]; then
  HF_DIR="$([[ -d "${IFEDOTOV_HF}" ]] && echo "${IFEDOTOV_HF}" || echo "$ROOT/data/hf")"
fi
OUT_DIR="${OUT_DIR_HOST:-$ROOT/out}"

mkdir -p "${HF_DIR}" "${OUT_DIR}"

docker run --rm --network "${NETWORK}" \
  -e HF_HOME=/hf \
  -e HF_DATASETS_OFFLINE=1 \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "${ROOT}:/work" \
  -v "${HF_DIR}:/hf" \
  -v "${OUT_DIR}:/out" \
  -w /work \
  python:3.12-slim \
  bash -lc "
    pip install -q --no-cache-dir --root-user-action=ignore httpx datasets pyarrow tenacity tqdm &&
    python scripts/index_bright.py \
      --retrieval-url '${RETRIEVAL_URL}' \
      --splits '${SPLIT}' \
      --docs-from-gold '${DOCS_FROM_GOLD}' \
      --project-id '${PROJECT_ID}' \
      --concurrency '${CONCURRENCY}' &&
    python scripts/bright_eval.py \
      --retrieval-url '${RETRIEVAL_URL}' \
      --bright --bright-split '${SPLIT}' \
      --bright-docs-from-gold '${DOCS_FROM_GOLD}' \
      --bright-limit '${BRIGHT_LIMIT}' \
      --out /out/bright_${SPLIT}_gold_only.jsonl \
      --project-id '${PROJECT_ID}' \
      --concurrency '${CONCURRENCY}' \
      --top-k '${TOP_K}' \
      --timeout '${TIMEOUT}' \
      \"\$@\"
  " -- "$@"
