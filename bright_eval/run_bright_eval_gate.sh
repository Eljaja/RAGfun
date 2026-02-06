#!/usr/bin/env bash
# BRIGHT eval via gate (full pipeline: gate + retrieval + optional judge).
# Uses pipeline-tests image. Alternative to bright_eval/run_bright_eval.sh (retrieval-only).
#
# Build first: docker build -t rugfunsota-pipeline-tests:latest -f pipeline-tests/Dockerfile pipeline-tests/
#
# Usage: ./bright_eval/run_bright_eval_gate.sh --domain biology --limit 50 --top_k 10

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK="${RAGFUN_DOCKER_NETWORK:-rugfunsota_rag-network}"
OUT_FILE_DEFAULT="/out/bright_${BRIGHT_DOMAIN:-biology}_gold_only.jsonl"

IFEDOTOV_HF="$(dirname "$(dirname "$ROOT")")/ifedotov/rag_fun/data/hf"
HF_DIR="${HF_HOME_HOST:-}"
[[ -z "${HF_DIR}" ]] && HF_DIR="$([[ -d "${IFEDOTOV_HF}" ]] && echo "${IFEDOTOV_HF}" || echo "$ROOT/data/hf")"
OUT_DIR="${OUT_DIR_HOST:-$ROOT/out}"
mkdir -p "${OUT_DIR}"

docker run --rm --network "${NETWORK}" \
  --env-file "${ROOT}/.env" 2>/dev/null || true \
  -e GATE_BASE_URL=http://rag-gate:8090 \
  -e RETRIEVAL_BASE_URL=http://retrieval:8080 \
  -e HF_HOME=/hf \
  -v "${HF_DIR}:/hf" \
  -v "${OUT_DIR}:/out" \
  rugfunsota-pipeline-tests:latest \
  python -m bench.bright_eval \
    --domain "${BRIGHT_DOMAIN:-biology}" \
    --limit "${BRIGHT_LIMIT:-0}" \
    --top_k "${BRIGHT_TOP_K:-10}" \
    --out "${BRIGHT_OUT:-$OUT_FILE_DEFAULT}" \
    "$@"
