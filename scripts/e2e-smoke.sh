#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8080}"
GATE_URL="${GATE_URL:-http://localhost:8090}"

UPSERT_PAYLOAD="${UPSERT_PAYLOAD:-${ROOT_DIR}/tests/data/upsert.json}"
SEARCH_PAYLOAD="${SEARCH_PAYLOAD:-${ROOT_DIR}/tests/data/search.json}"
CHAT_PAYLOAD="${CHAT_PAYLOAD:-${ROOT_DIR}/tests/data/chat.json}"

RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-40}"
RETRY_DELAY="${RETRY_DELAY:-5}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to run the smoke test" >&2
  exit 1
fi

wait_ready() {
  local url="$1"
  local name="$2"
  local attempt=1

  while (( attempt <= RETRY_ATTEMPTS )); do
    if response="$(curl -sS "$url")"; then
      ready_value="$(echo "$response" | jq -r '.ready // .ok // empty')" || ready_value=""
      if [[ "$ready_value" == "true" ]]; then
        echo "[$name] ready after $attempt attempt(s)"
        return 0
      fi
      echo "[$name] not ready yet (attempt $attempt/${RETRY_ATTEMPTS})"
    else
      echo "[$name] unreachable (attempt $attempt/${RETRY_ATTEMPTS})"
    fi
    ((attempt++))
    sleep "$RETRY_DELAY"
  done

  echo "[$name] failed to become ready after ${RETRY_ATTEMPTS} attempts" >&2
  return 1
}

expect_hits() {
  local response="$1"
  local hits
  hits="$(echo "$response" | jq '.hits | length')" || hits="0"
  if [[ "$hits" -lt 1 ]]; then
    echo "Expected at least one search hit but got $hits" >&2
    exit 1
  fi
}

expect_context() {
  local response="$1"
  local ctx
  ctx="$(echo "$response" | jq '.context | length')" || ctx="0"
  if [[ "$ctx" -lt 1 ]]; then
    echo "Expected chat context to include hits but got $ctx items" >&2
    exit 1
  fi
}

echo "Waiting for retrieval readiness at ${RETRIEVAL_URL}/v1/readyz"
wait_ready "${RETRIEVAL_URL}/v1/readyz" "retrieval"

echo "Waiting for gate readiness at ${GATE_URL}/v1/readyz"
wait_ready "${GATE_URL}/v1/readyz" "rag-gate"

echo "Upserting smoke-test content into retrieval"
upsert_response="$(curl -sS -X POST -H "Content-Type: application/json" --data @"${UPSERT_PAYLOAD}" "${RETRIEVAL_URL}/v1/index/upsert")"
echo "$upsert_response" | jq .
if [[ "$(echo "$upsert_response" | jq -r '.ok')" != "true" ]]; then
  echo "Upsert failed" >&2
  exit 1
fi

echo "Running search smoke query"
search_response="$(curl -sS -X POST -H "Content-Type: application/json" --data @"${SEARCH_PAYLOAD}" "${RETRIEVAL_URL}/v1/search")"
echo "$search_response" | jq .
expect_hits "$search_response"

echo "Running chat smoke query via rag-gate"
chat_response="$(curl -sS -X POST -H "Content-Type: application/json" --data @"${CHAT_PAYLOAD}" "${GATE_URL}/v1/chat")"
echo "$chat_response" | jq .
expect_context "$chat_response"

echo "Smoke test succeeded"
