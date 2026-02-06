#!/usr/bin/env bash
# Быстрый smoke-тест RAG (~15 сек). Retrieval на localhost:8085.
# Usage: ./scripts/smoke_test.sh
set -e
cd "$(dirname "$0")/.."
exec python scripts/smoke_test.py
