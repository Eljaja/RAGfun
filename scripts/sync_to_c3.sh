#!/bin/bash
# Перенос проекта RAGfun на хост c3
# Запускать с машины, где настроен SSH (Host c3 с ProxyJump jump)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

rsync -avz --progress \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude 'node_modules' \
  --exclude 'data/hf' \
  -e ssh \
  "$PROJECT_DIR/" c3:~/RAGfun/

echo "Done. Project synced to c3:~/RAGfun/"
