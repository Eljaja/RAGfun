FROM michaelf34/infinity:latest

# Infinity ships its own venv at /app/.venv. We upgrade Transformers to a version
# that recognizes the `qwen3` architecture (needed for Qwen3-Embedding/Reranker).
SHELL ["/bin/bash", "-lc"]

RUN set -euxo pipefail; \
    source /app/.venv/bin/activate; \
    pip install --no-cache-dir -U "transformers>=4.51.0" "colpali-engine"; \
    python -c "import transformers; print('transformers', transformers.__version__)"

