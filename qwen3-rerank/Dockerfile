FROM michaelf34/infinity:latest

# Reuse Infinity base (CUDA+torch+venv), but run our own small rerank server.
SHELL ["/bin/bash", "-lc"]

RUN set -euxo pipefail; \
    source /app/.venv/bin/activate; \
    pip install --no-cache-dir -U "transformers>=4.51.0" "colpali-engine"

WORKDIR /app
COPY qwen3_rerank /app/qwen3_rerank

ENV PYTHONUNBUFFERED=1
ENV RERANK_PORT=7998

# Base image has an ENTRYPOINT (infinity_emb). Reset it so our CMD runs.
ENTRYPOINT ["bash", "-lc"]
CMD ["source /app/.venv/bin/activate && uvicorn qwen3_rerank.main:app --host 0.0.0.0 --port ${RERANK_PORT} --log-level info"]



