FROM python:3.11-slim AS base

# System dep for python-magic
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Path dependencies required by pyproject.toml
COPY retrieval_contract/ /retrieval_contract/
COPY generator_contract/ /generator_contract/

# --- Dependency layer (cached) ---
COPY gate_v2/pyproject.toml gate_v2/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Source layer ---
COPY gate_v2/py/ .

EXPOSE 8912
CMD ["uv", "run", "python", "main.py"]
