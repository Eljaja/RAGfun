FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Path dependencies required by pyproject.toml
COPY retrieval_contract/ /retrieval_contract/
COPY generator_contract/ /generator_contract/

# --- Dependency layer (cached) ---
COPY generator_v2/pyproject.toml generator_v2/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Source layer ---
COPY generator_v2/py/ .

EXPOSE 8930
CMD ["uv", "run", "python", "main.py"]
