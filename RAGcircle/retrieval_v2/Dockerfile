FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Path dependency required by pyproject.toml
COPY retrieval_contract/ /retrieval_contract/

# --- Dependency layer (cached) ---
COPY retrieval_v2/pyproject.toml retrieval_v2/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Source layer ---
COPY retrieval_v2/py/ .

EXPOSE 8920
CMD ["uv", "run", "python", "main.py"]
