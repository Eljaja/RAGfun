FROM python:3.11-slim AS base

# System dep for python-magic
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# --- Dependency layer (cached) ---
# Copy only lock + manifest so this layer is rebuilt only when deps change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Source layer ---
COPY py/ .

EXPOSE 8912
CMD ["uv", "run", "python", "main.py"]
