FROM python:3.11-slim AS base

# System deps for pdf2image (poppler) and office conversion (libreoffice)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libreoffice-core \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# --- Dependency layer (cached) ---
# Copy only lock + manifest so this layer is rebuilt only when deps change
COPY py/pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Source layer ---
COPY py/ .

CMD ["uv", "run", "python", "main.py"]
