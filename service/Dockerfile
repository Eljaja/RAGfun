FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set to 1 only when using RAG_RERANK_PROVIDER=local (adds ~10GB: torch, sentence-transformers)
ARG INSTALL_RERANK=0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY requirements-rerank.txt /app/requirements-rerank.txt
RUN if [ "$INSTALL_RERANK" = "1" ]; then pip install --no-cache-dir -r /app/requirements-rerank.txt; fi

COPY app /app/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

