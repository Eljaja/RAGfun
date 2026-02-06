FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

COPY tests /work/tests
COPY bench /work/bench

ENV GATE_BASE_URL=http://rag-gate:8090 \
    STORAGE_BASE_URL=http://document-storage:8081 \
    RETRIEVAL_BASE_URL=http://retrieval:8080 \
    PYTHONPATH=/work

CMD ["pytest", "-q", "-ra", "--timeout=120", "/work/tests"]

















