FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

ARG PADDLE_GPU_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/
ARG PADDLE_GPU_PACKAGE=paddlepaddle-gpu==3.3.1

WORKDIR /app

# LibreOffice is used for DOC/DOCX conversion to PDF (best-effort).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libreoffice-core \
    fonts-dejavu \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --index-url ${PADDLE_GPU_INDEX_URL} ${PADDLE_GPU_PACKAGE}

COPY app /app/app

EXPOSE 8082

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]

















