FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pika
COPY bridge.py .
CMD ["python", "bridge.py"]