FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pika \
    boto3 \
    botocore \
    pydantic

# Copy the application
COPY bridge.py .

# Expose the port
EXPOSE 9999

# Run the application
CMD ["uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "9999"]