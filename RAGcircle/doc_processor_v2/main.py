"""
Document Processor Service v2

Entry point (FastAPI + lifespan):
- Build clients (VLM, embedder, stores, S3 client)
- Start RabbitMQ consumer task

Core logic is split into:
- `amqp_consumer.py`: RabbitMQ consume loop + retry/DLQ behavior
- `events.py`: event parsing + event helpers
- `pipeline.py`: download/extract/chunk/ingest + delete pipeline
- `config.py`: environment configuration
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from aiobotocore.session import get_session
from fastapi import FastAPI

from amqp_consumer import consume_rabbitmq
from config import AppConfig
from embed_caller import Embedder
from pipeline import PipelineDeps
from processing import Settings, VLMClient
from store import BM25Store, QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AppConfig.from_env()

    vlm = VLMClient(
        base_url=cfg.vlm_base_url,
        api_key=cfg.vlm_api_key,
        model=cfg.vlm_model,
        timeout_s=cfg.vlm_timeout,
    )

    settings = Settings(
        max_pages=cfg.proc_max_pages,
        max_px=cfg.proc_max_px,
        vlm_concurrency=cfg.proc_vlm_concurrency,
        chunk_size_chars=cfg.chunk_size_chars,
        chunk_overlap_chars=cfg.chunk_overlap_chars,
    )

    embedder = Embedder(base_url=cfg.embedder_url, model=cfg.embedder_model)

    qdrant = QdrantStore(url=cfg.qdrant_url, collection=cfg.qdrant_collection, dimension=cfg.embedder_dim)
    await qdrant.ensure_collection()

    opensearch = BM25Store(url=cfg.opensearch_url, index=cfg.opensearch_index)
    await opensearch.ensure_index()

    deps = PipelineDeps(
        vlm=vlm,
        settings=settings,
        embedder=embedder,
        qdrant=qdrant,
        opensearch=opensearch,
        embed_batch_size=cfg.embed_batch_size,
    )

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=cfg.s3_endpoint,
        aws_access_key_id=cfg.s3_access_key,
        aws_secret_access_key=cfg.s3_secret_key,
        region_name=cfg.s3_region,
    ) as s3_client:
        consumer_task = asyncio.create_task(consume_rabbitmq(s3_client=s3_client, deps=deps, cfg=cfg))
        try:
            yield
        finally:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
            await embedder.close()
            await qdrant.close()
            await opensearch.close()


app = FastAPI(
    lifespan=lifespan,
    title="Document Processor v2",
    description="Async document processing service - consumes S3 events from RabbitMQ",
)


@app.get("/")
async def root():
    return {
        "service": "doc-processor-v2",
        "status": "running",
        "description": "Async document processing from RabbitMQ events",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "doc-processor-v2"}


# Run with:
# uvicorn main:app --host 0.0.0.0 --port 9998

