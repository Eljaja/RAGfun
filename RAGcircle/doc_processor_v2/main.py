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


from db_ops import DocumentEventDB
import asyncpg


# @asynccontextmanager
async def lifespan(app: FastAPI = None):
    cfg = AppConfig()

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

    qdrant = QdrantStore(url=cfg.qdrant_url, dimension=cfg.embedder_dim)
    await qdrant.ensure_collection(cfg.qdrant_collection)

    opensearch = BM25Store(url=cfg.opensearch_url)
    await opensearch.ensure_index(cfg.opensearch_index)

    pool = await asyncpg.create_pool(cfg.db_addr)
    document_event_db = DocumentEventDB(pool)
    await document_event_db.ensure_schema()

    deps = PipelineDeps(
        vlm=vlm,
        settings=settings,
        embedder=embedder,
        qdrant=qdrant,
        opensearch=opensearch,
        qdrant_collection=cfg.qdrant_collection,
        opensearch_index=cfg.opensearch_index,
        embed_batch_size=cfg.embed_batch_size,
        event_db_docs=document_event_db,
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
        await consumer_task
        # try:
        #     yield
        # finally:
        #     consumer_task.cancel()
        #     #try:
        #     #    await consumer_task
        #     #except asyncio.CancelledError:
        #     await consumer_task    
        #     await embedder.close()
        #     await qdrant.close()
        #     await opensearch.close()




if __name__ == "__main__":
    import asyncio
    asyncio.run(lifespan())


