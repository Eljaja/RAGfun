"""
Document Processor Service v2

Entry point:
- Build clients (VLM, embedder, stores, S3 client)
- Start RabbitMQ consumer task
- Graceful shutdown on SIGTERM/SIGINT

Core logic is split into:
- `amqp_consumer.py`: RabbitMQ consume loop + retry/DLQ behavior
- `s3_events.py`: event parsing + event helpers
- `pipeline.py`: download/extract/chunk/ingest + delete pipeline
- `config.py`: environment configuration
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

import asyncpg
from aiobotocore.session import get_session

from amqp_consumer import consume_rabbitmq
from config import AppConfig
from embed_caller import Embedder
from pipeline import PipelineDeps
from processing import Settings, VLMClient
from store import BM25Store, QdrantStore
from db_ops import DocumentEventDB


def _setup_logging(level: str) -> None:
    """Configure root logging once at process start.

    All modules use ``logging.getLogger("data.processing.*")`` — configuring
    the root logger ensures every child logger inherits the level + format.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quieten noisy third-party libraries — even at DEBUG we only want
    # to debug our own code, not heartbeat frames and HTTP internals.
    for noisy in ("aiobotocore", "botocore", "urllib3", "aio_pika", "aiormq", "pamqp", "httpx", "httpcore", "opensearch", "opensearchpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def main() -> None:
    cfg = AppConfig()
    _setup_logging(cfg.log_level)

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

    # do not need to ensure index anymore
    qdrant = QdrantStore(url=cfg.qdrant_url, dimension=cfg.embedder_dim)
    opensearch = BM25Store(url=cfg.opensearch_url)

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

    stack = AsyncExitStack()
    s3_client = await stack.enter_async_context(
        get_session().create_client(
            "s3",
            endpoint_url=cfg.s3_endpoint,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
            region_name=cfg.s3_region,
        )
    )

    consumer_task = asyncio.create_task(
        consume_rabbitmq(s3_client=s3_client, deps=deps, cfg=cfg)
    )

    # SIGTERM/SIGINT → cancel the consumer gracefully
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, consumer_task.cancel)

    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    finally:
        await stack.aclose()
        await vlm.close()
        await embedder.close()
        await qdrant.close()
        await opensearch.close()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
