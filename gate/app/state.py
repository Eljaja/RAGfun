"""Application state and lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients import DocProcessorClient, DocumentStorageClient, LLMClient, RetrievalClient
from app.config import Settings, load_settings
from app.logging_setup import setup_json_logging
from app.queue import RabbitPublisher

logger = logging.getLogger("gate")


class AppState:
    """Global application state."""
    settings: Settings | None = None
    config_error: str | None = None
    retrieval: RetrievalClient | None = None
    llm: LLMClient | None = None
    storage: DocumentStorageClient | None = None
    doc_processor: DocProcessorClient | None = None
    publisher: RabbitPublisher | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for RAG Gate."""
    try:
        state.settings = load_settings()
    except Exception as e:
        state.config_error = str(e)
        setup_json_logging("INFO")
        logger.error("config_error", extra={"extra": {"error": state.config_error}})
        yield
        return

    setup_json_logging(state.settings.log_level)
    state.retrieval = RetrievalClient(base_url=str(state.settings.retrieval_url), timeout_s=state.settings.retrieval_timeout_s)
    state.llm = LLMClient(
        provider=state.settings.llm_provider,
        base_url=str(state.settings.llm_base_url) if state.settings.llm_base_url else None,
        api_key=state.settings.llm_api_key.get_secret_value() if state.settings.llm_api_key else None,
        model=state.settings.llm_model,
        timeout_s=state.settings.llm_timeout_s,
    )
    if state.settings.storage_url:
        state.storage = DocumentStorageClient(base_url=str(state.settings.storage_url), timeout_s=state.settings.storage_timeout_s)
    if state.settings.doc_processor_url:
        state.doc_processor = DocProcessorClient(
            base_url=str(state.settings.doc_processor_url),
            timeout_s=state.settings.doc_processor_timeout_s,
        )
    if state.settings.rabbit_url:
        state.publisher = RabbitPublisher(url=str(state.settings.rabbit_url), queue_name=state.settings.rabbit_queue)
        try:
            await state.publisher.start()
        except Exception as e:
            # Degrade gracefully, but keep the publisher object:
            # it can reconnect lazily on the first publish attempt.
            logger.error("rabbit_publisher_init_failed", extra={"extra": {"error": str(e)}})
    yield
    if state.publisher:
        await state.publisher.close()

