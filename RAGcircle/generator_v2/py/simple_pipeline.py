"""Simple /chat pipeline: preset -> retrieve -> generate -> reflect."""

from __future__ import annotations

import logging

import httpx

from assessment import reflect
from context import build_context, extract_sources
from llm import LLMClient
from models import ChatRequest, ChatResponse, ChunkResult
from plan_builder import from_preset
from prompts import ANSWER_SYSTEM, ANSWER_USER
from retrieval_client import retrieve

logger = logging.getLogger(__name__)


async def simple_pipeline(
    request: ChatRequest,
    *,
    llm: LLMClient,
    http_client: httpx.AsyncClient,
    retrieval_url: str,
    gen_model: str,
    reflection_model: str,
    reflection_enabled: bool,
    max_context_chars: int = 6000,
    max_chunk_chars: int = 1200,
) -> ChatResponse:
    query = request.query
    do_reflect = request.reflection_enabled and reflection_enabled
    max_retries = request.max_retries if do_reflect else 0
    retries_used = 0
    answer = ""
    chunks: list[ChunkResult] = []

    for attempt in range(max_retries + 1):
        plan = from_preset(
            request.preset,
            top_k=request.top_k,
            rerank=request.rerank,
        )

        chunks = await retrieve(
            http_client,
            retrieval_url,
            project_id=request.project_id,
            query=query,
            plan=plan,
        )

        if not chunks:
            return ChatResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                chunks_used=0,
                retries_used=retries_used,
                query=request.query,
            )

        context_text = build_context(
            chunks,
            max_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM.format(lang="the same language as the question")},
            {"role": "user", "content": ANSWER_USER.format(
                history="", query=query, context=context_text,
            )},
        ]
        answer = await llm.complete(gen_model, messages)

        if not do_reflect or attempt >= max_retries:
            break

        try:
            reflection = await reflect(
                llm, reflection_model, query, context_text, answer,
            )
            logger.info(
                "Reflection attempt %d: complete=%s, requery=%s",
                attempt, reflection.complete, reflection.requery,
            )
            if reflection.complete:
                break
            query = reflection.requery or query
            retries_used += 1
        except Exception:
            logger.warning("Reflection failed on attempt %d, using current answer", attempt, exc_info=True)
            break

    return ChatResponse(
        answer=answer,
        sources=extract_sources(chunks),
        chunks_used=len(chunks),
        retries_used=retries_used,
        query=request.query,
    )
