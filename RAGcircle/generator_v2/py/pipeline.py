from __future__ import annotations

import logging

import httpx

from llm import LLMClient, generate_answer, reflect
from models import ChunkResult, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


async def call_retrieval(
    client: httpx.AsyncClient,
    retrieval_url: str,
    project_id: str,
    query: str,
    top_k: int,
    strategy: str,
    rerank: bool,
) -> list[ChunkResult]:
    """Call the retrieval service over HTTP."""
    resp = await client.post(
        f"{retrieval_url.rstrip('/')}/retrieve",
        json={
            "project_id": project_id,
            "query": query,
            "top_k": top_k,
            "strategy": strategy,
            "rerank": rerank,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [ChunkResult(**c) for c in data["chunks"]]


async def rag_pipeline(
    request: ChatRequest,
    retrieval_url: str,
    http_client: httpx.AsyncClient,
    llm: LLMClient,
    gen_model: str,
    reflection_model: str,
    reflection_enabled: bool,
) -> ChatResponse:
    """
    Full RAG pipeline:
    1. Call retrieval service for chunks
    2. Generate answer via LLM
    3. Optionally reflect and re-query if answer is incomplete
    """
    query = request.query
    do_reflect = request.reflection_enabled and reflection_enabled
    max_retries = request.max_retries if do_reflect else 0
    retries_used = 0
    answer = ""
    chunks: list[ChunkResult] = []

    for attempt in range(max_retries + 1):
        chunks = await call_retrieval(
            client=http_client,
            retrieval_url=retrieval_url,
            project_id=request.project_id,
            query=query,
            top_k=request.top_k,
            strategy=request.strategy,
            rerank=request.rerank,
        )

        if not chunks:
            return ChatResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                chunks_used=0,
                retries_used=retries_used,
                query=request.query,
            )

        answer = await generate_answer(llm, gen_model, query, chunks)

        if not do_reflect:
            break

        try:
            reflection = await reflect(llm, reflection_model, query, chunks, answer)
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

    sources = list({c.source_id for c in chunks})

    return ChatResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(chunks),
        retries_used=retries_used,
        query=request.query,
    )
