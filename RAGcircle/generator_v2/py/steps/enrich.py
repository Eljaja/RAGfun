"""Enrich phase: chunks in, better chunks out.

Pure function: (steps, chunks, query, ...) -> (chunks, traces).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import Settings
from context import merge_chunks, stitch_segments
from models.chunks import ChunkResult, ScoreSource
from models.retrieval import ExecutionPlan
from models.steps import (
    BM25AnchorStep,
    FactoidExpandStep,
    PostRetrieveStep,
    QualityCheckStep,
    StitchStep,
    TwoPassStep,
)
from plan_builder import from_preset
from query_variants import extract_hint_terms, keyword_query, unique_source_count
from steps.retrieve import _safe_retrieve

logger = logging.getLogger(__name__)


async def enrich(
    steps: list[PostRetrieveStep],
    *,
    chunks: list[ChunkResult],
    query: str,
    project_id: str,
    is_factoid: bool,
    retrieval_plan: ExecutionPlan | None,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> tuple[list[ChunkResult], list[dict[str, Any]]]:
    """Run all post-retrieve steps. Returns (enriched_chunks, traces)."""
    traces: list[dict[str, Any]] = []

    for step in steps:
        match step:
            case QualityCheckStep(min_hits=min_hits, min_score=min_score):
                poor = (
                    not chunks
                    or len(chunks) < min_hits
                    or (chunks[0].score_source == ScoreSource.RERANK and chunks[0].score < min_score)
                )
                if poor:
                    traces.append({"kind": "thought", "label": "Quality", "content": "Retrieval quality is poor"})

            case TwoPassStep(min_unique_sources=min_src):
                n_unique = unique_source_count(chunks)
                if n_unique < min_src:
                    traces.append({
                        "kind": "action", "label": "TwoPass",
                        "content": f"Only {n_unique} unique sources, generating follow-up query",
                    })
                    hints = extract_hint_terms(chunks, max_terms=3)
                    if hints:
                        follow_up = f"{query} {' '.join(hints)}"
                        plan = retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
                        extra = await _safe_retrieve(
                            follow_up, plan,
                            project_id=project_id, http_client=http_client, settings=settings,
                        )
                        if extra:
                            chunks = merge_chunks([chunks, extra])

            case BM25AnchorStep(top_k=anchor_k):
                kw = keyword_query(query)
                if kw:
                    traces.append({"kind": "action", "label": "BM25Anchor", "content": f"kw={kw[:60]}"})
                    bm25_plan = from_preset("fast", top_k=anchor_k, rerank=False)
                    bm25_hits = await _safe_retrieve(
                        kw, bm25_plan,
                        project_id=project_id, http_client=http_client, settings=settings,
                    )
                    if bm25_hits:
                        chunks = merge_chunks([chunks, bm25_hits])

            case FactoidExpandStep():
                if is_factoid and chunks:
                    traces.append({
                        "kind": "action", "label": "FactoidExpand",
                        "content": "Expanding within top sources",
                    })
                    expand_plan = from_preset("fast", top_k=5, rerank=False)
                    extra = await _safe_retrieve(
                        query, expand_plan,
                        project_id=project_id, http_client=http_client, settings=settings,
                    )
                    if extra:
                        chunks = merge_chunks([chunks, extra])

            case StitchStep(max_per_segment=mps):
                before = len(chunks)
                chunks = stitch_segments(chunks, max_per_segment=mps)
                after = len(chunks)
                if before != after:
                    traces.append({
                        "kind": "action", "label": "Stitch",
                        "content": f"Stitched {before} chunks into {after} segments",
                    })

    return chunks, traces
