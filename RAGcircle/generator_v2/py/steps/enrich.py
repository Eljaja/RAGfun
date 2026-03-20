"""Enrich phase: chunks in, better chunks + retrieval requests out.

Pure function: (steps, chunks, query, ...) -> (chunks, requests, traces).
Steps that need more chunks emit EnrichRetrievalRequests for the pipeline to fetch.
"""

from __future__ import annotations

import logging
from typing import Any

from context import stitch_segments
from models.chunks import ChunkResult, ScoreSource
from models.plan import EnrichRetrievalRequest
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

logger = logging.getLogger(__name__)


async def enrich(
    steps: list[PostRetrieveStep],
    *,
    chunks: list[ChunkResult],
    query: str,
    is_factoid: bool,
    retrieval_plan: ExecutionPlan | None,
) -> tuple[list[ChunkResult], list[EnrichRetrievalRequest], list[dict[str, Any]]]:
    """Run all post-retrieve steps.

    Returns (chunks, retrieval_requests, traces).
    Steps that need more data emit EnrichRetrievalRequests instead of fetching.
    """
    traces: list[dict[str, Any]] = []
    requests: list[EnrichRetrievalRequest] = []

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
                    hints = extract_hint_terms(chunks, max_terms=3)
                    if hints:
                        follow_up = f"{query} {' '.join(hints)}"
                        plan = retrieval_plan or from_preset("hybrid", top_k=10, rerank=True)
                        requests.append(EnrichRetrievalRequest(query=follow_up, plan=plan))
                        traces.append({
                            "kind": "action", "label": "TwoPass",
                            "content": f"Only {n_unique} unique sources, follow-up query queued",
                        })

            case BM25AnchorStep(top_k=anchor_k):
                kw = keyword_query(query)
                if kw:
                    bm25_plan = from_preset("fast", top_k=anchor_k, rerank=False)
                    requests.append(EnrichRetrievalRequest(query=kw, plan=bm25_plan))
                    traces.append({"kind": "action", "label": "BM25Anchor", "content": f"kw={kw[:60]}"})

            case FactoidExpandStep():
                if is_factoid and chunks:
                    expand_plan = from_preset("fast", top_k=5, rerank=False)
                    requests.append(EnrichRetrievalRequest(query=query, plan=expand_plan))
                    traces.append({
                        "kind": "action", "label": "FactoidExpand",
                        "content": "Expanding within top sources",
                    })

            case StitchStep(max_per_segment=mps):
                before = len(chunks)
                chunks = stitch_segments(chunks, max_per_segment=mps)
                after = len(chunks)
                if before != after:
                    traces.append({
                        "kind": "action", "label": "Stitch",
                        "content": f"Stitched {before} chunks into {after} segments",
                    })

    return chunks, requests, traces
