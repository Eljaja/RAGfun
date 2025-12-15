from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from opentelemetry import trace
from qdrant_client.http import models as qm

from app.clients.embeddings import EmbeddingsClient
from app.clients.opensearch import OpenSearchClient
from app.clients.qdrant import QdrantFacade
from app.clients.rerank import RerankClient
from app.fusion import hybrid_fusion, rrf_fusion
from app.metrics import CAND, DEP_DEGRADED, ERRS, LAT
from app.models import SearchFilters, SearchHit, SearchRequest, SearchResponse, SourceObj
from app.utils import redact_uri

logger = logging.getLogger("rag.search")
tracer = trace.get_tracer("rag.search")


def _normalize_project_ids(filters: SearchFilters) -> list[str]:
    ids: list[str] = []
    if filters.project_id:
        ids.append(filters.project_id)
    if filters.project_ids:
        ids.extend([x for x in filters.project_ids if x])
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _os_filters(filters: SearchFilters | None, acl: list[str]) -> list[dict[str, Any]]:
    f: list[dict[str, Any]] = []
    if filters is None:
        filters = SearchFilters()
    if filters.source:
        f.append({"term": {"source": filters.source}})
    if filters.lang:
        f.append({"term": {"lang": filters.lang}})
    if filters.tenant_id:
        f.append({"term": {"tenant_id": filters.tenant_id}})
    proj_ids = _normalize_project_ids(filters)
    if proj_ids:
        if len(proj_ids) == 1:
            f.append({"term": {"project_id": proj_ids[0]}})
        else:
            f.append({"terms": {"project_id": proj_ids}})
    if filters.doc_ids:
        f.append({"terms": {"doc_id": filters.doc_ids}})
    if filters.tags:
        f.append({"terms": {"tags": filters.tags}})
    if acl:
        f.append({"terms": {"acl": acl}})
    return f


def _qdrant_filter(filters: SearchFilters | None, acl: list[str]) -> qm.Filter | None:
    if filters is None:
        filters = SearchFilters()
    must: list[qm.Condition] = []
    if filters.source:
        must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=filters.source)))
    if filters.lang:
        must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=filters.lang)))
    if filters.tenant_id:
        must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=filters.tenant_id)))
    proj_ids = _normalize_project_ids(filters)
    if proj_ids:
        if len(proj_ids) == 1:
            must.append(qm.FieldCondition(key="project_id", match=qm.MatchValue(value=proj_ids[0])))
        else:
            must.append(qm.FieldCondition(key="project_id", match=qm.MatchAny(any=proj_ids)))
    if filters.doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=filters.doc_ids)))
    if filters.tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=filters.tags)))
    if acl:
        must.append(qm.FieldCondition(key="acl", match=qm.MatchAny(any=acl)))
    if not must:
        return None
    return qm.Filter(must=must)


def _group_by_doc(hits: list[SearchHit], max_per_doc: int) -> list[SearchHit]:
    per_doc: dict[str, int] = defaultdict(int)
    out: list[SearchHit] = []
    for h in hits:
        if per_doc[h.doc_id] >= max_per_doc:
            continue
        out.append(h)
        per_doc[h.doc_id] += 1
    return out


def deduplicate_by_page(hits: list[SearchHit]) -> list[SearchHit]:
    """
    Deduplicate chunks by (doc_id, page), keeping only the best scoring chunk per page.
    Hits without page information are preserved as-is.
    Returns list of unique hits, one per page (plus hits without page).
    """
    seen_pages: dict[tuple[str, int], SearchHit] = {}
    hits_without_page: list[SearchHit] = []
    
    for hit in hits:
        locator = hit.metadata.get("locator") or {}
        page = locator.get("page") if isinstance(locator, dict) else None
        
        if page is not None:
            key = (hit.doc_id, page)
            if key not in seen_pages or hit.score > seen_pages[key].score:
                seen_pages[key] = hit
        else:
            # Preserve hits without page information
            hits_without_page.append(hit)
    
    # Return deduplicated hits with pages, plus hits without pages
    deduplicated = list(seen_pages.values())
    # Sort by score descending
    deduplicated.sort(key=lambda h: h.score, reverse=True)
    
    # Combine: first deduplicated by page, then hits without page
    return deduplicated + hits_without_page


async def get_parent_pages(
    hits: list[SearchHit],
    os_client: OpenSearchClient | None,
    qdrant: QdrantFacade | None,
) -> dict[tuple[str, int], str]:
    """
    Retrieve full page content for hits by collecting all chunks from the same page.
    Returns dict mapping (doc_id, page) -> full_page_text.
    """
    # Group hits by (doc_id, page)
    pages_to_fetch: set[tuple[str, int]] = set()
    
    for hit in hits:
        locator = hit.metadata.get("locator") or {}
        page = locator.get("page") if isinstance(locator, dict) else None
        
        if page is not None:
            pages_to_fetch.add((hit.doc_id, page))
    
    if not pages_to_fetch:
        return {}
    
    # Fetch all chunks for each page in parallel
    async def fetch_page_chunks(doc_id: str, page: int) -> tuple[tuple[str, int], str]:
        chunks: list[dict[str, Any]] = []
        
        # Try OpenSearch first, then Qdrant
        if os_client is not None:
            try:
                chunks = await asyncio.to_thread(os_client.get_chunks_by_page, doc_id, page)
            except Exception:
                logger.debug(f"Failed to fetch page from OpenSearch: {doc_id}/{page}", exc_info=True)
        
        if not chunks and qdrant is not None:
            try:
                chunks = await asyncio.to_thread(qdrant.get_chunks_by_page, doc_id, page)
            except Exception:
                logger.debug(f"Failed to fetch page from Qdrant: {doc_id}/{page}", exc_info=True)
        
        # Combine chunks into full page text
        if chunks:
            # Sort by chunk_index to maintain order
            chunks.sort(key=lambda c: c.get("chunk_index", 0))
            # Combine text from all chunks
            page_text = "\n\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))
            return ((doc_id, page), page_text)
        
        return ((doc_id, page), "")
    
    # Fetch all pages in parallel
    tasks = [fetch_page_chunks(doc_id, page) for doc_id, page in pages_to_fetch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Build result dict, handling exceptions
    page_map: dict[tuple[str, int], str] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.debug(f"Error fetching page: {result}", exc_info=True)
            continue
        key, text = result
        if text:  # Only add non-empty pages
            page_map[key] = text
    
    return page_map


async def search(
    *,
    req: SearchRequest,
    os_client: OpenSearchClient | None,
    qdrant: QdrantFacade | None,
    embedder: EmbeddingsClient,
    reranker: RerankClient | None,
    rerank_mode: str,
    rerank_timeout_s: float,
    rerank_max_candidates: int,
    rerank_auto_min_query_tokens: int,
    rerank_auto_min_intersection: int,
    top_k_default: int,
    bm25_top_k: int,
    vector_top_k: int,
    rrf_k: int,
    weight_bm25: float,
    weight_vector: float,
    fusion_alpha: float,
    max_chunks_per_doc: int,
    redact_uri_mode: str,
    enable_page_deduplication: bool = False,
    enable_parent_page_retrieval: bool = False,
) -> SearchResponse:
    top_k = req.top_k or top_k_default
    max_per_doc = req.max_chunks_per_doc or max_chunks_per_doc

    degraded: list[str] = []
    partial = False
    partial_rerank = False

    async def do_bm25() -> tuple[list[SearchHit], dict[str, float]]:
        if os_client is None:
            raise RuntimeError("opensearch_not_configured")
        with tracer.start_as_current_span("os_search"):
            with LAT.labels("os_search").time():
                r = await asyncio.to_thread(os_client.search, req.query, _os_filters(req.filters, req.acl), bm25_top_k)
        hits = []
        rank_ids = []
        for h in r.get("hits", {}).get("hits", []):
            src = h.get("_source", {}) or {}
            cid = src.get("chunk_id") or h.get("_id")
            rank_ids.append(cid)
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    doc_id=src.get("doc_id"),
                    score=float(h.get("_score") or 0.0),
                    source_scores={"bm25": float(h.get("_score") or 0.0)},
                    text=src.get("text"),
                    highlight=h.get("highlight"),
                    metadata={k: v for k, v in src.items() if k not in ("text",)},
                )
            )
        CAND.labels("bm25").observe(len(hits))
        return hits, {cid: float(i + 1) for i, cid in enumerate(rank_ids)}

    async def do_vector() -> tuple[list[SearchHit], dict[str, float]]:
        if qdrant is None:
            raise RuntimeError("qdrant_not_configured")
        with tracer.start_as_current_span("embed"):
            with LAT.labels("embed").time():
                vec = (await embedder.embed([req.query]))[0]
        with tracer.start_as_current_span("qdrant_search"):
            with LAT.labels("qdrant_search").time():
                pts = await asyncio.to_thread(qdrant.search, vec, _qdrant_filter(req.filters, req.acl), vector_top_k)
        hits: list[SearchHit] = []
        rank_ids: list[str] = []
        for p in pts:
            payload = p.payload or {}
            cid = str(payload.get("chunk_id") or p.id)
            rank_ids.append(cid)
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    doc_id=str(payload.get("doc_id")),
                    score=float(p.score or 0.0),
                    source_scores={"vector": float(p.score or 0.0)},
                    text=payload.get("text"),
                    highlight=None,
                    metadata={k: v for k, v in payload.items() if k not in ("text",)},
                )
            )
        CAND.labels("vector").observe(len(hits))
        return hits, {cid: float(i + 1) for i, cid in enumerate(rank_ids)}

    bm25_hits: list[SearchHit] = []
    vec_hits: list[SearchHit] = []
    bm25_rank: dict[str, float] = {}
    vec_rank: dict[str, float] = {}

    if req.mode == "bm25":
        try:
            bm25_hits, bm25_rank = await do_bm25()
        except Exception as e:
            ERRS.labels("os_search", type(e).__name__).inc()
            partial = True
            degraded.append("bm25_unavailable")
    elif req.mode == "vector":
        try:
            vec_hits, vec_rank = await do_vector()
        except Exception as e:
            ERRS.labels("qdrant_search", type(e).__name__).inc()
            partial = True
            degraded.append("vector_unavailable")
    else:
        # hybrid: parallel, degrade gracefully
        tasks = []
        tasks.append(asyncio.create_task(do_bm25()))
        tasks.append(asyncio.create_task(do_vector()))
        res = await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(res[0], Exception):
            partial = True
            degraded.append("bm25_unavailable")
            DEP_DEGRADED.labels("bm25").inc()
        else:
            bm25_hits, bm25_rank = res[0]
        if isinstance(res[1], Exception):
            partial = True
            degraded.append("vector_unavailable")
            DEP_DEGRADED.labels("vector").inc()
        else:
            vec_hits, vec_rank = res[1]

    # Build fused ranks
    src_lists: dict[str, list[str]] = {}
    if bm25_hits:
        src_lists["bm25"] = [h.chunk_id for h in bm25_hits]
    if vec_hits:
        src_lists["vector"] = [h.chunk_id for h in vec_hits]

    if req.mode == "bm25":
        fused_order = [h.chunk_id for h in bm25_hits][:top_k]
        fused_scores = {h.chunk_id: h.score for h in bm25_hits}
    elif req.mode == "vector":
        fused_order = [h.chunk_id for h in vec_hits][:top_k]
        fused_scores = {h.chunk_id: h.score for h in vec_hits}
    else:
        with tracer.start_as_current_span("fusion"):
            with LAT.labels("fusion").time():
                # Hybrid fusion: combine rank-based RRF with normalized per-source scores.
                fused_scores = hybrid_fusion(
                    ranked_lists=src_lists,
                    scored_lists={
                        "bm25": {h.chunk_id: float(h.score) for h in bm25_hits},
                        "vector": {h.chunk_id: float(h.score) for h in vec_hits},
                    },
                    weights={"bm25": weight_bm25, "vector": weight_vector},
                    rrf_k=rrf_k,
                    alpha=fusion_alpha,
                )
        fused_order = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    # Merge payloads: prefer OS for highlight, fallback to Qdrant
    by_id: dict[str, SearchHit] = {}
    for h in bm25_hits:
        by_id[h.chunk_id] = h
    for h in vec_hits:
        if h.chunk_id in by_id:
            by_id[h.chunk_id].source_scores.update(h.source_scores)
            # keep existing text/highlight from OS
        else:
            by_id[h.chunk_id] = h

    out: list[SearchHit] = []
    for cid in fused_order:
        h = by_id.get(cid)
        if not h:
            continue
        h.source_scores["fusion"] = float(fused_scores.get(cid, h.score))
        h.score = float(fused_scores.get(cid, h.score))
        out.append(h)

    # Optional rerank
    def want_rerank() -> bool:
        if req.rerank is not None:
            return bool(req.rerank)
        if rerank_mode == "disabled":
            return False
        if rerank_mode == "always":
            return True
        if rerank_mode == "auto":
            qtoks = len(req.query.split())
            if qtoks < rerank_auto_min_query_tokens:
                return False
            if bm25_hits and vec_hits:
                inter = len({h.chunk_id for h in bm25_hits}.intersection({h.chunk_id for h in vec_hits}))
                return inter <= rerank_auto_min_intersection
            # if only one backend returned results, rerank usually doesn't help
            return False
        return False

    if want_rerank() and reranker is not None and out:
        with tracer.start_as_current_span("rerank"):
            with LAT.labels("rerank").time():
                try:
                    candidates = []
                    for h in out[:rerank_max_candidates]:
                        if not h.text:
                            continue
                        candidates.append({"id": h.chunk_id, "text": h.text})
                    CAND.labels("rerank").observe(len(candidates))
                    # Use reranker client's timeout; also bound with asyncio timeout for safety.
                    scores = await asyncio.wait_for(reranker.rerank(req.query, candidates), timeout=rerank_timeout_s + 0.5)
                    # Apply scores
                    for h in out:
                        if h.chunk_id in scores:
                            h.rerank_score = float(scores[h.chunk_id])
                    # Sort by rerank_score where available (fallback to fusion)
                    out.sort(key=lambda h: (h.rerank_score is not None, h.rerank_score or 0.0, h.score), reverse=True)
                    # Final score becomes rerank_score for those reranked
                    for h in out:
                        if h.rerank_score is not None:
                            h.score = h.rerank_score
                except Exception as e:
                    ERRS.labels("rerank", type(e).__name__).inc()
                    partial_rerank = True
    elif want_rerank() and reranker is None:
        # configured request wants rerank but no client available
        partial_rerank = True

    # Page deduplication: remove duplicate chunks from the same page
    if enable_page_deduplication:
        with tracer.start_as_current_span("page_deduplication"):
            with LAT.labels("page_deduplication").time():
                out_before = len(out)
                out = deduplicate_by_page(out)
                logger.debug(f"Page deduplication: {out_before} -> {len(out)} hits")

    # Parent page retrieval: replace chunks with full page content
    if enable_parent_page_retrieval:
        with tracer.start_as_current_span("parent_page_retrieval"):
            with LAT.labels("parent_page_retrieval").time():
                try:
                    page_map = await get_parent_pages(out, os_client, qdrant)
                    # Replace chunk text with full page text
                    for hit in out:
                        locator = hit.metadata.get("locator") or {}
                        page = locator.get("page") if isinstance(locator, dict) else None
                        
                        if page is not None:
                            page_key = (hit.doc_id, page)
                            if page_key in page_map and page_map[page_key]:
                                # Replace chunk text with full page text
                                hit.text = page_map[page_key]
                                logger.debug(f"Replaced chunk text with full page for {hit.doc_id}:{page}")
                except Exception as e:
                    logger.warning(f"Failed to retrieve parent pages: {e}", exc_info=True)
                    # Don't fail the request, just log the error

    # grouping per doc if requested
    if req.group_by_doc:
        out = _group_by_doc(out, max_per_doc)

    with tracer.start_as_current_span("assemble_response"):
        # sources
        sources_out: list[SourceObj] | None = None
        if req.include_sources:
            sources_out = []
            for h in out:
                md = h.metadata or {}
                sources_out.append(
                    SourceObj(
                        doc_id=h.doc_id,
                        title=md.get("title"),
                        uri=redact_uri(md.get("uri"), redact_uri_mode),
                        locator=md.get("locator"),
                    )
                )
                h.source = sources_out[-1]

        return SearchResponse(
            ok=True,
            mode=req.mode,
            partial=partial,
            partial_rerank=partial_rerank,
            degraded=degraded,
            hits=out,
            sources=sources_out,
        )


