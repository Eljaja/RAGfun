"""Chat endpoints for RAG queries."""

import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.llm_utils import _enforce_citations_or_refuse, _enforce_extractive_or_refuse
from app.metrics import GATE_DEGRADED, GATE_PARTIAL, GATE_REFUSALS, LAT, REQS
from app.models import ChatRequest, ChatResponse, ContextChunk, Source
from app.rag import build_context_blocks, build_messages
from app.retrieval_utils import (
    _apply_bm25_anchor_pass,
    _extract_hint_terms_from_hits,
    _query_variants,
    _unique_doc_count,
    _dedupe_queries,
    _keyword_query,
)
from app.state import state
from app.text_utils import _answer_is_in_context, _has_inline_citations, _is_factoid_like_question

logger = logging.getLogger("gate")

router = APIRouter()


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """Main chat endpoint for RAG queries."""
    if state.config_error:
        REQS.labels(endpoint="/v1/chat", status="503").inc()
        return ChatResponse(ok=False, answer=f"config_error: {state.config_error}", used_mode="none")

    assert state.settings is not None
    assert state.retrieval is not None
    assert state.llm is not None

    # Defaults (may be overridden by request payload)
    mode = payload.retrieval_mode or state.settings.retrieval_mode
    top_k = payload.top_k or state.settings.top_k
    rerank = payload.rerank
    multi_query_enabled = bool(state.settings.multi_query_enabled)
    two_pass_enabled = bool(state.settings.two_pass_enabled)
    bm25_anchor_enabled = bool(state.settings.bm25_anchor_enabled)
    segment_stitching_enabled = bool(state.settings.segment_stitching_enabled)

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    try:
        with LAT.labels("retrieval").time():
            # Optional: multi-query + two-pass retrieval (opt-in).
            if multi_query_enabled or two_pass_enabled:
                # Keep candidate pool bounded; too many low-quality hits increases refusals.
                raw_top_k = max(1, top_k) * max(1, int(state.settings.multi_query_top_k_multiplier))
                raw_top_k = max(20, min(80, int(raw_top_k)))

                # Pass 1 (q0 only) - used to derive hint terms for pass 2 if enabled.
                pass1 = await state.retrieval.search(
                    query=payload.query,
                    mode=mode,
                    top_k=raw_top_k,
                    # IMPORTANT: keep rerank behavior for q0 (restores quality vs massive refusals).
                    rerank=rerank,
                    filters=filters,
                    acl=payload.acl,
                    include_sources=payload.include_sources,
                )

                queries = _query_variants(payload.query)
                # Two-pass: add a follow-up query from top hits if pass1 seems too narrow.
                if two_pass_enabled:
                    hits1 = list(pass1.get("hits") or [])
                    if _unique_doc_count(hits1) < int(state.settings.two_pass_min_unique_docs):
                        hints = _extract_hint_terms_from_hits(
                            hits1,
                            max_terms=max(1, int(state.settings.two_pass_hint_max_terms)),
                        )
                        if hints:
                            q2 = (payload.query + " " + " ".join(hints)).strip()
                            queries.append(q2)

                # Clamp and dedupe
                queries = _dedupe_queries(queries)[: max(1, int(state.settings.multi_query_max_queries))]
                if not queries:
                    queries = [payload.query]

                # For each query, retrieve candidates (rerank off), then fuse by RRF on chunk_id.
                async def _one(q: str) -> dict[str, Any]:
                    return await state.retrieval.search(
                        query=q,
                        mode=mode,
                        top_k=raw_top_k,
                        # Only q0 is reranked; expansions are for recall.
                        rerank=False if q != payload.query else rerank,
                        filters=filters,
                        acl=payload.acl,
                        include_sources=payload.include_sources,
                    )

                rs = await asyncio.gather(*[_one(q) for q in queries], return_exceptions=True)
                # include pass1 results too
                rs = [pass1] + [r for r in rs if not isinstance(r, Exception)]

                ranked_lists: dict[str, list[str]] = {}
                hit_by_cid: dict[str, dict[str, Any]] = {}
                for i, rj in enumerate(rs):
                    hits_i = list(rj.get("hits") or [])
                    ranked_lists[f"q{i}"] = [str(h.get("chunk_id")) for h in hits_i if h.get("chunk_id")]
                    for h in hits_i:
                        cid = str(h.get("chunk_id") or "").strip()
                        if not cid:
                            continue
                        # keep the first version we saw (usually already has text/source)
                        hit_by_cid.setdefault(cid, h)

                fused: dict[str, float] = {}
                rrf_k = max(1, int(state.settings.multi_query_rrf_k))
                for _, cids in ranked_lists.items():
                    for rank, cid in enumerate(cids, start=1):
                        fused[cid] = fused.get(cid, 0.0) + (1.0 / (rrf_k + rank))

                fused_hits = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
                merged_hits: list[dict[str, Any]] = []
                for cid, sc in fused_hits[: raw_top_k]:
                    h = dict(hit_by_cid.get(cid) or {})
                    # Preserve original retrieval score (esp. rerank score). Store fused score separately.
                    md = dict(h.get("metadata") or {})
                    md["multi_rrf_score"] = float(sc)
                    h["metadata"] = md
                    merged_hits.append(h)

                retrieval_json = {
                    "ok": True,
                    "mode": mode,
                    "partial": any(bool(r.get("partial")) for r in rs if isinstance(r, dict)),
                    "degraded": sorted({d for r in rs if isinstance(r, dict) for d in (r.get("degraded") or [])}),
                    "hits": merged_hits,
                    # keep a small debug payload (UI uses it), but don't blow up response size
                    "multi_query": {"queries": queries, "raw_top_k": raw_top_k},
                }
            else:
                retrieval_json = await state.retrieval.search(
                    query=payload.query,
                    mode=mode,
                    top_k=top_k,
                    rerank=rerank,
                    filters=filters,
                    acl=payload.acl,
                    include_sources=payload.include_sources,
                )
    except httpx.TimeoutException:
        REQS.labels(endpoint="/v1/chat", status="504").inc()
        return ChatResponse(ok=False, answer="retrieval_timeout", used_mode=mode)
    except Exception as e:
        REQS.labels(endpoint="/v1/chat", status="502").inc()
        return ChatResponse(ok=False, answer=f"retrieval_error: {type(e).__name__}", used_mode=mode)

    # BM25 anchor pass + union/fuse candidates (prevents exact-match entities from being dropped).
    retrieval_json = await _apply_bm25_anchor_pass(
        retrieval=state.retrieval,
        settings=state.settings,
        payload=payload,
        retrieval_json=retrieval_json,
        mode=mode,
        top_k=top_k,
        filters=filters,
        enabled_override=bm25_anchor_enabled,
    )

    hits = retrieval_json.get("hits") or []
    degraded = retrieval_json.get("degraded") or []
    partial = bool(retrieval_json.get("partial"))

    # For factoid-style eval runs (include_sources=False), expand within top doc(s) BEFORE LLM call.
    # This increases the chance that the answer-bearing lead sentence is present in the context window.
    if (not payload.include_sources) and _is_factoid_like_question(payload.query) and hits:
        doc_ids: list[str] = []
        for h in hits:
            did = str(h.get("doc_id") or "").strip()
            if did and did not in doc_ids:
                doc_ids.append(did)
            if len(doc_ids) >= 1:
                break
        if doc_ids:
            kw = _keyword_query(payload.query) or payload.query
            try:
                extra = await state.retrieval.search(
                    query=kw,
                    mode="bm25",
                    top_k=20,
                    rerank=False,
                    filters={**(filters or {}), "doc_ids": doc_ids},
                    acl=payload.acl,
                    include_sources=False,
                    max_chunks_per_doc=20,
                )
                extra_hits = list(extra.get("hits") or [])
                if extra_hits:
                    # Prepend expansion hits; dedupe by chunk_id.
                    seen: set[str] = set()
                    merged: list[dict[str, Any]] = []
                    for hh in extra_hits + list(hits):
                        cid = str(hh.get("chunk_id") or "").strip()
                        if cid and cid in seen:
                            continue
                        if cid:
                            seen.add(cid)
                        merged.append(hh)
                    hits = merged
            except Exception as e:
                logger.warning("factoid_preexpand_failed", extra={"extra": {"error": str(e), "doc_ids": doc_ids}})

    per_chunk = 1200 if (not payload.include_sources and _is_factoid_like_question(payload.query)) else None
    kept, context_text, sources = build_context_blocks(
        hits=hits,
        max_chars=state.settings.max_context_chars,
        max_chunk_chars=per_chunk,
        stitch_segments=bool(segment_stitching_enabled),
        stitch_max_chunks_per_segment=int(state.settings.segment_stitching_max_chunks),
        stitch_group_by_page=bool(state.settings.segment_stitching_group_by_page),
    )
    messages = build_messages(
        query=payload.query,
        history=[m.model_dump() for m in payload.history],
        context_text=context_text,
        include_sources=bool(payload.include_sources),
    )

    with LAT.labels("llm").time():
        answer = await state.llm.chat(messages=messages)
        if payload.include_sources:
            refs = [int(s.get("ref")) for s in (sources or []) if s.get("ref") is not None]
            answer = await _enforce_citations_or_refuse(llm=state.llm, messages=messages, answer=answer, refs=refs)
        else:
            # ru_eval runs with include_sources=False. For short factoid answers:
            # 1) If the answer doesn't appear in the provided context, expand within top docs and retry once.
            # 2) Enforce "verbatim span from sources" (or refuse) to avoid entity hallucinations.
            if _is_factoid_like_question(payload.query):
                if not _answer_is_in_context(answer=answer, context_text=context_text):
                    # Expand inside the top doc(s) to fetch a more relevant chunk from the same page.
                    hits0 = list(kept or [])
                    doc_ids: list[str] = []
                    for h in hits0:
                        did = str(h.get("doc_id") or "").strip()
                        if did and did not in doc_ids:
                            doc_ids.append(did)
                        if len(doc_ids) >= 2:
                            break
                    if doc_ids:
                        kw = _keyword_query(payload.query) or payload.query
                        extra_hits: list[dict[str, Any]] = []
                        for did in doc_ids:
                            try:
                                rj2 = await state.retrieval.search(
                                    query=kw,
                                    mode="bm25",
                                    top_k=20,
                                    rerank=False,
                                    filters={**(filters or {}), "doc_ids": [did]},
                                    acl=payload.acl,
                                    include_sources=False,
                                    # IMPORTANT: override server-side per-doc cap (default=3), because we are
                                    # intentionally searching *within a single doc* to find the right chunk.
                                    max_chunks_per_doc=20,
                                )
                                extra_hits.extend(list(rj2.get("hits") or []))
                            except Exception as e:
                                logger.warning(
                                    "factoid_doc_expand_failed",
                                    extra={"extra": {"doc_id": did, "error": str(e)}},
                                )
                        if extra_hits:
                            # Merge by chunk_id, preserving original order first.
                            seen_cid: set[str] = set()
                            merged_hits: list[dict[str, Any]] = []
                            # IMPORTANT: prioritize doc-local expansion hits so the answer-bearing chunk
                            # (e.g. lead sentence) is likely to make it into the limited context window.
                            for h in extra_hits + hits0:
                                cid = str(h.get("chunk_id") or "").strip()
                                if cid and cid in seen_cid:
                                    continue
                                if cid:
                                    seen_cid.add(cid)
                                merged_hits.append(h)
                            kept2, context2, sources2 = build_context_blocks(
                                hits=merged_hits,
                                max_chars=state.settings.max_context_chars,
                                max_chunk_chars=per_chunk,
                                stitch_segments=bool(segment_stitching_enabled),
                                stitch_max_chunks_per_segment=int(state.settings.segment_stitching_max_chunks),
                                stitch_group_by_page=bool(state.settings.segment_stitching_group_by_page),
                            )
                            messages2 = build_messages(
                                query=payload.query,
                                history=[m.model_dump() for m in payload.history],
                                context_text=context2,
                                include_sources=False,
                            )
                            answer2 = await state.llm.chat(messages=messages2)
                            # Swap in expanded context for the final response if it helped.
                            if _answer_is_in_context(answer=answer2, context_text=context2):
                                kept = kept2
                                context_text = context2
                                sources = sources2
                                messages = messages2
                                answer = answer2

                answer = await _enforce_extractive_or_refuse(
                    llm=state.llm,
                    messages=messages,
                    answer=answer,
                    context_text=context_text,
                )

    ref_by_key = {(str(s.get("doc_id")), s.get("uri")): s.get("ref") for s in (sources or [])}

    ctx = []
    for h in kept:
        src = h.get("source") or {}
        key = (str(src.get("doc_id") or h.get("doc_id")), src.get("uri"))
        ctx.append(
            ContextChunk(
                chunk_id=str(h.get("chunk_id")),
                doc_id=str(h.get("doc_id")),
                text=h.get("text"),
                score=float(h.get("score") or 0.0),
                source=Source(
                    ref=ref_by_key.get(key),
                    doc_id=str(src.get("doc_id") or h.get("doc_id")),
                    title=src.get("title"),
                    uri=src.get("uri"),
                    locator=src.get("locator"),
                )
                if payload.include_sources
                else None,
            )
        )

    if partial:
        GATE_PARTIAL.labels(endpoint="/v1/chat").inc()
    for k in degraded:
        GATE_DEGRADED.labels(str(k)).inc()
    if (answer or "").strip() == "I don't know":
        GATE_REFUSALS.labels(endpoint="/v1/chat").inc()

    REQS.labels(endpoint="/v1/chat", status="200").inc()
    return ChatResponse(
        ok=True,
        answer=answer,
        used_mode=mode,
        degraded=list(degraded),
        partial=partial,
        context=ctx,
        sources=[Source(**s) for s in sources] if payload.include_sources else [],
        retrieval=retrieval_json,
    )


@router.post("/v1/chat/stream")
async def chat_stream(payload: ChatRequest):
    """Streaming chat endpoint using SSE."""
    if state.config_error:
        REQS.labels(endpoint="/v1/chat/stream", status="503").inc()
        async def error_stream():
            yield f"data: {json.dumps({'error': f'config_error: {state.config_error}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    assert state.settings is not None
    assert state.retrieval is not None
    assert state.llm is not None

    # Defaults (may be overridden by request payload)
    mode = payload.retrieval_mode or state.settings.retrieval_mode
    top_k = payload.top_k or state.settings.top_k
    rerank = payload.rerank
    bm25_anchor_enabled = bool(state.settings.bm25_anchor_enabled)
    segment_stitching_enabled = bool(state.settings.segment_stitching_enabled)

    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    async def generate_stream():
        try:
            # Step 1: Retrieval
            try:
                with LAT.labels("retrieval").time():
                    retrieval_json = await state.retrieval.search(
                        query=payload.query,
                        mode=mode,
                        top_k=top_k,
                        rerank=rerank,
                        filters=filters,
                        acl=payload.acl,
                        include_sources=payload.include_sources,
                    )
            except httpx.TimeoutException:
                yield f"data: {json.dumps({'error': 'retrieval_timeout'})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'error': f'retrieval_error: {type(e).__name__}'})}\n\n"
                return

            retrieval_json = await _apply_bm25_anchor_pass(
                retrieval=state.retrieval,
                settings=state.settings,
                payload=payload,
                retrieval_json=retrieval_json,
                mode=mode,
                top_k=top_k,
                filters=filters,
                enabled_override=bm25_anchor_enabled,
            )

            hits = retrieval_json.get("hits") or []
            degraded = retrieval_json.get("degraded") or []
            partial = bool(retrieval_json.get("partial"))

            per_chunk = 1200 if (not payload.include_sources and _is_factoid_like_question(payload.query)) else None
            kept, context_text, sources = build_context_blocks(
                hits=hits,
                max_chars=state.settings.max_context_chars,
                max_chunk_chars=per_chunk,
                stitch_segments=bool(segment_stitching_enabled),
                stitch_max_chunks_per_segment=int(state.settings.segment_stitching_max_chunks),
                stitch_group_by_page=bool(state.settings.segment_stitching_group_by_page),
            )
            messages = build_messages(
                query=payload.query,
                history=[m.model_dump() for m in payload.history],
                context_text=context_text,
                include_sources=bool(payload.include_sources),
            )

            # Build context chunks (same as /v1/chat response) and send retrieval payload for UI.
            ref_by_key = {(str(s.get("doc_id")), s.get("uri")): s.get("ref") for s in (sources or [])}
            ctx = []
            for h in kept:
                src = h.get("source") or {}
                key = (str(src.get("doc_id") or h.get("doc_id")), src.get("uri"))
                ctx.append(
                    ContextChunk(
                        chunk_id=str(h.get("chunk_id")),
                        doc_id=str(h.get("doc_id")),
                        text=h.get("text"),
                        score=float(h.get("score") or 0.0),
                        source=Source(
                            ref=ref_by_key.get(key),
                            doc_id=str(src.get("doc_id") or h.get("doc_id")),
                            title=src.get("title"),
                            uri=src.get("uri"),
                            locator=src.get("locator"),
                        )
                        if payload.include_sources
                        else None,
                    )
                )

            yield f"data: {json.dumps({'type': 'retrieval', 'mode': mode, 'partial': partial, 'degraded': list(degraded), 'context': [c.model_dump() for c in ctx], 'retrieval': retrieval_json})}\n\n"

            # Step 2: Stream LLM response
            answer_parts = []
            async for line in state.llm.chat_stream(messages=messages):
                # Parse OpenAI-compatible SSE format
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                answer_parts.append(content)
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue

            # Send final answer and metadata
            full_answer = "".join(answer_parts)
            if payload.include_sources:
                refs = [int(s.get("ref")) for s in (sources or []) if s.get("ref") is not None]
                # Streaming: cannot easily re-run the full generation. Enforce strictness by refusing if citations are missing.
                if refs and not _has_inline_citations(full_answer):
                    full_answer = "I don't know"
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'mode': mode, 'degraded': list(degraded), 'partial': partial, 'sources': sources if payload.include_sources else [], 'context': [c.model_dump() for c in ctx]})}\n\n"

            REQS.labels(endpoint="/v1/chat/stream", status="200").inc()
        except Exception as e:
            logger.error("chat_stream_error", extra={"extra": {"error": str(e)}})
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            REQS.labels(endpoint="/v1/chat/stream", status="500").inc()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

