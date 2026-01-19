"""Retrieval-related utilities: query expansion, RRF merging, BM25 anchor pass."""

import logging
from typing import Any

from rapidfuzz import fuzz

from app.clients import RetrievalClient
from app.config import Settings
from app.models import ChatRequest
from app.text_utils import _keyword_query, _norm_query, _QUOTED_RE, _YEAR_RE, _TOKEN_RE

logger = logging.getLogger("gate")


def _dedupe_queries(qs: list[str], *, threshold: int = 92) -> list[str]:
    """Deduplicate queries using fuzzy matching."""
    out: list[str] = []
    norms: list[str] = []
    for q in qs:
        q = (q or "").strip()
        if not q:
            continue
        nq = _norm_query(q)
        if not nq:
            continue
        dup = False
        for prev in norms:
            # token_set_ratio is robust to word reordering
            if fuzz.token_set_ratio(nq, prev) >= threshold:
                dup = True
                break
        if dup:
            continue
        out.append(q)
        norms.append(nq)
    return out


def _query_variants(q: str) -> list[str]:
    """Generate query variants for multi-query retrieval."""
    q = (q or "").strip()
    if not q:
        return []
    out: list[str] = [q]
    kw = _keyword_query(q)
    if kw and kw != q:
        out.append(kw)
    # quoted phrases and years often anchor multi-hop
    phrases: list[str] = []
    for m in _QUOTED_RE.finditer(q):
        p = (m.group(1) or m.group(2) or "").strip()
        if p and len(p) >= 3:
            phrases.append(p)
    years = _YEAR_RE.findall(q)
    if phrases:
        out.append(" ".join(phrases))
    if years:
        out.append(" ".join(sorted(set(years))))
    return _dedupe_queries(out)


def _rrf_merge_hits_by_chunk_id(
    *,
    base_hits: list[dict[str, Any]],
    anchor_hits: list[dict[str, Any]],
    rrf_k: int,
    cap: int,
) -> list[dict[str, Any]]:
    """
    Merge two ranked hit lists by Reciprocal Rank Fusion (RRF), keyed by chunk_id.
    Keeps the first-seen hit payload for each chunk_id, but attaches RRF diagnostics in metadata.
    """
    rrf_k = max(1, int(rrf_k))
    cap = max(1, int(cap))

    def _ranked_ids(hits: list[dict[str, Any]]) -> list[str]:
        return [str(h.get("chunk_id")) for h in hits if h.get("chunk_id")]

    base_ids = _ranked_ids(base_hits)
    anch_ids = _ranked_ids(anchor_hits)

    hit_by_cid: dict[str, dict[str, Any]] = {}
    for h in base_hits:
        cid = str(h.get("chunk_id") or "").strip()
        if cid:
            hit_by_cid.setdefault(cid, h)
    for h in anchor_hits:
        cid = str(h.get("chunk_id") or "").strip()
        if cid:
            hit_by_cid.setdefault(cid, h)

    fused: dict[str, float] = {}
    for rank, cid in enumerate(base_ids, start=1):
        fused[cid] = fused.get(cid, 0.0) + (1.0 / (rrf_k + rank))
    for rank, cid in enumerate(anch_ids, start=1):
        fused[cid] = fused.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    merged: list[dict[str, Any]] = []
    for cid, sc in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:cap]:
        h = dict(hit_by_cid.get(cid) or {})
        md = dict(h.get("metadata") or {})
        md["bm25_anchor_rrf_score"] = float(sc)
        if cid in set(anch_ids) and cid not in set(base_ids):
            md["bm25_anchor_only"] = True
        elif cid in set(anch_ids):
            md["bm25_anchor_present"] = True
        h["metadata"] = md
        merged.append(h)
    return merged


async def _apply_bm25_anchor_pass(
    *,
    retrieval: RetrievalClient,
    settings: Settings,
    payload: ChatRequest,
    retrieval_json: dict[str, Any],
    mode: str,
    top_k: int,
    filters: dict[str, Any] | None,
    enabled_override: bool | None = None,
) -> dict[str, Any]:
    """
    Run a small BM25 lookup on a keyword-only query and fuse candidates into retrieval_json["hits"].
    This prevents exact-match entity chunks from being dropped by hybrid/rerank pipelines.
    """
    enabled = getattr(settings, "bm25_anchor_enabled", False) if enabled_override is None else bool(enabled_override)
    if not enabled:
        return retrieval_json

    base_hits = list(retrieval_json.get("hits") or [])
    anchor_q = _keyword_query(payload.query) or (payload.query or "").strip()
    if not anchor_q:
        return retrieval_json

    try:
        anchor_top_k = max(1, int(getattr(settings, "bm25_anchor_top_k", 30)))
        anchor_json = await retrieval.search(
            query=anchor_q,
            mode="bm25",
            top_k=anchor_top_k,
            rerank=False,
            filters=filters,
            acl=payload.acl,
            include_sources=payload.include_sources,
        )
    except Exception as e:
        logger.warning(
            "bm25_anchor_pass_failed",
            extra={"extra": {"error": str(e), "query": anchor_q, "mode": mode}},
        )
        return retrieval_json

    anchor_hits = list(anchor_json.get("hits") or [])
    if not anchor_hits:
        return retrieval_json

    # Fuse by RRF and cap to a reasonable candidate pool.
    cap = max(len(base_hits), max(1, int(top_k)), max(1, int(getattr(settings, "bm25_anchor_top_k", 30))))
    cap = max(20, min(80, int(cap)))
    rrf_k = max(1, int(getattr(settings, "bm25_anchor_rrf_k", 60)))
    merged_hits = _rrf_merge_hits_by_chunk_id(base_hits=base_hits, anchor_hits=anchor_hits, rrf_k=rrf_k, cap=cap)

    out = dict(retrieval_json)
    out["hits"] = merged_hits
    out["bm25_anchor"] = {"query": anchor_q, "top_k": int(getattr(settings, "bm25_anchor_top_k", 30)), "rrf_k": rrf_k}
    # propagate partial/degraded from anchor lookup
    out["partial"] = bool(out.get("partial")) or bool(anchor_json.get("partial"))
    out["degraded"] = sorted(set(list(out.get("degraded") or [])) | set(list(anchor_json.get("degraded") or [])))
    return out


def _extract_hint_terms_from_hits(hits: list[dict[str, Any]], *, max_terms: int) -> list[str]:
    """
    Best-effort: extract a few 'anchor' terms from top hits to build a follow-up query.
    Keep it conservative to avoid query drift.
    """
    cand: list[str] = []
    for h in hits[:8]:
        t = str(h.get("text") or "")
        # Pull years and capitalized-ish tokens (in practice, proper nouns in English pages)
        cand.extend(_YEAR_RE.findall(t))
        cand.extend([x for x in _TOKEN_RE.findall(t) if len(x) >= 4][:20])
    # Normalize and prefer longer distinct terms; dedupe with fuzzy matching
    cand = [c.strip() for c in cand if c and c.strip()]
    cand = sorted(set(cand), key=lambda s: (-len(s), s))[: max_terms * 3]
    # Keep only a few, fuzzy-deduped
    uniq: list[str] = []
    for x in cand:
        nx = _norm_query(x)
        if not nx:
            continue
        if any(fuzz.token_set_ratio(nx, _norm_query(u)) >= 92 for u in uniq):
            continue
        uniq.append(x)
        if len(uniq) >= max_terms:
            break
    return uniq


def _unique_doc_count(hits: list[dict[str, Any]]) -> int:
    """Count unique document IDs in hits."""
    s: set[str] = set()
    for h in hits:
        did = str(h.get("doc_id") or "").strip()
        if did:
            s.add(did)
    return len(s)

