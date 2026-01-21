"""Main FastAPI application for RAG Gate."""

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.clients import DocProcessorClient, DocumentStorageClient, LLMClient, RetrievalClient
from app.config import Settings, load_settings
from app.html_text import html_to_text
from app.logging_setup import setup_json_logging
from app.models import ChatRequest, ChatResponse, ContextChunk, Source
from app.queue import RabbitPublisher
from app.rag import build_context_blocks, build_messages
from rapidfuzz import fuzz

logger = logging.getLogger("gate")

REQS = Counter("gate_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("gate_latency_seconds", "Latency", ["stage"])

ING_PUB = Counter("gate_ingestion_tasks_published_total", "Ingestion tasks published", ["type", "status"])
ING_PUB_LAT = Histogram("gate_ingestion_publish_latency_seconds", "Publish latency", ["type"])

# Standard HTTP server metrics (shared names across services; Prometheus "job" label disambiguates)
HTTP_INFLIGHT = Gauge("http_server_inflight_requests", "In-flight HTTP requests", ["method", "route"])
HTTP_REQS = Counter("http_server_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LAT = Histogram(
    "http_server_request_duration_seconds",
    "HTTP request duration (seconds)",
    ["method", "route", "status"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60),
)
HTTP_REQ_SIZE = Histogram(
    "http_server_request_size_bytes",
    "HTTP request size (bytes), from Content-Length when available",
    ["method", "route"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)
HTTP_RESP_SIZE = Histogram(
    "http_server_response_size_bytes",
    "HTTP response size (bytes), from Content-Length when available",
    ["method", "route", "status"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)

# Business-quality metrics
GATE_REFUSALS = Counter("gate_refusals_total", "Refusals (answer == 'I don't know')", ["endpoint"])
GATE_DEGRADED = Counter("gate_degraded_total", "Degradation events", ["kind"])
GATE_PARTIAL = Counter("gate_partial_total", "Partial responses", ["endpoint"])

# Pre-create common label series so Grafana panels show 0 instead of "No data" right after startup.
GATE_REFUSALS.labels(endpoint="/v1/chat").inc(0)
GATE_PARTIAL.labels(endpoint="/v1/chat").inc(0)


_CIT_RE = re.compile(r"\[\d+\]")


def _has_inline_citations(text: str) -> bool:
    return bool(text) and bool(_CIT_RE.search(text))


_WS_RE = re.compile(r"\s+")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
# Unicode-friendly tokenization (EN + RU + digits). Keep it simple and fast.
# \w includes underscore; that's acceptable for search queries.
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _norm_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = _WS_RE.sub(" ", q)
    return q


def _strip_diacritics(s: str) -> str:
    # Helps matching answers like "Улья́новск" vs "Ульяновск" (combining accents).
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_text_for_contains(s: str) -> str:
    s = _strip_diacritics((s or "").lower())
    s = _WS_RE.sub(" ", s).strip()
    return s


_FACTOID_LEAD_RE = re.compile(
    r"^\s*(who|what|which|where|when|how many|how much|кто|что|какой|какая|какие|где|когда|сколько)\b",
    re.IGNORECASE,
)


def _is_factoid_like_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    toks = _TOKEN_RE.findall(q)
    # Keep it conservative: only short questions (common in ru_eval).
    if len(toks) > 14:
        return False
    if "?" in q:
        return True
    return bool(_FACTOID_LEAD_RE.search(q))


def _answer_is_in_context(*, answer: str, context_text: str) -> bool:
    ans = (answer or "").strip()
    ctx = (context_text or "").strip()
    if not ans or not ctx:
        return False
    n_ans = _norm_text_for_contains(ans)
    n_ctx = _norm_text_for_contains(ctx)
    return bool(n_ans and n_ctx and n_ans in n_ctx)


async def _enforce_extractive_or_refuse(
    *,
    llm: LLMClient,
    messages: list[dict[str, str]],
    answer: str,
    context_text: str,
) -> str:
    """
    Best-effort anti-hallucination for factoid QA when include_sources=False:
    if the produced short answer is not present in sources, do one rewrite attempt
    forcing a verbatim span. If it still fails, refuse.
    """
    ans = (answer or "").strip()
    if not ans:
        return ans
    ctx = (context_text or "").strip()
    if not ctx:
        return ans

    # Apply only to short answers (entity/one-liner). Lists are hard to substring-match reliably.
    ans_toks = _TOKEN_RE.findall(ans)
    if len(ans_toks) > 6:
        return ans
    if "," in ans or "\n" in ans:
        return ans

    n_ans = _norm_text_for_contains(ans)
    n_ctx = _norm_text_for_contains(ctx)
    if n_ans and n_ans in n_ctx:
        return ans

    rewrite_prompt = (
        "Your previous answer is NOT a verbatim span from the provided Sources.\n"
        "Rewrite your answer STRICTLY as follows:\n"
        "- Output ONLY the exact answer phrase copied verbatim from Sources (no extra words).\n"
        "- Do NOT add citations like [1].\n"
        "- If the exact answer phrase is not explicitly present in Sources, reply exactly with \"I don't know\" (English) OR \"Не знаю.\" (Russian), matching the user's language.\n"
    )
    retry_messages = list(messages) + [{"role": "user", "content": rewrite_prompt}]
    rewritten = (await llm.chat(messages=retry_messages)).strip()
    if rewritten == "I don't know":
        return rewritten
    n_rew = _norm_text_for_contains(rewritten)
    if n_rew and n_rew in n_ctx:
        return rewritten
    return "I don't know"


def _dedupe_queries(qs: list[str], *, threshold: int = 92) -> list[str]:
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


def _keyword_query(q: str) -> str:
    # Keep a compact keyword-only query (helps BM25 on long questions).
    toks = [t.lower() for t in _TOKEN_RE.findall(q or "")]
    if not toks:
        return ""
    stop = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "with",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "many",
        "much",
        "did",
        "does",
        "do",
        "have",
        "has",
        # RU stopwords (small list, just to avoid query noise)
        "и",
        "а",
        "но",
        "или",
        "да",
        "нет",
        "не",
        "это",
        "этот",
        "эта",
        "эти",
        "как",
        "что",
        "кто",
        "где",
        "когда",
        "почему",
        "зачем",
        "сколько",
        "какой",
        "какая",
        "какие",
        "каков",
        "каково",
        "каковы",
        "в",
        "на",
        "по",
        "к",
        "у",
        "с",
        "со",
        "из",
        "за",
        "для",
        "о",
        "об",
        "про",
        "над",
        "под",
        "при",
        "от",
        "до",
        "после",
        "между",
        "через",
        "все",
        "всё",
        "же",
        "ли",
        "бы",
        "были",
        "был",
        "будет",
        "есть",
        "является",
        "являются",
        "то",
        "там",
        "тут",
    }
    toks = [t for t in toks if len(t) >= 3 and t not in stop]
    if not toks:
        return ""
    # Prefer frequent + longer tokens
    c = CollCounter(toks)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    keep = [t for t, _ in ranked[:10]]
    return " ".join(keep)


def _query_variants(q: str) -> list[str]:
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
    s: set[str] = set()
    for h in hits:
        did = str(h.get("doc_id") or "").strip()
        if did:
            s.add(did)
    return len(s)


def _inc_count(d: dict[str, int], key: str, *, n: int = 1) -> None:
    if not key:
        key = "unknown"
    try:
        d[key] = int(d.get(key) or 0) + int(n)
    except Exception:
        d[key] = int(n)


def _ingestion_state_from_doc(doc: dict[str, Any]) -> str:
    ing = (doc or {}).get("extra") or {}
    ing = ing.get("ingestion") if isinstance(ing, dict) else None
    state = (ing or {}).get("state")
    return str(state).strip().lower() if state else "unknown"


async def _enforce_citations_or_refuse(
    *,
    llm: LLMClient,
    messages: list[dict[str, str]],
    answer: str,
    refs: list[int],
) -> str:
    """
    If sources exist and the model returned an answer without citations, do one strict rewrite attempt.
    If it still fails, refuse (to avoid "fake grounding" by auto-adding citations).
    """
    if not refs:
        return answer
    if _has_inline_citations(answer):
        return answer

    suffix = "".join([f"[{r}]" for r in refs])
    rewrite_prompt = (
        "Rewrite your answer to comply STRICTLY with citations.\n"
        f"- Every factual claim MUST include inline citations like {suffix}\n"
        "- If you cannot answer using the provided sources, reply exactly: \"I don't know\" (no citations).\n"
        "- Do NOT add any facts, numbers, commands, or code that are not explicitly present in the sources.\n"
        "- Do NOT use generic advice or filler.\n"
    )
    retry_messages = list(messages) + [{"role": "user", "content": rewrite_prompt}]
    rewritten = await llm.chat(messages=retry_messages)
    if _has_inline_citations(rewritten) or rewritten.strip() == "I don't know":
        return rewritten
    return "I don't know"


class AppState:
    settings: Settings | None = None
    config_error: str | None = None
    retrieval: RetrievalClient | None = None
    llm: LLMClient | None = None
    storage: DocumentStorageClient | None = None
    doc_processor: DocProcessorClient | None = None
    publisher: RabbitPublisher | None = None


state = AppState()


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combine multiple lifespan contexts."""
    async with AsyncExitStack() as stack:
        # Enter all context managers
        await stack.enter_async_context(rag_lifespan(app))
        await stack.enter_async_context(presign_lifespan(app))
        yield
        # All will be properly cleaned up even if errors occur


# Create FastAPI app
app = FastAPI(title="RAG Gate", version="0.1.0", lifespan=combined_lifespan)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(public_router)
app.include_router(protected_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configured in runtime after settings load; kept permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.middleware("http")(http_metrics_middleware)

