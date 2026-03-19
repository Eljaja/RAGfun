"""Zero-cost heuristic query expansion. No LLM calls.

Ported from gate: keyword extraction, query variants, fuzzy dedup,
hint term extraction, factoid detection, grounding check.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from models import ChunkResult
from shared import strip_thinking

try:
    from rapidfuzz import fuzz as _fuzz

    def _token_set_ratio(a: str, b: str) -> float:
        return _fuzz.token_set_ratio(a, b)
except ImportError:
    from difflib import SequenceMatcher

    def _token_set_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

_WS_RE = re.compile(r"\s+")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

_FACTOID_LEAD_RE = re.compile(
    r"^\s*(who|what|which|where|when|how many|how much|кто|что|какой|какая|какие|где|когда|сколько)\b",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "with", "what", "which",
    "who", "when", "where", "why", "how", "many", "much", "did", "does",
    "do", "have", "has",
    "и", "а", "но", "или", "да", "нет", "не", "это", "этот", "эта", "эти",
    "как", "что", "кто", "где", "когда", "почему", "зачем", "сколько",
    "какой", "какая", "какие", "каков", "каково", "каковы",
    "в", "на", "по", "к", "у", "с", "со", "из", "за", "для", "о", "об",
    "про", "над", "под", "при", "от", "до", "после", "между", "через",
    "все", "всё", "же", "ли", "бы", "были", "был", "будет", "есть",
    "является", "являются", "то", "там", "тут",
})


def _norm_query(q: str) -> str:
    return _WS_RE.sub(" ", (q or "").strip().lower())


def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


# ── Query expansion ──────────────────────────────────────


def keyword_query(q: str) -> str:
    """Compact keyword-only query (helps BM25 on long questions)."""
    toks = [t.lower() for t in _TOKEN_RE.findall(q or "")]
    toks = [t for t in toks if len(t) >= 3 and t not in _STOP_WORDS]
    if not toks:
        return ""
    c = Counter(toks)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return " ".join(t for t, _ in ranked[:10])


def query_variants(q: str) -> list[str]:
    """Generate heuristic query variants from the original query."""
    q = (q or "").strip()
    if not q:
        return []
    out: list[str] = [q]

    kw = keyword_query(q)
    if kw and kw != q:
        out.append(kw)

    phrases = [
        (m.group(1) or m.group(2) or "").strip()
        for m in _QUOTED_RE.finditer(q)
    ]
    phrases = [p for p in phrases if p and len(p) >= 3]
    if phrases:
        out.append(" ".join(phrases))

    years = _YEAR_RE.findall(q)
    if years:
        out.append(" ".join(sorted(set(years))))

    return dedupe_queries(out)


def dedupe_queries(qs: list[str], *, threshold: int = 92) -> list[str]:
    """Fuzzy-deduplicate a list of queries."""
    out: list[str] = []
    norms: list[str] = []
    for q in qs:
        q = (q or "").strip()
        if not q:
            continue
        nq = _norm_query(q)
        if not nq:
            continue
        if any(_token_set_ratio(nq, prev) >= threshold for prev in norms):
            continue
        out.append(q)
        norms.append(nq)
    return out


# ── Hint term extraction (two-pass) ─────────────────────


def extract_hint_terms(
    chunks: list[ChunkResult],
    *,
    max_terms: int = 3,
) -> list[str]:
    """Extract anchor terms from top chunks for a follow-up query."""
    raw: list[str] = []
    for chunk in chunks[:8]:
        raw.extend(_YEAR_RE.findall(chunk.text or ""))
        raw.extend(t for t in _TOKEN_RE.findall(chunk.text or "") if len(t) >= 4)

    candidates = sorted(
        {c.strip() for c in raw if c and c.strip()},
        key=lambda s: (-len(s), s),
    )[:max_terms * 3]

    uniq: list[str] = []
    for cand in candidates:
        norm = _norm_query(cand)
        if not norm:
            continue
        if any(_token_set_ratio(cand, u) >= 92 for u in uniq):
            continue
        uniq.append(cand)
        if len(uniq) >= max_terms:
            break
    return uniq


def unique_source_count(chunks: list[ChunkResult]) -> int:
    return len({c.source_id for c in chunks if c.source_id})


# ── Factoid detection + grounding ────────────────────────


def is_factoid_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    toks = _TOKEN_RE.findall(q)
    if len(toks) > 14:
        return False
    if "?" in q:
        return True
    return bool(_FACTOID_LEAD_RE.search(q))


def answer_is_grounded(*, answer: str, context_text: str) -> bool:
    """Check if the answer text appears as a substring of the context (post-normalization)."""
    ans = _strip_diacritics((answer or "").strip().lower())
    ctx = _strip_diacritics((context_text or "").strip().lower())
    ans = _WS_RE.sub(" ", ans).strip()
    ctx = _WS_RE.sub(" ", ctx).strip()
    return bool(ans and ctx and ans in ctx)


__all__ = [
    "keyword_query", "query_variants", "dedupe_queries",
    "extract_hint_terms", "unique_source_count",
    "is_factoid_question", "answer_is_grounded", "strip_thinking",
]
