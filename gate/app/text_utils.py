"""Text processing and normalization utilities."""

import re
import unicodedata
from collections import Counter as CollCounter

_CIT_RE = re.compile(r"\[\d+\]")
_WS_RE = re.compile(r"\s+")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
# Unicode-friendly tokenization (EN + RU + digits). Keep it simple and fast.
# \w includes underscore; that's acceptable for search queries.
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_FACTOID_LEAD_RE = re.compile(
    r"^\s*(who|what|which|where|when|how many|how much|кто|что|какой|какая|какие|где|когда|сколько)\b",
    re.IGNORECASE,
)


def _has_inline_citations(text: str) -> bool:
    """Check if text contains inline citations like [1], [2], etc."""
    return bool(text) and bool(_CIT_RE.search(text))


def _norm_query(q: str) -> str:
    """Normalize query: lowercase, strip, collapse whitespace."""
    q = (q or "").strip().lower()
    q = _WS_RE.sub(" ", q)
    return q


def _strip_diacritics(s: str) -> str:
    """Remove diacritics/combining accents for better matching."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_text_for_contains(s: str) -> str:
    """Normalize text for substring matching."""
    s = _strip_diacritics((s or "").lower())
    s = _WS_RE.sub(" ", s).strip()
    return s


def _is_factoid_like_question(q: str) -> bool:
    """Detect if query looks like a factoid question."""
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
    """Check if answer text is a substring of context."""
    ans = (answer or "").strip()
    ctx = (context_text or "").strip()
    if not ans or not ctx:
        return False
    n_ans = _norm_text_for_contains(ans)
    n_ctx = _norm_text_for_contains(ctx)
    return bool(n_ans and n_ctx and n_ans in n_ctx)


def _keyword_query(q: str) -> str:
    """Extract keywords from query, removing stopwords."""
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

