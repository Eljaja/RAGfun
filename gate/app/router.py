from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class RouterDecision(BaseModel):
    """
    Per-request retrieval knobs produced by the router model.
    Keep this intentionally small + stable.
    """

    retrieval_mode: Literal["bm25", "vector", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=40)
    rerank: bool | None = None
    use_multi_query: bool | None = None
    use_two_pass: bool | None = None
    use_bm25_anchor: bool | None = None
    use_segment_stitching: bool | None = None
    # Optional human-readable explanation for logs / debugging.
    reason: str | None = None


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_obj(text: str) -> str | None:
    """
    Best-effort extraction of a JSON object from a model response.
    Accepts raw JSON or JSON wrapped in markdown/codefences.
    """
    if not text:
        return None
    t = text.strip()
    # Strip common markdown fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()

    # If response is already JSON, keep it.
    if t.startswith("{") and t.endswith("}"):
        return t

    m = _JSON_OBJ_RE.search(t)
    if not m:
        return None
    return m.group(0).strip()


def parse_router_decision(text: str) -> RouterDecision | None:
    """
    Parse router output. Returns None if parsing/validation fails.
    """
    raw = _extract_json_obj(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return RouterDecision.model_validate(obj)
    except ValidationError:
        return None


def build_router_messages(*, query: str) -> list[dict[str, str]]:
    """
    Few-shot router prompt for selecting retrieval knobs.

    Output format: JSON object ONLY (no markdown).
    """
    sys = {
        "role": "system",
        "content": (
            "You are a query router for a RAG system.\n"
            "Your job: choose retrieval parameters for the NEXT retrieval call.\n\n"
            "Return ONLY a single JSON object (no markdown, no prose).\n"
            "If unsure, keep fields null or omit them.\n\n"
            "Available fields:\n"
            "- retrieval_mode: one of [\"bm25\",\"vector\",\"hybrid\"]\n"
            "- top_k: integer 1..40 (typical: 6..14)\n"
            "- rerank: boolean (true/false)\n"
            "- use_multi_query: boolean (improves recall for complex/multi-hop)\n"
            "- use_two_pass: boolean (second pass uses hint terms from pass1)\n"
            "- use_bm25_anchor: boolean (helps exact entity matching in hybrid)\n"
            "- use_segment_stitching: boolean (stitches chunks from same page/doc)\n"
            "- reason: short string for debugging\n\n"
            "Guidelines:\n"
            "- Short factoid/entity questions (who/what/where/when, 1-line answer): prefer bm25 or hybrid; low top_k; bm25_anchor=true; multi_query/two_pass usually false.\n"
            "- Multi-hop/comparison/\"why\" questions: prefer hybrid; higher top_k; multi_query=true; two_pass=true; segment_stitching=true.\n"
            "- Definitions/explanations: hybrid or vector; moderate top_k; rerank=true.\n"
            "- Keep top_k conservative to reduce noisy context.\n"
        ),
    }

    # Few-shot examples (RU-heavy; matches the project eval style)
    examples: list[dict[str, str]] = []

    examples += [
        {
            "role": "user",
            "content": "Query: Кто сочинил «Танец с саблями»?",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "retrieval_mode": "bm25",
                    "top_k": 8,
                    "rerank": True,
                    "use_multi_query": False,
                    "use_two_pass": False,
                    "use_bm25_anchor": True,
                    "use_segment_stitching": False,
                    "reason": "short factoid (composer name)",
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": "Query: Сравни подходы A и B и объясни, что лучше и почему.",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "retrieval_mode": "hybrid",
                    "top_k": 14,
                    "rerank": True,
                    "use_multi_query": True,
                    "use_two_pass": True,
                    "use_bm25_anchor": True,
                    "use_segment_stitching": True,
                    "reason": "multi-hop / comparison needs higher recall and stitched context",
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": "Query: Где находится гора Эверест?",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "retrieval_mode": "bm25",
                    "top_k": 6,
                    "rerank": True,
                    "use_multi_query": False,
                    "use_two_pass": False,
                    "use_bm25_anchor": True,
                    "use_segment_stitching": False,
                    "reason": "short factoid (location)",
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": "Query: Объясни, что такое фотосинтез простыми словами.",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "retrieval_mode": "hybrid",
                    "top_k": 10,
                    "rerank": True,
                    "use_multi_query": False,
                    "use_two_pass": False,
                    "use_bm25_anchor": False,
                    "use_segment_stitching": True,
                    "reason": "definition/explanation; prefer coherent stitched context",
                },
                ensure_ascii=False,
            ),
        },
    ]

    user = {
        "role": "user",
        "content": (
            "Choose parameters for this query.\n"
            f"Query: {query}\n"
            "Return ONLY JSON."
        ),
    }

    return [sys, *examples, user]


