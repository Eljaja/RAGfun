"""Quality evaluation and completeness assessment.

Includes: quality_is_poor, reflect, assess_completeness, detect_language.
"""

from __future__ import annotations

import json
import logging

from llm import LLMClient
from models import AssessmentResult, ChunkResult, ReflectionResult, ScoreSource
from prompts import (
    ASSESS_SYSTEM,
    ASSESS_USER,
    DETECT_LANG_SYSTEM,
    DETECT_LANG_USER,
    REFLECT_SYSTEM,
    REFLECT_USER,
)

logger = logging.getLogger(__name__)


def quality_is_poor(
    chunks: list[ChunkResult],
    *,
    min_hits: int = 3,
    min_score: float = 0.5,
) -> bool:
    """Check if retrieval quality is insufficient for answering.

    Score threshold is only applied when score_source is 'rerank' (absolute scale).
    For other score sources, only hit count heuristics are used.
    """
    if not chunks:
        return True
    if len(chunks) < min_hits:
        return True
    top = chunks[0]
    if top.score_source == ScoreSource.RERANK and top.score < min_score:
        return True
    return False


async def reflect(
    llm: LLMClient,
    model: str,
    query: str,
    context_text: str,
    answer: str,
) -> ReflectionResult:
    """Evaluate if the answer fully addresses the question. Used by simple pipeline."""
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": REFLECT_USER.format(
                query=query, context=context_text, answer=answer,
            )},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "reflection-result",
                "schema": ReflectionResult.model_json_schema(),
            },
        },
    )
    try:
        data = json.loads(raw)
        return ReflectionResult(
            complete=data.get("complete", True),
            missing_context=data.get("missing_context"),
            requery=data.get("requery"),
        )
    except Exception:
        logger.warning("Failed to parse reflection result, assuming complete")
        return ReflectionResult(complete=True)


async def assess_completeness(
    llm: LLMClient,
    model: str,
    question: str,
    answer: str,
    *,
    history_text: str = "",
) -> AssessmentResult:
    """Assess if the answer is complete. Used by agent pipeline for retry decisions."""
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": ASSESS_SYSTEM},
            {"role": "user", "content": ASSESS_USER.format(
                history=history_text, question=question, answer=answer,
            )},
        ],
        temperature=0.0,
    )
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return AssessmentResult()
        missing = data.get("missing_terms") or []
        if not isinstance(missing, list):
            missing = []
        return AssessmentResult(
            incomplete=bool(data.get("incomplete")),
            missing_terms=[str(t).strip() for t in missing if str(t).strip()],
            reason=str(data.get("reason") or ""),
        )
    except Exception:
        logger.warning("Failed to parse assessment result")
        return AssessmentResult()


async def detect_language(
    llm: LLMClient,
    model: str,
    text: str,
) -> str:
    raw = await llm.complete(
        model,
        [
            {"role": "system", "content": DETECT_LANG_SYSTEM},
            {"role": "user", "content": DETECT_LANG_USER.format(text=text)},
        ],
        temperature=0.0,
    )
    return (raw or "").strip() or "English"
