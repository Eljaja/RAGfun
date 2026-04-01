from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ScoreSource(StrEnum):
    RETRIEVAL = "retrieval"
    RRF = "rrf"
    RERANK = "rerank"


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float
    score_source: ScoreSource = ScoreSource.RETRIEVAL
