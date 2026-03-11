from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class RetrieveRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    strategy: str
    query: str
