from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from retrieval_contract.chunks import ChunkResult
from retrieval_contract.steps import MAX_QUERY_LEN, ExecutionPlan, validate_query_str


class _RequestBase(BaseModel):
    project_id: str
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, v: object) -> str:
        return validate_query_str(v, required=True)


class RetrieveRequest(_RequestBase):
    top_k: int = Field(default=5, ge=1, le=2000)
    rerank: bool = True
    rerank_top_n: int = Field(default=5, ge=1, le=2000)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    strategy: str
    query: str
    plan_used: ExecutionPlan | None = None


class ExecuteRequest(_RequestBase):
    plan: ExecutionPlan


class ExecuteResponse(BaseModel):
    chunks: list[ChunkResult]
    query: str
    plan: ExecutionPlan
