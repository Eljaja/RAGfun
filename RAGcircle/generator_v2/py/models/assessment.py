from __future__ import annotations

from pydantic import BaseModel, Field


class ReflectionResult(BaseModel):
    complete: bool
    missing_context: str | None = None
    requery: str | None = None


class AssessmentResult(BaseModel):
    incomplete: bool = False
    missing_terms: list[str] = Field(default_factory=list)
    reason: str = ""
