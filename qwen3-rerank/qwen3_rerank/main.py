from __future__ import annotations

import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


MODEL_ID = os.getenv("RERANK_MODEL_ID", "Qwen/Qwen3-Reranker-4B")
MAX_LENGTH = _env_int("RERANK_MAX_LENGTH", 1024)
BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 8)


app = FastAPI(title="Qwen3 Rerank", version="0.1.0")

_tokenizer = None
_model = None
_device = None


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int | None = None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "device": str(_device) if _device is not None else None,
    }


@app.on_event("startup")
def _startup() -> None:
    global _tokenizer, _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if _device.type == "cuda" else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    _model.to(_device)
    _model.eval()


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict[str, Any]:
    if _tokenizer is None or _model is None or _device is None:
        raise HTTPException(status_code=503, detail="model_not_ready")

    if req.model and req.model != MODEL_ID:
        # Keep OpenAI-style request shape, but only one model is served per container.
        raise HTTPException(status_code=400, detail=f"model_not_deployed: {req.model}")

    docs = [d for d in req.documents if isinstance(d, str) and d]
    if not docs:
        return {"results": []}

    q = req.query or ""
    pairs = [(q, d) for d in docs]

    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), max(1, BATCH_SIZE)):
            batch = pairs[i : i + BATCH_SIZE]
            qs = [x[0] for x in batch]
            ds = [x[1] for x in batch]
            enc = _tokenizer(
                qs,
                ds,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(_device) for k, v in enc.items()}
            out = _model(**enc)
            logits = out.logits
            if logits.ndim != 2:
                raise HTTPException(status_code=500, detail="unexpected_logits_shape")

            # Common patterns:
            # - regression: [B, 1]
            # - binary classification: [B, 2] (use positive class)
            if logits.shape[1] == 1:
                s = logits[:, 0]
            else:
                s = logits[:, -1]
            scores.extend([float(x) for x in s.detach().float().cpu().tolist()])

    results = [{"index": idx, "relevance_score": sc} for idx, sc in enumerate(scores)]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)

    if req.top_n is not None:
        results = results[: max(0, int(req.top_n))]

    return {"results": results}



