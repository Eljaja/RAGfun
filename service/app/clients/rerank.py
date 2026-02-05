from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("rag.rerank")


@runtime_checkable
class Reranker(Protocol):
    """
    Swappable reranker interface: cross-encoder over (query, document) pairs.
    Implementations: HttpRerankClient (external service), LocalReranker (in-process CPU).
    """

    async def rerank(self, query: str, candidates: list[dict[str, str]]) -> dict[str, float]:
        """
        Rerank candidates by relevance to query.
        candidates: list of {"id": chunk_id, "text": chunk_text}.
        Returns: dict chunk_id -> relevance score (higher = more relevant).
        """
        ...


class RerankClient:
    """
    HTTP reranker: external cross-encoder service (e.g. Infinity, qwen3-rerank).

    Expected (simple) contract:
      POST { "query": "...", "candidates": [{"id":"..","text":".."}, ...] }
      -> { "scores": [0.1, 0.2, ...] }

    We also accept:
      -> { "results": [{"id":"..","score":..}, ...] }
    """

    def __init__(self, url: str, model: str | None, api_key: str | None, timeout_s: float):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    @retry(wait=wait_random_exponential(multiplier=0.2, max=2.0), stop=stop_after_attempt(2))
    async def rerank(self, query: str, candidates: list[dict[str, str]]) -> dict[str, float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Two supported request/response styles:
        #
        # A) Our simple internal contract:
        #    POST { "query": "...", "candidates": [{"id":"..","text":".."}, ...] }
        #    -> { "scores": [..] } or { "results": [{"id":"..","score":..}, ...] }
        #
        # B) Infinity (OpenAI-aligned) rerank:
        #    POST { "model":"..", "query":"..", "documents":[...strings...] }
        #    -> { "results":[{"index":0,"relevance_score":..}, ...] } (or similar)
        if self.model:
            docs: list[str] = []
            ids: list[str] = []
            for c in candidates:
                ids.append(str(c.get("id", "")))
                docs.append(str(c.get("text", "")))
            payload = {"model": self.model, "query": query, "documents": docs}
        else:
            payload = {"query": query, "candidates": candidates}

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()

        scores = data.get("scores")
        if isinstance(scores, list):
            out: dict[str, float] = {}
            for c, s in zip(candidates, scores, strict=False):
                try:
                    out[c["id"]] = float(s)
                except Exception:
                    continue
            return out

        results = data.get("results")
        if isinstance(results, list):
            out = {}
            for it in results:
                if not isinstance(it, dict):
                    continue
                # Internal format: {"id": "...", "score": ...}
                cid = it.get("id")
                sc = it.get("score")
                if cid is not None and sc is not None:
                    try:
                        out[str(cid)] = float(sc)
                        continue
                    except Exception:
                        pass

                # Infinity format: {"index": 0, "relevance_score": ...} (map back to candidate ids)
                idx = it.get("index")
                rel = it.get("relevance_score")
                if idx is not None and rel is not None:
                    try:
                        i = int(idx)
                        if 0 <= i < len(candidates):
                            out[str(candidates[i].get("id"))] = float(rel)
                    except Exception:
                        continue
            return out

        raise RuntimeError("Bad rerank response: expected 'scores' or 'results'")


def _load_cross_encoder(model_id: str, device: str, max_length: int):
    """Lazy load CrossEncoder; raises ImportError if sentence_transformers not installed."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "Local reranker requires sentence_transformers. "
            "Install with: pip install sentence-transformers"
        ) from e
    return CrossEncoder(model_id, max_length=max_length, device=device)


class LocalReranker:
    """
    In-process cross-encoder reranker, CPU-compatible. Open-source via sentence-transformers.
    Use when RAG_RERANK_PROVIDER=local; no external service required.
    """

    def __init__(
        self,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_id, self.device, self.max_length)
            logger.info("local_reranker_loaded", extra={"model": self.model_id, "device": self.device})
        return self._model

    async def rerank(self, query: str, candidates: list[dict[str, str]]) -> dict[str, float]:
        if not candidates:
            return {}
        ids = [c.get("id", "") for c in candidates]
        texts = [c.get("text", "") or "" for c in candidates]
        pairs = [(query, t) for t in texts]
        model = self._get_model()
        # Run CPU-bound predict in thread to avoid blocking event loop
        scores_list = await asyncio.to_thread(
            _predict_batched,
            model,
            pairs,
            self.batch_size,
        )
        return {cid: float(s) for cid, s in zip(ids, scores_list, strict=True) if cid}


def _predict_batched(model, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Run model.predict in batches; returns list of scores in same order as pairs."""
    out: list[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        scores = model.predict(batch, convert_to_numpy=True, show_progress_bar=False)
        if hasattr(scores, "tolist"):
            out.extend(scores.tolist())
        else:
            out.extend([float(s) for s in scores])
    return out


