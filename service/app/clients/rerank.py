from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential


class RerankClient:
    """
    External cross-encoder rerank service.

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


