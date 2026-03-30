from __future__ import annotations

import base64
import logging
import re
import tempfile
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("processor.clients")


class StorageClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def get_metadata(self, *, doc_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            # Use query param to avoid FastAPI path parsing issues with colons
            r = await client.get(f"{self._base_url}/v1/documents/by-id/metadata", params={"doc_id": doc_id})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    async def get_file(self, *, doc_id: str) -> tuple[bytes, str | None]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.get(f"{self._base_url}/v1/documents/by-id", params={"doc_id": doc_id})
            r.raise_for_status()
            ct = r.headers.get("content-type")
            return (r.content, ct)

    async def patch_extra(self, *, doc_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/by-id/extra", json={"doc_id": doc_id, "patch": patch})
            r.raise_for_status()
            return r.json()

    async def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.delete(f"{self._base_url}/v1/documents/by-id", params={"doc_id": doc_id})
            r.raise_for_status()
            return r.json()


class RetrievalClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def index_upsert(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/upsert", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_delete(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/delete", json=payload)
            r.raise_for_status()
            return r.json()


class VLMClient:
    """
    OpenAI-compatible chat completions client for multimodal extraction in vLLM.
    """

    def __init__(self, *, base_url: str, api_key: str | None, model: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def page_to_text(self, *, png_bytes: bytes) -> str:
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Convert this document page to clean text. "
                                "Preserve structure (headings, lists, tables) as Markdown. "
                                "Do not invent content. Output ONLY the converted text."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            j = r.json()
        try:
            return str(j["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.error("vlm_unexpected_response", extra={"extra": {"keys": list(j.keys())}})
            raise RuntimeError("vlm_unexpected_response")


class OCRClient:
    """Thin wrapper around PaddleOCR's general OCR pipeline."""

    def __init__(
        self,
        *,
        lang: str,
        device: str,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
    ) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime install
            raise RuntimeError("paddleocr_not_installed") from exc

        parsed_langs = _parse_ocr_langs(lang)
        device_str = str(device or "cpu").strip().lower()
        paddle_device = "gpu" if device_str.startswith("gpu") else "cpu"

        self._engines: list[tuple[str, Any]] = []
        for one_lang in parsed_langs:
            kwargs = {
                "lang": one_lang,
                "device": paddle_device,
                "use_doc_orientation_classify": bool(use_doc_orientation_classify),
                "use_doc_unwarping": bool(use_doc_unwarping),
                "use_textline_orientation": bool(use_textline_orientation),
            }
            if paddle_device == "cpu":
                # PaddleOCR 3.4 + PaddlePaddle 3.3 on CPU currently behaves more reliably
                # with the conservative backend settings below.
                kwargs["enable_hpi"] = False
                kwargs["enable_mkldnn"] = False
                kwargs["cpu_threads"] = 4
            self._engines.append((one_lang, PaddleOCR(**kwargs)))
        logger.info("ocr_engines_ready", extra={"extra": {"langs": parsed_langs, "device": paddle_device}})

    def page_to_text(self, *, png_bytes: bytes) -> tuple[str, float | None]:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(png_bytes)
            tmp.flush()
            best_text = ""
            best_score: float | None = None
            best_chars = -1
            for lang_name, engine in self._engines:
                texts: list[str] = []
                scores: list[float] = []
                results = engine.predict(tmp.name)
                for item in results or []:
                    payload = getattr(item, "json", item)
                    if not isinstance(payload, dict):
                        continue
                    res = payload.get("res") if isinstance(payload.get("res"), dict) else payload
                    rec_texts = res.get("rec_texts") or []
                    rec_scores = res.get("rec_scores") or []
                    texts.extend(str(t).strip() for t in rec_texts if str(t).strip())
                    for score in rec_scores:
                        try:
                            scores.append(float(score))
                        except Exception:
                            continue
                text = "\n".join(texts).strip()
                avg_score = (sum(scores) / len(scores)) if scores else None
                chars = _meaningful_chars(text)
                score_cmp = avg_score if avg_score is not None else -1.0
                best_score_cmp = best_score if best_score is not None else -1.0
                if chars > best_chars or (chars == best_chars and score_cmp > best_score_cmp):
                    best_chars = chars
                    best_text = text
                    best_score = avg_score
                logger.debug(
                    "ocr_lang_result",
                    extra={
                        "extra": {
                            "lang": lang_name,
                            "chars": chars,
                            "avg_score": avg_score,
                        }
                    },
                )
        return (best_text, best_score)


def _parse_ocr_langs(value: str | None) -> list[str]:
    raw = (value or "").strip().lower()
    if not raw:
        return ["ru", "en"]
    # Friendly aliases for "multilingual" mode.
    if raw in {"multi", "multilingual", "auto"}:
        return ["ru", "en"]
    langs = [x.strip() for x in raw.split(",") if x.strip()]
    if not langs:
        return ["ru", "en"]
    # Deduplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for lang in langs:
        if lang not in seen:
            out.append(lang)
            seen.add(lang)
    return out


def _meaningful_chars(text: str) -> int:
    return len(re.findall(r"[0-9A-Za-zА-Яа-яЁё]", text or ""))





