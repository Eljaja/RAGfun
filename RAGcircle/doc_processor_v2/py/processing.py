from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from extraction import (
    get_pdf_page_count,
    pdf_to_page_pngs,
    extract_pdf_with_kreuzberg,
    _convert_office_to_pdf,
    _html_to_text,
    _xml_to_text,
    _xlsx_to_markdown,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
logger = logging.getLogger("data.processing.ocr")


# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """Processing settings for document conversion and chunking.

    All fields are required — values must come from the project config
    fetched from gate, never from hardcoded defaults.
    """
    vlm_model: str
    vlm_concurrency: int
    page_window: int
    max_px: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    ocr_mode: str = "expensive"


# ─────────────────────────────────────────────────────────────
# VLM Client
# ─────────────────────────────────────────────────────────────
# TODO: Move to separate module (e.g., clients/vlm.py)
# TODO: Add retry logic with exponential backoff

class VLMClient:
    """
    OpenAI-compatible chat completions client for multimodal extraction in vLLM.

    Creates a persistent httpx.AsyncClient — call close() on shutdown.
    The model is specified per-call from the project config.
    """

    def __init__(self, *, base_url: str, api_key: str | None, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_s,
        )

    async def page_to_text(self, *, png_bytes: bytes, model: str) -> str:
        """Extract text from a page image using the project's VLM model."""
        data_url = "data:image/png;base64," + \
            base64.b64encode(png_bytes).decode("ascii")

        payload = {
            "model": model,
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

        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        try:
            return str(r.json()["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError):
            raise RuntimeError("vlm_unexpected_response")

    async def close(self) -> None:
        await self._client.aclose()


class ToText(Protocol):
    """Anything that can become text."""

    def to_text(self) -> list[str]:
        ...


class WindowedExtractor(Protocol):
    """Document types that support windowed extract -> chunk -> ingest."""

    def to_text_windowed(self) -> AsyncIterator[tuple[int, list[str]]]:
        ...

# ─────────────────────────────────────────────────────────────
# CONVERTERS: Each format knows how to become text
# ─────────────────────────────────────────────────────────────


class PDFDocument:
    """Convert PDF to text via VLM page-by-page extraction."""

    def __init__(self, raw: bytes, vlm: VLMClient, settings: Settings):
        self.raw = raw
        self.vlm = vlm
        self.settings = settings
        self._vlm_model = settings.vlm_model

    async def to_text_windowed(self) -> AsyncIterator[tuple[int, list[str]]]:
        """
        Yield ``(page_offset, texts)`` for each window of pages.

        ``page_offset`` is the 0-based index of the first page in the window
        so the caller can compute correct global page numbers.
        Each ``texts`` list contains one string per page in that window.
        """
        total = get_pdf_page_count(self.raw)
        window = self.settings.page_window

        for start in range(0, total, window):
            first = start + 1                       # pdf2image is 1-indexed
            last = min(start + window, total)
            pngs = pdf_to_page_pngs(
                self.raw,
                first_page=first,
                last_page=last,
                max_side_px=self.settings.max_px,
            )
            texts = await self._vlm_extract_pages(pngs)
            yield start, texts

    async def to_text(self) -> list[str]:
        """Convenience wrapper: collect all windows into one flat list."""
        all_texts: list[str] = []
        async for _offset, texts in self.to_text_windowed():
            all_texts.extend(texts)
        return all_texts

    async def _vlm_extract_pages(self, pngs: list[bytes]) -> list[str]:
        """
        Extract text from page images using bounded parallel VLM calls.
        Returns list of text strings in page order.
        """
        if not pngs:
            return []

        num_pages = len(pngs)
        pages_text: list[str] = [""] * num_pages
        sem = asyncio.Semaphore(self.settings.vlm_concurrency)

        async def extract_one(idx: int, png_bytes: bytes) -> tuple[int, str]:
            async with sem:
                text = await self.vlm.page_to_text(png_bytes=png_bytes, model=self._vlm_model)
                return (idx, text)

        # now this is a thing for guard or some other limiter for requests 
        # this way we risk to overload the poor vllm engine 
        # need to think about smol scaling pipeline 
        tasks = [extract_one(i, png) for i, png in enumerate(pngs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                continue
            idx, text = r
            if text:
                pages_text[idx] = text

        return pages_text


class KreuzbergPDFDocument:
    """Cheap PDF extraction path using Kreuzberg."""

    def __init__(self, raw: bytes):
        self.raw = raw

    async def to_text(self) -> list[str]:
        extracted = extract_pdf_with_kreuzberg(self.raw)
        return extracted.pages_text


def normalize_ocr_mode(value: str | None) -> str:
    mode = (value or "").strip().lower()
    if mode in {"cheap", "kreuzberg"}:
        return "cheap"
    if mode in {"auto"}:
        return "auto"
    return "expensive"


def _is_weak_pdf_extraction(texts: list[str], *, page_count: int) -> bool:
    if not texts:
        return True
    merged = "\n".join(t for t in texts if t).strip()
    if not merged:
        return True

    char_count = len(merged)
    min_chars = max(800, page_count * 120)
    if char_count < min_chars:
        return True

    alpha_count = sum(1 for ch in merged if ch.isalpha())
    density = alpha_count / max(char_count, 1)
    if density < 0.2:
        return True
    return False


class AutoPDFDocument:
    """Try cheap OCR first and fallback to VLM when quality is weak."""

    def __init__(self, raw: bytes, vlm: VLMClient, settings: Settings):
        self.raw = raw
        self.vlm = vlm
        self.settings = settings

    async def to_text(self) -> list[str]:
        cheap = await KreuzbergPDFDocument(self.raw).to_text()
        page_count = max(get_pdf_page_count(self.raw), 1)
        if _is_weak_pdf_extraction(cheap, page_count=page_count):
            logger.info("auto_ocr_fallback_to_expensive page_count=%d", page_count)
            return await PDFDocument(self.raw, self.vlm, self.settings).to_text()
        logger.info("auto_ocr_kept_cheap page_count=%d", page_count)
        return cheap


class OfficeDocument:
    """Office docs are converted to PDF first, then processed identically."""

    def __init__(self, raw: bytes, ext: str, vlm: VLMClient, settings: Settings):
        self.raw = raw
        self.ext = ext
        self.vlm = vlm
        self.settings = settings

    def _as_pdf(self) -> PDFDocument:
        pdf_bytes = _convert_office_to_pdf(self.raw, self.ext)
        return PDFDocument(pdf_bytes, self.vlm, self.settings)

    async def to_text_windowed(self) -> AsyncIterator[tuple[int, list[str]]]:
        async for item in self._as_pdf().to_text_windowed():
            yield item

    async def to_text(self) -> list[str]:
        return await self._as_pdf().to_text()


class HTMLDocument:
    def __init__(self, raw: bytes):
        self.raw = raw

    async def to_text(self) -> list[str]:
        return [_html_to_text(self.raw)]


class XMLDocument:
    def __init__(self, raw: bytes):
        self.raw = raw

    async def to_text(self) -> list[str]:
        return [_xml_to_text(self.raw)]


class SpreadsheetDocument:
    def __init__(self, raw: bytes, filename: str | None):
        self.raw = raw
        self.filename = filename

    async def to_text(self) -> list[str]:
        return [_xlsx_to_markdown(self.raw, self.filename)]


class PlainTextDocument:
    def __init__(self, raw: bytes):
        self.raw = raw

    async def to_text(self) -> list[str]:
        return [self.raw.decode("utf-8", errors="replace")]


# TODO: Consider registry pattern instead of if/elif chain
# TODO: Return Result[ToText, UnsupportedFormatError] for unknown types
# TODO: lol you already have code to handle that in gate, so yeah move it here 
def document_from_bytes(
    raw: bytes,
    content_type: str | None,
    filename: str | None,
    vlm: VLMClient,
    settings: Settings,
) -> ToText:
    """Route to the right converter based on content type / filename."""

    ct = (content_type or "").lower()
    name = (filename or "").lower()

    # PDF
    if ct == "application/pdf" or name.endswith(".pdf"):
        mode = normalize_ocr_mode(settings.ocr_mode)
        if mode == "cheap":
            return KreuzbergPDFDocument(raw)
        if mode == "auto":
            return AutoPDFDocument(raw, vlm, settings)
        return PDFDocument(raw, vlm, settings)

    # Office
    if ct in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",] or name.endswith((".doc", ".docx")):
        ext = "docx" if name.endswith(".docx") else "doc"
        return OfficeDocument(raw, ext, vlm, settings)

    # Spreadsheet
    if ct == XLSX_TYPE or name.endswith(".xlsx"):
        return SpreadsheetDocument(raw, filename)

    # HTML
    if ct in {"text/html", "application/xhtml+xml"} or name.endswith((".html", ".htm")):
        return HTMLDocument(raw)

    # XML
    if ct in {"application/xml", "text/xml"} or name.endswith(".xml"):
        return XMLDocument(raw)

    # Fallback: plain text
    return PlainTextDocument(raw)


# ─────────────────────────────────────────────────────────────
# Main processing entry points
# ─────────────────────────────────────────────────────────────

def is_windowed(doc: ToText) -> bool:
    """True when the document supports windowed extract-chunk-ingest."""
    return hasattr(doc, "to_text_windowed")


async def file_to_texts(
    raw: bytes,
    content_type: str | None,
    filename: str | None,
    vlm: VLMClient,
    settings: Settings,
) -> list[str]:
    """
    Simple entry point: convert file bytes to a flat list of text segments.
    For windowed pipeline use, call ``document_from_bytes`` + ``is_windowed`` instead.
    """
    document = document_from_bytes(raw, content_type, filename, vlm, settings)
    return await document.to_text()
