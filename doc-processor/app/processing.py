from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple

from app.clients import OCRClient
from app.extraction import (
    count_meaningful_chars,
    extract_pdf_text_layer,
    extract_text_non_vlm,
    normalize_to_pdf,
    pdf_pages_to_pngs,
)
from app.logging_setup import logger


class FileProcessor(ABC):
    """Strategy interface for turning raw file bytes into text.

    Implementations return a tuple ``(pages_text, normalized_content_type)`` where
    ``pages_text`` is always a ``list[str]`` – a single‑page document is simply a
    list with one element.
    """

    @abstractmethod
    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        raise NotImplementedError


class NonVLMProcessor(FileProcessor):
    """Base class for processors that do not call the VLM endpoint."""


# ---------------------------------------------------------------------------
# Concrete strategies for the most common formats
# ---------------------------------------------------------------------------

class TextProcessor(NonVLMProcessor):
    """Plain‑text (or UTF‑8) files.

    The raw bytes are decoded directly. If decoding fails we fall back to the
    generic non‑VLM extractor which can handle a few edge‑cases.
    """

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        try:
            text = raw.decode("utf-8")
            return [text], content_type or "text/plain"
        except Exception:
            # Fallback to the generic extractor – it knows how to handle many
            # other encodings and edge cases.
            ed = extract_text_non_vlm(raw, content_type, filename)
            return ed.pages_text, ed.content_type or content_type


class OCRPDFProcessor(FileProcessor):
    """PDF-like documents processed with text-layer extraction first, OCR second."""

    def __init__(self, ocr: OCRClient | None, settings):
        self.ocr = ocr
        self.settings = settings
        self.last_path = "pdf_text"

    async def _extract_pdf_pages(self, pdf_bytes: bytes) -> List[str]:
        pages_text = extract_pdf_text_layer(pdf_bytes, max_pages=self.settings.max_pages)
        if not pages_text:
            return []

        if self.ocr is None:
            return pages_text

        ocr_indices = [
            i for i, text in enumerate(pages_text) if count_meaningful_chars(text) < self.settings.pdf_text_min_chars
        ]
        if not ocr_indices:
            return pages_text

        rendered_pages = pdf_pages_to_pngs(
            pdf_bytes,
            page_indices=ocr_indices,
            max_side_px=self.settings.max_image_side_px,
        )
        sem = asyncio.Semaphore(max(1, int(self.settings.ocr_max_concurrency)))

        async def run_one(i: int, png_bytes: bytes) -> Tuple[int, str]:
            async with sem:
                text, _score = await asyncio.to_thread(self.ocr.page_to_text, png_bytes=png_bytes)
                return i, text

        results = await asyncio.gather(*(run_one(i, png) for i, png in rendered_pages), return_exceptions=True)
        improved = False
        for result in results:
            if isinstance(result, Exception):
                logger.warning("ocr_page_failed", extra={"extra": {"error": str(result)}})
                continue
            i, text = result
            if count_meaningful_chars(text) > count_meaningful_chars(pages_text[i]):
                pages_text[i] = text
                improved = True
        if improved:
            self.last_path = "ocr"
        return pages_text

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        pages_text = await self._extract_pdf_pages(raw)
        return pages_text, content_type or "application/pdf"


class DocxProcessor(OCRPDFProcessor):
    """DOC/DOCX (and similar office formats) – convert to PDF, then apply PDF routing.
    """

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        pdf_bytes, norm_ct = normalize_to_pdf(raw, content_type, filename)
        if not pdf_bytes:
            return [], norm_ct or content_type
        pages_text = await self._extract_pdf_pages(pdf_bytes)
        return pages_text, norm_ct


class XMLProcessor(NonVLMProcessor):
    """XML/HTML files – processed without the VLM.
    """

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        ed = extract_text_non_vlm(raw, content_type, filename)
        return ed.pages_text, ed.content_type or content_type


class DefaultProcessor(NonVLMProcessor):
    """Catch‑all fallback – uses the generic non‑VLM extractor.
    """

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        ed = extract_text_non_vlm(raw, content_type, filename)
        return ed.pages_text, ed.content_type or content_type


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_processor(state, content_type: str | None, filename: str) -> FileProcessor:
    """Select a concrete ``FileProcessor`` based on the file extension.

    The order is important – more specific formats are checked first.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in {"txt", "md", "csv", "log"}:
        return TextProcessor()
    if ext == "pdf":
        return OCRPDFProcessor(state.ocr, state.settings)
    if ext in {"doc", "docx", "odt", "rtf"}:
        return DocxProcessor(state.ocr, state.settings)
    if ext in {"xml", "html", "htm"}:
        return XMLProcessor()
    return DefaultProcessor()

