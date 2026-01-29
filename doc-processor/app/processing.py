from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple

from app.clients import VLMClient
from app.extraction import (
    extract_text_non_vlm,
    normalize_to_pdf,
    pdf_to_page_pngs,
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


# ---------------------------------------------------------------------------
# Concrete strategies for the most common formats
# ---------------------------------------------------------------------------

class TextProcessor(FileProcessor):
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


class PDFProcessor(FileProcessor):
    """PDF files – processed via the VLM (OCR) path.

    The raw bytes are already a PDF, so we skip the ``normalize_to_pdf`` step.
    """

    def __init__(self, vlm: VLMClient, settings):
        self.vlm = vlm
        self.settings = settings

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        # Render PDF pages to PNGs.
        pngs = pdf_to_page_pngs(
            raw,
            max_pages=self.settings.max_pages,
            max_side_px=self.settings.max_image_side_px,
        )
        pages = len(pngs)
        if pages == 0:
            return [], content_type

        async def one(i: int, b: bytes) -> Tuple[int, str]:
            t = await self.vlm.page_to_text(png_bytes=b)
            return i, t

        sem = asyncio.Semaphore(4)

        async def run_one(i: int, b: bytes):
            async with sem:
                return await one(i, b)

        results = await asyncio.gather(
            *(run_one(i, b) for i, b in enumerate(pngs)), return_exceptions=True
        )
        pages_text: List[str] = ["" for _ in range(pages)]
        for r in results:
            if isinstance(r, Exception):
                raise r
            i, t = r
            if t:
                pages_text[i] = t
        return pages_text, content_type or "application/pdf"


class DocxProcessor(FileProcessor):
    """DOC/DOCX (and similar office formats) – first converted to PDF, then VLM.
    """

    def __init__(self, vlm: VLMClient, settings):
        self.vlm = vlm
        self.settings = settings

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        # Convert to PDF using the existing helper.
        pdf_bytes, norm_ct = normalize_to_pdf(raw, content_type, filename)
        # Re‑use the PDFProcessor logic.
        pdf_processor = PDFProcessor(self.vlm, self.settings)
        return await pdf_processor.to_text(pdf_bytes, filename, norm_ct)


class XMLProcessor(FileProcessor):
    """XML/HTML files – processed without the VLM.
    """

    async def to_text(
        self, raw: bytes, filename: str, content_type: str | None
    ) -> Tuple[List[str], str | None]:
        ed = extract_text_non_vlm(raw, content_type, filename)
        return ed.pages_text, ed.content_type or content_type


class DefaultProcessor(FileProcessor):
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
        return PDFProcessor(state.vlm, state.settings)
    if ext in {"doc", "docx", "odt", "rtf"}:
        return DocxProcessor(state.vlm, state.settings)
    if ext in {"xml", "html", "htm"}:
        return XMLProcessor()
    # Anything else – try the VLM path if we can convert to PDF, otherwise fallback.
    try:
        _ = normalize_to_pdf(b"", content_type, filename)  # type: ignore[arg-type]
        # If conversion is possible we use the generic VLM processor.
        return PDFProcessor(state.vlm, state.settings)
    except Exception:
        return DefaultProcessor()

