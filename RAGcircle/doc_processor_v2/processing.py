from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Protocol

import httpx

from extraction import pdf_to_page_pngs, _convert_office_to_pdf, _html_to_text, _xml_to_text, _xlsx_to_markdown

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """Processing settings for document conversion and chunking."""
    max_pages: int = 50
    max_px: int = 2048
    vlm_concurrency: int = 4
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 200


# ─────────────────────────────────────────────────────────────
# VLM Client
# ─────────────────────────────────────────────────────────────
# TODO: Move to separate module (e.g., clients/vlm.py)
# TODO: Add connection pooling / reuse httpx client
# TODO: Add retry logic with exponential backoff
# TODO: Return Result[str, VLMError] instead of raising

class VLMClient:
    """
    OpenAI-compatible chat completions client for multimodal extraction in vLLM.
    """

    def __init__(self, *, base_url: str, api_key: str | None, model: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        # TODO: Create persistent httpx.AsyncClient here, close in __aexit__

    async def page_to_text(self, *, png_bytes: bytes) -> str:
        data_url = "data:image/png;base64," + \
            base64.b64encode(png_bytes).decode("ascii")
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
            # logger.error("vlm_unexpected_response", extra={"extra": {"keys": list(j.keys())}})
            raise RuntimeError("vlm_unexpected_response")


class ToText(Protocol):
    """Anything that can become text"""

    def to_text(self) -> list[str]:
        """Returns list of text segments (pages, sections, etc.)"""
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

    async def to_text(self) -> list[str]:
        """
        Render PDF pages as images and extract text via VLM.
        Returns a list of strings, one per page.
        """
        pngs = pdf_to_page_pngs(
            self.raw,
            max_pages=self.settings.max_pages,
            max_side_px=self.settings.max_px,
        )
        return await self._vlm_extract_pages(pngs)

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
                text = await self.vlm.page_to_text(png_bytes=png_bytes)
                return (idx, text)

        tasks = [extract_one(i, png) for i, png in enumerate(pngs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                # Log and continue - partial extraction is better than nothing
                continue
            idx, text = r
            if text:
                pages_text[idx] = text

        return pages_text


class OfficeDocument:
    def __init__(self, raw: bytes, ext: str, vlm: VLMClient, settings: Settings):
        self.raw = raw
        self.ext = ext
        self.vlm = vlm
        self.settings = settings

    async def to_text(self) -> list[str]:
        pdf = _convert_office_to_pdf(self.raw, self.ext)
        return await PDFDocument(pdf, self.vlm, self.settings).to_text()


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
# Main processing entry point
# ─────────────────────────────────────────────────────────────

async def file_to_texts(
    raw: bytes,
    content_type: str | None,
    filename: str | None,
    vlm: VLMClient,
    settings: Settings | None = None,
) -> list[str]:
    """
    Convert file bytes to a list of text segments (pages/sections).
    
    This is the main entry point for document processing.
    Routes to the appropriate converter based on content type/filename,
    then extracts text.
    
    Args:
        raw: File content as bytes
        content_type: MIME type (e.g., "application/pdf")
        filename: Original filename (used for format detection)
        vlm: VLM client for image-based extraction
        settings: Processing settings (uses defaults if None)
    
    Returns:
        List of text strings (one per page for PDFs, single item for others)
    """
    if settings is None:
        settings = Settings()
    
    document = document_from_bytes(raw, content_type, filename, vlm, settings)
    return await document.to_text()
