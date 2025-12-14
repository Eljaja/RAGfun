from __future__ import annotations

import io
import logging
import subprocess
import tempfile
from dataclasses import dataclass

import fitz  # PyMuPDF
from lxml import etree

logger = logging.getLogger("processor.extraction")


@dataclass(frozen=True)
class ExtractedDocument:
    content_type: str | None
    pages_text: list[str]


def _xml_to_text(raw: bytes) -> str:
    # Best-effort: parse as XML and extract text nodes.
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(raw, parser=parser)
    text = " ".join(t.strip() for t in root.itertext() if t and t.strip())
    return text.strip()


def _convert_office_to_pdf(raw: bytes, ext: str) -> bytes:
    """
    Convert DOC/DOCX to PDF via LibreOffice (headless).
    Returns PDF bytes. Raises on conversion failure.
    """
    ext = ext.lower().lstrip(".")
    if ext not in {"doc", "docx"}:
        raise ValueError(f"unsupported_office_ext:{ext}")

    with tempfile.TemporaryDirectory(prefix="docproc_") as td:
        in_path = f"{td}/input.{ext}"
        out_dir = td
        with open(in_path, "wb") as f:
            f.write(raw)

        # LibreOffice output file name is based on input basename.
        # Use --nologo/--nolockcheck for containers.
        cmd = [
            "soffice",
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            out_dir,
            in_path,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"libreoffice_failed: rc={p.returncode} stderr={p.stderr[:500]}")

        pdf_path = f"{td}/input.pdf"
        with open(pdf_path, "rb") as f:
            return f.read()


def pdf_to_page_pngs(pdf_bytes: bytes, *, max_pages: int, max_side_px: int) -> list[bytes]:
    """
    Render PDF pages to PNG bytes using PyMuPDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[bytes] = []
    n = min(doc.page_count, max_pages)
    for i in range(n):
        page = doc.load_page(i)
        # Scale so the largest side is max_side_px
        rect = page.rect
        max_side = max(rect.width, rect.height)
        zoom = max_side_px / max_side if max_side > 0 else 1.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(pix.tobytes("png"))
    return out


def extract_text_non_vlm(raw: bytes, content_type: str | None, filename: str | None) -> ExtractedDocument:
    """
    Fast extraction for XML/plain text without VLM.
    """
    ct = (content_type or "").split(";")[0].strip().lower() or None
    name = (filename or "").lower()

    if ct in {"application/xml", "text/xml"} or name.endswith(".xml"):
        return ExtractedDocument(content_type=ct, pages_text=[_xml_to_text(raw)])

    # Plain-ish text fallback
    try:
        text = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        text = ""
    return ExtractedDocument(content_type=ct, pages_text=[text])


def normalize_to_pdf(raw: bytes, content_type: str | None, filename: str | None) -> tuple[bytes | None, str | None]:
    """
    Convert inputs to PDF bytes when possible (pdf/doc/docx).
    Returns (pdf_bytes, normalized_content_type).
    """
    ct = (content_type or "").split(";")[0].strip().lower() or None
    name = (filename or "").lower()

    if ct == "application/pdf" or name.endswith(".pdf"):
        return (raw, "application/pdf")

    if ct in {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    } or name.endswith(".docx") or name.endswith(".doc"):
        ext = "docx" if name.endswith(".docx") or ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" else "doc"
        return (_convert_office_to_pdf(raw, ext), "application/pdf")

    return (None, ct)




