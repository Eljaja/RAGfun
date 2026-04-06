from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

CHUNK_PAGES = 50


def split_pdf_into_chunks(src: Path, out_root: Path, pages_per_chunk: int = CHUNK_PAGES) -> None:
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    out_dir = out_root / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(src)
    try:
        total = len(doc)
        part = 1
        for start in range(0, total, pages_per_chunk):
            end = min(start + pages_per_chunk, total)  # end is exclusive for slicing logic
            chunk = fitz.open()
            try:
                chunk.insert_pdf(doc, from_page=start, to_page=end - 1)
                out_path = out_dir / f"part_{part:03d}_p{start + 1}-{end}.pdf"
                chunk.save(out_path)
            finally:
                chunk.close()
            part += 1
    finally:
        doc.close()


def main() -> None:
    # Resolve paths how you prefer: absolute, or relative to this file / CWD
    base = Path("/home/ubuntu/tnovik/main_rag_folder/RAGfun/RAGcircle/test_data")
    out_root = base / "chunked"  # or Path("./chunked_output").resolve()

    pdf_names = [
        "management-center-admin-10-0.pdf",
        "management-center-device-config-10-0.pdf",
        # "other.pdf",
    ]

    for name in pdf_names:
        split_pdf_into_chunks(base / name, out_root)


if __name__ == "__main__":
    main()