"""PDF rendering helpers (PyMuPDF/fitz)."""

from __future__ import annotations

import fitz  # pymupdf


def pdf_to_images(pdf_path: str, config: dict | None = None) -> list[bytes]:
    """Render each page to image bytes (``config["images"]`` dpi/format; default 150 DPI jpeg)."""
    img_cfg = (config or {}).get("images") or {}
    dpi = int(img_cfg.get("dpi", 150))
    fmt = img_cfg.get("format", "jpeg")
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    doc = fitz.open(pdf_path)
    try:
        return [
            page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB).tobytes(fmt)
            for page in doc
        ]
    finally:
        doc.close()


def pdf_first_page_text(pdf_path: str) -> str:
    """Return raw text of page 1 (``""`` for empty PDFs)."""
    doc = fitz.open(pdf_path)
    try:
        return doc[0].get_text() if doc.page_count else ""
    finally:
        doc.close()


def pdf_text_excl_first_page(pdf_path: str) -> str:
    """Return raw text of pages 2..n; page 1 of PsycTESTS PDFs is database boilerplate."""
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(doc[i].get_text() for i in range(1, doc.page_count))
    finally:
        doc.close()


def pdf_doc_stats(pdf_path: str) -> dict:
    """Return ``page_count``, ``image_count`` and ``char_count_excl_first_page``."""
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        image_count = sum(len(page.get_images()) for page in doc)
        char_count_excl_first_page = sum(
            len(doc[i].get_text()) for i in range(1, page_count)
        )
        return {
            "page_count": page_count,
            "image_count": image_count,
            "char_count_excl_first_page": char_count_excl_first_page,
        }
    finally:
        doc.close()
