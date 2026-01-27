# pdf_parser_v2.py — Hybrid PDF Extraction Engine (Production-Ready)
# Enhanced for ingestion_v2 pipeline with async-safe extraction, fallback logic,
# and unified LLM normalization toggle.
import pdfplumber
import fitz  # PyMuPDF
import asyncio
import os
from typing import Dict, Any, List, Optional

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ Added LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global flag

# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for PDF parser
# False → Force disable LLM normalization
# None  → Inherit from global ENABLE_LLM_NORMALIZATION
LOCAL_LLM_TOGGLE = None


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# PDF EXTRACTION (PRIMARY)
# -------------------------------------------------------------------
def extract_with_pdfplumber(file_path: str) -> List[str]:
    """Extracts text page-by-page using pdfplumber (high accuracy for structured PDFs)."""
    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
    except Exception as e:
        log_warning(f"[pdf_parser_v2] pdfplumber failed: {e}")
    return pages_text


# -------------------------------------------------------------------
# PDF EXTRACTION (FALLBACK)
# -------------------------------------------------------------------
def extract_with_pymupdf(file_path: str) -> List[str]:
    """Fallback extraction using PyMuPDF for scanned or unusual PDFs."""
    pages_text = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                pages_text.append(text)
    except Exception as e:
        log_warning(f"[pdf_parser_v2] PyMuPDF fallback failed: {e}")
    return pages_text


# -------------------------------------------------------------------
# PAGE MERGER
# -------------------------------------------------------------------
def merge_page_text(pages: List[str]) -> str:
    """Concatenates pages with structured separation markers."""
    return "\n\n---PAGE BREAK---\n\n".join(pages)


# -------------------------------------------------------------------
# PARALLEL PAGE EXTRACTION (ASYNC)
# -------------------------------------------------------------------
async def parallel_extract_pdf(file_path: str) -> List[str]:
    """Runs extraction concurrently using both pdfplumber and pymupdf for redundancy."""
    # use get_running_loop() — safer inside an active event loop
    loop = asyncio.get_running_loop()
    plumber_task = loop.run_in_executor(None, lambda: extract_with_pdfplumber(file_path))
    pymupdf_task = loop.run_in_executor(None, lambda: extract_with_pymupdf(file_path))

    results = await asyncio.gather(plumber_task, pymupdf_task, return_exceptions=True)
    plumber_pages, pymupdf_pages = results

    # Prefer pdfplumber if sufficient text found, otherwise use PyMuPDF fallback
    if plumber_pages and sum(len(p) for p in plumber_pages) > 200:
        return plumber_pages
    if pymupdf_pages:
        return pymupdf_pages
    raise ValueError(f"[pdf_parser_v2] No extractable text found in: {file_path}")


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Hybrid PDF parser with async concurrency and optional LLM normalization.
    Steps:
      1. Attempt parallel extraction (pdfplumber + PyMuPDF)
      2. Merge and clean results
      3. (Optional) Normalize with LLM
      4. Return standardized ingestion output
    """

    log_info(f"[pdf_parser_v2] Reading PDF: {file_path}")

    pages_text = await parallel_extract_pdf(file_path)

    if not pages_text:
        raise ValueError(f"[pdf_parser_v2] Empty extraction result: {file_path}")

    combined = merge_page_text(pages_text)
    cleaned = clean_text(combined)
    normalized_text = cleaned

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization
    # ----------------------------------------------------------------
    if is_llm_enabled():
        try:
            log_info(f"[pdf_parser_v2] Sending {len(pages_text)} pages for LLM normalization...")
            normalized_results = await rewrite_batch([cleaned])
            if normalized_results:
                normalized_text = normalized_results[0]
                log_info("[pdf_parser_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[pdf_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[pdf_parser_v2] LLM normalization skipped (disabled).")

    # ----------------------------------------------------------------
    # ✅ Return standardized output
    # ----------------------------------------------------------------
    return {
        "raw_text": combined,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "pages": len(pages_text),
        "source_type": "pdf",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "pages": len(pages_text),
            "parser": "pdfplumber + pymupdf hybrid + LLM optional",
            "llm_normalization": is_llm_enabled(),
        },
    }
