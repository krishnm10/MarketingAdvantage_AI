# text_parser_v2.py — Advanced Plain Text Parser (Production-Ready)
# Async-safe and fully aligned with ingestion_v2 pipeline
# Now includes unified LLM normalization toggle (global + local)

from typing import Dict, Any
import chardet
import asyncio
import os

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ Added for LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global toggle

# -------------------------------------------------------------------
# Local parser-level LLM toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for text parser
# False → Force disable LLM normalization
# None  → Inherit from global ENABLE_LLM_NORMALIZATION
LOCAL_LLM_TOGGLE = None


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# ENCODING DETECTION
# -------------------------------------------------------------------
def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet with safe fallback."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
            result = chardet.detect(raw)
            return result.get("encoding", "utf-8") or "utf-8"
    except Exception as e:
        log_warning(f"[text_parser_v2] Encoding detection failed for {file_path}: {e}")
        return "utf-8"


# -------------------------------------------------------------------
# FILE READER
# -------------------------------------------------------------------
def read_text_file(file_path: str) -> str:
    """Reads text file safely using detected encoding."""
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"[text_parser_v2] Failed to read {file_path}: {e}")


# -------------------------------------------------------------------
# ASYNC WRAPPER
# -------------------------------------------------------------------
async def async_read_text_file(file_path: str) -> str:
    """Async-compatible file reader for ingestion_v2 pipeline."""
    # use get_running_loop for safety when called within an active event loop
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: read_text_file(file_path))


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_text(file_path: str) -> Dict[str, Any]:
    """
    Asynchronous text parser for ingestion_v2.
    Steps:
      1. Detect and read text file safely
      2. Clean and normalize text
      3. (Optional) Rewrite via LLM if enabled
      4. Return standardized ingestion-ready dict
    """

    log_info(f"[text_parser_v2] Reading text file: {file_path}")

    raw = await async_read_text_file(file_path)

    if not raw.strip():
        raise ValueError(f"[text_parser_v2] Text file is empty: {file_path}")

    cleaned = clean_text(raw)
    normalized_text = cleaned

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization
    # ----------------------------------------------------------------
    if is_llm_enabled():
        try:
            log_info("[text_parser_v2] Sending text for LLM normalization...")
            normalized_results = await rewrite_batch([cleaned])
            if normalized_results:
                normalized_text = normalized_results[0]
                log_info("[text_parser_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[text_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[text_parser_v2] LLM normalization skipped (disabled).")

    # ----------------------------------------------------------------
    # ✅ Return standardized ingestion output
    # ----------------------------------------------------------------
    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "source_type": "text",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "parser": "text_v2",
            "encoding": detect_encoding(file_path),
            "length_chars": len(raw),
            "length_words": len(raw.split()),
            "llm_normalization": is_llm_enabled(),
        },
    }
