# =============================================
# csv_parser_v2.py — CSV Parser (Production-Ready)
# Now includes unified LLM toggle control (global + local)
# Fully compatible with ingestion_v2 + row_segmenter_v2 pipeline
# =============================================

import pandas as pd
import asyncio
import os
from typing import Dict, Any, Optional, List
from fastapi import HTTPException

from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ LLM Import
from app.utils.logger import log_info, log_warning
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global toggle import


# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for this parser
# False → Force disable LLM normalization for this parser
# None  → Inherit from global flag
LOCAL_LLM_TOGGLE = None

def is_llm_enabled() -> bool:
    """Returns the effective LLM toggle for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# FILE READER
# -------------------------------------------------------------------
def try_read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads CSV safely with utf-8 and fallback encodings.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        return df
    except Exception as e:
        raise ValueError(f"[csv_parser_v2] Failed to read CSV file: {file_path}: {e}")


# -------------------------------------------------------------------
# NORMALIZER
# -------------------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame before segmentation:
      - Strip column names
      - Fill NaNs with None
      - Trim whitespace from string columns
    """
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = df.where(pd.notnull(df), None)
    return df


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_csv(
    file_path: str,
    file_id: Optional[str] = None,
    business_id: Optional[str] = None,
    db_session=None,
) -> Dict[str, Any]:
    """
    CSV parser with LLM-enhanced normalization and dedup-aware chunking.
    Steps:
      1. Load file
      2. Normalize data
      3. Segment rows
      4. Optionally LLM normalize each chunk
      5. Return structured response
    """
    log_info(f"[csv_parser_v2] Reading CSV: {file_path}")

    # Read safely via background thread (non-blocking)
    try:
        df = await asyncio.to_thread(try_read_csv, file_path)
    except Exception as e:
        log_warning(f"[csv_parser_v2] Error reading CSV: {e}")
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"CSV file is empty: {file_path}")

    df = normalize_dataframe(df)
    log_info(f"[csv_parser_v2] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Step 1: Row segmentation (dedup-aware)
    try:
        parsed_chunks = await parse_dataframe_rows(
            df=df,
            file_id=file_id,
            source_type="csv",
            db_session=db_session,
            business_id=business_id,
        )
    except Exception as e:
        log_warning(f"[csv_parser_v2] Chunk segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"CSV segmentation failed: {e}")

    if not parsed_chunks:
        log_warning(f"[csv_parser_v2] No chunks extracted from {file_path}")
        return {
            "chunks": [],
            "source_type": "csv",
            "metadata": {"file_name": os.path.basename(file_path), "empty": True},
        }

    # Step 2: Optional LLM normalization (toggle-aware)
    if is_llm_enabled():
        try:
            texts = [chunk.get("text", "") for chunk in parsed_chunks if chunk.get("text")]
            if texts:
                log_info(f"[csv_parser_v2] Sending {len(texts)} chunks for LLM normalization...")
                normalized_texts = await rewrite_batch(texts)
                for i, chunk in enumerate(parsed_chunks):
                    if i < len(normalized_texts):
                        chunk["normalized_text"] = normalized_texts[i]
            else:
                log_warning("[csv_parser_v2] No valid text chunks to normalize.")
        except Exception as e:
            log_warning(f"[csv_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[csv_parser_v2] LLM normalization skipped (disabled).")

    # Step 3: Return final payload
    log_info(f"[csv_parser_v2] ✅ Parsed {len(parsed_chunks)} chunks successfully.")
    return {
        "chunks": parsed_chunks,
        "source_type": "csv",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "rows": len(df),
            "columns": list(df.columns),
            "parser": "csv_v2 (pandas)",
        },
    }
