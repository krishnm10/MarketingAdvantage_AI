# =============================================
# xml_parser_v2.py — XML Parser (Production-Ready)
# Now includes unified LLM toggle control (local + global)
# Fully compatible with ingestion_v2 pipeline and row_segmenter_v2
# =============================================

import xml.etree.ElementTree as ET
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import pandas as pd

from app.utils.logger import log_info, log_warning
from app.utils.text_cleaner_v2 import clean_text
from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.llm_rewriter import rewrite_batch
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
# XML PARSING UTILITIES
# -------------------------------------------------------------------
def _flatten_xml(element: ET.Element, parent_path: str = "") -> Dict[str, Any]:
    """Recursively flattens an XML element tree into a dict-like structure."""
    data = {}
    tag_path = f"{parent_path}.{element.tag}" if parent_path else element.tag

    # Include text
    if element.text and element.text.strip():
        data[tag_path] = element.text.strip()

    # Include attributes
    for attr, value in element.attrib.items():
        data[f"{tag_path}.@{attr}"] = value

    # Recursively handle children
    for child in element:
        data.update(_flatten_xml(child, tag_path))

    return data


# -------------------------------------------------------------------
# SAFE LOADER
# -------------------------------------------------------------------
async def _try_load_xml(file_path: str) -> ET.ElementTree:
    """Asynchronously load an XML file and parse safely."""
    try:
        return await asyncio.to_thread(ET.parse, file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse XML: {e}")


# -------------------------------------------------------------------
# CONVERSION HELPERS
# -------------------------------------------------------------------
def _to_dataframe(flattened_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts flattened XML records into a pandas DataFrame for segmentation."""
    if not flattened_records:
        return pd.DataFrame()
    return pd.DataFrame(flattened_records)


# -------------------------------------------------------------------
# MAIN ASYNC PARSER
# -------------------------------------------------------------------
async def parse_xml(
    file_path: str,
    file_id: Optional[str] = None,
    business_id: Optional[str] = None,
    db_session=None,
) -> Dict[str, Any]:
    """
    Parses an XML document into structured chunks.
    Steps:
      1. Parse XML
      2. Flatten elements into key-value pairs
      3. Convert to DataFrame
      4. Segment rows with row_segmenter_v2
      5. Optionally normalize text via LLM
    """

    log_info(f"[xml_parser_v2] Parsing XML file: {file_path}")

    try:
        tree = await _try_load_xml(file_path)
        root = tree.getroot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XML parse error: {e}")

    flattened_records: List[Dict[str, Any]] = []

    # Convert XML tree to flat dicts per element
    for child in root:
        flattened_records.append(_flatten_xml(child))

    if not flattened_records:
        log_warning(f"[xml_parser_v2] No data extracted from XML: {file_path}")
        raise HTTPException(status_code=400, detail="Empty or invalid XML structure")

    df = _to_dataframe(flattened_records)
    log_info(f"[xml_parser_v2] Extracted {len(df)} flattened records from XML.")

    # Step 1: Row segmentation (dedup-aware)
    try:
        chunks = await parse_dataframe_rows(
            df=df,
            file_id=file_id,
            source_type="xml",
            db_session=db_session,
            business_id=business_id,
        )
    except Exception as e:
        log_warning(f"[xml_parser_v2] Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks generated from XML")

    # Step 2: Optional LLM normalization
    if is_llm_enabled():
        try:
            texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
            if not texts:
                log_warning("[xml_parser_v2] No valid text chunks to normalize.")
            else:
                log_info(f"[xml_parser_v2] Sending {len(texts)} chunks for LLM normalization...")
                normalized_texts = await rewrite_batch(texts)

                for i, chunk in enumerate(chunks):
                    cleaned_text = clean_text(chunk.get("text", ""))
                    chunk["cleaned_text"] = cleaned_text
                    if i < len(normalized_texts):
                        chunk["normalized_text"] = normalized_texts[i]
        except Exception as e:
            log_warning(f"[xml_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[xml_parser_v2] LLM normalization skipped (disabled).")

    log_info(f"[xml_parser_v2] ✅ Parsed {len(chunks)} chunks successfully.")

    # Step 3: Return standardized result
    return {
        "chunks": chunks,
        "source_type": "xml",
        "metadata": {
            "file_name": file_path.split("/")[-1],
            "parser": "xml_v2 (ETree + Ollama)",
            "records": len(flattened_records),
        },
    }
