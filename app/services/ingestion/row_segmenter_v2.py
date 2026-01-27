# =============================================
# row_segmenter_v2.py — Structured Data Chunker (Production-Ready)
# Fully aligned with DB schema and IngestionServiceV2
# =============================================

from typing import List, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime

# <<< PATCH: use text_cleaner_v2 (newer version) >>>
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info
from app.services.ingestion.segmenter_v2 import make_chunk_dict

# -------------------------------------------------------------------
# MAIN FUNCTION: Parse structured/tabular data into semantic chunks
# -------------------------------------------------------------------
async def parse_dataframe_rows(
    df: pd.DataFrame,
    file_id: str,
    source_type: str,
    db_session=None,
    business_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Converts structured/tabular data (CSV, Excel, JSON array)
    into dedup-aware semantic chunks compatible with GlobalContentIndex.

    - Flattens each row into human-readable text
    - Cleans and normalizes text
    - Deduplicates using GlobalContentIndex (semantic hash)
    - Returns chunks fully aligned with ingestion DB schema
    """

    chunks: List[Dict[str, Any]] = []

    # compute columns once for efficiency and consistency
    columns = list(df.columns) if df is not None else []

    for row_index, row in df.iterrows():
        row_dict = row.to_dict()

        # Flatten row into readable text string safely (handle dicts/lists)
        parts = []
        for col, value in row_dict.items():
            if pd.isna(value):
                continue
            # convert complex structures to json where possible for readability
            try:
                if isinstance(value, (dict, list)):
                    sval = json.dumps(value, ensure_ascii=False)
                else:
                    sval = str(value)
            except Exception:
                sval = str(value)
            parts.append(f"{col}: {sval}")

        row_text = " | ".join(parts)

        if not row_text.strip():
            continue

        # Use make_chunk_dict which internally cleans text; pass raw row_text to keep single-source cleaning
        chunk_data = await make_chunk_dict(
            row_text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )

        # Defensive: ensure chunk_data contains expected keys
        if not chunk_data or not isinstance(chunk_data, dict):
            continue

        chunks.append(
            {
                "file_id": file_id,
                "source_type": source_type,
                "row_index": row_index,
                "columns": columns,

                # Semantic core
                "text": chunk_data.get("text"),
                "cleaned_text": chunk_data.get("cleaned_text"),
                "tokens": chunk_data.get("tokens"),
                "semantic_hash": chunk_data.get("semantic_hash"),
                "confidence": chunk_data.get("confidence"),

                # GlobalContentIndex link
                "global_content_id": chunk_data.get("global_content_id"),

                # Metadata (aligned with DB meta_data JSONB field)
                "metadata": {
                    "row_index": row_index,
                    "columns": columns,
                    "raw_row": row_dict,
                    "dedup": {
                        "semantic_hash": chunk_data.get("semantic_hash"),
                        "global_content_id": chunk_data.get("global_content_id"),
                    },
                },
            }
        )

    log_info(f"[row_segmenter_v2] Produced {len(chunks)} dedup-aware structured chunks")

    return chunks
