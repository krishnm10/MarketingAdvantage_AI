# =============================================
# json_parser_v2.py — Fully Async JSON Parser with Optional LLM Normalization
# Production-ready, compatible with ingestion_service_v2 and row_segmenter_v2
# =============================================

import json
from typing import Any, Dict, List, Union, Optional
import aiofiles
import asyncio
import os  # <<< ADDED

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_text, rewrite_batch
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global flag import


# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for this parser
# False → Force disable LLM normalization for this parser
# None  → Inherit from global flag
LOCAL_LLM_TOGGLE = None

def is_llm_enabled() -> bool:
    """Determine if LLM normalization should be used for JSON parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# ASYNC JSON FLATTENING
# -------------------------------------------------------------------
def flatten_json(obj: Union[Dict[str, Any], List[Any]], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten deeply nested JSON structures."""
    items = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            new_key = f"{parent_key}{sep}{index}" if parent_key else str(index)
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value
    return items


# -------------------------------------------------------------------
# ASYNC FILE LOADER
# -------------------------------------------------------------------
async def try_load_json(file_path: str) -> Any:
    """Asynchronously loads a JSON file with fallback encodings."""
    for enc in ["utf-8", "utf-16", "latin-1", "ISO-8859-1"]:
        try:
            async with aiofiles.open(file_path, mode="r", encoding=enc) as f:
                data = await f.read()
                return json.loads(data)
        except Exception as e:
            log_warning(f"[json_parser_v2] Failed {enc} decode: {e}")
            continue
    raise ValueError(f"[json_parser_v2] Unable to parse JSON file: {file_path}")


# -------------------------------------------------------------------
# TEXT CONVERSION
# -------------------------------------------------------------------
def stringify_flattened(flat: Dict[str, Any]) -> str:
    """Converts flattened JSON dictionary into readable text for semantic ingestion."""
    parts = []
    for key, value in flat.items():
        key = str(key).strip()
        value = str(value).replace("\n", " ").strip()
        if value:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


# -------------------------------------------------------------------
# MAIN ASYNC PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_json(file_path: str) -> Dict[str, Any]:
    """Full async JSON ingestion process for ingestion_v2 with optional LLM normalization."""

    log_info(f"[json_parser_v2] Parsing JSON file: {file_path}")

    # Step 1: Load JSON asynchronously
    json_data = await try_load_json(file_path)
    flat = flatten_json(json_data)

    if not flat:
        raise ValueError(f"[json_parser_v2] No readable content extracted from {file_path}")

    # Step 2: Convert and clean
    text = stringify_flattened(flat)
    cleaned = clean_text(text)

    normalized_text = cleaned  # default fallback

    # Step 3: Optional LLM normalization
    if is_llm_enabled():
        try:
            if len(cleaned) < 3000:
                normalized_text = await rewrite_text(cleaned)
            else:
                chunks = [cleaned[i:i+3000] for i in range(0, len(cleaned), 3000)]
                results = await rewrite_batch(chunks)
                normalized_text = " ".join(results)
            log_info(f"[json_parser_v2] ✅ LLM normalization complete for {file_path}")
        except Exception as e:
            log_warning(f"[json_parser_v2] ⚠️ LLM normalization failed: {e}")
            normalized_text = cleaned
    else:
        log_info("[json_parser_v2] LLM normalization skipped (disabled).")

    # Step 4: Return standardized ingestion payload
    return {
        "flat_dict": flat,
        "text": text,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "source_type": "json",
        "metadata": {
            "file_name": os.path.basename(file_path),  # <<< PATCHED
            "parser": "json_v2_async_llm_toggle",
            "keys": list(flat.keys()),
            "total_fields": len(flat),
        },
    }


# -------------------------------------------------------------------
# TEST ENTRYPOINT (optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        result = await parse_json("./sample.json")
        print(json.dumps(result, indent=2))

    asyncio.run(_test())
