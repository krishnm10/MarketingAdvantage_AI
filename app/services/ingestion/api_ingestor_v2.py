# api_ingestor_v2.py — Production-Ready API Ingestion Layer
# Minimal safe patches:
#  - use httpx.AsyncClient (async) instead of blocking requests
#  - parse XML string locally (parse_xml(file_path) was async/file-path oriented)
# All other logic (LLM flow, chunking, metadata) preserved.
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import httpx
import xml.etree.ElementTree as ET

from app.services.ingestion.json_parser_v2 import flatten_json, stringify_flattened
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ LLM normalization
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global flag import


# -------------------------------------------------------------------
# Local parser-level LLM toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for API ingestion only
# False → Force disable LLM normalization
# None  → Inherit from global ENABLE_LLM_NORMALIZATION
LOCAL_LLM_TOGGLE = None


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# HTTP LAYER (async httpx)
# -------------------------------------------------------------------
async def http_get(
    url: str, headers: Optional[Dict[str, str]] = None, retries: int = 3, timeout: int = 20
) -> httpx.Response:
    """Async GET with retry and logging using httpx.AsyncClient."""
    attempt = 0
    while attempt < retries:
        try:
            log_info(f"[api_ingestor_v2] GET attempt {attempt+1}/{retries}: {url}")
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response
        except Exception as e:
            log_warning(f"[api_ingestor_v2] Request error: {e}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"[api_ingestor_v2] Failed to fetch {url} after {retries} retries")


# -------------------------------------------------------------------
# FORMAT DETECTION
# -------------------------------------------------------------------
def detect_api_format(response: httpx.Response) -> str:
    """Detects response content format."""
    ctype = response.headers.get("content-type", "").lower()
    if "application/json" in ctype or ctype.endswith("+json"):
        return "json"
    if "xml" in ctype or "html" in ctype:
        return "xml"
    return "text"


# -------------------------------------------------------------------
# XML STRING -> dict helper (minimal and safe)
# -------------------------------------------------------------------
def _xml_element_to_dict(elem: ET.Element) -> Any:
    """Recursively convert an ElementTree element to nested dict/list structure."""
    # If element has no children, return its text
    children = list(elem)
    if not children:
        text = elem.text.strip() if elem.text and elem.text.strip() else ""
        # include attributes if present
        if elem.attrib:
            d = {f"@{k}": v for k, v in elem.attrib.items()}
            if text:
                d["#text"] = text
            return d
        return text

    result = {}
    # include attributes
    for k, v in elem.attrib.items():
        result[f"@{k}"] = v

    # group children by tag to keep lists where appropriate
    tag_map = {}
    for child in children:
        child_val = _xml_element_to_dict(child)
        tag_map.setdefault(child.tag, []).append(child_val)

    for tag, vals in tag_map.items():
        # if only one child with that tag, store single value
        if len(vals) == 1:
            result[tag] = vals[0]
        else:
            result[tag] = vals
    return result


def _parse_xml_string_to_dict(xml_str: str) -> Dict[str, Any]:
    """Parse XML string into a python dict structure suitable for flatten_json."""
    try:
        root = ET.fromstring(xml_str)
    except Exception as e:
        raise ValueError(f"[api_ingestor_v2] XML parsing failed: {e}")
    return {root.tag: _xml_element_to_dict(root)}


# -------------------------------------------------------------------
# PARSERS
# -------------------------------------------------------------------
def parse_api_json(body: Any) -> Dict[str, Any]:
    """Parse and flatten JSON API data."""
    flat = flatten_json(body)
    text = stringify_flattened(flat)
    cleaned = clean_text(text)
    return {
        "flat_dict": flat,
        "text": text,
        "cleaned_text": cleaned,
        "parser": "json",
    }


def parse_api_xml(content: str) -> Dict[str, Any]:
    """Parse XML API data safely from XML string content."""
    xml_dict = _parse_xml_string_to_dict(content)
    flat = flatten_json(xml_dict)
    text = stringify_flattened(flat)
    cleaned = clean_text(text)
    return {
        "flat_dict": flat,
        "text": text,
        "cleaned_text": cleaned,
        "parser": "xml",
    }


def parse_api_text(content: str) -> Dict[str, Any]:
    """Parse plain text API responses."""
    cleaned = clean_text(content)
    return {
        "text": content,
        "cleaned_text": cleaned,
        "parser": "text",
    }


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
async def ingest_api_data(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    db_session=None,  # ✅ Added safely for file_router_v2 compatibility
) -> Dict[str, Any]:
    """
    Unified async API ingestion pipeline.
    Steps:
      1. GET request
      2. Detect format (JSON/XML/Text)
      3. Parse + clean content
      4. (Optional) Normalize with LLM
      5. Return structured ingestion payload with chunks
    """

    log_info(f"[api_ingestor_v2] Fetching API: {url}")

    response = await http_get(url, headers)
    fmt = detect_api_format(response)
    raw_content = response.text

    # ----------------------------------------------------------------
    # ✅ Step 1: Parse according to format
    # ----------------------------------------------------------------
    if fmt == "json":
        try:
            body = response.json()
            parsed = parse_api_json(body)
        except Exception as e:
            raise ValueError(f"[api_ingestor_v2] JSON parsing failed: {e}")
    elif fmt == "xml":
        parsed = parse_api_xml(raw_content)
    else:
        parsed = parse_api_text(raw_content)

    # ----------------------------------------------------------------
    # ✅ Step 2: Optional LLM Normalization
    # ----------------------------------------------------------------
    cleaned_text = parsed.get("cleaned_text", "")
    normalized_text = cleaned_text

    if is_llm_enabled() and cleaned_text:
        try:
            log_info("[api_ingestor_v2] Sending API content for LLM normalization...")
            normalized_texts = await rewrite_batch([cleaned_text])
            if normalized_texts:
                normalized_text = normalized_texts[0]
                log_info("[api_ingestor_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[api_ingestor_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[api_ingestor_v2] LLM normalization skipped (disabled).")

    # ----------------------------------------------------------------
    # ✅ Step 3: Chunk generation for ingestion_service_v2 compatibility
    # ----------------------------------------------------------------
    chunks: List[Dict[str, Any]] = []
    for i, para in enumerate(normalized_text.split("\n")):
        if para.strip():
            # ✅ Stable SHA256 hash for deduplication
            semantic_hash = hashlib.sha256(para.strip().encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "chunk_id": f"{url}_chunk_{i}",
                    "text": para.strip(),
                    "cleaned_text": para.strip(),
                    "semantic_hash": semantic_hash,
                    "source_type": "api",
                    "metadata": {
                        "url": url,
                        "paragraph_index": i,
                        "parser": parsed.get("parser", fmt),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
            )

    log_info(f"[api_ingestor_v2] ✅ Generated {len(chunks)} structured chunks for ingestion.")

    # ----------------------------------------------------------------
    # ✅ Step 4: Final return (consistent with file_router_v2 & ingestion_service_v2)
    # ----------------------------------------------------------------
    return {
        "raw": raw_content,
        "parsed": parsed,
        "normalized_text": normalized_text,
        "chunks": chunks,
        "source_type": "api",
        "metadata": {
            "url": url,
            "status_code": response.status_code,
            "content_type": fmt,
            "parser": parsed.get("parser"),
            "content_length": len(raw_content),
            "retrieved_at": datetime.utcnow().isoformat(),
            "llm_normalization": is_llm_enabled(),
        },
    }
