# web_scraper_v2.py — Intelligent Web Ingestion Pipeline
# Enhanced for ingestion_v2 architecture (async-safe + content normalization)
# Includes unified LLM toggle control and safe chunked normalization
# =============================================

import httpx
import asyncio
import hashlib
from bs4 import BeautifulSoup
from readability import Document
from typing import Dict, Any, Optional, List

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global toggle import


# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for this parser
# False → Force disable LLM normalization for this parser
# None  → Inherit from global flag
LOCAL_LLM_TOGGLE = None


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# HTTP FETCHER (ASYNC-SAFE)
# -------------------------------------------------------------------
async def fetch_html(url: str, retries: int = 3, timeout: int = 20) -> str:
    """Fetch HTML content with retries using httpx.AsyncClient and async backoff."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }

    attempt = 0
    while attempt < retries:
        try:
            log_info(f"[web_scraper_v2] GET attempt {attempt+1}/{retries}: {url}")
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            log_warning(f"[web_scraper_v2] Retry {attempt+1}/{retries} failed: {e}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"[web_scraper_v2] Failed to fetch {url} after {retries} attempts")


# -------------------------------------------------------------------
# MAIN EXTRACTION PIPELINE
# -------------------------------------------------------------------
async def extract_main_text(html: str) -> Dict[str, Any]:
    """Extracts main article content using readability-lxml; falls back to BeautifulSoup."""
    try:
        doc = Document(html)
        title = doc.title() or ""
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        extracted_text = soup.get_text(separator=" ", strip=True)
        return {"title": title, "content_raw": extracted_text}
    except Exception as e:
        log_warning(f"[web_scraper_v2] Readability failed: {e} — falling back to BeautifulSoup.")
        return fallback_extract(html)


# -------------------------------------------------------------------
# FALLBACK EXTRACTION
# -------------------------------------------------------------------
def fallback_extract(html: str) -> Dict[str, Any]:
    """Manual text extraction fallback (removes scripts/styles)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    return {"title": "", "content_raw": text}


# -------------------------------------------------------------------
# WEB INGESTION PIPELINE (ASYNC)
# -------------------------------------------------------------------
async def ingest_webpage(url: str, db_session=None) -> Dict[str, Any]:
    """
    Complete async ingestion pipeline for web URLs.
    Steps:
      1. Fetch HTML (async + retry)
      2. Extract main readable text
      3. Clean content
      4. (Optional) Normalize via LLM
      5. Package for ingestion_v2 service
    """

    log_info(f"[web_scraper_v2] Scraping webpage: {url}")

    html = await fetch_html(url)
    extracted = await extract_main_text(html)

    raw_text = extracted.get("content_raw", "")
    cleaned = clean_text(raw_text)

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization (with chunk safety)
    # ----------------------------------------------------------------
    if is_llm_enabled():
        try:
            log_info("[web_scraper_v2] Sending webpage content for LLM normalization...")

            # Split large text to avoid LLM context overflow
            CHUNK_SIZE = 2000
            text_chunks = [cleaned[i:i + CHUNK_SIZE] for i in range(0, len(cleaned), CHUNK_SIZE)]
            log_info(f"[web_scraper_v2] Splitting text into {len(text_chunks)} chunks for safe LLM rewrite")

            normalized_chunks = await rewrite_batch(text_chunks)
            normalized_text = " ".join(normalized_chunks)
            log_info("[web_scraper_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[web_scraper_v2] ⚠️ LLM normalization failed: {e}")
            normalized_text = cleaned
    else:
        log_info("[web_scraper_v2] LLM normalization skipped (disabled).")
        normalized_text = cleaned

    # ----------------------------------------------------------------
    # ✅ New: Chunk preparation (for IngestionServiceV2 compatibility)
    # ----------------------------------------------------------------
    chunks = []
    for i, paragraph in enumerate(normalized_text.split("\n")):
        if paragraph.strip():
            # ✅ Stable semantic hash for deduplication (persistent across runs)
            stable_hash = hashlib.sha256(paragraph.strip().encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "chunk_id": f"{url}_chunk_{i}",
                    "text": paragraph.strip(),
                    "cleaned_text": paragraph.strip(),
                    "semantic_hash": stable_hash,
                    "source_type": "web",
                    "metadata": {
                        "url": url,
                        "title": extracted.get("title", ""),
                        "paragraph_index": i,
                    },
                }
            )

    log_info(f"[web_scraper_v2] ✅ Generated {len(chunks)} structured chunks for ingestion.")

    # ----------------------------------------------------------------
    # ✅ Return structured result (compatible with file_router_v2 + ingestion_service_v2)
    # ----------------------------------------------------------------
    return {
        "raw_html": html,
        "content_raw": raw_text,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "chunks": chunks,
        "source_type": "web",
        "metadata": {
            "url": url,
            "title": extracted.get("title", ""),
            "length_chars": len(raw_text),
            "parser": "web_scraper_v2 (readability + bs4 + LLM chunked optional)",
        },
    }
