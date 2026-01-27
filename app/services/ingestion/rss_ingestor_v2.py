# =============================================
# rss_ingestor_v2.py — RSS Feed Ingestor (Production-Ready)
# Fully aligned with ingestion_v2 architecture and GlobalContentIndexV2
# Unified global/local LLM normalization toggle + stable dedup hashes
# =============================================

import asyncio
import feedparser
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.utils.logger import log_info, log_warning
from app.utils.text_cleaner_v2 import clean_text
from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.llm_rewriter import rewrite_batch
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION


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
# ASYNC FEED FETCHER
# -------------------------------------------------------------------
async def fetch_rss_feed(source_url: str) -> feedparser.FeedParserDict:
    """Fetches and parses RSS feed asynchronously."""
    try:
        log_info(f"[rss_ingestor_v2] Fetching RSS feed: {source_url}")
        return await asyncio.to_thread(feedparser.parse, source_url)
    except Exception as e:
        log_warning(f"[rss_ingestor_v2] Failed to fetch RSS feed: {e}")
        raise


# -------------------------------------------------------------------
# CLEAN + TRANSFORM RSS ENTRIES
# -------------------------------------------------------------------
def extract_feed_entries(feed: feedparser.FeedParserDict) -> List[Dict[str, Any]]:
    """Extracts and cleans entries from parsed RSS feed."""
    if not feed or not feed.entries:
        log_warning("[rss_ingestor_v2] No entries found in feed.")
        return []

    cleaned_entries = []

    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        link = entry.get("link", "")
        published = entry.get("published", "") or entry.get("updated", "")

        combined_text = f"{title}. {summary}"
        cleaned_text = clean_text(combined_text)

        # ✅ Stable semantic hash for deduplication
        semantic_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()

        cleaned_entries.append(
            {
                "title": title,
                "summary": summary,
                "cleaned_text": cleaned_text,
                "semantic_hash": semantic_hash,
                "link": link,
                "published": published,
                "raw_text": combined_text,
            }
        )

    log_info(f"[rss_ingestor_v2] Extracted {len(cleaned_entries)} feed entries.")
    return cleaned_entries


# -------------------------------------------------------------------
# MAIN RSS PARSER
# -------------------------------------------------------------------
async def parse_rss(
    source_url: str,
    file_id: Optional[str] = None,
    business_id: Optional[str] = None,
    db_session=None,
) -> Dict[str, Any]:
    """
    Parses an RSS feed, cleans its entries, segments them into chunks, and returns structured data.
    Steps:
      1. Fetch RSS feed
      2. Extract + clean entries
      3. (Optional) Normalize via LLM
      4. Convert to DataFrame
      5. Segment via row_segmenter_v2
      6. Return ingestion-ready structure
    """

    log_info(f"[rss_ingestor_v2] Parsing RSS source: {source_url}")

    feed = await fetch_rss_feed(source_url)
    entries = extract_feed_entries(feed)

    if not entries:
        log_warning("[rss_ingestor_v2] No valid entries found for ingestion.")
        return {
            "chunks": [],
            "source_type": "rss",
            "metadata": {
                "url": source_url,
                "entry_count": 0,
                "parser": "rss_v2 (clean only)",
                "file_name": source_url,
            },
        }

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization
    # ----------------------------------------------------------------
    if is_llm_enabled():
        texts = [entry["cleaned_text"] for entry in entries if entry.get("cleaned_text")]
        if texts:
            log_info(f"[rss_ingestor_v2] Sending {len(texts)} entries for LLM normalization...")
            try:
                normalized_texts = await rewrite_batch(texts)
                for i, entry in enumerate(entries):
                    if i < len(normalized_texts):
                        entry["normalized_text"] = normalized_texts[i]
            except Exception as e:
                log_warning(f"[rss_ingestor_v2] ⚠️ LLM normalization failed: {e}")
        else:
            log_warning("[rss_ingestor_v2] No cleaned text available for LLM normalization.")
    else:
        log_info("[rss_ingestor_v2] LLM normalization skipped (disabled).")

    # ----------------------------------------------------------------
    # ✅ Continue with segmentation
    # ----------------------------------------------------------------
    df = pd.DataFrame(entries)

    # <<< SAFETY PATCH: ensure all DataFrame cells are strings to avoid non-str concatenation errors downstream >>>
    df = df.astype(str)

    try:
        chunks = await parse_dataframe_rows(
            df=df,
            file_id=file_id,
            source_type="rss",
            db_session=db_session,
            business_id=business_id,
        )
    except Exception as e:
        log_warning(f"[rss_ingestor_v2] Segmentation failed: {e}")
        chunks = []

    log_info(f"[rss_ingestor_v2] ✅ Parsed {len(chunks)} total chunks from RSS feed.")

    # ----------------------------------------------------------------
    # ✅ Return standardized structured output (aligned with ingestion_service_v2)
    # ----------------------------------------------------------------
    return {
        "chunks": chunks,
        "source_type": "rss",
        "metadata": {
            "url": source_url,
            "entry_count": len(entries),
            "parser": "rss_v2 (feedparser + cleaner + LLM optional)",
            "retrieved_at": datetime.utcnow().isoformat(),
            "llm_normalization": is_llm_enabled(),
            "file_name": source_url,
        },
    }
