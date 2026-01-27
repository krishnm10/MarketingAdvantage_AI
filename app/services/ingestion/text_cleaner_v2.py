# =============================================
# text_cleaner_v2.py — Advanced Text Normalizer (Production-Ready)
# Optimized: precompiled regexes + small URL handling robustness
# =============================================

import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional

# -------------------------------------------------------------------
# Precompiled regexes (module-level for performance)
# -------------------------------------------------------------------
# Emoji / symbol ranges (kept as a compiled pattern)
_EMOJI_PATTERN = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "]+",
    flags=re.UNICODE,
)

# URL pattern: match http(s):// or www. and stop at whitespace or common trailing punctuation
_URL_PATTERN = re.compile(r"""(?xi)
    \b
    (?:https?://|www\.)        # scheme or www
    [^\s<>"'(){}\[\]]+         # run of allowed URL chars (stops before brackets and quotes)
""")

# Control characters (invisible)
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x1F\x7F]+")

# Special symbol cleanup (bullets, arrows, repeated punctuation)
_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[•→←↔✔✖★☆●■□◆◇✓✗]+")

# Repeated punctuation limiter
_REPEATED_PUNCT_PATTERN = re.compile(r"([.,!?])\1{2,}")

# Whitespace collapse
_WHITESPACE_PATTERN = re.compile(r"\s+")


# -------------------------------------------------------------------
# HTML STRIPPING
# -------------------------------------------------------------------
def strip_html(text: Optional[str]) -> str:
    """Remove HTML tags safely using BeautifulSoup."""
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return text or ""


# -------------------------------------------------------------------
# UNICODE NORMALIZATION
# -------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Normalize text to NFKC (safe canonical form)."""
    return unicodedata.normalize("NFKC", text)


# -------------------------------------------------------------------
# EMOJI + SYMBOL REMOVAL
# -------------------------------------------------------------------
def remove_emojis(text: str) -> str:
    """Remove emoji/symbol ranges."""
    return _EMOJI_PATTERN.sub("", text)


# -------------------------------------------------------------------
# URL + CONTROL CHARACTER REMOVAL
# -------------------------------------------------------------------
def remove_urls(text: str) -> str:
    """Remove URLs from text. Uses a defensive regex to avoid swallowing trailing punctuation."""
    return _URL_PATTERN.sub("", text)


def remove_control_chars(text: str) -> str:
    """Remove invisible or control characters."""
    # convert sequences of control chars to a single space
    return _CONTROL_CHARS_PATTERN.sub(" ", text)


# -------------------------------------------------------------------
# SPECIAL SYMBOL CLEANUP
# -------------------------------------------------------------------
def remove_special_symbols(text: str) -> str:
    """Remove stray bullets, arrows, and reduce repeated punctuation."""
    text = _SPECIAL_SYMBOLS_PATTERN.sub(" ", text)
    text = _REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    return text


# -------------------------------------------------------------------
# WHITESPACE NORMALIZATION
# -------------------------------------------------------------------
def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


# -------------------------------------------------------------------
# MASTER CLEAN FUNCTION
# -------------------------------------------------------------------
def clean_text(text: Optional[str]) -> str:
    """
    Advanced hybrid text cleaner.
    Steps (preserved order):
      1. Strip HTML
      2. Normalize Unicode
      3. Remove emojis
      4. Remove URLs and control chars
      5. Remove special symbols
      6. Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # run each step (same order as original; semantics preserved)
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_control_chars(text)
    text = remove_special_symbols(text)
    text = collapse_whitespace(text)

    return text
