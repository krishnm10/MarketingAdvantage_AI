# app/utils/text_cleaner.py

import re
import unicodedata
from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to a safer form."""
    return unicodedata.normalize("NFKC", text)


def remove_emojis(text: str) -> str:
    """Remove emojis + non-text symbolic glyphs."""
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002700-\U000027BF"  
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Normalize excessive whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Hybrid Cleaner:
    - Strip HTML
    - Normalize unicode
    - Remove emojis
    - Remove control characters
    - Collapse whitespace
    """

    if not text:
        return ""

    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_emojis(text)

    # Remove control chars (tabs, weird unicode)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    text = collapse_whitespace(text)

    return text
