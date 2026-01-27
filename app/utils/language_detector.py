# app/utils/language_detector.py

import re

# Basic language patterns (expandable)
LANG_PATTERNS = {
    "en": r"[a-zA-Z]",
    "hi": r"[\u0900-\u097F]",
    "te": r"[\u0C00-\u0C7F]",
    "ta": r"[\u0B80-\u0BFF]",
    "kn": r"[\u0C80-\u0CFF]"
}


def detect_language(text: str) -> str:
    """
    Hybrid detection: lightweight & fast.
    Returns language code (default = 'en').
    """

    if not text:
        return "unknown"

    scores = {}

    for lang, pattern in LANG_PATTERNS.items():
        matches = re.findall(pattern, text)
        scores[lang] = len(matches)

    # Pick the language with highest probability
    best_lang = max(scores, key=scores.get)

    # Fallback if no meaningful match
    return best_lang if scores[best_lang] > 5 else "en"
