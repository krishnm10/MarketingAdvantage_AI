"""
LLM normalization service.

This is a SAFE STUB.
- No external calls yet
- No Ollama / OpenAI dependency
- Can be swapped later without changing APIs
"""

async def run_llm_normalization(text: str, mode: str) -> str:
    """
    Normalize text using LLM.

    mode:
      - factual  → clean, neutral, no creativity
      - creative → rewrite with light creativity
    """

    text = text.strip()

    if mode == "factual":
        # No creativity, just normalization
        return text

    if mode == "creative":
        # Placeholder for creative rewrite
        return f"{text}"

    return text
