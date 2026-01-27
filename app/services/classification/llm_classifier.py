# services/classification/llm_classifier.py

import json
import subprocess
from typing import Dict, Any, List

from app.utils.logger import log_info, log_warning


LLM_MODEL = "llama3.1:8b"   # Your local Ollama model


def call_llama(prompt: str) -> str:
    """
    Calls Ollama locally using subprocess to avoid async complexity.
    Ensures safe JSON-only outputs.
    """

    try:
        process = subprocess.Popen(
            ["ollama", "run", LLM_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(prompt, timeout=45)

        if stderr:
            log_warning(f"[llm_classifier] LLM stderr: {stderr}")

        return stdout.strip()

    except Exception as e:
        log_warning(f"[llm_classifier] Error calling LLaMA model: {e}")
        return "{}"


# -----------------------------
#  CLASSIFICATION PROMPT BUILDER
# -----------------------------

def build_prompt(chunk_text: str, embed_candidates: Dict[str, Any]) -> str:
    """
    Builds an extremely strict LLM instruction to produce ONLY JSON.

    Inputs:
        - text chunk
        - embedding-ranked taxonomy candidates
    """

    candidate_list = embed_candidates.get("candidates", [])
    strong_list = embed_candidates.get("strong_candidates", [])

    prompt = f"""
You are an enterprise-grade Business Classification AI.

You must classify text into 3 structured levels:
1. **industry** (root)
2. **sub_industry**
3. **sub_sub_industry**

You also detect synonyms and whether the taxonomy needs admin approval.

You will be given:
- The text from the document chunk
- Top taxonomy candidates generated from embeddings
- Strongly matched categories (high confidence)
- Rules:
    - ALWAYS stay within the provided candidates if possible.
    - Only create a NEW taxonomy if necessary.
    - You MUST return valid and STRICT JSON.

-------------------------
TEXT TO CLASSIFY:
\"\"\"{chunk_text}\"\"\"
-------------------------

EMBEDDING CANDIDATES (Weaker to Stronger):
{json.dumps(candidate_list, indent=2)}

STRONG MATCHES (Very likely):
{json.dumps(strong_list, indent=2)}

-------------------------
RETURN STRICT JSON ONLY:
Format:
{{
  "industry": "string or null",
  "sub_industry": "string or null",
  "sub_sub_industry": "string or null",
  "confidence": float (0-1),
  "synonyms_detected": ["list"],
  "requires_admin_approval": true/false,
  "proposed_new_taxonomy": "name or null",
  "reason": "explain classification logic"
}}
-------------------------
ONLY RETURN JSON. NO TALKING.
"""
    return prompt


# -----------------------------
#   PARSE LLM RESPONSE
# -----------------------------

def safe_json_parse(content: str) -> Dict[str, Any]:
    """
    Ensures LLM response is valid JSON.
    Attempts cleanup if needed.
    """

    # Find JSON block inside content (LLM may add trailing info)
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        payload = content[start:end]
        return json.loads(payload)
    except Exception:
        log_warning(f"[llm_classifier] Failed to parse JSON. Raw: {content}")
        return {
            "industry": None,
            "sub_industry": None,
            "sub_sub_industry": None,
            "confidence": 0.0,
            "synonyms_detected": [],
            "requires_admin_approval": False,
            "proposed_new_taxonomy": None,
            "reason": "json_parse_error"
        }


# -----------------------------
#   MAIN CLASSIFIER FUNCTION
# -----------------------------

def classify_chunk_with_llm(
    text: str,
    embed_candidates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs the LLM classifier pipeline.
    
    Input:
      - chunk text
      - embedding_ranker output

    Output:
      structured dict including taxonomy suggestions
    """

    prompt = build_prompt(text, embed_candidates)

    raw_output = call_llama(prompt)

    parsed = safe_json_parse(raw_output)

    final = {
        "industry": parsed.get("industry"),
        "sub_industry": parsed.get("sub_industry"),
        "sub_sub_industry": parsed.get("sub_sub_industry"),
        "confidence": float(parsed.get("confidence") or 0.0),
        "synonyms_detected": parsed.get("synonyms_detected") or [],
        "requires_admin_approval": bool(parsed.get("requires_admin_approval")),
        "proposed_new_taxonomy": parsed.get("proposed_new_taxonomy"),
        "reason": parsed.get("reason") or "",
        "llm_raw": parsed
    }

    return final
