# =============================================
# runtime.py
#
# Global runtime configuration for AI execution.
# This file MUST remain:
# - lightweight
# - import-safe
# - free of model logic
#
# Any change here should switch behavior
# system-wide without code refactors.
# =============================================

# -------------------------------------------------
# AI Execution Profile
#
# cpu  → local CPU-only models (current setup)
# gpu  → local GPU-backed models
# dist → distributed / remote inference (future)
# -------------------------------------------------
AI_PROFILE = "cpu"   # allowed: "cpu", "gpu", "dist"


# -------------------------------------------------
# Audio configuration
# -------------------------------------------------
AUDIO_MODEL_CPU = "base"     # whisper: tiny | base | small
AUDIO_MODEL_GPU = "large"   # future use


# -------------------------------------------------
# Image / Vision configuration (future)
# -------------------------------------------------
IMAGE_CAPTION_MODEL_CPU = "light"
IMAGE_CAPTION_MODEL_GPU = "large"


# -------------------------------------------------
# Video configuration (future)
# -------------------------------------------------
VIDEO_FRAME_SAMPLE_RATE = 1.0   # frames per second
VIDEO_MAX_DURATION_SEC = 1800   # safety cap (30 mins)


# -------------------------------------------------
# LLM behavior flags
# -------------------------------------------------
ENABLE_VISUAL_EXPLANATION = True
ENABLE_AUDIO_LANGUAGE_DETECTION = True


# -------------------------------------------------
# Safety / performance controls
# -------------------------------------------------
MAX_CONCURRENT_MEDIA_TASKS = 2      # keep CPU safe
MEDIA_TIMEOUT_SECONDS = 900         # hard timeout


# -------------------------------------------------
# Debug / observability
# -------------------------------------------------
LOG_AI_PROVIDER_SELECTION = True
