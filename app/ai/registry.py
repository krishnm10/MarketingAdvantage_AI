# =============================================
# registry.py
#
# AI Provider Registry
#
# This file resolves AI capability implementations
# based on runtime configuration.
#
# Rules:
# - This is the ONLY place that knows CPU vs GPU
# - Lazy imports only (avoid heavy startup cost)
# - No ingestion logic
# =============================================

from app.config.ai_config import (
    AI_PROFILE,
    AUDIO_MODEL_CPU,
    AUDIO_MODEL_GPU,
    LOG_AI_PROVIDER_SELECTION,
)


# -------------------------------------------------
# Audio: Speech → Text
# -------------------------------------------------
def get_speech_to_text():
    """
    Resolve SpeechToText provider based on AI_PROFILE.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU SpeechToText provider")

        from app.ai.providers.gpu.whisper_gpu_v1 import WhisperGPUSpeechToText
        return WhisperGPUSpeechToText(model_size=AUDIO_MODEL_GPU)

    # Default: CPU
    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU SpeechToText provider")

    from app.ai.providers.local.whisper_cpu_v1 import WhisperCPUSpeechToText
    return WhisperCPUSpeechToText(model_size=AUDIO_MODEL_CPU)


# -------------------------------------------------
# Image: Image → Caption
# -------------------------------------------------
def get_image_captioner():
    """
    Resolve ImageCaptioner provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU ImageCaptioner provider")

        from app.ai.providers.gpu.image_caption_gpu_v1 import ImageCaptionerGPU
        return ImageCaptionerGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU ImageCaptioner provider")

    from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU
    return ImageCaptionerCPU()


# -------------------------------------------------
# Video: Video → Text
# -------------------------------------------------
def get_video_to_text():
    """
    Resolve VideoToText provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU VideoToText provider")

        from app.ai.providers.gpu.video_to_text_gpu_v1 import VideoToTextGPU
        return VideoToTextGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU VideoToText provider")

    from app.ai.providers.local.video_to_text_cpu_v1 import VideoToTextCPU
    return VideoToTextCPU()


# -------------------------------------------------
# Visual Explanation (charts / tables)
# -------------------------------------------------
def get_visual_explainer():
    """
    Resolve VisualExplainer provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU VisualExplainer provider")

        from app.ai.providers.gpu.visual_explainer_gpu_v1 import VisualExplainerGPU
        return VisualExplainerGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU VisualExplainer provider")

    from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU
    return VisualExplainerCPU()
