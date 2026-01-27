# =============================================
# whisper_gpu_v1.py
#
# GPU-backed Whisper implementation (future use)
#
# Notes:
# - This file is NOT used unless AI_PROFILE == "gpu"
# - Safe to exist even without GPU today
# - Implements SpeechToText contract
# =============================================

import asyncio
from typing import Optional

from app.ai.contracts import SpeechToText
from app.utils.logger import log_info


class WhisperGPUSpeechToText(SpeechToText):
    """
    GPU Whisper speech-to-text provider.
    Intended for future GPU-enabled deployments.
    """

    def __init__(self, model_size: str = "large"):
        self.model_size = model_size
        self._model = None

    # -------------------------------------------------
    # Lazy model loader (GPU)
    # -------------------------------------------------
    def _load_model(self):
        if self._model is None:
            log_info(f"[WhisperGPU] Loading Whisper model (GPU): {self.model_size}")
            import whisper

            # IMPORTANT:
            # This will only work when CUDA is available.
            # This file is never loaded in CPU mode.
            self._model = whisper.load_model(self.model_size, device="cuda")

        return self._model

    # -------------------------------------------------
    # Transcription
    # -------------------------------------------------
    async def transcribe(self, audio_path: str) -> str:
        model = self._load_model()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_path)
        )

        text = result.get("text", "")
        return text.strip()
