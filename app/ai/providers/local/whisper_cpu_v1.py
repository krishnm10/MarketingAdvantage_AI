# =============================================
# whisper_cpu_v1.py
#
# Local CPU-only Whisper implementation
#
# Characteristics:
# - CPU-safe
# - Async-friendly (executor offload)
# - No side effects at import
# - Implements SpeechToText contract
# =============================================

import asyncio
from typing import Optional

from app.ai.contracts import SpeechToText
from app.utils.logger import log_info


class WhisperCPUSpeechToText(SpeechToText):
    """
    CPU-only Whisper speech-to-text provider.
    """

    def __init__(self, model_size: str = "base"):
        """
        Model is loaded lazily to avoid startup cost.
        """
        self.model_size = model_size
        self._model = None

    # -------------------------------------------------
    # Lazy model loader
    # -------------------------------------------------
    def _load_model(self):
        if self._model is None:
            log_info(f"[WhisperCPU] Loading Whisper model (CPU): {self.model_size}")
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    # -------------------------------------------------
    # Transcription
    # -------------------------------------------------
    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file into plain text.
        """
        model = self._load_model()

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_path)
        )

        text = result.get("text", "")
        return text.strip()
