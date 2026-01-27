# =============================================
# contracts.py
#
# AI Capability Interfaces
#
# This file defines WHAT the system needs,
# not HOW it is implemented.
#
# Rules:
# - No model imports
# - No config imports
# - No side effects
# - Async-first interfaces
# =============================================

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


# -------------------------------------------------
# Audio → Text
# -------------------------------------------------
class SpeechToText(ABC):
    """
    Converts audio into raw textual transcript.
    """

    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """
        Args:
            audio_path: Absolute or relative path to audio file

        Returns:
            Plain text transcript (no timestamps, no formatting)
        """
        raise NotImplementedError


# -------------------------------------------------
# Image → Caption / Text
# -------------------------------------------------
class ImageCaptioner(ABC):
    """
    Converts images into descriptive text.
    Used for photos, charts, diagrams.
    """

    @abstractmethod
    async def caption(self, image_path: str) -> Dict[str, Any]:
        """
        Args:
            image_path: Path to image file

        Returns:
            {
              "caption": str,
              "ocr_text": Optional[str],
              "objects": Optional[List[str]],
              "is_chart": Optional[bool]
            }
        """
        raise NotImplementedError


# -------------------------------------------------
# Video → Text
# -------------------------------------------------
class VideoToText(ABC):
    """
    Converts video into semantic text using
    audio + frame sampling.
    """

    @abstractmethod
    async def extract(self, video_path: str) -> Dict[str, Any]:
        """
        Args:
            video_path: Path to video file

        Returns:
            {
              "transcript": Optional[str],
              "frame_captions": List[str],
              "duration_sec": Optional[float]
            }
        """
        raise NotImplementedError


# -------------------------------------------------
# Visual / Chart Explanation
# -------------------------------------------------
class VisualExplainer(ABC):
    """
    Converts numeric / chart-like text into
    semantic explanation.
    """

    @abstractmethod
    async def explain(self, text: str) -> str:
        """
        Args:
            text: Raw chart / table / visual text

        Returns:
            Human-readable semantic explanation
        """
        raise NotImplementedError


# -------------------------------------------------
# Optional: Language Detection
# -------------------------------------------------
class LanguageDetector(ABC):
    """
    Detects language of text/audio.
    """

    @abstractmethod
    async def detect(self, text: str) -> str:
        """
        Returns:
            ISO language code (e.g. 'en', 'hi', 'fr')
        """
        raise NotImplementedError
