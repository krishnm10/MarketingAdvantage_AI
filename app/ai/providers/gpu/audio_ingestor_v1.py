# =============================================
# audio_ingestor_v1.py
#
# Audio → Text ingestion adapter
#
# Responsibilities:
# - Transcribe audio via AI registry
# - Lightly clean transcript
# - Hand off to IngestionServiceV2 (unchanged)
#
# This file must remain:
# - model-agnostic
# - ingestion-core-safe
# =============================================

import os
import re
from datetime import datetime
from typing import Optional

from app.utils.logger import log_info
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.session_v2 import async_session
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.ai.registry import get_speech_to_text


class AudioIngestorV1:
    """
    Audio ingestion adapter.
    Converts audio into semantic text and feeds the ingestion core.
    """

    def __init__(self):
        # Resolve provider based on ai_config.py
        self.transcriber = get_speech_to_text()

    # --------------------------------------------------
    # Public entrypoint
    # --------------------------------------------------
    async def ingest(
        self,
        file_id: str,
        audio_path: str,
        business_id: Optional[str] = None,
    ):
        """
        Ingest an audio file.
        
        Args:
            file_id: UUID for the ingested file
            audio_path: Path to audio file
            business_id: Optional business scope
        """
        async with async_session() as db:
            log_info(f"[AudioIngestorV1] 🎧 Ingesting audio: {audio_path}")
        
            # ----------------------------------------------
            # Ensure IngestedFile record exists
            # ----------------------------------------------
            file_record = await IngestionServiceV2._get_file_record(db, file_id)
        
            if not file_record:
                file_record = IngestedFileV2(
                    id=file_id,
                    file_name=os.path.basename(audio_path),
                    file_type="audio",
                    file_path=audio_path,
                    business_id=business_id,
                    meta_data={
                        "source_type": "audio",
                        "ingested_via": "audio_ingestor_v1",
                    },
                    status="uploaded",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(file_record)
                await db.commit()
        
            # ----------------------------------------------
            # Transcribe audio (AI registry decides CPU/GPU)
            # ----------------------------------------------
            raw_transcript = await self.transcriber.transcribe(audio_path)
        
            if not raw_transcript or not raw_transcript.strip():
                log_info(f"[AudioIngestorV1] ❌ Empty transcript: {audio_path}")
                return
        
            # ----------------------------------------------
            # Light cleanup (NO heavy rewriting)
            # ----------------------------------------------
            cleaned_transcript = self._clean_transcript(raw_transcript)
        
            # ----------------------------------------------
            # Build parsed payload (core-compatible)
            # ----------------------------------------------
            parsed_payload = {
                "raw_text": cleaned_transcript,
                "meta": {
                    "media_type": "audio",
                    "transcriber": self.transcriber.__class__.__name__,
                },
            }
        
            # ----------------------------------------------
            # Hand off to sealed ingestion core
            # ----------------------------------------------
            await IngestionServiceV2._run_pipeline(
                db=db,
                file_record=file_record,
                parsed_payload=parsed_payload,
            )
        
            log_info(f"[AudioIngestorV1] ✅ Completed audio ingestion: {file_id}")
    # --------------------------------------------------
    # Transcript cleanup (safe & minimal)
    # --------------------------------------------------
    def _clean_transcript(self, text: str) -> str:
        """
        Minimal cleanup:
        - remove timestamps
        - normalize whitespace
        - remove filler words
        """
        text = re.sub(r"\[\d+:\d+(:\d+)?\]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" uh ", " ").replace(" um ", " ")
        return text.strip()
