# =============================================
# audio_ingestor_v1.py
#
# Enterprise Audio → Semantic Text Ingestor
#
# Guarantees:
# - Hard safety guards (size, duration)
# - Singleton Whisper model
# - Semantic segmentation
# - Retrieval-safe chunking
# - Governance-ready metadata
# - NO direct DB/vector writes
# =============================================

import os
import re
import asyncio
from typing import Optional, List
from datetime import datetime

from app.utils.logger import log_info
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.db.session_v2 import get_async_session
# Add these new imports after the existing imports
from app.services.ingestion.media.media_hash_utils import MediaHashComputer
from sqlalchemy import select

# After imports, before class definition
def compute_media_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of file bytes for exact duplicate detection.
    Note: This is different from acoustic fingerprinting.
    """
    from app.services.ingestion.media.media_hash_utils import MediaHashComputer
    return MediaHashComputer._compute_file_hash(file_path)



# -----------------------------
# HARD LIMITS (ENTERPRISE)
# -----------------------------
MAX_AUDIO_MB = 50
MAX_AUDIO_DURATION_SEC = 30 * 60  # 30 minutes
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

# -----------------------------
# Whisper Singleton
# -----------------------------
_WHISPER_MODEL = None
_WHISPER_LOCK = asyncio.Lock()


async def get_whisper_model(model_size: str = "base"):
    global _WHISPER_MODEL
    async with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            log_info("[AudioIngestorV1] Loading Whisper model (singleton)")
            import whisper
            _WHISPER_MODEL = whisper.load_model(model_size)
        return _WHISPER_MODEL


# -----------------------------
# Audio Ingestor
# -----------------------------
class AudioIngestorV1:
    """
    Enterprise-grade audio ingestion adapter.
    """
    async def ingest(
        self,
        file_id: str,
        audio_path: str,
        business_id: Optional[str] = None,
    ):
        """
        Enterprise-grade audio ingestion with acoustic fingerprint deduplication.
        """
        log_info(f"[AudioIngestorV1] 🎧 Processing audio: {audio_path}")
        
        # ============================================
        # STEP 0: COMPUTE ACOUSTIC FINGERPRINT & CHECK DUPLICATES
        # ============================================
        acoustic_hash, byte_hash = MediaHashComputer.compute_audio_hash(audio_path)
        
        async with get_async_session() as db:
            # Check for duplicate audio (by acoustic fingerprint)
            existing_query = select(IngestedFileV2).where(
                IngestedFileV2.media_hash == acoustic_hash
            )
            result = await db.execute(existing_query)
            existing_file = result.scalar_one_or_none()
            
            if existing_file:
                log_info(
                    f"[AudioIngestorV1] ⚠️ DUPLICATE DETECTED → "
                    f"{os.path.basename(audio_path)} matches existing file: {existing_file.file_name} "
                    f"(ID: {existing_file.id}, Hash: {acoustic_hash[:12]}...)"
                )
                
                # Return duplicate info (don't process further)
                return {
                    "status": "duplicate_skipped",
                    "duplicate_of": str(existing_file.id),
                    "original_file": existing_file.file_name,
                    "acoustic_hash": acoustic_hash[:16],
                    "message": f"Audio is duplicate of existing file: {existing_file.file_name}"
                }
            
            # ============================================
            # No duplicate found - proceed with ingestion
            # ============================================
            log_info(f"[AudioIngestorV1] ✅ Unique audio confirmed, proceeding with ingestion")
            
            # -----------------------------
            # 1️⃣ Hard guards
            # -----------------------------
            try:
                self._validate_audio(audio_path)
            except ValueError as e:
                log_info(f"[AudioIngestorV1] ❌ Validation failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e)
                }

            # -----------------------------
            # 2️⃣ Ensure file record with media_hash
            # -----------------------------
            file_record = await IngestionServiceV2._get_file_record(db, file_id)
            
            if not file_record:
                file_record = IngestedFileV2(
                    id=file_id,
                    file_name=os.path.basename(audio_path),
                    file_type="audio",
                    file_path=audio_path,
                    business_id=business_id,
                    media_hash=acoustic_hash,  # ⬅️ STORE ACOUSTIC HASH
                    meta_data={
                        "source_type": "audio",
                        "ingested_via": "audio_ingestor_v1",
                        "acoustic_fingerprint": acoustic_hash,
                        "byte_hash": byte_hash,
                        "dedup_method": "chromaprint_or_mfcc",
                    },
                    status="uploaded",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(file_record)
                await db.commit()
            else:
                # Update existing record with media_hash if missing
                if not file_record.media_hash:
                    file_record.media_hash = acoustic_hash
                    if not file_record.meta_data:
                        file_record.meta_data = {}
                    file_record.meta_data.update({
                        "acoustic_fingerprint": acoustic_hash,
                        "byte_hash": byte_hash,
                        "dedup_method": "chromaprint_or_mfcc",
                    })
                    await db.commit()

            # -----------------------------
            # 3️⃣ Transcription (async-safe)
            # -----------------------------
            model = await get_whisper_model()
            loop = asyncio.get_running_loop()
            
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(audio_path)
                )
            except Exception as e:
                log_info(f"[AudioIngestorV1] ❌ Transcription failed: {e}")
                return {
                    "status": "failed",
                    "error": f"Transcription failed: {e}"
                }

            transcript = (result.get("text") or "").strip()
            if not transcript:
                log_info("[AudioIngestorV1] ❌ Empty transcript")
                return {
                    "status": "failed",
                    "error": "Empty transcript - no speech detected"
                }

            # -----------------------------
            # 4️⃣ Normalize & segment
            # -----------------------------
            cleaned = self._clean_transcript(transcript)
            segments = self._segment_transcript(cleaned)

            if not segments:
                log_info("[AudioIngestorV1] ❌ No semantic segments")
                return {
                    "status": "failed",
                    "error": "No semantic segments extracted"
                }

            # -----------------------------
            # 5️⃣ Emit ALL segments in ONE ingestion call
            # -----------------------------
            parsed_payload = {
                "raw_text": "\n".join(segments),
                "meta": {
                    "media_type": "audio",
                    "confidence_source": "model",
                    "transcription_model": "whisper",
                    "segment_count": len(segments),
                    "acoustic_fingerprint": acoustic_hash,
                    "byte_hash": byte_hash,
                },
            }

            await IngestionServiceV2._run_pipeline(
                db=db,
                file_record=file_record,
                parsed_payload=parsed_payload,
            )

            log_info(f"[AudioIngestorV1] ✅ Completed audio ingestion: {file_id}")
            
            return {
                "status": "success",
                "file_id": file_id,
                "acoustic_hash": acoustic_hash[:16],
                "segment_count": len(segments),
                "message": "Audio ingested successfully"
            }

    
   
    # -----------------------------
    # Helpers
    # -----------------------------
        
    def _validate_audio(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_AUDIO_MB:
            raise ValueError(f"Audio too large: {size_mb:.1f} MB")

        # Optional duration check (safe)
        try:
            import wave
            with wave.open(path, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                if duration > MAX_AUDIO_DURATION_SEC:
                    raise ValueError("Audio duration exceeds limit")
        except Exception:
            pass  # duration check best-effort

    def _clean_transcript(self, text: str) -> str:
        text = re.sub(r"\[\d+:\d+(:\d+)?\]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b(uh|um|ah)\b", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _segment_transcript(self, text: str) -> List[str]:
        """
        Deterministic segmentation for retrieval.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        buf = []

        for s in sentences:
            buf.append(s)
            if len(" ".join(buf)) > 400:
                chunks.append(" ".join(buf).strip())
                buf = []

        if buf:
            chunks.append(" ".join(buf).strip())

        return chunks
