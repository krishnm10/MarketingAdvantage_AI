# =============================================
# media_ingestion_hook_v1.py
#
# Post-ingestion Media Hook (Unified Dispatcher)
#
# Responsibilities:
# - Handle embedded visuals in documents (PDF, DOCX, XLS)
# - Route extracted visuals to ImageIngestorV1
# - Dispatch standalone media (image / audio / video)
# - Handle deduplication responses from all ingestors
# - MUST NOT call IngestionServiceV2 directly
# - MUST NOT write DB or vectors on its own
# =============================================
from typing import Optional, Dict, Any
from app.utils.logger import log_info, log_warning
from app.services.ingestion.media.image_ingestor_v1 import ImageIngestorV1
from app.services.ingestion.media.audio_ingestor_v1 import AudioIngestorV1
from app.services.ingestion.media.video_ingestor_v1 import VideoIngestorV1  # ✨ NEW!
from app.services.ingestion.media.document_visual_interceptor_v1 import (
    DocumentVisualInterceptorV1,
)
from pathlib import Path


class MediaIngestionHookV1:
    """
    Media ingestion hook executed AFTER file routing.
    Handles deduplication for all media types: image, audio, video.
    """

    def __init__(self):
        self.image_ingestor = ImageIngestorV1()
        self.audio_ingestor = AudioIngestorV1()
        self.video_ingestor = VideoIngestorV1()  # ✨ NEW!
        self.document_visual_interceptor = DocumentVisualInterceptorV1()

    async def handle(
        self,
        *,
        file_id: str,
        file_path: str,
        file_type: str,
        parsed_output: Dict[str, Any],
        business_id: Optional[str] = None,
        media_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Entry point for media handling with deduplication support.
        
        media_kind (optional):
            - None → legacy behavior (document visuals only)
            - "document" → document with embedded visuals
            - "image" → standalone image
            - "audio" → standalone audio
            - "video" → standalone video ✨ NOW SUPPORTED!
        
        Returns:
            Dict with status information (including duplicate detection)
        """
        log_info(f"[MediaHook] Handling media for file: {file_path}")

        # --------------------------------------------------
        # 1️⃣ Legacy behavior (UNCHANGED)
        # --------------------------------------------------
        if media_kind is None:
            if file_type in {"pdf", "docx", "xls", "xlsx"}:
                log_info(f"[MediaHook] Intercepting document visuals: {file_type}")
                await self.document_visual_interceptor.intercept(
                    file_path=file_path,
                    parsed_output=parsed_output or {},
                    file_type=file_type,
                    file_id=file_id,
                    business_id=business_id,
                )
                return {"status": "success", "media_kind": "document_visuals"}

        # --------------------------------------------------
        # 2️⃣ Explicit document visuals
        # --------------------------------------------------
        if media_kind == "document":
            if file_type in {"pdf", "docx", "xls", "xlsx"}:
                log_info(f"[MediaHook] Intercepting document visuals: {file_type}")
                await self.document_visual_interceptor.intercept(
                    file_path=file_path,
                    parsed_output=parsed_output or {},
                    file_type=file_type,
                    file_id=file_id,
                    business_id=business_id,
                )
                return {"status": "success", "media_kind": "document_visuals"}

        # --------------------------------------------------
        # 3️⃣ Standalone image (WITH DEDUPLICATION)
        # --------------------------------------------------
        if media_kind == "image":
            log_info("[MediaHook] Routing standalone image")
            result = await self.image_ingestor.ingest(
                file_id=file_id,
                image_path=file_path,
                business_id=business_id,
            )
            
            # Handle deduplication response
            if result and result.get("status") == "duplicate_skipped":
                log_warning(
                    f"[MediaHook] Image duplicate detected: {result.get('message')}"
                )
            
            return result or {"status": "failed", "error": "Image ingestion returned no result"}

        # --------------------------------------------------
        # 4️⃣ Standalone audio (WITH DEDUPLICATION)
        # --------------------------------------------------
        if media_kind == "audio":
            log_info("[MediaHook] Routing standalone audio")
            result = await self.audio_ingestor.ingest(
                file_id=file_id,
                audio_path=file_path,
                business_id=business_id,
            )
            
            # Handle deduplication response
            if result and result.get("status") == "duplicate_skipped":
                log_warning(
                    f"[MediaHook] Audio duplicate detected: {result.get('message')}"
                )
            
            return result or {"status": "failed", "error": "Audio ingestion returned no result"}

        # --------------------------------------------------
        # 5️⃣ Standalone video (WITH DEDUPLICATION) ✨ FULLY IMPLEMENTED!
        # --------------------------------------------------
        if media_kind == "video":
            log_info("[MediaHook] Routing standalone video (premium quality)")
            result = await self.video_ingestor.ingest(
                file_id=file_id,
                video_path=file_path,
                business_id=business_id,
            )
            
            # Handle deduplication response
            if result and result.get("status") == "duplicate_skipped":
                log_warning(
                    f"[MediaHook] Video duplicate detected: {result.get('message')}"
                )
            elif result and result.get("status") == "success":
                log_info(
                    f"[MediaHook] ✅ Video ingested successfully: "
                    f"{result.get('scenes_analyzed', 0)} scenes analyzed, "
                    f"text quality: {result.get('text_quality', 'standard')}"
                )
            elif result and result.get("status") == "failed":
                log_warning(
                    f"[MediaHook] Video ingestion failed: {result.get('error', 'Unknown error')}"
                )
            
            return result or {"status": "failed", "error": "Video ingestion returned no result"}

        # --------------------------------------------------
        # 6️⃣ Fallback (no-op)
        # --------------------------------------------------
        log_info("[MediaHook] No media action taken")
        return {
            "status": "skipped",
            "message": "No matching media handler for this file type"
        }
