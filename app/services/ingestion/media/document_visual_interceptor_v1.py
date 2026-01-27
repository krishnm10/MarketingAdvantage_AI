# =====================================================
# document_visual_interceptor_v1.py
#
# Document Visual Interception Layer
#
# Responsibilities:
# - Detect embedded visuals in documents
# - Extract visuals deterministically
# - Convert visuals to semantic explanations
# - Return text-only explanations to be merged
#
# IMPORTANT:
# - No DB writes
# - No chunking
# - No vector operations
# =====================================================

import os
import tempfile
from typing import List, Dict, Any
import time

from app.utils.logger import log_info
from app.services.ingestion.media.image_ingestor_v1 import ImageIngestorV1


class DocumentVisualInterceptorV1:
    """
    Intercepts document ingestion to extract and explain
    embedded visuals (charts, figures, tables, images).
    """

    def __init__(self):
        self.image_ingestor = ImageIngestorV1()

    # -------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------
    async def intercept(
        self,
        file_path: str,
        parsed_output: Dict[str, Any],
        file_type: str,
        file_id: str,
        business_id: str = None,
    ) -> List[str]:
        """
        Detects and processes document visuals.

        Returns:
            List of semantic explanations (text only)
        """
        visual_explanations: List[str] = []

        if file_type == "pdf":
            visuals = await self._extract_pdf_visuals(file_path)

        elif file_type == "docx":
            visuals = await self._extract_docx_visuals(file_path)

        elif file_type in ("xls", "xlsx"):
            visuals = await self._extract_excel_visuals(file_path)

        elif file_type in ("html", "web"):
            visuals = await self._extract_web_visuals(parsed_output)

        else:
            return []

        for visual_path, context in visuals:
            explanation = await self._process_visual(
                visual_path=visual_path,
                context=context,
                file_id=file_id,
                business_id=business_id,
            )
            if explanation:
                visual_explanations.append(explanation)

        return visual_explanations

    # -------------------------------------------------
    # Visual → explanation
    # -------------------------------------------------
    async def _process_visual(
        self,
        visual_path: str,
        context: Dict[str, Any],
        file_id: str,
        business_id: str,
    ) -> str:
        """
        Sends visual through image ingestion pipeline
        and returns semantic explanation.
        """
        try:
            explanation = await self.image_ingestor.ingest(
                #file_id=f"{file_id}::visual::{os.path.basename(visual_path)}",
                file_id=file_id,
                image_path=visual_path,
                business_id=business_id,
            )
            return explanation
        finally:
            self._safe_cleanup(visual_path)


    # -------------------------------------------------
    # PDF visuals
    # -------------------------------------------------
    async def _extract_pdf_visuals(self, file_path: str):
        """
        Extract embedded images/charts from PDF.
        """
        import fitz  # PyMuPDF

        visuals = []
        doc = fitz.open(file_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                temp_path = self._write_temp_image(image_bytes, image_ext)
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "pdf",
                            "page": page_index + 1,
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # DOCX visuals
    # -------------------------------------------------
    async def _extract_docx_visuals(self, file_path: str):
        """
        Extract images from DOCX.
        """
        from docx import Document

        visuals = []
        doc = Document(file_path)

        for rel in doc.part._rels.values():
            if "image" in rel.reltype:
                image_bytes = rel.target_part.blob
                image_ext = rel.target_ref.split(".")[-1]

                temp_path = self._write_temp_image(image_bytes, image_ext)
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "docx",
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # Excel visuals (charts as images)
    # -------------------------------------------------
    async def _extract_excel_visuals(self, file_path: str):
        """
        Extract charts rendered as images from Excel.
        """
        visuals = []

        # NOTE:
        # Excel charts extraction is implementation-specific.
        # This is intentionally isolated here so logic does not
        # pollute ingestion core.

        # Placeholder for future chart rendering logic.
        return visuals

    # -------------------------------------------------
    # Web / HTML visuals
    # -------------------------------------------------
    async def _extract_web_visuals(self, parsed_output: Dict[str, Any]):
        """
        Extract images from scraped web content.
        """
        visuals = []

        images = parsed_output.get("images", [])
        for img in images:
            if img.get("bytes"):
                temp_path = self._write_temp_image(
                    img["bytes"], img.get("ext", "png")
                )
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "web",
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # Temp image writer
    # -------------------------------------------------
    def _write_temp_image(self, image_bytes: bytes, ext: str) -> str:
        fd, path = tempfile.mkstemp(suffix=f".{ext}")
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return path
        
        
    def _safe_cleanup(self, path: str, retries: int = 5, delay: float = 0.2):
        """
        Windows-safe temp file cleanup.
        Retries deletion to avoid WinError 32 file locks.
        """
        # Everything inside here must be indented 8 spaces
        for _ in range(retries):
            try:
                if os.path.exists(path):
                    os.remove(path)
                return
            except PermissionError:
                time.sleep(delay)
            except Exception:
                return
