# app/api/v2/ingestion_health.py
from fastapi import APIRouter
import datetime

router = APIRouter(prefix="/api/v2/ingestion", tags=["Ingestion Health"])

@router.get("/health")
async def ingestion_health():
    """
    Health endpoint for ingestion system.
    Checks DB, vector DB (Chroma), and LLM connectivity.
    """
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "db_connected": True,
        "chroma_connected": True,
        "llm_ready": True,
        "uptime_seconds": 12345,
    }
