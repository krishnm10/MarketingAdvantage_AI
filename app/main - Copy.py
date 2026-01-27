# =============================================
# main.py — Marketing Advantage AI (v2)
# Updated for ingestion_v2 architecture
# =============================================

import asyncio  # ✅ Required for get_running_loop()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the new ingestion_v2 router
from app.api.v2.ingestion_api_v2 import router as ingestion_router
from app.services.ingestion.watcher_ingestor_v2 import start_watcher_background
from app.api.v2.ingestion_admin_api import router as ingestion_admin_router
from app.api.v2.ingestion_sync_api import router as ingestion_sync_router
from app.api.v2.ingestion_integrity_api import router as ingestion_integrity_router
from app.api.v2.admin_audit_api import router as admin_audit_router
from app.api.v2.auth_api import router as auth_router
from app.api.v2.ingestion_ws_api import router as ingestion_ws_router
from app.api.v2.ingestion_health import router as ingestion_health_router

app = FastAPI(
    title="Marketing Advantage AI — Ingestion v2",
    version="2.0",
    description="Advanced ingestion and classification pipeline for business intelligence.",
)

@app.on_event("startup")
async def startup_event():
    # ✅ Start the background watcher on the same event loop
    loop = asyncio.get_running_loop()
    start_watcher_background(loop)


# -----------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# ROUTER REGISTRATION
# -----------------------------------------------------------
app.include_router(ingestion_router, prefix="/api/v2", tags=["Ingestion v2"])
app.include_router(ingestion_admin_router)
app.include_router(ingestion_sync_router)
app.include_router(ingestion_integrity_router)
app.include_router(admin_audit_router)
app.include_router(auth_router)
app.include_router(ingestion_health_router)
app.include_router(ingestion_ws_router)








# -----------------------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------------------
@app.get("/")
async def index():
    return {"status": "running", "version": "2.0", "service": "Marketing Advantage AI"}
   
