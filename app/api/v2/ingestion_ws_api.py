from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from datetime import datetime
import asyncio
import json

router = APIRouter(prefix="/api/v2/ws", tags=["Ingestion Live Feed"])

# ----------------------------------
# ACTIVE CONNECTIONS STORAGE
# ----------------------------------
active_connections: List[WebSocket] = []

# ----------------------------------
# BROADCAST FUNCTION
# ----------------------------------
async def broadcast(message: Dict[str, Any]):
    """
    Broadcasts a JSON message to all connected WebSocket clients.
    Can be called by logger.py, watcher_ingestor_v2.py, or other modules.
    """
    if not active_connections:
        return

    text_data = json.dumps(message, ensure_ascii=False)
    disconnected = []

    for connection in list(active_connections):
        try:
            await connection.send_text(text_data)
        except Exception:
            disconnected.append(connection)

    # Remove any disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

# ----------------------------------
# WEBSOCKET ENDPOINT
# ----------------------------------
@router.websocket("/ingestion")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for frontend to receive live ingestion logs.
    URL: ws://127.0.0.1:8000/api/v2/ws/ingestion
    """
    await websocket.accept()
    active_connections.append(websocket)
    print(f"🔌 WebSocket connected ({len(active_connections)} clients)")

    try:
        # Initial connection message
        await websocket.send_json({
            "timestamp": datetime.utcnow().isoformat(),
            "stage": "connection",
            "status": "connected",
            "message": "✅ Connected to live ingestion feed.",
        })

        # Keep connection alive with regular pings
        while True:
            await asyncio.sleep(15)
            await websocket.send_json({
                "timestamp": datetime.utcnow().isoformat(),
                "stage": "heartbeat",
                "status": "alive",
                "message": "⏱ Keep-alive ping",
            })

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
