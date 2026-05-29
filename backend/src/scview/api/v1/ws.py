"""WebSocket endpoint for real-time progress updates (stub – Phase 1)."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/progress/{dataset_id}")
async def progress_websocket(websocket: WebSocket, dataset_id: str):
    """Stream progress updates for long-running operations."""
    await websocket.accept()
    try:
        # Placeholder – in Phase 2 this will relay conversion / analysis progress
        await websocket.send_json({
            "dataset_id": dataset_id,
            "status": "connected",
            "message": "WebSocket endpoint is a stub. Real progress tracking coming in Phase 2.",
        })
        # Keep connection open until the client disconnects
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        pass
