"""The backend's own state: its memory budget (``core/memory_budget.py``)."""

from __future__ import annotations

from fastapi import APIRouter

from scview.core.memory_budget import read_budget

router = APIRouter()


@router.get("/system/memory")
async def system_memory() -> dict:
    """How much memory scView may use (``limit_bytes``), uses now (``used_bytes``)
    and has left (``available_bytes``); ``limited`` when it is the container limit
    set by SCVIEW_MEMORY_GB rather than the machine's memory."""
    return read_budget().to_dict()
