"""The backend's own state: its memory budget (``core/memory_budget.py``)."""

from __future__ import annotations

from fastapi import APIRouter

from scview.core.memory_budget import read_budget, unclean_restart

router = APIRouter()


@router.get("/system/memory")
async def system_memory() -> dict:
    """How much memory scView may use (``limit_bytes``), uses now (``used_bytes``)
    and has left (``available_bytes``); ``limited`` when it is the container limit
    set by SCVIEW_MEMORY_GB rather than the machine's memory. ``unclean_restart``
    is set when the previous backend was killed, most often for running out of
    memory, and restarted."""
    return {**read_budget().to_dict(), "unclean_restart": unclean_restart()}
