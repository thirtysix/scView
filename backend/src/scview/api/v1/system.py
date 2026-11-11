"""The backend's own state: its memory budget (``core/memory_budget.py``) and its version
(``core/version_check.py``)."""

from __future__ import annotations

from fastapi import APIRouter

from scview.config import get_settings
from scview.core import version_check
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


@router.get("/system/version")
async def system_version() -> dict:
    """This scView's ``version``; with UPDATE_CHECK on (the default), ``latest`` release on
    GitHub (asked at most once a day, null when it could not be read) and
    ``update_available`` + ``release_url`` when it is newer. ``check`` is "on" or "off"."""
    return await version_check.status(get_settings().UPDATE_CHECK)
