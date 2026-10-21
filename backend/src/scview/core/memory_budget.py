"""How much memory the backend may use, and how much it uses now.

The start scripts write ``SCVIEW_MEMORY_GB`` to .env and docker-compose.yml turns
it into the container's memory limit, so a run that outgrows it stops scView,
not the computer (on Linux nothing else caps Docker or Podman). Inside the
container the limit is the cgroup's: ``memory.max`` (cgroup v2) or
``memory.limit_in_bytes`` (v1). Without one ("max", or v1's huge sentinel) the
budget is the memory of the machine the container sees, which under Docker
Desktop or Podman on Windows and macOS is their virtual machine.

A backend that outgrows the limit is killed and restarted (``restart:
on-failure``), and its cgroup counters start again from zero, so the restart is
recognised by a run marker instead: written at startup and removed at a clean
shutdown, in the container's own /tmp, which survives a restart but not a
recreate. A marker from the same kernel boot means the previous run was killed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
MEMINFO = Path("/proc/meminfo")
RUN_MARKER = Path("/tmp/scview-running")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
# cgroup v1 reports "no limit" as a page-rounded 2^63 - 1.
_V1_UNLIMITED = 1 << 60


@dataclass
class MemoryBudget:
    limit_bytes: int | None  # what scView may use; None when unknown
    used_bytes: int | None
    limited: bool  # a container limit (SCVIEW_MEMORY_GB), not the machine's memory
    source: str  # "container limit" | "machine" | "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available_bytes"] = (
            max(0, self.limit_bytes - self.used_bytes)
            if self.limit_bytes is not None and self.used_bytes is not None else None
        )
        return d


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _meminfo_kb(field: str, meminfo: Path) -> int | None:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_rss() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def read_budget(cgroup: Path = CGROUP, meminfo: Path = MEMINFO) -> MemoryBudget:
    """The backend's memory budget and current use (see the module docstring)."""
    if (cgroup / "memory.max").exists():  # cgroup v2
        limit = _read_int(cgroup / "memory.max")
        used = _read_int(cgroup / "memory.current")
    else:  # cgroup v1, or none
        limit = _read_int(cgroup / "memory" / "memory.limit_in_bytes")
        used = _read_int(cgroup / "memory" / "memory.usage_in_bytes")
        if limit is not None and limit >= _V1_UNLIMITED:
            limit = None
    if used is None:
        used = _process_rss()
    if limit is not None:
        return MemoryBudget(limit, used, True, "container limit")
    total_kb = _meminfo_kb("MemTotal", meminfo)
    if total_kb is not None:
        return MemoryBudget(total_kb * 1024, used, False, "machine")
    return MemoryBudget(None, used, False, "unknown")


_unclean_restart: dict | None = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def mark_running(marker: Path = RUN_MARKER, boot_id: Path = BOOT_ID) -> dict | None:
    """Write the run marker; return (and keep for ``unclean_restart``) when the
    previous run in this container ended without a clean shutdown, else None.
    A marker from another boot is a restarted computer, not a killed backend."""
    global _unclean_restart
    try:
        prev = json.loads(marker.read_text())
    except (OSError, ValueError):
        prev = None
    boot = _read_text(boot_id)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _unclean_restart = (
        {"restarted_at": now, "previous_start": prev.get("started_at")}
        if isinstance(prev, dict) and prev.get("boot_id") == boot else None
    )
    try:
        marker.write_text(json.dumps({"started_at": now, "boot_id": boot}))
    except OSError:
        pass
    return _unclean_restart


def mark_stopped(marker: Path = RUN_MARKER) -> None:
    """Clean shutdown: remove the run marker."""
    try:
        marker.unlink(missing_ok=True)
    except OSError:
        pass


def unclean_restart() -> dict | None:
    """Set when this backend started after the previous one was killed (see ``mark_running``)."""
    return _unclean_restart
