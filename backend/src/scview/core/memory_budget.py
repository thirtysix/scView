"""How much memory the backend may use, and how much it uses now.

The start scripts write ``SCVIEW_MEMORY_GB`` to .env and docker-compose.yml turns
it into the container's memory limit, so a run that outgrows it stops scView,
not the computer (on Linux nothing else caps Docker or Podman). Inside the
container the limit is the cgroup's: ``memory.max`` (cgroup v2) or
``memory.limit_in_bytes`` (v1). Without one ("max", or v1's huge sentinel) the
budget is the memory of the machine the container sees, which under Docker
Desktop or Podman on Windows and macOS is their virtual machine.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
MEMINFO = Path("/proc/meminfo")
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
