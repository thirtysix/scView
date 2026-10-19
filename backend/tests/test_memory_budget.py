"""The backend's memory budget from cgroup v2 / v1 files, else the machine's memory."""

from __future__ import annotations

from pathlib import Path

from scview.core.memory_budget import read_budget

GB = 1 << 30


def _meminfo(tmp: Path, total_kb: int = 16 * (1 << 20)) -> Path:
    p = tmp / "meminfo"
    p.write_text(f"MemTotal:       {total_kb} kB\nMemFree:        1000 kB\n")
    return p


def test_cgroup_v2_limit(tmp_path):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{4 * GB}\n")
    (cg / "memory.current").write_text(f"{GB}\n")
    b = read_budget(cg, _meminfo(tmp_path)).to_dict()
    assert b == {"limit_bytes": 4 * GB, "used_bytes": GB, "limited": True,
                 "source": "container limit", "available_bytes": 3 * GB}


def test_no_limit_falls_back_to_the_machine(tmp_path):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text("max\n")
    (cg / "memory.current").write_text(f"{GB}\n")
    b = read_budget(cg, _meminfo(tmp_path))
    assert (b.limit_bytes, b.limited, b.source) == (16 * GB, False, "machine")


def test_cgroup_v1_and_its_unlimited_sentinel(tmp_path):
    cg = tmp_path / "cg"
    (cg / "memory").mkdir(parents=True)
    (cg / "memory" / "memory.limit_in_bytes").write_text(f"{2 * GB}\n")
    (cg / "memory" / "memory.usage_in_bytes").write_text(f"{GB}\n")
    assert read_budget(cg, _meminfo(tmp_path)).limit_bytes == 2 * GB
    (cg / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert read_budget(cg, _meminfo(tmp_path)).source == "machine"


def test_nothing_readable(tmp_path):
    b = read_budget(tmp_path / "none", tmp_path / "no-meminfo")
    assert b.limit_bytes is None and b.source == "unknown"
    assert b.to_dict()["available_bytes"] is None


async def test_endpoint(client):
    body = (await client.get("/api/v1/system/memory")).json()
    assert set(body) == {"limit_bytes", "used_bytes", "limited", "source", "available_bytes"}
