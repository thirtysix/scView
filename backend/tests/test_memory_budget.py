"""The backend's memory budget from cgroup v2 / v1 files, else the machine's memory."""

from __future__ import annotations

from pathlib import Path

from scview.core.memory_budget import mark_running, mark_stopped, read_budget, unclean_restart

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
                 "source": "container limit", "cache_bytes": 0, "available_bytes": 3 * GB}


def test_reclaimable_file_cache_is_not_counted_as_used(tmp_path):
    # After reading a big file its pages sit in memory.current; the kernel drops
    # them before it kills anything (docker stats leaves them out too).
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{6 * GB}\n")
    (cg / "memory.current").write_text(f"{3 * GB}\n")
    (cg / "memory.stat").write_text(f"anon {2 * GB}\nfile {GB}\ninactive_file {GB}\n")
    b = read_budget(cg, _meminfo(tmp_path))
    assert (b.used_bytes, b.cache_bytes, b.to_dict()["available_bytes"]) == (2 * GB, GB, 4 * GB)
    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.limit_in_bytes").write_text(f"{6 * GB}\n")
    (v1 / "memory" / "memory.usage_in_bytes").write_text(f"{3 * GB}\n")
    (v1 / "memory" / "memory.stat").write_text(f"cache {GB}\ntotal_inactive_file {GB // 2}\n")
    assert read_budget(v1, _meminfo(tmp_path)).used_bytes == 3 * GB - GB // 2


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
    assert set(body) == {"limit_bytes", "used_bytes", "limited", "source", "cache_bytes", "available_bytes",
                         "unclean_restart"}


def test_run_marker_recognises_a_killed_backend(tmp_path):
    marker, boot = tmp_path / "running", tmp_path / "boot_id"
    boot.write_text("boot-a\n")
    assert mark_running(marker, boot) is None  # first start
    assert mark_running(marker, boot) is not None  # killed: no mark_stopped in between
    assert unclean_restart()["previous_start"]
    mark_stopped(marker)
    assert not marker.exists()
    assert mark_running(marker, boot) is None  # clean shutdown before
    boot.write_text("boot-b\n")
    assert mark_running(marker, boot) is None  # the computer restarted, not the backend
