"""scView's own version, and whether a newer release is out (shown in the sidebar).

The backend asks GitHub for the latest release at most once a day (once an hour after a
failure) and keeps the answer in memory, so the browser never calls GitHub. The request
names no user, dataset or machine: a generic User-Agent and no query. ``UPDATE_CHECK=false``
turns it off (offline or batch machines); then nothing is sent and only the version is shown.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

RELEASES_API = "https://api.github.com/repos/thirtysix/scView/releases/latest"
RELEASES_PAGE = "https://github.com/thirtysix/scView/releases"
CHECK_EVERY = 24 * 3600.0
RETRY_AFTER_FAILURE = 3600.0
TIMEOUT = 5.0

_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_lock = asyncio.Lock()
_cache: dict = {"checked_at": None, "latest": None, "url": None, "ok": False}


def current_version() -> str:
    """The installed scview package's version ("unknown" outside an install)."""
    try:
        return importlib.metadata.version("scview")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse(v: str | None) -> tuple[int, int, int] | None:
    """(major, minor, patch) of "0.69.1" or "v0.69.1"; None for anything else (a pre-release too)."""
    m = _VERSION.match((v or "").strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def is_newer(latest: str | None, current: str) -> bool:
    a, b = parse(latest), parse(current)
    return bool(a and b and a > b)


async def _fetch(client: httpx.AsyncClient | None = None) -> tuple[str, str] | None:
    """(tag, page url) of GitHub's latest release, or None when it cannot be read."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        r = await client.get(RELEASES_API, headers={
            "Accept": "application/vnd.github+json", "User-Agent": "scView-update-check"})
        r.raise_for_status()
        data = r.json()
        tag = str(data.get("tag_name") or "")
        if not parse(tag):
            return None
        return tag.lstrip("v"), str(data.get("html_url") or RELEASES_PAGE)
    except Exception as exc:  # offline, rate-limited, GitHub down: never an error for the user
        logger.info("Update check skipped: %s", exc)
        return None
    finally:
        if own:
            await client.aclose()


async def latest_release(client: httpx.AsyncClient | None = None, now: float | None = None) -> dict:
    """{"latest", "url"} from the cache, asking GitHub when the cached answer is stale."""
    now = time.monotonic() if now is None else now
    async with _lock:
        last = _cache["checked_at"]
        wait = CHECK_EVERY if _cache["ok"] else RETRY_AFTER_FAILURE
        if last is None or now - last >= wait:
            got = await _fetch(client)
            _cache.update(checked_at=now, ok=got is not None)
            if got:
                _cache.update(latest=got[0], url=got[1])
        return {"latest": _cache["latest"], "url": _cache["url"]}


async def status(enabled: bool, client: httpx.AsyncClient | None = None) -> dict:
    """What the sidebar shows: this version, and a newer release when there is one."""
    version = current_version()
    out = {"version": version, "check": "on" if enabled else "off",
           "latest": None, "update_available": False, "release_url": None}
    if not enabled:
        return out
    rel = await latest_release(client)
    out["latest"] = rel["latest"]
    if is_newer(rel["latest"], version):
        out.update(update_available=True, release_url=rel["url"] or RELEASES_PAGE)
    return out


def _reset() -> None:
    """Forget the cached answer (tests)."""
    _cache.update(checked_at=None, latest=None, url=None, ok=False)
