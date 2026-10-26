"""Download MSigDB gene-set JSON files from the Broad Institute.

scView does not ship the MSigDB collections (the KEGG and BioCarta sets carry
third-party terms), so each install fetches them from Broad's public release
directory into MSIGDB_DIR: the human ones on first start, the mouse ones (MH,
M1-M8; about 27 MB) when a mouse dataset first asks for them, so a human-only
install never grows. The mouse files are pinned by sha256 and their source is
written to ``msigdb_mouse.json``. MSIGDB_AUTO_DOWNLOAD=false turns both off.
Run by hand to fill in anything a failed start missed:

    docker compose exec backend python -m scview.core.msigdb_fetch [--mouse]

Licence: MSigDB v2022.1 and later is CC BY 4.0, except sets derived from KEGG
(KEGG_ / KEGG_MEDICUS_, human C2 only) and BioCarta (C2 and M2 BIOCARTA_), which
carry their owners' terms: https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import httpx

from scview.core.msigdb_loader import MOUSE_COLLECTIONS, MSIGDB_COLLECTIONS, MSigDBLoader

logger = logging.getLogger(__name__)

MSIGDB_RELEASE = "2026.1.Hs"
BASE_URL = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MSIGDB_RELEASE}"

# The newest mouse release with JSON files (Broad's 2026.2.Mm directory).
MOUSE_RELEASE = "2026.2.Mm"
MOUSE_BASE_URL = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MOUSE_RELEASE}"
LICENCE = ("CC BY 4.0 (MSigDB v2022.1+); BioCarta-derived sets (m2.cp.biocarta) carry BioCarta's terms. "
           "https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp")
MOUSE_META = "msigdb_mouse.json"
# sha256 of each 2026.2.Mm file as published (checked 2026-10-26).
MOUSE_SHA256: dict[str, str] = {
    "mh.all": "6344ef69c6c16e5a112b39240319f86d055113535b6cc65630e8c510db274479",
    "m1.all": "3b291fdcecf43fa208a8c2a4e1a60349267527fa9dd9a6361dbeaf8960e236ea",
    "m2.cgp": "e50292419456c75867ba0ceb473cb6751738a6f87f52dbcb83a181b50f0a127c",
    "m2.cp.biocarta": "2fc00600e2a9fd1057b6d5b41756e0db914764b610030dfc4a9b470ac8ed0943",
    "m2.cp.reactome": "3a131ca90b601913b0e025893dd029bd9416a4025e2a3742e7e09e57538321a1",
    "m2.cp.wikipathways": "e34b07a06f3d784bc09d73ef8afd45f54cc5573e7664325b2f75f9130b72b885",
    "m3.gtrd": "c42b7b6abc293edaa77f8e8a9c9412fb9199d10b8d9ed0a7e2f758fd77521cb5",
    "m3.mirdb": "be00df108dc198fae1c4dae755f03b1752226ed393094c6d83f3c58617627f96",
    "m5.go.bp": "f1bb7b8a940c7ff98a919ac35a7f0216e273b1829d2f9d1d9c5d65fcd56a2b02",
    "m5.go.cc": "1db6dd46fde8fa5d3e9e7a4b854c349218f5ff813142c4f7ef76a4e4a1204aef",
    "m5.go.mf": "40471ab325cd7a7eaf6fb3e08c2200d7adeadcf3cce660d84c72ffdc459303ca",
    "m5.mpo.mp": "1127ac75c559b87f45732d0fa4bfdbfe1c056097633581b6b451504cb9851ae2",
    "m5.mpo.mpt": "e1be2bead784d2db3e561a51872decbc1a01de8e12df2ff1f626c71f3ec5d99a",
    "m7.all": "2be57432162ba7f9325994cc9b96c09a71a9c2efd3050f9090acf996dcb50bc9",
    "m8.all": "3507a95fb31e06cb6f6d13e3255d05746aee6685a1f86773b868248e8dba6c06",
}
MOUSE_IDS = [c["id"] for c in MOUSE_COLLECTIONS]


def missing_collections(dest: Path) -> list[str]:
    """Human collection ids with no JSON file in ``dest`` (any MSigDB version counts)."""
    loader = MSigDBLoader(str(dest))
    return [c["id"] for c in MSIGDB_COLLECTIONS if loader._find_file(c["id"]) is None]


def download_collections(
    dest: Path, ids: list[str], client: httpx.Client | None = None, *,
    release: str = MSIGDB_RELEASE, base_url: str = BASE_URL,
    sha256: dict[str, str] | None = None, errors: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Fetch each collection to ``dest``; return (downloaded, failed) ids.

    Each file is written to a ``.part`` name and validated as JSON (and against
    ``sha256`` when given) before it is renamed into place, so an interrupted or
    garbled download never leaves a file the loader would pick up. ``errors``
    collects why each failed id failed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    done: list[str] = []
    failed: list[str] = []
    try:
        for cid in ids:
            name = f"{cid}.v{release}.json"
            target = dest / name
            part = dest / f"{name}.part"
            try:
                digest = hashlib.sha256()
                with client.stream("GET", f"{base_url}/{name}") as resp:
                    resp.raise_for_status()
                    with open(part, "wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                            digest.update(chunk)
                want = (sha256 or {}).get(cid)
                if want and digest.hexdigest() != want:
                    raise ValueError(f"sha256 mismatch for {name}")
                json.loads(part.read_bytes())
                part.replace(target)
                done.append(cid)
            except Exception as e:  # network, HTTP, disk, checksum or JSON error
                part.unlink(missing_ok=True)
                failed.append(cid)
                if errors is not None:
                    errors[cid] = str(e) or type(e).__name__
                logger.warning("MSigDB download failed for %s: %s", cid, e)
    finally:
        if own_client:
            client.close()
    return done, failed


def ensure_msigdb(dest: Path) -> None:
    """Download any missing collections. Never raises; logs what happened.

    Holds a lock file so the prod stack's two uvicorn workers don't both fetch.
    """
    try:
        if not missing_collections(dest):
            return
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / ".download.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return  # another worker is already downloading
            missing = missing_collections(dest)
            if not missing:
                return
            logger.info(
                "Downloading %d MSigDB collections (release %s) to %s",
                len(missing), MSIGDB_RELEASE, dest,
            )
            done, failed = download_collections(dest, missing)
            logger.info("MSigDB download: %d downloaded, %d failed", len(done), len(failed))
            if failed:
                logger.warning(
                    "Missing MSigDB collections stay unavailable until the next start: %s",
                    ", ".join(failed),
                )
    except OSError as e:
        logger.warning("Could not prepare MSigDB directory %s: %s", dest, e)


# ---------------------------------------------------------------------------
# Mouse collections: fetched on first use
# ---------------------------------------------------------------------------


class MSigDBUnavailable(RuntimeError):
    """Asked-for collections are not on disk and could not be fetched (a 503)."""


_mouse_lock = threading.Lock()
# What the last mouse fetch did, for the collection listing: idle | downloading | failed.
_mouse_state: dict = {"state": "idle", "error": ""}


def missing_mouse(dest: Path) -> list[str]:
    """Mouse collection ids with no JSON file in ``dest``."""
    loader = MSigDBLoader(str(dest))
    return [cid for cid in MOUSE_IDS if loader._find_file(cid) is None]


def _write_mouse_meta(dest: Path) -> None:
    loader = MSigDBLoader(str(dest))
    files = {}
    for cid in MOUSE_IDS:
        f = loader._find_file(cid)
        if f is not None:
            files[f.name] = MOUSE_SHA256.get(cid, "")
    meta = {"source": MOUSE_BASE_URL, "release": MOUSE_RELEASE, "licence": LICENCE,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256": files}
    (dest / MOUSE_META).write_text(json.dumps(meta, indent=1))


def fetch_mouse(dest: Path, client: httpx.Client | None = None) -> tuple[list[str], dict[str, str]]:
    """Download the missing mouse collections, waiting for a fetch already running.
    Returns (downloaded ids, {failed id: why}). Blocks: call off the event loop."""
    with _mouse_lock:
        missing = missing_mouse(dest)
        if not missing:
            return [], {}
        _mouse_state.update(state="downloading", error="")
        errors: dict[str, str] = {}
        done: list[str] = []
        try:
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / ".download-mouse.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)  # another process fetching: wait for it
                missing = missing_mouse(dest)
                logger.info("Downloading %d mouse MSigDB collections (release %s) to %s",
                            len(missing), MOUSE_RELEASE, dest)
                done, _ = download_collections(
                    dest, missing, client, release=MOUSE_RELEASE, base_url=MOUSE_BASE_URL,
                    sha256=MOUSE_SHA256, errors=errors,
                )
                if done:
                    _write_mouse_meta(dest)
        except OSError as e:
            errors = {cid: str(e) for cid in missing}
        _mouse_state.update(state="failed" if errors else "idle",
                            error=next(iter(errors.values()), ""))
        logger.info("Mouse MSigDB download: %d downloaded, %d failed", len(done), len(errors))
        return done, errors


def start_mouse_download(dest: Path) -> None:
    """Fetch the missing mouse collections in a background thread (one at a time)."""
    if _mouse_lock.locked() or not missing_mouse(dest):
        return
    _mouse_state.update(state="downloading", error="")
    threading.Thread(target=fetch_mouse, args=(dest,), daemon=True).start()


def _off_message() -> str:
    return ("The mouse MSigDB collections are not downloaded and MSIGDB_AUTO_DOWNLOAD is off. Fetch them "
            "with: docker compose exec backend python -m scview.core.msigdb_fetch --mouse")


def _failed_message(error: str) -> str:
    why = f" ({error})" if error else ""
    return ("The mouse MSigDB collections are not downloaded yet and fetching them from "
            f"data.broadinstitute.org failed{why}. Check the internet connection and try again, "
            "or choose the human collections (human symbols, matched ignoring case).")


def mouse_status(dest: Path, auto_download: bool) -> dict:
    """{"state": ready | downloading | failed | off | missing, "release", "message"},
    for the collection listing of a mouse dataset."""
    out = {"release": MOUSE_RELEASE, "message": ""}
    if not missing_mouse(dest):
        return {**out, "state": "ready"}
    if _mouse_lock.locked() or _mouse_state["state"] == "downloading":
        return {**out, "state": "downloading",
                "message": f"Downloading the mouse MSigDB collections (release {MOUSE_RELEASE}, about 27 MB)."}
    if not auto_download:
        return {**out, "state": "off", "message": _off_message()}
    if _mouse_state["state"] == "failed":
        return {**out, "state": "failed", "message": _failed_message(_mouse_state["error"])}
    return {**out, "state": "missing"}


def require(ids: list[str], dest: Path | None, auto_download: bool,
            client: httpx.Client | None = None) -> None:
    """Make sure the asked-for mouse collections are on disk, fetching them on first
    use; raise ``MSigDBUnavailable`` with a plain message when they cannot be. Human
    and custom ids are left alone (a missing human file stays an empty collection,
    as before). Blocks while fetching: call off the event loop."""
    want = [i for i in dict.fromkeys(str(x).lower() for x in ids) if i in MOUSE_IDS]
    if not want or dest is None:
        return
    if not any(i in missing_mouse(dest) for i in want):
        return
    if not auto_download:
        raise MSigDBUnavailable(_off_message())
    _, errors = fetch_mouse(dest, client)
    if any(i in missing_mouse(dest) for i in want):
        raise MSigDBUnavailable(_failed_message(next(iter(errors.values()), "")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = Path(os.environ.get("MSIGDB_DIR", "/msigdb"))
    if "--mouse" in sys.argv[1:]:
        _, failed = fetch_mouse(target)
        sys.exit(1 if failed else 0)
    ensure_msigdb(target)
