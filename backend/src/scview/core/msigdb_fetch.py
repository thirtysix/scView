"""Download MSigDB gene-set JSON files from the Broad Institute.

scView does not ship the MSigDB collections (the KEGG and BioCarta sets carry
third-party terms), so each install fetches them from Broad's public release
directory into MSIGDB_DIR on first start. Run by hand to fill in anything a
failed start missed:

    docker compose exec backend python -m scview.core.msigdb_fetch
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path

import httpx

from scview.core.msigdb_loader import MSIGDB_COLLECTIONS, MSigDBLoader

logger = logging.getLogger(__name__)

MSIGDB_RELEASE = "2026.1.Hs"
BASE_URL = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MSIGDB_RELEASE}"


def missing_collections(dest: Path) -> list[str]:
    """Collection ids with no JSON file in ``dest`` (any MSigDB version counts)."""
    loader = MSigDBLoader(str(dest))
    return [c["id"] for c in MSIGDB_COLLECTIONS if loader._find_file(c["id"]) is None]


def download_collections(
    dest: Path, ids: list[str], client: httpx.Client | None = None
) -> tuple[list[str], list[str]]:
    """Fetch each collection to ``dest``; return (downloaded, failed) ids.

    Each file is written to a ``.part`` name and validated as JSON before it is
    renamed into place, so an interrupted or garbled download never leaves a
    file the loader would pick up.
    """
    dest.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    done: list[str] = []
    failed: list[str] = []
    try:
        for cid in ids:
            name = f"{cid}.v{MSIGDB_RELEASE}.json"
            target = dest / name
            part = dest / f"{name}.part"
            try:
                with client.stream("GET", f"{BASE_URL}/{name}") as resp:
                    resp.raise_for_status()
                    with open(part, "wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                json.loads(part.read_bytes())
                part.replace(target)
                done.append(cid)
            except Exception as e:  # network, HTTP, disk or JSON error
                part.unlink(missing_ok=True)
                failed.append(cid)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_msigdb(Path(os.environ.get("MSIGDB_DIR", "/msigdb")))
