"""LIANA's consensus ligand-receptor resource, human and mouse, fetched on first use.

scView does not ship it. The table is LIANA's ``omni_resource.csv`` (Dimitrov et al.
2022, PMID 35680885), the file ``li.rs.select_resource`` reads, taken from the
liana-py v1.10.0 release on GitHub and pinned by sha256, so scView scores the
same pairs as ``li.mt.rank_aggregate``. Two of its resources are kept:

- ``consensus``: human, 4,620 interactions, LIANA's consensus of curated
  ligand-receptor resources compiled through OmniPath.
- ``mouseconsensus``: the same, converted to mouse symbols by LIANA (3,989).

A complex is written with its subunits joined by ``_`` (e.g. ``ITGA4_ITGB1``);
``core/ccc.py`` scores it by its least expressed subunit, as LIANA does.

Licence: liana-py is BSD 3-Clause. The interactions come from OmniPath, which
compiles resources that each carry their own licence (some academic use only),
which is one reason scView fetches the table at run time instead of
redistributing it. Fetch by hand, e.g. for an offline machine:
``docker compose exec backend python -m scview.core.lr_resource``.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/scverse/liana/v1.10.0/src/liana/resource/omni_resource.csv"
SHA256 = "b499b56ce084b7afde3462460e25b79c6abc047086833cd95ca42172d78e0b7c"
CITATION = "Dimitrov et al. 2022, Nat Commun (PMID 35680885); LIANA consensus resource"
VERSION = "liana-py 1.10.0"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # the file is ~2 MB
RESOURCES = {"consensus": "human", "mouseconsensus": "mouse"}
_FILE = "liana_lr_resource.tsv"
_META = "liana_lr_resource.json"


class ResourceUnavailable(RuntimeError):
    """The resource is not on disk and could not be fetched (offline, bad file)."""


def parse(raw: bytes) -> pd.DataFrame:
    """The two consensus resources from LIANA's file: resource / ligand / receptor,
    in file order, without duplicate pairs. Raises ValueError on a wrong file."""
    df = pd.read_csv(io.BytesIO(raw), index_col=False)
    missing = {"source_genesymbol", "target_genesymbol", "resource"} - set(df.columns)
    if missing:
        raise ValueError(f"LIANA resource file lacks columns {sorted(missing)}")
    df = df[df["resource"].isin(list(RESOURCES))]
    out = df.rename(columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"})
    out = out[["resource", "ligand", "receptor"]].dropna()
    out = out.drop_duplicates().reset_index(drop=True)
    counts = out["resource"].value_counts()
    for name in RESOURCES:
        if counts.get(name, 0) < 1000:
            raise ValueError(f"LIANA resource file has {counts.get(name, 0)} '{name}' rows")
    return out


def _table(dest: Path) -> Path:
    return dest / _FILE


def read_meta(dest: Path) -> dict | None:
    """What was fetched and when, or None when the resource is not on disk."""
    if not _table(dest).is_file():
        return None
    try:
        return json.loads((dest / _META).read_text())
    except (OSError, ValueError):
        return {}


def download(dest: Path, client: httpx.Client | None = None, sha256: str = SHA256) -> pd.DataFrame:
    """Fetch, check (sha256, then contents) and store the resource. Raises on any
    failure, leaving no partial file behind."""
    dest.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    tmp = dest / f"{_FILE}.part"
    try:
        buf = io.BytesIO()
        digest = hashlib.sha256()
        with client.stream("GET", URL) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                if buf.tell() + len(chunk) > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                digest.update(chunk)
                buf.write(chunk)
        got = digest.hexdigest()
        if sha256 and got != sha256:
            raise ValueError(f"sha256 {got[:12]}... is not the release scView expects ({sha256[:12]}...)")
        table = parse(buf.getvalue())
        table.to_csv(tmp, sep="\t", index=False)
        tmp.replace(_table(dest))
        meta = {
            "url": URL, "version": VERSION, "sha256": got, "download_bytes": buf.tell(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_interactions": {k: int(v) for k, v in table["resource"].value_counts().items()},
        }
        (dest / _META).write_text(json.dumps(meta, indent=1))
        logger.info("Fetched the LIANA resource: %s", meta["n_interactions"])
        return table
    finally:
        tmp.unlink(missing_ok=True)
        if own_client:
            client.close()


_lock = threading.Lock()
_loaded: dict[str, tuple[float, pd.DataFrame]] = {}


def load(dest: Path, client: httpx.Client | None = None) -> pd.DataFrame:
    """Both resources (resource / ligand / receptor), fetched first if not on disk.
    Raises ``ResourceUnavailable`` (with what to do) when that fails."""
    path = _table(dest)
    with _lock:
        if not path.is_file():
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".download.lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)  # one fetch across processes too
                    if not path.is_file():
                        download(dest, client=client)
            except Exception as e:
                logger.warning("Could not fetch the LIANA resource: %s", e)
                raise ResourceUnavailable(
                    "The LIANA ligand-receptor resource is not downloaded yet and fetching it "
                    f"failed ({e}). scView fetches it once (2 MB) from raw.githubusercontent.com; "
                    "check the internet connection and try again."
                ) from e
        mtime = path.stat().st_mtime
        cached = _loaded.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        _loaded[str(path)] = (mtime, table)
        return table


def select(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """One resource's ligand / receptor pairs."""
    if name not in RESOURCES:
        raise KeyError(f"Unknown resource '{name}'; choose from {', '.join(RESOURCES)}.")
    return table.loc[table["resource"] == name, ["ligand", "receptor"]].reset_index(drop=True)


def resource_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load(resource_dir())
