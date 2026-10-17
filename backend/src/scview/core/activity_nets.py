"""The prior-knowledge networks behind TF and pathway activity: CollecTRI and PROGENy.

scView does not ship them. Each is fetched once, on first use, into
``DATA_DIR/networks`` (the data volume), from where decoupler 2 fetches it, and
built into a source / target / weight table the same way decoupler builds it:

- CollecTRI (Müller-Dott et al. 2023, PMID 37843125): signed TF -> target edges
  (weight +1 activation, -1 repression), from the CollecTRI Zenodo record. As
  decoupler's ``op.collectri`` does, edges without a literature reference are
  dropped and AP1 / NFKB stay as complexes.
- PROGENy (Schubert et al. 2018, PMID 29295995): pathway -> responsive-gene
  weights from OmniPath's annotations, kept where the gene's p-value < 0.05, as
  decoupler's ``op.progeny`` does by default (all such genes, not a top 500).

Human only. decoupler maps them to mouse through orthologs; scView does not yet,
and a mouse dataset is refused with a note rather than matched by symbol case.

A download goes to a ``.part`` file and is parsed and checked (columns, row counts,
signs, known sources) before the table is written, so a captive-portal page or a
truncated file never becomes a network. Fetch by hand, e.g. for an offline
machine: ``docker compose exec backend python -m scview.core.activity_nets``.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # PROGENy's full annotation table is ~41 MB
PROGENY_MAX_PADJ = 0.05
PROGENY_PATHWAYS = frozenset({
    "Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT", "MAPK", "NFkB", "p53",
    "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "WNT",
})


class NetworkUnavailable(RuntimeError):
    """A network that is not on disk and could not be fetched (offline, bad file)."""


def _parse_collectri(raw: bytes) -> pd.DataFrame:
    ct = pd.read_csv(io.BytesIO(raw))
    missing = {"source", "target", "weight", "references"} - set(ct.columns)
    if missing:
        raise ValueError(f"CollecTRI file lacks columns {sorted(missing)}")
    # decoupler drops every row with a missing field: the edges without a reference.
    ct = ct.dropna()
    ct = ct[["source", "target", "weight"]].drop_duplicates(["source", "target"])
    ct["weight"] = pd.to_numeric(ct["weight"], errors="raise").astype(float)
    if not ct["weight"].isin([1.0, -1.0]).all():
        raise ValueError("CollecTRI weights should all be +1 or -1")
    for tf in ("STAT1", "IRF1", "NFKB", "TP53"):
        if tf not in set(ct["source"]):
            raise ValueError(f"CollecTRI file has no '{tf}' regulon")
    return ct.reset_index(drop=True)


def _parse_progeny(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), sep="\t")
    missing = {"genesymbol", "label", "value", "record_id"} - set(df.columns)
    if missing:
        raise ValueError(f"PROGENy annotation file lacks columns {sorted(missing)}")
    if not {"pathway", "weight", "p_value"} <= set(df["label"].unique()):
        raise ValueError("PROGENy annotations lack pathway / weight / p_value records")
    # One row per (gene, record): its pathway, weight and p-value.
    wide = df.pivot(index=["genesymbol", "record_id"], columns="label", values="value").reset_index()
    wide["weight"] = pd.to_numeric(wide["weight"], errors="raise").astype(float)
    wide["p_value"] = pd.to_numeric(wide["p_value"], errors="raise").astype(float)
    wide = wide.sort_values("p_value").sort_values(["pathway", "p_value"], kind="stable")
    net = wide.rename(columns={"pathway": "source", "genesymbol": "target", "p_value": "padj"})
    net = net[net["padj"] < PROGENY_MAX_PADJ][["source", "target", "weight", "padj"]]
    net = net.drop_duplicates(["source", "target"]).reset_index(drop=True)
    if not np.isfinite(net["weight"]).all():
        raise ValueError("PROGENy weights should all be finite")
    unknown = set(net["source"]) - PROGENY_PATHWAYS
    if unknown or len(set(net["source"])) < 10:
        raise ValueError(f"Unexpected PROGENy pathways: {sorted(unknown) or sorted(set(net['source']))}")
    return net


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    label: str
    kind: str  # "tf" | "pathway"
    url: str
    citation: str
    parse: Callable[[bytes], pd.DataFrame]
    min_edges: int
    min_sources: int


NETWORKS: dict[str, NetworkSpec] = {
    "collectri": NetworkSpec(
        name="collectri",
        label="CollecTRI",
        kind="tf",
        url="https://zenodo.org/records/8192729/files/CollecTRI_regulons.csv?download=1",
        citation="Müller-Dott et al. 2023, Nucleic Acids Res (PMID 37843125)",
        parse=_parse_collectri,
        min_edges=20_000,
        min_sources=500,
    ),
    "progeny": NetworkSpec(
        name="progeny",
        label="PROGENy",
        kind="pathway",
        url="https://omnipathdb.org/annotations?databases=PROGENy&license=academic",
        citation="Schubert et al. 2018, Nat Commun (PMID 29295995)",
        parse=_parse_progeny,
        min_edges=5_000,
        min_sources=10,
    ),
}


def spec(name: str) -> NetworkSpec:
    if name not in NETWORKS:
        raise KeyError(f"Unknown network '{name}'; choose from {', '.join(NETWORKS)}.")
    return NETWORKS[name]


def _table(dest: Path, name: str) -> Path:
    return dest / f"{name}_human.tsv"


def _meta_path(dest: Path, name: str) -> Path:
    return dest / f"{name}_human.json"


def read_meta(dest: Path, name: str) -> dict | None:
    """What was fetched and when, or None when the network is not on disk."""
    if not _table(dest, name).is_file():
        return None
    try:
        return json.loads(_meta_path(dest, name).read_text())
    except (OSError, ValueError):
        return {}


def build_from_bytes(name: str, raw: bytes) -> pd.DataFrame:
    """Parse and check a download; raises ValueError when it is not the network."""
    s = spec(name)
    net = s.parse(raw)
    n_sources = net["source"].nunique()
    if len(net) < s.min_edges or n_sources < s.min_sources:
        raise ValueError(
            f"{s.label} download looks truncated: {len(net)} edges from {n_sources} sources "
            f"(expected at least {s.min_edges} and {s.min_sources})."
        )
    return net


def download_network(name: str, dest: Path, client: httpx.Client | None = None) -> pd.DataFrame:
    """Fetch, check and store one network; return its table. Raises on any failure,
    leaving no partial file behind."""
    s = spec(name)
    dest.mkdir(parents=True, exist_ok=True)
    part = dest / f"{name}.download.part"
    own_client = client is None
    client = client or httpx.Client(timeout=300.0, follow_redirects=True)
    try:
        size = 0
        digest = hashlib.sha256()
        with client.stream("GET", s.url) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"{s.label} download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                    digest.update(chunk)
                    fh.write(chunk)
        net = build_from_bytes(name, part.read_bytes())
        tmp = dest / f"{name}_human.tsv.part"
        net.to_csv(tmp, sep="\t", index=False)
        tmp.replace(_table(dest, name))
        meta = {
            "name": name,
            "url": s.url,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": digest.hexdigest(),
            "download_bytes": size,
            "n_edges": int(len(net)),
            "n_sources": int(net["source"].nunique()),
        }
        _meta_path(dest, name).write_text(json.dumps(meta, indent=1))
        logger.info("Fetched %s: %d edges from %d sources", s.label, meta["n_edges"], meta["n_sources"])
        return net
    finally:
        part.unlink(missing_ok=True)
        (dest / f"{name}_human.tsv.part").unlink(missing_ok=True)
        if own_client:
            client.close()


_fetch_lock = threading.Lock()
_loaded: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}


def load_network(name: str, dest: Path, client: httpx.Client | None = None) -> pd.DataFrame:
    """The network table, fetched first if it is not on disk. Raises
    ``NetworkUnavailable`` (with what to do) when it can be neither read nor fetched."""
    s = spec(name)
    path = _table(dest, name)
    with _fetch_lock:  # one fetch at a time in this process ...
        if not path.is_file():
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".download.lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)  # ... and across processes
                    if not path.is_file():
                        download_network(name, dest, client=client)
            except Exception as e:
                logger.warning("Could not fetch %s: %s", s.label, e)
                raise NetworkUnavailable(
                    f"{s.label} is not downloaded yet and fetching it failed ({e}). scView "
                    f"fetches it once from {s.url.split('/')[2]}; check the internet "
                    "connection and try again."
                ) from e
        mtime = path.stat().st_mtime
        cached = _loaded.get((name, str(path)))
        if cached and cached[0] == mtime:
            return cached[1]
        net = pd.read_csv(path, sep="\t", dtype={"source": str, "target": str})
        _loaded[(name, str(path))] = (mtime, net)
        return net


def networks_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for n in NETWORKS:
        load_network(n, networks_dir())
