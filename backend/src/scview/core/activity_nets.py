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

Both are human. For a mouse dataset each is mapped through HGNC HCOP orthologs
(``core/orthologs.py``) as decoupler's ``op.collectri(organism="mouse")`` /
``op.progeny(organism="mouse")`` do: CollecTRI's TFs and targets, PROGENy's genes;
then the first edge per (source, target) is kept. The mouse table is cached next to
the human one (``<name>_mouse.tsv``) and rebuilt when either input changes.

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

from scview.core import orthologs
from scview.core.object_cache import ObjectCache

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


def translate_network(name: str, net: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """A human network mapped to mouse: CollecTRI's sources and targets, PROGENy's
    genes (its pathways are not genes), the first edge kept per (source, target)."""
    columns = ["source", "target"] if spec(name).kind == "tf" else ["target"]
    out = orthologs.translate_net(net, columns, table=table)
    return out.drop_duplicates(["source", "target"]).reset_index(drop=True)


def _mouse_table(dest: Path, name: str) -> Path:
    return dest / f"{name}_mouse.tsv"


def _mouse_meta_path(dest: Path, name: str) -> Path:
    return dest / f"{name}_mouse.json"


def read_mouse_meta(dest: Path, name: str) -> dict | None:
    """How the mouse network was mapped (counts, HCOP release), or None when not built."""
    if not _mouse_table(dest, name).is_file():
        return None
    try:
        return json.loads(_mouse_meta_path(dest, name).read_text())
    except (OSError, ValueError):
        return {}


def _build_mouse(name: str, dest: Path, human: pd.DataFrame, table: pd.DataFrame, key: dict) -> pd.DataFrame:
    net = translate_network(name, human, table)
    path = _mouse_table(dest, name)
    tmp = dest / f"{name}_mouse.tsv.part"
    net.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)
    hcop = orthologs.read_meta(dest) or {}
    human_genes = human["target"].astype(str).unique()
    mapped = orthologs.human_to_mouse(human_genes, table=table)
    _mouse_meta_path(dest, name).write_text(json.dumps({
        **key, "name": name, "mapped_by": orthologs.describe(),
        "hcop_pinned": hcop.get("pinned"), "hcop_last_modified": hcop.get("last_modified"),
        "n_human_edges": int(len(human)), "n_edges": int(len(net)),
        "n_human_sources": int(human["source"].nunique()), "n_sources": int(net["source"].nunique()),
        "n_human_targets": int(len(human_genes)),
        "n_targets_mapped": int(sum(bool(v) for v in mapped.values())),
        "n_mouse_targets": int(net["target"].nunique()),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    logger.info("Mapped %s to mouse: %d of %d edges kept", spec(name).label, len(net), len(human))
    return net


def load_mouse_network(name: str, dest: Path, client: httpx.Client | None = None) -> pd.DataFrame:
    """The network mapped to mouse, built from the human one and HCOP on first use
    (both fetched when missing). Raises ``NetworkUnavailable`` with what to do."""
    human = load_network(name, dest, client=client)
    try:
        table = orthologs.load(dest, client=client)
    except orthologs.OrthologsUnavailable as e:
        raise NetworkUnavailable(str(e)) from e
    key = {
        "human_sha256": (read_meta(dest, name) or {}).get("sha256"),
        "hcop_sha256": (orthologs.read_meta(dest) or {}).get("table_sha256"),
        "min_evidence": orthologs.MIN_EVIDENCE, "one_to_many": orthologs.ONE_TO_MANY,
    }
    path = _mouse_table(dest, name)
    with _fetch_lock:
        meta = read_mouse_meta(dest, name) or {}
        if not path.is_file() or any(meta.get(k) != v for k, v in key.items()):
            _build_mouse(name, dest, human, table, key)
        mtime = path.stat().st_mtime
        cached = _loaded.get((name, str(path)))
        if cached and cached[0] == mtime:
            return cached[1]
        net = pd.read_csv(path, sep="\t", dtype={"source": str, "target": str})
        _loaded[(name, str(path))] = (mtime, net)
        return net


def load_for(name: str, dest: Path, species: str, client: httpx.Client | None = None) -> pd.DataFrame:
    """The network in the dataset's species: mapped to mouse for mouse, human otherwise."""
    if species == "mouse":
        return load_mouse_network(name, dest, client=client)
    return load_network(name, dest, client=client)


# Mouse runs, for the co-pilot: loaded dataset -> {network: what was scored and how mapped}.
_scored = ObjectCache()
TAG = "result:activity"


def note_mouse_run(adata, name: str, dest: Path, n_sources: int, n_genes_measured: int) -> dict:
    """What a mouse activity run mapped (shown under the plot and kept as a co-pilot fact)."""
    meta = read_mouse_meta(dest, name) or {}
    info = {
        "network": name, "label": spec(name).label, "mapped_by": orthologs.describe(),
        "n_targets_mapped": meta.get("n_targets_mapped"), "n_human_targets": meta.get("n_human_targets"),
        "n_edges": meta.get("n_edges"), "n_human_edges": meta.get("n_human_edges"),
        "n_sources": int(n_sources), "n_genes_measured": int(n_genes_measured),
    }
    try:
        runs = dict(_scored.get(adata) or {})
        runs[name] = info
        _scored.put(adata, runs)
    except TypeError:  # not weak-referenceable (a test double)
        pass
    return info


def grounding_lines(adata) -> tuple[list[str], str | None]:
    """For the co-pilot: how TF / pathway activity was mapped on this mouse dataset."""
    runs = _scored.get(adata) or {}
    if not runs:
        return [], None
    detail = "TF / pathway activity (ULM) on a mouse dataset, human networks mapped through HCOP orthologs"
    lines = ["\n## TF and pathway activity (mouse; human networks mapped to mouse)"]
    for r in runs.values():
        lines.append(
            f"- activity ({r['label']}): {r['n_targets_mapped']} of {r['n_human_targets']} human target "
            f"genes mapped to mouse through {r['mapped_by']}; {r['n_edges']} of {r['n_human_edges']} edges "
            f"kept; {r['n_sources']} sources scored over {r['n_genes_measured']} measured genes [{TAG}].")
    return lines, detail


def networks_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for n in NETWORKS:
        load_network(n, networks_dir())
