"""Protein interaction subnetwork of a gene list, from STRING physical links.

A DE result's up genes, a group's markers, an NMF program or a typed list: map
the symbols to STRING proteins, take the links among them, and say which are the
hubs, which form communities, and whether the list is more connected than
chance (as STRING's "PPI enrichment p-value", but with a degree-matched null).

Data: STRING v12.0 physical subnetwork (Szklarczyk et al. 2023, Nucleic Acids
Res, PMID 36370105; CC BY 4.0), one links file and one protein-info file per
species, fetched once on first use into ``DATA_DIR/networks`` and checked
against pinned sha256. Links with combined_score >= 400 are kept, in a compact
npz; 400 / 700 / 900 (medium / high / highest confidence) is chosen per query.
Users' gene lists never leave the machine: there is no call to the STRING API.
``PPI_AUTO_DOWNLOAD=false`` turns fetching off; fetch by hand with
``docker compose exec backend python -m scview.core.ppi``.

Connectivity test: the observed number of links among the mapped proteins
against ``N_DRAWS`` random protein sets of the same size drawn from the whole
network, each protein replaced by a random one from the same degree bin (degree
in the whole network at the chosen score; a bin is widened to its neighbours
until it holds ten times the query's proteins in it), seeded. p = (1 + draws with at least
as many links) / (1 + draws). Matching on degree keeps a list of well-studied
hubs from looking connected only because hubs have many partners.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.object_cache import ObjectCache

logger = logging.getLogger(__name__)

STRING_VERSION = "12.0"
LICENCE = "CC BY 4.0"
CITATION = "Szklarczyk et al. 2023, Nucleic Acids Res (PMID 36370105)"
ATTRIBUTION = f"STRING v{STRING_VERSION} physical links (string-db.org, {LICENCE})"
MIN_SCORE = 400  # kept on disk
SCORES = (400, 700, 900)  # medium / high / highest confidence
MAX_GENES = 500
N_DRAWS = 1000
SEED = 0
N_BINS = 20  # degree quantile bins for the null
MIN_POOL = 10  # a bin offers at least this many proteins per query protein in it
N_HUBS = 10
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # the human links file is ~9 MB
TAG = "result:ppi"

_BASE = "https://stringdb-downloads.org/download"


class PPIUnavailable(RuntimeError):
    """The STRING network is not on disk and could not (or may not) be fetched."""


@dataclass(frozen=True)
class SpeciesSpec:
    species: str
    taxid: int
    links_sha256: str
    info_sha256: str
    min_proteins: int
    min_edges: int  # at MIN_SCORE

    @property
    def links_url(self) -> str:
        return f"{_BASE}/protein.physical.links.v{STRING_VERSION}/{self.taxid}.protein.physical.links.v{STRING_VERSION}.txt.gz"

    @property
    def info_url(self) -> str:
        return f"{_BASE}/protein.info.v{STRING_VERSION}/{self.taxid}.protein.info.v{STRING_VERSION}.txt.gz"


SPECIES: dict[str, SpeciesSpec] = {
    "human": SpeciesSpec(
        species="human", taxid=9606,
        links_sha256="1ca87209a93cf7685e9c2997fbcfbfc69fedc6a803ccc1df917318a651df0823",
        info_sha256="144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c",
        min_proteins=15_000, min_edges=50_000,
    ),
    "mouse": SpeciesSpec(
        species="mouse", taxid=10090,
        links_sha256="7a864b6ec28e03f61315af57ed3acf2f5df02512313f105b0a799763a3eb5a0d",
        info_sha256="ededc5cefddea16013c4d8f7d5a1080efea22d4b3f1f090c27019eb1d5209c08",
        min_proteins=15_000, min_edges=20_000,
    ),
}


def spec(species: str) -> SpeciesSpec:
    if species not in SPECIES:
        raise ValueError(f"No STRING network for species '{species}'; choose human or mouse.")
    return SPECIES[species]


# ---------------------------------------------------------------------------
# The network on disk
# ---------------------------------------------------------------------------


@dataclass
class Network:
    """STRING proteins (preferred names) and their links at score >= MIN_SCORE,
    each link once (i < j)."""

    species: str
    names: np.ndarray  # preferred name per protein
    ids: np.ndarray  # STRING id per protein
    i: np.ndarray  # int32
    j: np.ndarray  # int32
    score: np.ndarray  # int16
    meta: dict[str, Any] = field(default_factory=dict)
    _lookup: dict[str, int] | None = field(default=None, repr=False)
    _adj: dict[int, Any] = field(default_factory=dict, repr=False)

    @property
    def n_proteins(self) -> int:
        return int(self.names.size)

    def lookup(self) -> dict[str, int]:
        """Upper-case preferred name -> protein index (first one wins)."""
        if self._lookup is None:
            out: dict[str, int] = {}
            for k, n in enumerate(self.names.tolist()):
                out.setdefault(str(n).upper(), k)
            self._lookup = out
        return self._lookup

    def edges(self, min_score: int) -> tuple[np.ndarray, np.ndarray]:
        keep = self.score >= min_score
        return self.i[keep], self.j[keep]

    def weights(self) -> sp.csr_matrix:
        """Symmetric combined-score matrix (cached)."""
        if -1 not in self._adj:
            n = self.n_proteins
            data = np.concatenate([self.score, self.score]).astype(np.int16)
            W = sp.csr_matrix((data, (np.concatenate([self.i, self.j]), np.concatenate([self.j, self.i]))),
                              shape=(n, n))
            self._adj[-1] = W
        return self._adj[-1]

    def degree(self, min_score: int) -> np.ndarray:
        """Each protein's number of partners at ``min_score`` (cached)."""
        if min_score not in self._adj:
            self._adj[min_score] = np.bincount(np.concatenate(self.edges(min_score)),
                                               minlength=self.n_proteins)
        return self._adj[min_score]


def _read_info(raw: bytes, taxid: int) -> pd.DataFrame:
    info = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep="\t", usecols=[0, 1], dtype=str)
    info.columns = ["id", "name"]
    if info.empty or not info["id"].str.startswith(f"{taxid}.").all():
        raise ValueError(f"STRING protein info is not for taxon {taxid}")
    return info.drop_duplicates("id").reset_index(drop=True)


def _read_links(raw: bytes, taxid: int) -> pd.DataFrame:
    links = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep=" ",
                        dtype={"protein1": str, "protein2": str, "combined_score": np.int32})
    if list(links.columns) != ["protein1", "protein2", "combined_score"]:
        raise ValueError(f"STRING links file has columns {list(links.columns)}")
    if not links["protein1"].str.startswith(f"{taxid}.").all():
        raise ValueError(f"STRING links are not for taxon {taxid}")
    return links[links["combined_score"] >= MIN_SCORE]


def build_from_bytes(species: str, links_raw: bytes, info_raw: bytes) -> Network:
    """Parse and check both downloads; raises ValueError when they are not STRING."""
    s = spec(species)
    info = _read_info(info_raw, s.taxid)
    links = _read_links(links_raw, s.taxid)
    index = pd.Index(info["id"])
    a = index.get_indexer(links["protein1"])
    b = index.get_indexer(links["protein2"])
    ok = (a >= 0) & (b >= 0) & (a != b)
    a, b, w = a[ok], b[ok], links["combined_score"].to_numpy()[ok]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = pd.DataFrame({"i": lo, "j": hi, "s": w}).groupby(["i", "j"], sort=True)["s"].max().reset_index()
    if len(info) < s.min_proteins or len(pairs) < s.min_edges:
        raise ValueError(
            f"STRING {species} download looks truncated: {len(info)} proteins, {len(pairs)} links "
            f"(expected at least {s.min_proteins} and {s.min_edges})."
        )
    return Network(
        species=species, names=info["name"].to_numpy(dtype=str), ids=info["id"].to_numpy(dtype=str),
        i=pairs["i"].to_numpy(np.int32), j=pairs["j"].to_numpy(np.int32),
        score=pairs["s"].to_numpy(np.int16),
    )


def _npz(dest: Path, species: str) -> Path:
    return dest / f"string_v{STRING_VERSION}_{species}.npz"


def _meta_path(dest: Path, species: str) -> Path:
    return dest / f"string_v{STRING_VERSION}_{species}.json"


def read_meta(dest: Path, species: str) -> dict | None:
    """What was fetched and when, or None when the network is not on disk."""
    if not _npz(dest, species).is_file():
        return None
    try:
        return json.loads(_meta_path(dest, species).read_text())
    except (OSError, ValueError):
        return {}


def _fetch(client: httpx.Client, url: str, sha256: str) -> bytes:
    buf = io.BytesIO()
    digest = hashlib.sha256()
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            if buf.tell() + len(chunk) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"{url.rsplit('/', 1)[-1]} exceeds {MAX_DOWNLOAD_BYTES} bytes")
            digest.update(chunk)
            buf.write(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"{url.rsplit('/', 1)[-1]}: sha256 {digest.hexdigest()[:12]}... is not the "
                         f"release scView expects ({sha256[:12]}...)")
    return buf.getvalue()


def download_network(species: str, dest: Path, client: httpx.Client | None = None) -> Network:
    """Fetch, check and store one species' network. Raises on any failure, leaving
    no partial file behind."""
    s = spec(species)
    dest.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=300.0, follow_redirects=True)
    tmp = dest / f"string_{species}.npz.part"
    try:
        links_raw = _fetch(client, s.links_url, s.links_sha256)
        info_raw = _fetch(client, s.info_url, s.info_sha256)
        net = build_from_bytes(species, links_raw, info_raw)
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, names=net.names, ids=net.ids, i=net.i, j=net.j, score=net.score)
        tmp.replace(_npz(dest, species))
        meta = {
            "species": species, "taxid": s.taxid, "version": STRING_VERSION, "licence": LICENCE,
            "citation": CITATION, "links_url": s.links_url, "info_url": s.info_url,
            "links_sha256": s.links_sha256, "info_sha256": s.info_sha256,
            "download_bytes": len(links_raw) + len(info_raw),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_proteins": net.n_proteins, "n_links": int(net.i.size), "min_score": MIN_SCORE,
        }
        _meta_path(dest, species).write_text(json.dumps(meta, indent=1))
        net.meta = meta
        logger.info("Fetched STRING %s: %d proteins, %d links", species, net.n_proteins, net.i.size)
        return net
    finally:
        tmp.unlink(missing_ok=True)
        if own:
            client.close()


_fetch_lock = threading.Lock()
_loaded: dict[str, tuple[float, Network]] = {}


def load_network(species: str, dest: Path, *, auto_download: bool = True,
                 client: httpx.Client | None = None) -> Network:
    """The species' network, fetched first when it is not on disk (and fetching is
    on). Raises ``PPIUnavailable`` saying what to do."""
    s = spec(species)
    path = _npz(dest, species)
    with _fetch_lock:
        if not path.is_file():
            if not auto_download:
                raise PPIUnavailable(
                    f"The STRING {species} network is not downloaded and PPI_AUTO_DOWNLOAD is off. "
                    "Fetch it once with: docker compose exec backend python -m scview.core.ppi"
                )
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".download.lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not path.is_file():
                        download_network(species, dest, client=client)
            except Exception as e:
                logger.warning("Could not fetch STRING %s: %s", species, e)
                raise PPIUnavailable(
                    f"The STRING {species} network is not downloaded yet and fetching it failed ({e}). "
                    f"scView fetches it once (~{11 if s.taxid == 9606 else 6} MB) from stringdb-downloads.org; "
                    "check the internet connection and try again."
                ) from e
        mtime = path.stat().st_mtime
        key = f"{species}:{path}"
        cached = _loaded.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        with np.load(path, allow_pickle=False) as z:
            net = Network(species=species, names=z["names"], ids=z["ids"], i=z["i"], j=z["j"],
                          score=z["score"], meta=read_meta(dest, species) or {})
        _loaded[key] = (mtime, net)
        return net


def networks_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _clean(genes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for g in genes:
        g = str(g).strip()
        if g and g.upper() not in seen:
            seen.add(g.upper())
            out.append(g)
    return out


def degree_bins(degree: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Each protein's degree bin: quantile bins of the whole network's degrees,
    finer at the top (where hubs differ most), degree 0 a bin of its own."""
    pos = degree[degree > 0]
    if pos.size == 0:
        return np.zeros(degree.size, dtype=np.int32)
    qs = np.concatenate([np.linspace(0, 0.9, n_bins // 2, endpoint=False), np.linspace(0.9, 1, n_bins // 2 + 1)])
    edges = np.unique(np.quantile(pos, qs))
    bins = np.searchsorted(edges, degree, side="right").astype(np.int32)  # 1.. for degree > 0
    bins[degree == 0] = 0
    return bins


def _pool(bins: np.ndarray, b: int, size: int) -> np.ndarray:
    """The proteins of bin ``b``, widened to the neighbouring bins (above, then
    below) until there are at least ``size``: a bin barely larger than the query's
    share of it would hand the query back to itself (a list of the top hubs)."""
    lo = hi = b
    top = int(bins.max())
    pool = np.flatnonzero(bins == b)
    while pool.size < size and (lo > 0 or hi < top):
        if hi < top:
            hi += 1
        if lo > 0 and np.count_nonzero((bins >= lo) & (bins <= hi)) < size:
            lo -= 1
        pool = np.flatnonzero((bins >= lo) & (bins <= hi))
    return pool


def null_edge_counts(net: Network, idx: np.ndarray, min_score: int, n_draws: int = N_DRAWS,
                     seed: int = SEED) -> np.ndarray:
    """Links among ``n_draws`` random protein sets, each matched to ``idx`` on degree bin."""
    bins = degree_bins(net.degree(min_score))
    need = pd.Series(bins[idx]).value_counts()
    pools = {int(b): _pool(bins, int(b), MIN_POOL * int(k)) for b, k in need.items()}
    a, b = net.edges(min_score)
    rng = np.random.default_rng(seed)
    mask = np.zeros(net.n_proteins, dtype=bool)
    out = np.empty(n_draws, dtype=np.int64)
    for d in range(n_draws):
        mask[:] = False
        for bn, k in need.items():
            pool = pools[int(bn)]
            mask[rng.choice(pool, size=int(k), replace=k > pool.size)] = True
        out[d] = int(np.count_nonzero(mask[a] & mask[b]))
    return out


def _communities(n: int, edges: list[tuple[int, int]], connected: np.ndarray) -> np.ndarray:
    """Leiden (modularity, seed 0) on the proteins with a link; 1-based ids by
    size, largest first; 0 for a protein with no link."""
    import igraph as ig
    import leidenalg

    comm = np.zeros(n, dtype=np.int32)
    if not edges:
        return comm
    g = ig.Graph(n=n, edges=edges)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=SEED)
    labels = np.asarray(part.membership)
    groups = [c for c in pd.Series(labels[connected]).value_counts().index]
    for rank, c in enumerate(groups, start=1):
        comm[(labels == c) & connected] = rank
    return comm


def _layout(n: int, edges: list[tuple[int, int]], connected: np.ndarray) -> np.ndarray:
    """2D positions for the proteins with a link (seeded spring layout); NaN for the rest."""
    import networkx as nx

    pos = np.full((n, 2), np.nan)
    nodes = np.flatnonzero(connected)
    if nodes.size == 0:
        return pos
    g = nx.Graph()
    g.add_nodes_from(nodes.tolist())
    g.add_edges_from(edges)
    lay = nx.spring_layout(g, seed=SEED, k=1.5 / np.sqrt(max(nodes.size, 1)), iterations=100)
    for v, (x, y) in lay.items():
        pos[v] = (x, y)
    return pos


def format_p(p: float, n_draws: int) -> str:
    return f"p < {1 / n_draws:g}" if p <= 1 / n_draws else f"p = {p:.3f}"


def analyse(net: Network, genes: list[str], *, score: int = MIN_SCORE,
            logfc: dict[str, float] | None = None, n_draws: int = N_DRAWS,
            seed: int = SEED, max_genes: int = MAX_GENES) -> dict[str, Any]:
    """The subnetwork of ``genes`` at ``score``: nodes, links, hubs, communities,
    and the connectivity test against a degree-matched null."""
    import igraph as ig

    if score not in SCORES:
        raise ValueError(f"Score must be one of {', '.join(map(str, SCORES))}.")
    clean = _clean(genes)
    if not clean:
        raise ValueError("No genes given.")
    n_given = len(clean)
    used = clean[:max_genes]
    lookup = net.lookup()
    idx: list[int] = []
    shown: list[str] = []
    unmapped: list[str] = []
    seen: set[int] = set()
    for g in used:
        k = lookup.get(g.upper())
        if k is None:
            unmapped.append(g)
        elif k not in seen:
            seen.add(k)
            idx.append(k)
            shown.append(g)
    ix = np.asarray(idx, dtype=np.int64)
    n = ix.size
    sub = sp.triu(net.weights()[ix][:, ix], k=1).tocoo() if n else sp.coo_matrix((0, 0))
    keep = sub.data >= score
    rows, cols, wts = sub.row[keep], sub.col[keep], sub.data[keep]
    edges = list(zip(rows.tolist(), cols.tolist()))
    m = len(edges)

    g = ig.Graph(n=n, edges=edges)
    degree = np.asarray(g.degree(), dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    btw = np.asarray(g.betweenness(directed=False), dtype=float) if n else np.zeros(0)
    if n > 2:
        btw = btw / ((n - 1) * (n - 2) / 2)
    connected = degree > 0
    comm = _communities(n, edges, connected)
    pos = _layout(n, edges, connected)

    if n >= 2:
        null = null_edge_counts(net, ix, score, n_draws=n_draws, seed=seed)
        expected = float(null.mean())
        p = float((1 + np.count_nonzero(null >= m)) / (1 + n_draws))
    else:
        expected, p = 0.0, None

    order = np.lexsort((-btw, -degree)) if n else np.zeros(0, dtype=int)
    hubs = [int(v) for v in order if degree[v] >= 2][:N_HUBS]
    hub_set = set(hubs)
    lfc = {str(k).upper(): float(v) for k, v in (logfc or {}).items()}
    nodes = []
    for v in range(n):
        nodes.append({
            "gene": shown[v], "name": str(net.names[ix[v]]), "string_id": str(net.ids[ix[v]]),
            "degree": int(degree[v]), "betweenness": round(float(btw[v]), 4),
            "community": int(comm[v]) or None,
            "logfc": (round(lfc[shown[v].upper()], 3) if shown[v].upper() in lfc else None),
            "x": None if np.isnan(pos[v, 0]) else round(float(pos[v, 0]), 4),
            "y": None if np.isnan(pos[v, 1]) else round(float(pos[v, 1]), 4),
            "hub": v in hub_set,
        })
    n_comm = int(comm.max()) if n else 0
    return {
        "species": net.species, "score": score, "version": STRING_VERSION,
        "attribution": ATTRIBUTION, "citation": CITATION,
        "n_given": n_given, "n_used": len(used), "max_genes": max_genes,
        "truncated": n_given > len(used),
        "n_mapped": n, "unmapped": unmapped[:100], "n_unmapped": len(unmapped),
        "n_proteins": n, "n_links": m, "expected_links": round(expected, 2),
        "p_value": p, "n_draws": n_draws, "seed": seed,
        "n_isolated": int(np.count_nonzero(~connected)), "n_communities": n_comm,
        "nodes": nodes,
        "links": [{"source": int(a), "target": int(b), "score": int(w)}
                  for (a, b), w in zip(edges, wts.tolist())],
        "hubs": [shown[v] for v in hubs],
        "method": (
            f"STRING v{STRING_VERSION} physical links with combined score >= {score}. Expected links: the "
            f"mean over {n_draws} random protein sets of the same size from the whole STRING network, "
            "each protein matched on its number of partners (degree bin), seed "
            f"{seed}. Communities: Leiden (modularity, seed {SEED}). Genes are matched to STRING "
            "preferred names ignoring case."
        ),
    }


# ---------------------------------------------------------------------------
# Gene lists from a dataset
# ---------------------------------------------------------------------------

DIRECTIONS = ("up", "down", "both")
MAX_PADJ = 0.05
MIN_ABS_LFC = 0.25


def from_de(stored, n: int, direction: str) -> tuple[list[str], dict[str, float]]:
    """A stored DE result's significant genes in ``direction`` (adjusted p < 0.05 and
    |logFC| >= 0.25; a result without p-values by logFC alone), up to ``n``, the
    smallest p first."""
    genes = np.asarray(stored.genes, dtype=object)
    lfc = np.asarray(stored.logfoldchanges, dtype=float)
    p = stored.pvals_adj if stored.pvals_adj is not None else stored.pvals
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(lfc)
    if direction == "up":
        keep &= lfc >= MIN_ABS_LFC
    elif direction == "down":
        keep &= lfc <= -MIN_ABS_LFC
    else:
        keep &= np.abs(lfc) >= MIN_ABS_LFC
    has_p = np.isfinite(p).any()
    if has_p:
        keep &= p < MAX_PADJ
    idx = np.flatnonzero(keep)
    if has_p:
        idx = idx[np.lexsort((-np.abs(lfc[idx]), p[idx]))]
    else:
        idx = idx[np.argsort(-np.abs(lfc[idx]), kind="stable")]
    idx = idx[:n]
    return [str(g) for g in genes[idx]], {str(genes[k]): float(lfc[k]) for k in idx}


def _marker_lfc(adata, column: str, group: str, genes: list[str]) -> dict[str, float]:
    from scview.core.ora_background import resolve_markers_key

    key = resolve_markers_key(adata.uns, column)
    try:
        rgg = adata.uns[key]
        names = np.asarray(rgg["names"][group]).astype(str)
        lfcs = np.asarray(rgg["logfoldchanges"][group], dtype=float)
    except (KeyError, TypeError, ValueError, IndexError):
        return {}
    pos = {g: k for k, g in enumerate(names.tolist())}
    return {g: float(lfcs[pos[g]]) for g in genes if g in pos and pos[g] < lfcs.size}


def resolve_source(adata, dataset_id: str, source: str, n: int, direction: str
                   ) -> tuple[str, list[str], dict[str, float]]:
    """(label, genes, logFC per gene) for ``de:<id>``, ``markers:<column>/<group>`` or
    ``program:<k>``. ValueError for a malformed source, LookupError when it names
    something that is not there."""
    from scview.core import de_results, gene_programs
    from scview.core.marker_selection import group_markers

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}.")
    kind, _, rest = source.partition(":")
    if kind == "de" and rest:
        stored = de_results.recall(dataset_id, rest)
        if stored is None:
            raise LookupError("That DE result is no longer kept (only the last few are); run the DE again.")
        genes, lfc = from_de(stored, n, direction)
        return f"{stored.label or 'DE result'}, {direction} genes", genes, lfc
    if kind == "markers" and "/" in rest:
        column, _, group = rest.partition("/")
        if column not in adata.obs.columns:
            raise LookupError(f"Column '{column}' not found.")
        markers = group_markers(adata, column, n=n)
        if not markers:
            raise LookupError(f"No markers computed for '{column}' yet: compute them in the Markers tab.")
        if group not in markers:
            raise LookupError(f"'{group}' is not a group of '{column}' with stored markers.")
        genes = markers[group]
        return f"{column} {group} markers", genes, _marker_lfc(adata, column, group, genes)
    if kind == "program" and rest:
        try:
            k = int(rest.lstrip("Pp"))
        except ValueError:
            raise ValueError(f"Program must be a number, not '{rest}'.")
        info = gene_programs.stored(adata)
        if info is None:
            raise LookupError("No gene programs for this dataset yet: run them in the Programs tab.")
        if not 1 <= k <= info["k"]:
            raise LookupError(f"There are {info['k']} programs (P1 to P{info['k']}); no P{k}.")
        name = info["programs"][k - 1].get("name") or f"P{k}"
        return f"program {name}", gene_programs.query_genes(info, k, n), {}
    raise ValueError("source must be de:<result id>, markers:<column>/<group> or program:<k>.")


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------

KEEP_LAST = 3
# loaded dataset object -> the last few PPI summaries; dropped with the object
RESULTS = ObjectCache()


def remember(adata, label: str, result: dict[str, Any], cache: ObjectCache = RESULTS) -> None:
    """Keep a short summary of the last few dataset PPI queries against the loaded
    dataset object, for the co-pilot."""
    summary = {k: result[k] for k in ("score", "n_proteins", "n_links", "expected_links", "p_value",
                                      "n_draws", "hubs", "n_communities", "n_unmapped")}
    prev = [s for s in (cache.get(adata, "ppi") or []) if s["label"] != label]
    cache.put(adata, ([{"label": label, **summary}] + prev)[:KEEP_LAST], key="ppi")


def grounding_lines(adata, cache: ObjectCache = RESULTS) -> tuple[list[str], str]:
    """The last PPI queries on this dataset as facts; nothing before one is run."""
    last = cache.get(adata, "ppi") or []
    if not last:
        return [], ""
    lines = ["\n## Protein interactions (STRING)"]
    for s in last:
        p = format_p(s["p_value"], s["n_draws"]) if s["p_value"] is not None else "p n/a"
        hubs = f"; hubs {', '.join(s['hubs'][:5])}" if s["hubs"] else ""
        lines.append(
            f"- PPI (STRING v{STRING_VERSION} physical >= {s['score']}): {s['label']}: {s['n_proteins']} "
            f"proteins, {s['n_links']} interactions vs {s['expected_links']:g} expected (degree-matched "
            f"random sets), {p}{hubs}; {s['n_communities']} communities [{TAG}]"
        )
    return lines, f"{len(last)} protein-interaction quer{'y' if len(last) == 1 else 'ies'}"


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for sp_name in sys.argv[1:] or list(SPECIES):
        load_network(sp_name, networks_dir())
