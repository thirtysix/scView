"""Human -> mouse orthologs from HGNC HCOP, mapped the way decoupler 2 maps its resources.

Human prior knowledge (CollecTRI, PROGENy, the CytoSig signature, gene-set libraries)
is written as human gene symbols. For a mouse dataset decoupler 2.x
(``decoupler.op.translate``, read in 2.3.0) maps it through HCOP, the HGNC
comparison of orthology predictions (Yates et al. 2021, PMID 33959747), which
pools a dozen orthology resources per gene pair. scView does the same, with
decoupler's defaults:

- a pair is kept when at least ``MIN_EVIDENCE`` (3) resources support it;
- a human gene with more than ``ONE_TO_MANY`` (5) mouse orthologs is dropped
  (counted per HCOP row, as decoupler counts them);
- a human gene with several orthologs within the cap maps to all of them;
- a name joined by "_" (a complex) is mapped per subunit and kept only when every
  subunit maps; its mouse names are every combination of the subunits' orthologs;
- many-to-one (two human genes, one mouse ortholog) is left to the caller:
  CollecTRI and PROGENy keep the first edge per (source, target), as decoupler does.

The table (``human_mouse_hcop_fifteen_column.txt.gz``, ~3.5 MB, CC0) is fetched
once, on first use, from the HGNC download bucket into ``DATA_DIR/networks`` and
stored as a small parquet of (human, mouse, evidence) rows. HGNC replaces the file
every month and keeps no archive, so the sha256 pinned here is the release scView
was checked against (``RELEASE``): a download that matches it is recorded as
pinned; a newer one is accepted only when it parses and passes the content checks,
and the meta json says it is unpinned. The stored table is checked against the
sha256 in its meta json on every read. ``ORTHOLOGS_AUTO_DOWNLOAD=false`` turns
fetching off (a table placed there by hand, with its meta, is still used). Fetch
by hand, e.g. for an offline machine: ``python -m scview.core.orthologs``.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz"
RELEASE = "2026-10-01"
SHA256 = "0cfb78e4eb273751e0557e8230445a60e844bfaf958e32740d4a266fb41cdb35"  # the RELEASE file
LICENCE = "CC0 1.0 (HGNC)"
CITATION = "HGNC HCOP: Yates et al. 2021, Brief Bioinform (PMID 33959747)"
LABEL = "HCOP orthologs"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024  # the file is ~3.5 MB
MIN_EVIDENCE = 3  # decoupler 2.x translate default
ONE_TO_MANY = 5  # decoupler 2.x translate default
MIN_PAIRS = 10_000  # pairs at MIN_EVIDENCE; the 2026-10 release has ~21,900
KNOWN_PAIRS = (("TP53", "Trp53"), ("CD3E", "Cd3e"), ("STAT1", "Stat1"))
TABLE = "hcop_human_mouse.parquet"
META = "hcop_human_mouse.json"


class OrthologsUnavailable(RuntimeError):
    """The ortholog table is not on disk and could not be fetched (offline, off, bad file)."""


def parse(raw: bytes) -> pd.DataFrame:
    """The HCOP file as (human, mouse, evidence) rows in file order; raises
    ValueError when the bytes are not the human-mouse table."""
    try:
        df = pd.read_csv(io.BytesIO(raw), sep="\t", compression="gzip", dtype=str,
                         usecols=lambda c: c in {"human_symbol", "mouse_symbol", "support"})
    except Exception as e:  # noqa: BLE001 - any parse failure means a bad file
        raise ValueError(f"not a gzipped HCOP table ({e})") from None
    missing = {"human_symbol", "mouse_symbol", "support"} - set(df.columns)
    if missing:
        raise ValueError(f"HCOP file lacks columns {sorted(missing)}")
    df = df.dropna()
    # "-" is HCOP's empty field: a gene with no symbol in one species.
    df = df[(df["human_symbol"] != "-") & (df["mouse_symbol"] != "-")]
    out = pd.DataFrame({
        "human": df["human_symbol"].to_numpy(str),
        "mouse": df["mouse_symbol"].to_numpy(str),
        "evidence": df["support"].str.split(",").str.len().to_numpy("int16"),
    })
    strong = out[out["evidence"] >= MIN_EVIDENCE]
    if len(strong) < MIN_PAIRS:
        raise ValueError(f"only {len(strong)} pairs with {MIN_EVIDENCE}+ sources; expected {MIN_PAIRS}+")
    pairs = set(zip(strong["human"], strong["mouse"]))
    for h, m in KNOWN_PAIRS:
        if (h, m) not in pairs:
            raise ValueError(f"no {h} -> {m} pair")
    return out.reset_index(drop=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_meta(dest: Path) -> dict | None:
    """What was fetched and when, or None when the table is not on disk."""
    if not (dest / TABLE).is_file():
        return None
    try:
        return json.loads((dest / META).read_text())
    except (OSError, ValueError):
        return {}


def download(dest: Path, client: httpx.Client | None = None) -> pd.DataFrame:
    """Fetch, check and store the table. Raises on any failure, leaving no partial file."""
    dest.mkdir(parents=True, exist_ok=True)
    part, tmp = dest / "hcop.download.part", dest / f"{TABLE}.part"
    own = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        digest, size = hashlib.sha256(), 0
        with client.stream("GET", URL) as resp:
            resp.raise_for_status()
            modified = resp.headers.get("last-modified")
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                    digest.update(chunk)
                    fh.write(chunk)
        table = parse(part.read_bytes())
        table.to_parquet(tmp, index=False)
        tmp.replace(dest / TABLE)
        sha = digest.hexdigest()
        meta = {
            "name": "HCOP human-mouse orthologs", "url": URL, "licence": LICENCE, "citation": CITATION,
            "sha256": sha, "pinned": sha == SHA256, "pinned_release": RELEASE,
            "last_modified": modified, "download_bytes": size,
            "table_sha256": _sha256((dest / TABLE).read_bytes()),
            "n_rows": int(len(table)),
            "n_pairs_min_evidence": int((table["evidence"] >= MIN_EVIDENCE).sum()),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if not meta["pinned"]:
            meta["note"] = (f"A newer HCOP release than the {RELEASE} one scView was checked against; "
                            "accepted because it passed the content checks.")
        (dest / META).write_text(json.dumps(meta, indent=1))
        logger.info("Fetched HCOP human-mouse orthologs: %d pairs (%s)", meta["n_pairs_min_evidence"],
                    "pinned release" if meta["pinned"] else "newer than the pinned release")
        return table
    finally:
        part.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        if own:
            client.close()


def orthologs_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks"


def _auto_download() -> bool:
    from scview.config import get_settings

    return bool(get_settings().ORTHOLOGS_AUTO_DOWNLOAD)


_lock = threading.Lock()
_loaded: dict[str, tuple[float, pd.DataFrame]] = {}
_maps: dict[tuple, tuple[pd.DataFrame, dict[str, list[str]]]] = {}


def load(dest: Path | None = None, *, auto_download: bool | None = None,
         client: httpx.Client | None = None) -> pd.DataFrame:
    """The (human, mouse, evidence) table, fetched first when missing. Raises
    ``OrthologsUnavailable`` with what to do."""
    dest = orthologs_dir() if dest is None else dest
    auto_download = _auto_download() if auto_download is None else auto_download
    path = dest / TABLE
    with _lock:
        if not path.is_file():
            if not auto_download:
                raise OrthologsUnavailable(
                    "The HCOP ortholog table is not downloaded and ORTHOLOGS_AUTO_DOWNLOAD is off, so "
                    "human resources cannot be mapped to mouse. Run `python -m scview.core.orthologs` "
                    "on a machine with internet, or turn the switch on.")
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".hcop.lock", "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    if not path.is_file():
                        download(dest, client=client)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not fetch HCOP orthologs: %s", e)
                raise OrthologsUnavailable(
                    f"The HCOP ortholog table (to map human resources to mouse) is not downloaded yet "
                    f"and fetching it failed ({e}). scView fetches it once from storage.googleapis.com; "
                    "check the internet connection and try again.") from e
        mtime = path.stat().st_mtime
        cached = _loaded.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        raw = path.read_bytes()
        want = (read_meta(dest) or {}).get("table_sha256")
        if want and _sha256(raw) != want:
            raise OrthologsUnavailable(
                f"{path} does not match the sha256 in {META}; delete both to fetch the table again.")
        table = pd.read_parquet(io.BytesIO(raw))
        _loaded[str(path)] = (mtime, table)
        return table


def mapping(table: pd.DataFrame, min_evidence: int = MIN_EVIDENCE,
            one_to_many: int = ONE_TO_MANY) -> dict[str, list[str]]:
    """human symbol -> its mouse orthologs (file order, each once), for the genes that
    map: evidence >= ``min_evidence`` and at most ``one_to_many`` HCOP rows."""
    key = (id(table), len(table), min_evidence, one_to_many)
    hit = _maps.get(key)
    if hit is not None and hit[0] is table:  # the same object, not a new one at a reused id
        return hit[1]
    kept = table[table["evidence"] >= min_evidence]
    out: dict[str, list[str]] = {}
    for h, mice in kept.groupby("human", sort=False)["mouse"]:
        if len(mice) <= one_to_many:  # decoupler counts rows, a pair listed twice included
            out[str(h)] = list(dict.fromkeys(mice))
    if len(_maps) > 8:
        _maps.clear()
    _maps[key] = (table, out)
    return out


def _table(table: pd.DataFrame | None) -> pd.DataFrame:
    return load() if table is None else table


def human_to_mouse(symbols: Iterable[str], *, table: pd.DataFrame | None = None,
                   min_evidence: int = MIN_EVIDENCE, one_to_many: int = ONE_TO_MANY) -> dict[str, list[str]]:
    """Every symbol given -> its mouse orthologs ([] when none passes the cuts)."""
    m = mapping(_table(table), min_evidence, one_to_many)
    return {str(s): list(m.get(str(s), [])) for s in symbols}


def mouse_to_human(symbols: Iterable[str], *, table: pd.DataFrame | None = None,
                   min_evidence: int = MIN_EVIDENCE, one_to_many: int = ONE_TO_MANY) -> dict[str, list[str]]:
    """Every mouse symbol given -> its human orthologs ([] when none passes the cuts):
    the same rules the other way round (a mouse gene with more than ``one_to_many``
    human rows is dropped)."""
    t = _table(table)
    key = ("m2h", id(t), len(t), min_evidence, one_to_many)
    hit = _maps.get(key)
    if hit is not None and hit[0] is t:
        m = hit[1]
    else:
        kept = t[t["evidence"] >= min_evidence]
        m = {}
        for mouse, humans in kept.groupby("mouse", sort=False)["human"]:
            if len(humans) <= one_to_many:
                m[str(mouse)] = list(dict.fromkeys(humans))
        if len(_maps) > 8:
            _maps.clear()
        _maps[key] = (t, m)
    return {str(s): list(m.get(str(s), [])) for s in symbols}


def _translations(values: Iterable[str], m: dict[str, list[str]]) -> pd.DataFrame:
    """(original, translated) rows: every combination of the subunits' orthologs."""
    rows = []
    for v in dict.fromkeys(values):
        parts = [m.get(p) for p in str(v).split("_")]
        if all(parts):
            rows.extend((v, "_".join(c)) for c in product(*parts))
    return pd.DataFrame(rows, columns=["_orig", "_translated"])


def translate_net(df: pd.DataFrame, columns: Iterable[str] = ("source", "target"), *,
                  table: pd.DataFrame | None = None, min_evidence: int = MIN_EVIDENCE,
                  one_to_many: int = ONE_TO_MANY) -> pd.DataFrame:
    """``df`` with each of ``columns`` (that it has) mapped human -> mouse, as
    ``decoupler.op.translate`` does: a row whose name does not map is dropped, one
    that maps to several orthologs becomes several rows, in the original row order.
    Rows are not de-duplicated (decoupler's callers drop duplicate source/target
    pairs, keeping the first)."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError(f"None of {list(columns)} is a column of the table.")
    m = mapping(_table(table), min_evidence, one_to_many)
    out = df
    for col in cols:
        tr = _translations(out[col].astype(str), m).set_index("_orig")
        out = out.merge(tr, left_on=col, right_index=True, how="inner")
        out[col] = out.pop("_translated")
    return out.reset_index(drop=True)


def map_gene_sets(sets: dict[str, list[str]], *, table: pd.DataFrame | None = None,
                  min_evidence: int = MIN_EVIDENCE,
                  one_to_many: int = ONE_TO_MANY) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Human gene sets as mouse gene sets: each gene replaced by all its orthologs
    (each mouse gene once, in order). Returns (sets, coverage), coverage per set name:
    ``{"n_genes": human genes, "n_mapped": of them with an ortholog, "n_mouse": mouse
    genes}``. A set with no mapped gene comes back empty; the caller decides."""
    m = mapping(_table(table), min_evidence, one_to_many)
    out: dict[str, list[str]] = {}
    coverage: dict[str, dict] = {}
    for name, genes in sets.items():
        uniq = list(dict.fromkeys(str(g) for g in genes))
        mice: dict[str, None] = {}
        n_mapped = 0
        for g in uniq:
            hits = m.get(g)
            if hits:
                n_mapped += 1
                mice.update(dict.fromkeys(hits))
        out[name] = list(mice)
        coverage[name] = {"n_genes": len(uniq), "n_mapped": n_mapped, "n_mouse": len(mice)}
    return out, coverage


def describe(min_evidence: int = MIN_EVIDENCE, one_to_many: int = ONE_TO_MANY) -> str:
    """The mapping rule in one phrase, for notes and co-pilot facts."""
    return (f"HCOP orthologs (pairs backed by {min_evidence}+ orthology resources, "
            f"genes with more than {one_to_many} orthologs dropped, as decoupler does)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load(auto_download=True)
