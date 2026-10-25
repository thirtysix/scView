"""Where else in the body a group's markers are expressed: Human Protein Atlas tissues and cell types.

A marker can be specific among a dataset's clusters and still be expressed in many
other tissues or cell types, which matters for sorting, imaging panels and drug
targets. This module looks each gene up in two Human Protein Atlas (HPA) tables:

- ``rna_tissue_consensus``: nTPM per tissue, the consensus of HPA and GTEx RNA-seq;
- ``rna_single_cell_type``: nCPM per cell type, from HPA's single-cell atlas.

Per gene and per table it gives:

- **tau** (Yanai et al. 2005, the definition in ``core/marker_specificity.py``) over
  log2(value + 1), the scale Kryuchkova-Mostacci & Robinson-Rechavi (2017, Brief
  Bioinform 18:205) recommend: 0 even everywhere, 1 in one tissue only;
- the **HPA specificity category**, implemented from HPA's published rules
  (https://www.proteinatlas.org/about/assays+annotation#classification_rna;
  Uhlén et al. 2015 Science 347:1260419, Karlsson et al. 2021 Sci Adv 7:eabh2169):

  - *not detected*: below 1 in every tissue;
  - *enriched*: at least 4x the next highest tissue;
  - *group enriched*: the mean of a group of 2-5 tissues (2-10 cell types) at
    least 4x the highest tissue outside it;
  - *enhanced*: one or more tissues at least 4x the mean of all the other tissues;
  - *low specificity*: detected (>= 1) but none of the above.

  Every elevated tissue must itself be at >= 1. HPA computes these with its own
  code on its own data; this is a reimplementation of the stated rules, so a gene
  near a cut-off can land one category apart from the HPA website.
- the top 3 tissues / cell types with their values.

A gene's **verdict** combines the two tables: *restricted* (enriched or group
enriched in either), *enhanced* (enhanced across tissues, restricted in neither),
*broad* (low tissue specificity, restricted in neither: "enhanced" among 154 cell
types alone does not count, nearly every gene is 4x the mean in some cell type),
*not detected* (in both), or *not in HPA* (symbol not found). The tissue table
leads when a gene is in only the cell-type table, that table decides. A marker is **flagged** when it is broad in HPA
but specific in the dataset (in-dataset tau >= ``TAU_SPECIFIC``, or tau unknown:
the gene is a marker of the group either way).

The brain regions of the tissue table (amygdala, basal ganglia, cerebellum, ...)
are merged into one "brain" tissue, as the maximum over regions, so a brain gene
is not split across nine regions and called "enhanced" instead of "enriched".

Data: HPA version 25.1 (Ensembl 109), CC BY-SA 4.0. scView does not ship it: each
zip is fetched once, on first use, into ``DATA_DIR/hpa`` (raw zip under ``raw/``,
checked against its pinned sha256), then converted into a compact ``.npz`` plus a
``.json`` meta (URL, version, licence, sha256, sizes). A zip placed in ``raw/`` by
hand is used without a download. ``HPA_AUTO_DOWNLOAD=false`` turns fetching off.
Fetch by hand: ``docker compose exec backend python -m scview.core.offtarget``.

Human symbols. Mouse genes are looked up by their upper-case symbol, which is the
human orthologue's symbol for most genes (not all: the result says so).
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import re
import threading
import weakref
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from scview.core.marker_specificity import tau as _tau

logger = logging.getLogger(__name__)

TAG = "result:offtarget"
HPA_VERSION = "25.1"
HPA_ENSEMBL = "109"
LICENCE = "CC BY-SA 4.0"
ATTRIBUTION = (f"Human Protein Atlas v{HPA_VERSION} (proteinatlas.org), {LICENCE}: tissue RNA consensus "
               "(HPA + GTEx) and single-cell type RNA.")
GENE_URL = "https://www.proteinatlas.org/{ensg}"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
TIMEOUT = httpx.Timeout(120.0, connect=20.0)

FOLD = 4.0  # HPA's fold change for every elevated category
MIN_LEVEL = 1.0  # nTPM / nCPM: detected, and the floor for an elevated tissue
TOP_N = 3
TAU_SPECIFIC = 0.5
MAX_GENES = 200
N_DEFAULT = 20
N_MAX = 50

BRAIN_REGIONS = frozenset({
    "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
    "hippocampal formation", "hypothalamus", "midbrain", "spinal cord",
})

CATEGORY_TEXT = {
    "enriched": "enriched",
    "group_enriched": "group enriched",
    "enhanced": "enhanced",
    "low_specificity": "low specificity",
    "not_detected": "not detected",
}
VERDICT_TEXT = {
    "restricted": "restricted",
    "enhanced": "enhanced",
    "broad": "broad",
    "not_detected": "not detected",
    "not_in_hpa": "not in HPA",
}
MOUSE_NOTE = ("Mouse genes are matched to HPA (human) by upper-case symbol, the human orthologue's "
              "symbol for most genes; a gene with a different human symbol shows as not in HPA.")


class HpaUnavailable(RuntimeError):
    """An HPA table that is not on disk and could not be fetched (offline, disabled, bad file)."""


@dataclass(frozen=True)
class HpaFile:
    kind: str  # "tissue" | "cell_type"
    label: str
    url: str
    sha256: str
    label_column: str
    value_column: str
    unit: str
    max_group: int  # group enriched: 2..max_group tissues / cell types

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


FILES: dict[str, HpaFile] = {
    "tissue": HpaFile(
        kind="tissue",
        label="HPA tissue RNA consensus",
        url="https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip",
        sha256="b001903da23f802c8dfbd1b6b3a8c20b831fedd4932f4bb721b06622d83ae60b",  # 5,293,680 bytes
        label_column="Tissue",
        value_column="nTPM",
        unit="nTPM",
        max_group=5,
    ),
    "cell_type": HpaFile(
        kind="cell_type",
        label="HPA single-cell type RNA",
        url="https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip",
        sha256="2ddd3d90fb050a8efea508792ba7741c510fb5aa3d89f352aab99794ec2c9d64",  # 16,346,850 bytes
        label_column="Cell type",
        value_column="nCPM",
        unit="nCPM",
        max_group=10,
    ),
}


def hpa_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "hpa"


def auto_download_enabled() -> bool:
    from scview.config import get_settings

    return bool(getattr(get_settings(), "HPA_AUTO_DOWNLOAD", True))


# ---------------------------------------------------------------------------
# Parse, fetch, store
# ---------------------------------------------------------------------------


@dataclass
class Table:
    """One HPA table as a genes x labels matrix."""

    kind: str
    ensg: np.ndarray  # (n_genes,) str
    symbols: np.ndarray  # (n_genes,) str
    labels: np.ndarray  # (n_labels,) str
    values: np.ndarray  # (n_genes, n_labels) float32
    meta: dict

    def __post_init__(self) -> None:
        self._by_symbol: dict[str, int] = {}
        self._by_upper: dict[str, int] = {}
        # A symbol on two Ensembl ids: the one with the higher peak wins.
        order = np.argsort(-self.values.max(axis=1), kind="stable") if len(self.values) else []
        for i in order:
            s = str(self.symbols[i])
            self._by_symbol.setdefault(s, int(i))
            self._by_upper.setdefault(s.upper(), int(i))
        self._by_ensg = {str(e): i for i, e in enumerate(self.ensg)}

    def find(self, gene: str, *, mouse: bool = False) -> int | None:
        g = str(gene).strip()
        if g.upper().startswith("ENSG"):
            return self._by_ensg.get(g.split(".")[0].upper())
        if not mouse and g in self._by_symbol:
            return self._by_symbol[g]
        return self._by_upper.get(g.upper())


def parse_zip(raw: bytes, spec: HpaFile) -> Table:
    """An HPA download (zip of one TSV) as a Table; ValueError when it is not one."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if len(names) != 1:
            raise ValueError(f"{spec.label}: expected one .tsv in the zip, found {len(names)}")
        with zf.open(names[0]) as fh:
            cols = ["Gene", "Gene name", spec.label_column, spec.value_column]
            df = pd.read_csv(fh, sep="\t", usecols=lambda c: c in cols,
                             dtype={"Gene": "category", "Gene name": str, spec.label_column: "category"})
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{spec.label} file lacks columns {sorted(missing)}")
    df[spec.value_column] = pd.to_numeric(df[spec.value_column], errors="coerce").fillna(0.0)
    labels = df[spec.label_column].astype(str)
    if spec.kind == "tissue":
        labels = labels.where(~labels.isin(BRAIN_REGIONS), "brain")
    lab = pd.Categorical(labels)
    gene = df["Gene"].cat.remove_unused_categories()
    values = np.zeros((len(gene.cat.categories), len(lab.categories)), dtype=np.float32)
    # max, so merged brain regions keep their highest region (other cells are unique)
    np.maximum.at(values, (gene.cat.codes.to_numpy(), lab.codes),
                  df[spec.value_column].to_numpy(dtype=np.float32))
    first = df.drop_duplicates("Gene").set_index("Gene")["Gene name"]
    ensg = np.asarray(gene.cat.categories.astype(str))
    symbols = np.asarray([str(s) if isinstance(s, str) and s else e
                          for e, s in zip(ensg, first.reindex(ensg).to_numpy())])
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{spec.label}: no genes or fewer than two {spec.label_column.lower()}s")
    return Table(spec.kind, ensg, symbols, np.asarray(lab.categories.astype(str)), values, {})


def _npz(dest: Path, kind: str) -> Path:
    return dest / f"hpa_{kind}.npz"


def _meta_path(dest: Path, kind: str) -> Path:
    return dest / f"hpa_{kind}.json"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _download(spec: HpaFile, part: Path, client: httpx.Client) -> None:
    size = 0
    with client.stream("GET", spec.url) as resp:
        resp.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                fh.write(chunk)


def install(kind: str, dest: Path, *, auto_download: bool = True, client: httpx.Client | None = None) -> None:
    """Fetch (or take from ``raw/``), check the pinned sha256, convert and store one
    table. Raises on any failure and leaves no partial file behind."""
    spec = FILES[kind]
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / spec.filename
    served = "local file"
    if not raw.is_file():
        if not auto_download:
            raise HpaUnavailable(
                f"The {spec.label} table is not downloaded and HPA_AUTO_DOWNLOAD is off. "
                f"Put {spec.filename} in {raw_dir} or turn the download on.")
        part = raw_dir / f"{spec.filename}.part"
        own = client is None
        c = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
        try:
            _download(spec, part, c)
            digest = _sha256_bytes(part.read_bytes())
            if digest != spec.sha256:
                raise ValueError(f"sha256 {digest[:12]}... is not HPA v{HPA_VERSION} as scView expects "
                                 f"({spec.sha256[:12]}...)")
            part.replace(raw)
            served = spec.url
        finally:
            part.unlink(missing_ok=True)
            if own:
                c.close()
    data = raw.read_bytes()
    digest = _sha256_bytes(data)
    if digest != spec.sha256:
        raise ValueError(f"{raw} is not HPA v{HPA_VERSION} as scView expects (sha256 {digest[:12]}...)")
    table = parse_zip(data, spec)
    tmp = dest / f"hpa_{kind}.part.npz"
    try:
        np.savez_compressed(tmp, ensg=table.ensg.astype(str), symbols=table.symbols.astype(str),
                            labels=table.labels.astype(str),
                            values=table.values)
        tmp.replace(_npz(dest, kind))
    finally:
        tmp.unlink(missing_ok=True)
    meta = {
        "kind": kind, "label": spec.label, "url": spec.url, "served_from": served,
        "version": HPA_VERSION, "ensembl": HPA_ENSEMBL, "licence": LICENCE,
        "attribution": ATTRIBUTION, "sha256": digest, "download_bytes": len(data),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unit": spec.unit, "n_genes": int(table.values.shape[0]), "n_labels": int(table.values.shape[1]),
        "merged": "brain regions merged into 'brain' (max)" if kind == "tissue" else None,
    }
    _meta_path(dest, kind).write_text(json.dumps(meta, indent=1))
    logger.info("HPA %s: %d genes x %d %s", kind, meta["n_genes"], meta["n_labels"], spec.label_column)


_lock = threading.Lock()
_loaded: dict[tuple[str, str], tuple[float, Table]] = {}


def load_table(kind: str, dest: Path | None = None, *, auto_download: bool | None = None,
               client: httpx.Client | None = None) -> Table:
    """One HPA table, installed first when missing (blocking: call from a worker thread).
    Raises ``HpaUnavailable`` with what to do."""
    dest = dest or hpa_dir()
    auto = auto_download_enabled() if auto_download is None else auto_download
    spec = FILES[kind]
    path = _npz(dest, kind)
    with _lock:
        if not path.is_file():
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".download.lock", "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)  # across processes
                    if not path.is_file():
                        install(kind, dest, auto_download=auto, client=client)
            except HpaUnavailable:
                raise
            except Exception as e:
                logger.warning("Could not install %s: %s", spec.label, e)
                raise HpaUnavailable(
                    f"The {spec.label} table is not downloaded yet and fetching it failed ({e}). scView "
                    "fetches it once from www.proteinatlas.org; check the internet connection and try "
                    "again.") from e
        mtime = path.stat().st_mtime
        hit = _loaded.get((kind, str(path)))
        if hit and hit[0] == mtime:
            return hit[1]
        with np.load(path, allow_pickle=False) as z:
            try:
                meta = json.loads(_meta_path(dest, kind).read_text())
            except (OSError, ValueError):
                meta = {}
            table = Table(kind, z["ensg"], z["symbols"], z["labels"], z["values"].astype(np.float32), meta)
        _loaded[(kind, str(path))] = (mtime, table)
        return table


def status(dest: Path | None = None) -> dict:
    """What is on disk (meta per table), without fetching."""
    dest = dest or hpa_dir()
    out = {}
    for kind in FILES:
        try:
            out[kind] = json.loads(_meta_path(dest, kind).read_text()) if _npz(dest, kind).is_file() else None
        except (OSError, ValueError):
            out[kind] = {}
    return {"tables": out, "auto_download": auto_download_enabled(), "version": HPA_VERSION,
            "licence": LICENCE, "attribution": ATTRIBUTION}


# ---------------------------------------------------------------------------
# Categories and the per-gene record
# ---------------------------------------------------------------------------


def classify(values: np.ndarray, max_group: int) -> tuple[str, list[int]]:
    """HPA's specificity category for one gene's values across tissues, and the
    indices of the elevated tissues (best first)."""
    v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    if v.size == 0 or v.max() < MIN_LEVEL:
        return "not_detected", []
    order = np.argsort(-v, kind="stable")
    s = v[order]
    if s.size == 1 or s[0] >= FOLD * s[1]:
        return "enriched", [int(order[0])]
    for k in range(2, min(max_group, s.size - 1) + 1):
        group = s[:k]
        if group.min() >= MIN_LEVEL and group.mean() >= FOLD * s[k]:
            return "group_enriched", [int(i) for i in order[:k]]
    total = v.sum()
    n = v.size
    others_mean = (total - v) / (n - 1)
    hit = [int(i) for i in order if v[i] >= MIN_LEVEL and v[i] >= FOLD * others_mean[i]]
    if hit:
        return "enhanced", hit
    return "low_specificity", []


def _side(table: Table, i: int) -> dict:
    row = table.values[i]
    cat, elevated = classify(row, FILES[table.kind].max_group)
    t = _tau(np.log2(row.astype(np.float64) + 1.0)[:, None])[0]
    top = np.argsort(-row, kind="stable")[:TOP_N]
    return {
        "category": cat,
        "category_text": CATEGORY_TEXT[cat],
        "tau": None if not np.isfinite(t) else round(float(t), 3),
        "max": round(float(row.max()), 1),
        "top": [{"name": str(table.labels[j]), "value": round(float(row[j]), 1)} for j in top if row[j] > 0],
        "elevated": [str(table.labels[j]) for j in elevated],
    }


def _verdict(tissue: dict | None, cell: dict | None) -> str:
    cats = [s["category"] for s in (tissue, cell) if s]
    if any(c in ("enriched", "group_enriched") for c in cats):
        return "restricted"
    if all(c == "not_detected" for c in cats):
        return "not_detected"
    # Among 154 cell types nearly every gene is 4x the mean somewhere (ACTB in
    # megakaryocytes), so "enhanced" in cell types alone does not make it less broad.
    lead = tissue["category"] if tissue else cats[0]
    return "enhanced" if lead == "enhanced" else "broad"


def gene_records(genes: list[str], tissue: Table, cell: Table, *, mouse: bool = False,
                 in_dataset_tau: list[float | None] | None = None) -> list[dict]:
    """One record per gene, in the order given."""
    out = []
    for k, g in enumerate(genes):
        it, ic = tissue.find(g, mouse=mouse), cell.find(g, mouse=mouse)
        dtau = in_dataset_tau[k] if in_dataset_tau is not None else None
        dtau = None if dtau is None or not np.isfinite(dtau) else round(float(dtau), 3)
        rec: dict = {"gene": str(g), "in_dataset_tau": dtau}
        if it is None and ic is None:
            rec.update(found=False, hpa_symbol=None, ensg=None, url=None, tissue=None, cell_type=None,
                       verdict="not_in_hpa", verdict_text=VERDICT_TEXT["not_in_hpa"], flag=False)
            out.append(rec)
            continue
        ensg = str(tissue.ensg[it]) if it is not None else str(cell.ensg[ic])
        sym = str(tissue.symbols[it]) if it is not None else str(cell.symbols[ic])
        sides = {}
        if it is not None:
            sides["tissue"] = _side(tissue, it)
        if ic is not None:
            sides["cell_type"] = _side(cell, ic)
        verdict = _verdict(sides.get("tissue"), sides.get("cell_type"))
        rec.update(found=True, hpa_symbol=sym, ensg=ensg, url=GENE_URL.format(ensg=ensg),
                   tissue=sides.get("tissue"), cell_type=sides.get("cell_type"),
                   verdict=verdict, verdict_text=VERDICT_TEXT[verdict],
                   flag=verdict == "broad" and (dtau is None or dtau >= TAU_SPECIFIC))
        out.append(rec)
    return out


def heatmap(records: list[dict], tissue: Table) -> dict:
    """log2(nTPM + 1), genes found in the tissue table x tissues."""
    rows, names = [], []
    for r in records:
        i = tissue.find(r["ensg"]) if r.get("ensg") else None
        if i is None:
            continue
        rows.append(np.round(np.log2(tissue.values[i].astype(np.float64) + 1.0), 3).tolist())
        names.append(r["gene"])
    return {"genes": names, "tissues": [str(t) for t in tissue.labels], "log2_ntpm": rows,
            "unit": "log2(nTPM + 1)"}


def summarise(records: list[dict]) -> dict:
    counts = {v: 0 for v in VERDICT_TEXT}
    for r in records:
        counts[r["verdict"]] += 1
    return {"n_genes": len(records), "counts": counts,
            "n_flagged": sum(r["flag"] for r in records),
            "flagged": [r["gene"] for r in records if r["flag"]]}


def _clean_genes(genes: list[str]) -> list[str]:
    seen, out = set(), []
    for g in genes:
        g = re.sub(r"\s+", "", str(g))
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def lookup(genes: list[str], *, species: str = "human", dest: Path | None = None,
           auto_download: bool | None = None, client: httpx.Client | None = None,
           in_dataset_tau: list[float | None] | None = None) -> dict:
    """Records, heatmap and summary for an explicit gene list (worker thread)."""
    genes = _clean_genes(genes)
    if not genes:
        raise ValueError("No genes given.")
    if len(genes) > MAX_GENES:
        raise ValueError(f"At most {MAX_GENES} genes at a time ({len(genes)} given).")
    tissue = load_table("tissue", dest, auto_download=auto_download, client=client)
    cell = load_table("cell_type", dest, auto_download=auto_download, client=client)
    mouse = species == "mouse"
    records = gene_records(genes, tissue, cell, mouse=mouse, in_dataset_tau=in_dataset_tau)
    return {
        "species": species,
        "note": MOUSE_NOTE if mouse else None,
        "genes": records,
        "heatmap": heatmap(records, tissue),
        "summary": summarise(records),
        "attribution": ATTRIBUTION,
        "version": HPA_VERSION,
        "licence": LICENCE,
        "cutoffs": {"fold": FOLD, "min_level": MIN_LEVEL, "group_tissues": FILES["tissue"].max_group,
                    "group_cell_types": FILES["cell_type"].max_group, "tau_specific": TAU_SPECIFIC},
    }


def dataset_species(adata) -> str:
    from scview.core.species import detect_species

    call = detect_species(adata).species
    return call if call in ("human", "mouse") else "human"


def for_group(adata, group_col: str, group: str, n: int = N_DEFAULT, *, dest: Path | None = None,
              auto_download: bool | None = None, client: httpx.Client | None = None) -> dict:
    """The off-target view of one group's markers (``marker_selection.group_markers``)."""
    from scview.core import marker_specificity
    from scview.core.marker_selection import group_markers
    from scview.core.ora_background import resolve_markers_key

    per = group_markers(adata, group_col, n=n)
    if not per:
        raise LookupError(f"No markers are stored for '{group_col}'. Compute them in the Markers table first.")
    if str(group) not in per:
        raise KeyError(f"Group '{group}' is not in the markers of '{group_col}'.")
    genes = per[str(group)]
    taus: list[float | None] | None = None
    key = resolve_markers_key(adata.uns, group_col)
    rgg = adata.uns.get(key) if key else None
    if isinstance(rgg, dict) and genes:
        cols = marker_specificity.columns_for_group(rgg, str(group), genes)
        if "tau" in cols:
            taus = [None if not np.isfinite(x) else float(x) for x in cols["tau"]]
    species = dataset_species(adata)
    if not genes:
        res = {"species": species, "note": MOUSE_NOTE if species == "mouse" else None, "genes": [],
               "heatmap": {"genes": [], "tissues": [], "log2_ntpm": [], "unit": "log2(nTPM + 1)"},
               "summary": summarise([]), "attribution": ATTRIBUTION, "version": HPA_VERSION,
               "licence": LICENCE}
    else:
        res = lookup(genes, species=species, dest=dest, auto_download=auto_download, client=client,
                     in_dataset_tau=taus)
    res.update(group_col=group_col, group=str(group), n=n,
               marker_rule=("group_markers: detected in >= 20% of the group, adjusted p < 0.05, "
                            "by log fold change, no ribosomal/mitochondrial genes"))
    return res


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: dict) -> None:
    """Keep the latest group results per dataset (by group) for the co-pilot."""
    try:
        with _last_lock:
            per = _last.setdefault(adaptor, {})
            per[(result["group_col"], result["group"])] = result
            while len(per) > 5:
                per.pop(next(iter(per)))
    except TypeError:  # not weak-referenceable
        pass


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """One line per group checked against HPA in this session."""
    try:
        with _last_lock:
            per = dict(_last.get(adaptor) or {})
    except TypeError:
        return [], None
    rows = [r for (col, _), r in per.items() if col in obs.columns and r["summary"]["n_genes"]]
    if not rows:
        return [], None
    lines = [f"\n## Off-target expression of markers (HPA v{HPA_VERSION} tissues + cell types)"]
    for r in rows:
        s = r["summary"]
        c = s["counts"]
        broad = [g["gene"] for g in r["genes"] if g["verdict"] == "broad"]
        eg = f" (e.g. {', '.join(broad[:3])})" if broad else ""
        lines.append(
            f"- off-target (HPA): {r['group_col']} {r['group']} markers: {c['broad']}/{s['n_genes']} broadly "
            f"expressed{eg}, {c['restricted']} restricted, {c['enhanced']} enhanced, "
            f"{c['not_in_hpa']} not in HPA; {s['n_flagged']} specific here but broad in HPA [{TAG}]")
    lines.append("- Caveat: HPA categories are recomputed from HPA's published rules on its tables; "
                 "broad in the body does not make a marker wrong within this dataset.")
    first = rows[0]
    detail = (f"{len(rows)} group(s) checked against HPA; {first['group_col']} {first['group']}: "
              f"{first['summary']['counts']['broad']}/{first['summary']['n_genes']} broad")
    return lines, detail


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for k in FILES:
        load_table(k, auto_download=True)
