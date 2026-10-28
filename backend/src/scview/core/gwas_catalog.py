"""GWAS trait links: which traits the GWAS Catalog maps near a gene, and which
traits a group's marker genes overlap more than random genes do.

Data: the NHGRI-EBI GWAS Catalog (Sollis et al. 2023, Nucleic Acids Res, PMID
36350656), one PINNED dated release: the full associations file (ontology
annotated, a zip of one TSV) and its EFO trait mappings (label and parent term
per trait). Fetched once into ``DATA_DIR/gwas`` and checked against pinned sha256;
the two files can also be placed there by hand (same names), and are then used
without a download. ``GWAS_AUTO_DOWNLOAD=false`` turns fetching off. The
associations are parsed streaming from the zip (pandas chunks, a few columns) into
a compact npz cache; only genome-wide significant rows (p <= 5e-8) are kept.
Nothing about a dataset leaves the machine: there is no call to any GWAS API.

Traits the mappings file does not list by URI (about 1 in 7: newer MONDO/OBA URIs
for diseases and protein levels) take the parent its rows' reported traits have in
the mappings file, else "Other measurement" when the label reads as a level or
amount, else "Not classified".

Trait gene sets (one per EFO trait URI; a row naming several URIs counts for each):
the Ensembl genes the Catalog maps the SNP into (SNP_GENE_IDS), else MAPPED_GENE,
both flanking genes when the SNP is intergenic ("A - B"). Haplotype ("a; b") and
interaction ("a x b") rows give one record per SNP when their parts line up. SNPs
in the extended MHC (chr6:25-34 Mb, GRCh38) are left out: its long-range LD makes
dozens of genes per signal. A trait's loci are its lead SNPs merged when within
1 Mb of each other on a chromosome.

Cluster test, per group of a column: the group's markers (``marker_selection.
group_markers``) against the genes measured here that the Catalog maps to any
trait (the background). Per trait (10-500 background genes; "Other measurement"
traits, mostly protein and metabolite levels, hidden unless asked): hypergeometric
p on the genes in common, Benjamini-Hochberg across the traits tested in the
group, and an empirical p from random gene sets matched to the markers on gene
degree (number of traits a gene is linked to; deciles of the background, seeded),
on the number of distinct loci the overlap covers. A trait is listed when the
overlap covers at least 3 loci. The degree match keeps long, SNP-dense and
much-studied genes (ABO, CFH, APOE) from making any list look trait-linked.
This is a lead, not heritability enrichment: that needs summary statistics
(LDSC-SEG, MAGMA, scDRS).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
import weakref
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from scview.core import results_store

logger = logging.getLogger(__name__)

RELEASE = "2026-10-22"
_BASE = "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/2026/10/22"
ASSOC_NAME = "gwas-catalog-associations_ontology-annotated-full.zip"
ASSOC_URL = f"{_BASE}/{ASSOC_NAME}"
ASSOC_SHA256 = "601a302f767aecf65583521605ee20e7ff26c69113b5cc9fbba0a3630f0acdf3"
ASSOC_BYTES = 73_038_962
MAPPINGS_NAME = "gwas-efo-trait-mappings.tsv"
MAPPINGS_URL = f"{_BASE}/{MAPPINGS_NAME}"
MAPPINGS_SHA256 = "f124b50053ec48031dab88e4e1a7989426b4f2e0b9aea4258a98e398ba38fc8d"
MAPPINGS_BYTES = 19_481_277
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
CACHE_VERSION = 1
LICENCE = "CC0 or EMBL-EBI terms of use"
CITATION = "Sollis et al. 2023, Nucleic Acids Res (PMID 36350656)"
ATTRIBUTION = f"NHGRI-EBI GWAS Catalog, release {RELEASE} (ebi.ac.uk/gwas)"
KIND = "gwas_traits"
STEP = "gwas_traits"
TAG = "result:gwas"

P_CUT = 5e-8
MLOG_CUT = -math.log10(P_CUT)
XMHC = (6, 25_000_000, 34_000_000)  # chr6, GRCh38
LOCUS_GAP = 1_000_000
MIN_LOCI = 3
MIN_TRAIT = 10
MAX_TRAIT = 500
MEASUREMENT_PARENT = "Other measurement"
N_DRAWS = 1000
REFINE_DRAWS = 10_000
REFINE_AT = 10  # traits with at most this many null draws reaching them get REFINE_DRAWS
N_BINS = 10  # gene-degree deciles
SEED = 0
N_MARKERS = 100
MIN_QUERY = 5
MAX_ROWS = 50  # traits listed per group
CHUNK_ROWS = 100_000

CAVEAT = (
    "Overlap between this cluster's marker genes and genes the GWAS Catalog maps near trait-associated "
    "variants. A mapped gene is usually the nearest gene, not a proven causal gene; long and heavily "
    "studied genes are over-represented, and one locus can contribute several genes. P-values compare "
    "against random gene sets matched for how many traits each gene is linked to, among genes measured "
    "here. Treat a hit as a lead for follow-up, not evidence that this cell type mediates the trait: "
    "that needs heritability methods on summary statistics (LDSC-SEG, MAGMA, scDRS)."
)

USECOLS = ["PUBMEDID", "CHR_ID", "CHR_POS", "MAPPED_GENE", "UPSTREAM_GENE_ID", "DOWNSTREAM_GENE_ID",
           "SNP_GENE_IDS", "INTERGENIC", "P-VALUE", "PVALUE_MLOG", "MAPPED_TRAIT", "MAPPED_TRAIT_URI",
           "DISEASE/TRAIT"]
_CHROMS = {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24, "MT": 25, "M": 25}
CHROM_NAMES = {v: k for k, v in _CHROMS.items() if k != "M"}
_MULTI = re.compile(r"\s*;\s*|\s+x\s+")
_ENSG = re.compile(r"^ENSG\d+$")


class GwasUnavailable(RuntimeError):
    """The catalog is not on disk and could not (or may not) be fetched or built."""


def auto_download() -> bool:
    """``GWAS_AUTO_DOWNLOAD`` (default on)."""
    return os.environ.get("GWAS_AUTO_DOWNLOAD", "true").strip().lower() not in {"0", "false", "no", "off"}


def gwas_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "gwas"


def cache_path(dest: Path) -> Path:
    return dest / f"gwas_catalog_{RELEASE}_v{CACHE_VERSION}.npz"


def meta_path(dest: Path) -> Path:
    return dest / f"gwas_catalog_{RELEASE}_v{CACHE_VERSION}.json"


def read_meta(dest: Path) -> dict | None:
    """What was built and when, or None when the cache is not on disk."""
    if not cache_path(dest).is_file():
        return None
    try:
        return json.loads(meta_path(dest).read_text())
    except (OSError, ValueError):
        return {}


def raw_present(dest: Path) -> bool:
    return (dest / ASSOC_NAME).is_file() and (dest / MAPPINGS_NAME).is_file()


NOT_DOWNLOADED = (
    f"The GWAS Catalog ({RELEASE} release) is not downloaded yet. Download it once in the Traits tab "
    f"(about {round((ASSOC_BYTES + MAPPINGS_BYTES) / 1e6)} MB from ftp.ebi.ac.uk; preparing it takes a "
    "minute or two)."
)


# ---------------------------------------------------------------------------
# Parsing the associations file
# ---------------------------------------------------------------------------


def _split(value: Any, sep: str = ",") -> list[str]:
    if not isinstance(value, str):
        return []
    return [s.strip() for s in value.split(sep) if s.strip()]


def _gene_list(part: str) -> list[str]:
    """Symbols of one MAPPED_GENE part: "A - B" (intergenic flanks), "A, B" or "A"."""
    sep = " - " if " - " in part else ","
    return [g for g in _split(part, sep) if g not in {"NR", "-"}]


def _chrom(value: str) -> int:
    return _CHROMS.get(value.strip().upper().removeprefix("CHR"), 0)


def _pos(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def row_records(chr_id: Any, chr_pos: Any, mapped: Any, up_id: Any, down_id: Any, snp_ids: Any,
                intergenic: Any) -> list[tuple[str, str, int, int]]:
    """(symbol, ensembl or "", chrom, pos) per gene a row maps, one SNP at a time.
    Symbol "" means an Ensembl id whose symbol is resolved later."""
    if not isinstance(mapped, str) and not isinstance(snp_ids, str):
        return []
    chr_s = chr_id if isinstance(chr_id, str) else ""
    pos_s = chr_pos if isinstance(chr_pos, str) else ""
    if _MULTI.search(chr_s) or _MULTI.search(pos_s):
        chrs, poss = _MULTI.split(chr_s.strip()), _MULTI.split(pos_s.strip())
        parts = _MULTI.split(mapped.strip()) if isinstance(mapped, str) else []
        if not (len(chrs) == len(poss) == len(parts)):
            return []
        out = []
        for c, p, part in zip(chrs, poss, parts):
            out.extend((g, "", _chrom(c), _pos(p)) for g in _gene_list(part))
        return out
    c, p = _chrom(chr_s), _pos(pos_s)
    ids = [i for i in _split(snp_ids) if _ENSG.match(i)]
    symbols = _gene_list(mapped) if isinstance(mapped, str) else []
    if ids:
        if len(symbols) == len(ids):
            return [(s, i, c, p) for s, i in zip(symbols, ids)]
        # symbols and ids do not line up: ids resolved through other rows, symbols as they are
        return [*(("", i, c, p) for i in ids), *((s, "", c, p) for s in symbols)]
    is_intergenic = str(intergenic).strip() == "1" or (isinstance(mapped, str) and " - " in mapped)
    if is_intergenic and len(symbols) == 2:
        flank = [i if isinstance(i, str) and _ENSG.match(i.strip()) else "" for i in (up_id, down_id)]
        return [(symbols[0], flank[0].strip(), c, p), (symbols[1], flank[1].strip(), c, p)]
    return [(s, "", c, p) for s in symbols]


@dataclass
class _Builder:
    genes: dict[str, int] = field(default_factory=dict)  # symbol (or "@ENSG" placeholder) -> id
    ens_votes: dict[tuple[str, str], int] = field(default_factory=dict)  # (symbol, ensembl) -> rows
    traits: dict[str, int] = field(default_factory=dict)
    trait_labels: dict[str, str] = field(default_factory=dict)
    reported_parent: dict[str, str] = field(default_factory=dict)  # reported trait -> parent (mappings)
    parent_votes: dict[tuple[str, str], int] = field(default_factory=dict)  # (uri, parent) -> rows
    parts: list[pd.DataFrame] = field(default_factory=list)
    n_rows: int = 0
    n_significant: int = 0
    n_xmhc: int = 0
    n_no_position: int = 0
    n_unparsed: int = 0

    def gene_id(self, symbol: str, ens: str) -> int:
        key = symbol or f"@{ens}"
        if ens and symbol:
            self.ens_votes[(symbol, ens)] = self.ens_votes.get((symbol, ens), 0) + 1
        k = self.genes.get(key)
        if k is None:
            k = self.genes[key] = len(self.genes)
        return k

    def trait_id(self, uri: str) -> int:
        k = self.traits.get(uri)
        if k is None:
            k = self.traits[uri] = len(self.traits)
        return k

    def add_chunk(self, df: pd.DataFrame) -> None:
        self.n_rows += len(df)
        p = pd.to_numeric(df["P-VALUE"], errors="coerce")
        mlog = pd.to_numeric(df["PVALUE_MLOG"], errors="coerce")
        mlog = mlog.where(mlog.notna(), -np.log10(p.where(p > 0)))
        keep = ((p <= P_CUT) | (p.isna() & (mlog >= MLOG_CUT - 1e-9))) & df["MAPPED_TRAIT_URI"].notna()
        df = df[keep.to_numpy()]
        mlog = mlog[keep].fillna(MLOG_CUT).to_numpy(np.float64)
        self.n_significant += len(df)
        pmid = pd.to_numeric(df["PUBMEDID"], errors="coerce").fillna(0).to_numpy(np.int64)
        tr, gn, ch, ps, ml, pm = [], [], [], [], [], []
        cols = [df[c].to_numpy(object) for c in ("CHR_ID", "CHR_POS", "MAPPED_GENE", "UPSTREAM_GENE_ID",
                                                 "DOWNSTREAM_GENE_ID", "SNP_GENE_IDS", "INTERGENIC",
                                                 "MAPPED_TRAIT_URI", "MAPPED_TRAIT", "DISEASE/TRAIT")]
        for k in range(len(df)):
            recs = row_records(*(c[k] for c in cols[:7]))
            if not recs:
                self.n_unparsed += 1
                continue
            uris = _split(cols[7][k])
            labels = _split(cols[8][k]) if isinstance(cols[8][k], str) else []
            if len(labels) == len(uris):
                for u, lab in zip(uris, labels):
                    self.trait_labels.setdefault(u, lab)
            rp = self.reported_parent.get(cols[9][k]) if isinstance(cols[9][k], str) else None
            if rp:
                for u in uris:
                    self.parent_votes[(u, rp)] = self.parent_votes.get((u, rp), 0) + 1
            tids = [self.trait_id(u) for u in uris]
            for sym, ens, c, pos in recs:
                if c == 0 or pos < 0:
                    self.n_no_position += 1
                    continue
                if c == XMHC[0] and XMHC[1] <= pos <= XMHC[2]:
                    self.n_xmhc += 1
                    continue
                g = self.gene_id(sym, ens)
                for t in tids:
                    tr.append(t)
                    gn.append(g)
                    ch.append(c)
                    ps.append(pos)
                    ml.append(mlog[k])
                    pm.append(pmid[k])
        if tr:
            part = pd.DataFrame({"trait": np.asarray(tr, np.int32), "gene": np.asarray(gn, np.int32),
                                 "chrom": np.asarray(ch, np.int8), "pos": np.asarray(ps, np.int32),
                                 "mlog": np.asarray(ml, np.float32), "pmid": np.asarray(pm, np.int32)})
            self.parts.append(_dedupe(part))


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trait, gene, SNP position): the smallest p and its study."""
    df = df.sort_values("mlog", ascending=False, kind="stable")
    return df.drop_duplicates(["trait", "gene", "chrom", "pos"], keep="first")


def assign_loci(trait: np.ndarray, chrom: np.ndarray, pos: np.ndarray, gap: int = LOCUS_GAP) -> np.ndarray:
    """A locus id per row: per trait, its SNP positions on a chromosome merged when
    within ``gap`` of the previous one (single linkage), ids 0.. across all traits."""
    n = trait.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.lexsort((pos, chrom, trait))
    t, c, p = trait[order], chrom[order], pos[order].astype(np.int64)
    new = np.ones(n, dtype=bool)
    new[1:] = (t[1:] != t[:-1]) | (c[1:] != c[:-1]) | ((p[1:] - p[:-1]) > gap)
    ids = np.cumsum(new) - 1
    out = np.empty(n, dtype=np.int32)
    out[order] = ids
    return out


def read_mappings(source: Any) -> tuple[pd.DataFrame, dict[str, str]]:
    """The EFO trait mappings file: (uri, label, parent per trait URI, first row each;
    reported trait -> its parent term)."""
    m = pd.read_csv(source, sep="\t", dtype=str, usecols=["Disease trait", "EFO term", "EFO URI", "Parent term"])
    m.columns = [c.strip() for c in m.columns]
    m = m.dropna(subset=["EFO URI"])
    rep = m.dropna(subset=["Disease trait", "Parent term"])
    rep = rep[rep["Parent term"] != "NR"].drop_duplicates("Disease trait")
    m = m.drop_duplicates("EFO URI")
    traits = pd.DataFrame({"uri": m["EFO URI"].str.strip(), "label": m["EFO term"].fillna(""),
                           "parent": m["Parent term"].fillna("")})
    return traits, dict(zip(rep["Disease trait"], rep["Parent term"]))


_MEASURE_RX = re.compile(r"\bmeasurement\b|^amount of |\blevels?\b|\bratio\b", re.IGNORECASE)
UNCLASSIFIED = "Not classified"


def unclassified_parent(label: str) -> str:
    """A parent term for a trait the mappings file does not list yet (newer than its
    curation): "Other measurement" when the label reads as a level or amount (most
    such traits are new protein-level studies), else "Not classified"."""
    return MEASUREMENT_PARENT if _MEASURE_RX.search(label) else UNCLASSIFIED


def _member(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.endswith(".tsv")]
    if len(names) != 1:
        raise ValueError(f"expected one .tsv in the GWAS Catalog zip, found {names}")
    return names[0]


@dataclass
class Catalog:
    """Genome-wide significant GWAS Catalog records, one per (trait, gene, SNP)."""

    symbols: np.ndarray  # str per gene
    ensembl: np.ndarray  # str per gene ("" when unknown)
    trait_uri: np.ndarray
    trait_label: np.ndarray
    trait_parent: np.ndarray
    trait: np.ndarray  # int32 per record
    gene: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    mlog: np.ndarray
    pmid: np.ndarray
    locus: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    _lookup: dict[str, int] | None = field(default=None, repr=False)
    _mats: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_genes(self) -> int:
        return int(self.symbols.size)

    @property
    def n_traits(self) -> int:
        return int(self.trait_uri.size)

    @property
    def n_loci(self) -> int:
        return int(self.locus.max()) + 1 if self.locus.size else 0

    def lookup(self) -> dict[str, int]:
        """Upper-case symbol (and Ensembl id) -> gene index."""
        if self._lookup is None:
            out: dict[str, int] = {}
            for k, s in enumerate(self.symbols.tolist()):
                out.setdefault(str(s).upper(), k)
            for k, e in enumerate(self.ensembl.tolist()):
                if e:
                    out.setdefault(str(e).upper(), k)
            self._lookup = out
        return self._lookup

    def gene_trait(self) -> sp.csr_matrix:
        """Genes x traits, 1 where the Catalog maps the gene for the trait."""
        if "gt" not in self._mats:
            m = sp.csr_matrix((np.ones(self.trait.size, np.int8), (self.gene, self.trait)),
                              shape=(self.n_genes, self.n_traits))
            m.data[:] = 1
            self._mats["gt"] = m
        return self._mats["gt"]

    def gene_locus(self) -> sp.csr_matrix:
        """Genes x loci, 1 where a gene is mapped at the locus."""
        if "gl" not in self._mats:
            m = sp.csr_matrix((np.ones(self.locus.size, np.int8), (self.gene, self.locus)),
                              shape=(self.n_genes, self.n_loci))
            m.data[:] = 1
            self._mats["gl"] = m
        return self._mats["gl"]

    def locus_trait(self) -> np.ndarray:
        """The trait of each locus."""
        if "lt" not in self._mats:
            lt = np.zeros(self.n_loci, dtype=np.int32)
            lt[self.locus] = self.trait
            self._mats["lt"] = lt
        return self._mats["lt"]


def build(assoc: Any, mappings: Any, *, progress: Callable[[dict], None] | None = None,
          chunk_rows: int = CHUNK_ROWS) -> Catalog:
    """Parse the associations zip (path or file object) and the trait mappings into
    a ``Catalog``. Raises ValueError when they are not the GWAS Catalog files."""
    maps, reported_parent = read_mappings(mappings)
    b = _Builder(reported_parent=reported_parent)
    with zipfile.ZipFile(assoc) as zf, zf.open(_member(zf)) as fh:
        reader = pd.read_csv(fh, sep="\t", dtype=str, chunksize=chunk_rows, low_memory=True,
                             usecols=lambda c: c.strip() in USECOLS)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            missing = set(USECOLS) - set(chunk.columns)
            if missing:
                raise ValueError(f"The GWAS Catalog file lacks columns {sorted(missing)}")
            b.add_chunk(chunk)
            if progress:
                progress({"stage": "parse", "rows": b.n_rows})
    if not b.parts:
        raise ValueError("No genome-wide significant associations with a mapped gene in the file.")
    rec = _dedupe(pd.concat(b.parts, ignore_index=True))
    b.parts.clear()

    # Genes: placeholders for Ensembl ids without a symbol take the symbol seen with that id.
    ens_of: dict[str, str] = {}
    sym_of: dict[str, str] = {}
    for (s, e), _n in sorted(b.ens_votes.items(), key=lambda kv: -kv[1]):
        ens_of.setdefault(s, e)
        sym_of.setdefault(e, s)
    keys = list(b.genes)
    names = [k if not k.startswith("@") else sym_of.get(k[1:], "") for k in keys]
    final: dict[str, int] = {}
    remap = np.full(len(keys), -1, dtype=np.int64)
    for k, name in enumerate(names):
        if name:
            remap[k] = final.setdefault(name, len(final))
    g = remap[rec["gene"].to_numpy()]
    rec = rec[g >= 0].assign(gene=g[g >= 0].astype(np.int32))
    rec = _dedupe(rec).reset_index(drop=True)
    symbols = np.asarray(list(final), dtype=str)
    ensembl = np.asarray([ens_of.get(s, "") for s in final], dtype=str)

    by_uri = maps.set_index("uri")
    voted: dict[str, str] = {}
    for (u, parent), _n in sorted(b.parent_votes.items(), key=lambda kv: -kv[1]):
        voted.setdefault(u, parent)
    uris = list(b.traits)
    labels, parents = [], []
    for u in uris:
        if u in by_uri.index:
            labels.append(str(by_uri.at[u, "label"]) or b.trait_labels.get(u, u.rsplit("/", 1)[-1]))
            parents.append(str(by_uri.at[u, "parent"]))
        else:
            label = b.trait_labels.get(u, u.rsplit("/", 1)[-1])
            labels.append(label)
            parents.append(voted.get(u) or unclassified_parent(label))
    trait = rec["trait"].to_numpy(np.int32)
    chrom = rec["chrom"].to_numpy(np.int8)
    pos = rec["pos"].to_numpy(np.int32)
    cat = Catalog(
        symbols=symbols, ensembl=ensembl, trait_uri=np.asarray(uris, dtype=str),
        trait_label=np.asarray(labels, dtype=str), trait_parent=np.asarray(parents, dtype=str),
        trait=trait, gene=rec["gene"].to_numpy(np.int32), chrom=chrom, pos=pos,
        mlog=rec["mlog"].to_numpy(np.float32), pmid=rec["pmid"].to_numpy(np.int32),
        locus=assign_loci(trait, chrom, pos),
    )
    cat.meta = {
        "release": RELEASE, "n_rows": b.n_rows, "n_significant": b.n_significant, "n_xmhc": b.n_xmhc,
        "n_no_position": b.n_no_position, "n_unparsed": b.n_unparsed, "n_records": int(trait.size),
        "n_genes": cat.n_genes, "n_traits": int(np.unique(trait).size), "n_loci": cat.n_loci,
    }
    return cat


_ARRAYS = ("symbols", "ensembl", "trait_uri", "trait_label", "trait_parent", "trait", "gene", "chrom",
           "pos", "mlog", "pmid", "locus")


def save(cat: Catalog, dest: Path, extra_meta: dict | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (cache_path(dest).name + ".part")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **{k: getattr(cat, k) for k in _ARRAYS})
        tmp.replace(cache_path(dest))
    finally:
        tmp.unlink(missing_ok=True)
    meta = {**cat.meta, **(extra_meta or {}), "cache_version": CACHE_VERSION, "licence": LICENCE,
            "citation": CITATION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta_path(dest).write_text(json.dumps(meta, indent=1))
    cat.meta = meta


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(client: httpx.Client, url: str, sha256: str, total: int, out: Path,
           progress: Callable[[dict], None] | None) -> None:
    """Stream ``url`` to ``out`` (via .part), checking the sha256."""
    tmp = out.with_name(out.name + ".part")
    digest = hashlib.sha256()
    done = 0
    last = 0.0
    try:
        with client.stream("GET", url) as resp, open(tmp, "wb") as fh:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(1 << 16):
                done += len(chunk)
                if done > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"{out.name} exceeds {MAX_DOWNLOAD_BYTES} bytes")
                digest.update(chunk)
                fh.write(chunk)
                now = time.monotonic()
                if progress and now - last > 0.5:
                    last = now
                    progress({"stage": "download", "file": out.name, "bytes": done, "total": total})
        if digest.hexdigest() != sha256:
            raise ValueError(f"{out.name}: sha256 {digest.hexdigest()[:12]}... is not the release scView "
                             f"expects ({sha256[:12]}...)")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(dest: Path, *, allow_download: bool = True, client: httpx.Client | None = None,
            progress: Callable[[dict], None] | None = None) -> Catalog:
    """Download the two pinned files when missing (and allowed), check them, and
    build the cache. Files placed by hand are used as they are once their sha256 matches."""
    dest.mkdir(parents=True, exist_ok=True)
    assoc, maps = dest / ASSOC_NAME, dest / MAPPINGS_NAME
    if not (assoc.is_file() and maps.is_file()) and not allow_download:
        raise GwasUnavailable(
            f"The GWAS Catalog is not downloaded and GWAS_AUTO_DOWNLOAD is off. Place {ASSOC_NAME} and "
            f"{MAPPINGS_NAME} from {_BASE}/ in {dest}, or turn the switch on.")
    own = client is None
    try:
        for path, url, sha, size in ((assoc, ASSOC_URL, ASSOC_SHA256, ASSOC_BYTES),
                                     (maps, MAPPINGS_URL, MAPPINGS_SHA256, MAPPINGS_BYTES)):
            if path.is_file():
                if progress:
                    progress({"stage": "check", "file": path.name})
                got = _sha256_file(path)
                if got != sha:
                    raise GwasUnavailable(f"{path} is not the {RELEASE} release scView expects (sha256 "
                                          f"{got[:12]}..., want {sha[:12]}...); delete it to fetch it again.")
                continue
            client = client or httpx.Client(timeout=600.0, follow_redirects=True)
            _fetch(client, url, sha, size, path, progress)
    finally:
        if own and client is not None:
            client.close()
    if progress:
        progress({"stage": "parse", "rows": 0})
    t0 = time.time()
    cat = build(assoc, maps, progress=progress)
    save(cat, dest, {"assoc_url": ASSOC_URL, "assoc_sha256": ASSOC_SHA256, "mappings_url": MAPPINGS_URL,
                     "mappings_sha256": MAPPINGS_SHA256, "build_seconds": round(time.time() - t0, 1)})
    logger.info("Built GWAS Catalog %s: %d genes, %d traits, %d records", RELEASE, cat.n_genes,
                cat.meta["n_traits"], cat.trait.size)
    return cat


_lock = threading.Lock()  # the loaded cache (held briefly)
_prepare_lock = threading.Lock()  # one download / build at a time (held for minutes)
_loaded: dict[str, tuple[float, Catalog]] = {}


def ensure(dest: Path, *, allow_download: bool | None = None, client: httpx.Client | None = None,
           progress: Callable[[dict], None] | None = None) -> Catalog:
    """The catalog, prepared first when the cache is missing (from files on disk, or
    downloaded when allowed). Raises ``GwasUnavailable`` saying what to do."""
    allow = auto_download() if allow_download is None else allow_download
    path = cache_path(dest)
    with _prepare_lock:  # never _lock: a status or load request must not wait on a download
        if not path.is_file():
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".prepare.lock", "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    if not path.is_file():
                        prepare(dest, allow_download=allow, client=client, progress=progress)
            except GwasUnavailable:
                raise
            except Exception as e:
                logger.warning("Could not prepare the GWAS Catalog: %s", e)
                raise GwasUnavailable(
                    f"Fetching or reading the GWAS Catalog failed ({e}). scView fetches it once from "
                    "ftp.ebi.ac.uk; check the internet connection and try again.") from e
    with _lock:
        return _load_locked(dest)


def _load_locked(dest: Path) -> Catalog:
    path = cache_path(dest)
    mtime = path.stat().st_mtime
    cached = _loaded.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    with np.load(path, allow_pickle=False) as z:
        cat = Catalog(**{k: z[k] for k in _ARRAYS}, meta=read_meta(dest) or {})
    _loaded.clear()
    _loaded[str(path)] = (mtime, cat)
    return cat


def load(dest: Path) -> Catalog:
    """The built catalog; ``GwasUnavailable`` (with ``NOT_DOWNLOADED``) when it is not built."""
    with _lock:
        if not cache_path(dest).is_file():
            raise GwasUnavailable(NOT_DOWNLOADED)
        return _load_locked(dest)


# ---------------------------------------------------------------------------
# One gene's traits
# ---------------------------------------------------------------------------

MAX_GENE_TRAITS = 50
MAX_PMIDS = 5


def p_from_mlog(mlog: float) -> float:
    return float(10.0 ** (-float(mlog)))


def gene_traits(cat: Catalog, symbol: str, limit: int = MAX_GENE_TRAITS) -> dict[str, Any] | None:
    """The traits the Catalog maps ``symbol`` (or an Ensembl id) for, smallest p first,
    with the loci and PubMed ids; None when the gene is not in the Catalog."""
    g = cat.lookup().get(str(symbol).strip().upper())
    if g is None:
        return None
    rows = np.flatnonzero(cat.gene == g)
    df = pd.DataFrame({"trait": cat.trait[rows], "locus": cat.locus[rows], "mlog": cat.mlog[rows],
                       "pmid": cat.pmid[rows]}).sort_values("mlog", ascending=False, kind="stable")
    out = []
    for t, sub in df.groupby("trait", sort=False):
        pm = [int(x) for x in dict.fromkeys(sub["pmid"].tolist()) if x > 0][:MAX_PMIDS]
        out.append({
            "uri": str(cat.trait_uri[t]), "trait": str(cat.trait_label[t]), "parent": str(cat.trait_parent[t]),
            "mlog": round(float(sub["mlog"].iloc[0]), 3), "p": p_from_mlog(sub["mlog"].iloc[0]),
            "n_loci": int(sub["locus"].nunique()), "pmids": pm,
        })
    out.sort(key=lambda r: -r["mlog"])
    return {"gene": str(cat.symbols[g]), "ensembl": str(cat.ensembl[g]) or None, "n_traits": len(out),
            "traits": out[:limit], "shown": min(limit, len(out)), "release": RELEASE,
            "attribution": ATTRIBUTION, "citation": CITATION}


# ---------------------------------------------------------------------------
# Cluster test
# ---------------------------------------------------------------------------


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p (as statsmodels fdr_bh)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def hypergeom_sf(k: np.ndarray, N: int, K: np.ndarray, n: int) -> np.ndarray:
    """P(X >= k) for X ~ Hypergeometric(N, K, n), per trait."""
    return np.clip(stats.hypergeom.sf(np.asarray(k) - 1, N, np.asarray(K), n), 0.0, 1.0)


def odds_ratio(k: np.ndarray, N: int, K: np.ndarray, n: int) -> np.ndarray:
    """Sample odds ratio of the 2x2 table, 0.5 added to every cell when one is zero."""
    a = np.asarray(k, float)
    b = n - a
    c = np.asarray(K, float) - a
    d = N - n - c
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = (np.where(zero, x + 0.5, x) for x in (a, b, c, d))
    return (a * d) / (b * c)


def degree_bins(degree: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-size degree bins (deciles) by rank; ties split by gene order (deterministic)."""
    n = degree.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(degree, kind="stable")] = np.arange(n)
    return np.minimum((rank * n_bins) // n, n_bins - 1).astype(np.int32)


def random_sets(bins: np.ndarray, query: np.ndarray, n_draws: int, seed: int) -> sp.csr_matrix:
    """``n_draws`` random sets (rows) over the background, each with as many genes
    from every degree bin as ``query`` has, drawn without replacement, seeded."""
    rng = np.random.default_rng(seed)
    need = np.bincount(bins[query], minlength=int(bins.max()) + 1 if bins.size else 0)
    pools = [np.flatnonzero(bins == b) for b in range(need.size)]
    m = int(need.sum())
    cols = np.empty((n_draws, m), dtype=np.int64)
    for d in range(n_draws):
        at = 0
        for b, k in enumerate(need):
            if k:
                cols[d, at:at + k] = rng.choice(pools[b], size=int(k), replace=False)
                at += k
    rows = np.repeat(np.arange(n_draws), m)
    return sp.csr_matrix((np.ones(rows.size, np.int32), (rows, cols.ravel())), shape=(n_draws, bins.size))


@dataclass
class Background:
    """The catalog restricted to the genes measured in a dataset."""

    genes: np.ndarray  # catalog gene index per background gene
    gt: sp.csr_matrix  # background genes x traits
    gl: sp.csr_matrix  # background genes x loci
    sizes: np.ndarray  # background genes per trait
    degree: np.ndarray  # traits per background gene
    bins: np.ndarray
    index: dict[int, int]  # catalog gene -> background row


def background(cat: Catalog, measured: Iterable[str]) -> Background:
    look = cat.lookup()
    idx = sorted({look[s.upper()] for s in (str(x) for x in measured) if s.upper() in look})
    genes = np.asarray(idx, dtype=np.int64)
    gt = cat.gene_trait()[genes]
    gl = cat.gene_locus()[genes]
    degree = np.asarray(gt.sum(axis=1)).ravel()
    return Background(genes=genes, gt=gt.tocsc(), gl=gl, sizes=np.asarray(gt.sum(axis=0)).ravel(),
                      degree=degree, bins=degree_bins(degree), index={int(g): k for k, g in enumerate(idx)})


def eligible_traits(cat: Catalog, bg: Background, include_measurements: bool) -> np.ndarray:
    ok = (bg.sizes >= MIN_TRAIT) & (bg.sizes <= MAX_TRAIT)
    if not include_measurements:
        ok &= cat.trait_parent != MEASUREMENT_PARENT
    return np.flatnonzero(ok)


def loci_hits(gl: sp.csr_matrix, sets: sp.csr_matrix, locus_trait: np.ndarray, traits: np.ndarray,
              n_traits: int) -> np.ndarray:
    """Distinct loci of ``traits`` each set (row) covers: sets x len(traits)."""
    want = np.zeros(n_traits, dtype=bool)
    want[traits] = True
    cols = np.flatnonzero(want[locus_trait])
    hit = (sets @ gl[:, cols]).tocsr()
    hit.data[:] = 1
    col_of = np.full(n_traits, -1, dtype=np.int64)
    col_of[traits] = np.arange(traits.size)
    agg = sp.csr_matrix((np.ones(cols.size, np.int32), (np.arange(cols.size), col_of[locus_trait[cols]])),
                        shape=(cols.size, traits.size))
    return np.asarray((hit.astype(np.int32) @ agg).todense())


def group_traits(cat: Catalog, bg: Background, query_rows: np.ndarray, *, include_measurements: bool = False,
                 n_draws: int = N_DRAWS, refine_draws: int = REFINE_DRAWS, seed: int = SEED,
                 max_rows: int = MAX_ROWS, min_loci: int = MIN_LOCI) -> dict[str, Any]:
    """One group's traits: rows (overlap >= ``min_loci`` loci) smallest p first, and counts."""
    traits = eligible_traits(cat, bg, include_measurements)
    N, n = int(bg.genes.size), int(query_rows.size)
    base = {"n_query": n, "n_tested": int(traits.size), "n_with_loci": 0, "n_leads": 0, "rows": []}
    if n < MIN_QUERY or traits.size == 0:
        return base
    qmask = np.zeros(N, dtype=np.int32)
    qmask[query_rows] = 1
    qvec = sp.csr_matrix(qmask)
    k = np.asarray((qvec @ bg.gt[:, traits]).todense()).ravel()
    K = bg.sizes[traits]
    p = hypergeom_sf(k, N, K, n)
    padj = bh(p)
    lt = cat.locus_trait()
    loci = loci_hits(bg.gl, qvec, lt, traits, cat.n_traits).ravel()
    cand = np.flatnonzero(loci >= min_loci)
    base["n_with_loci"] = int(cand.size)
    if cand.size == 0:
        return base
    emp = np.ones(cand.size)
    draws = np.full(cand.size, n_draws, dtype=np.int64)
    null = loci_hits(bg.gl, random_sets(bg.bins, query_rows, n_draws, seed), lt, traits[cand], cat.n_traits)
    ge = (null >= loci[cand][None, :]).sum(axis=0)
    emp[:] = (1 + ge) / (1 + n_draws)
    refine = np.flatnonzero(ge <= REFINE_AT)
    if refine.size and refine_draws > n_draws:
        null2 = loci_hits(bg.gl, random_sets(bg.bins, query_rows, refine_draws, seed + 1), lt,
                          traits[cand[refine]], cat.n_traits)
        ge2 = (null2 >= loci[cand[refine]][None, :]).sum(axis=0)
        emp[refine] = (1 + ge2) / (1 + refine_draws)
        draws[refine] = refine_draws
    orr = odds_ratio(k[cand], N, K[cand], n)
    order = np.lexsort((-loci[cand], p[cand]))
    rows = []
    for j in order:
        t = int(traits[cand[j]])
        members = bg.gt[:, t].indices  # background rows in the trait
        over = np.intersect1d(members, query_rows)
        rows.append({
            "uri": str(cat.trait_uri[t]), "trait": str(cat.trait_label[t]), "parent": str(cat.trait_parent[t]),
            "n_trait_genes": int(K[cand[j]]), "n_overlap": int(k[cand[j]]), "n_loci": int(loci[cand[j]]),
            "genes": [str(cat.symbols[bg.genes[r]]) for r in over],
            "odds_ratio": float(orr[j]), "p": float(p[cand[j]]), "p_adj": float(padj[cand[j]]),
            "p_empirical": float(emp[j]), "n_draws": int(draws[j]),
            "lead": bool(padj[cand[j]] < 0.05 and emp[j] < 0.05),
        })
    base["n_leads"] = sum(r["lead"] for r in rows)
    base["rows"] = rows[:max_rows]
    return base


def human_map(symbols: Iterable[str], species: str,
              to_human: Callable[[list[str]], dict[str, list[str]]] | None = None) -> dict[str, list[str]] | None:
    """Dataset symbol -> human symbols (HCOP) for a mouse dataset; None for human."""
    if species != "mouse":
        return None
    if to_human is None:
        from scview.core import orthologs

        to_human = orthologs.mouse_to_human
    return to_human(list(dict.fromkeys(str(s) for s in symbols)))


def humanise(genes: Iterable[str], m: dict[str, list[str]] | None) -> tuple[list[str], dict[str, str]]:
    """(human genes, upper-case human symbol -> the dataset gene it came from)."""
    q: list[str] = []
    back: dict[str, str] = {}
    for g in genes:
        for h in ([g] if m is None else m.get(g, [])):
            if h.upper() not in back:
                back[h.upper()] = g
                q.append(h)
    return q, back


def run(cat: Catalog, adata, column: str, *, include_measurements: bool = False, n_markers: int = N_MARKERS,
        species: str = "human", n_draws: int = N_DRAWS, refine_draws: int = REFINE_DRAWS, seed: int = SEED,
        to_human: Callable[[list[str]], dict[str, list[str]]] | None = None,
        dataset_id: str | None = None) -> dict[str, Any]:
    """Every group of ``column``: its markers' trait overlaps. KeyError for a missing
    column, ValueError without markers."""
    from scview.core.marker_selection import group_markers

    t0 = time.time()
    if column not in adata.obs.columns:
        raise KeyError(f"Column '{column}' not found.")
    markers = group_markers(adata, column, n=n_markers)
    if not markers:
        raise ValueError(f"No markers computed for '{column}' yet: compute them in the Markers tab first.")
    var = adata.raw.var_names if adata.raw is not None else adata.var_names
    names = list(dict.fromkeys([*map(str, adata.var_names), *map(str, var)]))
    m = human_map(names, species, to_human)
    measured, _ = humanise(names, m)
    bg = background(cat, measured)
    look = cat.lookup()
    groups = []
    for group, genes in markers.items():
        q, back = humanise(genes, m)
        rows = sorted({bg.index[look[g.upper()]] for g in q if look.get(g.upper(), -1) in bg.index})
        res = group_traits(cat, bg, np.asarray(rows, dtype=np.int64), include_measurements=include_measurements,
                           n_draws=n_draws, refine_draws=refine_draws, seed=seed)
        for r in res["rows"]:
            r["dataset_genes"] = [back.get(g.upper(), g) for g in r["genes"]]
        groups.append({"group": str(group), "n_markers": len(genes), **res})
    return {
        "result_id": uuid.uuid4().hex[:12], "dataset_id": dataset_id, "column": column,
        "include_measurements": include_measurements, "n_markers": n_markers, "species": species,
        "mapping": ("mouse genes mapped to human with HCOP orthologs" if species == "mouse" else None),
        "n_background": int(bg.genes.size), "n_measured": len(names),
        "min_loci": MIN_LOCI, "trait_size": [MIN_TRAIT, MAX_TRAIT], "n_draws": n_draws,
        "refine_draws": refine_draws, "seed": seed, "max_rows": MAX_ROWS,
        "release": RELEASE, "attribution": ATTRIBUTION, "citation": CITATION, "caveat": CAVEAT,
        "groups": groups, "elapsed_seconds": round(time.time() - t0, 2), "created_at": time.time(),
        "cells": results_store.cells_hash(adata),
        "method": (
            f"GWAS Catalog {RELEASE}, associations with p <= 5e-8, genes mapped to each EFO trait (both "
            "flanking genes for intergenic SNPs), extended MHC (chr6:25-34 Mb) left out, lead SNPs within "
            f"1 Mb merged into loci. Query: each group's top {n_markers} markers (>= 20% of the group, "
            "adjusted p < 0.05, by fold change, no ribosomal or mitochondrial genes). Background: genes "
            f"measured here that the Catalog maps to any trait. Traits with {MIN_TRAIT}-{MAX_TRAIT} "
            "background genes; hypergeometric p, Benjamini-Hochberg across the traits tested per group; "
            f"empirical p on distinct loci from {n_draws} random gene sets matched on gene-degree decile "
            f"(seed {seed}; {refine_draws} for the strongest). Listed when the overlap covers >= {MIN_LOCI} loci."
        ),
    }


# ---------------------------------------------------------------------------
# Kept results and the co-pilot
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def record(adata, result: dict[str, Any]) -> None:
    """The run in provenance (History): adds no data, so other results stay current."""
    from scview.core import provenance

    provenance.record_step(
        adata, step=STEP, tool=f"scview.gwas_catalog (GWAS Catalog {RELEASE})",
        params={"column": result["column"], "n_markers": result["n_markers"],
                "include_measurements": result["include_measurements"], "release": RELEASE,
                "species": result["species"], "n_background": result["n_background"],
                "n_draws": result["n_draws"], "seed": result["seed"],
                "n_leads": sum(g["n_leads"] for g in result["groups"])},
        resources={"seconds": result["elapsed_seconds"]},
    )


def remember(adaptor, result: dict[str, Any]) -> None:
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # a test double
        pass
    results_store.keep(adaptor, KIND, result, key=result["column"], result_id=result["result_id"],
                       label=f"GWAS trait links by {result['column']} ({RELEASE})")


def last(adaptor, column: str | None = None) -> dict[str, Any] | None:
    """The last run (on ``column``, when given) while the dataset holds the same cells."""
    r = results_store.remembered(_last, _last_lock, adaptor, KIND)
    if r is not None and column is not None and r.get("column") != column:
        r = results_store.latest(adaptor, KIND, column)
    if r is None:
        return None
    try:
        if r.get("cells") and r["cells"] != results_store.cells_hash(adaptor.adata):
            return None
    except Exception:  # a test double
        pass
    return r


def _fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def grounding_lines(adaptor, highlighted: dict | None, n: int = 5) -> tuple[list[str], str]:
    """The highlighted group's top traits from the last run on its column; nothing
    before a run, or for another column."""
    if not isinstance(highlighted, dict) or not highlighted.get("value"):
        return [], ""
    column, value = str(highlighted.get("column") or ""), str(highlighted["value"])
    r = last(adaptor, column)
    if r is None or r.get("column") != column:
        return [], ""
    g = next((x for x in r["groups"] if x["group"] == value), None)
    if g is None:
        return [], ""
    lines = [f"\n## GWAS trait links ({column} = {value})"]
    head = (f"- GWAS Catalog {r['release']} traits overlapping the {g['n_markers']} markers of {column} {value} "
            f"(background {r['n_background']:,} genes; overlap counted in loci, >= {r['min_loci']}). Caveat: a "
            "mapped gene is usually the nearest gene, not a proven causal one; a hit is a lead for follow-up, "
            "not evidence that this cell type mediates the trait (that needs LDSC-SEG, MAGMA or scDRS)")
    if not g["rows"]:
        lines.append(head + f"; no trait reaches {r['min_loci']} loci [{TAG}:{column}:{value}]")
        return lines, f"GWAS traits of {column} {value}"
    parts = []
    for row in g["rows"][:n]:
        parts.append(f"{row['trait']} ({row['n_loci']} loci, genes {', '.join(row['genes'][:6])}; p "
                     f"{_fmt_p(row['p'])}, adjusted p {_fmt_p(row['p_adj'])}, degree-matched empirical p "
                     f"{_fmt_p(row['p_empirical'])})")
    lines.append(f"{head}: {'; '.join(parts)}; {g['n_leads']} of {g['n_tested']} tested traits pass adjusted "
                 f"p < 0.05 and empirical p < 0.05 [{TAG}:{column}:{value}]")
    return lines, f"GWAS traits of {column} {value}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure(gwas_dir(), allow_download=True,
           progress=lambda e: logger.info("%s", e) if e.get("stage") != "download" else None)
