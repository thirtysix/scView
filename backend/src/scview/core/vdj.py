"""TCR / BCR contigs joined to the cells: chains, clonotypes and repertoire (C098).

10x 5' immune profiling gives, beside the expression matrix, a table of V(D)J
contigs per cell barcode. This module reads that table, joins it to the cells
by barcode, calls each cell's chains, defines clonotypes, and computes the
repertoire views (clone sizes, diversity, overlap between groups, V/J usage).

**Input.** Cell Ranger's ``filtered_contig_annotations.csv`` or
``all_contig_annotations.csv`` (columns ``barcode``, ``chain``, ``cdr3``,
``productive``, ``umis``...), or an AIRR rearrangement TSV (``cell_id``,
``junction_aa``, ``v_call``, ``productive``, and ``locus`` / ``umi_count``,
or 10x's AIRR export, which has neither: the locus is then read from the V, J
or C call and the UMIs from ``duplicate_count``). Plain or gzipped.

**Chains kept** (scirpy's ``read_10x_vdj(filtered=True)`` and
``pp.index_chains`` defaults; Sturm et al. 2020, Bioinformatics 36:4817, PMID
32614448): productive; high-confidence and in a called cell where the file
says (10x columns ``high_confidence`` / ``is_cell``); an IMGT locus (TRA TRB
TRG TRD IGH IGK IGL); a CDR3 amino-acid sequence. A contig repeated in the file
(same barcode and contig id) counts once. Contigs that fail do not count
towards multichain. Per cell, the VJ chains (TRA, TRG, IGK, IGL) and the VDJ
chains (TRB, TRD, IGH) are each ranked by UMI count, then reads, then the CDR3
nucleotide and amino-acid strings (scirpy's sort keys), and the top two of
each arm are kept as primary and secondary; more than two of either arm makes
the cell **multichain**. A cell with T- and B-cell chains is **ambiguous**.

**Chain status** (``vdj_status``, from scirpy's ``chain_pairing``): paired (one
VJ + one VDJ chain), paired + extra chain (a second VJ and/or VDJ chain: dual
TCR-alpha is common), single chain (one arm only, "orphan"), multichain,
ambiguous, no usable chain (contigs, none kept), no contig (the cell's sample
was profiled, this barcode had no contig), not profiled (no contig table was
attached for its sample).

**Clonotype** (``clonotype``): cells with the same receptor type (TCR or BCR)
and identical CDR3 amino-acid sequences of their primary VJ and primary VDJ
chains, i.e. scirpy's ``define_clonotype_clusters(sequence="aa",
metric="identity", receptor_arms="all", within_group="receptor_type")`` with
``dual_ir="primary_only"`` as in scirpy's tutorial; ``same_v_gene`` also
requires the same V genes. A single-chain cell shares a clonotype only with
other cells that have that one arm and the same CDR3 (receptor_arms="all"),
never with a paired cell. Multichain and ambiguous cells get no clonotype
(scirpy's tutorial removes multichain cells as likely doublets). Clonotypes are
defined over the whole dataset, so one clonotype can span samples (that is what
overlap measures), and they are numbered by size: ``TCR1`` is the largest T
cell clonotype. ``clone_size`` = cells of the dataset in the clonotype;
``clone_expansion`` bins it (1, 2-5, 6-20, >20 cells).

**Barcode join.** A contig barcode matches the cell of the same name first;
otherwise its barcode core (the run of A/C/G/T, with its ``-1`` GEM suffix)
is matched against the cores of the cells' names, so a merged dataset's
``AAAC...-1:sampleA`` or ``sampleA_AAAC...-1`` is found. Matching is
restricted to one sample's cells when the dataset holds several (a sample
column, or the cells' barcode suffix); a core found in two candidate cells is
ambiguous and left unmatched. Both directions are counted.

The functions between the "replayed" markers use only numpy, pandas and the
standard library: the replay script (``core/replay_script.py``) writes their
source into its text, so it repeats the attach exactly.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- replayed: constants -------------------------------------------------------

VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")
VDJ_LOCI = ("TRB", "TRD", "IGH")
TCR_LOCI = ("TRA", "TRB", "TRG", "TRD")
LOCI = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
SLOTS = ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2")
TEXT_FIELDS = ("locus", "cdr3", "cdr3_nt", "v", "d", "j", "c")
NUMBER_FIELDS = ("umis", "reads")
CHAINS_KEY = "vdj_chains"
STATUS_COL = "vdj_status"
RECEPTOR_COL = "vdj_receptor"
CLONOTYPE_COL = "clonotype"
SIZE_COL = "clone_size"
EXPANSION_COL = "clone_expansion"
STATUS_ORDER = ("paired", "paired + extra chain", "single chain", "multichain", "ambiguous",
                "no usable chain", "no contig", "not profiled")
RECEPTOR_ORDER = ("TCR alpha-beta", "TCR gamma-delta", "TCR mixed", "BCR", "TCR + BCR", "none")
EXPANSION_ORDER = ("1 cell", "2-5 cells", "6-20 cells", ">20 cells", "no clonotype")
NO_CLONOTYPE = "no clonotype"
CLONOTYPE_PAIRINGS = ("single pair", "extra VJ", "extra VDJ", "two full chains", "orphan VJ",
                      "orphan VDJ")
SUFFIX_COLUMN = "__barcode_suffix__"
UNS_KEY = "scview_vdj"

# --- end replayed constants ------------------------------------------------------

DROP_REASONS = ("duplicate", "no_barcode", "not_cell", "low_confidence", "non_productive", "unknown_locus", "no_cdr3")
# Fewer cells with a clonotype than this: no diversity for the group.
MIN_CELLS = 10
N_DRAWS = 20
MAX_GROUPS_SHOWN = 40
MAX_GENES_SHOWN = 40
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
TAG = "result:vdj"

_NAME_RX = re.compile(r"^[a-z]{3}_cdr3$")


class VdjError(ValueError):
    """A request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --- replayed: reading, joining, calling chains and clonotypes ---------------------


def sniff_contigs(header: str) -> str | None:
    """'10x' or 'airr' from a contig table's header line, else None."""
    sep = "\t" if header.count("\t") > header.count(",") else ","
    cols = {c.strip().strip('"').lower() for c in header.rstrip("\r\n").split(sep)}
    if {"barcode", "chain", "cdr3", "productive"} <= cols:
        return "10x"
    if {"cell_id", "junction_aa", "v_call", "productive"} <= cols:
        return "airr"
    return None


def _open_contigs(path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    if gz:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def read_contigs(path) -> tuple[pd.DataFrame, str]:
    """The contig table in one layout (barcode, contig_id, locus, cdr3, cdr3_nt,
    v, d, j, c, umis, reads, productive, high_confidence, is_cell) and its format."""
    with _open_contigs(path) as fh:
        header = fh.readline()
    fmt = sniff_contigs(header)
    if fmt is None:
        raise VdjError(
            "This is not a V(D)J contig table: expected Cell Ranger's contig annotations CSV "
            "(barcode, chain, cdr3, productive ...) or an AIRR rearrangement TSV (cell_id, "
            "junction_aa, v_call, productive ...).")
    sep = "\t" if header.count("\t") > header.count(",") else ","
    wanted = {"barcode", "contig_id", "chain", "cdr3", "cdr3_nt", "v_gene", "d_gene", "j_gene",
              "c_gene", "umis", "reads", "productive", "high_confidence", "is_cell", "cell_id",
              "sequence_id", "locus", "junction_aa", "junction", "v_call", "d_call", "j_call",
              "c_call", "umi_count", "duplicate_count", "consensus_count"}
    with _open_contigs(path) as fh:
        raw = pd.read_csv(fh, sep=sep, dtype=str, keep_default_na=False,
                          usecols=lambda c: c.strip().lower() in wanted)
    raw.columns = [c.strip().lower() for c in raw.columns]
    n = len(raw)

    def col(*names, default=""):
        for name in names:
            if name in raw.columns:
                return raw[name].astype(str).str.strip()
        return pd.Series([default] * n, index=raw.index, dtype=object)

    if fmt == "10x":
        df = pd.DataFrame({
            "barcode": col("barcode"), "contig_id": col("contig_id"), "locus": col("chain"),
            "cdr3": col("cdr3"), "cdr3_nt": col("cdr3_nt"), "v": col("v_gene"), "d": col("d_gene"),
            "j": col("j_gene"), "c": col("c_gene"), "umis": col("umis"), "reads": col("reads"),
            "productive": col("productive"), "high_confidence": col("high_confidence", default="true"),
            "is_cell": col("is_cell", default="true"),
        })
    else:
        locus = col("locus")
        calls = [col("v_call"), col("j_call"), col("c_call")]
        for call in calls:  # 10x's AIRR export has no locus column
            guess = call.str[:3].str.upper()
            locus = locus.where(locus != "", guess.where(guess.isin(LOCI), ""))
        umis = col("umi_count")
        umis = umis.where(umis != "", col("duplicate_count"))
        df = pd.DataFrame({
            "barcode": col("cell_id"), "contig_id": col("sequence_id"), "locus": locus,
            "cdr3": col("junction_aa"), "cdr3_nt": col("junction"), "v": col("v_call"),
            "d": col("d_call"), "j": col("j_call"), "c": col("c_call"), "umis": umis,
            "reads": col("consensus_count"), "productive": col("productive"),
            "high_confidence": col("high_confidence", default="true"),
            "is_cell": col("is_cell", default="true"),
        })
    missing = {"", "none", "na", "nan", "null"}
    for name in ("barcode", "contig_id", "locus", "cdr3", "cdr3_nt", "v", "d", "j", "c"):
        s = df[name]
        df[name] = s.where(~s.str.lower().isin(missing), "")
    for name in ("v", "d", "j", "c"):  # the first call, without its allele: TRBV6-5*01,... -> TRBV6-5
        df[name] = df[name].str.split(",").str[0].str.split("*").str[0].str.strip()
    df["locus"] = df["locus"].str.upper()
    for name in ("productive", "high_confidence", "is_cell"):
        df[name] = df[name].str.lower().isin(("true", "t", "1", "yes"))
    for name in ("umis", "reads"):
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
    return df.reset_index(drop=True), fmt


def keep_chains(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(contigs without repeats, the kept chains, contigs dropped by reason). Each
    dropped contig counts once, under the first test it fails."""
    has_id = df["contig_id"] != ""
    dup = (df.duplicated(["barcode", "contig_id"]) & has_id) | (df.duplicated() & ~has_id)
    no_barcode = ~dup & (df["barcode"] == "")
    contigs = df[~dup & ~no_barcode]
    tests = (("not_cell", ~contigs["is_cell"]), ("low_confidence", ~contigs["high_confidence"]),
             ("non_productive", ~contigs["productive"]),
             ("unknown_locus", ~contigs["locus"].isin(LOCI)), ("no_cdr3", contigs["cdr3"] == ""))
    dropped = {"duplicate": int(dup.sum()), "no_barcode": int(no_barcode.sum())}
    left = pd.Series(True, index=contigs.index)
    for name, fails in tests:
        dropped[name] = int((left & fails).sum())
        left &= ~fails
    return contigs, contigs[left], dropped


_CORE_RX = re.compile(r"[ACGTN]{10,}")


def barcode_core(name) -> tuple[str | None, str | None]:
    """(sequence, GEM suffix) of a barcode inside a cell name: the longest run of
    A/C/G/T/N (10+), and the digits of a "-N" right after it (None without)."""
    s = str(name)
    best = None
    for m in _CORE_RX.finditer(s):
        if best is None or len(m.group()) > len(best.group()):
            best = m
    if best is None:
        return None, None
    gem = re.match(r"-(\d+)", s[best.end():])
    return best.group(), (gem.group(1) if gem else None)


def library_tags(names) -> np.ndarray:
    """Each cell name with its barcode replaced by "*" ("*-1:sampleA", "*-2"): cells
    from one 10x library share a tag."""
    out = []
    for name in names:
        s = str(name)
        seq, _ = barcode_core(s)
        out.append(s.replace(seq, "*", 1) if seq else s)
    return np.asarray(out, dtype=object)


def candidate_cells(adata, sample_column: str | None, sample: str | None) -> np.ndarray | None:
    """Mask of the cells one sample's contigs may match (None: all cells)."""
    if not sample_column:
        return None
    if sample is None or str(sample) == "":
        raise VdjError("Name the sample these contigs belong to.")
    if sample_column == SUFFIX_COLUMN:
        mask = library_tags(adata.obs_names) == str(sample)
    elif sample_column in adata.obs.columns:
        mask = (adata.obs[sample_column].astype(str) == str(sample)).to_numpy()
    else:
        raise VdjError(f"Column '{sample_column}' not found.", status=404)
    if not mask.any():
        raise VdjError(f"No cells have {sample_column} = '{sample}'.", status=404)
    return mask


def match_barcodes(barcodes, obs_names, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """For each contig barcode, the index of its cell (-1: none, -2: ambiguous) and
    whether it matched by exact name. Exact names first, then the barcode core
    with its GEM suffix, then the core alone; a core in several candidate cells,
    or a cell two barcodes claim, is ambiguous."""
    names = np.asarray(obs_names).astype(str)
    idx = np.arange(len(names)) if mask is None else np.flatnonzero(mask)
    exact_of = {names[i]: int(i) for i in idx}
    by_full: dict = {}
    by_seq: dict = {}
    for i in idx:
        seq, gem = barcode_core(names[i])
        if seq:
            by_full.setdefault((seq, gem), []).append(int(i))
            by_seq.setdefault(seq, []).append(int(i))
    cells = np.full(len(barcodes), -1, dtype=np.int64)
    exact = np.zeros(len(barcodes), dtype=bool)
    for k, b in enumerate(barcodes):
        b = str(b)
        if b in exact_of:
            cells[k], exact[k] = exact_of[b], True
            continue
        seq, gem = barcode_core(b)
        if not seq:
            continue
        hits = by_full.get((seq, gem)) if gem is not None else None
        hits = hits or by_seq.get(seq)
        if hits:
            cells[k] = hits[0] if len(hits) == 1 else -2
    # A cell claimed by two barcodes keeps the exact one, else neither.
    hit = cells >= 0
    claimed = pd.Series(cells[hit]).value_counts()
    for cell in claimed.index[claimed.to_numpy() > 1]:
        ks = np.flatnonzero(cells == cell)
        keep = ks[exact[ks]]
        cells[ks] = -2
        if len(keep) == 1:
            cells[keep[0]] = cell
    return cells, exact


def empty_table(obs_names) -> pd.DataFrame:
    """The per-cell chain table with no chains."""
    n = len(obs_names)
    cols: dict = {}
    for slot in SLOTS:
        for f in TEXT_FIELDS:
            cols[f"{slot}_{f}"] = pd.Series([np.nan] * n, dtype=object)
        for f in NUMBER_FIELDS:
            cols[f"{slot}_{f}"] = np.full(n, np.nan)
    for f in ("n_vj", "n_vdj", "n_contigs"):
        cols[f] = np.zeros(n, dtype=np.int64)
    cols["profiled"] = np.zeros(n, dtype=bool)
    table = pd.DataFrame(cols)
    table.index = pd.Index(np.asarray(obs_names).astype(str))
    return table


def read_table(adata) -> pd.DataFrame | None:
    """The stored per-cell chain table (obsm ``vdj_chains``), text as object."""
    if CHAINS_KEY not in adata.obsm:
        return None
    stored = adata.obsm[CHAINS_KEY]
    table = empty_table(adata.obs_names)
    for name in table.columns:
        if name not in stored.columns:
            continue
        values = stored[name].to_numpy()
        if table[name].dtype == object:
            table[name] = pd.Series(stored[name].astype(object).to_numpy(), index=table.index)
        else:
            table[name] = np.asarray(values, dtype=table[name].dtype)
    return table


def _arm_rows(table: pd.DataFrame, cells: np.ndarray) -> list[pd.DataFrame]:
    """The chains already in ``table`` for ``cells``, one frame per slot in use."""
    rows = []
    part = table.iloc[cells]
    for slot in SLOTS:
        have = part[f"{slot}_locus"].notna().to_numpy()
        if not have.any():
            continue
        sub = pd.DataFrame({f: part[f"{slot}_{f}"].to_numpy()[have] for f in TEXT_FIELDS + NUMBER_FIELDS})
        sub["cell"] = cells[have]
        rows.append(sub)
    return rows


def merge_chains(table: pd.DataFrame, contigs: pd.DataFrame, kept: pd.DataFrame, cell_of: dict,
                 mask: np.ndarray | None, replace: bool) -> pd.DataFrame:
    """``table`` with one file's chains added to its matched cells (``cell_of``:
    barcode -> cell index): each arm re-ranked over the old and new chains and the
    top two kept. ``mask``: the profiled sample's cells (None: all)."""
    table = table.copy()
    sample = np.arange(len(table)) if mask is None else np.flatnonzero(mask)
    slot_cols = [f"{s}_{f}" for s in SLOTS for f in TEXT_FIELDS + NUMBER_FIELDS]
    if replace:
        table.iloc[sample, [table.columns.get_loc(c) for c in slot_cols]] = np.nan
        for f in ("n_vj", "n_vdj", "n_contigs"):
            table.iloc[sample, table.columns.get_loc(f)] = 0
    table.iloc[sample, table.columns.get_loc("profiled")] = True

    per_cell = contigs["barcode"].map(cell_of)
    n_contigs = per_cell[per_cell.notna()].astype(np.int64).value_counts()
    loc = table.columns.get_loc("n_contigs")
    table.iloc[n_contigs.index.to_numpy(), loc] = table["n_contigs"].to_numpy()[n_contigs.index.to_numpy()] \
        + n_contigs.to_numpy()

    new = kept.assign(cell=kept["barcode"].map(cell_of))
    new = new[new["cell"].notna()].copy()
    if new.empty:
        return table
    new["cell"] = new["cell"].astype(np.int64)
    cells = np.unique(new["cell"].to_numpy())
    long = pd.concat([*_arm_rows(table, cells), new[[*TEXT_FIELDS, *NUMBER_FIELDS, "cell"]]],
                     ignore_index=True)
    long["cell"] = long["cell"].astype(np.int64)
    long["arm"] = np.where(long["locus"].isin(VJ_LOCI), "VJ", "VDJ")
    for f in TEXT_FIELDS:
        long[f] = long[f].fillna("").astype(str)
    for f in NUMBER_FIELDS:
        long[f] = pd.to_numeric(long[f], errors="coerce").fillna(0.0)
    long = long.sort_values(["cell", "arm", "umis", "reads", "cdr3_nt", "cdr3"],
                            ascending=[True, True, False, False, False, False], kind="mergesort")
    long["rank"] = long.groupby(["cell", "arm"]).cumcount()

    table.iloc[cells, [table.columns.get_loc(c) for c in slot_cols]] = np.nan
    for arm in ("VJ", "VDJ"):
        added = new[new["locus"].isin(VJ_LOCI if arm == "VJ" else VDJ_LOCI)]["cell"].value_counts()
        col = "n_vj" if arm == "VJ" else "n_vdj"
        at = added.index.to_numpy()
        table.iloc[at, table.columns.get_loc(col)] = table[col].to_numpy()[at] + added.to_numpy()
        for rank in (0, 1):
            sub = long[(long["arm"] == arm) & (long["rank"] == rank)]
            slot = f"{arm}_{rank + 1}"
            at = sub["cell"].to_numpy()
            for f in TEXT_FIELDS:
                vals = sub[f].to_numpy(dtype=object)
                table.iloc[at, table.columns.get_loc(f"{slot}_{f}")] = np.where(vals == "", None, vals)
            for f in NUMBER_FIELDS:
                table.iloc[at, table.columns.get_loc(f"{slot}_{f}")] = sub[f].to_numpy(dtype=float)
    for slot in SLOTS:
        for f in TEXT_FIELDS:
            c = f"{slot}_{f}"
            table[c] = table[c].astype(object).where(table[c].notna(), np.nan)
    return table


def _slot_loci(table: pd.DataFrame) -> list[np.ndarray]:
    return [table[f"{s}_locus"].astype(object).to_numpy() for s in SLOTS]


def receptor_kind(table: pd.DataFrame) -> np.ndarray:
    """Each cell's receptor type from its kept chains: TCR, BCR, ambiguous or ""."""
    tcr = np.zeros(len(table), dtype=bool)
    bcr = np.zeros(len(table), dtype=bool)
    for loci in _slot_loci(table):
        present = pd.notna(loci)
        is_t = np.isin(loci, TCR_LOCI)
        tcr |= present & is_t
        bcr |= present & ~is_t
    return np.where(tcr & bcr, "ambiguous", np.where(tcr, "TCR", np.where(bcr, "BCR", "")))


def chain_pairing(table: pd.DataFrame) -> np.ndarray:
    """scirpy's ``chain_pairing`` per cell: no IR, multichain, ambiguous, single pair,
    extra VJ, extra VDJ, two full chains, orphan VJ, orphan VDJ."""
    has = {s: table[f"{s}_cdr3"].notna().to_numpy() for s in SLOTS}
    multi = (table["n_vj"].to_numpy() > 2) | (table["n_vdj"].to_numpy() > 2)
    kind = receptor_kind(table)
    vj, vdj = has["VJ_1"], has["VDJ_1"]
    out = np.full(len(table), "no IR", dtype=object)
    out[vj & ~vdj] = "orphan VJ"
    out[vdj & ~vj] = "orphan VDJ"
    pair = vj & vdj
    out[pair] = "single pair"
    out[pair & has["VJ_2"] & ~has["VDJ_2"]] = "extra VJ"
    out[pair & has["VDJ_2"] & ~has["VJ_2"]] = "extra VDJ"
    out[pair & has["VJ_2"] & has["VDJ_2"]] = "two full chains"
    out[kind == "ambiguous"] = "ambiguous"
    out[multi] = "multichain"
    return out


def define_clonotypes(table: pd.DataFrame, same_v_gene: bool = False) -> tuple[pd.Series, pd.Series]:
    """(clonotype id, clone size) per cell; NaN for cells without one. Ids are
    numbered by size within receptor type (TCR1 = largest T-cell clonotype)."""
    pairing = chain_pairing(table)
    kind = receptor_kind(table)
    ok = np.isin(pairing, CLONOTYPE_PAIRINGS) & np.isin(kind, ("TCR", "BCR"))
    parts = [pd.Series(kind, index=table.index), table["VJ_1_cdr3"].fillna("").astype(str),
             table["VDJ_1_cdr3"].fillna("").astype(str)]
    if same_v_gene:
        parts += [table["VJ_1_v"].fillna("").astype(str), table["VDJ_1_v"].fillna("").astype(str)]
    key = parts[0].str.cat(parts[1:], sep="|").where(ok)
    counts = key.value_counts()
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ids, seen = {}, {"TCR": 0, "BCR": 0}
    for k, _ in order:
        prefix = k.split("|", 1)[0]
        seen[prefix] += 1
        ids[k] = f"{prefix}{seen[prefix]}"
    clonotype = key.map(ids)
    size = key.map(counts).astype(float)
    return clonotype, size


def expansion_label(size) -> np.ndarray:
    """1 cell, 2-5, 6-20, >20 cells, or no clonotype (NaN size)."""
    s = np.asarray(size, dtype=float)
    out = np.full(s.shape, NO_CLONOTYPE, dtype=object)
    out[s == 1] = EXPANSION_ORDER[0]
    out[(s >= 2) & (s <= 5)] = EXPANSION_ORDER[1]
    out[(s >= 6) & (s <= 20)] = EXPANSION_ORDER[2]
    out[s > 20] = EXPANSION_ORDER[3]
    return out


def _ordered(values, order) -> pd.Categorical:
    used = set(pd.unique(np.asarray(values, dtype=object)))
    return pd.Categorical(values, categories=[c for c in order if c in used])


def write_columns(adata, table: pd.DataFrame, same_v_gene: bool = False) -> None:
    """Store the chain table (obsm ``vdj_chains``) and write the cell columns from
    it: status, receptor, clonotype, clone size and expansion, primary CDR3 per locus."""
    store = table.copy()
    for slot in SLOTS:
        for f in TEXT_FIELDS:
            store[f"{slot}_{f}"] = pd.Categorical(store[f"{slot}_{f}"])
    adata.obsm[CHAINS_KEY] = store

    pairing = chain_pairing(table)
    status = np.full(len(table), "not profiled", dtype=object)
    profiled = table["profiled"].to_numpy(dtype=bool)
    status[profiled] = "no contig"
    status[profiled & (table["n_contigs"].to_numpy() > 0)] = "no usable chain"
    for label, group in (("paired", ("single pair",)),
                         ("paired + extra chain", ("extra VJ", "extra VDJ", "two full chains")),
                         ("single chain", ("orphan VJ", "orphan VDJ")),
                         ("multichain", ("multichain",)), ("ambiguous", ("ambiguous",))):
        status[np.isin(pairing, group)] = label
    obs = adata.obs
    obs[STATUS_COL] = _ordered(status, STATUS_ORDER)

    loci = _slot_loci(table)
    kind = receptor_kind(table)
    ab = np.zeros(len(table), dtype=bool)
    gd = np.zeros(len(table), dtype=bool)
    for lv in loci:
        ab |= np.isin(lv, ("TRA", "TRB"))
        gd |= np.isin(lv, ("TRG", "TRD"))
    receptor = np.full(len(table), "none", dtype=object)
    receptor[kind == "BCR"] = "BCR"
    receptor[kind == "ambiguous"] = "TCR + BCR"
    t = kind == "TCR"
    receptor[t & ab & ~gd] = "TCR alpha-beta"
    receptor[t & gd & ~ab] = "TCR gamma-delta"
    receptor[t & ab & gd] = "TCR mixed"
    obs[RECEPTOR_COL] = _ordered(receptor, RECEPTOR_ORDER)

    clonotype, size = define_clonotypes(table, same_v_gene)
    ids = list(pd.unique(clonotype.dropna()))
    ids.sort(key=lambda s: (0 if s.startswith("TCR") else 1, int(re.sub(r"\D", "", s))))
    obs[CLONOTYPE_COL] = pd.Categorical(clonotype.to_numpy(), categories=ids)
    obs[SIZE_COL] = size.to_numpy(dtype=float)
    obs[EXPANSION_COL] = _ordered(expansion_label(size.to_numpy()), EXPANSION_ORDER)

    for locus in LOCI:
        obs.drop(columns=[f"{locus.lower()}_cdr3"], inplace=True, errors="ignore")
    for locus in LOCI:
        slot = "VJ_1" if locus in VJ_LOCI else "VDJ_1"
        lv = table[f"{slot}_locus"].astype(object).to_numpy()
        if not (lv == locus).any():
            continue
        cdr3 = table[f"{slot}_cdr3"].astype(object).to_numpy()
        obs[f"{locus.lower()}_cdr3"] = pd.Categorical(np.where(lv == locus, cdr3, None))


def clear(adata) -> None:
    """Remove every V(D)J column, the chain table and the attach record."""
    if CHAINS_KEY in adata.obsm:
        del adata.obsm[CHAINS_KEY]
    cols = [STATUS_COL, RECEPTOR_COL, CLONOTYPE_COL, SIZE_COL, EXPANSION_COL]
    cols += [f"{locus.lower()}_cdr3" for locus in LOCI]
    adata.obs.drop(columns=[c for c in cols if c in adata.obs.columns], inplace=True)
    adata.uns.pop(UNS_KEY, None)


def attach_file(adata, path, sample_column: str | None = None, sample: str | None = None,
                replace: bool = False, same_v_gene: bool = False) -> dict:
    """Join one contig table to the cells and rewrite the V(D)J columns; returns
    what was read and matched. Raises VdjError (nothing changed) when the dataset
    holds several libraries and no sample was named."""
    contigs_all, fmt = read_contigs(path)
    contigs, kept, dropped = keep_chains(contigs_all)
    mask = candidate_cells(adata, sample_column, sample)
    barcodes = pd.unique(contigs["barcode"])
    cells, exact = match_barcodes(barcodes, adata.obs_names, mask)
    if mask is None and ((cells >= 0) & ~exact).any():
        tags = pd.unique(library_tags(adata.obs_names))
        if len(tags) > 1:
            raise VdjError(
                f"This dataset holds cells from {len(tags)} libraries (barcode suffixes such as "
                f"{', '.join(map(str, tags[:3]))}). Pick the sample these contigs belong to, so a "
                "barcode is only matched within that sample's cells.", status=409)
    matched = cells >= 0
    cell_of = dict(zip(np.asarray(barcodes)[matched], cells[matched]))
    table = read_table(adata)
    if table is None:
        table = empty_table(adata.obs_names)
    table = merge_chains(table, contigs, kept, cell_of, mask, replace)
    write_columns(adata, table, same_v_gene)

    usable = set(kept["barcode"])
    is_usable = np.array([b in usable for b in barcodes], dtype=bool)
    unmatched = cells == -1
    n_candidates = int(len(adata) if mask is None else mask.sum())
    return {
        "format": fmt,
        "n_rows": int(len(contigs_all)),
        "n_contigs": int(len(contigs)),
        "n_kept_chains": int(len(kept)),
        "dropped": dropped,
        "kept_by_locus": {k: int(v) for k, v in kept["locus"].value_counts().items()},
        "n_barcodes": int(len(barcodes)),
        "n_barcodes_usable": int(is_usable.sum()),
        "matched": int(matched.sum()),
        "matched_exact": int((matched & exact).sum()),
        "matched_core": int((matched & ~exact).sum()),
        "matched_usable": int((matched & is_usable).sum()),
        "unmatched": int(unmatched.sum()),
        "unmatched_usable": int((unmatched & is_usable).sum()),
        "ambiguous": int((cells == -2).sum()),
        "unmatched_examples": [str(b) for b in np.asarray(barcodes)[unmatched & is_usable][:5]],
        "n_candidate_cells": n_candidates,
        "cells_without_contigs": int(n_candidates - matched.sum()),
    }


def redefine(adata, same_v_gene: bool = False) -> None:
    """Rewrite the clonotype columns from the stored chain table (after cells were
    removed, or with another definition)."""
    table = read_table(adata)
    if table is None:
        raise VdjError("No V(D)J contigs are attached to this dataset.", status=404)
    write_columns(adata, table, same_v_gene)


# --- end replayed ---------------------------------------------------------------

REPLAYED_CONSTANTS = ("VJ_LOCI", "VDJ_LOCI", "TCR_LOCI", "LOCI", "SLOTS", "TEXT_FIELDS",
                      "NUMBER_FIELDS", "CHAINS_KEY", "STATUS_COL", "RECEPTOR_COL", "CLONOTYPE_COL",
                      "SIZE_COL", "EXPANSION_COL", "STATUS_ORDER", "RECEPTOR_ORDER", "EXPANSION_ORDER",
                      "NO_CLONOTYPE", "CLONOTYPE_PAIRINGS", "SUFFIX_COLUMN", "UNS_KEY")
REPLAYED_FUNCTIONS = ("VdjError", "sniff_contigs", "_open_contigs", "read_contigs", "keep_chains",
                      "barcode_core", "library_tags", "candidate_cells", "match_barcodes",
                      "empty_table", "read_table", "_arm_rows", "merge_chains", "_slot_loci",
                      "receptor_kind", "chain_pairing", "define_clonotypes", "expansion_label",
                      "_ordered", "write_columns", "clear", "attach_file", "redefine")


def replay_source() -> str:
    """The constants and functions above as Python source, for the replay script."""
    import inspect

    out = [f"{name} = {globals()[name]!r}" for name in REPLAYED_CONSTANTS]
    out.append(f"_CORE_RX = re.compile({_CORE_RX.pattern!r})")
    for name in REPLAYED_FUNCTIONS:
        out.append("")
        out.append("")
        out.append(inspect.getsource(globals()[name]).rstrip("\n"))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The attach record (uns) and the steps scView records
# ---------------------------------------------------------------------------

DEFINITION = (
    "Clonotype = same receptor type and identical CDR3 amino-acid sequences of the primary VJ "
    "(TRA/TRG/IGK/IGL) and primary VDJ (TRB/TRD/IGH) chains; a single-chain cell matches only "
    "single-chain cells with the same CDR3. Multichain and TCR+BCR cells get none. As scirpy's "
    "define_clonotype_clusters(sequence='aa', metric='identity', receptor_arms='all', "
    "dual_ir='primary_only', within_group='receptor_type') (Sturm et al. 2020, PMID 32614448).")


def stored_info(adata) -> dict | None:
    """The attach record (uns ``scview_vdj``): files, definition, columns."""
    raw = adata.uns.get(UNS_KEY)
    if raw is None:
        return None
    try:
        info = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _save_info(adata, info: dict) -> None:
    adata.uns[UNS_KEY] = json.dumps(info, default=str)


def has_vdj(adata) -> bool:
    return CHAINS_KEY in adata.obsm and CLONOTYPE_COL in adata.obs.columns


def our_columns(adata) -> list[str]:
    cols = [STATUS_COL, RECEPTOR_COL, CLONOTYPE_COL, SIZE_COL, EXPANSION_COL]
    return cols + [c for c in adata.obs.columns if _NAME_RX.match(str(c)) and str(c)[:3].upper() in LOCI]


def _foreign_columns(adata) -> list[str]:
    """Columns with this module's names that it did not write (an author's own
    ``clonotype``): attaching would overwrite them."""
    if stored_info(adata) is not None:
        return []
    names = [STATUS_COL, RECEPTOR_COL, CLONOTYPE_COL, SIZE_COL, EXPANSION_COL]
    names += [f"{locus.lower()}_cdr3" for locus in LOCI]
    return [c for c in names if c in adata.obs.columns]


def attach(adata, path, *, file_name: str, sha256: str, sample_column: str | None = None,
           sample: str | None = None, replace: bool = False, same_v_gene: bool | None = None,
           stored_as: str | None = None) -> dict:
    """Attach one contig table (``attach_file``), record it in uns and the history.
    Everything is checked before ``adata`` changes."""
    info = stored_info(adata) or {"version": 1, "files": []}
    if same_v_gene is None:
        same_v_gene = bool(info.get("same_v_gene", False))
    sample_column = sample_column or None
    sample = None if sample is None or sample_column is None else str(sample)
    foreign = _foreign_columns(adata)
    if foreign:
        raise VdjError(f"The dataset already has column(s) {', '.join(foreign)} that scView did not "
                       "write; attaching contigs would overwrite them.", status=409)
    same_sample = [f for f in info["files"]
                   if f.get("sample_column") == sample_column and f.get("sample") == sample]
    if not replace and any(f.get("sha256") == sha256 for f in same_sample):
        raise VdjError(f"'{file_name}' is already attached"
                       + (f" to {sample_column} = {sample}" if sample_column else "")
                       + ". Tick 'replace' to read it again.", status=409)
    stats = attach_file(adata, path, sample_column, sample, replace, bool(same_v_gene))
    files = [f for f in info["files"] if not (replace and (sample_column is None or f in same_sample))]
    record = {"name": file_name, "sha256": sha256, "sample_column": sample_column, "sample": sample,
              "stored_as": stored_as, "attached_at": datetime.now(timezone.utc).isoformat(), **stats}
    files.append(record)
    info.update({"version": 1, "files": files, "same_v_gene": bool(same_v_gene),
                 "definition": DEFINITION})
    _save_info(adata, info)
    _forget(adata)
    _record(adata, "vdj_attach", {
        "file": file_name, "sha256": sha256, "format": stats["format"], "sample_column": sample_column,
        "sample": sample, "replace": bool(replace), "same_v_gene": bool(same_v_gene),
        "matched_cells": stats["matched"], "unmatched_barcodes": stats["unmatched"],
        "ambiguous_barcodes": stats["ambiguous"], "definition": DEFINITION,
    })
    return record


def set_definition(adata, same_v_gene: bool) -> dict:
    """Recompute the clonotypes from the stored chains (``redefine``) and record it."""
    info = stored_info(adata)
    if info is None or CHAINS_KEY not in adata.obsm:
        raise VdjError("No V(D)J contigs are attached to this dataset.", status=404)
    redefine(adata, bool(same_v_gene))
    info["same_v_gene"] = bool(same_v_gene)
    _save_info(adata, info)
    _forget(adata)
    n = int(adata.obs[CLONOTYPE_COL].nunique())
    _record(adata, "vdj_clonotypes", {"same_v_gene": bool(same_v_gene), "n_clonotypes": n,
                                      "definition": DEFINITION})
    return {"same_v_gene": bool(same_v_gene), "n_clonotypes": n}


def remove(adata) -> dict:
    if stored_info(adata) is None and CHAINS_KEY not in adata.obsm:
        raise VdjError("No V(D)J contigs are attached to this dataset.", status=404)
    cols = [c for c in our_columns(adata) if c in adata.obs.columns]
    clear(adata)
    _record(adata, "vdj_remove", {"columns": cols})
    return {"removed_columns": cols}


def _forget(adata) -> None:
    """Markers and enrichment computed for an old version of the label columns."""
    from scview.core.obs_writes import forget_column_results

    for col in (STATUS_COL, RECEPTOR_COL, CLONOTYPE_COL, EXPANSION_COL):
        forget_column_results(adata, col)


def _record(adata, step: str, params: dict) -> None:
    from scview.core import provenance

    try:
        provenance.record_step(adata, step=step, tool="scview.vdj", params=params)
    except Exception as e:  # provenance must never block the edit
        logger.warning("Could not record %s provenance: %s", step, e)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

HOW_TO = ("No TCR/BCR contigs are attached. Add Cell Ranger's filtered_contig_annotations.csv "
          "(from the vdj_t / vdj_b output) or an AIRR rearrangement TSV here, or upload it beside the "
          "expression files when importing (same filename prefix). Cells are matched by barcode.")


def sample_choices(adata) -> list[dict]:
    """Columns a contig table's sample can be named by: sample-like columns first,
    then the barcode suffix when the cells come from several libraries."""
    from scview.core.replicates import donor_candidates

    obs = adata.obs
    like = re.compile(r"sample|batch|library|orig|gem|channel|lane|donor|patient", re.I)
    names = list(donor_candidates(obs))
    for col in obs.columns:
        name = str(col)
        if name in names or name in our_columns(adata) or not like.search(name):
            continue
        if 2 <= obs[col].nunique(dropna=True) <= 100:
            names.append(name)
    out = [{"name": n, "values": [str(v) for v in pd.unique(obs[n].dropna().astype(str))][:100]}
           for n in names if 2 <= obs[n].nunique(dropna=True) <= 100]
    tags = pd.Series(library_tags(adata.obs_names)).value_counts()
    if len(tags) > 1:
        out.append({"name": SUFFIX_COLUMN, "label": "barcode suffix",
                    "values": [str(t) for t in tags.index[:100]]})
    return out


def _counts(series: pd.Series) -> dict[str, int]:
    vc = series.value_counts(sort=False)
    return {str(k): int(v) for k, v in vc.items() if v > 0}


def summary(adata) -> dict:
    """What is attached, what matched both ways, the chain status and clonotypes."""
    n_obs = int(adata.n_obs)
    if not has_vdj(adata):
        return {"has_vdj": False, "n_cells": n_obs, "how_to": HOW_TO,
                "sample_choices": sample_choices(adata)}
    info = stored_info(adata) or {}
    obs = adata.obs
    table = read_table(adata)
    pairing = pd.Series(chain_pairing(table))
    clon = obs[CLONOTYPE_COL]
    live = clon.astype(object).map(clon.value_counts()).astype(float)
    stored_size = pd.to_numeric(obs[SIZE_COL], errors="coerce")
    stale = bool(not np.allclose(live.fillna(-1).to_numpy(), stored_size.fillna(-1).to_numpy()))
    sizes = clon.value_counts()
    sizes = sizes[sizes > 0]
    expansion = {}
    for label in EXPANSION_ORDER[:-1]:
        in_bin = sizes[expansion_label(sizes.to_numpy()) == label]
        expansion[label] = {"clonotypes": int(len(in_bin)), "cells": int(in_bin.sum())}
    loci = {}
    for locus in LOCI:
        cells = np.zeros(n_obs, dtype=bool)
        for lv in _slot_loci(table):
            cells |= lv == locus
        if cells.any():
            loci[locus] = int(cells.sum())
    files = info.get("files", [])
    profiled = int(table["profiled"].sum())
    return {
        "has_vdj": True,
        "n_cells": n_obs,
        "files": files,
        "match": {
            "cells_profiled": profiled,
            "cells_with_contigs": int((table["n_contigs"] > 0).sum()),
            "cells_with_chain": int((table["VJ_1_cdr3"].notna() | table["VDJ_1_cdr3"].notna()).sum()),
            "cells_without_contigs": int((table["profiled"] & (table["n_contigs"] == 0)).sum()),
            "cells_not_profiled": int(n_obs - profiled),
            "barcodes_without_cells": int(sum(f.get("unmatched", 0) for f in files)),
            "barcodes_without_cells_usable": int(sum(f.get("unmatched_usable", 0) for f in files)),
            "ambiguous_barcodes": int(sum(f.get("ambiguous", 0) for f in files)),
        },
        "status": _counts(obs[STATUS_COL]),
        "receptor": _counts(obs[RECEPTOR_COL]),
        "pairing": {str(k): int(v) for k, v in pairing.value_counts().items()},
        "loci": loci,
        "n_clonotypes": int(len(sizes)),
        "n_cells_in_clonotypes": int(sizes.sum()),
        "n_expanded": int((sizes >= 2).sum()),
        "n_cells_expanded": int(sizes[sizes >= 2].sum()),
        "expansion": expansion,
        "same_v_gene": bool(info.get("same_v_gene", False)),
        "definition": DEFINITION,
        "stale": stale,
        "stale_note": ("Cells were removed after the contigs were attached, so the stored clone sizes "
                       "count cells that are gone. Recompute the clonotypes to update them.") if stale else None,
        "columns": [c for c in our_columns(adata) if c in obs.columns],
        "sample_choices": sample_choices(adata),
        "how_to": HOW_TO,
    }


# ---------------------------------------------------------------------------
# Repertoire views
# ---------------------------------------------------------------------------


def _require(adata) -> None:
    if not has_vdj(adata):
        raise VdjError(HOW_TO, status=404)


def _groups(adata, group_by: str) -> tuple[list[str], np.ndarray]:
    from scview.core.qc_groups import group_codes

    if group_by not in adata.obs.columns:
        raise VdjError(f"Column '{group_by}' not found.", status=404)
    labels, codes = group_codes(adata.obs[group_by])
    if len(labels) > 100:
        raise VdjError(f"'{group_by}' has {len(labels)} groups; pick a column with at most 100.")
    return labels, codes


def clonotype_codes(adata) -> tuple[list[str], np.ndarray]:
    """The clonotype ids in category order and each cell's code (-1: none)."""
    col = adata.obs[CLONOTYPE_COL]
    if not isinstance(col.dtype, pd.CategoricalDtype):
        col = col.astype("category")
    return [str(c) for c in col.cat.categories], np.asarray(col.cat.codes)


def default_group(adata, kind: str = "sample") -> str | None:
    """A grouping when none is named: the samples (diversity, overlap, usage) or the
    cell types (where the top clonotypes sit), else the clustering."""
    from scview.core.de_composition import default_column
    from scview.core.qc_groups import sample_grouping

    if kind == "sample":
        col, _ = sample_grouping(adata)
        if col:
            return col
    return default_column(adata)


def shannon(counts) -> float:
    """Shannon entropy, natural log: -sum p ln p over the clonotypes."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def simpson(counts) -> float:
    """Gini-Simpson index 1 - sum p^2: the chance two cells drawn with replacement
    are from different clonotypes (scikit-bio's ``simpson``)."""
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()
    return float(1.0 - (p * p).sum())


def clonality(counts) -> float | None:
    """1 - Shannon / ln(richness) (1 - Pielou's evenness); None below 2 clonotypes."""
    c = np.asarray(counts, dtype=float)
    s = int((c > 0).sum())
    if s < 2:
        return None
    return float(1.0 - shannon(c) / np.log(s))


def gini(counts) -> float:
    """Gini coefficient of the clone sizes: 0 when every clonotype has the same
    number of cells, towards 1 when a few hold most of them."""
    x = np.sort(np.asarray(counts, dtype=float)[np.asarray(counts) > 0])
    n = len(x)
    if n == 0:
        return float("nan")
    ranks = np.arange(1, n + 1)
    return float(2.0 * (ranks * x).sum() / (n * x.sum()) - (n + 1) / n)


METRICS = ("richness", "shannon", "simpson", "clonality", "gini")


def metrics(counts) -> dict[str, float | None]:
    c = np.asarray(counts)
    c = c[c > 0]
    return {"richness": float(len(c)), "shannon": shannon(c), "simpson": simpson(c),
            "clonality": clonality(c), "gini": gini(c)}


def _rarefied(codes: np.ndarray, depth: int, rng: np.random.Generator, n_draws: int) -> dict:
    draws: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_draws):
        pick = codes[rng.choice(len(codes), size=depth, replace=False)]
        for m, v in metrics(np.bincount(pick)).items():
            if v is not None:
                draws[m].append(v)
    out: dict[str, Any] = {}
    for m, vals in draws.items():
        out[m] = float(np.mean(vals)) if vals else None
        out[f"{m}_sd"] = float(np.std(vals)) if vals else None
    return out


def diversity(adata, group_by: str, *, rarefy: bool = False, seed: int = 0,
              n_draws: int = N_DRAWS) -> dict:
    """Richness, Shannon, Gini-Simpson, clonality and Gini per group, over the
    group's cells with a clonotype. All of them grow or shrink with the number of
    cells sampled; ``rarefy`` subsamples every group to the smallest group's cells
    (groups with MIN_CELLS or more), ``n_draws`` times from ``seed``, and reports
    the mean and SD."""
    _require(adata)
    labels, codes = _groups(adata, group_by)
    _, clon = clonotype_codes(adata)
    rows = []
    member = {}
    for g, name in enumerate(labels):
        in_g = codes == g
        c = clon[in_g & (clon >= 0)]
        member[g] = c
        row: dict[str, Any] = {"name": name, "n_cells": int(in_g.sum()), "n_with_clonotype": int(len(c))}
        if len(c) < MIN_CELLS:
            row.update({m: None for m in METRICS})
            row["note"] = f"fewer than {MIN_CELLS} cells with a clonotype"
        else:
            row.update(metrics(np.bincount(c)))
            row["note"] = None
        rows.append(row)
    usable = [g for g in member if len(member[g]) >= MIN_CELLS]
    depth = min((len(member[g]) for g in usable), default=None)
    if rarefy and depth:
        rng = np.random.default_rng(seed)
        for g in usable:
            rows[g]["rarefied"] = _rarefied(member[g], depth, rng, n_draws)
    sizes = [len(member[g]) for g in usable]
    caption = (f"Over the cells with a clonotype in each group of '{group_by}' "
               f"({min(sizes):,} to {max(sizes):,} cells)." if sizes else
               f"No group of '{group_by}' has {MIN_CELLS} or more cells with a clonotype.")
    if rarefy and depth:
        caption += (f" Rarefied: each group subsampled to {depth:,} cells (the smallest group), "
                    f"{n_draws} draws from seed {seed}; mean shown.")
    elif sizes:
        caption += (" Not rarefied: richness and Shannon grow with the number of cells sampled, so "
                    "compare groups of similar size or rarefy.")
    return {"group_by": group_by, "rarefy": bool(rarefy and depth), "depth": depth if rarefy else None,
            "seed": seed, "n_draws": n_draws, "min_cells": MIN_CELLS, "groups": rows,
            "caption": caption}


def _shown_groups(labels: list[str], sizes: np.ndarray, cap: int) -> list[int]:
    keep = [g for g in range(len(labels)) if sizes[g] > 0]
    if len(keep) > cap:
        keep = sorted(sorted(keep, key=lambda g: -sizes[g])[:cap])
    return keep


def morisita_horn(x, y) -> float:
    """Morisita-Horn overlap of two clone-size vectors over the same clonotypes:
    2 sum x_i y_i / ((Dx + Dy) X Y), D = sum n_i^2 / N^2 (Horn 1966)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X, Y = x.sum(), y.sum()
    if X == 0 or Y == 0:
        return float("nan")
    dx, dy = (x * x).sum() / (X * X), (y * y).sum() / (Y * Y)
    return float(2.0 * (x * y).sum() / ((dx + dy) * X * Y))


def jaccard(x, y) -> float:
    a, b = np.asarray(x) > 0, np.asarray(y) > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def overlap(adata, group_by: str, metric: str = "morisita_horn") -> dict:
    """Clonotype overlap between every two groups: Morisita-Horn (weighted by clone
    size, little affected by how many cells each group has) or Jaccard (shared
    clonotypes / all clonotypes of the two, every singleton counted equally)."""
    _require(adata)
    if metric not in ("morisita_horn", "jaccard"):
        raise VdjError("metric must be morisita_horn or jaccard.")
    labels, codes = _groups(adata, group_by)
    ids, clon = clonotype_codes(adata)
    ok = (codes >= 0) & (clon >= 0)
    counts = np.zeros((len(labels), len(ids)), dtype=np.int64)
    np.add.at(counts, (codes[ok], clon[ok]), 1)
    sizes = counts.sum(axis=1)
    keep = _shown_groups(labels, sizes, MAX_GROUPS_SHOWN)
    fn = morisita_horn if metric == "morisita_horn" else jaccard
    m = [[fn(counts[a], counts[b]) for b in keep] for a in keep]
    shared = [[int(((counts[a] > 0) & (counts[b] > 0)).sum()) for b in keep] for a in keep]
    n_nonzero = sum(1 for s in sizes if s > 0)
    return {"group_by": group_by, "metric": metric, "groups": [labels[g] for g in keep],
            "n_cells": [int(sizes[g]) for g in keep],
            "n_clonotypes": [int((counts[g] > 0).sum()) for g in keep],
            "matrix": [[None if np.isnan(v) else round(v, 4) for v in row] for row in m],
            "shared": shared, "n_groups_total": n_nonzero, "n_groups_shown": len(keep)}


def usage(adata, group_by: str, chain: str | None = None, gene: str = "v",
          count: str = "clonotypes") -> dict:
    """V or J gene usage of one locus per group: the share of the group's clonotypes
    (each once) or cells whose primary chain of that locus uses each gene."""
    _require(adata)
    if gene not in ("v", "j"):
        raise VdjError("gene must be v or j.")
    if count not in ("clonotypes", "cells"):
        raise VdjError("count must be clonotypes or cells.")
    table = read_table(adata)
    present = {}
    for locus in LOCI:
        slot = "VJ_1" if locus in VJ_LOCI else "VDJ_1"
        present[locus] = int((table[f"{slot}_locus"].astype(object) == locus).sum())
    if chain is None:
        chain = max(("TRB", "IGH", "TRA", "IGK", "IGL", "TRD", "TRG"), key=lambda lc: present[lc])
    chain = chain.upper()
    if chain not in LOCI:
        raise VdjError(f"chain must be one of {', '.join(LOCI)}.")
    labels, codes = _groups(adata, group_by)
    _, clon = clonotype_codes(adata)
    slot = "VJ_1" if chain in VJ_LOCI else "VDJ_1"
    lv = table[f"{slot}_locus"].astype(object).to_numpy()
    genes = table[f"{slot}_{gene}"].astype(object).to_numpy()
    ok = (lv == chain) & (clon >= 0) & (codes >= 0) & pd.notna(genes)
    frame = pd.DataFrame({"group": codes[ok], "clonotype": clon[ok], "gene": genes[ok].astype(str)})
    if count == "clonotypes":
        frame = frame.drop_duplicates()
    tab = pd.crosstab(frame["group"], frame["gene"]) if len(frame) else pd.DataFrame()
    totals = tab.sum(axis=0).sort_values(ascending=False, kind="mergesort") if len(tab) else pd.Series(dtype=int)
    top = list(totals.index[:MAX_GENES_SHOWN])
    group_n = tab.sum(axis=1) if len(tab) else pd.Series(dtype=int)
    sizes = np.array([int(group_n.get(g, 0)) for g in range(len(labels))])
    keep = _shown_groups(labels, sizes, MAX_GROUPS_SHOWN)
    counts = [[int(tab.at[g, x]) if g in tab.index else 0 for x in top] for g in keep]
    share = [[c / sizes[g] if sizes[g] else 0.0 for c in row] for g, row in zip(keep, counts)]
    return {"group_by": group_by, "chain": chain, "gene": gene, "count": count,
            "chains_present": {k: v for k, v in present.items() if v},
            "genes": top, "n_genes_total": int(len(totals)), "groups": [labels[g] for g in keep],
            "n": [int(sizes[g]) for g in keep], "counts": counts,
            "share": [[round(v, 5) for v in row] for row in share],
            "n_groups_total": int((sizes > 0).sum()), "n_groups_shown": len(keep)}


def _chain_text(table: pd.DataFrame, i: int) -> list[dict]:
    out = []
    for slot in ("VJ_1", "VDJ_1"):
        locus = table.iat[i, table.columns.get_loc(f"{slot}_locus")]
        if pd.isna(locus):
            continue
        out.append({"arm": slot[:-2], "locus": str(locus),
                    "cdr3": str(table.iat[i, table.columns.get_loc(f"{slot}_cdr3")]),
                    "v": _str(table.iat[i, table.columns.get_loc(f"{slot}_v")]),
                    "j": _str(table.iat[i, table.columns.get_loc(f"{slot}_j")])})
    return out


def _str(v) -> str | None:
    return None if pd.isna(v) else str(v)


def clonotypes(adata, group_by: str | None = None, group: str | None = None, n: int = 25) -> dict:
    """The largest clonotypes (in ``group`` of ``group_by`` when named): cells,
    expansion, chains, cells per group of ``group_by`` and the cell indices."""
    _require(adata)
    n = max(1, min(int(n), 200))
    ids, clon = clonotype_codes(adata)
    labels, codes = (_groups(adata, group_by) if group_by else ([], np.full(len(clon), -1)))
    pick = clon >= 0
    if group is not None:
        if not group_by:
            raise VdjError("Name the column the group belongs to (group_by).")
        if group not in labels:
            raise VdjError(f"'{group}' is not a group of '{group_by}'.", status=404)
        pick &= codes == labels.index(group)
    size_all = np.bincount(clon[clon >= 0], minlength=len(ids))
    size_pick = np.bincount(clon[pick], minlength=len(ids))
    total = int((size_pick > 0).sum())
    order = sorted(np.flatnonzero(size_pick > 0), key=lambda c: (-size_pick[c], c))[:n]
    table = read_table(adata)
    receptor = adata.obs[RECEPTOR_COL].astype(str).to_numpy()
    rows = []
    for c in order:
        cells = np.flatnonzero(clon == c)
        by_group: dict[str, int] = {}
        if group_by:
            gc = np.bincount(codes[cells][codes[cells] >= 0], minlength=len(labels))
            by_group = {labels[g]: int(gc[g]) for g in np.flatnonzero(gc)}
        top = max(by_group.items(), key=lambda kv: kv[1]) if by_group else None
        rows.append({
            "id": ids[c], "n_cells": int(size_all[c]),
            "n_in_group": int(size_pick[c]) if group is not None else None,
            "expansion": str(expansion_label([size_all[c]])[0]),
            "receptor": receptor[cells[0]], "chains": _chain_text(table, int(cells[0])),
            "by_group": by_group, "top_group": top[0] if top else None,
            "top_share": (top[1] / len(cells)) if top else None,
            "indices": cells.tolist(),
        })
    return {"group_by": group_by, "group": group, "n_clonotypes": total, "n_shown": len(rows),
            "n_cells_in_clonotypes": int(size_pick.sum()), "rows": rows}


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """One co-pilot fact: how many cells have chains, the clonotypes, and the
    largest expanded ones with the groups they sit in. Nothing without contigs."""
    adata = adaptor.adata
    if not has_vdj(adata):
        return [], ""
    s = summary(adata)
    where = default_group(adata, kind="celltype")
    top = clonotypes(adata, group_by=where, n=5)["rows"]
    expanded = [r for r in top if r["n_cells"] >= 2]

    def one(r: dict) -> str:
        chains = "/".join(f"{c['locus']} {c['cdr3']}" for c in r["chains"])
        place = (f", {r['top_share']:.0%} in {where} '{r['top_group']}'"
                 if r.get("top_group") is not None else "")
        return f"{r['id']} ({r['n_cells']} cells; {chains}{place})"

    m = s["match"]
    line = (f"- TCR/BCR contigs: {m['cells_with_chain']:,} of {s['n_cells']:,} cells have a usable chain "
            f"({m['cells_not_profiled']:,} cells not profiled; {m['barcodes_without_cells_usable']:,} contig "
            f"barcodes match no cell). {s['n_clonotypes']:,} clonotypes over {s['n_cells_in_clonotypes']:,} "
            f"cells; {s['n_expanded']:,} expanded (2+ cells) holding {s['n_cells_expanded']:,} cells. "
            + ("Largest expanded: " + "; ".join(one(r) for r in expanded) + ". " if expanded else
               "No clonotype has more than one cell. ")
            + "Clonotype = identical CDR3 amino acids of the primary VJ and VDJ chains (scirpy's "
            f"definition); clone sizes count the cells in this dataset [{TAG}]")
    detail = f"{s['n_clonotypes']:,} clonotypes, {s['n_expanded']:,} expanded"
    return [line], detail
