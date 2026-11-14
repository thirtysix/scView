"""Per-cluster quality flags: which clusters stand out against the others.

Deterministic and reproducible (no LLM). For a grouping column (default: the
active clustering) each cluster's cells are summarised (size, median genes and
counts per cell, median % mitochondrial and % ribosomal, doublet fraction or
score) and a cluster is flagged when it sits far from the typical cluster on
robust statistics: the median across clusters and the MAD around it, so one bad
cluster cannot hide by inflating the spread. Absolute floors stop near-identical
clusters from turning a tiny difference into a flag.

Flags: ``low_quality`` (few genes AND high mito), ``high_mito``, ``low_genes``,
``doublet_rich``, ``doublet_like`` (its top markers are mostly those of two other
clusters and its cells carry more RNA than either: a cluster of their doublets,
``doublet_like``), ``stress_driven`` (dissociation-stress score far above the
others AND stress genes among its top markers; see ``core/stress.py``),
``hb_ambient`` (haemoglobin RNA in a cluster whose markers are not erythroid;
see ``core/haemoglobin.py``), ``cycling`` (mostly cells with clear S/G2M scores;
see ``core/cell_cycle.py``), ``low_malat1`` (little of the nuclear-retained MALAT1
RNA: fragments or cytoplasm without a nucleus) and ``small``. An author's
intronic / unspliced share column, when obs has one, is summarised (never computed). Erythroid clusters get a note, not a
flag: their haemoglobin is their own. Top markers whose signal in the other clusters
looks like ambient RNA are listed per cluster (``ambient_markers``, ``core/ambient.py``),
also a note, not a flag. At the dataset level, haemoglobin spread
across the non-erythroid clusters (ambient RNA) and technical scores that track
the leading PCs of the clustering's space (``core/qc_drivers.py``) are reported.
The same numbers ground the co-pilot, so "which clusters look low quality?" is
answered from them (``grounding_block``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from scview.core import ambient as ambient_rna
from scview.core import (
    cell_cycle,
    doublet_source,
    gene_filters,
    haemoglobin,
    marker_selection,
    qc_drivers,
    spatial,
    stress,
    targeted_panel,
)
from scview.core.anndata_adaptor import per_cell_qc

logger = logging.getLogger(__name__)

MIN_GROUPS = 3  # comparing against "the other clusters" needs a few of them
MAX_GROUPS = 200
_MAD_TO_SD = 1.4826
_Z_SINGLE = 3.0  # one metric on its own: scater's "3 MADs" outlier rule
_Z_PAIR = 2.0  # each of two metrics that agree (few genes and high mito)
# Floors on the robust spread, in each metric's own units (log for genes/counts).
_SCALE_FLOOR = {
    "log_genes": 0.05,
    "pct_mt": 1.0,
    "doublet_fraction": 0.02,
    "doublet_score": 0.02,
    "stress": stress.SCALE_FLOOR,
}
_SMALL_MIN_CELLS = 20
_SMALL_FRACTION = 0.0025
# low_malat1: a cluster whose median MALAT1 share is below this fraction of the
# median cell's (Montserrat-Ayuso & Esteve-Codina 2024).
LOW_MALAT1_FRACTION = 1 / 3
# ... and only when MALAT1 is captured at all: below this share in the median cell
# (paul15, MARS-seq: 0.02 %) a third of it is noise, and most clusters sit at 0.
MALAT1_MIN_MEDIAN_PCT = 0.5

# doublet_like: a cluster that looks like doublets of two others (C056). Its top markers
# (marker_selection.group_markers: the ONE marker choice, DOUBLET_LIKE_MARKERS of them)
# must be mostly theirs (DOUBLET_LIKE_SHARE of them in the two clusters' top markers
# together, at least DOUBLET_LIKE_EACH from each one: a doublet carries both identities),
# its cells must carry about two cells' RNA (median counts at least the larger cluster's
# plus DOUBLET_LIKE_RNA of the smaller one's: a doublet holds a little under the sum of its
# two cells' in practice, where a state between the two holds one cell's, about their mean;
# on hgmm_10k_v3 the cross-species doublet cluster holds 0.95 of the sum, two subtypes of
# the larger cluster 0.86-0.90 and are not flagged) and no fewer genes, and it must
# be smaller than the larger of the two (a cluster bigger than both is a real population).
# A cluster or a parent with under DOUBLET_LIKE_MIN_MARKERS markers is not judged: too few
# to tell. EACH was 0.3 and the size rule "smaller than both" until 2026-11-09: they missed
# 7 of 12 visible doublets in the annotation benchmark's T4 sets (one parent's genes lead
# the fold-change ranking; a doublet group bigger than a rare parent), and 0.2 with the
# larger parent flagged no clean group in ~210 (bench author + Leiden columns, paul15, Kang).
DOUBLET_LIKE_MARKERS = 25
DOUBLET_LIKE_MIN_MARKERS = 10
DOUBLET_LIKE_SHARE = 0.5
DOUBLET_LIKE_EACH = 0.2
DOUBLET_LIKE_RNA = 0.5
_NOT_A_PARENT = {"low_quality", "low_genes"}  # damaged cells or ambient RNA, not a cell type

# nuclei_mito: in data where the typical cluster's median cell has under NUCLEI_MITO_MEDIAN % mito
# (single-nucleus data: nuclei hold no mitochondria, so median % mito reads 0 and the % mito
# flags cannot see anything), a cluster whose mean % mito is NUCLEI_MITO_FOLD times the typical
# cluster's (and NUCLEI_MITO_MIN_DIFF points more) AND whose share of cells with any mito read is
# NUCLEI_MITO_SHARE_DIFF above the typical cluster's. Measured 2026-11-12 on 15 groupings of 9
# datasets (research/v075-qa): fires on the Emont 2022 mouse fat Leiden 10 (mito reads in
# 100% of nuclei vs 4%) and author mAd3 (43% vs 17%, mean 0.20% vs 0.02%), on no Wilson 2022
# kidney cluster (PT included), and never on cell data (the median gate is off: hepatocytes,
# kidney PT at 12-15% in Tabula Sapiens). A caution, not damage: mAd3 is a population the
# authors named.
NUCLEI_MITO_MEDIAN = 1.0
NUCLEI_MITO_FOLD = 4.0
NUCLEI_MITO_MIN_DIFF = 0.1
NUCLEI_MITO_SHARE_DIFF = 0.25
# low_content: a cluster with the identity of another (at least LOW_CONTENT_SHARE of its top
# markers are that cluster's top markers, of at least DOUBLET_LIKE_MIN_MARKERS) whose cells carry
# under LOW_CONTENT_COUNTS of its median counts and LOW_CONTENT_GENES of its genes: the same
# cells with less RNA (damaged cells, nuclei losing RNA, ambient droplets), split off by depth.
# Wilson 2022 kidney Seurat cluster 0 (13 of 25 markers PT's cluster 1, 0.36 of its counts, 0.48
# of its genes) fires; in the 14 other groupings none did. Real low-RNA types sit apart: Kang B cells
# vs DC share 7 MHC genes (0.34 counts, 0.54 genes), Smart-seq2 read counts move with genes
# the other way (Tabula Muris oligodendrocytes 0.45 counts but 1.5 genes). A caution.
LOW_CONTENT_SHARE = 0.4
LOW_CONTENT_COUNTS = 0.5
LOW_CONTENT_GENES = 0.6
# noncoding_markers: at least NONCODING_SHARE of a cluster's top markers (the annotator's list) are
# small RNAs, multi-copy genes or pseudogenes (NONCODING_RE), or more than half of its top
# UNINFORMATIVE_TOP (unnamed genes kept) are MALAT1, NEAT1, mouse Gm / Rik genes or unnamed ids.
# Tabula Sapiens liver Leiden 14 (Smart-seq2 cells among 10x ones; 16 of 25) and its author
# "myeloid cell" (13 of 25) fire; in the 13 other groupings at most 4 of 25 (hepatocyte pseudogenes
# FYTTD1P1, CES1P1). A caution: a technical cluster (protocol, ambient RNA, damage), not a type.
NONCODING_SHARE = 1 / 3
UNINFORMATIVE_TOP = 10
NONCODING_RE = re.compile(
    r"^(?:MIR\d+[A-Z]?(?:-\d+)?|SNOR[AD]\d+\w*|SCARNA\d+\w*|RN7S[KL]\w*|RNU\d\w*|RNU\w+ATAC\w*|RNY\d\w*"
    r"|VTRNA\w*|POTE[A-Z]\w*|MT(?:ND\d|CO\d|ATP\d|CYB|RNR\d)\w*P\d+|MTRNR2L\d+"
    # Processed pseudogenes named by their gene plus P and a number (FTH1P20, CES1P1, H3P16), and
    # mouse -ps ones. A digit before the P keeps real genes (CASP3, MAP4, VAMP2) out.
    r"|[A-Z][A-Z0-9]*\dP\d+|\w+-ps\d*)$", re.I)
UNINFORMATIVE_RE = re.compile(r"^(?:(?i:MALAT1|NEAT1)|Gm\d+|\w+Rik\d*)$")  # Gm / Rik: mouse, case kept
# sex_markers: a sex-linked gene (sex_check.SEX_LINKED_GENES) among a cluster's top SEX_TOP
# markers. Emont 2022 mouse fat Leiden 1, 2 and 5 (Xist first or second): adipocytes split by the
# mice's sex. None of the 14 other groupings had one in its top 5.
SEX_TOP = 3

FLAG_HELP = {
    "low_quality": (
        "Likely low quality: far fewer genes per cell and much more mitochondrial RNA "
        "than the typical cluster, the signature of damaged or dying cells."
    ),
    "high_mito": (
        "High mitochondrial content compared with the other clusters: stressed or "
        "dying cells, or a cell type with naturally high respiration."
    ),
    "low_genes": (
        "Far fewer genes per cell than the other clusters: empty droplets or ambient "
        "RNA, or naturally small cells such as platelets or red blood cells."
    ),
    "doublet_rich": (
        "Many predicted doublets: the cluster may be two cells captured together "
        "rather than a real population."
    ),
    "doublet_like": (
        "Looks like doublets of two other clusters: most of its top marker genes are the "
        "top markers of those two, and its cells carry more RNA than either (a doublet holds "
        "two cells' RNA). It may be a real state between the two (a transitional or "
        "dual-identity cell type); check the doublet calls and both clusters' markers in it "
        "before naming it. The two named are the best match by top markers: when one cell type "
        "is split into several clusters, the genes they share are none of their top markers, "
        "so one side of the doublets may be named by a related cluster."
    ),
    "stress_driven": stress.FLAG_HELP,
    "hb_ambient": haemoglobin.FLAG_HELP,
    "cycling": cell_cycle.FLAG_HELP,
    "low_malat1": (
        "Little MALAT1 RNA: its median share of reads is below a third of the median cell's. "
        "MALAT1 is kept in the nucleus, so cells with little of it may be fragments or "
        "cytoplasm that lost its nucleus; in single-nucleus data, cytoplasmic debris. "
        "Cells without a nucleus (red cells, platelets) have little by nature. "
        "Montserrat-Ayuso & Esteve-Codina 2024, BMC Genomics, doi:10.1186/s12864-024-11015-5."
    ),
    "small": (
        "Very few cells: its markers and DE results are unreliable; it may be a rare "
        "cell type or an artifact."
    ),
    "nuclei_mito": (
        "Mitochondrial RNA in far more of its cells than in the typical cluster, in data where most "
        "cells have almost none (single-nucleus data: nuclei hold no mitochondria). It points to "
        "cytoplasm still on the nuclei, or whole cells: contamination, or nuclei of a type that keep "
        "more cytoplasm. Check its markers for cytoplasmic genes (ferritin, Cox genes)."
    ),
    "low_content": (
        "Shares the identity of another cluster (most of its top markers are that cluster's) but its "
        "cells carry far less RNA: likely the same cells damaged or losing RNA, or droplets of "
        "ambient RNA, split off by sequencing depth rather than biology. Its markers against that "
        "cluster are often ambient genes from other cell types."
    ),
    "noncoding_markers": (
        "Its top markers are mostly small RNAs (miRNA, snoRNA, 7SL), multi-copy genes or "
        "pseudogenes, or unnamed genes and MALAT1: reads a cell type's marker list does not hold. "
        "Typical of a technical cluster (another protocol or chemistry, ambient RNA, damaged "
        "cells), not a cell type."
    ),
    "sex_markers": (
        "Sex-linked genes (XIST, TSIX, Y-chromosome genes) lead its markers: it may hold one sex's "
        "cells of a type that is split by sex (or by donor), not a type of its own. Compare its "
        "make-up by sex or donor before naming it."
    ),
}


class ClusterQC(BaseModel):
    group: str
    n_cells: int
    fraction: float  # share of all cells
    median_genes: float | None = None
    median_counts: float | None = None
    median_pct_mt: float | None = None
    median_pct_ribo: float | None = None
    doublet_fraction: float | None = None
    median_doublet_score: float | None = None
    median_stress_score: float | None = None
    stress_markers: list[str] = []  # signature genes among its top markers
    mean_pct_hb: float | None = None  # mean, not median: most cells have no Hb read
    erythroid: bool = False  # red-cell lineage by its markers (or a red-cell Hb level)
    erythroid_markers: list[str] = []  # haemoglobin / erythroid genes among its top markers
    cycling_fraction: float | None = None  # share of cells with a clear S or G2M score
    doublet_like_of: list[str] = []  # doublet_like: the two clusters it looks like a mix of
    # doublet_like: [its top markers among the first's, among the second's, among either's, its markers]
    doublet_like_shared: list[int] = []
    median_pct_malat1: float | None = None  # % of a cell's counts from MALAT1
    median_pct_intronic: float | None = None  # the author's intronic/unspliced share, %
    # Top markers whose signal outside the cluster looks like ambient RNA (core/ambient.py)
    ambient_markers: list[str] = []
    mean_pct_mt: float | None = None  # mean % mito: a median of 0 in nuclei hides contamination
    mt_detected: float | None = None  # share of its cells with any mitochondrial read
    low_content_of: str | None = None  # low_content: the cluster whose identity it shares
    low_content_shared: list[int] = []  # low_content: [its markers among that cluster's, its markers]
    noncoding_markers: list[str] = []  # small RNAs / pseudogenes / unnamed genes among its top markers
    sex_markers: list[str] = []  # sex-linked genes among its top SEX_TOP markers
    flags: list[str] = []
    reasons: list[str] = []  # one plain-language sentence per flag, with the numbers
    notes: list[str] = []  # not a problem, but worth saying (erythroid: Hb is real)


NO_MT_NOTE = ("No mitochondrial (MT-) genes in this dataset, so % mito can't be measured "
              "(a mito filter had nothing to remove).")


class ClusterQCReport(BaseModel):
    column: str | None
    n_cells: int
    n_groups: int = 0
    metrics: list[str] = []  # per-cluster metrics that could be computed
    typical: dict[str, float] = {}  # median across clusters, per metric
    # The median CELL's genes and counts (what the QC plots show), not a cluster's
    cell_median: dict[str, float] = {}
    after_filtering: bool = False  # the dataset's cell filtering step has run
    clusters: list[ClusterQC] = []
    flag_help: dict[str, str] = FLAG_HELP
    computed_on_demand: bool = False  # genes/counts/mito worked out from counts
    note: str = ""
    ambient_hb: str = ""  # haemoglobin RNA across the non-erythroid clusters, if any
    drivers: list[qc_drivers.Driver] = []  # technical scores tracking a leading PC
    drivers_checked: list[str] = []  # the scores looked at
    drivers_space: str | None = None  # obsm key of the clustering's space
    drivers_n_pcs: int = 0  # leading PCs looked at
    dataset_median_pct_malat1: float | None = None  # the median cell's, for low_malat1
    intronic_column: str | None = None  # obs column of an intronic/unspliced share, if any
    doublet_source: str | None = None  # whose doublet calls doublet_fraction counts (doublet_source)
    # Ambient-looking top markers per cluster are from the soup share alone (no raw matrix)
    ambient_markers_approximate: bool = False
    ambient_markers_help: str = ""
    doublet_like_checked: bool = False  # markers were there to look for doublet-like clusters
    # Nuclei-like data (typical cluster's median % mito under NUCLEI_MITO_MEDIAN): mito is
    # judged by the share of cells with a mito read (nuclei_mito)
    nuclei_like_mito: bool = False

    @property
    def flagged(self) -> list[ClusterQC]:
        return [c for c in self.clusters if c.flags]


def doublet_calls(obs: pd.DataFrame) -> np.ndarray | None:
    """Per-cell doublet calls as booleans from the source the doublet views use:
    cell hashing when it was run, else Scrublet, else a text class column
    (``doublet_source.resolve``, source "auto")."""
    used = doublet_source.resolve(obs)
    return None if used is None else used.calls


# An author's per-cell intronic / unspliced read share (Cell Ranger, velocyto,
# STARsolo summaries): shown per cluster when present, never computed here.
INTRONIC_COLUMNS = (
    "pct_intronic", "pct_counts_intronic", "pct_reads_intronic", "intronic_fraction",
    "fraction_intronic", "frac_intronic", "unspliced_fraction", "fraction_unspliced",
    "frac_unspliced", "unspliced_ratio", "pct_unspliced",
)
_INTRONIC_RE = re.compile(r"intron|unspliced", re.I)
_SHARE_RE = re.compile(r"pct|percent|frac|ratio|prop|share", re.I)


def intronic_share(obs: pd.DataFrame) -> tuple[str, np.ndarray] | None:
    """(column, % per cell) of an author's intronic / unspliced share column; a
    0-1 fraction is read as a share and given as %. None when obs has none."""
    names = [c for c in INTRONIC_COLUMNS if c in obs.columns]
    names += [str(c) for c in obs.columns if str(c) not in names
              and _INTRONIC_RE.search(str(c)) and _SHARE_RE.search(str(c))]
    for name in names:
        v = pd.to_numeric(pd.Series(np.asarray(obs[name])), errors="coerce").to_numpy(dtype=np.float64)
        fin = v[np.isfinite(v)]
        if fin.size == 0 or fin.min() < 0 or fin.max() > 100:
            continue
        return name, v * 100.0 if fin.max() <= 1.0 else v
    return None


def _robust_z(values: pd.Series, floor: float) -> pd.Series:
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    return (values - med) / max(_MAD_TO_SD * mad, floor)


def default_grouping(adata) -> str | None:
    """The column cluster QC uses when none is named: the active clustering."""
    active = adata.uns.get("scview_active_clustering")
    if active and active in adata.obs.columns:
        return str(active)
    for c in ("leiden", "louvain", "cluster", "clusters", "seurat_clusters"):
        if c in adata.obs.columns:
            return c
    return None


def cluster_qc(
    adata, groupby: str | None = None, cells: dict | Callable[[], dict] | None = None
) -> ClusterQCReport:
    """Per-cluster QC summary + flags for ``groupby`` (default: the active clustering).

    ``cells`` are the per-cell QC arrays (``AnnDataAdaptor.qc_per_cell``), or a
    function returning them that is only called once the column checks out; they
    are computed here when not given. Raises ``ValueError`` for a column that is
    not a usable grouping."""
    n = int(adata.n_obs)
    column = groupby or default_grouping(adata)
    if not column:
        return ClusterQCReport(
            column=None, n_cells=n,
            note="No clustering yet: run clustering, or pick a grouping column.",
        )
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found in obs.")
    labels = adata.obs[column]
    n_groups = int(labels.nunique(dropna=True))
    if n_groups < 2 or n_groups > MAX_GROUPS:
        raise ValueError(
            f"'{column}' has {n_groups} distinct values; cluster QC needs 2 to {MAX_GROUPS} groups."
        )

    if callable(cells):
        cells = cells()
    elif cells is None:
        cells = per_cell_qc(adata, ribo=True)
    frame = pd.DataFrame(
        {"genes": cells["n_genes"], "counts": cells["total_counts"]}, index=adata.obs_names
    )
    metrics = ["genes", "counts"]
    if cells.get("has_mt", True):
        frame["pct_mt"] = cells["pct_mt"]
        metrics.append("pct_mt")
        pct_mt = np.asarray(cells["pct_mt"], dtype=np.float64)
        frame["mt_any"] = np.where(np.isfinite(pct_mt), (pct_mt > 0).astype(np.float64), np.nan)
    if cells.get("pct_ribo") is not None:
        frame["pct_ribo"] = cells["pct_ribo"]
        metrics.append("pct_ribo")
    used = doublet_source.resolve(adata.obs)
    if used is not None:
        frame["doublet"] = used.calls.astype(np.float64)
        metrics.append("doublet_fraction")
    if "doublet_score" in adata.obs.columns:
        frame["doublet_score"] = pd.to_numeric(adata.obs["doublet_score"], errors="coerce").to_numpy()
        metrics.append("doublet_score")
    sig_markers: dict[str, list[str]] | None = None
    if stress.SCORE_COLUMN in adata.obs.columns:
        frame["stress"] = pd.to_numeric(adata.obs[stress.SCORE_COLUMN], errors="coerce").to_numpy()
        metrics.append("stress_score")
        sig_markers = stress.signature_markers(adata, column)
    ery_markers: dict[str, list[str]] | None = None
    if cells.get("pct_hb") is not None:
        frame["pct_hb"] = cells["pct_hb"]
        metrics.append("pct_hb")
        ery_markers = haemoglobin.erythroid_markers(adata, column)
    cycling = cell_cycle.cycling_cells(adata.obs)
    if cycling is not None:
        frame["cycling"] = cycling.astype(np.float64)
        metrics.append("cycling_fraction")
    malat1_median = None
    if cells.get("pct_malat1") is not None:
        frame["pct_malat1"] = cells["pct_malat1"]
        metrics.append("pct_malat1")
        malat1_median = _num(np.nanmedian(cells["pct_malat1"]), 4)
    intronic = intronic_share(adata.obs)
    if intronic is not None:
        frame["intronic"] = intronic[1]
        metrics.append("pct_intronic")

    grouped = frame.groupby(labels.astype(str).where(labels.notna()).to_numpy(), sort=False)
    sizes = grouped.size()
    medians = grouped.median()
    if "doublet" in frame.columns:
        medians["doublet"] = grouped["doublet"].mean()  # a fraction, not a median
    if "cycling" in frame.columns:
        medians["cycling"] = grouped["cycling"].mean()
    if "pct_hb" in frame.columns:
        medians["hb"] = grouped["pct_hb"].mean()  # the median cell has no Hb read
    if "mt_any" in frame.columns:
        medians["mt_any"] = grouped["mt_any"].mean()  # a share of cells
        medians["mt_mean"] = grouped["pct_mt"].mean()  # the median nucleus has no mito read
    # Keep the column's own category order (cluster 0, 1, 2 …), not first-seen order.
    order = (
        [str(c) for c in labels.cat.categories] if hasattr(labels, "cat")
        else sorted(sizes.index, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
    )
    order = [g for g in order if g in sizes.index]
    sizes, medians = sizes.loc[order], medians.loc[order]

    typical = {k: float(np.nanmedian(v)) for k, v in medians.items()}
    # Haemoglobin: erythroid clusters first; the others are compared with each
    # other (a typical cluster's Hb is a typical NON-erythroid cluster's).
    erythroid: set[str] = set()
    others = list(order)
    if "hb" in medians:
        erythroid = {g for g in order if haemoglobin.is_erythroid(
            None if ery_markers is None else ery_markers.get(g, []), medians.loc[g, "pct_hb"])}
        others = [g for g in order if g not in erythroid]
        typical.pop("hb")
        if others:
            typical["hb"] = float(np.nanmedian(medians.loc[others, "hb"]))
    # Red-cell genes besides haemoglobin detected in the clusters' own cells: when every
    # cluster is red-cell lineage, markers cannot show it and their haemoglobin is their own.
    lineage = haemoglobin.lineage_detected(adata, labels.astype(str).to_numpy(), others) \
        if "hb" in medians and others else None
    z: dict[str, pd.Series] = {}
    if len(order) >= MIN_GROUPS:
        z["genes"] = _robust_z(np.log1p(medians["genes"]), _SCALE_FLOOR["log_genes"])
        if "pct_mt" in medians:
            z["pct_mt"] = _robust_z(medians["pct_mt"], _SCALE_FLOOR["pct_mt"])
        if "doublet" in medians:
            z["doublet"] = _robust_z(medians["doublet"], _SCALE_FLOOR["doublet_fraction"])
        if "doublet_score" in medians:
            z["doublet_score"] = _robust_z(medians["doublet_score"], _SCALE_FLOOR["doublet_score"])
        if "stress" in medians:
            z["stress"] = _robust_z(medians["stress"], _SCALE_FLOOR["stress"])
        if "hb" in medians and len(others) >= MIN_GROUPS:
            # Centre and spread from the non-erythroid clusters only.
            hb = medians["hb"]
            med = float(np.nanmedian(hb.loc[others]))
            mad = float(np.nanmedian(np.abs(hb.loc[others] - med)))
            z["hb"] = (hb - med) / max(_MAD_TO_SD * mad, haemoglobin.SCALE_FLOOR)

    small_cut = max(_SMALL_MIN_CELLS, _SMALL_FRACTION * n)
    nuclei_like = len(order) >= MIN_GROUPS and "mt_any" in medians \
        and typical.get("pct_mt", np.inf) < NUCLEI_MITO_MEDIAN
    if nuclei_like:
        metrics.append("mt_detected")
    rows: list[ClusterQC] = []
    for g in order:
        m = medians.loc[g]
        row = ClusterQC(
            group=g,
            n_cells=int(sizes.loc[g]),
            fraction=round(float(sizes.loc[g]) / max(n, 1), 4),
            median_genes=_num(m.get("genes")),
            median_counts=_num(m.get("counts")),
            median_pct_mt=_num(m.get("pct_mt")),
            median_pct_ribo=_num(m.get("pct_ribo")),
            doublet_fraction=_num(m.get("doublet"), 4),
            median_doublet_score=_num(m.get("doublet_score"), 4),
            median_stress_score=_num(m.get("stress"), 3),
            stress_markers=(sig_markers or {}).get(g, []),
            mean_pct_hb=_num(m.get("hb"), 4),
            erythroid=g in erythroid,
            erythroid_markers=(ery_markers or {}).get(g, []),
            cycling_fraction=_num(m.get("cycling"), 3),
            median_pct_malat1=_num(m.get("pct_malat1"), 4),
            median_pct_intronic=_num(m.get("intronic"), 1),
            mean_pct_mt=_num(m.get("mt_mean"), 4),
            mt_detected=_num(m.get("mt_any"), 4),
        )
        _flag(row, {k: float(v.loc[g]) for k, v in z.items()}, typical, small_cut,
              markers_checked=sig_markers is not None, hb_checked=ery_markers is not None,
              compare=len(order) >= MIN_GROUPS, median_pct_hb=_num(m.get("pct_hb")))
        _flag_malat1(row, malat1_median, compare=len(order) >= MIN_GROUPS)
        if nuclei_like:
            _flag_nuclei_mito(row, typical)
        rows.append(row)
    like_checked = len(order) >= MIN_GROUPS and _flag_doublet_like(adata, column, rows)
    if len(order) >= MIN_GROUPS:
        _flag_marker_makeup(adata, column, rows, small_cut)

    names = {"genes": "median_genes", "counts": "median_counts", "pct_mt": "median_pct_mt",
             "pct_ribo": "median_pct_ribo", "doublet": "doublet_fraction",
             "doublet_score": "median_doublet_score", "stress": "median_stress_score",
             "hb": "mean_pct_hb", "cycling": "cycling_fraction",
             "pct_malat1": "median_pct_malat1", "intronic": "median_pct_intronic",
             "mt_any": "mt_detected", "mt_mean": "mean_pct_mt"}
    # A score outlier that couldn't be checked against markers isn't flagged: say so.
    unchecked = [] if sig_markers is not None or "stress" not in z else [
        g for g in order if _stress_outlier(float(z["stress"].loc[g]),
                                            medians.loc[g].get("stress"), typical.get("stress"))
    ]
    hb_unchecked = [] if ery_markers is not None or "hb" not in z else [
        g for g in others if haemoglobin.ambient_outlier(
            float(z["hb"].loc[g]), medians.loc[g, "hb"], typical.get("hb"))
    ]
    ambient = lineage_note = ""
    own = {g for g, share in (lineage or {}).items() if share >= haemoglobin.LINEAGE_SHARE}
    for row in rows:
        if row.group in own and "hb_ambient" in row.flags:
            i = row.flags.index("hb_ambient")
            row.flags.pop(i)
            row.reasons = [r for r in row.reasons if not r.startswith("Haemoglobin RNA without erythroid markers")]
    wide_own = bool(lineage) and float(np.median(list(lineage.values()))) >= haemoglobin.LINEAGE_SHARE
    if typical.get("hb") is not None and typical["hb"] >= haemoglobin.AMBIENT_WIDE_PCT and wide_own:
        share = float(np.median(list(lineage.values())))
        lineage_note = (
            f"Haemoglobin RNA is in every cluster ({_pct(typical['hb'])} of the reads in a typical one), "
            f"but so are red-cell genes besides it: in a typical cluster {share:.0%} of the cells detect "
            f"{haemoglobin.MIN_ERYTHROID_MARKERS} or more of {', '.join(haemoglobin.ERYTHROID_GENES[:4])} ... "
            "These are red-cell lineage cells and the haemoglobin is their own, not ambient RNA."
        )
    elif typical.get("hb") is not None and typical["hb"] >= haemoglobin.AMBIENT_WIDE_PCT:
        ambient = (
            f"Haemoglobin RNA is spread across the non-erythroid clusters ({_pct(typical['hb'])} "
            "of the reads in a typical one): likely ambient RNA from lysed red cells. It adds "
            "red-cell genes to every cluster and can blur markers and DE; ambient-RNA removal "
            "before import (SoupX, CellBender) cleans it."
        )
    drivers = _drivers(adata, labels, cells, frame, erythroid)
    amb, amb_approx = _ambient(adata, column)
    for row in rows:
        row.ambient_markers = amb.get(row.group, [])
    return ClusterQCReport(
        column=column,
        n_cells=n,
        n_groups=len(order),
        metrics=metrics,
        typical={names[k]: round(v, 4) for k, v in typical.items() if k in names},
        cell_median={"median_genes": round(float(np.nanmedian(frame["genes"])), 4),
                     "median_counts": round(float(np.nanmedian(frame["counts"])), 4)},
        after_filtering=_filtered(adata),
        clusters=rows,
        computed_on_demand=bool(cells.get("computed")),
        note=" ".join(n for n in (
            lineage_note,
            "" if len(order) >= MIN_GROUPS
            else f"Only {len(order)} groups: too few to say which stands out; only size is flagged.",
            "" if cells.get("has_mt", True)
            else targeted_panel.NO_MITO_NOTE if spatial.targeted_panel(adata) else NO_MT_NOTE,
            "" if malat1_median is None or malat1_median >= MALAT1_MIN_MEDIAN_PCT
            else _malat1_floor_note(malat1_median),
            "" if not unchecked
            else (("Clusters " + ", ".join(unchecked) + " have" if len(unchecked) > 1
                   else f"Cluster {unchecked[0]} has")
                  + " a high dissociation-stress score, but "
                  f"'{column}' has no marker genes to check it against: compute markers to "
                  "see whether it is stress-driven."),
            "" if not hb_unchecked
            else (("Clusters " + ", ".join(hb_unchecked) + " have" if len(hb_unchecked) > 1
                   else f"Cluster {hb_unchecked[0]} has")
                  + " far more haemoglobin RNA than the other clusters, but "
                  f"'{column}' has no marker genes to tell erythroid cells from ambient RNA: "
                  "compute markers to see which it is."),
        ) if n),
        ambient_hb=ambient,
        drivers=drivers.drivers,
        drivers_checked=drivers.checked,
        drivers_space=drivers.representation,
        drivers_n_pcs=drivers.n_pcs,
        dataset_median_pct_malat1=malat1_median,
        doublet_source=None if used is None else used.label,
        doublet_like_checked=like_checked,
        nuclei_like_mito=nuclei_like,
        intronic_column=intronic[0] if intronic is not None else None,
        ambient_markers_approximate=amb_approx,
        ambient_markers_help=(ambient_rna.HELP["flag_approximate"] if amb_approx
                              else ambient_rna.HELP["flag"]) if amb else "",
    )


def _ambient(adata, column: str) -> tuple[dict[str, list[str]], bool]:
    """Ambient-looking top markers per cluster (``core/ambient.py``), and whether they come
    from the soup share alone. Empty for a large or backed dataset (it reads every
    cell's counts) and when they can't be worked out; never fails the table."""
    from scview.core import large_mode

    if large_mode.is_large(adata) or getattr(adata, "isbacked", False):
        return {}, False
    try:
        body = ambient_rna.diagnose_cached(adata, column)
    except Exception as exc:  # noqa: BLE001 - optional
        logger.debug("cluster QC: no ambient markers: %s", exc)
        return {}, False
    groups = body["markers"]["groups"]
    return {g: [r["gene"] for r in rows] for g, rows in groups.items()}, bool(body["markers"]["approximate"])
def doublet_like_pair(markers: dict[str, set[str]], rows: dict[str, "ClusterQC"],
                      group: str) -> tuple[str, str, int, int, int] | None:
    """The two clusters ``group`` looks like a mix of (``DOUBLET_LIKE_*``), as (a, b, its
    markers among a's, among b's, among either's), or None. ``markers``: each cluster's
    top markers; ``rows``: their QC rows (cells, median counts and genes, flags: a
    low-quality or low-genes cluster is no parent, its few genes are ambient RNA). Of the
    pairs that pass, the one whose markers cover most of the cluster's, then the more even
    split between the two, then the RNA closer to the sum of the two, then the order of the clusters."""
    mine, me = markers.get(group, set()), rows[group]
    if len(mine) < DOUBLET_LIKE_MIN_MARKERS or not me.median_counts or me.median_genes is None:
        return None
    others = [g for g in rows if g != group and len(markers.get(g, ())) >= DOUBLET_LIKE_MIN_MARKERS
              and rows[g].median_counts and rows[g].median_genes is not None
              and not set(rows[g].flags) & _NOT_A_PARENT]
    best, key = None, None
    for i, a in enumerate(others):
        in_a = mine & markers[a]
        if len(in_a) < DOUBLET_LIKE_EACH * len(mine):
            continue
        for b in others[i + 1:]:
            in_b = mine & markers[b]
            either = len(in_a | in_b)
            if len(in_b) < DOUBLET_LIKE_EACH * len(mine) or either < DOUBLET_LIKE_SHARE * len(mine):
                continue
            ra, rb = rows[a], rows[b]
            if max(ra.n_cells, rb.n_cells) <= me.n_cells:
                continue
            big, small = sorted((ra.median_counts, rb.median_counts), reverse=True)
            if me.median_counts < big + DOUBLET_LIKE_RNA * small \
                    or me.median_genes < max(ra.median_genes, rb.median_genes):
                continue
            k = (either, min(len(in_a), len(in_b)), me.median_counts / (big + small))
            if key is None or k > key:
                best, key = (a, b, len(in_a), len(in_b), either), k
    return best


def _flag_doublet_like(adata, column: str, rows: list["ClusterQC"]) -> bool:
    """Flag ``doublet_like`` clusters (``doublet_like_pair``) on their stored markers;
    False when the column has none (nothing to judge by). Never fails the table."""
    try:
        # Unnamed genes kept: the doublet-like rule was measured on the full lists.
        top = marker_selection.group_markers(adata, column, n=DOUBLET_LIKE_MARKERS, skip_unnamed=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cluster QC: no markers for doublet-like clusters: %s", exc)
        return False
    if not top:
        return False
    markers = {g: set(v) for g, v in top.items()}
    by = {r.group: r for r in rows}
    for row in rows:
        found = doublet_like_pair(markers, by, row.group)
        if found is None:
            continue
        a, b, in_a, in_b, either = found
        ra, rb = by[a], by[b]
        k = len(markers[row.group])
        row.flags.append("doublet_like")
        row.doublet_like_of = [a, b]
        row.doublet_like_shared = [in_a, in_b, either, k]
        row.reasons.append(
            f"Looks like doublets of {a} + {b}: {either} of its top {k} markers are theirs "
            f"({in_a} of {a}'s, {in_b} of {b}'s), and its cells carry more RNA, median "
            f"{_fmt(row.median_counts)} counts vs {_fmt(ra.median_counts)} and {_fmt(rb.median_counts)}."
        )
    return True


def _flag_nuclei_mito(row: ClusterQC, typical: dict[str, float]) -> None:
    """``nuclei_mito`` (``NUCLEI_MITO_*``): far more cells with mito reads, and a far higher mean
    % mito, than the typical cluster, where the typical cell has almost none."""
    share, mean = row.mt_detected, row.mean_pct_mt
    t_share, t_mean = typical.get("mt_any"), typical.get("mt_mean")
    if None in (share, mean) or t_share is None or t_mean is None:
        return
    if mean >= NUCLEI_MITO_FOLD * t_mean and mean - t_mean >= NUCLEI_MITO_MIN_DIFF \
            and share - t_share >= NUCLEI_MITO_SHARE_DIFF:
        row.flags.append("nuclei_mito")
        row.reasons.append(
            f"Mitochondrial RNA in nuclei: {share:.0%} of its cells have mito reads vs {t_share:.0%} in a "
            f"typical cluster, a mean of {mito_text(mean)} of their reads vs {mito_text(t_mean)}. Where most "
            "cells have almost none, this points to cytoplasm on the nuclei or whole cells (contamination)."
        )


def _flag_marker_makeup(adata, column: str, rows: list[ClusterQC], small_cut: float) -> None:
    """``low_content``, ``noncoding_markers`` and ``sex_markers`` from the stored markers (none
    without them). Never fails the table."""
    from scview.core import sex_check  # sex_check imports qc_groups, which imports this module

    try:
        named = marker_selection.group_markers(adata, column, n=DOUBLET_LIKE_MARKERS)
        full = marker_selection.group_markers(adata, column, n=DOUBLET_LIKE_MARKERS, skip_unnamed=False)
        sex_ids = sex_check.sex_linked_ids(adata)  # Ensembl-id names -> symbols
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cluster QC: no markers for the marker checks: %s", exc)
        return
    if not named and not full:
        return
    sex_genes = {g for genes in sex_check.SEX_LINKED_GENES.values() for g in genes}
    by ={r.group: r for r in rows}
    for row in rows:
        mine = named.get(row.group, [])
        top = full.get(row.group, [])
        # Led by sex-linked genes
        sex = [sex_ids.get(g, g) for g in mine[:SEX_TOP] if g in sex_genes or g in sex_ids]
        if sex:
            row.sex_markers = sex
            row.flags.append("sex_markers")
            row.reasons.append(
                f"Sex genes lead its markers ({', '.join(sex)} in its top {SEX_TOP}): it may hold one sex's "
                "cells of a type split by sex or donor, not a type of its own."
            )
        # Small RNAs, pseudogenes, unnamed genes
        junk = [g for g in mine if NONCODING_RE.match(g)]
        head = top[:UNINFORMATIVE_TOP]
        vague = [g for g in head if UNINFORMATIVE_RE.match(g) or gene_filters.is_unnamed(g)]
        if len(mine) >= DOUBLET_LIKE_MIN_MARKERS and len(junk) >= NONCODING_SHARE * len(mine):
            row.noncoding_markers = junk
            row.flags.append("noncoding_markers")
            row.reasons.append(
                f"Non-coding markers: {len(junk)} of its top {len(mine)} markers are small RNAs, multi-copy "
                f"genes or pseudogenes ({', '.join(junk[:6])}): a technical cluster (another protocol, "
                "ambient RNA, damage) more likely than a cell type."
            )
        elif len(head) >= UNINFORMATIVE_TOP and len(vague) > len(head) / 2:
            row.noncoding_markers = vague
            row.flags.append("noncoding_markers")
            row.reasons.append(
                f"Uninformative markers: {len(vague)} of its top {len(head)} are unnamed genes, MALAT1 / NEAT1 "
                f"or predicted genes ({', '.join(vague[:6])}): little to name a cell type by."
            )
    # The same identity with less RNA
    markers = {g: set(v) for g, v in named.items()}
    for row in rows:
        mine = markers.get(row.group, set())
        if len(mine) < DOUBLET_LIKE_MIN_MARKERS or not row.median_counts or not row.median_genes:
            continue
        best, shared = None, 0
        for other, genes in markers.items():
            o = by.get(other)
            if other == row.group or o is None or o.n_cells < small_cut or set(o.flags) & _NOT_A_PARENT:
                continue
            k = len(mine & genes)
            if k > shared:
                best, shared = o, k
        if best is None or shared < LOW_CONTENT_SHARE * len(mine) or not best.median_counts \
                or not best.median_genes:
            continue
        rc, rg = row.median_counts / best.median_counts, row.median_genes / best.median_genes
        if rc <= LOW_CONTENT_COUNTS and rg <= LOW_CONTENT_GENES:
            row.low_content_of = best.group
            row.low_content_shared = [shared, len(mine)]
            row.flags.append("low_content")
            row.reasons.append(
                f"Low RNA for its identity: {shared} of its top {len(mine)} markers are {best.group}'s, but its "
                f"cells carry {rc:.2f}x the counts (median {_fmt(row.median_counts)} vs "
                f"{_fmt(best.median_counts)}) and {rg:.2f}x the genes of {best.group}: likely the same cells "
                "damaged or losing RNA, or ambient droplets, split off by depth."
            )


def _drivers(adata, labels: pd.Series, cells: dict, frame: pd.DataFrame,
             erythroid: set[str]) -> qc_drivers.DriverCheck:
    """Technical scores against the leading PCs (``core/qc_drivers.py``). Never
    fails the per-cluster table."""
    try:
        lab = labels.astype(str).where(labels.notna()).to_numpy()
        scores: dict[str, np.ndarray] = {
            "log_counts": np.log1p(np.asarray(cells["total_counts"], dtype=np.float64)),
        }
        if "pct_mt" in frame.columns:
            scores["pct_mt"] = frame["pct_mt"].to_numpy(dtype=np.float64)
        if "pct_ribo" in frame.columns:
            scores["pct_ribo"] = frame["pct_ribo"].to_numpy(dtype=np.float64)
        masks: dict[str, np.ndarray] = {}
        if "pct_hb" in frame.columns:
            scores["pct_hb"] = np.log1p(frame["pct_hb"].to_numpy(dtype=np.float64))
            masks["pct_hb"] = ~np.isin(lab, list(erythroid))
        cc = cell_cycle.cell_cycle_scores(adata.obs)
        if cc is not None:
            scores["S_score"], scores["G2M_score"] = cc
        return qc_drivers.for_adata(adata, lab, scores, masks)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cluster QC: driver check failed: %s", exc)
        return qc_drivers.DriverCheck()


def for_adaptor(adaptor, groupby: str | None = None) -> ClusterQCReport:
    """``cluster_qc`` on a loaded dataset, reusing its cached per-cell QC arrays.

    Accepts anything with ``.adata`` (the co-pilot's insight tests pass a bare
    namespace); the arrays are then computed here."""
    adata = adaptor.adata
    fn = getattr(adaptor, "qc_per_cell", None)
    if callable(fn):
        return cluster_qc(adata, groupby, lambda: fn(ribo=True))
    return cluster_qc(adata, groupby)


def _filtered(adata) -> bool:
    """Whether the dataset's provenance records the cell filtering step."""
    try:
        from scview.core import provenance

        return any(h.get("step") == "filtering"
                   for h in provenance.read_provenance(adata).get("history", []))
    except Exception:  # pragma: no cover - defensive: a malformed record
        return False


def _num(v, digits: int = 2) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def _fmt(v: float) -> str:
    """Genes or counts per cell: whole numbers."""
    return f"{v:,.0f}"


def _pct(v: float) -> str:
    """A haemoglobin share, readable from red cells (50%) down to ambient traces."""
    if v >= 1:
        return f"{v:.1f}%"
    if v >= 0.01:
        return f"{v:.2f}%"
    return "<0.01%" if v > 0 else "0%"


def mito_text(v: float) -> str:
    """A % mito readable from damaged cells (25%) down to nuclei (0.008%): the digits it needs."""
    if round(v, 1) >= 1:  # decided on the rounded value: 0.0999 reads 0.10%, like 0.1
        return f"{v:.1f}%"
    if round(v, 2) >= 0.1:
        return f"{v:.2f}%"
    if round(v, 3) >= 0.001:
        return f"{v:.3f}%"
    return "<0.001%" if v > 0 else "0%"


def _stress_outlier(z: float, score, typical) -> bool:
    try:
        return z >= stress.Z_OUTLIER and float(score) - float(typical) >= stress.MIN_DIFF
    except (TypeError, ValueError):
        return False


def _flag(row: ClusterQC, z: dict[str, float], typical: dict[str, float], small_cut: float,
          markers_checked: bool = False, hb_checked: bool = False, compare: bool = True,
          median_pct_hb: float | None = None) -> None:
    """Attach flags + plain-language reasons (and notes) to one cluster's row.

    ``compare`` is False when there are too few clusters to say one stands out."""
    genes, mito = row.median_genes, row.median_pct_mt
    t_genes, t_mito = typical.get("genes"), typical.get("pct_mt")
    few_genes = genes is not None and bool(t_genes) and z.get("genes", 0.0) <= -_Z_PAIR \
        and genes <= 0.7 * t_genes
    more_mito = mito is not None and t_mito is not None and z.get("pct_mt", 0.0) >= _Z_PAIR \
        and mito - t_mito >= 3.0

    if few_genes and more_mito:
        row.flags.append("low_quality")
        row.reasons.append(
            f"Likely low quality: median {_fmt(genes)} genes per cell vs {_fmt(t_genes)} in a "
            f"typical cluster, and {mito:.1f}% mitochondrial reads vs {t_mito:.1f}%."
        )
    else:
        if mito is not None and t_mito is not None and z.get("pct_mt", 0.0) >= _Z_SINGLE \
                and mito - t_mito >= 5.0:
            row.flags.append("high_mito")
            row.reasons.append(
                f"High mitochondrial content: median {mito:.1f}% vs {t_mito:.1f}% in a typical cluster."
            )
        if genes is not None and t_genes and z.get("genes", 0.0) <= -_Z_SINGLE \
                and genes <= 0.5 * t_genes:
            row.flags.append("low_genes")
            row.reasons.append(
                f"Few genes per cell: median {_fmt(genes)} vs {_fmt(t_genes)} in a typical cluster."
            )

    frac, t_frac = row.doublet_fraction, typical.get("doublet")
    score, t_score = row.median_doublet_score, typical.get("doublet_score")
    if frac is not None and t_frac is not None:
        if z.get("doublet", 0.0) >= _Z_SINGLE and frac >= 2 * t_frac and frac - t_frac >= 0.1:
            row.flags.append("doublet_rich")
            row.reasons.append(
                f"Doublet-rich: {frac:.0%} of its cells are predicted doublets vs {t_frac:.0%} "
                "in a typical cluster."
            )
    elif score is not None and t_score is not None:
        if z.get("doublet_score", 0.0) >= _Z_SINGLE and score >= 1.5 * t_score:
            row.flags.append("doublet_rich")
            row.reasons.append(
                f"Doublet-rich: median doublet score {score:.2f} vs {t_score:.2f} in a typical cluster."
            )

    st, t_st = row.median_stress_score, typical.get("stress")
    hits = row.stress_markers
    if markers_checked and _stress_outlier(z.get("stress", 0.0), st, t_st) \
            and len(hits) >= stress.MIN_SIGNATURE_MARKERS:
        row.flags.append("stress_driven")
        row.reasons.append(
            f"Possibly dissociation stress: median stress score {st:.2f} vs {t_st:.2f} in "
            f"a typical cluster, and {len(hits)} of its top {stress.TOP_MARKERS} markers are "
            f"stress genes ({', '.join(hits[:6])})."
        )

    hb, t_hb = row.mean_pct_hb, typical.get("hb")
    if row.erythroid:
        # Said only where its haemoglobin is high enough to be mistaken for ambient RNA.
        if hb is not None and hb > 0 and (t_hb is None or hb >= haemoglobin.AMBIENT_FOLD * t_hb):
            lineage = haemoglobin.lineage_markers(row.erythroid_markers)
            if len(lineage) >= haemoglobin.MIN_ERYTHROID_MARKERS:
                genes = ", ".join(row.erythroid_markers[:5])
                why = f"its top markers include red-cell genes ({genes})"
            else:
                level = _pct(median_pct_hb or 0.0)
                why = f"its median cell is {level} haemoglobin, a red-cell level"
            vs = f" vs {_pct(t_hb)} in a typical non-erythroid cluster" if t_hb is not None else ""
            row.notes.append(
                f"Erythroid: {why}, and haemoglobin is {_pct(hb)} of its reads{vs}. Real "
                "red-cell expression, not ambient contamination."
            )
    elif hb_checked and compare and haemoglobin.ambient_outlier(z.get("hb", 0.0), hb, t_hb):
        row.flags.append("hb_ambient")
        row.reasons.append(
            f"Haemoglobin RNA without erythroid markers: {_pct(hb)} of its reads are haemoglobin "
            f"vs {_pct(t_hb)} in a typical non-erythroid cluster, and its top markers hold no "
            "red-cell genes besides haemoglobin. Likely ambient RNA from lysed red cells, or "
            "red-cell doublets."
        )

    cyc, t_cyc = row.cycling_fraction, typical.get("cycling")
    if compare and cyc is not None and t_cyc is not None and cyc >= cell_cycle.CYCLING_SHARE \
            and cyc - t_cyc >= cell_cycle.CYCLING_MARGIN:
        row.flags.append("cycling")
        row.reasons.append(
            f"Mostly cycling cells: {cyc:.0%} have a clear S or G2M score (above "
            f"{cell_cycle.NEAR_ZERO_MARGIN:g}) vs {t_cyc:.0%} in a typical cluster. The cell cycle "
            "may be what sets it apart (a dividing cell type, or a dividing state of another); "
            "regress out the S and G2M scores if that is not the biology of interest (Scaling "
            "step, Regress out; S - G2M keeps dividing cells apart from resting ones)."
        )

    if row.n_cells < small_cut:
        row.flags.append("small")
        row.reasons.append(
            f"Very small: {row.n_cells:,} cells ({row.fraction:.1%} of the data); "
            "markers and DE for it are unreliable."
        )


def _flag_malat1(row: ClusterQC, dataset_median: float | None, compare: bool) -> None:
    """``low_malat1``: the cluster's median MALAT1 share below ``LOW_MALAT1_FRACTION``
    of the median cell's. Not for erythroid clusters (red cells have no nucleus)."""
    m = row.median_pct_malat1
    if not compare or row.erythroid or m is None or dataset_median is None \
            or dataset_median < MALAT1_MIN_MEDIAN_PCT:
        return
    cut = LOW_MALAT1_FRACTION * dataset_median
    if m < cut:
        row.flags.append("low_malat1")
        row.reasons.append(
            f"Low MALAT1: median {_pct(m)} of its reads vs {_pct(dataset_median)} in the median "
            f"cell (flagged below a third of it, {_pct(cut)}). MALAT1 is kept in the nucleus, "
            "so these may be cell fragments or cytoplasm without its nucleus."
        )


def _malat1_floor_note(median: float) -> str:
    return (f"MALAT1 is barely captured here ({_pct(median)} of the median cell's reads, under "
            f"{MALAT1_MIN_MEDIAN_PCT:g}%): too little to judge nuclear loss, so no cluster is "
            "flagged for low MALAT1.")


def malat1_line(report: ClusterQCReport, tag: str) -> str:
    """One co-pilot fact line on MALAT1: the median cell's share and the
    low-MALAT1 clusters (or none), with the rule. Empty when not measured."""
    d = report.dataset_median_pct_malat1
    if "pct_malat1" not in report.metrics or d is None or report.n_groups < MIN_GROUPS:
        return ""
    if d < MALAT1_MIN_MEDIAN_PCT:
        return f"- MALAT1: {_malat1_floor_note(d)} [{tag}]"
    low = [c for c in report.clusters if "low_malat1" in c.flags]
    which = ", ".join(f"{c.group} ({_pct(c.median_pct_malat1 or 0.0)})" for c in low) or "none"
    return (f"- MALAT1 (nuclear-retained RNA; low share = fragments or lost nuclei): "
            f"{_pct(d)} of the median cell's reads; clusters with a median below a third of "
            f"that ({_pct(LOW_MALAT1_FRACTION * d)}): {which} [{tag}]")


def qc_tag(column: str, group: str) -> str:
    """Citation tag for one cluster's QC numbers (a chip that jumps to the cluster)."""
    return f"result:qc:{column}:{group}"


def grounding_block(report: ClusterQCReport) -> tuple[list[str], list[tuple[str, str]]]:
    """Compact prompt lines + ``(tag, detail)`` citations for the co-pilot.

    Only the typical values, the flagged clusters and the ones with a note
    (erythroid), so the prompt stays small; an unflagged cluster is covered by the
    "within the normal range" line. Dataset-level findings (ambient haemoglobin,
    technical drivers of the PCs) cite the column's own tag."""
    if not report.column or not report.clusters:
        return [], []
    t = report.typical
    typ = [f"{_fmt(t['median_genes'])} genes/cell" if "median_genes" in t else "",
           f"{_fmt(t['median_counts'])} counts/cell" if "median_counts" in t else "",
           f"{mito_text(t['median_pct_mt'])} mito" if "median_pct_mt" in t else "",
           f"mito reads in {t['mt_detected']:.0%} of cells (mean {mito_text(t.get('mean_pct_mt', 0.0))})"
           if report.nuclei_like_mito and "mt_detected" in t else "",
           f"{t['doublet_fraction']:.0%} doublets" if "doublet_fraction" in t else "",
           f"stress score {t['median_stress_score']:.2f}" if "median_stress_score" in t else "",
           f"{_pct(t['mean_pct_hb'])} haemoglobin" if "mean_pct_hb" in t else "",
           f"{t['cycling_fraction']:.0%} cycling cells" if "cycling_fraction" in t else ""]
    head = f"result:qc:{report.column}"
    lines = [f"\n## Per-cluster QC (deterministic; column '{report.column}', {report.n_groups} groups)"]
    cm = report.cell_median
    if "median_genes" in cm and "median_counts" in cm:
        # The number the QC plots show; a model read the typical cluster's as this.
        when = "after QC filtering" if report.after_filtering else "no QC filtering recorded"
        lines.append(f"- Median cell (all {report.n_cells:,} cells, {when}): "
                     f"{_fmt(cm['median_genes'])} genes/cell, {_fmt(cm['median_counts'])} "
                     f"counts/cell [{head}]")
    lines.append("- Typical cluster (the median of the per-cluster medians, not the median cell): "
                 + ", ".join(s for s in typ if s) + f" [{head}]")
    cites: list[tuple[str, str]] = [(head, f"{report.column}: {len(report.flagged)} of "
                                            f"{report.n_groups} clusters flagged")]
    flagged = report.flagged
    listed = [c for c in report.clusters if c.flags or c.notes]
    for c in listed:
        tag = qc_tag(report.column, c.group)
        lines.append(f"- **{c.group}** ({c.n_cells:,} cells) [{tag}]: " + " ".join(c.reasons + c.notes))
        cites.append((tag, f"{c.group}: " + "; ".join(c.flags + (["erythroid"] if c.notes else []))))
    rest = report.n_groups - len(listed)
    if not flagged:
        checked = ["genes", "counts"] + [w for m, w in (("pct_mt", "mito"), ("doublet_fraction", "doublets"),
                                                        ("doublet_like", "doublet-like markers"),
                                                        ("mt_detected", "mito reads in nuclei"),
                                                        ("doublet_like", "marker make-up (low RNA, "
                                                                         "non-coding or sex genes)"),
                                                        ("stress_score", "dissociation stress"),
                                                        ("pct_hb", "haemoglobin"),
                                                        ("cycling_fraction", "cell cycle"),
                                                        ("pct_malat1", "MALAT1"))
                                         if m in report.metrics
                                         or (m == "doublet_like" and report.doublet_like_checked)]
        lines.append("- No cluster stands out: all are within the normal range on "
                     + ", ".join(checked[:-1]) + " and " + checked[-1]
                     + (" (an erythroid cluster's haemoglobin is its own)." if listed else "."))
    elif rest:
        lines.append(f"- The other {rest} clusters are within the normal range.")
    if report.ambient_hb:
        lines.append(f"- Ambient haemoglobin: {report.ambient_hb} [{head}]")
    if mal := malat1_line(report, head):
        lines.append(mal)
    if report.drivers_n_pcs:
        space = f"first {report.drivers_n_pcs} PCs of {report.drivers_space}"
        for d in report.drivers:
            lines.append(f"- Technical driver ({space}): {d.reason} [{head}]")
        if not report.drivers and report.drivers_checked:
            names = [s[:1].lower() + s[1:] if s[1:2].islower() else s
                     for s in report.drivers_checked]  # "log total counts", "G2M score"
            lines.append(f"- Technical scores vs the clustering's PCs ({space}): none of "
                         + ", ".join(names) + f" correlates strongly with one (|r| >= "
                         f"{qc_drivers.DRIVER_R:g}). [{head}]")
    if report.note:
        lines.append(f"- Note: {report.note}")
    return lines, cites
