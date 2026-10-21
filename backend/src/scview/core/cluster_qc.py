"""Per-cluster quality flags: which clusters stand out against the others.

Deterministic and reproducible (no LLM). For a grouping column (default: the
active clustering) each cluster's cells are summarised (size, median genes and
counts per cell, median % mitochondrial and % ribosomal, doublet fraction or
score) and a cluster is flagged when it sits far from the typical cluster on
robust statistics: the median across clusters and the MAD around it, so one bad
cluster cannot hide by inflating the spread. Absolute floors stop near-identical
clusters from turning a tiny difference into a flag.

Flags: ``low_quality`` (few genes AND high mito), ``high_mito``, ``low_genes``,
``doublet_rich``, ``stress_driven`` (dissociation-stress score far above the
others AND stress genes among its top markers; see ``core/stress.py``),
``hb_ambient`` (haemoglobin RNA in a cluster whose markers are not erythroid;
see ``core/haemoglobin.py``), ``cycling`` (mostly cells with clear S/G2M scores;
see ``core/cell_cycle.py``), ``low_malat1`` (little of the nuclear-retained MALAT1
RNA: fragments or cytoplasm without a nucleus) and ``small``. An author's
intronic / unspliced share column, when obs has one, is summarised (never computed). Erythroid clusters get a note, not a
flag: their haemoglobin is their own. At the dataset level, haemoglobin spread
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

from scview.core import cell_cycle, haemoglobin, qc_drivers, stress
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
}

# Doublet calls stored as text by other tools (scDblFinder, demuxlet, DoubletFinder).
_DOUBLET_CLASS_COLS = ("scDblFinder.class", "scDblFinder_class", "multiplets", "doublet_class")
_DOUBLET_CLASS_PREFIX = "DF.classification"


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
    median_pct_malat1: float | None = None  # % of a cell's counts from MALAT1
    median_pct_intronic: float | None = None  # the author's intronic/unspliced share, %
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

    @property
    def flagged(self) -> list[ClusterQC]:
        return [c for c in self.clusters if c.flags]


def doublet_calls(obs: pd.DataFrame) -> np.ndarray | None:
    """Per-cell doublet calls as booleans, from Scrublet's bool column or a text one."""
    if "predicted_doublet" in obs.columns:
        col = obs["predicted_doublet"]
        if pd.api.types.is_bool_dtype(col):
            return col.to_numpy(dtype=bool)
        # Categorical/text "True"/"False" after a round trip: astype(bool) would make
        # every non-empty string True.
        return col.astype(str).str.lower().isin(("true", "1", "doublet")).to_numpy()
    names = [c for c in _DOUBLET_CLASS_COLS if c in obs.columns]
    names += [c for c in obs.columns if str(c).startswith(_DOUBLET_CLASS_PREFIX)]
    for name in names:
        return obs[name].astype(str).str.lower().eq("doublet").to_numpy()
    return None


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
    if cells.get("pct_ribo") is not None:
        frame["pct_ribo"] = cells["pct_ribo"]
        metrics.append("pct_ribo")
    calls = doublet_calls(adata.obs)
    if calls is not None:
        frame["doublet"] = calls.astype(np.float64)
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
        )
        _flag(row, {k: float(v.loc[g]) for k, v in z.items()}, typical, small_cut,
              markers_checked=sig_markers is not None, hb_checked=ery_markers is not None,
              compare=len(order) >= MIN_GROUPS, median_pct_hb=_num(m.get("pct_hb")))
        _flag_malat1(row, malat1_median, compare=len(order) >= MIN_GROUPS)
        rows.append(row)

    names = {"genes": "median_genes", "counts": "median_counts", "pct_mt": "median_pct_mt",
             "pct_ribo": "median_pct_ribo", "doublet": "doublet_fraction",
             "doublet_score": "median_doublet_score", "stress": "median_stress_score",
             "hb": "mean_pct_hb", "cycling": "cycling_fraction",
             "pct_malat1": "median_pct_malat1", "intronic": "median_pct_intronic"}
    # A score outlier that couldn't be checked against markers isn't flagged: say so.
    unchecked = [] if sig_markers is not None or "stress" not in z else [
        g for g in order if _stress_outlier(float(z["stress"].loc[g]),
                                            medians.loc[g].get("stress"), typical.get("stress"))
    ]
    hb_unchecked = [] if ery_markers is not None or "hb" not in z else [
        g for g in others if haemoglobin.ambient_outlier(
            float(z["hb"].loc[g]), medians.loc[g, "hb"], typical.get("hb"))
    ]
    ambient = ""
    if typical.get("hb") is not None and typical["hb"] >= haemoglobin.AMBIENT_WIDE_PCT:
        ambient = (
            f"Haemoglobin RNA is spread across the non-erythroid clusters ({_pct(typical['hb'])} "
            "of the reads in a typical one): likely ambient RNA from lysed red cells. It adds "
            "red-cell genes to every cluster and can blur markers and DE; ambient-RNA removal "
            "before import (SoupX, CellBender) cleans it."
        )
    drivers = _drivers(adata, labels, cells, frame, erythroid)
    return ClusterQCReport(
        column=column,
        n_cells=n,
        n_groups=len(order),
        metrics=metrics,
        typical={names[k]: round(v, 4) for k, v in typical.items() if k in names},
        clusters=rows,
        computed_on_demand=bool(cells.get("computed")),
        note=" ".join(n for n in (
            "" if len(order) >= MIN_GROUPS
            else f"Only {len(order)} groups: too few to say which stands out; only size is flagged.",
            "" if cells.get("has_mt", True)
            else NO_MT_NOTE,
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
        intronic_column=intronic[0] if intronic is not None else None,
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
            "regress out the S and G2M scores if that is not the biology of interest."
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
    if not compare or row.erythroid or m is None or not dataset_median or dataset_median <= 0:
        return
    cut = LOW_MALAT1_FRACTION * dataset_median
    if m < cut:
        row.flags.append("low_malat1")
        row.reasons.append(
            f"Low MALAT1: median {_pct(m)} of its reads vs {_pct(dataset_median)} in the median "
            f"cell (flagged below a third of it, {_pct(cut)}). MALAT1 is kept in the nucleus, "
            "so these may be cell fragments or cytoplasm without its nucleus."
        )


def malat1_line(report: ClusterQCReport, tag: str) -> str:
    """One co-pilot fact line on MALAT1: the median cell's share and the
    low-MALAT1 clusters (or none), with the rule. Empty when not measured."""
    d = report.dataset_median_pct_malat1
    if "pct_malat1" not in report.metrics or d is None or report.n_groups < MIN_GROUPS:
        return ""
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
           f"{t['median_pct_mt']:.1f}% mito" if "median_pct_mt" in t else "",
           f"{t['doublet_fraction']:.0%} doublets" if "doublet_fraction" in t else "",
           f"stress score {t['median_stress_score']:.2f}" if "median_stress_score" in t else "",
           f"{_pct(t['mean_pct_hb'])} haemoglobin" if "mean_pct_hb" in t else "",
           f"{t['cycling_fraction']:.0%} cycling cells" if "cycling_fraction" in t else ""]
    head = f"result:qc:{report.column}"
    lines = [
        f"\n## Per-cluster QC (deterministic; column '{report.column}', {report.n_groups} groups)",
        "- Typical cluster (median across clusters): " + ", ".join(s for s in typ if s)
        + f" [{head}]",
    ]
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
                                                        ("stress_score", "dissociation stress"),
                                                        ("pct_hb", "haemoglobin"),
                                                        ("cycling_fraction", "cell cycle"),
                                                        ("pct_malat1", "MALAT1"))
                                         if m in report.metrics]
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
