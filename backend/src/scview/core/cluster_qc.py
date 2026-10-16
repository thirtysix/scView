"""Per-cluster quality flags: which clusters stand out against the others.

Deterministic and reproducible (no LLM). For a grouping column (default: the
active clustering) each cluster's cells are summarised (size, median genes and
counts per cell, median % mitochondrial and % ribosomal, doublet fraction or
score) and a cluster is flagged when it sits far from the typical cluster on
robust statistics: the median across clusters and the MAD around it, so one bad
cluster cannot hide by inflating the spread. Absolute floors stop near-identical
clusters from turning a tiny difference into a flag.

Flags: ``low_quality`` (few genes AND high mito), ``high_mito``, ``low_genes``,
``doublet_rich`` and ``small``. The same numbers ground the co-pilot, so "which
clusters look low quality?" is answered from them (``grounding_block``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

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
}
_SMALL_MIN_CELLS = 20
_SMALL_FRACTION = 0.0025

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
    flags: list[str] = []
    reasons: list[str] = []  # one plain-language sentence per flag, with the numbers


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

    grouped = frame.groupby(labels.astype(str).where(labels.notna()).to_numpy(), sort=False)
    sizes = grouped.size()
    medians = grouped.median()
    if "doublet" in frame.columns:
        medians["doublet"] = grouped["doublet"].mean()  # a fraction, not a median
    # Keep the column's own category order (cluster 0, 1, 2 …), not first-seen order.
    order = (
        [str(c) for c in labels.cat.categories] if hasattr(labels, "cat")
        else sorted(sizes.index, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
    )
    order = [g for g in order if g in sizes.index]
    sizes, medians = sizes.loc[order], medians.loc[order]

    typical = {k: float(np.nanmedian(v)) for k, v in medians.items()}
    z: dict[str, pd.Series] = {}
    if len(order) >= MIN_GROUPS:
        z["genes"] = _robust_z(np.log1p(medians["genes"]), _SCALE_FLOOR["log_genes"])
        if "pct_mt" in medians:
            z["pct_mt"] = _robust_z(medians["pct_mt"], _SCALE_FLOOR["pct_mt"])
        if "doublet" in medians:
            z["doublet"] = _robust_z(medians["doublet"], _SCALE_FLOOR["doublet_fraction"])
        if "doublet_score" in medians:
            z["doublet_score"] = _robust_z(medians["doublet_score"], _SCALE_FLOOR["doublet_score"])

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
        )
        _flag(row, {k: float(v.loc[g]) for k, v in z.items()}, typical, small_cut)
        rows.append(row)

    names = {"genes": "median_genes", "counts": "median_counts", "pct_mt": "median_pct_mt",
             "pct_ribo": "median_pct_ribo", "doublet": "doublet_fraction",
             "doublet_score": "median_doublet_score"}
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
        ) if n),
    )


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


def _flag(row: ClusterQC, z: dict[str, float], typical: dict[str, float], small_cut: float) -> None:
    """Attach flags + plain-language reasons to one cluster's row."""
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

    if row.n_cells < small_cut:
        row.flags.append("small")
        row.reasons.append(
            f"Very small: {row.n_cells:,} cells ({row.fraction:.1%} of the data); "
            "markers and DE for it are unreliable."
        )


def qc_tag(column: str, group: str) -> str:
    """Citation tag for one cluster's QC numbers (a chip that jumps to the cluster)."""
    return f"result:qc:{column}:{group}"


def grounding_block(report: ClusterQCReport) -> tuple[list[str], list[tuple[str, str]]]:
    """Compact prompt lines + ``(tag, detail)`` citations for the co-pilot.

    Only the typical values and the flagged clusters, so the prompt stays small;
    an unflagged cluster is covered by the "within the normal range" line."""
    if not report.column or not report.clusters:
        return [], []
    t = report.typical
    typ = [f"{_fmt(t['median_genes'])} genes/cell" if "median_genes" in t else "",
           f"{_fmt(t['median_counts'])} counts/cell" if "median_counts" in t else "",
           f"{t['median_pct_mt']:.1f}% mito" if "median_pct_mt" in t else "",
           f"{t['doublet_fraction']:.0%} doublets" if "doublet_fraction" in t else ""]
    head = f"result:qc:{report.column}"
    lines = [
        f"\n## Per-cluster QC (deterministic; column '{report.column}', {report.n_groups} groups)",
        "- Typical cluster (median across clusters): " + ", ".join(s for s in typ if s)
        + f" [{head}]",
    ]
    cites: list[tuple[str, str]] = [(head, f"{report.column}: {len(report.flagged)} of "
                                            f"{report.n_groups} clusters flagged")]
    flagged = report.flagged
    for c in flagged:
        tag = qc_tag(report.column, c.group)
        lines.append(f"- **{c.group}** ({c.n_cells:,} cells) [{tag}]: " + " ".join(c.reasons))
        cites.append((tag, f"{c.group}: " + "; ".join(c.flags)))
    rest = report.n_groups - len(flagged)
    if not flagged:
        checked = ["genes", "counts"] + [w for m, w in (("pct_mt", "mito"), ("doublet_fraction", "doublets"))
                                         if m in report.metrics]
        lines.append("- No cluster stands out: all are within the normal range on "
                     + ", ".join(checked[:-1]) + " and " + checked[-1] + ".")
    elif rest:
        lines.append(f"- The other {rest} clusters are within the normal range.")
    if report.note:
        lines.append(f"- Note: {report.note}")
    return lines, cites
