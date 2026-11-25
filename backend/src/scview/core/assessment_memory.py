"""What Data Assessment's requests add to memory at their peak, before they run.

Each heavy request the page sends is held in the memory budget while it runs
(``assessment_gate.Budget``): these are their estimates, from the dataset's shape and stored
values, 0 for work a cache already holds. Measured on a deep 23,578 x 34,209 file (5,650 values
a cell, CSC counts as the R converter writes them) and on Kang (13,836 x 14,053): see each
constant. Each estimate is at or above the measured peak RSS growth: a request is let in only
when all of it fits (``working_copy.reading``).

``ROUTES`` names every route the page sends and how it is held; a test calls each one and
checks it went through the budget, so a new route cannot skip it.
"""

from __future__ import annotations

import logging

import numpy as np

from scview.core.anndata_adaptor import ROW_BLOCK_VALUES

logger = logging.getLogger(__name__)

MiB = 1024**2

# Every Data Assessment route that computes over the matrix, held through ``assessment_gate``
# with this label as its refusal's subject (tests/test_assessment_memory.py calls each one).
ROUTES: dict[str, str] = {
    "/datasets/{dataset_id}/qc": "QC metrics",
    "/datasets/{dataset_id}/qc/ambient": "The ambient RNA estimate",
    "/datasets/{dataset_id}/cluster-qc": "Cluster QC",
    "/datasets/{dataset_id}/clusters/evidence": "Cluster evidence",
    "/datasets/{dataset_id}/qc/by-group": "QC by group",
    "/datasets/{dataset_id}/qc/sex": "The sex check",
    "/datasets/{dataset_id}/qc/libraries": "Library concordance",
    "/datasets/{dataset_id}/qc/pseudobulk": "Pseudobulk sample QC",
    "/datasets/{dataset_id}/qc/filter-preview": "The filter preview",
    "/datasets/{dataset_id}/qc/doublets": "The doublet summary",
    "/datasets/{dataset_id}/qc-report": "The QC report",
    "/datasets/{dataset_id}/qc/mito-by-type": "Mito by cell type",
    "/datasets/{dataset_id}/qc/states": "QC states",
    "/datasets/{dataset_id}/qc/mean-variance": "The mean-variance plot",
}
# The page's other GET routes in api/v1/assessment.py, and how each is held (or why not).
HELD_OTHERWISE: dict[str, str] = {
    "/datasets/{dataset_id}/assessment": "working_copy.reading with assessor.assessment_need",
    "/datasets/{dataset_id}/qc/mito-refilter": "working_copy.reading with its own estimate",
    "/datasets/{dataset_id}/rerun-plan": "provenance only",
}

# The per-cell QC pass (``anndata_adaptor._row_sums``): one block of about ROW_BLOCK_VALUES
# stored values at a time, copied, compared and column-sliced. Deep file, row blocks of a CSC:
# +215 MB; CSC spans: +XXX MB.
PER_CELL_BYTES_PER_BLOCK_VALUE = 28
PER_CELL_BYTES_PER_CELL = 128  # the per-cell arrays and their pieces
FIXED = 32 * MiB  # imports, small frames, the answer itself

# Ambient RNA (``ambient.diagnose``): per channel sums (k x genes, dense) and, with a soup from
# empty droplets, a CSR copy of the counts, the expressed-above-cut copy and the sample's rows.
AMBIENT_DENSE_COPIES = 4
AMBIENT_SOUP_COPIES = 3

# Mean-variance (``normalization.mean_variance``): blocks of MV_BLOCK_VALUES stored values (a
# slice, its rows, its square and its shifted log) and residual blocks of MV_RESIDUAL_VALUES.
MV_BYTES_PER_BLOCK_VALUE = 48
MV_DENSE_COPIES = 6

# Sample pseudobulk QC: per sample sums (dense) and the rows of one cell type when asked.
PB_DENSE_COPIES = 4


def _values(M) -> tuple[int, int]:
    """(stored values, bytes per value with its index) of a matrix, in memory or on disk."""
    nnz = getattr(M, "nnz", None)
    item = int(getattr(getattr(M, "dtype", None), "itemsize", 8) or 8)
    if isinstance(nnz, (int, np.integer)):
        return int(nnz), item + 4
    n, g = (int(x) for x in M.shape)
    return n * g, item


def _per_cell_pending(adaptor, ribo: bool) -> bool:
    """Whether ``qc_per_cell(ribo)`` would pass over the counts: not cached, and a core metric
    is missing from obs (or ``ribo`` asks for the extras of an in-memory matrix)."""
    from scview.core.anndata_adaptor import _QC_OBS_NAMES

    cached = getattr(adaptor, "qc_cached", None)
    if callable(cached) and cached(ribo):
        return False
    adata = adaptor.adata
    obs = adata.obs.columns

    def missing(key: str) -> bool:
        return not any(name in obs for name in _QC_OBS_NAMES[key])

    if any(missing(k) for k in ("total_counts", "n_genes", "pct_mt")):
        return True
    return bool(ribo and not getattr(adata, "isbacked", False)
                and any(missing(k) for k in ("pct_ribo", "pct_hb", "pct_malat1")))


def per_cell_need(adaptor, ribo: bool = False) -> int:
    """The per-cell QC pass (``AnnDataAdaptor.qc_per_cell``); 0 when it is cached or obs
    holds what it would compute."""
    if not _per_cell_pending(adaptor, ribo):
        return 0
    n = int(adaptor.adata.n_obs or 0)
    return int(ROW_BLOCK_VALUES * PER_CELL_BYTES_PER_BLOCK_VALUE + n * PER_CELL_BYTES_PER_CELL + FIXED)


def ambient_need(adata, column: str | None) -> int:
    """``ambient.diagnose_cached``; 0 when its answer is cached."""
    from scview.core import ambient, cluster_qc, doublets

    try:
        if ambient.is_cached(adata, column):
            return 0
        counts, _ = ambient.counts_matrix(adata)
    except Exception:  # no counts: it answers "not applicable" at once
        return FIXED
    n, g = (int(x) for x in counts.shape)
    try:
        k = max(1, len(doublets.channels(adata).names))
    except Exception:  # noqa: BLE001 - one channel
        k = 1
    col = column or cluster_qc.default_grouping(adata)
    if col and col in adata.obs.columns:
        k = max(k, int(adata.obs[col].nunique()))
    need = k * g * 8 * AMBIENT_DENSE_COPIES + n * 32 + FIXED
    if ambient.read_soups(adata):
        values, per = _values(counts)
        need += AMBIENT_SOUP_COPIES * values * per
    return int(need)


def cluster_qc_need(adaptor, column: str | None) -> int:
    """``cluster_qc.for_adaptor``: the per-cell pass with the extras, and the ambient flags
    of the grouping's markers (``ambient.flagged_markers``)."""
    from scview.core import cluster_qc

    adata = adaptor.adata
    col = column or cluster_qc.default_grouping(adata)
    if not col or col not in adata.obs.columns:
        return 0  # no grouping: answered from obs
    return per_cell_need(adaptor, ribo=True) + ambient_need(adata, col)


def evidence_need(adaptor, column: str | None) -> int:
    """``cluster_evidence.for_adaptor``: cluster QC's, 0 when its report is cached."""
    from scview.core import cluster_evidence

    try:
        if cluster_evidence.cached(adaptor, column or None) is not None:
            return 0
    except Exception:  # noqa: BLE001 - estimate it
        pass
    return cluster_qc_need(adaptor, column)


def qc_report_need(adaptor) -> int:
    """The QC report's facts (``qc_report.build_facts``): the per-cell pass with the extras and
    an assessment."""
    from scview.core.assessor import assessment_need

    return per_cell_need(adaptor, ribo=True) + assessment_need(adaptor.adata)


def mean_variance_need(adata) -> int:
    """``normalization.mean_variance`` on the counts (sampled cells, blocks)."""
    from scview.core import normalization

    try:
        counts, _, _ = normalization.counts_all_genes(adata, read_only=True)
    except Exception:  # no counts: a 409 at once
        return FIXED
    n, g = (int(x) for x in counts.shape)
    ns = min(n, normalization.DIAGNOSTIC_MAX_CELLS)
    return int(normalization.MV_BLOCK_VALUES * MV_BYTES_PER_BLOCK_VALUE
               + normalization.MV_RESIDUAL_VALUES * 8 * MV_DENSE_COPIES
               + g * 8 * 8 + ns * 8 * 4 + FIXED)


def pseudobulk_need(adaptor, celltype_column: str | None, celltype: str | None) -> int:
    """Sample pseudobulk QC: per-sample sums, the sex check's genes, and one cell type's rows
    when ``celltype`` limits it."""
    from scview.core.pseudobulk import raw_counts

    adata = adaptor.adata
    try:
        M, _, _ = raw_counts(adata)
    except Exception:  # no counts: answered with the reason
        return FIXED
    n, g = (int(x) for x in M.shape)
    need = 0
    if celltype and celltype_column and celltype_column in adata.obs.columns:
        share = float((adata.obs[celltype_column].astype(str) == str(celltype)).mean())
        values, per = _values(M)
        need += int(share * values * per * 2)
    from scview.core import qc_groups

    k = 1
    try:
        col = qc_groups.sample_grouping(adata)[0]
        if col:
            k = max(1, int(adata.obs[col].nunique()))
    except Exception:  # noqa: BLE001 - one sample
        pass
    k = max(k, 2) * 4  # samples split by a condition
    return int(need + k * g * 8 * PB_DENSE_COPIES + n * 32
               + ROW_BLOCK_VALUES * PER_CELL_BYTES_PER_BLOCK_VALUE + FIXED)


def sex_need(adata) -> int:
    """The sex check: a few gene columns and every cell's total (measured +1 MB on the deep
    file)."""
    return int(int(adata.n_obs or 0) * 64 + FIXED)


# Placing the cells filtering removed (``mito_refilter.assigned_removed``): their counts read
# from the original upload and log-normalised, and a log-normalised copy of the kept cells'
# counts when the dataset holds no log values.
REFILTER_COPIES = 3


def mito_by_type_need(dm, dataset_id: str, adaptor) -> int:
    """Mito by cell type: the per-cell pass, and after filtering the removed cells placed by
    expression (0 once placed for this file)."""
    from scview.core import mito_refilter, qc_filter

    adata = adaptor.adata
    need = per_cell_need(adaptor)
    names = qc_filter.recorded_names(adata)
    if not names or mito_refilter.assigned_cached(dataset_id, adaptor):
        return need
    if mito_refilter.source_file(dm, dataset_id) is None:
        return need
    n = max(1, int(adata.n_obs or 0))
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    values, per = _values(counts)
    per_cell = values / n
    need += int(len(names) * per_cell * per * REFILTER_COPIES)
    try:
        holds_log = mito_refilter.holds_log(adata)
    except Exception:  # noqa: BLE001 - count the copy
        holds_log = False
    if not holds_log:
        need += int(values * per * REFILTER_COPIES)
    return need
