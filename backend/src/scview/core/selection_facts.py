"""Facts about the user's current cell selection, for the co-pilot.

A lasso, a gate or a cell filter picks cells; the co-pilot is then asked "what
are these cells?" or "how do they differ from the rest?". The frontend sends the
selection in the view context, encoded as session files keep one (``lib/session.ts``
``encodeSelection``: the indices, or a base64 bitmask with cell i at bit i % 8 of
byte i // 8, whichever is shorter). This module turns it into a few cheap facts:

- its size, and its share of the dataset;
- its composition by the colour-by column and the dataset's cluster, cell-type,
  sample and condition columns: the top categories with their share of the
  selection, their share of the dataset and the ratio of the two;
- QC medians (genes and counts per cell, % mito) of the selection vs the rest;
- the genes up (and down) in the selection vs the rest, under the rules of the
  shared marker choice (``marker_selection``: detected in >= 20% of the cells,
  log2 fold change >= 0.25, adjusted p < 0.05, by fold change, no ribosomal or
  mitochondrial genes). The candidates passing the detection and fold-change rules
  get a Wilcoxon rank-sum test (normal approximation, tie-corrected, as scanpy's
  ``wilcoxon`` with ``tie_correct=True``), read off the sparse columns; every other
  gene counts as p = 1 in the Benjamini-Hochberg adjustment over all genes, which
  can only raise the adjusted p of the genes listed (conservative);
- whether a stored DE result (``de_results``) compared exactly these cells with
  the rest.

The expensive part (markers, QC) is cached per dataset and selection hash. A large
dataset (its matrix on disk) gets no markers, with a line saying why.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

from scview.core.gene_filters import is_ribo_mito
from scview.core.marker_selection import MAX_PADJ, MIN_LOGFC, MIN_PTS

logger = logging.getLogger(__name__)

TAG = "result:selection"
N_UP = 15
N_DOWN = 5
MAX_CANDIDATES = 240  # per direction, by fold change, before the rank test
_BATCH = 60  # candidates per direction tested at a time
TOP_CATEGORIES = 4
MAX_COLUMNS = 5
MAX_UNIQUE = 60  # a column with more categories is not a grouping worth listing
_CACHE_KEEP = 16
_MAX_BITMASK_CHARS = 2_000_000  # ~12M cells

# Questions about the selection itself (the router and the API use this).
_REFERS_RE = re.compile(
    r"\b(?:these|those|the selected|my selected|selected)\s+(?:\d[\d,]*\s+)?(?:cells?|ones|points|barcodes)\b"
    r"|\b(?:this|my|the current|current)\s+(?:selection|lasso|gate|subset)\b"
    r"|\b(?:selection|selected|lasso(?:ed)?|gated)\b",
    re.I,
)

_cache: OrderedDict[tuple, dict] = OrderedDict()
_lock = threading.Lock()


def refers_to_selection(query: str) -> bool:
    """True when a question is about the selected cells ("these cells", "my lasso")."""
    return bool(_REFERS_RE.search(query or ""))


def present(view_context: dict | None) -> bool:
    """Whether the view context carries a cell selection."""
    sel = (view_context or {}).get("selection")
    return (isinstance(sel, dict) and isinstance(sel.get("n_cells"), int) and sel["n_cells"] > 0
            and isinstance(sel.get("cells"), dict))


def wanted(query: str, view_context: dict | None) -> bool:
    """Whether to put the selection's facts in the context: the question is about
    it, or the user has just made it (``fresh``). Otherwise the call stays small."""
    if not present(view_context):
        return False
    return refers_to_selection(query) or bool(view_context["selection"].get("fresh"))


def build(adaptor: Any, view_context: dict) -> tuple[str, list]:
    """The selection's context block and its citable source, for the co-pilot."""
    from scview.core.assistant import ChatSource

    sel = view_context.get("selection") or {}
    lines, detail = grounding_lines(adaptor, sel, color_by=view_context.get("color_by"))
    if not lines:
        return "", []
    source = ChatSource(kind="result", ref=TAG, detail=detail or "the selection could not be used",
                        evidence="\n".join(lines[1:]))
    return "\n".join(lines).strip(), [source]


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------


def decode_cells(cells: Any, n_obs: int) -> np.ndarray:
    """The selected cell indices (sorted, unique) from ``{n_cells, indices}`` or
    ``{n_cells, bitmask}``. ValueError when the payload is malformed or was made on
    another number of cells (the dataset was filtered since)."""
    if not isinstance(cells, dict):
        raise ValueError("The selection is not in a known form.")
    n_total = cells.get("n_cells")
    if not isinstance(n_total, int) or n_total != n_obs:
        raise ValueError(f"The selection was made on {n_total} cells; the dataset now has {n_obs:,}.")
    if "indices" in cells:
        raw = cells["indices"]
        if not isinstance(raw, list):
            raise ValueError("The selection's indices are not a list.")
        idx = np.asarray([i for i in raw if isinstance(i, int) and not isinstance(i, bool)], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_obs)]
        return np.unique(idx)
    mask_text = cells.get("bitmask")
    if not isinstance(mask_text, str) or len(mask_text) > _MAX_BITMASK_CHARS:
        raise ValueError("The selection has neither indices nor a bitmask.")
    try:
        raw_bytes = base64.b64decode(mask_text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The selection's bitmask is not base64.") from exc
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8), bitorder="little")[:n_obs]
    return np.flatnonzero(bits).astype(np.int64)


def selection_hash(idx: np.ndarray, n_obs: int) -> str:
    """A short hash of a selection (its cells among ``n_obs``): the cache key, and
    how a stored DE result names the cells it compared."""
    h = hashlib.sha256(f"{int(n_obs)}\n".encode())
    h.update(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes())
    return h.hexdigest()[:16]


def describe_source(source: Any) -> str:
    """How the cells were picked, for the context line."""
    s = str(source or "")
    if s.startswith("gate:") and len(s) > 5:
        return f"saved gate '{s[5:][:80]}'"
    return {"lasso": "lasso", "filter": "cell filter", "group": "legend/group click"}.get(s, "selection")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

_CLUSTER_RE = re.compile(r"leiden|louvain|cluster|seurat_clusters", re.I)
_LABEL_RE = re.compile(r"cell_?type|celltypeanno|annotation|predicted_labels|majority_voting|^labels?$", re.I)
_SAMPLE_RE = re.compile(r"donor|patient|individual|^ind$|subject|participant|replicate|sample|orig[._]ident", re.I)
_CONDITION_RE = re.compile(r"condition|stim|treat|disease|status|genotype|perturb|timepoint|^time$|group", re.I)


def _usable(obs: pd.DataFrame, col: str) -> bool:
    s = obs[col]
    if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool):
        return False
    n = s.nunique(dropna=True)
    return 1 < n <= MAX_UNIQUE


def composition_columns(obs: pd.DataFrame, color_by: str | None = None) -> list[str]:
    """The colour-by column, then the cluster, cell-type, sample and condition
    columns (the first matching name of each), at most ``MAX_COLUMNS``."""
    out: list[str] = []
    if color_by and color_by in obs.columns and _usable(obs, color_by):
        out.append(color_by)
    for rx in (_CLUSTER_RE, _LABEL_RE, _SAMPLE_RE, _CONDITION_RE):
        for col in obs.columns:
            name = str(col)
            if name in out or name.startswith("_") or not rx.search(name):
                continue
            if _usable(obs, name):
                out.append(name)
                break
    return out[:MAX_COLUMNS]


def composition(obs: pd.DataFrame, mask: np.ndarray, column: str, top: int = TOP_CATEGORIES) -> list[dict]:
    """The selection's largest categories in ``column``: cells, % of the selection,
    % of the dataset and their ratio (> 1 = more common in the selection)."""
    values = obs[column].astype(object).where(obs[column].notna(), "NA").astype(str).to_numpy()
    n_sel = int(mask.sum())
    n_all = values.size
    if n_sel == 0 or n_all == 0:
        return []
    sel_counts = pd.Series(values[mask]).value_counts()
    all_counts = pd.Series(values).value_counts()
    rows = []
    for cat, n in sel_counts.items():
        pct_sel = 100.0 * n / n_sel
        pct_all = 100.0 * all_counts[cat] / n_all
        rows.append({"category": str(cat), "n": int(n), "pct_selection": pct_sel,
                     "pct_dataset": pct_all, "ratio": pct_sel / pct_all if pct_all else float("nan")})
    rows.sort(key=lambda r: (-r["n"], r["category"]))
    return rows[:top]


# ---------------------------------------------------------------------------
# QC medians
# ---------------------------------------------------------------------------


def _qc_arrays(adaptor: Any, adata: Any, large: bool) -> dict[str, np.ndarray | None]:
    from scview.core.anndata_adaptor import _QC_OBS_NAMES, per_cell_qc

    if large or adaptor is None:
        # From obs only on a large dataset (computing them reads the counts sidecar).
        out: dict[str, np.ndarray | None] = {}
        for key in ("total_counts", "n_genes", "pct_mt"):
            col = next((c for c in _QC_OBS_NAMES[key] if c in adata.obs.columns), None)
            out[key] = None if col is None else pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(float)
        if large or all(v is not None for v in out.values()):
            return out
        qc = per_cell_qc(adata)
    else:
        qc = adaptor.qc_per_cell()
    has_mt = qc.get("has_mt", True)
    return {"total_counts": qc.get("total_counts"), "n_genes": qc.get("n_genes"),
            "pct_mt": qc.get("pct_mt") if has_mt is not False else None}


def qc_medians(qc: dict[str, np.ndarray | None], mask: np.ndarray) -> dict[str, tuple[float, float]]:
    """Median of each QC metric in the selection and in the rest."""
    out: dict[str, tuple[float, float]] = {}
    for key, arr in qc.items():
        if arr is None or len(arr) != mask.size or mask.all() or not mask.any():
            continue
        a = np.asarray(arr, dtype=float)
        sel, rest = a[mask], a[~mask]
        if np.isfinite(sel).any() and np.isfinite(rest).any():
            out[key] = (float(np.nanmedian(sel)), float(np.nanmedian(rest)))
    return out


# ---------------------------------------------------------------------------
# Marker genes: selection vs rest
# ---------------------------------------------------------------------------


def _log_matrix(adata: Any):
    """The log-normalised matrix and its gene names: ``.raw`` when it holds log
    values, else X (normalised + log1p here when X holds counts), as DE does."""
    from scview.core.pipeline import looks_like_counts

    use_raw = adata.raw is not None and not looks_like_counts(adata.raw.X)
    M = adata.raw.X if use_raw else adata.X
    names = adata.raw.var_names if use_raw else adata.var_names
    M = sp.csr_matrix(M) if not sp.issparse(M) else M.tocsr()
    if looks_like_counts(M):
        totals = np.asarray(M.sum(axis=1)).ravel()
        scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
        M = sp.diags(scale) @ M
        M = M.tocsr()
        M.data = np.log1p(M.data)
    return M, np.asarray(names).astype(str)


def _col_stats(M: sp.csr_matrix, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Per gene: sum and number of cells with a value > 0 (over ``rows``, or all)."""
    sub = M if rows is None else M[rows]
    sums = np.asarray(sub.sum(axis=0)).ravel()
    pos = sub.data > 0
    nnz = np.bincount(sub.indices[pos], minlength=M.shape[1])
    return sums, nnz, sub.shape[0]


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def _ranksum_z(col: sp.csc_matrix, j: int, in_sel: np.ndarray, n1: int) -> float:
    """Wilcoxon rank-sum z of the selection for column ``j`` (tie-corrected, no
    continuity correction), from its stored values; the zeros share one tied rank."""
    n = in_sel.size
    n2 = n - n1
    lo, hi = col.indptr[j], col.indptr[j + 1]
    vals = col.data[lo:hi]
    rows = col.indices[lo:hi]
    if vals.size and vals.min() < 0:  # not a log matrix: rank every cell
        dense = np.zeros(n)
        dense[rows] = vals
        ranks = rankdata(dense)
        r1 = float(ranks[in_sel].sum())
        _, counts = np.unique(dense, return_counts=True)
        ties = float(((counts.astype(float) ** 3) - counts).sum())
    else:
        keep = vals != 0
        vals, rows = vals[keep], rows[keep]
        t0 = n - vals.size
        ranks = rankdata(vals) + t0 if vals.size else np.zeros(0)
        sel_nz = in_sel[rows]
        r1 = float(ranks[sel_nz].sum()) + (n1 - int(sel_nz.sum())) * (t0 + 1) / 2.0
        _, counts = np.unique(vals, return_counts=True) if vals.size else (None, np.zeros(0))
        ties = float(((counts.astype(float) ** 3) - counts).sum()) + float(t0) ** 3 - t0
    u = r1 - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 0.0
    return (u - n1 * n2 / 2.0) / np.sqrt(var)


def _ranksum_z_many(M: sp.csr_matrix, cand: np.ndarray, in_sel: np.ndarray, n1: int) -> np.ndarray:
    """``_ranksum_z`` for every column in ``cand`` at once, for a matrix without
    negative values: one sort of the candidates' stored values by (column, value)."""
    n = in_sel.size
    n2 = n - n1
    k = cand.size
    pos = np.full(M.shape[1], -1, dtype=np.int64)
    pos[cand] = np.arange(k)
    rows_all = np.repeat(np.arange(n, dtype=np.int64), np.diff(M.indptr))
    keep = (pos[M.indices] >= 0) & (M.data != 0)
    c = pos[M.indices[keep]]
    v = M.data[keep]
    r = rows_all[keep]
    # One float key orders by column, then value (values >= 0; float32 values stay
    # distinct in float64 under the offset); a plain sort beats lexsort here.
    v = v.astype(np.float64)
    order = np.argsort(c * (float(v.max(initial=0.0)) * 2.0 + 1.0) + v)
    c, v, r = c[order], v[order], r[order]
    nnz = np.bincount(c, minlength=k)
    t0 = n - nnz
    col_start = np.concatenate([[0], np.cumsum(nnz)[:-1]])
    # tie groups: runs of equal (column, value)
    new = np.ones(c.size, dtype=bool)
    if c.size > 1:
        new[1:] = (c[1:] != c[:-1]) | (v[1:] != v[:-1])
    gid = np.cumsum(new) - 1
    g_start = np.flatnonzero(new)
    g_size = np.diff(np.append(g_start, c.size))
    # average rank within the column's stored values (1-based), after its zeros
    within = g_start - col_start[c[g_start]]
    g_rank = within + (g_size + 1) / 2.0
    ranks = g_rank[gid] + t0[c]
    sel = in_sel[r]
    r1 = np.bincount(c[sel], weights=ranks[sel], minlength=k)
    nz_sel = np.bincount(c[sel], minlength=k)
    r1 = r1 + (n1 - nz_sel) * (t0 + 1) / 2.0
    gs = g_size.astype(float)
    ties = np.bincount(c[g_start], weights=gs ** 3 - gs, minlength=k) + t0.astype(float) ** 3 - t0
    u = r1 - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, (u - n1 * n2 / 2.0) / np.sqrt(np.maximum(var, 1e-300)), 0.0)
    return z


def selection_markers(adata: Any, idx: np.ndarray, *, n_up: int = N_UP, n_down: int = N_DOWN) -> dict:
    """Genes up and down in the selected cells vs the rest (module docstring)."""
    M, names = _log_matrix(adata)
    n = M.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    n1, n2 = int(mask.sum()), int(n - mask.sum())
    if n1 < 3 or n2 < 3:
        return {"up": [], "down": [], "n_tested": 0, "reason": "too few cells on one side"}
    tot_sum, tot_nnz, _ = _col_stats(M)
    sel_sum, sel_nnz, _ = _col_stats(M, np.flatnonzero(mask))
    mean_s, mean_r = sel_sum / n1, (tot_sum - sel_sum) / n2
    pts_s, pts_r = sel_nnz / n1, (tot_nnz - sel_nnz) / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        lfc = np.log2((np.expm1(mean_s) + 1e-9) / (np.expm1(mean_r) + 1e-9))
    lfc = np.nan_to_num(lfc, nan=0.0)
    usable = np.fromiter((not is_ribo_mito(g) for g in names), dtype=bool, count=names.size)

    up = np.flatnonzero(usable & (pts_s >= MIN_PTS) & (lfc >= MIN_LOGFC))
    down = np.flatnonzero(usable & (pts_r >= MIN_PTS) & (lfc <= -MIN_LOGFC))
    up = up[np.argsort(-lfc[up], kind="stable")][:MAX_CANDIDATES]
    down = down[np.argsort(lfc[down], kind="stable")][:MAX_CANDIDATES]
    p = np.ones(names.size)
    nonneg = M.data.size == 0 or M.data.min() >= 0

    def test(cand: np.ndarray) -> None:
        if not cand.size:
            return
        if nonneg:
            p[cand] = 2.0 * norm.sf(np.abs(_ranksum_z_many(M, cand, mask, n1)))
            return
        col = M[:, cand].tocsc()  # negative values (not a log matrix): rank every cell
        for k, j in enumerate(cand):
            p[j] = float(2.0 * norm.sf(abs(_ranksum_z(col, k, mask, n1))))

    # Tested in batches by fold change until enough pass: testing more genes can only
    # lower the adjusted p of the ones tested, and an untested gene has a smaller fold
    # change than every tested one, so it could not enter the lists.
    done = {"up": 0, "down": 0}
    want = {"up": n_up, "down": n_down}
    lists = {"up": up, "down": down}
    padj = _bh(p)
    while True:
        batch = []
        for d in ("up", "down"):
            tested = lists[d][: done[d]]
            if done[d] < lists[d].size and int((padj[tested] < MAX_PADJ).sum()) < want[d]:
                batch.append(lists[d][done[d]: done[d] + _BATCH])
                done[d] = min(done[d] + _BATCH, lists[d].size)
        if not batch:
            break
        test(np.concatenate(batch))
        padj = _bh(p)
    up, down = up[: done["up"]], down[: done["down"]]
    cand = np.concatenate([up, down])

    def rows(genes: np.ndarray, limit: int) -> list[dict]:
        out = []
        for j in genes:
            if padj[j] < MAX_PADJ:
                out.append({"gene": str(names[j]), "log2fc": float(lfc[j]), "pct_selection": 100 * float(pts_s[j]),
                            "pct_rest": 100 * float(pts_r[j]), "padj": float(padj[j])})
            if len(out) >= limit:
                break
        return out

    return {"up": rows(up, n_up), "down": rows(down, n_down), "n_tested": int(cand.size),
            "n_genes": int(names.size), "reason": None}


# ---------------------------------------------------------------------------
# Stored DE for exactly these cells
# ---------------------------------------------------------------------------


def stored_de(dataset_id: str | None, sel_hash: str) -> dict | None:
    """The newest kept DE result that compared exactly these cells with the rest."""
    if not dataset_id:
        return None
    from scview.core import de_results

    try:
        kept = de_results.list_results(dataset_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("selection facts: DE results unavailable: %s", exc)
        return None
    for rid, r in kept:
        c = r.comparison or {}
        if c.get("mode") == "selection" and c.get("cells") == sel_hash:
            n_sig = None
            if r.pvals_adj is not None:
                n_sig = int(np.sum(np.nan_to_num(r.pvals_adj, nan=1.0) < 0.05))
            return {"id": rid, "label": r.label, "n_sig": n_sig, "n_tested": len(r.genes)}
    return None


# ---------------------------------------------------------------------------
# Facts -> context lines
# ---------------------------------------------------------------------------


def _cached(key: tuple, build) -> dict:
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    value = build()
    with _lock:
        _cache[key] = value
        while len(_cache) > _CACHE_KEEP:
            _cache.popitem(last=False)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def cache_size() -> int:
    with _lock:
        return len(_cache)


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def facts(adaptor: Any, cells: Any, *, dataset_id: str | None = None) -> dict:
    """The selection's facts (``cells`` as the view context sends them). The markers
    and QC medians are cached per dataset and selection."""
    from scview.core import large_mode

    adata = adaptor.adata if hasattr(adaptor, "adata") else adaptor
    real_adaptor = adaptor if hasattr(adaptor, "adata") else None
    n_obs = int(adata.n_obs)
    idx = decode_cells(cells, n_obs)
    sel_hash = selection_hash(idx, n_obs)
    mask = np.zeros(n_obs, dtype=bool)
    mask[idx] = True
    large = bool(real_adaptor is not None and large_mode.is_large(real_adaptor))

    def build() -> dict:
        out: dict[str, Any] = {}
        try:
            out["qc"] = qc_medians(_qc_arrays(real_adaptor, adata, large), mask)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("selection facts: QC failed: %s", exc)
            out["qc"] = {}
        if large:
            out["markers"] = {"up": [], "down": [], "reason": "large dataset: its expression matrix stays on disk, "
                              "so the selection's genes are not ranked here"}
        elif getattr(adata, "isbacked", False):
            out["markers"] = {"up": [], "down": [], "reason": "the expression matrix is on disk"}
        elif idx.size < 3 or n_obs - idx.size < 3:
            out["markers"] = {"up": [], "down": [], "reason": "too few cells on one side"}
        else:
            out["markers"] = selection_markers(adata, idx)
        return out

    key = (dataset_id or "", id(adata), n_obs, int(adata.n_vars), sel_hash)
    heavy = _cached(key, build)
    return {"n": int(idx.size), "n_total": n_obs, "hash": sel_hash, "mask": mask, **heavy}


def grounding_lines(
    adaptor: Any, selection: dict, *, color_by: str | None = None, dataset_id: str | None = None,
) -> tuple[list[str], str | None]:
    """Context lines for the co-pilot about the selected cells, and a one-line detail
    for the source list; ``([], None)`` when the selection cannot be read."""
    adata = adaptor.adata if hasattr(adaptor, "adata") else adaptor
    if dataset_id is None:
        try:
            from scview.core import results_store
            dataset_id = results_store.dataset_id_of(adaptor)
        except Exception:  # pragma: no cover - defensive
            dataset_id = None
    try:
        f = facts(adaptor, selection.get("cells"), dataset_id=dataset_id)
    except ValueError as exc:
        return [f"\n## Selected cells [{TAG}]", f"- The selection could not be used: {exc}"], None
    n, n_total, mask = f["n"], f["n_total"], f["mask"]
    if n == 0:
        return [], None
    how = describe_source(selection.get("source"))
    share = 100.0 * n / n_total
    detail = f"{n:,} selected cells ({how}) = {share:.1f}% of {n_total:,}"
    lines = [f"\n## Selected cells (the user's current {how}; 'these cells') [{TAG}]",
             f"- {n:,} cells selected, {share:.1f}% of the dataset's {n_total:,} cells."]

    # One line per category, so the claims check ties each share to its group.
    obs = adata.obs
    for col in composition_columns(obs, color_by):
        rows = composition(obs, mask, col)
        if not rows:
            continue
        lines.append(f"- By {col}, the selection's largest groups (share of the selection vs of the dataset):")
        for r in rows:
            lines.append(f"  - {r['category']} ({col}): {r['n']:,} cells, {_fmt_pct(r['pct_selection'])} of the "
                         f"selection vs {_fmt_pct(r['pct_dataset'])} of the dataset ({r['ratio']:.1f}x)")

    qc = f.get("qc") or {}
    names = {"n_genes": ("genes/cell", "{:,.0f}"), "total_counts": ("counts/cell", "{:,.0f}"),
             "pct_mt": ("% mito", "{:.1f}")}
    qc_parts = [f"{label} {fmt.format(qc[k][0])} vs {fmt.format(qc[k][1])}"
                for k, (label, fmt) in names.items() if k in qc]
    if qc_parts:
        lines.append("- QC medians, selection vs rest: " + "; ".join(qc_parts) + ".")

    mk = f.get("markers") or {}
    if mk.get("reason"):
        lines.append(f"- Genes of the selection not ranked: {mk['reason']}.")
    else:
        rule = (f"Wilcoxon vs the rest; in >= {MIN_PTS:.0%} of the cells, log2FC >= {MIN_LOGFC:g}, "
                f"adj. p < {MAX_PADJ:g}; by fold change; ribosomal/mito left out")
        # Fold changes only: detection shares here would match unrelated % claims.
        if mk.get("up"):
            ups = ", ".join(f"{g['gene']} (log2FC {g['log2fc']:.2f})" for g in mk["up"])
            lines.append(f"- Genes up in the selection ({rule}): {ups}.")
        else:
            lines.append(f"- No gene passes the marker rules for the selection ({rule}).")
        if mk.get("down"):
            downs = ", ".join(f"{g['gene']} (log2FC {g['log2fc']:.2f})" for g in mk["down"])
            lines.append(f"- Genes lower in the selection than in the rest: {downs}.")

    de = stored_de(dataset_id, f["hash"])
    if de:
        sig = f"; {de['n_sig']:,} of {de['n_tested']:,} genes at adj. p < 0.05" if de["n_sig"] is not None else ""
        lines.append(f"- A stored DE result compared exactly these cells with the rest: {de['label']}{sig}.")
    else:
        lines.append("- No DE result is stored for exactly these cells; the Unified View's DE tab "
                     "runs selection vs rest (the user clicks Run).")
    lines.append("- Caveat: the selection is the user's choice, not a cluster, so its genes describe "
                 "the cells picked, not a tested population.")
    return lines, detail
