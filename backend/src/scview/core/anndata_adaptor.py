"""AnnDataAdaptor – read-only interface to an h5ad dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import category_order, expression_range, haemoglobin, marker_specificity, protein
from scview.core.ora_background import resolve_markers_key
from scview.core.species import detect_species
from scview.core.uns_keys import markers_column, repair_slash_keys
from scview.core.user_labels import user_columns

# Obs columns taken as the clustering when none is recorded (a file processed elsewhere).
CLUSTERING_CANDIDATES = (
    "leiden", "louvain", "cluster", "clusters", "seurat_clusters", "cell_type",
)

logger = logging.getLogger(__name__)


def _dense_rows(block) -> np.ndarray:
    return block.toarray() if hasattr(block, "toarray") else np.asarray(block)


def _looks_like_counts(M) -> bool:
    """Non-negative integers in the first rows: raw counts, not log or scaled values."""
    sample = _dense_rows(M[: min(200, M.shape[0])])
    return sample.size == 0 or (sample.min() >= 0 and np.allclose(sample, np.round(sample)))


def _counts_matrix(adata):
    """The best available counts matrix and its gene names: layers['counts'], .raw, X."""
    candidates = []
    if "counts" in adata.layers:
        candidates.append((adata.layers["counts"], adata.var_names))
    if adata.raw is not None:
        candidates.append((adata.raw.X, adata.raw.var_names))
    candidates.append((adata.X, adata.var_names))
    for M, names in candidates:
        try:
            if _looks_like_counts(M):
                return M, names
        except Exception:  # an unreadable layer: try the next one
            continue
    return adata.X, adata.var_names


# obsm entries named like embeddings that are not coordinates: gene-program usages
# (core/gene_programs.py), one column per program.
NOT_EMBEDDINGS = ("X_nmf_usage", "X_nmf_usage_raw")


def _row_sums(M, masks: list[np.ndarray], chunk: int = 20000):
    """Per-cell totals, detected genes and one sum per gene mask, read in row
    chunks so a backed (on-disk) matrix, which has no .sum, works too."""
    tc, ng = [], []
    sums: list[list[np.ndarray]] = [[] for _ in masks]
    for i in range(0, M.shape[0], chunk):
        block = M[i:i + chunk]
        if not hasattr(block, "toarray"):
            block = np.asarray(block)
        tc.append(np.asarray(block.sum(axis=1)).ravel())
        ng.append(np.asarray((block > 0).sum(axis=1)).ravel())
        for parts, mask in zip(sums, masks):
            if mask.any():
                parts.append(np.asarray(block[:, mask].sum(axis=1)).ravel())
    cat = lambda parts: np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)  # noqa: E731
    total = cat(tc)
    return total, cat(ng), [cat(p) if p else np.zeros_like(total) for p in sums]


def _row_stats(M, mt_mask: np.ndarray, chunk: int = 20000):
    """Per-cell totals, detected genes and mito counts (see ``_row_sums``)."""
    total, detected, (mt,) = _row_sums(M, [mt_mask], chunk)
    return total, detected, mt


# obs names for the same per-cell metric: scanpy, older scanpy / Kang, Seurat.
_QC_OBS_NAMES = {
    "total_counts": ("total_counts", "n_counts", "nCount_RNA"),
    "n_genes": ("n_genes_by_counts", "n_genes", "nFeature_RNA"),
    "pct_mt": ("pct_counts_mt", "percent.mt", "pct_counts_mito", "percent_mito"),
    "pct_ribo": ("pct_counts_ribo", "percent.ribo", "percent.rb", "pct_counts_rb"),
    "pct_hb": ("pct_counts_hb", "percent.hb", "percent.HB", "pct_counts_hemo"),
    "pct_malat1": ("pct_counts_malat1", "percent.malat1", "percent.MALAT1", "pct_malat1"),
}

# MALAT1 is retained in the nucleus: a cell with little of it may be cytoplasm
# without its nucleus (core/cluster_qc.py ``low_malat1``).
MALAT1_NAMES = ("MALAT1",)  # matched case-insensitively (mouse Malat1)


def malat1_mask(var_names: pd.Index) -> np.ndarray:
    """The MALAT1 / Malat1 column(s) of a gene index."""
    return np.asarray(pd.Index(var_names).str.upper().isin(MALAT1_NAMES))


def per_cell_qc(adata, ribo: bool = False) -> dict:
    """Per-cell QC arrays: ``total_counts``, ``n_genes``, ``pct_mt`` and, when they
    can be had cheaply, ``pct_ribo``, ``pct_hb`` (haemoglobin genes for the
    dataset's species, ``core/haemoglobin.py``) and ``pct_malat1`` (share of
    counts from MALAT1); each None when not worked out or the genes are absent.

    Read from ``obs`` when present, otherwise computed from the counts matrix in
    one chunked pass (no mutation of the AnnData, no disk write). ``computed`` is
    True when one of the three core metrics had to be computed; ``has_mt`` is
    False when it was computed and no MT- genes exist (pct_mt is then all zero).
    """
    obs = adata.obs

    def _first(names: tuple[str, ...]) -> np.ndarray | None:
        for n in names:
            if n in obs.columns:
                try:
                    return np.asarray(obs[n].to_numpy(), dtype=np.float64)
                except (TypeError, ValueError):
                    continue
        return None

    out = {k: _first(names) for k, names in _QC_OBS_NAMES.items()}
    need = [k for k in ("total_counts", "n_genes", "pct_mt") if out[k] is None]
    # Ribosomal, haemoglobin and MALAT1 % are a bonus: worked out in the pass the
    # core metrics need, or, when ``ribo`` asks for them, in a pass of their own if the
    # matrix is in memory (one sparse pass is cheap; a backed file would be read
    # again from disk).
    extra_ok = bool(need) or (ribo and not adata.isbacked)
    want_ribo = out["pct_ribo"] is None and extra_ok
    want_hb = out["pct_hb"] is None and extra_ok
    want_malat1 = out["pct_malat1"] is None and extra_ok
    has_mt = True
    if need or want_ribo or want_hb or want_malat1:
        M, var_names = _counts_matrix(adata)
        mt_mask = np.asarray(var_names.str.startswith("MT-") | var_names.str.startswith("mt-"))
        none = np.zeros(len(var_names), dtype=bool)
        ribo_mask = np.asarray(var_names.str.startswith(("RPS", "RPL", "Rps", "Rpl"))) \
            if want_ribo else none
        hb_mask = haemoglobin.hb_mask(var_names, detect_species(adata).species) if want_hb else none
        mal_mask = malat1_mask(var_names) if want_malat1 else none
        if (want_ribo or want_hb or want_malat1) and not _looks_like_counts(M):
            ribo_mask = hb_mask = mal_mask = none  # log or scaled X: no meaningful %
        if need or ribo_mask.any() or hb_mask.any() or mal_mask.any():
            tc, ng, (mt_counts, ribo_counts, hb_counts, mal_counts) = _row_sums(
                M, [mt_mask, ribo_mask, hb_mask, mal_mask])
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = lambda part: np.where(tc > 0, part / tc * 100.0, 0.0)  # noqa: E731
                if out["pct_mt"] is None:
                    has_mt = bool(mt_mask.any())
                    out["pct_mt"] = pct(mt_counts) if has_mt else np.zeros_like(tc)
                if ribo_mask.any():
                    out["pct_ribo"] = pct(ribo_counts)
                if hb_mask.any():
                    out["pct_hb"] = pct(hb_counts)
                if mal_mask.any():
                    out["pct_malat1"] = pct(mal_counts)
            if out["total_counts"] is None:
                out["total_counts"] = tc
            if out["n_genes"] is None:
                out["n_genes"] = ng
    if has_mt and out["pct_mt"] is not None and not np.any(out["pct_mt"] > 0) and not _has_mt_genes(adata):
        # A % mito column of zeros on a matrix without MT- genes (the QC step ran
        # but had nothing to count) measures nothing: report it as unmeasurable,
        # not as a clean 0 %. A non-zero column is kept even then: authors often
        # drop MT genes after computing it.
        has_mt = False
    return {**out, "computed": bool(need), "has_mt": has_mt, "ribo_checked": bool(need) or ribo}


def _has_mt_genes(adata) -> bool:
    """Whether the matrix (or .raw) has any mitochondrial MT-/mt- genes."""
    names = [adata.var_names]
    if getattr(adata, "raw", None) is not None:
        names.append(adata.raw.var_names)
    return any(bool(n.str.startswith(("MT-", "mt-")).any()) for n in names)


# Layer key -> obsm key of the protein modality (core/protein.py).
PROTEIN_LAYERS = {"protein": protein.CLR_KEY, "protein_counts": protein.COUNTS_KEY}

class AnnDataAdaptor:
    """Provides read-only access to a single h5ad file.

    Loads the AnnData object lazily on first access.
    """

    def __init__(self, h5ad_path: str) -> None:
        self.h5ad_path = h5ad_path
        self._adata: ad.AnnData | None = None
        self._qc_cache: dict | None = None
        self._qc_cells: dict | None = None
        self._fingerprint: tuple[pd.Index, str] | None = None  # (obs_names hashed, hash)
        self._ranges: dict[tuple, tuple[float, float]] = {}  # (layer, id, shape) -> colour range

    @property
    def adata(self) -> ad.AnnData:
        if self._adata is None:
            logger.info("Loading %s", self.h5ad_path)
            file_size = Path(self.h5ad_path).stat().st_size
            # Use backed mode for very large files (>2GB)
            if file_size > 2 * 1024**3:
                self._adata = ad.read_h5ad(self.h5ad_path, backed="r")
                logger.info("Loaded in backed mode (%d bytes)", file_size)
            else:
                self._adata = ad.read_h5ad(self.h5ad_path)
                logger.info("Loaded in memory (%d bytes)", file_size)
            if repair_slash_keys(self._adata.uns):
                logger.info("Re-keyed cached results saved under '/' keys")
            if not self._adata.isbacked:
                # Protein counts that came without the CLR (a converted Seurat ADT
                # assay, totalVI's obsm): normalised here, in memory.
                protein.adopt_obsm(self._adata)
        return self._adata

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def n_cells(self) -> int:
        return self.adata.n_obs

    def cells_fingerprint(self) -> str:
        """A short hash of the cell barcodes in order: equal for two datasets
        whose cell i is the same cell, so a saved session's cell indices and
        camera still mean the same cells. Labels, embeddings and genes don't
        change it; filtering or reordering cells does."""
        names = self.adata.obs_names
        if self._fingerprint is not None and self._fingerprint[0] is names:
            return self._fingerprint[1]
        h = hashlib.sha256(f"{len(names)}\n".encode())
        for start in range(0, len(names), 100_000):
            h.update("\n".join(map(str, names[start:start + 100_000])).encode())
            h.update(b"\n")
        fingerprint = h.hexdigest()[:24]
        self._fingerprint = (names, fingerprint)
        return fingerprint

    def known_cells_fingerprint(self) -> str | None:
        """The fingerprint when it is already hashed for the current cells, else
        None: never reads the file or hashes (the dataset list asks for it)."""
        if self._adata is None or self._fingerprint is None:
            return None
        names, fingerprint = self._fingerprint
        return fingerprint if names is self._adata.obs_names else None

    def n_genes(self) -> int:
        return self.adata.n_vars

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def available_embeddings(self) -> list[dict]:
        """Return list of embedding info dicts (name + dimensions)."""
        results = []
        for key in self.adata.obsm_keys():
            if key.startswith("X_") and key not in NOT_EMBEDDINGS:
                arr = self.adata.obsm[key]
                results.append({"name": key, "dimensions": arr.shape[1] if arr.ndim > 1 else 1})
        return results

    def get_embedding(self, name: str) -> np.ndarray:
        """Return embedding coordinates as float32 array, shape (n_cells, dims)."""
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {self.adata.obsm_keys()}")
        arr = np.asarray(self.adata.obsm[name], dtype=np.float32)
        return arr

    # ------------------------------------------------------------------
    # Metadata (obs columns)
    # ------------------------------------------------------------------

    def obs_columns_info(self) -> list[dict]:
        """Return info about each obs column: name, dtype, n_unique, values (if categorical)."""
        results = []
        for col in self.adata.obs.columns:
            series = self.adata.obs[col]
            info: dict = {"name": col, "dtype": str(series.dtype)}
            if hasattr(series, "cat") or series.dtype.name == "category":
                cats = series.cat.categories.tolist()
                info["n_unique"] = len(cats)
                # Only inline values for low-cardinality categoricals usable as
                # discrete selectors/colour groups. High-cardinality columns
                # (e.g. cell_id, mis-typed continuous coords) would bloat the
                # payload by tens of MB across all datasets — mirror the object
                # branch and omit values beyond the cap.
                if len(cats) <= 100:
                    info["values"] = [str(c) for c in cats]
                    # Palette slot per value once a reorder pinned colours to labels.
                    slots = category_order.color_slots(self.adata, col)
                    if slots is not None:
                        info["color_slots"] = slots
            elif pd.api.types.is_bool_dtype(series):
                # A two-level categorical; the colour codes are int(value), see embeddings.
                info["n_unique"] = int(series.nunique())
                info["values"] = ["False", "True"]
            elif series.dtype == object:
                unique = series.unique()
                info["n_unique"] = len(unique)
                if len(unique) <= 100:
                    info["values"] = [str(v) for v in unique]
            else:
                info["n_unique"] = int(series.nunique())
            results.append(info)
        return results

    def get_obs_column(self, name: str) -> pd.Series:
        """Return a single obs column as a pandas Series."""
        if name not in self.adata.obs.columns:
            raise KeyError(f"Column '{name}' not found in obs.")
        return self.adata.obs[name]

    def get_obs_summary(self, groupby: str) -> dict:
        """Return cell counts per group for a categorical obs column."""
        col = self.get_obs_column(groupby)
        counts = col.value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    # ------------------------------------------------------------------
    # QC metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _qc_summary(arr: np.ndarray, n_bins: int) -> dict:
        """Histogram + summary stats for one QC metric."""
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0,
                "hist": {"bin_edges": [], "counts": []},
            }
        counts, edges = np.histogram(arr, bins=n_bins)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "hist": {"bin_edges": edges.tolist(), "counts": counts.tolist()},
        }

    def qc_per_cell(self, ribo: bool = False) -> dict:
        """Per-cell QC arrays (see ``per_cell_qc``), cached for the dataset's
        lifetime in memory; shared by the QC plots and the per-cluster flags.
        ``ribo`` also works out % ribosomal when obs lacks it (an extra pass)."""
        cells = self._qc_cells
        if cells is None or (ribo and not cells["ribo_checked"]):
            self._qc_cells = per_cell_qc(self.adata, ribo=ribo)
        return self._qc_cells

    def qc_distributions(self, n_scatter: int = 5000, n_bins: int = 40) -> dict:
        """Per-cell QC distributions for the Data Assessment tab.

        Reads ``total_counts`` / ``n_genes_by_counts`` / ``pct_counts_mt`` from
        ``obs`` when present, otherwise computes them on demand from ``X`` (no
        mutation of the AnnData, no disk write). Returns per-metric histograms +
        summary stats and a downsampled total_counts-vs-genes scatter (coloured
        by % mito) so the payload stays small even for hundreds of thousands of
        cells. Result is cached on the adaptor for the dataset's lifetime.
        """
        if self._qc_cache is not None:
            return self._qc_cache

        obs = self.adata.obs

        def _col(name: str) -> np.ndarray | None:
            if name in obs.columns:
                return np.asarray(obs[name].to_numpy(), dtype=np.float64)
            return None

        # Counts, not whatever X holds: summing a scaled X gave negative
        # "counts per cell" (see _counts_matrix). ``ribo``: the extras too (MALAT1 %),
        # in the pass cluster QC needs anyway (cached).
        cells = self.qc_per_cell(ribo=True)
        total_counts, n_genes, pct_mt = cells["total_counts"], cells["n_genes"], cells["pct_mt"]
        computed = cells["computed"]

        has_mt = bool(cells.get("has_mt", True))
        metrics = {
            "n_genes_by_counts": self._qc_summary(n_genes, n_bins),
            "total_counts": self._qc_summary(total_counts, n_bins),
        }
        if has_mt:  # else no MT genes: a histogram of zeros would read as "clean"
            metrics["pct_counts_mt"] = self._qc_summary(pct_mt, n_bins)
        # Optional metrics contributed by later QC analyses (doublet detection,
        # cell-cycle scoring, ribo/hemo QC) — included once they exist in obs, so
        # the plot set grows as the user runs more steps.
        for name in ("doublet_score", "S_score", "G2M_score", "dissociation_stress_score",
                     "pct_counts_ribo", "pct_counts_hb"):
            col = _col(name)
            if col is not None:
                metrics[name] = self._qc_summary(col, n_bins)
        if cells.get("pct_malat1") is not None:  # None: no MALAT1 gene, or not counts
            metrics["pct_malat1"] = self._qc_summary(cells["pct_malat1"], n_bins)

        n = int(len(total_counts))
        idx = np.linspace(0, n - 1, n_scatter).astype(int) if n > n_scatter else np.arange(n)
        scatter = {
            "x": total_counts[idx].tolist(),
            "y": n_genes[idx].tolist(),
            "color": pct_mt[idx].tolist() if has_mt else [],
            "x_label": "total_counts",
            "y_label": "n_genes_by_counts",
            "color_label": "pct_counts_mt" if has_mt else "",
            "n_shown": int(len(idx)),
        }

        result = {
            "n_cells": n,
            "computed_on_demand": computed,
            "has_mt": has_mt,
            "metrics": metrics,
            "scatter": scatter,
        }
        self._qc_cache = result
        return result

    # ------------------------------------------------------------------
    # Expression layers
    # ------------------------------------------------------------------

    def _var_names_for_layer(self, layer: str) -> list[str]:
        """Return gene names for a given expression layer key. The protein
        modality (CITE-seq) is read as two more layers: ``protein`` (CLR) and
        ``protein_counts``, with the protein names as its genes."""
        if layer in PROTEIN_LAYERS and protein.has_protein(self.adata):
            return protein.protein_names(self.adata)
        if layer == "raw":
            if self.adata.raw is not None:
                return self.adata.raw.var_names.tolist()
            raise KeyError("Layer 'raw' requested but adata.raw is not available.")
        if layer == "X":
            return self.adata.var_names.tolist()
        # Named layer in adata.layers — shares var_names with adata.X
        if layer in self.adata.layers:
            return self.adata.var_names.tolist()
        raise KeyError(
            f"Layer '{layer}' not found. Available: {[lyr['key'] for lyr in self.available_expression_layers()]}"
        )

    def _get_matrix_for_layer(self, layer: str):
        """Return the expression matrix for a given layer key."""
        if layer in PROTEIN_LAYERS and protein.has_protein(self.adata):
            return self.adata.obsm[PROTEIN_LAYERS[layer]].to_numpy(dtype=np.float32)
        if layer == "raw":
            if self.adata.raw is not None:
                return self.adata.raw.X
            raise KeyError("Layer 'raw' requested but adata.raw is not available.")
        if layer == "X":
            return self.adata.X
        if layer in self.adata.layers:
            return self.adata.layers[layer]
        raise KeyError(f"Layer '{layer}' not found.")

    def available_expression_layers(self) -> list[dict]:
        """Return list of available expression layers with metadata.

        Each entry has keys: key, label, n_genes.
        """
        layers: list[dict] = []

        # raw (log-normalized, all genes)
        if self.adata.raw is not None:
            layers.append({
                "key": "raw",
                "label": "Log-normalized (raw)",
                "n_genes": len(self.adata.raw.var_names),
            })

        # X (main matrix — may be scaled/HVG-subset)
        n_x = self.adata.n_vars
        if self.adata.raw is not None:
            label = f"Scaled/HVG ({n_x} genes)" if n_x < len(self.adata.raw.var_names) else "Main matrix (X)"
        else:
            label = "Main matrix (X)"
        layers.append({"key": "X", "label": label, "n_genes": n_x})

        # Named layers (counts, spliced, unspliced, etc.)
        for name in self.adata.layers:
            layers.append({
                "key": name,
                "label": name,
                "n_genes": self.adata.n_vars,
            })

        return layers

    def protein_info(self) -> dict | None:
        """The protein modality (CITE-seq antibodies), or None: names in panel
        order, the isotype controls among them, and the layer keys to read it."""
        names = protein.protein_names(self.adata)
        if not names:
            return None
        return {
            "names": names,
            "controls": [n for n in names if protein.is_control(n)],
            "layer": "protein",
            "counts_layer": "protein_counts",
            "normalisation": "CLR per cell",
        }

    def default_expression_layer(self) -> str:
        """Return the default layer key — prefers 'raw' if available, else 'X'."""
        if self.adata.raw is not None:
            return "raw"
        return "X"

    # ------------------------------------------------------------------
    # Gene names and expression
    # ------------------------------------------------------------------

    def _expression_var_names(self) -> list[str]:
        """Return gene names from the best available expression source.

        Prefers adata.raw.var_names (log-normalized, all genes) over
        adata.var_names (which may be scaled/subset to HVGs after pipeline).
        """
        return self._var_names_for_layer(self.default_expression_layer())

    def gene_names(self) -> list[str]:
        """Return all gene/variable names."""
        return self._expression_var_names()

    def search_genes(self, query: str, limit: int = 20) -> list[str]:
        """Case-insensitive prefix search for gene names."""
        q = query.lower()
        results = [
            name
            for name in self._expression_var_names()
            if name.lower().startswith(q)
        ]
        results.sort(key=str.lower)
        return results[:limit]

    def get_expression(
        self, gene_names: list[str], layer: str | None = None
    ) -> np.ndarray:
        """Return expression matrix for requested genes, shape (n_cells, n_genes).

        Parameters
        ----------
        gene_names : list[str]
            Gene names to retrieve.
        layer : str, optional
            Expression layer key (e.g. "raw", "X", "counts"). Defaults to
            ``default_expression_layer()``.

        Returns dense float32 array with NaN replaced by 0.
        """
        if layer is None:
            layer = self.default_expression_layer()

        var_names = self._var_names_for_layer(layer)
        X = self._get_matrix_for_layer(layer)

        # Find gene indices
        indices = []
        for g in gene_names:
            if g in var_names:
                indices.append(var_names.index(g))

        if not indices:
            return np.zeros((self.n_cells(), 0), dtype=np.float32)

        # Extract expression data — handle scipy sparse, backed sparse datasets,
        # and dense arrays via duck typing
        # Read the columns in increasing order, once each (an on-disk h5py matrix
        # rejects anything else), then put them back in the requested order.
        order, inverse = np.unique(np.asarray(indices), return_inverse=True)
        sliced = X[:, order]
        if hasattr(sliced, 'toarray'):
            data = sliced.toarray()
        else:
            data = np.asarray(sliced)
        data = data[:, inverse]

        return np.nan_to_num(data.astype(np.float32), nan=0.0)

    def get_expression_for_violin(
        self, gene: str, groupby: str, layer: str | None = None
    ) -> dict[str, list[float]]:
        """Return expression values grouped for violin plot rendering."""
        expr = self.get_expression([gene], layer=layer)
        if expr.shape[1] == 0:
            return {}

        values = expr[:, 0]
        groups = self.get_obs_column(groupby)
        result: dict[str, list[float]] = {}
        for group_val in groups.unique():
            mask = groups == group_val
            result[str(group_val)] = values[mask].tolist()
        return result

    def expression_by_group(
        self, gene_names: list[str], groupby: str, layer: str | None = None
    ) -> dict:
        """Mean expression and fraction of expressing cells per group, for the
        marker dot plot and heatmap. Genes missing from the layer are dropped.

        ``mean``, ``mean_expressing`` and ``fraction`` are indexed [group][gene].
        ``mean`` is over all of a group's cells, so it mixes how many cells
        express a gene with how much; ``mean_expressing`` is over the cells
        with a value above 0 only (0 when none), so it carries the level alone
        (Seurat / scanpy's ``mean_only_expressed``). ``scale_range`` is the
        layer's fixed colour range (``core/expression_range.py``), the same
        whatever genes and groups are asked for.
        """
        if layer is None:
            layer = self.default_expression_layer()
        present = set(self._var_names_for_layer(layer))
        genes = [g for g in dict.fromkeys(gene_names) if g in present]
        col = self.get_obs_column(groupby)
        if isinstance(col.dtype, pd.CategoricalDtype):
            groups = [str(c) for c in col.cat.categories if (col == c).any()]
        else:
            groups = sorted(str(v) for v in col.dropna().unique())
        labels = col.astype(str).to_numpy()
        expr = self.get_expression(genes, layer=layer) if genes else np.zeros((len(labels), 0))
        mean, mean_expressing, fraction = [], [], []
        for g in groups:
            sub = expr[labels == g]
            expressing = sub > 0
            n_expressing = expressing.sum(axis=0)
            level = np.where(sub > 0, sub, 0).sum(axis=0) / np.maximum(n_expressing, 1)
            mean.append([round(float(v), 4) for v in sub.mean(axis=0)])
            mean_expressing.append([round(float(v), 4) for v in level])
            fraction.append([round(float(v), 4) for v in expressing.mean(axis=0)])
        return {
            "groupby": groupby, "layer": layer, "groups": groups, "genes": genes,
            "mean": mean, "mean_expressing": mean_expressing, "fraction": fraction,
            "scale_range": list(self.layer_scale_range(layer)),
        }

    def layer_scale_range(self, layer: str) -> tuple[float, float]:
        """The layer's fixed colour range for marker plots, cached per matrix."""
        source = (
            self.adata.obsm[PROTEIN_LAYERS[layer]]
            if layer in PROTEIN_LAYERS and protein.has_protein(self.adata)
            else self._get_matrix_for_layer(layer)
        )
        key = (layer, id(source), tuple(source.shape))
        if key not in self._ranges:
            self._ranges[key] = expression_range.layer_range(self._get_matrix_for_layer(layer))
        return self._ranges[key]

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    _CLUSTERING_CANDIDATES = CLUSTERING_CANDIDATES

    def active_clustering_column(self) -> str | None:
        """Return the best clustering column name.

        Checks adata.uns['scview_active_clustering'] first (set by pipeline
        re-clustering), then falls back to standard candidate column names.
        """
        active = self.adata.uns.get("scview_active_clustering")
        if active and active in self.adata.obs.columns:
            return active
        for candidate in self._CLUSTERING_CANDIDATES:
            if candidate in self.adata.obs.columns:
                return candidate
        return None

    def user_annotation_columns(self) -> list[str]:
        """The hand-label columns (uns ``scview_user_annotations``) still in obs."""
        return user_columns(self.adata)

    # ------------------------------------------------------------------
    # Marker genes
    # ------------------------------------------------------------------

    def has_markers(self, column: str | None = None) -> bool:
        if column:
            return resolve_markers_key(self.adata.uns, column) is not None
        return "rank_genes_groups" in self.adata.uns or any(
            markers_column(k) is not None for k in self.adata.uns
        )

    def marker_columns(self) -> list[str]:
        """Return list of obs columns that have pre-computed markers."""
        cols = []
        for key in self.adata.uns:
            if markers_column(key) is not None:
                cols.append(markers_column(key))
        if not cols and "rank_genes_groups" in self.adata.uns:
            rgg = self.adata.uns["rank_genes_groups"]
            groupby = rgg.get("params", {}).get("groupby", "")
            if groupby:
                cols.append(groupby)
        return cols

    def get_markers(
        self, groupby: str | None = None, column: str | None = None, n_genes: int = 100,
    ) -> pd.DataFrame | None:
        """Extract marker genes from adata.uns.

        Parameters
        ----------
        groupby : str, optional
            Filter to a specific group value.
        column : str, optional
            Obs column whose markers to retrieve. If None, uses default key.
        n_genes : int
            Max genes to return per group (default 100).

        Returns a DataFrame with columns: group, gene, logfoldchange, pval, pval_adj,
        pct_in, pct_out, and the specificity columns of ``core/marker_specificity.py``
        (tau, pct_max_other, max_other_group, pct_ratio) when they can be computed.
        """
        # A file's own markers are often under the bare key with params.groupby set.
        uns_key = resolve_markers_key(self.adata.uns, column or "")
        if uns_key is None:
            return None

        rgg = self.adata.uns[uns_key]
        groups = list(rgg["names"].dtype.names) if hasattr(rgg["names"].dtype, "names") else []

        if not groups:
            return None

        rows = []
        for group in groups:
            if groupby and group != groupby:
                continue
            total_genes = len(rgg["names"][group])
            limit = min(n_genes, total_genes)
            genes = [str(g) for g in rgg["names"][group][:limit]]
            spec = marker_specificity.columns_for_group(rgg, str(group), genes)
            for i in range(limit):
                gene_name = str(rgg["names"][group][i])
                row = {
                    "group": str(group),
                    "gene": gene_name,
                }
                if "logfoldchanges" in rgg:
                    row["logfoldchange"] = float(rgg["logfoldchanges"][group][i])
                if "pvals" in rgg:
                    row["pval"] = float(rgg["pvals"][group][i])
                if "pvals_adj" in rgg:
                    row["pval_adj"] = float(rgg["pvals_adj"][group][i])
                if "pts" in rgg and rgg["pts"] is not None:
                    pts = rgg["pts"]
                    try:
                        if hasattr(pts, "loc"):
                            # pts is a DataFrame (scanpy >= 1.8) — index by gene name
                            row["pct_in"] = float(pts.loc[gene_name, group])
                        elif hasattr(pts.dtype, "names") and group in pts.dtype.names:
                            # pts is a recarray
                            row["pct_in"] = float(pts[group][i])
                    except (KeyError, IndexError):
                        pass
                if "pts_rest" in rgg and rgg["pts_rest"] is not None:
                    pts_rest = rgg["pts_rest"]
                    try:
                        if hasattr(pts_rest, "loc"):
                            row["pct_out"] = float(pts_rest.loc[gene_name, group])
                        elif hasattr(pts_rest.dtype, "names") and group in pts_rest.dtype.names:
                            row["pct_out"] = float(pts_rest[group][i])
                    except (KeyError, IndexError):
                        pass
                for field, values in spec.items():
                    value = values[i]
                    row[field] = value if value is None or isinstance(value, str) else float(value)
                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Trajectory / pseudotime
    # ------------------------------------------------------------------

    def available_pseudotime_columns(self) -> list[str]:
        """Find obs columns that look like pseudotime values."""
        candidates = []
        for col in self.adata.obs.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ("pseudotime", "dpt_", "latent_time", "monocle")):
                if pd.api.types.is_numeric_dtype(self.adata.obs[col]):
                    candidates.append(col)
        return candidates

    # ------------------------------------------------------------------
    # Cell-level metadata
    # ------------------------------------------------------------------

    def get_cell_metadata(self, index: int) -> dict:
        """Return all obs values for a single cell at the given integer index."""
        if index < 0 or index >= self.n_cells():
            raise IndexError(f"Cell index {index} out of range [0, {self.n_cells()}).")
        row = self.adata.obs.iloc[index]
        result: dict = {}
        for col in self.adata.obs.columns:
            val = row[col]
            # Convert numpy/pandas types to JSON-serializable Python types
            if pd.isna(val):
                result[col] = None
            elif hasattr(val, "item"):
                result[col] = val.item()
            else:
                result[col] = str(val) if not isinstance(val, (int, float, bool, str)) else val
        return result

    # ------------------------------------------------------------------
    # Cross-tabulation
    # ------------------------------------------------------------------

    def get_obs_crosstab(self, row_col: str, col_col: str, facet_col: str | None = None) -> dict:
        """Return a cross-tabulation of two obs columns.

        Returns dict with keys: row_values, col_values, counts (2-D list). With
        ``facet_col`` (e.g. the condition, for one bar per sample split by
        condition) also facet_values and facet_counts[facet][row][col], on the
        same rows and columns.
        """
        row_series = self.get_obs_column(row_col)
        col_series = self.get_obs_column(col_col)
        ct = pd.crosstab(row_series, col_series)
        out = {
            "row_values": [str(v) for v in ct.index.tolist()],
            "col_values": [str(v) for v in ct.columns.tolist()],
            "counts": ct.values.tolist(),
        }
        if facet_col:
            facet = self.get_obs_column(facet_col)
            if isinstance(facet.dtype, pd.CategoricalDtype):
                present = set(facet.dropna().astype(str))
                facet_values = [str(c) for c in facet.cat.categories if str(c) in present]
            else:
                facet_values = sorted({str(v) for v in facet.dropna()})
            labels = facet.astype(str).where(facet.notna())
            out["facet_values"] = facet_values
            out["facet_counts"] = [
                pd.crosstab(row_series[(labels == f).to_numpy()], col_series[(labels == f).to_numpy()])
                .reindex(index=ct.index, columns=ct.columns, fill_value=0)
                .astype(int).values.tolist()
                for f in facet_values
            ]
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Called when the dataset manager drops this adaptor. A backed file is closed
        (reading it again is cheap). An in-memory dataset is kept for the requests
        that took this adaptor before the drop: dropping it made their next ``adata``
        read the whole file again outside the manager's cache and its memory check
        (after a stopped run on ovary, 3.2 GB beside the copy being reopened: the
        backend was killed). It is freed when the last of them lets go."""
        if self._adata is not None and getattr(self._adata, "isbacked", False):
            if self._adata.file is not None:
                self._adata.file.close()
            self._adata = None
