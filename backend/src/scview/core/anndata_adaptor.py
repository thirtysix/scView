"""AnnDataAdaptor – read-only interface to an h5ad dataset."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


class AnnDataAdaptor:
    """Provides read-only access to a single h5ad file.

    Loads the AnnData object lazily on first access.
    """

    def __init__(self, h5ad_path: str) -> None:
        self.h5ad_path = h5ad_path
        self._adata: ad.AnnData | None = None
        self._qc_cache: dict | None = None

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
        return self._adata

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def n_cells(self) -> int:
        return self.adata.n_obs

    def n_genes(self) -> int:
        return self.adata.n_vars

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def available_embeddings(self) -> list[dict]:
        """Return list of embedding info dicts (name + dimensions)."""
        results = []
        for key in self.adata.obsm_keys():
            if key.startswith("X_"):
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

        adata = self.adata
        obs = adata.obs
        computed = False

        def _col(name: str) -> np.ndarray | None:
            if name in obs.columns:
                return np.asarray(obs[name].to_numpy(), dtype=np.float64)
            return None

        total_counts = _col("total_counts")
        n_genes = _col("n_genes_by_counts")
        if n_genes is None:
            n_genes = _col("n_genes")
        pct_mt = _col("pct_counts_mt")

        if total_counts is None or n_genes is None or pct_mt is None:
            computed = True
            X = adata.X
            tc = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
            ng = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)
            var_names = adata.var_names
            mt_mask = np.asarray(
                var_names.str.startswith("MT-") | var_names.str.startswith("mt-")
            )
            if mt_mask.any():
                mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel().astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pm = np.where(tc > 0, mt_counts / tc * 100.0, 0.0)
            else:
                pm = np.zeros_like(tc)
            total_counts = tc if total_counts is None else total_counts
            n_genes = ng if n_genes is None else n_genes
            pct_mt = pm if pct_mt is None else pct_mt

        doublet = _col("doublet_score")

        metrics = {
            "n_genes_by_counts": self._qc_summary(n_genes, n_bins),
            "total_counts": self._qc_summary(total_counts, n_bins),
            "pct_counts_mt": self._qc_summary(pct_mt, n_bins),
        }
        if doublet is not None:
            metrics["doublet_score"] = self._qc_summary(doublet, n_bins)

        n = int(len(total_counts))
        idx = np.linspace(0, n - 1, n_scatter).astype(int) if n > n_scatter else np.arange(n)
        scatter = {
            "x": total_counts[idx].tolist(),
            "y": n_genes[idx].tolist(),
            "color": pct_mt[idx].tolist(),
            "x_label": "total_counts",
            "y_label": "n_genes_by_counts",
            "color_label": "pct_counts_mt",
            "n_shown": int(len(idx)),
        }

        result = {
            "n_cells": n,
            "computed_on_demand": computed,
            "metrics": metrics,
            "scatter": scatter,
        }
        self._qc_cache = result
        return result

    # ------------------------------------------------------------------
    # Expression layers
    # ------------------------------------------------------------------

    def _var_names_for_layer(self, layer: str) -> list[str]:
        """Return gene names for a given expression layer key."""
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
            f"Layer '{layer}' not found. Available: {[l['key'] for l in self.available_expression_layers()]}"
        )

    def _get_matrix_for_layer(self, layer: str):
        """Return the expression matrix for a given layer key."""
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
        sliced = X[:, indices]
        if hasattr(sliced, 'toarray'):
            data = sliced.toarray()
        else:
            data = np.asarray(sliced)

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

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    _CLUSTERING_CANDIDATES = (
        "leiden", "louvain", "cluster", "clusters", "seurat_clusters", "cell_type",
    )

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

    # ------------------------------------------------------------------
    # Marker genes
    # ------------------------------------------------------------------

    def has_markers(self, column: str | None = None) -> bool:
        if column:
            return f"rank_genes_groups__{column}" in self.adata.uns
        return "rank_genes_groups" in self.adata.uns or any(
            k.startswith("rank_genes_groups__") for k in self.adata.uns
        )

    def marker_columns(self) -> list[str]:
        """Return list of obs columns that have pre-computed markers."""
        cols = []
        for key in self.adata.uns:
            if key.startswith("rank_genes_groups__"):
                cols.append(key.split("__", 1)[1])
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

        Returns a DataFrame with columns: group, gene, logfoldchange, pval, pval_adj, pct.
        """
        if column:
            uns_key = f"rank_genes_groups__{column}"
        else:
            uns_key = "rank_genes_groups"

        if uns_key not in self.adata.uns:
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

    def get_obs_crosstab(self, row_col: str, col_col: str) -> dict:
        """Return a cross-tabulation of two obs columns.

        Returns dict with keys: row_values, col_values, counts (2-D list).
        """
        row_series = self.get_obs_column(row_col)
        col_series = self.get_obs_column(col_col)
        ct = pd.crosstab(row_series, col_series)
        return {
            "row_values": [str(v) for v in ct.index.tolist()],
            "col_values": [str(v) for v in ct.columns.tolist()],
            "counts": ct.values.tolist(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._adata is not None:
            if hasattr(self._adata, "file") and self._adata.file is not None:
                self._adata.file.close()
            self._adata = None
