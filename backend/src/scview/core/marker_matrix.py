"""The matrix marker genes are ranked on: log-normalised values, never raw counts.

Markers ranked ``.raw`` whenever it was there. A CELLxGENE file keeps the raw COUNTS in
``.raw`` beside a log-normalised X, so a fresh import got a Wilcoxon test on counts and fold
changes of counts read as log values (kidney MAGI2 log2FC 437, mouse fat Mageb18 61 and NaN
for Malat1; research/v075-qa). The choice is now the gene-set scoring's
(``activity.lognorm_source``): ``.raw`` when it holds log values, else X when it does, else the
shifted log of the counts. Each table records it in ``params["matrix"]``.

A table stored before that (no ``matrix``) is stale when only counts can explain it
(``on_counts``): a fold change no log-normalised matrix can reach (or an infinite one), or it says it
ranked ``.raw`` (or X) and that matrix holds counts. ``ora_background.resolve_markers_key``
skips such a table, so every reader treats the column as having no markers and the markers
route computes them again.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_LOG = "log-normalised .raw"
X_LOG = "log-normalised X"

# |log2 fold change| above this needs counts: on log1p of values summing to 1e6 per cell (CPM),
# a gene holding all of one group's counts and none of the rest's gives 19.9 + 29.9 (scanpy's
# 1e-9 floor) = 50. Counts give hundreds (MAGI2 437), or inf / NaN once expm1 overflows.
STALE_LFC = 55.0


def choose(adata) -> tuple[str, Any]:
    """(label, ``activity.Lognorm`` or None): the matrix to rank. None when the dataset has
    neither log-normalised values nor counts: then what is stored is ranked as it is."""
    from scview.core.activity import lognorm_source

    try:
        choice = lognorm_source(adata)
    except ValueError:
        return (".raw as stored" if adata.raw is not None else "X as stored"), None
    return choice.source, choice


def light(adata, column: str, labels: pd.Series, choice: Any = None):
    """A light AnnData on the chosen matrix (``choice``, else ``choose``'s), with ``labels``
    as its one obs column ``column``: (AnnData, label). Nothing of ``adata`` is copied except
    the shifted log when the matrix holds counts; a backed dataset's matrix is read whole."""
    import anndata as ad

    if choice is None:
        label, choice = choose(adata)
    else:
        label = choice.source
    if choice is None:
        from scview.core.working_copy import read_matrix

        src = adata.raw if adata.raw is not None else adata
        X, names = read_matrix(src.X), src.var_names
    else:
        X, names = choice.whole(), choice.names
    work = ad.AnnData(X=X, obs=pd.DataFrame({column: labels.values}, index=adata.obs_names),
                      var=pd.DataFrame(index=pd.Index(names).astype(str)))
    if choice is not None and choice.counts:
        work.uns["log1p"] = {"base": None}  # the shifted log is natural
    elif "log1p" in adata.uns:  # the base the fold changes undo, as for the pipeline's
        work.uns["log1p"] = adata.uns["log1p"]
    return work, label


def rank(adata, groupby: str, *, key_added: str | None = None, groups: Any = "all",
         specificity: bool = True) -> str:
    """Wilcoxon markers with pts of ``groupby`` into ``adata.uns[key_added or
    "rank_genes_groups"]`` (``core/rank_genes.py``), on the log-normalised matrix; returns the
    matrix's label, also stored as the table's ``params["matrix"]``. ``.raw`` or X is ranked
    in place (as before when ``.raw`` holds log values); counts on a light copy."""
    from scview.core import rank_genes

    key = key_added or "rank_genes_groups"
    label, choice = choose(adata)
    if choice is not None and choice.counts:
        work, _ = light(adata, groupby, adata.obs[groupby], choice)
        rank_genes.rank_genes_groups(work, groupby, method="wilcoxon", pts=True, groups=groups,
                                     use_raw=False, key_added=key, specificity=specificity)
        adata.uns[key] = work.uns[key]
    else:
        use_raw = None if choice is None else choice.source == RAW_LOG
        rank_genes.rank_genes_groups(adata, groupby, method="wilcoxon", pts=True, groups=groups,
                                     use_raw=use_raw, key_added=key_added, specificity=specificity)
    adata.uns[key].setdefault("params", {})["matrix"] = label
    return label


def _fold_changes(rgg) -> np.ndarray | None:
    lfc = rgg.get("logfoldchanges") if isinstance(rgg, dict) else None
    if lfc is None:
        return None
    arr = np.asarray(lfc)
    if arr.dtype.names:
        from numpy.lib.recfunctions import structured_to_unstructured

        try:
            arr = structured_to_unstructured(arr)
        except (TypeError, ValueError):
            return None
    try:
        return np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def on_counts(rgg, adata=None) -> bool:
    """Whether a stored marker table was ranked on raw counts. A table that names its
    ``matrix`` was not. ``adata`` adds the check of the matrix it says it ranked."""
    params = rgg.get("params") if isinstance(rgg, dict) else None
    if not isinstance(params, dict) or params.get("matrix"):
        return False
    lfc = _fold_changes(rgg)
    if lfc is not None and lfc.size:
        if np.isinf(lfc).any():  # expm1 of a count past 709 (NaN alone can be a file's own)
            return True
        finite = np.abs(lfc[np.isfinite(lfc)])
        if finite.size and float(finite.max()) > STALE_LFC:
            return True
    if adata is None:
        return False
    from scview.core.pipeline import looks_like_counts

    use_raw = params.get("use_raw")
    try:
        if use_raw in (True, "True", 1) and adata.raw is not None:
            return looks_like_counts(adata.raw.X)
        if use_raw in (False, "False", 0):
            return looks_like_counts(adata.X)
    except Exception:  # noqa: BLE001 - an unreadable matrix: no claim about it
        return False
    return False
