"""A Python script that replays a dataset's recorded processing (C006).

The provenance history (``core/provenance.py``) records each step scView ran with
its parameters and what the step found (the doublet rate it used, each sample's
QC thresholds, the dissociation genes present). ``build_script`` turns that
history into a standalone scanpy script: run on the dataset's input file it
repeats the processing outside scView, with no scView import.

Each step maps to code that does what scView's own step does (``pipeline.py``,
``subcluster.py``, ``doublets.py``...), with the recorded values written in.
``describe`` says how faithful each step's code is:

- ``exact``: the same computation (thread count and BLAS can still move the
  last digits of PCA / UMAP / Harmony, and with them a few cells' clusters);
- ``approximate``: the closest scanpy code, with a ``# NOTE:`` on what differs;
- ``recorded``: the step's outcome written in from the record (an LLM's cell-type
  names), not computed again;
- ``not_replayable``: a comment with the recorded values (manual labels, scView's
  own confidence checks);
- ``unknown``: a step type this version does not know; a placeholder comment.

Everything taken from the dataset goes into the script as a Python literal
(``repr``) or a sanitised comment, so a crafted provenance block cannot inject
code into the script.
"""

from __future__ import annotations

import json
import pprint
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from scview.core import msigdb_species, provenance, vdj

EXACT = "exact"
APPROXIMATE = "approximate"
RECORDED = "recorded"
NOT_REPLAYABLE = "not_replayable"
UNKNOWN = "unknown"
STATUSES = (EXACT, APPROXIMATE, RECORDED, NOT_REPLAYABLE, UNKNOWN)


@dataclass
class Snippet:
    """One step's code (top-level lines) and how faithful it is."""

    status: str
    code: list[str]
    note: str = ""


@dataclass
class _Ctx:
    """What later steps need from earlier ones while the script is written."""

    adata: Any
    helpers: set[str] = field(default_factory=set)
    # The matrix the last scaling step planned for PCA (core/scaling.py), or None: an
    # older scView's scaling wrote it into X, so PCA reads X as it is.
    scaling: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Text safety
# ---------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x1f\x7f  ]+")


def comment_text(value: object, limit: int = 300) -> str:
    """One line of plain text, safe inside a ``#`` comment."""
    text = _CONTROL.sub(" ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _lit(value: object) -> str:
    """A Python literal of a JSON value (str, number, bool, None, list, dict);
    a long list or dict wraps over lines (inside its brackets, so still valid
    wherever it stands)."""
    return pprint.pformat(value, width=96, sort_dicts=False)


def _comment_block(lines: list[str]) -> list[str]:
    return [f"# {comment_text(line)}" if line else "#" for line in lines]


def _param_comment(params: dict) -> list[str]:
    """The recorded params, one per comment line (long values cut)."""
    out = []
    for k, v in params.items():
        out.append(f"#   {comment_text(k, 80)} = {comment_text(json.dumps(v, default=str), 200)}")
    return out or ["#   (no parameters recorded)"]


# ---------------------------------------------------------------------------
# Helpers the script defines when a step needs them (no scView import)
# ---------------------------------------------------------------------------

_HELPERS: dict[str, str] = {
    "looks_like_counts": '''
def looks_like_counts(M, sample=20_000):
    """Non-negative integers with some signal: a raw-count matrix (as scView checks)."""
    if M is None:
        return False
    if sp.issparse(M):
        vals = np.asarray(M.data[:sample])
    else:
        arr = np.asarray(M)
        vals = arr.ravel()[: sample * 4]
        vals = vals[vals != 0][:sample]
    if vals.size == 0:
        return False
    vals = vals.astype(np.float64, copy=False)
    return bool(np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))
''',
    "keep_raw_counts": '''
def keep_raw_counts(adata):
    """Copy counts held only in .raw into layers['counts'] (scaling replaces .raw)."""
    if "counts" in adata.layers or adata.raw is None:
        return
    if not looks_like_counts(adata.raw.X):
        return
    if len(adata.var_names.difference(adata.raw.var_names)):
        return
    adata.layers["counts"] = adata.raw[:, adata.var_names].X.copy()
''',
    "find_counts": '''
def find_counts(adata):
    """Raw counts for X's genes: layers['counts'], then .raw, then X."""
    if "counts" in adata.layers and looks_like_counts(adata.layers["counts"]):
        return adata.layers["counts"]
    raw = adata.raw
    if raw is not None and not len(adata.var_names.difference(raw.var_names)) \\
            and looks_like_counts(raw.X):
        return raw[:, adata.var_names].X
    if looks_like_counts(adata.X):
        return adata.X
    raise ValueError("No raw counts in layers['counts'], .raw or X.")
''',
    "var_from_hvg_frame": '''
def var_from_hvg_frame(adata, df):
    """Replace the variable-gene columns in var with one flavour's results."""
    for col in ("dispersions", "dispersions_norm", "variances", "variances_norm",
                "highly_variable_rank", "residual_variances"):
        if col in adata.var.columns and col not in df.columns:
            del adata.var[col]
    for col in df.columns:
        adata.var[col] = df[col].to_numpy()
    adata.var["highly_variable"] = adata.var["highly_variable"].astype(bool)
''',
    "counts_hvg": '''
def counts_hvg(adata, flavor, n_top_genes, batch_key, theta=None):
    """Variable genes ranked on the counts (seurat_v3, or Pearson residual variance)."""
    keep_raw_counts(adata)
    tmp = ad.AnnData(X=find_counts(adata), obs=adata.obs[[batch_key]] if batch_key else None)
    tmp.var_names = adata.var_names
    n = min(n_top_genes, adata.n_vars)
    if flavor == "pearson_residuals":
        df = sc.experimental.pp.highly_variable_genes(
            tmp, flavor="pearson_residuals", n_top_genes=n, batch_key=batch_key, theta=theta,
            inplace=False)
    else:
        df = sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=n,
                                         batch_key=batch_key, inplace=False)
    var_from_hvg_frame(adata, df)
''',
    "pearson_residuals": '''
def pearson_residuals(counts, theta, clip, chunk=1000, columns=None):
    """Analytic Pearson residuals (float32), a block of genes at a time: the values
    of sc.experimental.pp.normalize_pearson_residuals without its dense copies.
    ``columns``: only those genes' residuals (the totals still over every gene)."""
    n, g = counts.shape
    cells = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    genes = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    total = float(genes.sum())
    if columns is not None:
        counts, genes, g = counts[:, columns], genes[columns], len(columns)
    out = np.empty((n, g), dtype=np.float32)
    for j0 in range(0, g, chunk):
        j1 = min(g, j0 + chunk)
        C = counts[:, j0:j1]
        C = C.toarray() if sp.issparse(C) else np.asarray(C)
        mu = np.outer(cells, genes[j0:j1] / total) if total > 0 else np.zeros(C.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = (C - mu) / np.sqrt(mu + mu**2 / theta)
        R[~np.isfinite(R)] = 0.0
        np.clip(R, -clip, clip, out=R)
        out[:, j0:j1] = R
    return out
''',
    "per_cell_qc": '''
QC_OBS_NAMES = {
    "total_counts": ("total_counts", "n_counts", "nCount_RNA"),
    "n_genes": ("n_genes_by_counts", "n_genes", "nFeature_RNA"),
    "pct_mt": ("pct_counts_mt", "percent.mt", "pct_counts_mito", "percent_mito"),
}


def per_cell_qc(adata):
    """Counts, genes and % mito per cell: from obs when present, else from the counts.
    pct_mt is None when it cannot be measured (no MT- genes)."""
    out = {}
    for key, names in QC_OBS_NAMES.items():
        col = next((n for n in names if n in adata.obs.columns), None)
        out[key] = None if col is None else adata.obs[col].to_numpy(dtype=np.float64)
    var_names = adata.var_names
    has_mt_genes = bool(var_names.str.startswith(("MT-", "mt-")).any()) or (
        adata.raw is not None and bool(adata.raw.var_names.str.startswith(("MT-", "mt-")).any()))
    if any(out[k] is None for k in out):
        cands = []
        if "counts" in adata.layers:
            cands.append((adata.layers["counts"], adata.var_names))
        if adata.raw is not None:
            cands.append((adata.raw.X, adata.raw.var_names))
        cands.append((adata.X, adata.var_names))
        M, names = next(((m, nm) for m, nm in cands if looks_like_counts(m)), cands[-1])
        tc = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
        ng = np.asarray((M != 0).sum(axis=1), dtype=np.float64).ravel()
        mt = np.asarray(names.str.startswith("MT-") | names.str.startswith("mt-"))
        mt_counts = np.asarray(M[:, mt].sum(axis=1), dtype=np.float64).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(tc > 0, mt_counts / tc * 100.0, 0.0)
        if out["total_counts"] is None:
            out["total_counts"] = tc
        if out["n_genes"] is None:
            out["n_genes"] = ng
        if out["pct_mt"] is None:
            out["pct_mt"] = pct if mt.any() else None
    elif not has_mt_genes and not np.any(out["pct_mt"] > 0):
        out["pct_mt"] = None  # a column of zeros without MT- genes measures nothing
    return out
''',
    "doublet_calls": '''
def doublet_calls(obs, source="legacy"):
    """Per-cell doublet calls as booleans from the source scView's filtering used
    (core/doublet_source.py): "hashing" (hash_class), "scrublet" (predicted_doublet),
    "classifier" (predicted_doublet_classifier), "consensus" (both detectors; the one
    that ran alone), "either" (any of hashing, Scrublet, the classifier), "auto"
    (hashing, else the classifier, else Scrublet, else a text column), a text column's
    name, or "legacy" (a filtering recorded before the choice existed: Scrublet's
    column first, then the text ones)."""
    def text_in(col, values):
        return col.astype(str).str.lower().isin(values).to_numpy()

    def flag(name):
        if name not in obs.columns:
            return None
        col = obs[name]
        return (col.to_numpy(dtype=bool) if pd.api.types.is_bool_dtype(col)
                else text_in(col, ("true", "1", "1.0", "doublet")))

    hashing = text_in(obs["hash_class"], ("doublet",)) if "hash_class" in obs.columns else None
    scrublet = flag("predicted_doublet")
    classifier = flag("predicted_doublet_classifier")
    named = [c for c in ("scDblFinder.class", "scDblFinder_class", "multiplets", "doublet_class")
             if c in obs.columns]
    finder = [c for c in obs.columns if str(c).startswith("DF.classification")]
    if source == "hashing" or (source == "auto" and hashing is not None):
        return hashing
    if source == "either":
        found = [c for c in (hashing, scrublet, classifier) if c is not None]
        return np.logical_or.reduce(found) if found else None
    if source == "scrublet":
        return scrublet
    if source == "classifier":
        return classifier
    if source == "consensus":
        if scrublet is not None and classifier is not None:
            return scrublet & classifier
        return scrublet if classifier is None else classifier
    if source == "auto" and (classifier is not None or scrublet is not None):
        return classifier if classifier is not None else scrublet
    if source == "legacy" and scrublet is not None:
        return scrublet
    if source == "legacy" and hashing is not None and not named:
        return hashing  # hash_class came after the named text columns, before DoubletFinder's
    if source not in ("auto", "legacy"):  # a text column another tool left
        return text_in(obs[source], ("doublet",)) if source in obs.columns else None
    texts = named + finder
    return text_in(obs[texts[0]], ("doublet",)) if texts else None


def hash_negatives(obs):
    """Cells cell hashing calls Negative (no tag above background), or None."""
    if "hash_class" not in obs.columns:
        return None
    return obs["hash_class"].astype(str).str.lower().eq("negative").to_numpy()
''',
    "mad_thresholds": '''
def mad_thresholds(adata, qc, column, k, k_mt, upper, ceiling, min_cells=100):
    """Each sample's QC bounds as scView works them out (Heumos et al. 2023): below
    median - k MADs of log1p counts / genes (above + k too when upper), above
    median + k_mt MADs of % mito, capped by ceiling. Small samples pool."""
    import math

    def median_mad(v):
        v = v[np.isfinite(v)]
        if not v.size:
            return math.nan, math.nan
        med = float(np.median(v))
        return med, float(np.median(np.abs(v - med)))

    values = {"total_counts": np.log1p(qc["total_counts"]), "n_genes": np.log1p(qc["n_genes"])}
    if qc["pct_mt"] is not None:
        values["pct_mt"] = qc["pct_mt"]

    def bounds(stats):
        out = {}
        for key, below, above in (("total_counts", "min_counts", "max_counts"),
                                  ("n_genes", "min_genes", "max_genes")):
            med, mad = stats[key]
            on = mad > 0
            out[below] = max(0.0, math.expm1(med - k * mad)) if on else None
            out[above] = math.expm1(med + k * mad) if on and upper else None
        mt = None
        if "pct_mt" in stats:
            med, mad = stats["pct_mt"]
            mt = med + k_mt * mad if mad > 0 else math.inf
            mt = min(mt, ceiling) if ceiling is not None else mt
        out["max_pct_mt"] = mt if mt is not None and math.isfinite(mt) else None
        return out

    pooled = bounds({m: median_mad(v) for m, v in values.items()})
    samples = []
    if column and column in adata.obs.columns:
        col = adata.obs[column]
        labels = col.astype(str).where(col.notna())
        for name in labels.dropna().unique():
            m = (labels == name).to_numpy()
            own = m.sum() >= min_cells
            samples.append({"name": str(name), **(bounds({key: median_mad(v[m])
                            for key, v in values.items()}) if own else pooled)})
    return {"column": column, "upper": upper, "ceiling_pct_mt": ceiling if "pct_mt" in values
            else None, "samples": samples, "pooled": pooled}
''',
    "apply_thresholds": '''
def mad_drop(adata, qc, rec):
    """The cells outside their sample's recorded bounds (pooled bounds for cells
    without a sample, or of a sample not listed)."""
    n = adata.n_obs
    rows = {str(r["name"]): r for r in rec.get("samples") or []}
    pooled = rec["pooled"]
    column = rec.get("column")
    if column and column in adata.obs.columns:
        col = adata.obs[column]
        labels = col.astype(str).where(col.notna())
        per_cell = [pooled if s is None or s != s else rows.get(s, pooled) for s in labels]
    else:
        per_cell = [pooled] * n

    def bound(key, missing, log):
        b = np.array([np.nan if r.get(key) is None else float(r[key]) for r in per_cell])
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log1p(b) if log else b
        return np.where(np.isnan(b), missing, v)

    drop = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        for metric, below, above in (("total_counts", "min_counts", "max_counts"),
                                     ("n_genes", "min_genes", "max_genes")):
            v = np.log1p(qc[metric])
            ok = np.isfinite(v)
            drop |= ok & (v < bound(below, -np.inf, True))
            if rec.get("upper", True):
                drop |= ok & (v > bound(above, np.inf, True))
        if qc["pct_mt"] is not None:
            pct = qc["pct_mt"]
            ok = np.isfinite(pct)
            drop |= ok & (pct > bound("max_pct_mt", np.inf, False))
            if rec.get("ceiling_pct_mt") is not None:
                drop |= ok & (pct >= float(rec["ceiling_pct_mt"]))
    return drop
''',
    "fixed_drop": '''
def fixed_drop(qc, min_genes, max_genes, min_counts, max_pct_mt):
    """The cells outside fixed cut-offs (0 or None turns one off)."""
    genes, counts = qc["n_genes"], qc["total_counts"]
    drop = np.zeros(len(genes), dtype=bool)
    with np.errstate(invalid="ignore"):
        if min_genes:
            drop |= np.isfinite(genes) & (genes < min_genes)
        if max_genes:
            drop |= np.isfinite(genes) & (genes > max_genes)
        if min_counts:
            drop |= np.isfinite(counts) & (counts < min_counts)
        if max_pct_mt is not None and qc["pct_mt"] is not None:
            pct = qc["pct_mt"]
            drop |= np.isfinite(pct) & (pct >= max_pct_mt)
    return drop
''',
    "type_mito_drop": '''
def mito_only(qc):
    """The QC values with counts and genes blanked: a drop helper then judges % mito alone."""
    nan = np.full(len(qc["n_genes"]), np.nan)
    return {**qc, "total_counts": nan, "n_genes": nan}


def type_mito_drop(adata, qc, rule, global_mito):
    """% mito per cell type, as scView's per-type filter: a type with a recorded bound
    loses its cells above it (or at the cap); the other cells keep the global rule."""
    if qc["pct_mt"] is None:
        return np.zeros(adata.n_obs, dtype=bool)
    col = adata.obs[rule["column"]]
    labels = col.astype(str).where(col.notna())
    bounds = {str(r["name"]): float(r["bound"]) for r in rule["types"] if r.get("bound") is not None}
    hi = labels.map(bounds).to_numpy(dtype=np.float64)  # NaN: no bound of its own
    pct = qc["pct_mt"]
    with np.errstate(invalid="ignore"):
        mine = np.isfinite(pct) & ((pct > hi) | (pct >= rule["cap"]))
    return np.where(~np.isnan(hi), mine, global_mito)
''',
    "refilter_types": '''
def refilter_types(full, pass1, column, target_sum=1e4, n_genes=2000, min_genes=50):
    """Each cell's type for scView's per-type % mito filter, as scView gives it: a
    cell pass 1 kept takes its label by barcode; one it removed, the group whose
    centroid (mean log-normalised expression of the group's cells over the variable
    genes, MT- genes left out) its own expression is most correlated with."""
    def lognorm(C):
        C = sp.csr_matrix(C).astype(np.float64)
        total = np.asarray(C.sum(axis=1), dtype=np.float64).ravel()
        scale = np.divide(target_sum, total, out=np.zeros_like(total), where=total > 0)
        L = sp.csr_matrix(sp.diags(scale) @ C)
        L.data = np.log1p(L.data)
        return L

    if pass1.raw is not None and not looks_like_counts(pass1.raw.X):
        L, names = pass1.raw.X, pass1.raw.var_names
    elif not looks_like_counts(pass1.X) and float(pass1.X.min()) >= 0:
        L, names = pass1.X, pass1.var_names
    else:
        L, names = lognorm(pass1.layers["counts"] if "counts" in pass1.layers else pass1.X), pass1.var_names
    if "counts" in full.layers and looks_like_counts(full.layers["counts"]):
        counts = full.layers["counts"]
    elif looks_like_counts(full.X):
        counts = full.X
    else:
        counts = full.raw[:, full.var_names].X
    no_mt = ~names.str.startswith(("MT-", "mt-"))
    var = pass1.var
    if "highly_variable" in var.columns and var["highly_variable"].astype(bool).any():
        hv = set(var.index[var["highly_variable"].astype(bool).to_numpy()])
        genes = [g for g, ok in zip(names, no_mt) if ok and g in hv]
    else:
        M = sp.csr_matrix(L)
        mean = np.asarray(M.mean(axis=0)).ravel()
        sq = np.asarray(M.multiply(M).mean(axis=0)).ravel()
        genes = [names[i] for i in np.argsort(-(sq - mean**2), kind="stable") if no_mt[i]][:n_genes]
    have = set(full.var_names)
    genes = [g for g in genes if g in have]
    col = pass1.obs[column]
    if col.dtype.name == "category":
        used = set(col.dropna().astype(str))
        groups = [str(c) for c in col.cat.categories if str(c) in used]
    else:
        present = set(col.dropna().astype(str))
        groups = sorted(present, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
    labels = pd.Series("", index=full.obs_names, dtype=object)
    labels.loc[pass1.obs_names] = col.astype(str).where(col.notna(), "").to_numpy()
    gone = np.flatnonzero(~full.obs_names.isin(pass1.obs_names))
    if gone.size and len(genes) >= min_genes:
        codes = pd.Categorical(col.astype(str).where(col.notna()), categories=groups).codes
        gi = names.get_indexer(genes)
        Lg = L[:, gi]
        lab = codes >= 0
        ind = sp.csr_matrix((np.ones(int(lab.sum())), (codes[lab], np.flatnonzero(lab))),
                            shape=(len(groups), pass1.n_obs))
        sizes = np.asarray(ind.sum(axis=1)).ravel()
        sums = ind @ Lg
        cent = (sums.toarray() if sp.issparse(sums) else np.asarray(sums)) / np.maximum(sizes, 1)[:, None]
        R = lognorm(sp.csr_matrix(counts)[gone])[:, full.var_names.get_indexer(genes)].toarray()

        def unit(M):
            M = M - M.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(M, axis=1, keepdims=True)
            return np.divide(M, norm, out=np.zeros_like(M), where=norm > 0)

        r = unit(R) @ unit(cent).T
        r[:, sizes == 0] = -np.inf
        best = r.argmax(axis=1)
        ok = np.isfinite(r[np.arange(gone.size), best]) & (np.abs(R).sum(axis=1) > 0)
        labels.iloc[gone] = np.where(ok, np.asarray(groups, dtype=object)[best], "")
    values = labels.to_numpy()
    return pd.Categorical(np.where(values == "", None, values), categories=groups)
''',
    "markers_key": '''
def markers_key(column):
    """Where scView stores a column's markers in uns."""
    return "rank_genes_groups__" + str(column).replace("/", "%2F")


def markers_for(adata, column):
    """Wilcoxon markers of one obs column (2-100 groups), as scView stores them."""
    if column not in adata.obs.columns or not 2 <= adata.obs[column].nunique() <= 100:
        return
    if adata.obs[column].dtype.name != "category":
        adata.obs[column] = adata.obs[column].astype(str).astype("category")
    sc.tl.rank_genes_groups(adata, groupby=column, method="wilcoxon", pts=True,
                            key_added=markers_key(column))
''',
    "louvain": '''
def louvain_igraph(adata, resolution, key_added, seed=0):
    """Louvain on the neighbour graph with igraph's multilevel method, which honours
    the resolution; clusters numbered by size, largest first (as scView does)."""
    import random

    import igraph as ig

    adj = sp.triu(adata.obsp["connectivities"], k=1).tocoo()
    g = ig.Graph(n=adata.n_obs, edges=np.column_stack([adj.row, adj.col]).tolist(), directed=False)
    g.es["weight"] = adj.data.tolist()
    ig.set_random_number_generator(random.Random(seed))
    try:
        membership = np.asarray(g.community_multilevel(weights="weight",
                                                       resolution=resolution).membership)
    finally:
        ig.set_random_number_generator(random)
    order = np.argsort(-np.bincount(membership), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    labels = rank[membership].astype(str)
    adata.obs[key_added] = pd.Categorical(labels, categories=[str(i) for i in range(order.size)])
''',
    "harmony": '''
def harmony(X, obs, batch_key):
    """Harmony-corrected PCs (cells x PCs) from harmonypy, either orientation."""
    import harmonypy

    Z = np.asarray(harmonypy.run_harmony(X, obs, [batch_key]).Z_corr)
    if Z.shape[0] != X.shape[0] and Z.ndim == 2 and Z.shape[1] == X.shape[0]:
        Z = Z.T
    return np.ascontiguousarray(Z)
''',
    "scanorama": '''
def scanorama(adata, batch_key):
    """Scanorama's corrected PCs (sc.external.pp.scanorama_integrate with exact neighbours,
    as scView ran it; pip install scanorama) for batches in any cell order: scanpy's
    wrapper wants each batch's cells together, so they are gathered and put back."""
    b = adata.obs[batch_key].astype(str).to_numpy()
    names, first = np.unique(b, return_index=True)
    order = np.concatenate([np.flatnonzero(b == n) for n in names[np.argsort(first)]])
    tmp = ad.AnnData(X=np.zeros((b.size, 1), np.float32),  # the wrapper reads X.shape
                     obs=pd.DataFrame({batch_key: b[order]}, index=adata.obs_names[order]))
    tmp.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])[order]
    sc.external.pp.scanorama_integrate(tmp, batch_key, approx=False, verbose=0)
    Z = np.empty(tmp.obsm["X_scanorama"].shape)
    Z[order] = tmp.obsm["X_scanorama"]
    return Z
''',
    "lognorm_scoring": '''
def lognorm_scoring(adata):
    """(use_raw, gene names) of the log-normalised matrix: .raw after scaling, else X."""
    if adata.raw is not None and not looks_like_counts(adata.raw.X):
        return True, list(adata.raw.var_names)
    if not looks_like_counts(adata.X):
        return False, list(adata.var_names)
    raise ValueError("Scoring needs log-normalised expression.")


def match_genes(genes, names, aliases=None):
    """The dataset's own spelling of each gene it has, ignoring case."""
    lookup = {}
    for n in names:
        lookup.setdefault(str(n).upper(), str(n))
    out = []
    for g in genes:
        for cand in (g, *(aliases or {}).get(g, ())):
            hit = lookup.get(cand.upper())
            if hit is not None:
                if hit not in out:
                    out.append(hit)
                break
    return out
''',
    "subcluster": '''
def subcluster(adata, column, group, resolution, batch_key=None, n_top_genes=2000, n_pcs=30,
               n_neighbors=15, seed=0, n_iterations=-1):
    """Split one group of a column on its own graph into <column>_sub, as scView does:
    log-normalised expression of the group's cells, Seurat HVGs, scaled PCA,
    Harmony when the dataset was corrected, a neighbour graph and Leiden
    (two passes for fast Leiden, when the dataset's clustering was made that way)."""
    labels = adata.obs[column]
    mask = (labels.astype(str).where(labels.notna()) == group).to_numpy()
    n = int(mask.sum())

    def from_counts(X, names):
        s = ad.AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=names))
        sc.pp.normalize_total(s, target_sum=1e4)
        sc.pp.log1p(s)
        return s

    if "counts" in adata.layers:
        sub = from_counts(adata.layers["counts"][mask], adata.var_names)
    else:
        sources = [(adata.raw.X, adata.raw.var_names)] if adata.raw is not None else []
        for X, names in [*sources, (adata.X, adata.var_names)]:
            X = X[mask]
            if looks_like_counts(X):
                sub = from_counts(X, names)
                break
            if float(X.min()) >= 0:
                sub = ad.AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=names))
                break
    sub.obs_names = adata.obs_names[mask]
    use_batch = bool(batch_key and batch_key in adata.obs.columns
                     and adata.obs.loc[mask, batch_key].nunique() >= 2)
    if use_batch:
        sub.obs[batch_key] = adata.obs.loc[mask, batch_key].astype(str).to_numpy()
    sc.pp.highly_variable_genes(sub, n_top_genes=min(n_top_genes, sub.n_vars), flavor="seurat")
    sub = sub[:, sub.var["highly_variable"]].copy()
    sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=max(2, min(n_pcs, n - 1, sub.n_vars - 1)), random_state=seed)
    rep = "X_pca"
    if use_batch:
        sub.obsm["X_pca_harmony"] = harmony(sub.obsm["X_pca"], sub.obs, batch_key)
        rep = "X_pca_harmony"
    sc.pp.neighbors(sub, n_neighbors=min(n_neighbors, n - 1), use_rep=rep, random_state=seed)
    sc.tl.leiden(sub, resolution=resolution, key_added="s", random_state=seed,
                 flavor="leidenalg", n_iterations=n_iterations)
    codes = sub.obs["s"].astype(str).to_numpy()
    new = [f"{group}.{k}" for k in sorted(set(codes), key=int)]
    target = column if column.endswith("_sub") else f"{column}_sub"
    base = adata.obs[target] if target in adata.obs.columns else labels
    values = base.astype(str).where(base.notna()).astype(object).to_numpy()
    values[mask] = [f"{group}.{k}" for k in codes]
    cats = list(base.cat.categories) if hasattr(base, "cat") else sorted(base.dropna().unique())
    order = []
    for c in map(str, cats):
        if c == group or c.startswith(f"{group}."):
            if new[0] not in order:
                order.extend(new)
        else:
            order.append(c)
    order += [c for c in new if c not in order]
    adata.obs[target] = pd.Categorical(values, categories=order)
''',
    "gene_programs": """
def gene_programs(adata, genes, k, seeds, source="raw", solver="cd", init="random",
                  beta_loss="frobenius", tol=1e-4, max_iter=1000, kmeans_seed=1, kmeans_starts=10,
                  target_sum=1e4):
    \"\"\"Consensus NMF as scView ran it: the log-normalised variable genes, each divided
    by its standard deviation; one NMF per seed; k-means of the unit-length spectra and
    each group's median, scaled to sum 1; usages refitted with those spectra fixed and
    scaled to sum 1 per cell; programs ordered by mean usage share. Returns
    (usage shares, usages, spectra), cells x k, cells x k, k x genes.\"\"\"
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF, non_negative_factorization

    if source == "raw":
        L = adata.raw[:, genes].X
    elif source == "X":
        L = adata[:, genes].X
    else:  # the shifted log of the counts
        C = find_counts(adata)
        totals = np.asarray(C.sum(axis=1), dtype=np.float64).ravel()
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        cols = adata.var_names.get_indexer(genes)
        L = sp.csr_matrix(sp.diags(scale.astype(np.float32)) @ sp.csr_matrix(C[:, cols], dtype=np.float32))
        L.data = np.log1p(L.data)
    X = sp.csr_matrix(L, dtype=np.float32)
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0), dtype=np.float64).ravel()
    sd = np.sqrt(np.maximum(sq - mean * mean, 0.0) * (n / max(n - 1, 1)))
    X = sp.csr_matrix(X @ sp.diags((1.0 / sd).astype(np.float32)), dtype=np.float32)
    X.eliminate_zeros()
    spectra = []
    for seed in seeds:
        model = NMF(n_components=k, init=init, solver=solver, beta_loss=beta_loss, tol=tol,
                    max_iter=max_iter, random_state=seed)
        model.fit_transform(X)
        spectra.append(np.asarray(model.components_, dtype=np.float64))
    S = np.vstack(spectra)
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    U = np.divide(S, norms, out=np.zeros_like(S), where=norms > 0)
    labels = (KMeans(n_clusters=k, n_init=kmeans_starts, random_state=kmeans_seed).fit_predict(U)
              if len(seeds) > 1 else np.arange(k))
    H = np.zeros((k, S.shape[1]))
    for c in range(k):
        if (labels == c).any():
            H[c] = np.median(U[labels == c], axis=0)
    tot = H.sum(axis=1, keepdims=True)
    H = np.divide(H, tot, out=np.zeros_like(H), where=tot > 0)
    W, _, _ = non_negative_factorization(X, H=H.astype(X.dtype), n_components=k, init=None,
                                         update_H=False, solver=solver, beta_loss=beta_loss, tol=tol,
                                         max_iter=max_iter)
    W = np.asarray(W, dtype=np.float64)
    tot = W.sum(axis=1, keepdims=True)
    share = np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)
    order = np.argsort(-share.mean(axis=0), kind="stable")
    return share[:, order], W[:, order], H[order]
""",
}

_HELPERS["infer_cnv"] = """
def infer_cnv(adata, genes, chromosomes, ref_column, ref_groups, window=100, step=10,
              threshold=1.5, lfc_clip=3.0, source="raw", target_sum=1e4, n_pcs=50,
              n_neighbors=15, resolution=1.0, seed=0, chunk=1000):
    \"\"\"Copy number from expression as scView inferred it (inferCNV-style, as infercnvpy):
    the genes in genomic order; log values minus the reference groups' means (bounded
    difference between the groups' lowest and highest mean), clipped; a pyramid-weighted
    running mean over `window` genes per chromosome, every `step`th window; each cell's
    median subtracted; values under `threshold` SD of all values set to 0. Returns
    (cells x windows, per-cell score, CNV clusters).\"\"\"
    if source == "raw":
        M, names = adata.raw.X, adata.raw.var_names
    elif source == "X":
        M, names = adata.X, adata.var_names
    elif adata.raw is not None and adata.raw.n_vars > adata.n_vars and looks_like_counts(adata.raw.X):
        M, names = adata.raw.X, adata.raw.var_names
    else:
        M, names = find_counts(adata), adata.var_names
    cols = pd.Index(names).get_indexer(genes)

    def block(s, e):
        B = M[s:e]
        if source == "counts":
            B = sp.csr_matrix(B, dtype=np.float32)
            tot = np.asarray(B.sum(axis=1), dtype=np.float64).ravel()
            scale = np.divide(target_sum, tot, out=np.zeros_like(tot), where=tot > 0).astype(np.float32)
            return np.log1p(B[:, cols].toarray() * scale[:, None])
        B = B[:, cols]
        return (B.toarray() if sp.issparse(B) else np.asarray(B)).astype(np.float32)

    chroms = np.asarray(chromosomes, dtype=object)
    rows, cols_w, vals = [], [], []
    bounds = np.flatnonzero(np.r_[True, chroms[1:] != chroms[:-1], True])
    j = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = b - a
        if window < n:
            r = np.arange(1, window + 1, dtype=np.float64)
            w = np.minimum(r, r[::-1]) / np.minimum(r, r[::-1]).sum()
            starts = np.arange(0, n - window + 1, step)
        else:
            w, starts = np.full(n, 1.0 / n), np.array([0])
        for st in starts:
            rows.append(np.arange(a + st, a + st + len(w)))
            cols_w.append(np.full(len(w), j))
            vals.append(w)
            j += 1
    WT = sp.csr_matrix((np.concatenate(vals), (np.concatenate(cols_w), np.concatenate(rows))),
                       shape=(j, len(genes)))
    lab = adata.obs[ref_column].astype(str).to_numpy()
    sums = np.zeros((len(ref_groups), len(genes)))
    sizes = np.zeros(len(ref_groups))
    for s in range(0, adata.n_obs, chunk):
        B = block(s, s + chunk)
        for i, g in enumerate(ref_groups):
            m = lab[s:s + chunk] == g
            sums[i] += B[m].sum(axis=0, dtype=np.float64)
            sizes[i] += m.sum()
    ref = (sums / np.maximum(sizes, 1)[:, None]).astype(np.float32)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    X = np.empty((adata.n_obs, j), dtype=np.float32)
    for s in range(0, adata.n_obs, chunk):
        B = block(s, s + chunk)
        if len(ref_groups) == 1:
            D = B - ref[0]
        else:
            D = np.zeros_like(B)
            np.subtract(B, hi, out=D, where=B > hi)
            np.subtract(B, lo, out=D, where=B < lo)
        np.clip(D, -lfc_clip, lfc_clip, out=D)
        S = np.asarray(WT @ D.T.astype(np.float64)).T
        X[s:s + chunk] = S - np.median(S, axis=1, keepdims=True)
    if threshold:
        X[np.abs(X) < threshold * float(X.astype(np.float64).std())] = 0
    score = np.abs(X).mean(axis=1)
    tmp = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float64))
    k = max(1, min(n_pcs, X.shape[1] - 1, X.shape[0] - 1))
    sc.tl.pca(tmp, n_comps=k, svd_solver="arpack", zero_center=False, random_state=seed)
    sc.pp.neighbors(tmp, n_neighbors=min(n_neighbors, X.shape[0] - 1), use_rep="X_pca", random_state=seed)
    sc.tl.leiden(tmp, resolution=resolution, random_state=seed, key_added="cnv_leiden")
    return X, score, tmp.obs["cnv_leiden"].astype(str).to_numpy()
"""

_HELPERS["read_hashtags"] = '''
def read_hashtags(path, cells, features, sources):
    """The tag counts scView demultiplexed, read again from the input file (cells x
    tags, in the call's order): an .h5ad's tag table in obsm or its features in X,
    a 10x .h5 or matrix folder's features of any type (read_input keeps genes only)."""
    p = str(path)
    tags = list(features)
    if p.endswith(".h5ad"):
        full = ad.read_h5ad(p, backed="r")
        out = {}
        for t in tags:
            src = str(sources.get(t) or "")
            keys = ([src[5:]] if src.startswith("obsm ") else []) + [
                "hto_counts", "protein_counts", "protein_expression", "protein", "adt", "ADT"]
            for key in keys:
                table = full.obsm[key] if key in full.obsm else None
                name = next((n for n in (t, features[t]) if isinstance(table, pd.DataFrame)
                             and n in table.columns), None)
                if name is not None:
                    out[t] = table[name].to_numpy(dtype=np.float64)
                    break
            else:  # a feature in X (10x feature types read into an .h5ad)
                col = full.X[:, [full.var_names.get_loc(features[t])]]
                out[t] = (col.toarray() if sp.issparse(col) else np.asarray(col)).ravel().astype(np.float64)
        names = pd.Index(full.obs_names)
    else:
        if p.endswith(".h5"):
            full = sc.read_10x_h5(p, gex_only=False)
        else:
            full = sc.read_10x_mtx(p, var_names="gene_symbols", gex_only=False)
        full.var_names_make_unique()
        out = {}
        for t in tags:
            col = full[:, features[t]].X
            out[t] = (col.toarray() if sp.issparse(col) else np.asarray(col)).ravel().astype(np.float64)
        names = pd.Index(full.obs_names)
    if not names.is_unique:
        names = pd.Index(ad.utils.make_index_unique(names))
    frame = pd.DataFrame(out, index=names)
    return frame.loc[list(cells), tags].to_numpy(dtype=np.float64)
'''


def _hashing_helper() -> str:
    """scView's own demultiplexing code (core/hashing.py: the HTODemux and
    hashedDrops ports), copied into the script so the replay computes exactly
    what scView did, without importing scView."""
    import inspect

    from scview.core import hashing

    lines = ["import math", "from dataclasses import dataclass, field", "from typing import Any", ""]
    lines += [f"{name} = {_lit(getattr(hashing, name))}" for name in hashing.REPLAY_CONSTANTS]
    for obj in hashing.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["hashing"] = _hashing_helper()


# The V(D)J attach: core/vdj.py's own functions with their source written in (they use
# only numpy, pandas and the standard library), so the script repeats it exactly.
_HELPERS["vdj"] = "\nimport gzip\nimport os\nimport re\n\n\n" + vdj.replay_source() + '''

def vdj_file(name):
    """A contig table by name: in the working directory, else beside the input file."""
    for folder in (".", os.path.dirname(os.path.abspath(INPUT))):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    sys.exit(f"Contig table {name} not found: put it beside this script or the input file.")
'''

# The order helpers are written in (later ones call earlier ones).
_HELPER_ORDER = ("looks_like_counts", "keep_raw_counts", "find_counts", "var_from_hvg_frame",
                 "counts_hvg", "pearson_residuals", "per_cell_qc", "doublet_calls",
                 "mad_thresholds", "apply_thresholds", "fixed_drop", "type_mito_drop", "refilter_types",
                 "markers_key", "louvain",
                 "harmony", "scanorama", "lognorm_scoring", "subcluster", "gene_programs", "infer_cnv",
                 "read_hashtags", "hashing", "vdj")

# Helpers another helper calls.
_HELPER_NEEDS = {
    "keep_raw_counts": ("looks_like_counts",),
    "find_counts": ("looks_like_counts",),
    "counts_hvg": ("keep_raw_counts", "find_counts", "var_from_hvg_frame"),
    "per_cell_qc": ("looks_like_counts",),
    "lognorm_scoring": ("looks_like_counts",),
    "subcluster": ("looks_like_counts", "harmony"),
    "refilter_types": ("looks_like_counts",),
    "gene_programs": ("find_counts",),
    "infer_cnv": ("looks_like_counts", "find_counts"),
}


def _scrublet_helper() -> str:
    """core/scrublet_lean.py's code (Scrublet on the counts alone, the scores
    sc.pp.scrublet gives without its copies of the dataset), copied into the script so
    the replay scores what scView scored, in the same memory, without importing scView."""
    import inspect

    from scview.core import scrublet_lean

    lines = [f"{name} = {_lit(getattr(scrublet_lean, name))}" for name in scrublet_lean.REPLAY_CONSTANTS]
    for obj in scrublet_lean.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["scrublet_counts"] = _scrublet_helper()


def _barcode_channels_helper() -> str:
    """core/doublets.py's reading of the capture channels from 10x barcodes (numpy,
    pandas and re only), copied so the script finds the channels scView found."""
    import inspect

    from scview.core import doublets

    lines = ["import re", ""]
    lines += [f"{name} = {_lit(getattr(doublets, name))}" for name in doublets.REPLAY_CONSTANTS]
    for obj in doublets.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["barcode_channels"] = _barcode_channels_helper()


def _classifier_helper() -> str:
    """core/doublet_classifier.py's code (the scDblFinder-style classifier: numpy, scipy
    and scikit-learn), copied so the script scores what scView scored."""
    import inspect

    from scview.core import doublet_classifier

    lines = [f"{name} = {_lit(getattr(doublet_classifier, name))}"
             for name in doublet_classifier.REPLAY_CONSTANTS]
    for obj in doublet_classifier.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["doublet_classifier"] = _classifier_helper()
_HELPER_NEEDS["doublet_classifier"] = ("scrublet_counts",)  # channel_rows
_HELPER_ORDER += ("scrublet_counts", "barcode_channels", "doublet_classifier")


def _ambient_helper() -> str:
    """core/ambient.py's soup reading and count subtraction (numpy, scipy, pandas, h5py,
    anndata only), copied so the script removes what scView removed; the soup is read
    again from each sample's raw matrix, found beside the script or the input file."""
    import inspect

    from scview.core import ambient

    lines = ["import os", "", "import h5py", ""]
    lines += [f"{name} = {_lit(getattr(ambient, name))}" for name in ambient.REPLAY_CONSTANTS]
    for obj in ambient.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def raw_soup(files):
    """The soup of a sample's raw matrix (one .h5, or the MEX matrix, barcodes, features)."""
    paths = []
    for name in files:
        for folder in (".", os.path.dirname(os.path.abspath(INPUT))):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                paths.append(path)
                break
        else:
            sys.exit(f"Raw matrix {name} not found: put it beside this script or the input file.")
    if len(paths) == 1:
        return soup_from_h5(paths[0], SOUP_UMI_MAX)
    return soup_from_mex(*paths, umi_max=SOUP_UMI_MAX)
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["ambient"] = _ambient_helper()
_HELPER_ORDER += ("ambient",)


def _label_transfer_helper() -> str:
    """core/label_transfer.py's mapping code (numpy, scipy, pandas, anndata, scanpy,
    scikit-learn, harmonypy; celltypist for that method), copied so the script maps the
    cells as scView did; the reference is read from its file beside the script or input."""
    import inspect

    from scview.core import label_transfer

    lines = ["import os", ""]
    lines += [f"{name} = {_lit(getattr(label_transfer, name))}" for name in label_transfer.REPLAY_CONSTANTS]
    for obj in label_transfer.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def reference_file(name):
    """A reference dataset's file by name: in the working directory, else beside the input."""
    for folder in (".", os.path.dirname(os.path.abspath(INPUT))):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    sys.exit(f"Reference file {name} not found: export the reference dataset from scView as .h5ad "
             "and put it beside this script or the input file.")
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["label_transfer"] = _label_transfer_helper()
_HELPER_ORDER += ("label_transfer",)


def _learned_gate_helper() -> str:
    """core/learned_gate.py's scoring code (numpy, scipy, pandas), copied so the script
    scores the cells as scView did; the gate itself is written inline at its step."""
    import inspect

    from scview.core import learned_gate

    lines: list[str] = []
    for obj in learned_gate.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    return "\n".join(lines).strip("\n") + "\n"


_HELPERS["learned_gate"] = _learned_gate_helper()
_HELPER_ORDER += ("learned_gate",)


def _dsb_helper() -> str:
    """core/dsb.py's empty-droplet reading and DSB maths (numpy, pandas, h5py only), copied
    so the script normalises the antibodies as scView did; the antibody counts are read
    again from the input, the empty droplets from each sample's raw matrix."""
    import inspect

    from scview.core import dsb

    lines = ["import math", "import os", "", "import h5py", ""]
    lines += [f"{name} = {_lit(getattr(dsb, name))}" for name in dsb.REPLAY_CONSTANTS]
    for obj in dsb.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def raw_paths(files):
    """A sample's raw matrix files (one .h5, or the MEX matrix, barcodes, features)."""
    paths = []
    for name in files:
        for folder in (".", os.path.dirname(os.path.abspath(INPUT))):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                paths.append(path)
                break
        else:
            sys.exit(f"Raw matrix {name} not found: put it beside this script or the input file.")
    return paths


def input_cells():
    """The input's cells (excluded from the empty droplets, as scView excluded the dataset's)."""
    p = str(INPUT)
    if p.endswith(".h5ad"):
        return list(ad.read_h5ad(p, backed="r").obs_names.astype(str))
    return list(read_input(p).obs_names.astype(str))


def protein_counts_for(adata, proteins, features):
    """The cells' antibody counts (cells x proteins, in the recorded order), read again from
    the input: an .h5ad's protein table in obsm or its antibody features in X, or a 10x
    file's Antibody Capture features (read_input keeps genes only)."""
    p = str(INPUT)
    if p.endswith(".h5ad"):
        full = ad.read_h5ad(p, backed="r")
        rows = full.obs_names.get_indexer(adata.obs_names)
        for key in ("protein_counts", "protein_expression", "protein", "adt", "ADT"):
            table = full.obsm[key] if key in full.obsm else None
            if isinstance(table, pd.DataFrame) and all(n in table.columns for n in proteins):
                return table[proteins].to_numpy(dtype=np.float64)[rows]
        full = full.to_memory()
    elif p.endswith(".h5"):
        full = sc.read_10x_h5(p, gex_only=False)
    else:
        full = sc.read_10x_mtx(p, var_names="gene_symbols", gex_only=False)
    if "feature_types" not in full.var.columns:
        sys.exit("The input has no antibody counts to normalise.")
    ab = full[:, full.var["feature_types"].astype(str).eq("Antibody Capture").to_numpy()]
    names = [str(n) for n in ab.var_names]
    cols = [names.index(f) for f in features]
    X = ab.X[:, cols]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return X[ab.obs_names.get_indexer(adata.obs_names)].astype(np.float64)
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["dsb"] = _dsb_helper()
_HELPER_ORDER += ("dsb",)


def _spatial_autocorr_helper() -> str:
    """core/spatial_autocorr.py's neighbour graph and Moran's I (numpy, scipy, pandas),
    copied so the script scores the genes as scView (= squidpy 1.6's defaults) did."""
    import inspect

    from scview.core import spatial_autocorr as sa

    lines = ["from scipy import stats", "from scipy.spatial import cKDTree", ""]
    lines += [f"{name} = {_lit(getattr(sa, name))}" for name in sa.REPLAY_CONSTANTS]
    for obj in sa.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def spatial_genes(adata, names, genes):
    """Column indices of the genes scored: the variable genes ("hvg"), all, or a list."""
    names = pd.Index([str(n) for n in names])
    if genes == "hvg":
        hv = adata.var["highly_variable"].to_numpy(dtype=bool)
        idx = names.get_indexer(adata.var_names[hv].astype(str))
        return np.sort(idx[idx >= 0])
    if genes == "all":
        return np.arange(len(names))
    idx = names.get_indexer(pd.Index([str(g) for g in genes]).unique())
    return idx[idx >= 0]


def spatial_moran(adata, M, names, genes, library_id, grid):
    """Moran's I per gene on the spots (one section's, or all), as squidpy's
    uns["moranI"]: genes as the index, I, pval_norm, var_norm, pval_norm_fdr_bh."""
    libs = (adata.obs["library_id"].astype(str).to_numpy() if "library_id" in adata.obs.columns
            else np.full(adata.n_obs, ""))
    rows = np.flatnonzero(libs == str(library_id)) if library_id else np.arange(adata.n_obs)
    idx = spatial_genes(adata, names, genes)
    M = M[rows][:, idx]
    W = spatial_weights(np.asarray(adata.obsm["spatial"])[rows, :2], libs[rows], grid)
    order = spatial_order(np.asarray(adata.obsm["spatial"])[rows, :2], libs[rows])
    table, _ = morans_i(W, M, np.asarray([str(n) for n in names])[idx], order)
    return pd.DataFrame({"I": table["I"].to_numpy(), "pval_norm": table["pval"].to_numpy(),
                         "var_norm": table["var_I"].to_numpy(),
                         "pval_norm_fdr_bh": table["pval_adj"].to_numpy()},
                        index=pd.Index(table["gene"].astype(str).to_numpy()))
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["spatial_autocorr"] = _spatial_autocorr_helper()
_HELPER_ORDER += ("spatial_autocorr",)


def _wnn_helper() -> str:
    """core/wnn.py's WNN code (numpy, scipy, pynndescent, umap-learn: Seurat's
    FindMultiModalNeighbors), copied so the script weights the modalities as scView did;
    the antibody counts are read again from the input (the DSB helper's reader)."""
    import inspect

    from scview.core import wnn

    lines = [f"{name} = {_lit(getattr(wnn, name))}" for name in wnn.REPLAY_CONSTANTS]
    for obj in wnn.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def wnn_clr(counts):
    """Centred log-ratio per cell over the whole panel (Seurat's CLR, margin = 2), float32."""
    x = np.asarray(counts, dtype=np.float64)
    denom = np.exp(np.log1p(x).sum(axis=1, keepdims=True) / x.shape[1])
    return np.log1p(x / denom).astype(np.float32)
'''.strip("\n")]
    return "\n".join(lines) + "\n"


def _spot_deconvolution_helper() -> str:
    """core/spot_deconvolution.py's model (numpy, scipy, pandas), copied so the script fits
    the spots as scView did; the reference is read from its file beside the script or input."""
    import inspect

    from scview.core import spot_deconvolution as sd

    lines = ["import math", "import os", "", "from scipy.optimize import nnls", "from scipy.special import gammaln", ""]
    lines += [f"{name} = {_lit(getattr(sd, name))}" for name in sd.REPLAY_CONSTANTS]
    for obj in sd.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def spot_reference_file(name):
    """The reference dataset's file by name: in the working directory, else beside the input."""
    for folder in (".", os.path.dirname(os.path.abspath(INPUT))):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    sys.exit(f"Reference file {name} not found: export the reference dataset from scView as .h5ad, "
             "name it so and put it beside this script or the input file.")


def spot_counts(adata):
    """Raw counts and their gene names, as scView looks: layers['counts'], .raw, then X."""
    if "counts" in adata.layers and looks_like_counts(adata.layers["counts"]):
        return adata.layers["counts"], adata.var_names
    if adata.raw is not None and looks_like_counts(adata.raw.X):
        return adata.raw.X, adata.raw.var_names
    if looks_like_counts(adata.X):
        return adata.X, adata.var_names
    sys.exit("No raw counts (layers['counts'], .raw or X): spot composition fits counts.")


def first_positions(names, wanted):
    first = {}
    for i, g in enumerate(map(str, names)):
        first.setdefault(g, i)
    missing = [g for g in wanted if g not in first]
    if missing:
        sys.exit(f"{len(missing)} of the recorded genes are missing, e.g. {missing[:3]}")
    return np.asarray([first[g] for g in wanted], dtype=np.int64)


def spot_composition(adata, ref, label, types, genes, ref_genes, fit_genes, s_e, method):
    """Each spot's shares of the reference's types on the recorded genes, written where
    scView writes them (obsm spot_deconv / spot_deconv_cells, obs spot_deconv_top(_share))."""
    RM, rnames = spot_counts(ref)
    SM, snames = spot_counts(adata)
    labels = ref.obs[label]
    codes = np.asarray(pd.Categorical(labels.astype(str).where(labels.notna(), None).to_numpy(),
                                      categories=types).codes, dtype=np.int64)
    totals = np.asarray(RM.sum(axis=1), dtype=np.float64).ravel()
    codes = np.where(totals > 0, codes, -1)
    mu, S, _ = sd_profiles(RM, totals, codes, len(types))
    mu = mu[:, first_positions(rnames, ref_genes)]
    spos = first_positions(snames, genes)
    Y = sp.csr_matrix(SM[:, spos] if sp.issparse(SM) else np.asarray(SM)[:, spos], dtype=np.float32)
    N = np.asarray(SM.sum(axis=1), dtype=np.float64).ravel()
    if "library_id" in adata.obs.columns:
        lib = adata.obs["library_id"]
        libs = lib.astype(str).where(lib.notna(), "section1").to_numpy(dtype=object).astype(str)
    else:
        libs = np.full(adata.n_obs, "section1", dtype=object).astype(str)
    out = None
    for event, data in sd_iter_deconvolve(mu, S, Y, N, libs, method, np.ones(len(genes), dtype=bool),
                                          genes=(np.arange(len(genes)), np.asarray(fit_genes)), s_e=s_e):
        if event == "fitted":
            out = data
    beta, fitted = out["beta"], out["fitted"]
    top_idx = np.where(fitted, np.nanargmax(np.where(np.isfinite(beta), beta, -1.0), axis=1), -1)
    top = np.where(fitted, np.asarray(types, dtype=object)[np.maximum(top_idx, 0)], "low counts").astype(str)
    adata.obsm["spot_deconv"] = beta.astype(np.float32)
    adata.obsm["spot_deconv_cells"] = out["cells"].astype(np.float32)
    present = [t for t in types if (top == t).any()]
    low = ["low counts"] if (top == "low counts").any() else []
    adata.obs["spot_deconv_top"] = pd.Categorical(top, categories=present + low)
    adata.obs["spot_deconv_top_share"] = np.where(
        fitted, beta[np.arange(beta.shape[0]), np.maximum(top_idx, 0)], np.nan).astype(np.float32)
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["wnn"] = _wnn_helper()
_HELPER_ORDER += ("wnn",)
_HELPER_NEEDS["wnn"] = ("dsb",)  # protein_counts_for
_HELPERS["spot_deconvolution"] = _spot_deconvolution_helper()
_HELPER_ORDER += ("spot_deconvolution",)
_HELPER_NEEDS["spot_deconvolution"] = ("looks_like_counts",)


def _spatial_domains_helper() -> str:
    """core/spatial_domains.py's graph, features, mixture, enrichment and niches (numpy,
    scipy, pandas, sklearn), copied so the script finds them as scView did."""
    import inspect

    from scview.core import spatial_domains as sd

    lines = ["from scipy.spatial import Delaunay, QhullError", "from sklearn.cluster import KMeans",
             "from sklearn.mixture import GaussianMixture"]
    for obj in sd.REPLAY_CODE:
        lines += ["", "", inspect.getsource(obj).rstrip("\n")]
    lines += ["", "", '''
def spatial_sections(adata):
    """Each observation's section (the whole dataset as one when none is recorded)."""
    if "library_id" in adata.obs.columns:
        return adata.obs["library_id"].astype(str).to_numpy()
    return np.full(adata.n_obs, "", dtype=object)
'''.strip("\n")]
    return "\n".join(lines) + "\n"


_HELPERS["spatial_domains"] = _spatial_domains_helper()
_HELPER_ORDER += ("spatial_domains",)
_HELPER_NEEDS["spatial_domains"] = ("spatial_autocorr",)  # knn_edges


def _need(ctx: _Ctx, *names: str) -> None:
    for name in names:
        if name not in ctx.helpers:
            ctx.helpers.add(name)
            _need(ctx, *_HELPER_NEEDS.get(name, ()))


# ---------------------------------------------------------------------------
# Step code
# ---------------------------------------------------------------------------


def _block(text: str) -> list[str]:
    """Lines of a code block written flush left (leading newline dropped)."""
    return text.strip("\n").splitlines()


def _reset_to_counts(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(EXACT, _block('''
if "counts_uncorrected" in adata.layers:  # the counts before an ambient RNA correction
    adata.layers["counts"] = adata.layers["counts_uncorrected"]
    del adata.layers["counts_uncorrected"]
if "counts" in adata.layers or adata.raw is not None:
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata = adata.raw.to_adata()
    adata.raw = None
    adata.var.drop(columns=[c for c in (
        "highly_variable", "means", "dispersions", "dispersions_norm", "variances",
        "residual_variances", "highly_variable_rank", "highly_variable_nbatches",
        "highly_variable_intersection") if c in adata.var.columns], inplace=True)
    for k in [k for k in list(adata.obsm.keys()) if k.startswith("X_")]:
        del adata.obsm[k]
    for k in list(adata.obsp.keys()):
        del adata.obsp[k]
    for k in ("neighbors", "rank_genes_groups", "scview_active_clustering", "pca", "umap",
              "leiden", "louvain", "scview_batch_mixing"):
        adata.uns.pop(k, None)
    for k in [k for k in list(adata.uns) if k.startswith(("rank_genes_groups__", "enrichment__"))]:
        del adata.uns[k]
    adata.obs.drop(columns=[c for c in adata.obs.columns
                            if c.startswith("scview_") and c != "scview_mito_type"],
                   inplace=True, errors="ignore")
'''))


def _qc_metrics(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(EXACT, _block('''
adata.var["mt"] = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
'''))


# Where the doublet step took the counts from (its recorded ``input``; none: X).
_DOUBLET_INPUT = {"layers['counts']": 'adata.layers["counts"]', ".raw": "adata.raw.X"}


def _doublet_detection(p: dict, ctx: _Ctx) -> Snippet:
    method = str(p.get("doublet_method") or "scrublet")  # recorded before the choice: Scrublet
    if method == "scdblfinder-style":
        return _doublet_classifier(p, ctx, _DOUBLET_INPUT.get(p.get("input"), "adata.X"))
    found = _doublet_scrublet(p, ctx)
    if method != "both":
        return found
    more = _doublet_classifier(p, ctx, _DOUBLET_INPUT.get(p.get("input"), "adata.X"), codes_known=True)
    status = found.status if more.status == EXACT else more.status
    return Snippet(status, found.code + more.code, "; ".join(n for n in (found.note, more.note) if n))


def _doublet_classifier(p: dict, ctx: _Ctx, counts: str, codes_known: bool = False) -> Snippet:
    """The scDblFinder-style classifier in each capture channel, as scView ran it
    (core/doublets.py ``run_classifier``): its recorded channels and rates."""
    _need(ctx, "doublet_classifier")
    rec = p.get("classifier") if isinstance(p.get("classifier"), dict) else {}
    channels = [c for c in rec.get("channels") or [] if isinstance(c, dict)]
    if not channels:
        return Snippet(APPROXIMATE, ["# NOTE: the doublet classifier's channels were not recorded; skipped."],
                       "doublet classifier channels not recorded")
    names = [str(c.get("name")) for c in channels]
    rates = [float(c["expected_doublet_rate"]) if isinstance(c.get("expected_doublet_rate"), (int, float))
             else 0.05 for c in channels]
    min_cells = p.get("min_channel_cells")
    min_cells = int(min_cells) if isinstance(min_cells, (int, float)) else 100
    code = ["# The scDblFinder-style classifier (classify_by_channel above) in each capture channel",
            "# alone, with the rates scView used; a channel under "
            f"{min_cells} cells is not scored."]
    if codes_known:
        if names != [str(c.get("name")) for c in p.get("channels") or [] if isinstance(c, dict)]:
            return Snippet(APPROXIMATE, ["# NOTE: the classifier ran in other channels than Scrublet; skipped."],
                           "doublet classifier channels differ")
    else:
        code += _channel_codes(p, ctx, names)
    code += [f"CLS_RATES = {_lit(rates)}",
             f"cls_score, cls_call, cls_runs = classify_by_channel({counts}, codes, CLS_RATES, "
             f"{_lit(min_cells)}, rows=channel_rows)",
             'adata.obs["doublet_score_classifier"] = cls_score',
             'adata.obs["predicted_doublet_classifier"] = cls_call',
             'adata.uns["scview_doublet_classifier"] = {"scview_channels": {',
             f'    "name": {_lit(names)}, "n_cells": [r["n_cells"] for r in cls_runs],',
             '    "n_scored": [r["n_scored"] for r in cls_runs], "expected_doublet_rate": CLS_RATES,',
             '    "threshold": [np.nan if r["threshold"] is None else r["threshold"] for r in cls_runs],',
             '    "n_called": [r["n_called"] for r in cls_runs], "status": [r["status"] for r in cls_runs]},',
             f'    "scview_channel_reason": {_lit(str(p.get("channel_reason") or ""))}}}']
    column = p.get("channel_column")
    if column is not None:
        code.append(f'adata.uns["scview_doublet_classifier"]["scview_channel_column"] = {_lit(str(column))}')
        if not codes_known and str(p.get("channel_reason") or "").startswith("barcode"):
            code.append('adata.obs["capture_channel"] = pd.Categorical.from_codes(codes, categories=CHANNELS)')
    return Snippet(EXACT, code, "")


def _channel_codes(p: dict, ctx: _Ctx, names: list[str]) -> list[str]:
    """Code that sets ``codes`` (each cell's channel, an index into CHANNELS; -1: none)
    as the doublet step found them: the recorded column, the barcodes, or all cells."""
    column, reason = p.get("channel_column"), str(p.get("channel_reason") or "")
    code = [f"CHANNELS = {_lit(names)}"]
    if column is None:
        code += ["# All cells as one channel.", "codes = np.zeros(adata.n_obs, dtype=np.int64)"]
    elif reason.startswith("barcode"):
        _need(ctx, "barcode_channels")
        code += ["# The channels read from the cell barcodes (barcode_channels above), as scView did.",
                 "found = barcode_channels(adata.obs_names)",
                 "if found is None:",
                 '    sys.exit("The cell barcodes do not carry the capture channels scView read from them.")',
                 "labels = np.asarray(found[0], dtype=object)[found[1]]",
                 "codes = pd.Categorical(labels, categories=CHANNELS).codes.astype(np.int64)"]
    else:
        col = _lit(str(column))
        code += [f"# The channels: obs column {comment_text(column, 80)}.",
                 f"if {col} not in adata.obs.columns:",
                 f"    sys.exit('obs column ' + {col} + ' (the capture channels) is missing.')",
                 f"labels = adata.obs[{col}].astype(str).where(adata.obs[{col}].notna())",
                 "codes = pd.Categorical(labels, categories=CHANNELS).codes.astype(np.int64)"]
    return code


def _doublet_scrublet(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "scrublet_counts")
    counts = _DOUBLET_INPUT.get(p.get("input"), "adata.X")
    channels = p.get("channels")
    if isinstance(channels, list) and channels and all(isinstance(c, dict) for c in channels):
        return _doublet_channels(p, ctx, counts, channels)
    rate = p.get("expected_doublet_rate")
    recorded = isinstance(rate, (int, float)) and rate > 0
    approx = {"exact": "False", "approximate": "True"}.get(p.get("neighbors"), "None")
    code = ["# Scrublet on the counts alone (scrublet_counts above): sc.pp.scrublet's scores, "
            "without its copies of the dataset.",
            f"score, call, uns = scrublet_counts({counts}, "
            f"expected_doublet_rate={_lit(float(rate) if recorded else 0.05)}, "
            f"use_approx_neighbors={approx})",
            'adata.obs["doublet_score"] = score',
            "# A cell Scrublet dropped (under 3 genes) has no call: not a doublet (as scView).",
            'adata.obs["predicted_doublet"] = np.isfinite(score) & (call == True)',
            'adata.uns["scrublet"] = uns']
    if recorded:
        if p.get("expected_rate_from"):
            code.insert(0, "# The rate scView suggested from the cells per sample "
                           f"({comment_text(p['expected_rate_from'], 80)}), as it ran.")
        return Snippet(EXACT, code, "")
    code.insert(0, "# NOTE: the expected doublet rate was not recorded; scanpy's default (0.05) is used.")
    return Snippet(APPROXIMATE, code, "doublet rate not recorded: scanpy's default")


def _doublet_channels(p: dict, ctx: _Ctx, counts: str, channels: list[dict]) -> Snippet:
    """Scrublet in each capture channel, as scView ran it: the channels from the recorded
    column (or the barcodes, or all cells as one), each with its recorded rate and
    neighbours (core/doublets.py ``run_scrublet``)."""
    column, reason = p.get("channel_column"), str(p.get("channel_reason") or "")
    names = [str(c.get("name")) for c in channels]
    rates = [float(c["expected_doublet_rate"]) if isinstance(c.get("expected_doublet_rate"), (int, float))
             else 0.05 for c in channels]
    approx = [c.get("neighbors") == "approximate" for c in channels]
    min_cells = p.get("min_channel_cells")
    min_cells = int(min_cells) if isinstance(min_cells, (int, float)) else 100
    status, note = EXACT, ""
    code = ["# Scrublet in each capture channel alone (scrublet_by_channel above), one after another,",
            "# each with the expected rate and neighbours scView used; a channel under",
            f"# {min_cells} cells is not scored (NaN score, no call).",
            f"RATES = {_lit(rates)}",
            f"APPROX = {_lit(approx)}"]
    code += _channel_codes(p, ctx, names)
    code += [f"score, call, runs = scrublet_by_channel({counts}, codes, RATES, APPROX, {_lit(min_cells)})",
             'adata.obs["doublet_score"] = score',
             'adata.obs["predicted_doublet"] = call']
    if column is not None and reason.startswith("barcode"):
        code.append('adata.obs["capture_channel"] = pd.Categorical.from_codes(codes, categories=CHANNELS)')
    code += ["# Scrublet's own uns when one run covered every cell, and each channel's record.",
             "one = len(runs) == 1 and runs[0]['uns'] is not None",
             'adata.uns["scrublet"] = dict(runs[0]["uns"]) if one else {}',
             'adata.uns["scrublet"]["scview_channels"] = {',
             '    "name": CHANNELS, "n_cells": [r["n_cells"] for r in runs],',
             '    "n_scored": [r["n_scored"] for r in runs], "expected_doublet_rate": RATES,',
             '    "threshold": [np.nan if r["threshold"] is None else r["threshold"] for r in runs],',
             '    "n_called": [r["n_called"] for r in runs],',
             '    "neighbors": ["approximate" if x else "exact" for x in APPROX],',
             '    "status": [r["status"] for r in runs]}',
             f'adata.uns["scrublet"]["scview_channel_reason"] = {_lit(reason)}']
    if column is not None:
        code.append(f'adata.uns["scrublet"]["scview_channel_column"] = {_lit(str(column))}')
    if reason in ("donor", "batch"):
        code.insert(0, "# NOTE: scView took the samples as the channels (no lane or GEM well column): a guess.")
    return Snippet(status, code, note)


def _mito_rule(p: dict) -> dict | None:
    """A per-type filtering's recorded rule, as the script needs it."""
    rule = p.get("qc_mito_types") if p.get("qc_mito_mode") == "per_type" else None
    if not isinstance(rule, dict) or not isinstance(rule.get("types"), list) or not rule.get("column"):
        return None
    return {"column": rule["column"], "k": rule.get("k"), "cap": rule.get("cap", 50.0),
            "types": [{"name": r.get("name"), "n_cells": r.get("n_cells"), "median": r.get("median"),
                       "bound": r.get("bound")} for r in rule["types"] if isinstance(r, dict)]}


def _floor_lines(global_mito: str) -> list[str]:
    """The per-type % mito filter with scView's floor (``qc_mito_type_floor``): a
    cell goes only when its type's bound and the global rule both remove it."""
    return ["# Never stricter than the global rule: each cell's bound is the higher of its type's",
            "# own and the global rule's for that cell, so it goes only when both remove it.",
            f"GLOBAL_MITO = {global_mito}",
            "drop |= type_mito_drop(adata, qc, MITO_TYPES, GLOBAL_MITO) & GLOBAL_MITO"]


def _filtering(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "per_cell_qc")
    mode = p.get("qc_threshold_mode") or "fixed"
    min_cells = p.get("min_cells", 3)
    code = ["qc = per_cell_qc(adata)"]
    status, note = EXACT, ""
    rule = _mito_rule(p)
    floor = rule is not None and bool(p.get("qc_mito_type_floor"))  # absent (older records): off
    if rule is not None:
        _need(ctx, "type_mito_drop")
        code += ["# % mito per cell type (scView's per-type filter): each type's recorded bound,",
                 "# median + k MADs; types without one keep the global rule below.",
                 f"MITO_TYPES = {_lit(rule)}"]
    if mode == "mad":
        rec = p.get("qc_thresholds")
        _need(ctx, "apply_thresholds")
        if isinstance(rec, dict) and isinstance(rec.get("pooled"), dict):
            code.append("# Each sample's thresholds as scView worked them out and applied them.")
            code.append(f"THRESHOLDS = {_lit(rec)}")
        else:
            _need(ctx, "mad_thresholds")
            status, note = APPROXIMATE, "thresholds not recorded: worked out again"
            ceiling = p.get("max_pct_mt")
            code += [
                "# NOTE: this filtering's thresholds were not recorded; they are worked out "
                "again from the cells here.",
                f"THRESHOLDS = mad_thresholds(adata, qc, {_lit(p.get('qc_sample_column') or None)}, "
                f"k={_lit(p.get('qc_mad_k', 5.0))}, k_mt={_lit(p.get('qc_mad_k_mt', 3.0))}, "
                f"upper={_lit(bool(p.get('qc_mad_upper', False)))}, ceiling={_lit(ceiling)})",
            ]
        if rule is None:
            code.append("drop = mad_drop(adata, qc, THRESHOLDS)")
        elif floor:
            code += ['drop = mad_drop(adata, {**qc, "pct_mt": None}, THRESHOLDS)',
                     *_floor_lines("mad_drop(adata, mito_only(qc), THRESHOLDS)")]
        else:
            code += ['drop = mad_drop(adata, {**qc, "pct_mt": None}, THRESHOLDS)',
                     "drop |= type_mito_drop(adata, qc, MITO_TYPES, mad_drop(adata, mito_only(qc), THRESHOLDS))"]
    else:
        _need(ctx, "fixed_drop")
        cut = (f"min_genes={_lit(p.get('min_genes', 0))}, max_genes={_lit(p.get('max_genes', 0))}, "
               f"min_counts={_lit(p.get('min_counts', 0))}")
        if rule is None:
            code.append(f"drop = fixed_drop(qc, {cut}, max_pct_mt={_lit(p.get('max_pct_mt'))})")
        elif floor:
            code += [f"drop = fixed_drop(qc, {cut}, max_pct_mt=None)",
                     *_floor_lines(f"fixed_drop(mito_only(qc), 0, 0, 0, {_lit(p.get('max_pct_mt'))})")]
        else:
            code += [f"drop = fixed_drop(qc, {cut}, max_pct_mt=None)",
                     "drop |= type_mito_drop(adata, qc, MITO_TYPES, fixed_drop(mito_only(qc), 0, 0, 0, "
                     f"{_lit(p.get('max_pct_mt'))}))"]
        if "min_genes" not in p:
            status, note = APPROXIMATE, "fixed cut-offs not recorded"
            code.insert(0, "# NOTE: this record has no cut-offs (an older scView): none applied.")
    if p.get("drop_doublets"):
        _need(ctx, "doublet_calls")
        code += [*_doublet_source_code(p), "if calls is not None:", "    drop |= calls"]
    if p.get("drop_hash_negatives"):
        _need(ctx, "doublet_calls")  # defines hash_negatives too
        code += ["negatives = hash_negatives(adata.obs)  # cell hashing's Negative cells",
                 "if negatives is not None:", "    drop |= negatives"]
    code += [
        "if drop.any():",
        "    adata = adata[~drop, :].copy()",
    ]
    clones = p.get("vdj_clonotypes")
    if isinstance(clones, dict):
        _need(ctx, "vdj")
        code += ["# V(D)J clone sizes recomputed on the cells kept, as scView's filtering step does.",
                 "if CHAINS_KEY in adata.obsm:",
                 f"    redefine(adata, same_v_gene={_lit(bool(clones.get('same_v_gene')))})"]
    code.append(f"sc.pp.filter_genes(adata, min_cells={_lit(min_cells)})")
    return Snippet(status, code, note)


def _doublet_source_code(p: dict) -> list[str]:
    """The doublet calls a filtering dropped: the source it recorded (the one its
    choice stood for), or Scrublet first for a record from before the choice."""
    if "qc_doublet_source" not in p:
        return ["calls = doublet_calls(adata.obs)"]
    used = p.get("qc_doublet_source_used") or p.get("qc_doublet_source") or "auto"
    return [f"# Doublets by {comment_text(used, 60)} (scView's choice: {comment_text(p['qc_doublet_source'], 20)}).",
            f"calls = doublet_calls(adata.obs, {_lit(str(used))})"]


def _mito_refilter(p: dict, ctx: _Ctx) -> Snippet:
    """scView's two-pass % mito filter (core/mito_refilter.py): the steps above are
    pass 1; this starts again from the input with every cell typed."""
    column, type_col = p.get("column"), p.get("type_column")
    if not isinstance(column, str) or not isinstance(type_col, str):
        return Snippet(NOT_REPLAYABLE, ["# Re-filter record without its columns. Recorded:",
                                        *_param_comment(p)], "columns not recorded")
    code = ["# Pass 2 of scView's per-type % mito filter: start again from the input (the steps",
            "# above were pass 1), every cell typed for the per-type filter that follows.",
            "pass1 = adata",
            "adata = read_input(INPUT)"]
    if p.get("reset_to_counts"):
        code += _reset_to_counts({}, ctx).code
    if type_col != column:
        _need(ctx, "refilter_types")
        target = (p.get("assignment") or {}).get("target_sum", 1e4)
        code.append(f"adata.obs[{_lit(type_col)}] = refilter_types(adata, pass1, {_lit(column)}, "
                    f"target_sum={_lit(target)})")
    code.append("del pass1")
    return Snippet(EXACT, code)


def _ambient_correction(p: dict, ctx: _Ctx) -> Snippet:
    """Each sample's recorded rho, its soup read again from its raw matrix (by the file
    names recorded), subtracted per cell as core/ambient.py does."""
    samples = p.get("samples")
    if not isinstance(samples, list) or not samples or not all(
            isinstance(s, dict) and s.get("raw_files") for s in samples):
        return Snippet(NOT_REPLAYABLE, ["# Ambient RNA correction: the raw matrices are not recorded, "
                                        "so the soup can't be read again. Recorded:",
                                        *_param_comment(p)], "raw matrices not recorded")
    _need(ctx, "ambient")
    rows = [[str(s.get("sample")), s.get("column"), float(s.get("rho", 0.0)),
             [str(f) for f in s.get("raw_files") or []]] for s in samples]
    return Snippet(EXACT, [
        "# Ambient RNA removed from the counts (SoupX-style subtraction per cell, core/ambient.py):",
        "# per sample (name, obs column or None for every cell, rho scView used, raw matrix files),",
        "# its soup read again from the raw matrix's empty droplets.",
        f"AMBIENT_SAMPLES = {_lit(rows)}",
        *_block('''
original = adata.layers["counts_uncorrected"] if "counts_uncorrected" in adata.layers else adata.X
original = sp.csr_matrix(original)
groups = []
for sample, column, rho, files in AMBIENT_SAMPLES:
    frac, _ = soup_fractions(raw_soup(files), adata.var_names, _var_ids(adata))
    rows = (np.flatnonzero(adata.obs[column].astype(str).to_numpy() == sample) if column
            else np.arange(adata.n_obs))
    groups.append((rows, rho, frac))
corrected = correct_counts(original, groups)
if "counts_uncorrected" not in adata.layers:
    adata.layers["counts_uncorrected"] = original
adata.X = corrected
if "counts" in adata.layers:
    adata.layers["counts"] = corrected.copy()
del original, groups, corrected
'''),
    ])


def _normalization(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "keep_raw_counts")
    target = p.get("target_sum", 1e4)
    return Snippet(EXACT, _block(f'''
if "counts" not in adata.layers:
    if adata.raw is None or looks_like_counts(adata.X):
        adata.layers["counts"] = adata.X.copy()
    else:
        keep_raw_counts(adata)
sc.pp.normalize_total(adata, target_sum={_lit(float(target))})
'''))


def _log_transform(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(EXACT, ["sc.pp.log1p(adata)"])


def _hvg(p: dict, ctx: _Ctx) -> Snippet:
    n = p.get("n_top_genes", 2000)
    batch = p.get("hvg_batch_key") or None
    if "pearson_residuals" in (p.get("normalization_method"), p.get("hvg_flavor")):
        _need(ctx, "counts_hvg")
        theta = p.get("theta", 100.0)
        return Snippet(EXACT, [f'counts_hvg(adata, "pearson_residuals", {_lit(n)}, {_lit(batch)}, '
                               f"theta={_lit(theta)})"])
    flavor = p.get("hvg_flavor") or "seurat"
    if flavor == "seurat_v3":
        _need(ctx, "counts_hvg")
        return Snippet(EXACT, [f'counts_hvg(adata, "seurat_v3", {_lit(n)}, {_lit(batch)})'])
    return Snippet(EXACT, [f"sc.pp.highly_variable_genes(adata, n_top_genes={_lit(n)}, "
                           f"flavor={_lit(flavor)}, batch_key={_lit(batch)})"])


def _scaling(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "keep_raw_counts")
    pearson = p.get("normalization_method") == "pearson_residuals"
    if "scaled_genes" in p:
        # Since the step plans the matrix PCA reads (core/scaling.py): X keeps the log
        # values; the PCA step below scales a copy of the variable genes.
        regress = [str(k) for k in (p.get("regress_out") or [])]
        ctx.scaling = {"method": "pearson" if pearson else "zscore",
                       "theta": p.get("theta", 100.0), "max_value": p.get("max_value", 10.0),
                       "regress_out": regress}
        raw = _block('''
if adata.raw is None or not adata.X[:2000].min() < 0:  # z-scores an older scView left in X: kept
    adata.raw = adata  # the log-normalised values, which X keeps too
''')
        if regress and not pearson:
            return Snippet(EXACT, ["keep_raw_counts(adata)", *raw,
                                   *_regress_covariates(regress, ctx),
                                   "# The variable genes' fit on these is regressed out of a copy at the PCA step,",
                                   "# then the z-scores of the residuals are made there."])
        if pearson:
            _need(ctx, "find_counts", "pearson_residuals")
            return Snippet(EXACT, ["keep_raw_counts(adata)",
                                   "find_counts(adata)  # the residuals for PCA need the counts",
                                   "if looks_like_counts(adata.X):",
                                   '    raise ValueError("Run normalization and log transform before the '
                                   'Pearson residuals.")', *raw,
                                   "pearson_clip = np.sqrt(adata.n_obs)  # for the PCA step's residuals"])
        return Snippet(EXACT, ["keep_raw_counts(adata)", *raw,
                               "# The z-scores of the variable genes are made at the PCA step."])
    ctx.scaling = None  # an older scView: the scaled values go into X
    if pearson:
        _need(ctx, "find_counts", "pearson_residuals")
        theta = p.get("theta", 100.0)
        clip = "np.sqrt(adata.n_obs)"
        return Snippet(EXACT, _block(f'''
keep_raw_counts(adata)
counts = find_counts(adata)
if looks_like_counts(adata.X):
    raise ValueError("Run normalization and log transform before the Pearson residuals.")
adata.raw = adata  # .raw keeps the log-normalised values; X gets the residuals
adata.X = pearson_residuals(counts, theta={_lit(theta)}, clip={clip})
'''))
    return Snippet(EXACT, _block('''
keep_raw_counts(adata)
adata.raw = adata
X = adata.X.astype(np.float32)  # dense float32, a new array: .raw stays log-normalised
adata.X = X.toarray() if sp.issparse(X) else X
sc.pp.scale(adata, max_value=10)
'''))


def _regress_covariates(keys: list[str], ctx: _Ctx) -> list[str]:
    """The cell-cycle covariates the scaling step scored before regress-out (S, G2M and
    S - G2M), as the cell-cycle step scores them."""
    from scview.core import scaling

    if not set(keys) & set(scaling.CELL_CYCLE_COVARIATES):
        return []
    code = ["# The cell-cycle scores to regress out, made first (as scView's scaling step does):",
            *_cell_cycle({}, ctx).code]
    if scaling.CC_DIFFERENCE in keys:
        code.append(f'adata.obs[{_lit(scaling.CC_DIFFERENCE)}] = (adata.obs["S_score"].to_numpy(np.float64)'
                    ' - adata.obs["G2M_score"].to_numpy(np.float64))  # S - G2M')
    return code


def _pca(p: dict, ctx: _Ctx) -> Snippet:
    n = p.get("n_comps", 50)
    plan = ctx.scaling if p.get("pca_input") else None
    if plan is None:  # scaling skipped, or an older scView's scaled X
        return Snippet(EXACT, [
            f"sc.tl.pca(adata, n_comps=min({_lit(n)}, adata.n_obs - 1, adata.n_vars - 1), "
            'svd_solver="arpack")',
            *_DROP_CORRECTED,
        ])
    if plan["method"] == "pearson":
        _need(ctx, "find_counts", "pearson_residuals")
        make = [f"X = pearson_residuals(find_counts(adata), theta={_lit(plan['theta'])}, clip=pearson_clip,",
                "                      columns=None if mask is None else np.flatnonzero(mask))",
                "sub = ad.AnnData(X=X)"]
    else:
        regress = plan.get("regress_out") or []
        make = ["X = (adata.X if mask is None else adata.X[:, mask]).astype(np.float32)",
                "sub = ad.AnnData(X=X.toarray() if sp.issparse(X) else np.ascontiguousarray(X))"]
        if regress:
            make += [f"sub.obs = adata.obs[{_lit(regress)}].copy()",
                     "sub.obs.index = sub.obs.index.astype(str)",
                     f"sc.pp.regress_out(sub, {_lit(regress)})  # each gene's least-squares fit taken off"]
        make.append(f"sc.pp.scale(sub, max_value={_lit(plan['max_value'])})")
    return Snippet(EXACT, [
        "# PCA reads the scaled variable genes; scView scales a copy of them for PCA alone (X",
        "# keeps the log-normalised values): a copy, scaled, PCA on it, the results copied back.",
        'mask = adata.var["highly_variable"].to_numpy(dtype=bool) if "highly_variable" in adata.var.columns else None',
        *make,
        "if mask is not None:",
        '    sub.var["highly_variable"] = True  # PCA reads it through the mask, as scView\'s did',
        f"n_comps = min({_lit(n)}, adata.n_obs - 1, adata.n_vars - 1)",
        'sc.tl.pca(sub, n_comps=n_comps, svd_solver="arpack")',
        'adata.obsm["X_pca"] = sub.obsm["X_pca"]',
        'adata.uns["pca"] = sub.uns["pca"]',
        "if mask is None:",
        '    adata.varm["PCs"] = sub.varm["PCs"]',
        "else:",
        '    adata.varm["PCs"] = np.zeros((adata.n_vars, n_comps))  # zero rows for the other genes',
        '    adata.varm["PCs"][mask] = sub.varm["PCs"]',
        "del sub, X",
        *_DROP_CORRECTED,
    ])


def _uncorrected_embedding(p: dict, ctx: _Ctx) -> Snippet:
    k, md = p.get("n_neighbors", 15), p.get("umap_min_dist", 0.5)
    return Snippet(EXACT, _block(f'''
sc.pp.neighbors(adata, n_neighbors={_lit(k)}, use_rep="X_pca", key_added="uncorrected")
sc.tl.umap(adata, min_dist={_lit(md)}, neighbors_key="uncorrected", key_added="X_umap_uncorrected")
for k in ("uncorrected_connectivities", "uncorrected_distances"):
    adata.obsp.pop(k, None)
adata.uns.pop("uncorrected", None)
adata.uns.pop("X_umap_uncorrected", None)
'''))


_DROP_CORRECTED = ['adata.obsm.pop("X_pca_harmony", None)', 'adata.obsm.pop("X_scanorama", None)']


def _batch_correction(p: dict, ctx: _Ctx) -> Snippet:
    key = p.get("batch_key")
    if not key:
        return Snippet(NOT_REPLAYABLE, ["# No batch key was recorded, so nothing is replayed."],
                       "no batch key recorded")
    method = p.get("batch_method") or "harmony"
    if method == "scanorama":
        _need(ctx, "scanorama")
        return Snippet(EXACT, [
            'adata.obsm.pop("X_pca_harmony", None)',
            f'adata.obsm["X_scanorama"] = scanorama(adata, {_lit(key)})'])
    if method == "bbknn":
        w, n_pcs, trim = p.get("neighbors_within_batch"), p.get("n_pcs"), p.get("trim")
        if w is None:
            return Snippet(NOT_REPLAYABLE, ["# BBKNN's neighbours per batch were not recorded."],
                           "BBKNN settings not recorded")
        return Snippet(EXACT, [
            "import bbknn  # pip install bbknn (MIT): the neighbour graph, built batch by batch",
            *_DROP_CORRECTED,
            f"bbknn.bbknn(adata, batch_key={_lit(key)}, neighbors_within_batch={_lit(w)}, "
            f"n_pcs={_lit(n_pcs or 50)}, trim={_lit(trim)}, computation=\"cKDTree\")",
            "# The graph is BBKNN's: the neighbours step below keeps it."])
    _need(ctx, "harmony")
    code = ['adata.obsm.pop("X_scanorama", None)',
            f'adata.obsm["X_pca_harmony"] = harmony(adata.obsm["X_pca"], adata.obs, {_lit(key)})']
    if method != "harmony":
        code.insert(0, f"# NOTE: the record names the method {comment_text(method, 40)!s}; "
                       "it is replayed with Harmony.")
    return Snippet(EXACT, code)


def _neighbors(p: dict, ctx: _Ctx) -> Snippet:
    if p.get("kept"):
        return Snippet(EXACT, ["# The BBKNN graph from batch correction is the neighbour graph "
                               "(kept, as scView did)."])
    k = p.get("n_neighbors", 15)
    return Snippet(EXACT, [
        'use_rep = next((r for r in ("X_pca_harmony", "X_scanorama") if r in adata.obsm), "X_pca")',
        f"sc.pp.neighbors(adata, n_neighbors={_lit(k)}, use_rep=use_rep)",
    ])


def _iterations(p: dict) -> int:
    """Leiden's recorded passes (-1 before fast Leiden existed, or a malformed record)."""
    v = p.get("leiden_iterations")
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) == v \
        and (v == -1 or v > 0) else -1


def _leiden_or_louvain(method: str, resolution: Any, column: str, ctx: _Ctx,
                       on: str = "adata", n_iterations: int = -1) -> list[str]:
    """Clustering of ``on``'s graph: scView's Leiden child process runs exactly
    sc.tl.leiden's leidenalg flavour (core/leiden.py), stopped after ``n_iterations``
    passes for fast Leiden (2)."""
    if method == "louvain":
        _need(ctx, "louvain")
        return [f"louvain_igraph({on}, {_lit(resolution)}, {_lit(column)})"]
    passes = f", n_iterations={_lit(n_iterations)}" if n_iterations != -1 else ""
    return [f"sc.tl.leiden({on}, resolution={_lit(resolution)}, key_added={_lit(column)}, "
            f'flavor="leidenalg"{passes})']


def _clustering(p: dict, ctx: _Ctx) -> Snippet:
    method = p.get("clustering_method") or "leiden"
    res = p.get("clustering_resolution", 0.5)
    column = f"scview_{method}_r{res}"
    code = _leiden_or_louvain(method, res, column, ctx, n_iterations=_iterations(p))
    code.append(f'adata.uns["scview_active_clustering"] = {_lit(column)}')
    return Snippet(EXACT, code)


def _embeddings(p: dict, ctx: _Ctx) -> Snippet:
    md = p.get("umap_min_dist", 0.5)
    # Runs before umap_3d was recorded always made the 3D UMAP
    three_d = "" if p.get("umap_3d") == "off" else (
        f'sc.tl.umap(adata, min_dist={_lit(md)}, n_components=3)\n'
        'adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()\n')
    return Snippet(EXACT, _block(f'''
# umap-learn trims the graph it is given in place: keep the one clustering used.
key = (adata.uns.get("neighbors") or {{}}).get("connectivities_key", "connectivities")
graph = adata.obsp[key].copy() if key in adata.obsp else None
{three_d}sc.tl.umap(adata, min_dist={_lit(md)}, n_components=2)
if graph is not None:
    adata.obsp[key] = graph
'''))


def _marker_genes(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "markers_key")
    columns = [str(c) for c in (p.get("marker_columns") or [])]
    return Snippet(EXACT, _block(f'''
columns = {_lit(columns)}
active = adata.uns.get("scview_active_clustering")
if active and active in adata.obs.columns and active not in columns:
    columns.insert(0, active)
if not columns:
    columns = [c for c in ("leiden", "louvain", "cluster", "clusters", "seurat_clusters",
                           "cell_type") if c in adata.obs.columns][:1]
for column in columns:
    markers_for(adata, column)
if active and markers_key(active) in adata.uns:
    adata.uns["rank_genes_groups"] = adata.uns[markers_key(active)]
'''), "")


def _enrichment(p: dict, ctx: _Ctx) -> Snippet:
    cols = p.get("enrichment_columns") or []
    lines = [
        "# NOTE: enrichment is not replayed. scView ran over-representation tests of each",
        "# group's top marker genes against local MSigDB collections, with the tested genes",
        "# as background. To repeat it, run gseapy.enrich on each group's top markers.",
        "# Recorded:",
        *_param_comment({"enrichment_columns": cols,
                         "enrichment_n_genes": p.get("enrichment_n_genes"),
                         "enrichment_collections": p.get("enrichment_collections")}),
    ]
    # Which species' gene sets they were (a mouse run uses MSigDB's mouse sets).
    named = p.get("enrichment_gene_set_species") or msigdb_species.describe(p.get("enrichment_collections") or [])
    if named:
        lines.append(f"# Gene sets: {named}.")
    return Snippet(APPROXIMATE, lines, "MSigDB over-representation: not replayed (needs the gene sets)")


def _cell_cycle(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core.pipeline import CELL_CYCLE_ALIASES, CELL_CYCLE_G2M_GENES, CELL_CYCLE_S_GENES

    _need(ctx, "lognorm_scoring")
    return Snippet(EXACT, [
        "# Tirosh et al. 2016 S and G2/M genes, matched ignoring case (mouse: Mcm5).",
        f"S_GENES = {_lit(list(CELL_CYCLE_S_GENES))}",
        f"G2M_GENES = {_lit(list(CELL_CYCLE_G2M_GENES))}",
        f"ALIASES = {_lit({k: list(v) for k, v in CELL_CYCLE_ALIASES.items()})}",
        "use_raw, names = lognorm_scoring(adata)",
        "sc.tl.score_genes_cell_cycle(adata, s_genes=match_genes(S_GENES, names, ALIASES),",
        "                             g2m_genes=match_genes(G2M_GENES, names, ALIASES), "
        "use_raw=use_raw)",
    ])


def _dissociation_stress(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core.stress import SCORE_COLUMN, STRESS_GENES

    _need(ctx, "lognorm_scoring")
    genes = p.get("genes")
    code = ["use_raw, names = lognorm_scoring(adata)"]
    status, note = EXACT, ""
    if isinstance(genes, list) and genes:
        code.append(f"genes = {_lit([str(g) for g in genes])}  # the signature genes present")
    else:
        status, note = APPROXIMATE, "genes not recorded: matched again"
        code.append(f"genes = match_genes({_lit(list(STRESS_GENES))}, names)")
    code.append(f"sc.tl.score_genes(adata, genes, score_name={_lit(SCORE_COLUMN)}, use_raw=use_raw, "
                "ctrl_size=max(1, min(50, len(names) - len(genes))), random_state=0)")
    return Snippet(status, code, note)


def _groupby_code(p: dict, ctx: _Ctx) -> list[str]:
    """``groupby`` as the annotation step resolves it."""
    return [
        f"groupby = {_lit(p.get('annotation_groupby') or '')} or "
        'adata.uns.get("scview_active_clustering") or ""',
        "if groupby not in adata.obs.columns:",
        '    groupby = next((c for c in ("cluster", "leiden", "louvain") if c in adata.obs.columns',
        '                    and adata.obs[c].nunique() > 1), "")',
    ]


def _cell_type_annotation(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "markers_key")
    method = (p.get("annotation_method") or "llm").lower()
    target = str(p.get("annotation_target") or "cell_type")
    # The LLM falls back to CellTypist without a key; then the record carries its fit.
    if method == "llm" and any(k in p for k in ("model_species", "gene_overlap", "dataset_species")):
        method = "celltypist"
    code = _groupby_code(p, ctx)
    status, note = EXACT, ""
    if method == "celltypist":
        model = str(p.get("celltypist_model") or "Immune_All_Low.pkl")
        if not model.endswith(".pkl"):
            model += ".pkl"
        code += _block(f'''
import celltypist
from celltypist import models

models.download_models(model=[{_lit(model)}], force_update=False)
if "counts" in adata.layers:
    counts, var_names = adata.layers["counts"], list(adata.var_names)
elif adata.raw is not None:
    counts, var_names = adata.raw.X, list(adata.raw.var_names)
else:
    counts, var_names = adata.X, list(adata.var_names)
inp = ad.AnnData(X=counts.copy())
inp.obs_names = adata.obs_names
inp.var_names = var_names
sc.pp.normalize_total(inp, target_sum=1e4)
sc.pp.log1p(inp)
kwargs = {{"model": models.Model.load(model={_lit(model)}), "majority_voting": True}}
if groupby:
    inp.obs[groupby] = adata.obs[groupby].values
    kwargs["over_clustering"] = groupby
pred = celltypist.annotate(inp, **kwargs)
labels = pred.predicted_labels
col = "majority_voting" if "majority_voting" in labels.columns else "predicted_labels"
adata.obs[{_lit(target)}] = labels[col].astype(str).astype("category").values
if col != "predicted_labels":
    adata.obs[{_lit(target + "_percell")}] = labels["predicted_labels"].astype(str).astype("category").values
adata.obs[{_lit(target + "_confidence")}] = np.asarray(pred.probability_matrix.max(axis=1), dtype="float32")
''')
        code.insert(0, "# NOTE: CellTypist downloads the model's current release; a model "
                       "updated since this run can name a few cells differently.")
    elif method == "llm":
        mapping = _uns_mapping(ctx.adata, f"{target}_llm_mapping")
        if mapping:
            status, note = RECORDED, "LLM labels written in from the record"
            code += [
                "# NOTE: an LLM named each cluster from its top markers; asking again can",
                "# give other names, so the names scView stored for this column are used.",
                f"mapping = {_lit(mapping)}",
                f"adata.obs[{_lit(target)}] = adata.obs[groupby].astype(str).map(",
                '    lambda g: mapping.get(g, "Unknown")).astype("category")',
            ]
        else:
            return Snippet(NOT_REPLAYABLE, [
                "# An LLM named each cluster from its top markers; its answer is not stored",
                "# in this dataset, so the labels cannot be replayed. Recorded:",
                *_param_comment(p),
            ], "LLM labels not stored")
    elif method == "marker_score" and _marker_db_mapping(p, ctx, target):
        # Named from the marker databases (core/marker_db.py), which the script
        # does not download: the names scView stored for this column are used.
        status, note = RECORDED, "marker-database labels written in from the record"
        code += [
            "# NOTE: scView named each cluster from the PanglaoDB + CellMarker 2.0 marker",
            "# databases (fetched on first use, not by this script), so the names it",
            "# stored for this column are used.",
            f"mapping = {_lit(_marker_db_mapping(p, ctx, target))}",
            f"adata.obs[{_lit(target)}] = adata.obs[groupby].astype(str).map(",
            '    lambda g: mapping.get(g, "Unknown")).astype("category")',
        ]
    elif method == "marker_score":
        sets = {k: list(v) for k, v in _builtin_marker_sets(p, ctx, target).items()}
        code += _block(f'''
MARKER_SETS = {_lit(sets)}
upper = {{str(g).upper(): str(g) for g in adata.var_names}}
score_cols = {{}}
for ct, genes in MARKER_SETS.items():
    present = [upper[g.upper()] for g in genes if g.upper() in upper]
    if len(present) < 2:
        continue
    sc.tl.score_genes(adata, present, score_name=f"_ms__{{ct}}", ctrl_size=min(50, adata.n_vars - 1))
    score_cols[ct] = f"_ms__{{ct}}"
inv = {{v: k for k, v in score_cols.items()}}
means = adata.obs.groupby(groupby, observed=True)[list(score_cols.values())].mean()
mapping = {{str(i): inv[means.loc[i].idxmax()] for i in means.index}}
adata.obs[{_lit(target)}] = adata.obs[groupby].astype(str).map(
    lambda g: mapping.get(g, "Unknown")).astype("category")
adata.obs.drop(columns=list(score_cols.values()), inplace=True)
''')
    else:
        return Snippet(UNKNOWN, [f"# Unknown annotation method {comment_text(method, 40)}. Recorded:",
                                 *_param_comment(p)], "unknown annotation method")
    code.append(f"markers_for(adata, {_lit(target)})")
    return Snippet(status, code, note)


def _marker_db_mapping(p: dict, ctx: _Ctx, target: str) -> dict[str, str] | None:
    """The stored marker-score labels when the marker databases made them, else None."""
    source = getattr(ctx.adata, "uns", {}).get(f"{target}_markerscore_source") if ctx.adata is not None else None
    from_db = p.get("marker_source") == "database" or (
        hasattr(source, "get") and source.get("marker_source") == "database")
    return _uns_mapping(ctx.adata, f"{target}_markerscore_mapping") if from_db else None


def _builtin_marker_sets(p: dict, ctx: _Ctx, target: str) -> dict[str, list[str]]:
    """The built-in marker sets the run used: runs recorded before the version 2 sets
    (pDC, CD16+ monocyte) carry no version and replay with the old ones."""
    from scview.core.pipeline import _MARKER_SETS, _MARKER_SETS_V1

    source = getattr(ctx.adata, "uns", {}).get(f"{target}_markerscore_source") if ctx.adata is not None else None
    version = p.get("marker_sets_version") or (source.get("marker_sets_version") if hasattr(source, "get") else None)
    try:
        return _MARKER_SETS if int(version) >= 2 else _MARKER_SETS_V1
    except (TypeError, ValueError):
        return _MARKER_SETS_V1


def _uns_mapping(adata, key: str) -> dict[str, str] | None:
    raw = getattr(adata, "uns", {}).get(key) if adata is not None else None
    if not hasattr(raw, "items"):
        return None
    try:
        return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        return None


def _ingest(p: dict, ctx: _Ctx) -> Snippet:
    lines = ["# The import itself is read_input() at the top of this script."]
    pub = p.get("public_import")
    if isinstance(pub, dict):  # a CELLxGENE Discover dataset version (core/public_import.py)
        return Snippet(EXACT, lines + [
            f"# Downloaded from {comment_text(pub.get('source') or 'CELLxGENE Discover', 40)}: "
            f"{comment_text(pub.get('url'), 120)}",
            f"# (dataset version {comment_text(pub.get('dataset_version_id'), 40)}, sha256 "
            f"{comment_text(pub.get('sha256'), 64)}, licence {comment_text(pub.get('license'), 20)}).",
            "# Either that file or scView's stored copy works as INPUT: gene symbols become the",
            "# gene index (Ensembl ids kept in var['gene_ids']), in X and in .raw.",
            "def _symbols(var):",
            "    ids = var.index.astype(str)",
            "    names = var[\"feature_name\"].astype(str).to_numpy()",
            "    names = [n if n and n.lower() != \"nan\" else i for n, i in zip(names, ids)]",
            "    var = var.copy()",
            "    var[\"gene_ids\"] = list(ids)",
            "    var.index = ad.utils.make_index_unique(pd.Index(names))",
            "    return var",
            "if \"feature_name\" in adata.var.columns and \"gene_ids\" not in adata.var.columns:",
            "    if adata.raw is not None and \"feature_name\" in adata.raw.var.columns:",
            "        raw = adata.raw.to_adata()",
            "        raw.var = _symbols(raw.var)",
            "        adata.raw = raw",
            "    adata.var = _symbols(adata.var)",
        ])
    if isinstance(p.get("feature_split"), dict):  # an .h5ad upload with antibody / tag features in X
        lines += [
            "# scView kept only the 'Gene Expression' features in X; the antibody (CITE-seq) and",
            "# CellPlex tag features went to obsm['protein_counts'] / obsm['hto_counts'].",
            "if \"feature_types\" in adata.var.columns:",
            "    adata = adata[:, adata.var[\"feature_types\"].astype(str).eq(\"Gene Expression\").to_numpy()].copy()",
        ]
        return Snippet(APPROXIMATE, lines, "antibody and tag features dropped from X, not moved to obsm")
    if any(k.startswith("protein") or k.startswith("n_protein") for k in p):
        lines.append("# NOTE: scView moved the antibody (CITE-seq) features out of X into "
                     "obsm['protein_counts']; read_input keeps only the gene features.")
        return Snippet(APPROXIMATE, lines, "antibody features dropped, not moved to obsm")
    return Snippet(EXACT, lines)


def _integrate(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(NOT_REPLAYABLE, [
        "# This dataset was made by merging other scView datasets; the merge is not",
        "# replayed. Start from the merged counts file (the input above). Recorded:",
        *_param_comment(p),
    ], "merge of datasets: start from the merged file")


def _subcluster(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core import subcluster as sub

    _need(ctx, "subcluster")
    target = str(p.get("target") or sub.target_column(str(p.get("column"))))
    return Snippet(EXACT, [
        f"# Writes adata.obs[{comment_text(repr(target), 80)}].",
        f"subcluster(adata, {_lit(str(p.get('column')))}, {_lit(str(p.get('group')))}, "
        f"{_lit(p.get('resolution', sub.DEFAULT_RESOLUTION))}, batch_key={_lit(p.get('batch_key'))}, "
        f"n_top_genes={_lit(sub.N_TOP_GENES)}, n_pcs={_lit(sub.N_PCS)}, "
        f"n_neighbors={_lit(sub.N_NEIGHBORS)}, seed={_lit(sub.SEED)}"
        + (f", n_iterations={_lit(_iterations(p))}" if _iterations(p) != -1 else "") + ")",
    ])


def _keep_resolution(p: dict, ctx: _Ctx) -> Snippet:
    method = p.get("clustering_method") or "leiden"
    res = p.get("clustering_resolution", 0.5)
    column = str(p.get("column") or f"scview_{method}_r{res}")
    code = ["# A clustering kept from the resolution sweep (the active one does not change)."]
    code += [f"if {_lit(column)} not in adata.obs.columns:"]
    code += ["    " + line for line in _leiden_or_louvain(method, res, column, ctx,
                                                          n_iterations=_iterations(p))]
    return Snippet(EXACT, code)


def _keep_hvg_setting(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "harmony")
    column = str(p.get("column") or "scview_hvg_check")
    flavor = p.get("hvg_flavor") or "seurat"
    method = p.get("clustering_method") or "leiden"
    res = p.get("clustering_resolution", 0.5)
    code = [
        "# NOTE: scView's variable-gene check re-clustered under another HVG setting; this",
        "# is the same chain in plain scanpy (log values, scaled PCA), so the clusters can",
        "# differ a little from the check's.",
        "tmp = adata.raw.to_adata() if adata.raw is not None else adata.copy()",
    ]
    if flavor in ("seurat", "cell_ranger"):
        code.append(f"sc.pp.highly_variable_genes(tmp, n_top_genes={_lit(p.get('n_top_genes', 2000))}, "
                    f"flavor={_lit(flavor)})")
    else:
        code.append(f"# NOTE: flavour {comment_text(flavor, 40)} ranks the counts in scView; the "
                    "Seurat flavour on the log values stands in for it here.")
        code.append(f"sc.pp.highly_variable_genes(tmp, n_top_genes={_lit(p.get('n_top_genes', 2000))})")
    code += [
        'tmp = tmp[:, tmp.var["highly_variable"]].copy()',
        "sc.pp.scale(tmp, max_value=10)",
        f"sc.tl.pca(tmp, n_comps=min({_lit(p.get('n_comps', 50))}, tmp.n_obs - 1, tmp.n_vars - 1), "
        'svd_solver="arpack")',
        'rep = "X_pca"',
    ]
    if p.get("batch_key"):
        code += [f'tmp.obsm["X_pca_harmony"] = harmony(tmp.obsm["X_pca"], tmp.obs, '
                 f"{_lit(p['batch_key'])})", 'rep = "X_pca_harmony"']
    code.append(f"sc.pp.neighbors(tmp, n_neighbors={_lit(p.get('n_neighbors', 15))}, use_rep=rep)")
    code += _leiden_or_louvain(method, res, "c", ctx, on="tmp", n_iterations=_iterations(p))
    code.append(f'adata.obs[{_lit(column)}] = tmp.obs["c"].values')
    return Snippet(APPROXIMATE, code, "re-run in plain scanpy; clusters can differ a little")


def _pseudotime(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core.trajectory import DIFFMAP_COMPS, DPT_COLUMN, N_DCS

    root = p.get("root_cell")
    if not root:
        return Snippet(NOT_REPLAYABLE, ["# Pseudotime: no root cell recorded. Recorded:",
                                        *_param_comment(p)], "no root cell recorded")
    return Snippet(APPROXIMATE, _block(f'''
# NOTE: scView runs DPT on the cells its graph connects to the root and gives the
# others an infinite pseudotime; this runs it on every cell of the main graph.
sc.tl.diffmap(adata, n_comps={_lit(p.get("diffmap_n_comps", DIFFMAP_COMPS))})
adata.uns["iroot"] = int(adata.obs_names.get_loc({_lit(str(root))}))
sc.tl.dpt(adata, n_dcs={_lit(p.get("n_dcs", N_DCS))})
adata.obs[{_lit(str(p.get("column") or DPT_COLUMN))}] = adata.obs["dpt_pseudotime"].to_numpy()
'''), "DPT on the whole graph, not only the root's component")


def _paga(p: dict, ctx: _Ctx) -> Snippet:
    groupby = str(p.get("groupby") or "")
    return Snippet(APPROXIMATE, [
        "# NOTE: scView runs PAGA on a copy of the graph, over the cells with a group.",
        f"sc.tl.paga(adata, groups={_lit(groupby)})",
    ], "PAGA on all cells")


def _annotation_confidence(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(NOT_REPLAYABLE, [
        "# scView's label-fit check (split conformal, cross-fit classifier) has no scanpy",
        "# equivalent; it is not replayed. Recorded:",
        *_param_comment(p),
    ], "scView's own check")


def _label_cells(p: dict, ctx: _Ctx) -> Snippet:
    column = str(p.get("column") or "")
    return Snippet(NOT_REPLAYABLE, [
        "# Cells labelled by hand (lasso, legend or filter selection); which cells is not",
        "# recorded. To carry the final labels over, join the column from scView's cell",
        "# metadata export by barcode, e.g.:",
        "#   meta = pd.read_csv('cell_metadata.csv', index_col=0)",
        f"#   adata.obs[{comment_text(repr(column), 80)}] = meta.loc[adata.obs_names, "
        f"{comment_text(repr(column), 80)}].astype('category')",
        "# Recorded:",
        *_param_comment(p),
    ], "manual labels: cells not recorded")


def _annotation_crosscheck(p: dict, ctx: _Ctx) -> Snippet:
    outs = [str(c) for c in (p.get("outputs") or []) if c]
    join = [f"#   adata.obs[{comment_text(repr(c), 80)}] = meta.loc[adata.obs_names, {comment_text(repr(c), 80)}]"
            for c in outs]
    return Snippet(NOT_REPLAYABLE, [
        "# scView's annotation cross-check (consensus + agreement over several label",
        "# columns) has no scanpy equivalent; it is not replayed. To carry its columns",
        "# over, join them from scView's cell metadata export by barcode, e.g.:",
        "#   meta = pd.read_csv('cell_metadata.csv', index_col=0)",
        *join,
        "# Recorded:",
        *_param_comment(p),
    ], "scView's own cross-check")


def _bulk_deconvolution(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(NOT_REPLAYABLE, [
        "# Bulk deconvolution (scView's MuSiC-style weighted NNLS) needs the uploaded bulk file and",
        "# only adds a result; it is not replayed. Deconvolution > Download reference exports this",
        "# dataset as a reference for MuSiC, BayesPrism or CIBERSORTx. Recorded:",
        *_param_comment(p),
    ], "needs the bulk file")


def _spot_deconvolution(p: dict, ctx: _Ctx) -> Snippet:
    """Spot composition (core/spot_deconvolution.py) on the recorded genes, noise level and
    types; the reference read from its file (scView read it from the library)."""
    from scview.core import spot_deconvolution as sd

    genes, ref_genes, fit, types = p.get("genes"), p.get("ref_genes"), p.get("fit_genes"), p.get("types")
    label, ref_file, method, s_e = p.get("celltype_column"), p.get("reference_file"), p.get("method"), p.get("s_e")
    ok = (isinstance(genes, list) and genes and isinstance(ref_genes, list) and len(ref_genes) == len(genes)
          and isinstance(fit, list) and fit and all(isinstance(i, int) and 0 <= i < len(genes) for i in fit)
          and isinstance(types, list) and len(types) >= 2 and isinstance(label, str) and label
          and isinstance(ref_file, str) and ref_file and method in sd.METHODS
          and (method == "nnls" or isinstance(s_e, (int, float))))
    if not ok:
        return Snippet(NOT_REPLAYABLE, ["# Spot composition: its genes, reference file or settings are not "
                                        "recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items()
                                                         if x not in ("genes", "ref_genes", "fit_genes")})],
                       "settings not recorded")
    _need(ctx, "spot_deconvolution")
    status, notes = EXACT, []  # the recorded names are each file's own, also after a re-indexing
    for who, src in (("reference", p.get("reference_source")), ("spots", p.get("spot_source"))):
        if src and src not in ("layers['counts']", ".raw", "X"):
            status = APPROXIMATE
            notes.append(f"scView read the {who}' counts from {comment_text(src, 60)}; here from the file's own.")
    return Snippet(status, [
        f"# Spot composition from {comment_text(p.get('reference'), 80)} (its column {comment_text(label, 60)}), "
        f"{comment_text(sd.METHOD_NAMES.get(method, method), 60)}.",
        *[f"# NOTE: {n}" for n in notes],
        "SPOT_GENES = [",
        *_packed([repr(str(g)) for g in genes]),
        "]",
        "SPOT_REF_GENES = [",
        *_packed([repr(str(g)) for g in ref_genes]),
        "]",
        f"SPOT_FIT = {_lit([int(i) for i in fit])}",
        f"SPOT_TYPES = {_lit([str(t) for t in types])}",
        f"ref = read_input(spot_reference_file({_lit(ref_file)}))",
        f"spot_composition(adata, ref, {_lit(label)}, SPOT_TYPES, SPOT_GENES, SPOT_REF_GENES, SPOT_FIT, "
        f"{_lit(s_e)}, {_lit(method)})",
        "del ref",
    ], "; ".join(notes)[:200] if notes else "")


def _merge_clusters(p: dict, ctx: _Ctx) -> Snippet:
    column, target, mapping = p.get("column"), p.get("target"), p.get("mapping")
    if not column or not target or not isinstance(mapping, dict):
        return Snippet(NOT_REPLAYABLE, ["# Merged clusters: the merge is not recorded. Recorded:",
                                        *_param_comment(p)], "merge not recorded")
    mapping = {str(k): str(v) for k, v in mapping.items()}
    order = [str(c) for c in (p.get("categories") or dict.fromkeys(mapping.values()))]
    return Snippet(EXACT, [
        "# Clusters merged after scView's merge check (count splitting; the check itself is",
        "# not replayed: it only suggested the merges written here).",
        f"merged = {_lit(mapping)}",
        f"adata.obs[{_lit(str(target))}] = pd.Categorical(adata.obs[{_lit(str(column))}].astype(str).map(merged), "
        f"categories={_lit(order)})",
    ])


def _protein_dsb(p: dict, ctx: _Ctx) -> Snippet:
    """DSB values of the antibodies (core/dsb.py): the counts read again from the input,
    the empty droplets from each sample's raw matrix (by the file names recorded)."""
    mode = p.get("mode")
    proteins, features = p.get("proteins"), p.get("features")
    if mode not in ("empty_droplets", "model_negative") or not proteins or not features:
        return Snippet(NOT_REPLAYABLE, ["# DSB protein normalisation: the run is not recorded. Recorded:",
                                        *_param_comment(p)], "DSB settings not recorded")
    samples = p.get("samples") or []
    if mode == "empty_droplets" and not all(isinstance(s, dict) and s.get("raw_files") for s in samples):
        return Snippet(NOT_REPLAYABLE, ["# DSB protein normalisation: the raw matrices are not recorded, "
                                        "so the empty droplets can't be read again. Recorded:",
                                        *_param_comment(p)], "raw matrices not recorded")
    _need(ctx, "dsb")
    settings = {"mode": mode, "denoise": bool(p.get("denoise", True)),
                "isotypes": [str(i) for i in p.get("isotypes") or []],
                "rna_window": p.get("rna_window"), "prot_window": p.get("prot_window"),
                "proteins": [str(x) for x in proteins], "features": [str(x) for x in features],
                "samples": [{"sample": str(s.get("sample")), "column": s.get("column"),
                             "raw_files": [str(f) for f in s.get("raw_files") or []],
                             "n_droplets": s.get("n_droplets")} for s in samples]}
    how = ("empty droplets of each sample's raw matrix" if mode == "empty_droplets"
           else "the cells themselves (dsb ModelNegativeADTnorm: no empty droplets)")
    return Snippet(EXACT, [
        "# DSB protein normalisation (Mule et al. 2022; dsb 2.0.1's method, core/dsb.py): the",
        f"# antibody counts against the background in {how},",
        "# then the per-cell technical component (isotypes + background mean) regressed out.",
        f"DSB = {_lit(settings)}",
        *_block('''
counts = protein_counts_for(adata, DSB["proteins"], DSB["features"])
iso = [DSB["proteins"].index(i) for i in DSB["isotypes"]]
if DSB["mode"] == "empty_droplets":
    norm = np.full(counts.shape, np.nan)
    cells = input_cells()
    for s in DSB["samples"]:
        files = raw_paths(s["raw_files"])
        bg = (background_from_h5(files[0], cells) if len(files) == 1
              else background_from_mex(*files, cells=cells))
        n, mean, sd = window_stats(bg, DSB["rna_window"], DSB["prot_window"])
        if n != s["n_droplets"]:
            print(f"note: {n:,} empty droplets in the window for {s['sample']}; scView used {s['n_droplets']:,}")
        order = [list(bg["features"]).index(f) for f in DSB["features"]]
        rows = (np.flatnonzero(adata.obs[s["column"]].astype(str).to_numpy() == s["sample"])
                if s["column"] else np.arange(adata.n_obs))
        norm[rows] = step1_empty(counts[rows], mean[order], sd[order])
else:
    norm = step1_model_negative(counts)[0]
if DSB["denoise"]:
    norm = step2(norm, iso)[0]
adata.obsm["protein_dsb"] = pd.DataFrame(norm.astype(np.float32), index=adata.obs_names,
                                         columns=DSB["proteins"])
del counts, norm
'''),
    ])


def _qc_state_label(p: dict, ctx: _Ctx) -> Snippet:
    out = str(p.get("output") or "")
    return Snippet(NOT_REPLAYABLE, [
        "# scView's QC state check (1 vs 2 Gaussian components per group on the RP share",
        "# of non-mito counts and % mito) is not replayed. To carry its label over, join",
        "# it from scView's cell metadata export by barcode, e.g.:",
        "#   meta = pd.read_csv('cell_metadata.csv', index_col=0)",
        f"#   adata.obs[{comment_text(repr(out), 80)}] = meta.loc[adata.obs_names, "
        f"{comment_text(repr(out), 80)}].astype('category')",
        "# Recorded:",
        *_param_comment(p),
    ], "scView's own QC state check")


def _gwas_traits(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(NOT_REPLAYABLE, [
        "# scView's GWAS trait links (each group's markers against the GWAS Catalog",
        f"# {comment_text(p.get('release'), 20)} release) read the data and add nothing to it: not replayed.",
        "# Recorded:",
        *_param_comment(p),
    ], "a lookup against the GWAS Catalog; adds no data")


def _doublet_threshold(p: dict, ctx: _Ctx) -> Snippet:
    t = p.get("threshold")
    if t == "scrublet":
        return Snippet(EXACT, _block('''
# Back to Scrublet's own calls.
if "predicted_doublet_scrublet" in adata.obs.columns:
    adata.obs["predicted_doublet"] = adata.obs.pop("predicted_doublet_scrublet").to_numpy(dtype=bool)
'''))
    if isinstance(t, (int, float)):
        return Snippet(EXACT, _block(f'''
if "predicted_doublet_scrublet" not in adata.obs.columns and "predicted_doublet" in adata.obs.columns:
    adata.obs["predicted_doublet_scrublet"] = adata.obs["predicted_doublet"].to_numpy(dtype=bool)
scores = adata.obs["doublet_score"].to_numpy(dtype=np.float64)
adata.obs["predicted_doublet"] = np.isfinite(scores) & (scores > {_lit(float(t))})
'''))
    return Snippet(NOT_REPLAYABLE, ["# Doublet threshold not recorded. Recorded:",
                                    *_param_comment(p)], "threshold not recorded")


def _packed(items: list[str], width: int = 92) -> list[str]:
    """Literal list items, several to a line (a 2,000-gene list one per line is 2,000 lines)."""
    lines, cur = [], ""
    for item in items:
        if cur and len(cur) + len(item) + 2 > width:
            lines.append("    " + cur.rstrip())
            cur = ""
        cur += item + ", "
    if cur:
        lines.append("    " + cur.rstrip())
    return lines


def _gene_programs(p: dict, ctx: _Ctx) -> Snippet:
    genes, k, seeds = p.get("genes"), p.get("k"), p.get("seeds")
    if not isinstance(genes, list) or not genes or not isinstance(k, int) or not isinstance(seeds, list):
        return Snippet(NOT_REPLAYABLE, ["# Gene programs: genes, k or seeds not recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items() if x != "genes"})],
                       "genes, k or seeds not recorded")
    _need(ctx, "gene_programs")
    src = str(p.get("source") or "")
    source = "raw" if src.startswith("log-normalised .raw") else "X" if src.startswith("log-normalised X") \
        else "counts"
    return Snippet(EXACT, [
        "# NOTE: the same seeds and settings as scView's run; another scikit-learn version, BLAS",
        "# or thread count can still change the result in the last digits or, rarely, which",
        "# replicate spectra k-means groups together, so exact reproduction is not guaranteed.",
        "PROGRAM_GENES = [",
        *_packed([repr(str(g)) for g in genes]),
        "]",
        f"share, usage, spectra = gene_programs(adata, PROGRAM_GENES, {_lit(int(k))}, "
        f"{_lit([int(x) for x in seeds])}, source={_lit(source)}, solver={_lit(p.get('solver', 'cd'))}, "
        f"init={_lit(p.get('init', 'random'))}, beta_loss={_lit(p.get('beta_loss', 'frobenius'))}, "
        f"tol={_lit(p.get('tol', 1e-4))}, max_iter={_lit(p.get('max_iter', 1000))}, "
        f"kmeans_seed={_lit(p.get('kmeans_seed', 1))}, kmeans_starts={_lit(p.get('kmeans_starts', 10))}, "
        f"target_sum={_lit(p.get('target_sum', 1e4))})",
        'adata.obsm["X_nmf_usage"] = share.astype(np.float32)',
        'adata.obsm["X_nmf_usage_raw"] = usage.astype(np.float32)',
    ])


def _cnv(p: dict, ctx: _Ctx) -> Snippet:
    genes, chroms = p.get("genes"), p.get("chromosomes")
    col, groups = p.get("reference_column"), p.get("reference_groups")
    if (not isinstance(genes, list) or not genes or not isinstance(chroms, list)
            or len(chroms) != len(genes) or not col or not isinstance(groups, list) or not groups):
        return Snippet(NOT_REPLAYABLE, ["# Copy number: genes or reference not recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items()
                                                         if x not in ("genes", "chromosomes")})],
                       "genes or reference not recorded")
    _need(ctx, "infer_cnv")
    return Snippet(EXACT, [
        "# Copy number inferred from expression (inferCNV-style): the genes scView placed on the",
        f"# {p.get('assembly', 'genome')} genome (Ensembl {p.get('ensembl_release', '?')}), in genomic order.",
        "CNV_GENES = [",
        *_packed([repr(str(g)) for g in genes]),
        "]",
        "CNV_CHROMOSOMES = [",
        *_packed([repr(str(c)) for c in chroms]),
        "]",
        f"cnv_x, cnv_score, cnv_leiden = infer_cnv(adata, CNV_GENES, CNV_CHROMOSOMES, {_lit(str(col))}, "
        f"{_lit([str(g) for g in groups])}, window={_lit(p.get('window', 100))}, "
        f"step={_lit(p.get('step', 10))}, threshold={_lit(p.get('threshold', 1.5))}, "
        f"lfc_clip={_lit(p.get('lfc_clip', 3.0))}, source={_lit(p.get('source', 'raw'))}, "
        f"target_sum={_lit(p.get('target_sum', 1e4))}, n_pcs={_lit(p.get('n_pcs', 50))}, "
        f"n_neighbors={_lit(p.get('n_neighbors', 15))}, resolution={_lit(p.get('resolution', 1.0))}, "
        f"seed={_lit(p.get('seed', 0))})",
        'adata.obsm["X_cnv"] = cnv_x',
        'adata.obs["cnv_score"] = cnv_score.astype(np.float32)',
        'adata.obs["cnv_leiden"] = pd.Categorical(cnv_leiden)',
    ])

def _keep_layout(p: dict, ctx: _Ctx) -> Snippet:
    key, seed, k = p.get("key"), p.get("seed"), p.get("n_neighbors")
    if not key or seed is None or k is None:
        return Snippet(NOT_REPLAYABLE, ["# A kept layout: its seed or neighbours are not recorded. "
                                        "Recorded:", *_param_comment(p)], "layout not recorded")
    if p.get("min_dist") is None and p.get("a") is not None and p.get("b") is not None:
        curve = f"a={_lit(p['a'])}, b={_lit(p['b'])}"
    else:
        curve = f"min_dist={_lit(p.get('min_dist', 0.5))}"
    n_pcs = f", n_pcs={_lit(p['n_pcs'])}" if p.get("n_pcs") is not None else ""
    return Snippet(EXACT, [
        "# A layout kept from scView's layout stability check: the graph rebuilt from the",
        "# same representation with this neighbour count, laid out with this seed.",
        f"sc.pp.neighbors(adata, n_neighbors={_lit(k)}, use_rep={_lit(p.get('use_rep') or 'X_pca')}"
        f"{n_pcs}, metric={_lit(p.get('metric') or 'euclidean')}, key_added=\"layout_check\")",
        f"sc.tl.umap(adata, {curve}, random_state={_lit(seed)}, neighbors_key=\"layout_check\", "
        f"key_added={_lit(str(key))})",
        'for k in ("layout_check_connectivities", "layout_check_distances"):',
        "    adata.obsp.pop(k, None)",
        'adata.uns.pop("layout_check", None)',
        f"adata.uns.pop({_lit(str(key))}, None)",
    ])

_WNN_CODE = '''
counts = protein_counts_for(adata, WNN["panel"], WNN["features"])
clr = wnn_clr(counts)[:, [WNN["panel"].index(x) for x in WNN["proteins"]]]
pcs = protein_pcs(scale_proteins(clr, SCALE_MAX), WNN["n_pcs_protein"])
if WNN["batch_key"]:
    pcs = harmony(pcs, adata.obs, WNN["batch_key"])
pcs = np.asarray(pcs, dtype=np.float32)
rna = np.asarray(adata.obsm[WNN["rep"]])[:, :WNN["n_pcs_rna"]].astype(np.float32)
out = wnn_neighbors(rna, pcs, WNN["k"], WNN["knn_range"], WNN["exact_max_cells"])
conn, dists = wnn_graphs(out["idx"], out["dist"])
adata.obs["wnn_rna_weight"] = out["rna_weight"].astype(np.float32)
adata.obs["wnn_protein_weight"] = out["protein_weight"].astype(np.float32)
adata.obsm["protein_pca"] = pcs
adata.obsp["wnn_connectivities"] = conn.copy()  # umap-learn trims the graph it is given
adata.obsp["wnn_distances"] = dists
adata.uns["wnn"] = {"connectivities_key": "wnn_connectivities", "distances_key": "wnn_distances",
                    "params": {"method": "umap", "n_neighbors": WNN["k"] + 1, "use_rep": WNN["rep"],
                               "n_pcs": WNN["n_pcs_rna"], "metric": "euclidean", "random_state": 0}}
sc.tl.umap(adata, neighbors_key="wnn", key_added="X_umap_wnn", min_dist=WNN["min_dist"], random_state=0)
adata.obsp["wnn_connectivities"] = conn
sc.pp.neighbors(adata, n_neighbors=WNN["k_protein_umap"], use_rep="protein_pca", key_added="protein",
                random_state=0)
sc.tl.umap(adata, neighbors_key="protein", key_added="X_umap_protein", min_dist=WNN["min_dist"],
           random_state=0)
for k in ("protein_connectivities", "protein_distances"):
    adata.obsp.pop(k, None)
for k in ("protein", "X_umap_wnn", "X_umap_protein"):
    adata.uns.pop(k, None)
'''


def _wnn(p: dict, ctx: _Ctx) -> Snippet:
    """WNN joint RNA + protein neighbours (core/wnn.py): weights, the WNN graph under its
    own key, the WNN and protein UMAPs, Leiden on the WNN graph."""
    need = ("proteins", "panel", "features", "rep", "n_pcs_rna", "n_pcs_protein")
    if any(not p.get(x) for x in need) or len(p["panel"]) != len(p["features"]):
        return Snippet(NOT_REPLAYABLE, ["# WNN: the panel or the PCs are not recorded. Recorded:",
                                        *_param_comment(p)], "panel or PCs not recorded")
    _need(ctx, "wnn")
    if p.get("batch_key"):
        _need(ctx, "harmony")
    keys = ("proteins", "panel", "features", "rep", "n_pcs_rna", "n_pcs_protein", "batch_key", "k",
            "knn_range", "k_protein_umap", "min_dist", "resolution", "n_iterations", "exact_max_cells")
    settings = {x: p.get(x) for x in keys}
    for x, default in (("k", 20), ("knn_range", 200), ("exact_max_cells", 4096), ("k_protein_umap", 15),
                       ("min_dist", 0.5), ("resolution", 1.0)):
        settings[x] = settings[x] if settings[x] is not None else default
    n_iter = p.get("n_iterations")
    passes = f", n_iterations={_lit(n_iter)}" if n_iter not in (None, -1) else ""
    return Snippet(EXACT, [
        "# WNN (Hao et al. 2021; Seurat's FindMultiModalNeighbors, core/wnn.py): per-cell RNA and",
        "# protein weights and a joint 20-NN graph, kept under its own key ('wnn'). Above",
        "# exact_max_cells cells the neighbour search is pynndescent's (seeded, approximate).",
        f"WNN = {_lit(settings)}",
        *_block(_WNN_CODE),
        'sc.tl.leiden(adata, resolution=WNN["resolution"], key_added="wnn_leiden", neighbors_key="wnn", '
        f'flavor="leidenalg"{passes})',
    ])


def _label_transfer(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core import label_transfer as lt

    genes, vote, label, prefix = p.get("novelty_genes"), p.get("genes"), p.get("label_column"), p.get("prefix")
    method, k, ref_file = p.get("method"), p.get("k"), p.get("reference_file")
    ok = (isinstance(genes, list) and genes and isinstance(vote, list) and vote
          and set(map(str, vote)) <= set(map(str, genes)) and isinstance(label, str) and label
          and isinstance(prefix, str) and prefix and method in lt.METHODS and isinstance(k, int)
          and isinstance(ref_file, str) and ref_file)
    if not ok:
        return Snippet(NOT_REPLAYABLE, ["# Reference mapping: its genes, reference file or settings are not "
                                        "recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items()
                                                         if x not in ("genes", "novelty_genes")})],
                       "settings not recorded")
    _need(ctx, "label_transfer")
    cols = lt.columns_for(prefix)
    status, notes = EXACT, []
    if p.get("renamed"):
        status = APPROXIMATE
        notes.append(f"scView matched genes after re-indexing {comment_text(p['renamed'], 80)}; here they "
                     "are matched by name, so the reference must name them as this dataset does.")
    if method == lt.METHOD_CELLTYPIST:
        status = APPROXIMATE
        notes.append("scView trained CellTypist on every shared gene; here on the recorded variable "
                     "genes only, so a few labels can differ.")
    code = [
        f"# Reference mapping onto {comment_text(p.get('reference'), 80)} (its column "
        f"{comment_text(label, 60)}), {comment_text(lt.METHOD_TEXT.get(method, method), 60)}, k={k}.",
        "# NOTE: scView read counts from the dataset or its source files; this reads them (or",
        "# log-normalised values) from the reference file and the data at this point.",
        *[f"# NOTE: {n}" for n in notes],
        "MAP_GENES = [",
        *_packed([repr(str(g)) for g in genes]),
        "]",
        "VOTE_GENES = [",
        *_packed([repr(str(g)) for g in vote]),
        "]",
        f"ref = read_input(reference_file({_lit(ref_file)}))",
        f"ref = ref[ref.obs[{_lit(label)}].notna().to_numpy()].copy()",
        "ref_log, _ = lt_expression(ref, MAP_GENES)",
        "query_log, _ = lt_expression(adata, MAP_GENES)",
        "ref_umap = np.asarray(ref.obsm[\"X_umap\"])[:, :2] if \"X_umap\" in ref.obsm else None",
        "where = {g: i for i, g in enumerate(MAP_GENES)}",
        f"mapped = lt_map(ref_log, query_log, ref.obs[{_lit(label)}].astype(str).to_numpy(), {_lit(method)}, "
        f"{_lit(k)}, np.array([where[g] for g in VOTE_GENES]), np.arange(len(MAP_GENES)), "
        f"n_pcs={_lit(p.get('n_pcs', lt.N_PCS))}, clip={_lit(lt.CLIP)}, seed={_lit(lt.SEED)}, ref_umap=ref_umap, "
        f"percentile={_lit(p.get('percentile', lt.UNMAPPED_PERCENTILE))}, genes=MAP_GENES)",
        f"adata.obs[{_lit(cols['label'])}] = pd.Categorical(mapped[\"label\"])",
        f"adata.obs[{_lit(cols['score'])}] = mapped[\"score\"]",
        f"adata.obs[{_lit(cols['unmapped'])}] = mapped[\"unmapped\"]",
        "if mapped[\"umap\"] is not None:",
        f"    adata.obsm[{_lit(cols['embedding'])}] = mapped[\"umap\"]",
        "del ref, ref_log, query_log",
    ]
    return Snippet(status, code, "; ".join(notes)[:200] if notes else "")


def _learned_gate_train(p: dict, ctx: _Ctx) -> Snippet:
    return Snippet(RECORDED, [
        f"# Learned gate {comment_text(p.get('name'), 80)} trained on {comment_text(p.get('n_selected'), 20)} "
        "selected cells: it adds nothing to the data.",
        "# The gate is written inline wherever a learned gate apply step follows. Recorded:",
        *_param_comment(p),
    ], "training only adds a gate to the library")


def _learned_gate_apply(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core import learned_gate as lg

    gate, targets = p.get("gate"), p.get("target_genes")
    col, score_col = p.get("column"), p.get("score_column")
    try:
        parsed = lg.parse_gate(gate) if isinstance(gate, dict) else None
    except lg.LearnedGateError:
        parsed = None
    ok = (parsed is not None and isinstance(targets, list) and len(targets) == len(parsed.genes)
          and isinstance(col, str) and col and isinstance(score_col, str) and score_col)
    if not ok:
        return Snippet(NOT_REPLAYABLE, ["# Learned gate: its model or columns are not recorded. Recorded:",
                                        *_param_comment({k: v for k, v in p.items() if k != "gate"})],
                       "gate not recorded")
    _need(ctx, "learned_gate")
    model = {k: gate[k] for k in ("name", "genes", "coef", "intercept", "mean", "std", "clip", "threshold")}
    here = [t if isinstance(t, str) else None for t in targets]
    return Snippet(EXACT, [
        f"# Learned gate {comment_text(parsed.name, 80)} (an L1 logistic regression on "
        f"{len(parsed.genes)} genes): score = the class-balanced",
        "# model probability, in = score >= threshold. A gene this dataset lacks takes the training mean.",
        f"# Genes found {comment_text(p.get('via') or 'by name', 120)}.",
        "# NOTE: scView read counts from the dataset or its source files; this reads them (or",
        "# log-normalised values) from the data at this point.",
        f"LEARNED_GATE = {_lit(model)}",
        f"GATE_GENES_HERE = {_lit(here)}",
        'gate_X = lg_expression(adata, [g if g is not None else "" for g in GATE_GENES_HERE])',
        "gate_score, gate_in = lg_apply(gate_X, LEARNED_GATE)",
        f'adata.obs[{_lit(col)}] = pd.Categorical(np.where(gate_in, "in", "out"), categories=["in", "out"])',
        f"adata.obs[{_lit(score_col)}] = gate_score",
        "del gate_X, gate_score, gate_in",
    ])


def _sample_code(column: object, mapping: object) -> list[str] | None:
    if not isinstance(column, str) or not column or not isinstance(mapping, dict) \
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        return None
    order = list(dict.fromkeys(mapping.values()))
    return [
        f"mapping = {_lit(mapping)}  # tag -> sample; doublets and negatives get none",
        f"adata.obs[{_lit(column)}] = pd.Categorical(adata.obs[\"hash_call\"].astype(object).map(mapping), "
        f"categories={_lit(order)})",
    ]


def _hashing_demux(p: dict, ctx: _Ctx) -> Snippet:
    from scview.core import hashing

    tags, method = p.get("tags"), p.get("method")
    features, sources = p.get("features") or {}, p.get("sources") or {}
    ok = (isinstance(tags, list) and tags and all(isinstance(t, str) for t in tags)
          and method in hashing.METHODS and isinstance(features, dict) and isinstance(sources, dict)
          and all(isinstance(features.get(t), str) for t in tags))
    if not ok:
        return Snippet(NOT_REPLAYABLE, ["# Cell hashing call: the tags or method are not recorded. Recorded:",
                                        *_param_comment(p)], "hashing call not recorded")
    _need(ctx, "hashing", "read_hashtags")
    feats = {t: features[t] for t in tags}
    srcs = {t: str(sources.get(t) or "") for t in tags}
    if method == hashing.HTODEMUX:
        q = p.get("positive_quantile", hashing.POSITIVE_QUANTILE)
        run = [f"call = htodemux(counts, tags, {_lit(float(q))})  # = Seurat 5.1.0 HTODemux"]
    else:
        run = ["call = hashed_drops(counts, tags)  # = DropletUtils 1.24 hashedDrops, defaults"]
    code = [
        "# Cell hashing: each cell's tag from its hashtag counts (scView's port, copied above).",
        f"tags = {_lit(list(tags))}",
        f"counts = read_hashtags(INPUT, adata.obs_names, {_lit(feats)}, {_lit(srcs)})",
        *run,
        "for col, values in call_columns(call).items():",
        "    adata.obs[col] = values",
        "adata.obsm[\"hto_counts\"] = pd.DataFrame(counts.astype(np.float32), index=adata.obs_names, columns=tags)",
    ]
    redo = p.get("sample_column")
    if isinstance(redo, dict):
        sample = _sample_code(redo.get("column"), redo.get("mapping"))
        if sample:
            code += ["# The sample column follows the new call.", *sample]
    return Snippet(EXACT, code)


def _hashing_samples(p: dict, ctx: _Ctx) -> Snippet:
    code = _sample_code(p.get("column"), p.get("mapping"))
    if code is None:
        return Snippet(NOT_REPLAYABLE, ["# Sample column from the hashing call: not recorded. Recorded:",
                                        *_param_comment(p)], "mapping not recorded")
    return Snippet(EXACT, ["# The sample column from the hashing call (tag -> sample name).", *code])


def _vdj_start(ctx: _Ctx) -> list[str]:
    """Start from no V(D)J columns at the first V(D)J step, as scView did (an input
    saved after the attach already has them)."""
    first = "vdj" not in ctx.helpers
    _need(ctx, "vdj")
    return ["clear(adata)  # start from no V(D)J columns"] if first else []


def _vdj_attach(p: dict, ctx: _Ctx) -> Snippet:
    name = str(p.get("file") or "")
    if not name:
        return Snippet(NOT_REPLAYABLE, ["# TCR/BCR contigs attached; the file is not recorded. Recorded:",
                                        *_param_comment(p)], "contig file not recorded")
    start = _vdj_start(ctx)
    return Snippet(EXACT, [
        "# TCR/BCR contigs joined to the cells by barcode, chains called and clonotypes defined",
        "# (identical CDR3 amino acids of the primary VJ and VDJ chains) as core/vdj.py does.",
        f"# Contig table: {comment_text(name, 160)} (sha256 {comment_text(p.get('sha256'), 64)});",
        "# scView keeps a copy beside the dataset's uploaded files (uploads/<dataset id>/).",
        *start,
        f"attach_file(adata, vdj_file({_lit(name)}), sample_column={_lit(p.get('sample_column'))}, "
        f"sample={_lit(p.get('sample'))}, replace={_lit(bool(p.get('replace')))}, "
        f"same_v_gene={_lit(bool(p.get('same_v_gene')))})",
    ])


def _vdj_clonotypes(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "vdj")
    return Snippet(EXACT, ["# Clonotypes recomputed from the stored chains.",
                           f"redefine(adata, same_v_gene={_lit(bool(p.get('same_v_gene')))})"])


def _vdj_remove(p: dict, ctx: _Ctx) -> Snippet:
    _need(ctx, "vdj")
    return Snippet(EXACT, ["clear(adata)  # every V(D)J column removed"])


def _step_skipped(p: dict, ctx: _Ctx) -> Snippet:
    """A step that does not apply to the data (core/pipeline.py ``StepNotApplicable``:
    doublet detection on Visium spots): skipped here too, with scView's reason."""
    step = comment_text(p.get("step") or "this step", 40)
    reason = comment_text(p.get("reason") or "it does not apply", 160)
    return Snippet(EXACT, [f"# {step} was skipped: {reason}."])


def _tf_motif_enrichment(p: dict, ctx: _Ctx) -> Snippet:
    """Upstream TF candidates (core/tf_motifs.py): a result only, the data are unchanged; it
    needs scView's promoter motif table, so it is not computed again here."""
    return Snippet(NOT_REPLAYABLE, ["# Upstream TF candidates (promoter motifs): a result only, the data are "
                                    "unchanged. Recorded:", *_param_comment(p)], "result only")


def _run_errors(p: dict, ctx: _Ctx) -> Snippet:
    """A run that finished with errors (core/step_failures.py): which steps failed and were
    skipped (not in the history, so not replayed), and the earlier results it removed as
    stale, removed here too."""
    failed = p.get("failed") if isinstance(p.get("failed"), dict) else {}
    skipped = p.get("skipped") if isinstance(p.get("skipped"), dict) else {}
    cleared = p.get("cleared") if isinstance(p.get("cleared"), dict) else {}
    code = ["# The run of the steps above finished with errors: these steps were not run."]
    for step, err in failed.items():
        code.append(f"#   {comment_text(step, 40)} failed: {comment_text(err, 100)}")
    for step, because in skipped.items():
        code.append(f"#   {comment_text(step, 40)} skipped: it needs {comment_text(because, 40)}")
    names = {where: [str(k) for k in keys] for where, keys in cleared.items()
             if isinstance(keys, list) and keys}
    if names:
        code.append("# Results of an earlier run they would have replaced were removed as stale:")
    for where in ("obsm", "varm", "obsp", "uns"):
        if names.get(where):
            code.append(f"for k in {_lit(names[where])}:")
            code.append(f"    adata.{where}.pop(k, None)")
    for where in ("var", "obs"):
        if names.get(where):
            code.append(f"adata.{where} = adata.{where}.drop(columns=[c for c in {_lit(names[where])} "
                        f"if c in adata.{where}.columns])")
    if names.get("raw"):
        code.append("adata.raw = None")
    return Snippet(EXACT, code)


def _spatial_autocorr(p: dict, ctx: _Ctx) -> Snippet:
    genes, src, kind = p.get("genes"), str(p.get("matrix_source") or ""), p.get("neighbors")
    if isinstance(genes, str):
        genes = p.get("genes_used") if p.get("genes_used") in ("hvg", "all") else genes
    ok_genes = genes in ("hvg", "all") or (isinstance(genes, list) and genes)
    layer = src[len("layer:"):] if src.startswith("layer:") else None
    if src.startswith("log-normalised .raw"):
        values = ["M, names = adata.raw.X, adata.raw.var_names"]
    elif src.startswith("log-normalised X"):
        values = ["M, names = adata.X, adata.var_names"]
    elif layer:
        values = [f"M, names = adata.layers[{_lit(layer)}], adata.var_names"]
    elif src.startswith("shifted log"):
        _need(ctx, "find_counts")
        values = ["counts = sp.csr_matrix(find_counts(adata), dtype=np.float64)",
                  "totals = np.asarray(counts.sum(axis=1)).ravel()",
                  "M = sp.csr_matrix(sp.diags(np.divide(1e4, totals, out=np.zeros_like(totals), "
                  "where=totals > 0)) @ counts)",
                  "M.data = np.log1p(M.data)",
                  "names = adata.var_names"]
    else:
        values = []
    if not ok_genes or kind not in ("grid", "knn") or not values:
        return Snippet(NOT_REPLAYABLE, ["# Spatially variable genes: genes, values or neighbours not recorded. "
                                        "Recorded:", *_param_comment({x: v for x, v in p.items() if x != "genes"})],
                       "settings not recorded")
    _need(ctx, "spatial_autocorr")
    gene_lines = ([f"SPATIAL_GENES = {_lit(genes)}"] if isinstance(genes, str)
                  else ["SPATIAL_GENES = [", *_packed([repr(str(g)) for g in genes]), "]"])
    return Snippet(EXACT, [
        f"# Spatially variable genes: Moran's I per gene ({'Visium grid' if kind == 'grid' else 'kNN'} "
        f"neighbours, {p.get('n_neighs', 6)} per spot, within each section), = squidpy 1.6's",
        "# spatial_neighbors + spatial_autocorr defaults; saved where squidpy saves it.",
        *values,
        *gene_lines,
        f"adata.uns[\"moranI\"] = spatial_moran(adata, M, names, SPATIAL_GENES, "
        f"{_lit(p.get('library_id'))}, grid={_lit(kind == 'grid')})",
    ])


def _spatial_domains(p: dict, ctx: _Ctx) -> Snippet:
    """Spatial domains (core/spatial_domains.py) at the k the run ended on; no scan: the
    stability curve only advised on k."""
    column, k, rep, n_pcs, n_hops, seed = (p.get(x) for x in ("column", "k", "rep", "n_pcs", "n_hops", "seed"))
    grid = p.get("grid_neighbours")
    if (not isinstance(column, str) or not isinstance(k, int) or not isinstance(rep, str) or not isinstance(n_pcs, int)
            or not isinstance(n_hops, int) or not isinstance(seed, int)):
        return Snippet(NOT_REPLAYABLE, ["# Spatial domains: settings not recorded. Recorded:", *_param_comment(p)],
                       "settings not recorded")
    _need(ctx, "spatial_domains")
    return Snippet(EXACT, [
        f"# Spatial domains (CellCharter's method, Varrone et al. 2024): k = {k} on the first {n_pcs} components of "
        f"{comment_text(rep, 40)}, {n_hops} hops.",
        f"adj = spatial_graph(np.asarray(adata.obsm[\"spatial\"])[:, :2], spatial_sections(adata), {_lit(grid)})",
        f"features = aggregate_neighbors(adj, np.asarray(adata.obsm[{_lit(rep)}])[:, :{n_pcs}], {n_hops})",
        f"domains, _ = fit_domains(features, {k}, {seed})",
        f"adata.obs[{_lit(column)}] = pd.Categorical.from_codes(domains, categories=[str(i) for i in range({k})])",
    ])


def _nhood_enrichment(p: dict, ctx: _Ctx) -> Snippet:
    """Neighbourhood enrichment (core/spatial_domains.py, = squidpy's), saved where squidpy saves it."""
    from scview.core.spatial_domains import squidpy_key

    column, labels, n_perms, seed = (p.get(x) for x in ("column", "labels", "n_perms", "seed"))
    if (not isinstance(column, str) or not isinstance(labels, list) or len(labels) < 2 or not isinstance(n_perms, int)
            or not isinstance(seed, int)):
        return Snippet(NOT_REPLAYABLE, ["# Neighbourhood enrichment: settings not recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items() if x != "labels"})],
                       "settings not recorded")
    _need(ctx, "spatial_domains")
    return Snippet(EXACT, [
        f"# Neighbourhood enrichment of {comment_text(column, 60)}, {n_perms} permutations (= squidpy's "
        "nhood_enrichment); the same seed gives the same z here.",
        f"codes = pd.Categorical(adata.obs[{_lit(column)}].astype(str), categories={_lit([str(x) for x in labels])}).codes",
        "keep = codes >= 0",
        f"adj = spatial_graph(np.asarray(adata.obsm[\"spatial\"])[:, :2], spatial_sections(adata), {_lit(p.get('grid_neighbours'))})",
        f"z, count = nhood_enrichment(adj[keep][:, keep], codes[keep], {len(labels)}, spatial_sections(adata)[keep], "
        f"{n_perms}, {seed})",
        f"adata.uns[{_lit(squidpy_key(column))}] = {{\"zscore\": z, \"count\": count}}",
    ])


def _spatial_niches(p: dict, ctx: _Ctx) -> Snippet:
    """Composition niches (core/spatial_domains.py; Schurch et al. 2020)."""
    column, label, labels, window, n, seed = (p.get(x) for x in ("column", "label_column", "labels", "k_window",
                                                                 "n_niches", "seed"))
    if (not isinstance(column, str) or not isinstance(label, str) or not isinstance(labels, list) or len(labels) < 2
            or not isinstance(window, int) or not isinstance(n, int) or not isinstance(seed, int)):
        return Snippet(NOT_REPLAYABLE, ["# Composition niches: settings not recorded. Recorded:",
                                        *_param_comment({x: v for x, v in p.items() if x != "labels"})],
                       "settings not recorded")
    _need(ctx, "spatial_domains")
    return Snippet(EXACT, [
        f"# Composition niches (Schurch et al. 2020): k-means on the {comment_text(label, 60)} fractions among each "
        f"observation's {window} nearest, {n} niches.",
        f"codes = pd.Categorical(adata.obs[{_lit(label)}].astype(str), categories={_lit([str(x) for x in labels])}).codes",
        "keep = codes >= 0",
        f"fractions = window_fractions(np.asarray(adata.obsm[\"spatial\"])[:, :2][keep], codes[keep], {len(labels)}, "
        f"spatial_sections(adata)[keep], {window})",
        "niches = np.full(adata.n_obs, -1, dtype=np.int32)",
        f"niches[keep] = fit_niches(fractions, {n}, {seed})",
        f"adata.obs[{_lit(column)}] = pd.Categorical.from_codes(niches, categories=[str(i) for i in range({n})])",
    ])


_STEP_CODE: dict[str, Callable[[dict, _Ctx], Snippet]] = {
    "ingest": _ingest,
    "integrate": _integrate,
    "reset_to_counts": _reset_to_counts,
    "qc_metrics": _qc_metrics,
    "doublet_detection": _doublet_detection,
    "filtering": _filtering,
    "ambient_correction": _ambient_correction,
    "normalization": _normalization,
    "log_transform": _log_transform,
    "highly_variable_genes": _hvg,
    "scaling": _scaling,
    "pca": _pca,
    "uncorrected_embedding": _uncorrected_embedding,
    "batch_correction": _batch_correction,
    "neighbors": _neighbors,
    "clustering": _clustering,
    "embeddings": _embeddings,
    "marker_genes": _marker_genes,
    "enrichment": _enrichment,
    "cell_cycle": _cell_cycle,
    "dissociation_stress": _dissociation_stress,
    "cell_type_annotation": _cell_type_annotation,
    "subcluster": _subcluster,
    "keep_resolution": _keep_resolution,
    "keep_hvg_setting": _keep_hvg_setting,
    "pseudotime": _pseudotime,
    "paga": _paga,
    "annotation_confidence": _annotation_confidence,
    "annotation_confidence_unknown": _annotation_confidence,
    "label_cells": _label_cells,
    "undo_label_cells": _label_cells,
    "doublet_threshold": _doublet_threshold,
    "annotation_crosscheck": _annotation_crosscheck,
    "merge_clusters": _merge_clusters,
    "protein_dsb": _protein_dsb,
    "qc_state_label": _qc_state_label,
    "gwas_traits": _gwas_traits,
    "mito_refilter": _mito_refilter,
    "gene_programs": _gene_programs,
    "cnv": _cnv,
    "keep_layout": _keep_layout,
    "label_transfer": _label_transfer,
    "hashing_demux": _hashing_demux,
    "hashing_samples": _hashing_samples,
    "vdj_attach": _vdj_attach,
    "vdj_clonotypes": _vdj_clonotypes,
    "vdj_remove": _vdj_remove,
    "spatial_autocorr": _spatial_autocorr,
    "tf_motif_enrichment": _tf_motif_enrichment,
    "bulk_deconvolution": _bulk_deconvolution,
    "learned_gate_train": _learned_gate_train,
    "learned_gate_apply": _learned_gate_apply,
    "wnn": _wnn,
    "spot_deconvolution": _spot_deconvolution,
    "spatial_domains": _spatial_domains,
    "nhood_enrichment": _nhood_enrichment,
    "spatial_niches": _spatial_niches,
    "run_errors": _run_errors,
    "doublet_detection_skipped": _step_skipped,
}


def _snippet(step: dict, ctx: _Ctx) -> Snippet:
    """One step's code; a step type this version does not know, or one whose code
    cannot be written from its record, gets a placeholder comment."""
    name = str(step.get("step") or "")
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    fn = _STEP_CODE.get(name)
    if fn is None:
        return Snippet(UNKNOWN, ["# This scView version has no code for this step type; it is",
                                 "# not replayed. Recorded:", *_param_comment(params)],
                       "step type unknown to this version")
    try:
        return fn(params, ctx)
    except Exception as e:  # a malformed record never breaks the script
        return Snippet(UNKNOWN, [f"# Could not write code for this step ({comment_text(e, 120)}). "
                                 "Recorded:", *_param_comment(params)], "record not readable")


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

_READ_INPUT = '''
def read_input(path):
    """The dataset as scView imported it: .h5ad as is; 10x .h5 / matrix folders and
    .loom read as scView reads them (gene features only)."""
    p = str(path)
    if p.endswith(".h5ad"):
        adata = sc.read_h5ad(p)
    elif p.endswith(".h5"):
        adata = sc.read_10x_h5(p, gex_only=True)
    elif p.endswith(".loom"):
        adata = sc.read_loom(p)
    else:
        adata = sc.read_10x_mtx(p, var_names="gene_symbols", gex_only=True)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata
'''

_SUMMARY = '''
print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")
if "highly_variable" in adata.var.columns:
    print(f"variable genes: {int(adata.var['highly_variable'].sum()):,}")
for col in adata.obs.columns:
    if str(col).startswith("scview_") and adata.obs[col].dtype.name == "category":
        print(f"{col}: {adata.obs[col].nunique()} groups")
adata.write_h5ad(OUTPUT)
print("written", OUTPUT)
'''


def _steps(adata) -> list[dict]:
    hist = provenance.read_provenance(adata).get("history", [])
    return [h for h in hist if isinstance(h, dict)]


def describe(adata) -> list[dict[str, Any]]:
    """Each recorded step with how the script replays it (``status`` and ``note``)."""
    ctx = _Ctx(adata=adata)
    out = []
    for h in _steps(adata):
        snip = _snippet(h, ctx)
        out.append({"status": snip.status, "note": snip.note})
    return out


def input_name(adata) -> str:
    """The file the script reads by default: the source's file name, else input.h5ad."""
    name = str(provenance.read_provenance(adata).get("source", {}).get("original_filename") or "")
    if not name or "," in name or "+" in name:
        return "input.h5ad"
    return name


def build_script(adata, *, dataset_id: str = "", dataset_name: str = "",
                 input_file: dict | None = None, app_version: str | None = None) -> str:
    """The replay script for ``adata``'s recorded history.

    ``input_file`` (``run_manifest.input_file``): the stored source file's name and
    content hash, quoted in the header so the right input can be checked."""
    ctx = _Ctx(adata=adata)
    steps = _steps(adata)
    version = app_version or provenance._scview_version()
    source = provenance.read_provenance(adata).get("source", {}) or {}
    body: list[str] = []
    counts = {s: 0 for s in STATUSES}
    for i, h in enumerate(steps, 1):
        snip = _snippet(h, ctx)
        counts[snip.status] = counts.get(snip.status, 0) + 1
        when = comment_text(h.get("timestamp") or "time not recorded", 40)
        ver = comment_text(h.get("scview_version") or "?", 20)
        body.append("")
        body.append(f"# --- Step {i}: {comment_text(h.get('step'), 60)} "
                    f"({comment_text(h.get('tool') or '', 80)}) ---")
        body.append(f"# Ran {when} in scView {ver}; replay: {snip.status}"
                    + (f" ({comment_text(snip.note, 120)})" if snip.note else ""))
        if (h.get("params") or {}).get("chunked") is True:
            body.append("# scView ran it in large-dataset mode (matrices on disk, in row blocks): the "
                        "same results; this script runs it in memory.")
        effect = h.get("effect") if isinstance(h.get("effect"), dict) else {}
        if effect.get("n_cells") is not None:
            body.append(f"# scView had {comment_text(effect.get('n_cells'), 20)} cells x "
                        f"{comment_text(effect.get('n_genes'), 20)} genes after it.")
        body.extend(snip.code)

    header = [
        "#!/usr/bin/env python",
        f"# Replays the processing scView recorded for dataset {comment_text(dataset_name, 120)!s}",
        f"# (id {comment_text(dataset_id, 60)}): {len(steps)} step(s), from the dataset's provenance.",
        f"# Written by scView {comment_text(version, 20)} on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "#",
        "# Usage:  python this_script.py INPUT [OUTPUT.h5ad]",
        "# INPUT is the dataset as it was before the first step: an .h5ad, a 10x .h5 or",
        "# matrix folder, or a .loom.",
    ]
    if source:
        header.append(f"# Recorded source: {comment_text(source.get('original_filename') or '?', 160)} "
                      f"({comment_text(source.get('format') or '?', 30)}, "
                      f"{comment_text(source.get('n_cells'), 12)} cells x "
                      f"{comment_text(source.get('n_genes'), 12)} genes when added).")
    if input_file and input_file.get("sha256"):
        header.append(f"# scView's stored copy: {comment_text(input_file.get('name'), 120)}, "
                      f"sha256 {comment_text(input_file['sha256'], 64)}.")
    header += [
        "#",
        "# Each step says how it is replayed: exact (the same computation), approximate",
        "# (closest scanpy code; see its NOTE), recorded (scView's stored result written in),",
        "# not_replayable or unknown (a comment with the recorded values).",
        "# Here: " + ", ".join(f"{n} {s}" for s, n in counts.items() if n) + "."
        if steps else "# No steps are recorded for this dataset.",
        "# PCA, UMAP, Harmony and the neighbour graph are seeded, but thread count and BLAS",
        "# can move their last digits, and with them a few cells' clusters.",
    ]
    n_chunked = sum(1 for h in steps if (h.get("params") or {}).get("chunked") is True)
    if n_chunked:
        header += [
            f"# scView ran {n_chunked} of these steps in large-dataset mode: the matrices stayed",
            "# on disk and were read in row blocks (the same results; scaling and PCA to float",
            "# precision). Run in memory, this script needs the memory the dataset would take.",
        ]
    header += [
        "",
        "import sys",
        "",
        "import anndata as ad",
        "import numpy as np",
        "import pandas as pd",
        "import scanpy as sc",
        "import scipy.sparse as sp",
        "",
        f"INPUT = sys.argv[1] if len(sys.argv) > 1 else {_lit(input_name(adata))}",
        'OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "replayed.h5ad"',
    ]
    helpers = [_HELPERS[name].strip("\n") for name in _HELPER_ORDER if name in ctx.helpers]
    out = header + [""] + _READ_INPUT.strip("\n").splitlines()
    for h in helpers:
        out += ["", ""] + h.splitlines()
    out += ["", "", "adata = read_input(INPUT)", 'print(f"read {adata.n_obs:,} cells x '
            '{adata.n_vars:,} genes")']
    out += body
    out += [""] + _SUMMARY.strip("\n").splitlines()
    return "\n".join(out) + "\n"
