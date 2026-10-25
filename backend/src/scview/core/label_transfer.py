"""Reference mapping: label the open dataset's cells from an annotated reference (C042).

The user picks another library dataset with a cell-type column (the reference). Its
cells and the open dataset's (the query) are put in one space, each query cell takes
the labels of its ``k`` nearest reference cells (a vote weighted by distance), and a
cell further from the reference than the reference's own cells are from each other
is flagged "not in reference".

Three ways to label the cells:

- ``projection`` (default, like scanpy's ``sc.tl.ingest``): log-normalised values of
  the reference's variable genes, scaled with the REFERENCE's means and spreads, a PCA
  fitted on the reference alone, the query projected into it. The reference is not
  changed, so a query cell sits where a reference cell with the same expression would.
  It assumes no batch effect between the two: a query from another lab or protocol is
  shifted, and lands near the wrong cells or far from all of them.
- ``integration``: the same genes scaled and a PCA fitted on both together, then
  Harmony (Korsunsky et al. 2019, Nat Methods 16:1289) with reference/query as the
  batch, so a shift shared by every query cell is removed before the vote. It can
  also blend a state found only in the query into its nearest reference type, which
  the "not in reference" check then sees less often.
- ``celltypist`` (optional): a CellTypist logistic-regression model trained on the
  reference (``celltypist.train``, SGD mini-batches) predicts each query cell; its
  score is the model's probability for the label (~20 s to train on 6.5k cells).

The "not in reference" check tunes nothing on the query (it has no labels). It uses
the genes variable in EITHER dataset (a type the reference lacks keeps the genes that
mark it; the reference's variable genes alone may leave them out) in a joint PCA of
both (Harmony-corrected for ``integration``; ``projection`` assumes no batch effect,
so its check adds no correction either). Each query cell's mean distance to its
``k`` nearest reference cells is divided by the median of those neighbours' own mean
distances to their ``k`` nearest other reference cells (``lt_novelty``, a local
density ratio after the local outlier factor). The reference's own cells get the
same ratio, leave one out; a query cell above the reference's 99th percentile is
"not in reference", so about 1% of query cells are flagged by chance even when the
reference holds every type. A cluster with most of its cells flagged is a candidate
state the reference lacks.

Why not the plain distance in the vote's space: on Kang split by donor (4 donors as
reference with B cells removed, the other 4 as query) a 99th-percentile cut on the
mean kNN distance in the reference's PCA flagged 1.5% of the query's B cells (0.1% of
the rest): the reference's PCA has no axis for what marks B cells, and projected B
cells land among dense types. The check as built flagged 48% of them (0.8% of the
rest); with pDC removed instead 98%, Mk 62%, NK 36%, CD14 monocytes 4% (CD16
monocytes and DCs resemble them). These numbers chose the design, on this one split.
Being flagged does not mean a cell is a new type: low-quality cells, doublets and a
shift between the datasets flag too, and a type the reference lacks can pass
unflagged when it resembles one it has.

The functions in ``REPLAY_CODE`` use numpy, scipy, pandas, anndata, scanpy,
scikit-learn and harmonypy only: the replay script (``core/replay_script.py``)
copies them in, so it maps the cells as scView did.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import provenance

logger = logging.getLogger(__name__)

METHOD_PROJECTION = "projection"
METHOD_INTEGRATION = "integration"
METHOD_CELLTYPIST = "celltypist"
METHODS = (METHOD_PROJECTION, METHOD_INTEGRATION, METHOD_CELLTYPIST)

DEFAULT_K = 15  # sc.tl.ingest's default (the reference's neighbour count, 15 by default)
MIN_K = 3
MAX_K = 100
N_HVG = 2000  # variable genes of the reference, among the shared genes
N_PCS = 30
CLIP = 10.0  # scaled values clipped, as scView's scaling step does
TARGET_SUM = 1e4
SEED = 0
UNMAPPED_PERCENTILE = 99.0
# A cluster with more than this share of its cells "not in reference" is a candidate
# state absent from the reference. 30 times the 1% chance rate: with B cells left out of
# a Kang reference, the query's B cluster was 40% unmapped (other clusters at most 6%).
CLUSTER_UNMAPPED_SHARE = 0.3
# Fewer shared genes than this and the space is built on too little to place cells:
# a reference of another species, a pre-filtered gene list, or Ensembl ids against
# symbols that could not be matched.
MIN_SHARED_GENES = 500
MIN_LABELS = 2
MAX_LABELS = 200
MIN_REFERENCE_CELLS = 50
MIN_QUERY_CELLS = 10
HIST_BINS = 20
# CellTypist training on the reference (method C): SGD mini-batches, a fixed seed.
CT_EPOCHS = 10
CT_BATCH_SIZE = 1000

UNS_KEY = "scview_label_transfer"
STEP = "label_transfer"

METHOD_TEXT = {
    METHOD_PROJECTION: "projected onto the reference",
    METHOD_INTEGRATION: "integrated with the reference (Harmony)",
    METHOD_CELLTYPIST: "CellTypist model trained on the reference",
}

NOTE = (
    "\"Not in reference\" means a cell sits further from its nearest reference cells, "
    "measured against how closely those cells sit to each other, than 99% of the "
    "reference's own cells do. It does not mean the cell is a new type: low-quality cells, "
    "doublets and a shift between the datasets flag too, and a type the reference lacks "
    "can pass unflagged when it resembles one it has. About 1% of cells are flagged by chance."
)


class LabelTransferError(RuntimeError):
    """A transfer that cannot run as asked (``status`` is the HTTP code)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class LabelTransferCancelled(RuntimeError):
    """The client went away: the run stopped and nothing was saved."""


# ---------------------------------------------------------------------------
# The computation (REPLAY_CODE: copied into the replay script)
# ---------------------------------------------------------------------------


def lt_lognorm(M, target_sum=1e4):
    """Counts -> log1p of counts per ``target_sum`` (CP10k), float32 CSR."""
    X = sp.csr_matrix(M, dtype=np.float32, copy=True)
    totals = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    X = sp.csr_matrix(sp.diags(scale.astype(np.float32)) @ X)
    np.log1p(X.data, out=X.data)
    return X


def lt_variable_genes(ref_log, n_top):
    """Positions of the reference's ``n_top`` most variable genes (scanpy's seurat
    flavour on log-normalised values), in gene order."""
    import scanpy as sc

    a = ad.AnnData(X=ref_log)
    n_top = int(min(n_top, ref_log.shape[1]))
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_top)
    return np.flatnonzero(a.var["highly_variable"].to_numpy())


def lt_scale_fit(X):
    """Column means and spreads (ddof=1, as sc.pp.scale); a constant column gets 1."""
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32, copy=False)
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, ddof=1, dtype=np.float64) if X.shape[0] > 1 else np.ones(X.shape[1])
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def lt_scale(X, mean, std, clip=10.0):
    """Dense float32 (X - mean) / std, clipped at +-``clip``."""
    X = X.toarray() if sp.issparse(X) else np.array(X, dtype=np.float32)
    X = X.astype(np.float32, copy=False)
    X -= mean
    X /= std
    np.clip(X, -clip, clip, out=X)
    return X


def lt_space(ref_x, query_x, method, n_pcs=30, clip=10.0, seed=0):
    """Reference and query cells in one space: (reference coordinates, query
    coordinates). ``projection``: scaled by the reference, PCA fitted on the reference,
    the query projected; ``joint``: scaled and PCA on both together; ``integration``:
    ``joint`` then Harmony with reference/query as the batch."""
    from sklearn.decomposition import PCA

    n_ref = ref_x.shape[0]
    if method == "projection":
        mean, std = lt_scale_fit(ref_x)
        zr = lt_scale(ref_x, mean, std, clip)
        n = int(min(n_pcs, min(zr.shape) - 1))
        pca = PCA(n_components=n, svd_solver="arpack", random_state=seed).fit(zr)
        ref_rep = pca.transform(zr).astype(np.float32)
        del zr
        zq = lt_scale(query_x, mean, std, clip)
        return ref_rep, pca.transform(zq).astype(np.float32)
    both = sp.vstack([sp.csr_matrix(ref_x), sp.csr_matrix(query_x)], format="csr")
    mean, std = lt_scale_fit(both)
    z = lt_scale(both, mean, std, clip)
    del both
    n = int(min(n_pcs, min(z.shape) - 1))
    out = PCA(n_components=n, svd_solver="arpack", random_state=seed).fit_transform(z).astype(np.float32)
    del z
    if method == "integration":
        import harmonypy

        batch = pd.DataFrame({"batch": ["reference"] * n_ref + ["query"] * (out.shape[0] - n_ref)})
        Z = np.asarray(harmonypy.run_harmony(out, batch, ["batch"], random_state=seed, verbose=False).Z_corr)
        if Z.shape[0] != out.shape[0] and Z.ndim == 2 and Z.shape[1] == out.shape[0]:
            Z = Z.T  # an older harmonypy returns PCs x cells
        out = np.ascontiguousarray(Z, dtype=np.float32)
    return out[:n_ref], out[n_ref:]


def lt_neighbours(ref_rep, points, k, skip_self=False):
    """Distances and indices of each point's ``k`` nearest reference cells (exact,
    Euclidean); ``skip_self``: the points are the reference itself, each cell's own
    entry left out (leave one out)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1 if skip_self else k).fit(ref_rep)
    d, idx = nn.kneighbors(points)
    if skip_self:
        d, idx = d[:, 1:], idx[:, 1:]
    return d.astype(np.float32), idx


def lt_weights(d):
    """Gaussian weights of each cell's neighbours, its own mean neighbour distance as
    the width: near neighbours count more, and the scale needs no tuning."""
    sigma = d.mean(axis=1, keepdims=True)
    sigma[sigma <= 0] = 1.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def lt_vote(idx, w, ref_codes, n_labels):
    """Each cell's label (most weight among its neighbours) and its weight share."""
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    votes = np.zeros((idx.shape[0], n_labels), dtype=np.float64)
    np.add.at(votes, (rows, ref_codes[idx].ravel()), w.ravel())
    total = votes.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    share = votes / total
    return share.argmax(axis=1), share.max(axis=1).astype(np.float32)


def lt_place(ref_umap, idx, w):
    """Query cells on the reference's UMAP: the weighted mean of their neighbours'
    positions."""
    pts = np.asarray(ref_umap, dtype=np.float32)[idx]
    return ((pts * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]).astype(np.float32)


def lt_celltypist(ref_log, ref_labels, query_log, genes, seed=0, epochs=10, batch_size=1000):
    """CellTypist trained on the reference (SGD mini-batches, ``celltypist.train``):
    each query cell's best label and its probability, the query scaled and clipped as
    CellTypist's own classifier does before predicting."""
    import celltypist

    genes = pd.Index([str(g) for g in genes])
    # Mini-batches whenever the reference holds more than one batch: on Kang's 6.5k reference
    # cells they took 20 s (86% of the held-out donors right), plain SGD 150 s (83%).
    model = celltypist.train(ad.AnnData(X=ref_log, var=pd.DataFrame(index=genes)),
                             labels=[str(x) for x in ref_labels], use_SGD=True,
                             mini_batch=ref_log.shape[0] > batch_size, epochs=epochs, batch_size=batch_size,
                             check_expression=False, n_jobs=1, random_state=seed)
    pos = genes.get_indexer(list(model.features))
    labels, scores = [], []
    for start in range(0, query_log.shape[0], 2000):  # blocks: a dense float64 copy of each
        q = query_log[start:start + 2000][:, pos]
        q = q.toarray() if sp.issparse(q) else np.asarray(q)
        q = (q - model.scaler.mean_) / model.scaler.scale_
        q[q > 10] = 10
        _, prob, label = model.predict_labels_and_prob(q, mode="best match")
        labels.append(np.asarray(label).ravel().astype(str))
        scores.append(np.asarray(prob).max(axis=1).astype(np.float32))
    return np.concatenate(labels), np.concatenate(scores)


def lt_novelty(ref_rep, query_rep, k):
    """Each query cell's distance ratio and the reference's own (leave one out): the
    cell's mean distance to its ``k`` nearest reference cells over the median of those
    neighbours' own mean distances to their ``k`` nearest other reference cells, a
    local-density ratio after the local outlier factor (Breunig et al. 2000, SIGMOD).
    A ratio, not a raw distance: a dense type's cells sit closer together than a rare
    type's, so one distance cut flags rare types and passes cells between dense ones."""
    d, idx = lt_neighbours(ref_rep, query_rep, k)
    rd, ridx = lt_neighbours(ref_rep, ref_rep, k, skip_self=True)
    own = rd.mean(axis=1)
    floor = max(float(np.median(own)) * 1e-6, 1e-12)
    ref_ratio = own / np.maximum(np.median(own[ridx], axis=1), floor)
    ratio = d.mean(axis=1) / np.maximum(np.median(own[idx], axis=1), floor)
    return ratio.astype(np.float32), ref_ratio.astype(np.float32)


def lt_map(ref_log, query_log, ref_labels, method, k, vote_genes, novelty_genes, n_pcs=30, clip=10.0,
           seed=0, ref_umap=None, percentile=99.0, genes=None, stop=None):
    """Map the query onto the reference. ``ref_log`` / ``query_log``: log-normalised
    values on the shared genes (same columns); ``ref_labels``: the reference cells'
    labels; ``vote_genes``: columns of the reference's variable genes (the space the
    labels are voted in); ``novelty_genes``: columns variable in either dataset (the
    "not in reference" check, so a type the reference lacks keeps the genes that mark
    it). Returns a dict: ``label`` and ``score`` per query cell, ``ratio`` (the
    distance ratio of ``lt_novelty``), ``unmapped``, ``threshold`` (the reference's
    ``percentile`` of its own ratio), ``umap`` (query cells on the reference's UMAP,
    or None). The check's space is a joint PCA of both datasets, Harmony-corrected for
    ``integration`` only: projection assumes no batch effect, so it adds no correction
    either. ``celltypist`` needs ``genes`` (the shared genes' names)."""
    def check():
        if stop is not None and stop():
            raise InterruptedError("stopped")

    labels = pd.Categorical(np.asarray(ref_labels, dtype=object).astype(str))
    vote_space = "integration" if method == "integration" else "projection"
    ref_rep, query_rep = lt_space(ref_log[:, vote_genes], query_log[:, vote_genes], vote_space,
                                  n_pcs=n_pcs, clip=clip, seed=seed)
    check()
    d, idx = lt_neighbours(ref_rep, query_rep, k)
    w = lt_weights(d)
    best, score = lt_vote(idx, w, labels.codes, len(labels.categories))
    label = np.asarray(labels.categories)[best]
    umap = lt_place(ref_umap, idx, w) if ref_umap is not None else None
    del ref_rep, query_rep
    check()
    if method == "celltypist":
        label, score = lt_celltypist(ref_log, labels.astype(str), query_log, genes, seed=seed)
        check()
    nr, nq = lt_space(ref_log[:, novelty_genes], query_log[:, novelty_genes], "joint" if method != "integration"
                      else "integration", n_pcs=n_pcs, clip=clip, seed=seed)
    ratio, ref_ratio = lt_novelty(nr, nq, k)
    threshold = float(np.percentile(ref_ratio, percentile))
    return {"label": np.asarray(label, dtype=object).astype(str), "score": np.asarray(score, np.float32),
            "ratio": ratio, "unmapped": ratio > threshold, "threshold": threshold,
            "ref_ratio": ref_ratio, "umap": umap}


def lt_expression(adata, genes=None):
    """Log-normalised values (CSR float32) of ``adata`` and their gene names: counts
    from layers['counts'], .raw or X, normalised to CP10k and logged; else a
    log-normalised .raw or X as it is. ``genes``: only these, in this order."""
    def counts_like(M):
        vals = M.data[:20000] if sp.issparse(M) else np.asarray(M).ravel()[:80000]
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[vals != 0][:20000]
        return vals.size > 0 and bool(np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))

    def lognorm_like(M):
        vals = M.data[:20000] if sp.issparse(M) else np.asarray(M).ravel()[:80000]
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[vals != 0]
        return vals.size > 0 and bool(np.all(vals >= 0) and vals.max() < 50 and np.any(np.mod(vals, 1) != 0))

    found = []
    if "counts" in adata.layers:
        found.append((adata.layers["counts"], adata.var_names))
    if adata.raw is not None:
        found.append((adata.raw.X, adata.raw.var_names))
    found.append((adata.X, adata.var_names))
    pick = next(((M, names, True) for M, names in found if counts_like(M)), None)
    if pick is None:
        pick = next(((M, names, False) for M, names in found[1:] if lognorm_like(M)), None)
    if pick is None:
        raise ValueError("Neither raw counts nor log-normalised values found (layers['counts'], .raw, X).")
    M, names, is_counts = pick
    X = lt_lognorm(M) if is_counts else sp.csr_matrix(M, dtype=np.float32)
    names = pd.Index(np.asarray(names).astype(str))
    if genes is not None:
        pos = names.get_indexer(list(genes))
        if (pos < 0).any():
            raise ValueError(f"{int((pos < 0).sum())} of the recorded genes are missing, e.g. "
                             f"{[g for g, i in zip(genes, pos) if i < 0][:3]}")
        X = X[:, pos]
        names = pd.Index(list(genes))
    return X, names


def _lognorm_like(M) -> bool:
    """Non-negative, not all whole numbers, below 50: log-normalised values (scaled
    values have negatives, counts are integers)."""
    if M is None:
        return False
    vals = M.data[:20000] if sp.issparse(M) else np.asarray(M).ravel()[:80000]
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[vals != 0]
    return vals.size > 0 and bool(np.all(vals >= 0) and vals.max() < 50 and np.any(np.mod(vals, 1) != 0))


def expression_of(adata, fallback_paths: list[str]) -> tuple[sp.csr_matrix, pd.DataFrame, str]:
    """``adata``'s log-normalised values (CSR float32), their var and where they came
    from: raw counts from the dataset or its source files (``integration.recover_counts``),
    log-normalised to CP10k here; else a log-normalised .raw or X as it is, which is
    what ``lt_expression`` (the replay) reads too."""
    from scview.core.integration import IntegrationError, recover_counts

    try:
        rc = recover_counts(adata, fallback_paths)
        return lt_lognorm(rc.X, TARGET_SUM), rc.var, f"counts ({rc.source}), log-normalised here"
    except IntegrationError:
        pass
    if adata.raw is not None and _lognorm_like(adata.raw.X):
        return sp.csr_matrix(adata.raw.X, dtype=np.float32), adata.raw.var.copy(), "log-normalised .raw"
    if _lognorm_like(adata.X):
        return sp.csr_matrix(adata.X, dtype=np.float32), adata.var.copy(), "log-normalised X"
    raise LabelTransferError("No raw counts or log-normalised values found (layers['counts'], .raw, X, "
                             "and the dataset's source files): mapping compares log-normalised values.", 422)


REPLAY_CONSTANTS = ()
REPLAY_CODE = (lt_lognorm, lt_variable_genes, lt_scale_fit, lt_scale, lt_space, lt_neighbours, lt_weights,
               lt_vote, lt_place, lt_celltypist, lt_novelty, lt_map, lt_expression)


# ---------------------------------------------------------------------------
# Inputs: labels, species, genes
# ---------------------------------------------------------------------------


def slug(name: str) -> str:
    """A column-safe prefix from the reference's name ("Kang PBMC" -> "kang_pbmc")."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_").lower()[:30].strip("_")
    return s or "reference"


def label_columns(adata) -> list[dict[str, Any]]:
    """The reference's obs columns usable as labels: categorical or text with 2-200
    values, not one per cell. Each with its label count."""
    out = []
    n = adata.n_obs
    for col in adata.obs.columns:
        s = adata.obs[col]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_object_dtype(s)):
            continue
        k = int(s.nunique(dropna=True))
        if MIN_LABELS <= k <= MAX_LABELS and k <= 0.5 * n:
            out.append({"column": str(col), "n_labels": k})
    return out


def check_labels(adata, column: str) -> pd.Series:
    """The reference's labels as strings (cells without one dropped later), or a
    plain refusal."""
    if column not in adata.obs.columns:
        raise LabelTransferError(f"The reference has no column '{column}'.", 404)
    s = adata.obs[column]
    if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        raise LabelTransferError(f"'{column}' holds numbers, not labels: pick a cell-type column.", 422)
    k = int(s.nunique(dropna=True))
    if k < MIN_LABELS:
        raise LabelTransferError(f"'{column}' has {k} label(s); a reference needs at least {MIN_LABELS}.", 422)
    if k > MAX_LABELS:
        raise LabelTransferError(f"'{column}' has {k:,} values; a label column has at most {MAX_LABELS}.", 422)
    return s.astype(object).where(s.notna(), None)


def check_species(ref_species: str, query_species: str) -> str | None:
    """A refusal when the two are known and differ; None otherwise."""
    if ref_species in ("human", "mouse") and query_species in ("human", "mouse") \
            and ref_species != query_species:
        return (f"The reference looks {ref_species} and this dataset {query_species} (from their "
                "gene names). Cell types do not map across species by gene name: pick a "
                f"{query_species} reference.")
    return None


@dataclass
class GeneMatch:
    shared: list[str]  # shared genes, in the query's names
    ref_names: list[str]  # the same genes in the reference's names
    n_ref: int
    n_query: int
    renamed: str | None = None  # "reference via var 'gene_ids'" when one side was re-indexed
    message: str = ""


def match_genes(ref_var: pd.DataFrame, query_var: pd.DataFrame) -> GeneMatch:
    """Shared genes of the two, by name; when they label genes differently (Ensembl ids
    against symbols) the ingestion merge engine's reconciliation re-indexes one side
    through a var column of the other's kind (``core/ingestion/merge.py``)."""
    from scview.core.ingestion.merge import plan_merge

    ref_names = pd.Index(ref_var.index.astype(str))
    query_names = pd.Index(query_var.index.astype(str))
    skel = {
        "reference": ad.AnnData(obs=pd.DataFrame(index=pd.Index([], dtype=str)),
                                var=ref_var.set_axis(ref_names)),
        "query": ad.AnnData(obs=pd.DataFrame(index=pd.Index([], dtype=str)),
                            var=query_var.set_axis(query_names)),
    }
    plan = plan_merge(skel, counts_like={"reference": True, "query": True})
    rmap = pd.Series(ref_names, index=ref_names)
    qmap = pd.Series(query_names, index=query_names)
    renamed = None
    recon = plan.reconciliation
    if recon is not None and recon.needed and recon.feasible and recon.overlap_after > plan.intersection:
        for reset in recon.resets:
            var = ref_var if reset.sample == "reference" else query_var
            names = ref_names if reset.sample == "reference" else query_names
            new = pd.Series(names, index=var[reset.via_column].astype(str).to_numpy())
            new = new[~new.index.duplicated()]
            if reset.sample == "reference":
                rmap = new
            else:
                qmap = new
            renamed = f"{'the reference' if reset.sample == 'reference' else 'this dataset'} " \
                      f"by its var '{reset.via_column}'"
    common = [g for g in qmap.index if g in set(rmap.index)]
    return GeneMatch(shared=[str(qmap[g]) for g in common], ref_names=[str(rmap[g]) for g in common],
                     n_ref=len(ref_names), n_query=len(query_names), renamed=renamed,
                     message=recon.message if recon is not None else "")


def too_few_genes(m: GeneMatch) -> str | None:
    if len(m.shared) >= MIN_SHARED_GENES:
        return None
    why = (" They label genes differently and could not be matched (no var column with the other's "
           "kind of id)." if m.message and m.renamed is None and "differently" in m.message else
           " Either one is a pre-filtered gene list, or they are different species.")
    return (f"The reference and this dataset share {len(m.shared):,} genes (reference {m.n_ref:,}, this "
            f"dataset {m.n_query:,}); mapping needs at least {MIN_SHARED_GENES}, or the space is built "
            f"on too little to place cells.{why}")


# ---------------------------------------------------------------------------
# A run: prepared inputs -> streamed steps -> a result to save
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Everything a run needs, copied out of both datasets (they can close meanwhile)."""

    ref_log: sp.csr_matrix  # reference cells with a label x shared genes
    query_log: sp.csr_matrix  # query cells x shared genes (same order)
    genes: list[str]  # shared genes, query names
    ref_genes: list[str]
    ref_labels: np.ndarray
    ref_umap: np.ndarray | None
    query_clusters: pd.Series | None
    cluster_column: str | None
    info: dict[str, Any] = field(default_factory=dict)  # reference id/name/path/mtime, sources


def prepare(ref: ad.AnnData, query: ad.AnnData, *, label_column: str, ref_info: dict[str, Any],
            ref_expression: Callable[[ad.AnnData], tuple[Any, pd.DataFrame, str]],
            query_expression: Callable[[ad.AnnData], tuple[Any, pd.DataFrame, str]],
            cluster_column: str | None) -> Prepared:
    """Check the two datasets and copy out what the run needs. ``*_expression`` give a
    dataset's log-normalised matrix, its var and where it came from (counts recovery,
    ``api/v1/label_transfer.py``)."""
    from scview.core.species import detect_species

    labels = check_labels(ref, label_column)
    refusal = check_species(detect_species(ref).species, detect_species(query).species)
    if refusal:
        raise LabelTransferError(refusal, 422)
    rx, rvar, rsource = ref_expression(ref)
    qx, qvar, qsource = query_expression(query)
    match = match_genes(rvar, qvar)
    refusal = too_few_genes(match)
    if refusal:
        raise LabelTransferError(refusal, 422)
    rpos = pd.Index(rvar.index.astype(str)).get_indexer(match.ref_names)
    qpos = pd.Index(qvar.index.astype(str)).get_indexer(match.shared)
    has = labels.notna().to_numpy()
    if int(has.sum()) < MIN_REFERENCE_CELLS:
        raise LabelTransferError(f"Only {int(has.sum())} reference cells have a label in '{label_column}'; "
                                 f"a reference needs at least {MIN_REFERENCE_CELLS}.", 422)
    if query.n_obs < MIN_QUERY_CELLS:
        raise LabelTransferError(f"This dataset has {query.n_obs} cells; mapping needs {MIN_QUERY_CELLS}.", 422)
    ref_log = sp.csr_matrix(rx[np.flatnonzero(has)][:, rpos], dtype=np.float32)
    query_log = sp.csr_matrix(qx[:, qpos], dtype=np.float32)
    umap = None
    if "X_umap" in ref.obsm and np.asarray(ref.obsm["X_umap"]).shape[1] >= 2:
        umap = np.asarray(ref.obsm["X_umap"], dtype=np.float32)[has, :2].copy()
    clusters = None
    if cluster_column and cluster_column in query.obs.columns:
        clusters = query.obs[cluster_column].astype(str).copy()
    return Prepared(ref_log=ref_log, query_log=query_log, genes=list(match.shared), ref_genes=list(match.ref_names),
                    ref_labels=labels[has].astype(str).to_numpy(), ref_umap=umap, query_clusters=clusters,
                    cluster_column=cluster_column if clusters is not None else None,
                    info={**ref_info, "label_column": label_column, "reference_source": rsource,
                          "query_source": qsource, "n_ref_genes": match.n_ref, "n_query_genes": match.n_query,
                          "renamed": match.renamed, "n_reference_cells": int(has.sum()),
                          "n_reference_unlabelled": int((~has).sum())})


def estimate_seconds(n_ref: int, n_query: int, method: str) -> float:
    """Rough wall time of a run at 2 CPUs, measured on Kang split by donor (6.5k + 7.3k
    cells): projection and integration ~16 s, CellTypist ~36 s (its training)."""
    secs = 2.0 + 1.0e-3 * (n_ref + n_query)
    if method == METHOD_CELLTYPIST:
        secs += 3.0e-3 * n_ref
    return round(secs, 1)


def iter_transfer(p: Prepared, *, method: str, k: int, n_hvg: int = N_HVG, n_pcs: int = N_PCS,
                  stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """The run as events: ``progress`` per step, then ``mapped`` with the outcome."""
    t0 = time.monotonic()

    def halt():
        if stop is not None and stop():
            raise LabelTransferCancelled()

    if method not in METHODS:
        raise LabelTransferError(f"Unknown method '{method}': one of {', '.join(METHODS)}.")
    if not MIN_K <= k <= MAX_K:
        raise LabelTransferError(f"k must be between {MIN_K} and {MAX_K}.")
    k_used = int(min(k, p.ref_log.shape[0] - 1))
    yield {"type": "progress", "step": "genes", "message": "Choosing the variable genes"}
    hv = lt_variable_genes(p.ref_log, n_hvg)
    novelty = np.union1d(hv, lt_variable_genes(p.query_log, n_hvg))
    halt()
    yield {"type": "progress", "step": "space", "message": {
        METHOD_PROJECTION: "Fitting the reference's PCA, projecting this dataset, voting",
        METHOD_INTEGRATION: "Joint PCA and Harmony on both datasets, voting",
        METHOD_CELLTYPIST: "Training CellTypist on the reference (about a minute)",
    }[method]}
    try:
        out = lt_map(p.ref_log, p.query_log, p.ref_labels, method, k_used, hv, novelty, n_pcs=n_pcs,
                     clip=CLIP, seed=SEED, ref_umap=p.ref_umap, percentile=UNMAPPED_PERCENTILE,
                     genes=p.genes, stop=stop)
    except InterruptedError:
        raise LabelTransferCancelled()
    halt()
    out["hvg"] = [p.genes[i] for i in hv]
    out["novelty_genes"] = [p.genes[i] for i in novelty]
    out["k"] = k_used
    out["seconds"] = round(time.monotonic() - t0, 1)
    yield {"type": "mapped", "out": out}


# ---------------------------------------------------------------------------
# Results: summary, columns, provenance
# ---------------------------------------------------------------------------


def columns_for(prefix: str) -> dict[str, str]:
    return {"label": f"{prefix}_transfer", "score": f"{prefix}_transfer_score",
            "unmapped": f"{prefix}_unmapped", "embedding": f"X_umap_ref_{prefix}"}


def cluster_table(clusters: pd.Series | None, unmapped: np.ndarray, label: np.ndarray) -> list[dict[str, Any]]:
    """Per query cluster: cells, share "not in reference", its commonest label and that
    label's share, and whether most of its cells are unmapped."""
    if clusters is None:
        return []
    df = pd.DataFrame({"c": clusters.to_numpy(), "u": unmapped, "l": label})
    rows = []
    for c, g in df.groupby("c", sort=False, observed=True):
        top = g["l"].value_counts()
        share = float(g["u"].mean())
        rows.append({"cluster": str(c), "n_cells": int(len(g)), "unmapped_share": round(share, 4),
                     "top_label": str(top.index[0]), "top_label_share": round(float(top.iloc[0] / len(g)), 4),
                     "candidate_absent": share > CLUSTER_UNMAPPED_SHARE})
    rows.sort(key=lambda r: (-r["unmapped_share"], r["cluster"]))
    return rows


def summarize(p: Prepared, out: dict[str, Any], *, method: str, prefix: str) -> dict[str, Any]:
    label, score, unmapped = out["label"], out["score"], out["unmapped"]
    counts = pd.Series(label).value_counts()
    mean_score = pd.Series(score).groupby(label).mean()
    unm = pd.Series(unmapped).groupby(label).mean()
    hist, edges = np.histogram(score, bins=HIST_BINS, range=(0.0, 1.0))
    ref_counts = pd.Series(p.ref_labels).value_counts()
    return {
        "prefix": prefix,
        "columns": columns_for(prefix) if out.get("umap") is not None
        else {k_: v for k_, v in columns_for(prefix).items() if k_ != "embedding"},
        "method": method,
        "method_text": METHOD_TEXT[method],
        "k": out["k"],
        "n_cells": int(len(label)),
        "n_shared_genes": len(p.genes),
        "n_hvg": len(out["hvg"]),
        "n_pcs": N_PCS,
        "threshold": round(float(out["threshold"]), 4),
        "percentile": UNMAPPED_PERCENTILE,
        "n_unmapped": int(unmapped.sum()),
        "unmapped_share": round(float(unmapped.mean()), 4),
        "median_score": round(float(np.median(score)), 4),
        "labels": [{"label": str(lab), "n_cells": int(n), "mean_score": round(float(mean_score[lab]), 4),
                    "unmapped_share": round(float(unm[lab]), 4)} for lab, n in counts.items()],
        "reference_labels": [{"label": str(lab), "n_cells": int(n),
                              "n_query": int(counts.get(lab, 0))} for lab, n in ref_counts.items()],
        "score_hist": {"counts": hist.tolist(), "edges": [round(float(e), 3) for e in edges]},
        "cluster_column": p.cluster_column,
        "clusters": cluster_table(p.query_clusters, unmapped, label),
        "reference": {key: p.info.get(key) for key in ("id", "name", "mtime", "label_column", "path",
                                                       "n_reference_cells", "n_reference_unlabelled")},
        "sources": {"reference": p.info.get("reference_source"), "query": p.info.get("query_source"),
                    "renamed": p.info.get("renamed")},
        "seconds": out.get("seconds"),
        "note": NOTE,
    }


def save(adata: ad.AnnData, p: Prepared, out: dict[str, Any], summary: dict[str, Any],
         resources: dict | None = None) -> dict[str, Any]:
    """Write the label, score and "not in reference" columns (and the query on the
    reference's UMAP), the summary in uns, and the provenance step."""
    import json

    cols = columns_for(summary["prefix"])
    adata.obs[cols["label"]] = pd.Categorical(out["label"])
    adata.obs[cols["score"]] = out["score"].astype(np.float32)
    adata.obs[cols["unmapped"]] = out["unmapped"].astype(bool)
    if out.get("umap") is not None:
        adata.obsm[cols["embedding"]] = out["umap"]
    elif cols["embedding"] in adata.obsm:
        del adata.obsm[cols["embedding"]]  # a re-run on a reference without a UMAP
    stored = read_all(adata)
    stored[summary["prefix"]] = summary
    adata.uns[UNS_KEY] = json.dumps(stored)  # a JSON string: replaced, never mutated in place
    try:
        ref = summary["reference"]
        provenance.record_step(
            adata, step=STEP, tool=f"scview reference mapping ({summary['method']})",
            params={
                "reference": ref.get("name"), "reference_id": ref.get("id"),
                "reference_file": os.path.basename(str(ref.get("path") or "")),
                "reference_mtime": ref.get("mtime"), "label_column": ref.get("label_column"),
                "method": summary["method"], "k": summary["k"], "n_pcs": N_PCS, "n_hvg": summary["n_hvg"],
                "n_shared_genes": summary["n_shared_genes"], "percentile": UNMAPPED_PERCENTILE,
                "threshold": summary["threshold"], "renamed": summary["sources"]["renamed"],
                "outputs": list(summary["columns"].values()),
                "unmapped_share": summary["unmapped_share"], "prefix": summary["prefix"],
                "has_umap": out.get("umap") is not None,
                "genes": list(out["hvg"]), "novelty_genes": list(out["novelty_genes"]),
            },
            resources=resources,
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record reference mapping provenance: %s", e)
    return summary


def read_all(adata) -> dict[str, Any]:
    """Saved mappings by prefix ({} when none or unreadable)."""
    import json

    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    if raw is None:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def saved(adata) -> list[dict[str, Any]]:
    """Saved mappings whose columns are still in the dataset, newest last."""
    out = []
    for prefix, s in read_all(adata).items():
        if isinstance(s, dict) and s.get("columns", {}).get("label") in adata.obs.columns:
            out.append(s)
    return out
