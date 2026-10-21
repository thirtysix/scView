"""Milo: differential abundance between conditions in neighbourhoods of the
cell graph, with no clusters needed (Dann et al. 2022, Nat Biotech 40:245,
PMID 34594043). Follows pertpy's ``Milo`` (the milopy port) step by step:

1. **Neighbourhoods.** A random 10% of the cells (``prop``; 5% above 30k cells,
   Milo's advice) is drawn with Python's ``random`` seeded at 0, as pertpy does,
   and each drawn cell is moved to the neighbour closest to the median position
   of its neighbours in the representation the graph was built from (PCA or
   Harmony-corrected PCA). This "refined sampling" stops index cells piling up
   at the edge of a dense region; duplicates collapse. A neighbourhood is the
   index cell's neighbours in the dataset's own kNN graph (the binarised
   connectivities), so they overlap, and the membership matrix stays sparse.
2. **Counts.** Cells of each sample (one donor in one condition) in each
   neighbourhood.
3. **Test.** A negative binomial GLM per neighbourhood, ``~ [donor +]
   [covariate +] condition``, fitted by DESeq2 (pydeseq2, already used for
   pseudobulk DE; pertpy offers it as Milo's ``pydeseq2`` solver) and a Wald
   test of the condition. Milo's own solver is edgeR's quasi-likelihood F-test
   with TMM, which needs R in the image. On Kang (Harmony graph, 779
   neighbourhoods, ``~ donor + stim``) the results match pertpy's pydeseq2
   solver to 5e-7, and edgeR's log fold changes to r = 0.989: edgeR calls 304
   neighbourhoods at spatial FDR < 0.1, DESeq2 234, 233 of them shared and all
   in the same direction. On donor-level label shuffles both call nothing in 9
   of 10 runs, DESeq2 a little more conservatively. Donors sampled in both
   conditions are paired (a donor term), as in the composition test.
4. **Spatial FDR.** Neighbourhoods overlap, so plain BH over them counts the
   same cells many times where the graph is dense. Milo weights each
   neighbourhood's p-value by 1 / (distance from its index cell to the k-th
   neighbour) in a weighted Benjamini-Hochberg (cydar's spatial FDR): dense,
   heavily overlapping regions weigh less each.
5. **Labels.** Each neighbourhood gets the commonest label of a chosen column
   and that label's share; below 0.7 it is "Mixed" (the miloR vignette's cut).

Caveats. Neighbourhoods are built on the embedding: one corrected for the
condition itself pulls the conditions together and can hide shifts, and one
where the conditions separate entirely (a strong stimulation response, an
uncorrected batch) turns a change of state into apparent abundance changes
everywhere. Both are flagged. Milo wants neighbourhoods of about 5 cells per
sample or more; with a small k and many samples most counts are 0-2 and the
test has little power.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.composition import MIN_CELLS_PER_SAMPLE, MIN_SAMPLES, _labels, _order
from scview.core.confounding import check, for_comparison

logger = logging.getLogger(__name__)

PROP = 0.1  # share of cells sampled as index cells (Milo's default)
PROP_LARGE = 0.05  # Milo: enough above ~30k cells
LARGE_DATASET = 30_000
MIN_PROP, MAX_PROP = 0.01, 0.3
SEED = 0  # pertpy's default
MAX_CELLS = 200_000
MAX_NHOODS = 10_000
MAX_SAMPLES = 500
MAX_LABELS = 100
MAX_EDGES = 30_000  # neighbourhood graph edges sent to the browser (most shared first)
ALPHA = 0.1  # spatial FDR threshold for "significant" (Milo's plots)
MIXED_FRACTION = 0.7  # below: the neighbourhood is labelled "Mixed"
MIXED = "Mixed"
CELLS_PER_SAMPLE = 5  # Milo's rule of thumb for the mean neighbourhood size
ONE_CONDITION_SHARE = 0.95
ONE_CONDITION_WARN = 0.5  # share of neighbourhoods that are one condition before warning
# DESeq2 fits one model per neighbourhood: one process keeps it in this worker
# thread, as pseudobulk does.
N_CPUS = 1
# Seconds per (sampled index cell x sample), measured on Kang: 1,384 and 2,767
# sampled cells (780 and 1,193 neighbourhoods after refinement) x 16 samples ran
# in 2.3 and 2.6 s on one laptop core, nearly all of it the DESeq2 fit.
SECONDS_PER_CELL_OF_TABLE = 0.0001
SECONDS_BASE = 1.0


class MiloError(ValueError):
    """A request Milo cannot serve (no graph, too few samples): a 400."""


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------


def graph(adata) -> tuple[sp.csr_matrix, sp.csr_matrix | None]:
    """(connectivities, distances) of the dataset's neighbour graph; MiloError
    when it has none."""
    nb = adata.uns.get("neighbors")
    nb = nb if isinstance(nb, dict) else {}
    ckey = nb.get("connectivities_key", "connectivities")
    dkey = nb.get("distances_key", "distances")
    if ckey not in adata.obsp:
        raise MiloError(
            "This dataset has no neighbour graph, which Milo's neighbourhoods are built on. "
            "Run the processing pipeline (through the neighbours step) in Data Assessment first.")
    conn = sp.csr_matrix(adata.obsp[ckey])
    dist = sp.csr_matrix(adata.obsp[dkey]) if dkey in adata.obsp else None
    return conn, dist


def representation(adata) -> tuple[str, np.ndarray]:
    """The space the graph was built in (as the resolution sweep finds it)."""
    from scview.core.resolution_sweep import _representation

    rep, X = _representation(adata)
    if rep is None:
        raise MiloError("No PCA (or Harmony) representation found for the neighbour graph.")
    return rep, X


def default_prop(n_cells: int) -> float:
    return PROP if n_cells < LARGE_DATASET else PROP_LARGE


@dataclass
class Nhoods:
    index: np.ndarray  # index cell of each neighbourhood (sorted)
    membership: sp.csc_matrix  # cells x neighbourhoods, 1 = member
    kth_distance: np.ndarray  # per neighbourhood: index cell to its farthest kNN

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.membership.indptr)


def make_nhoods(conn: sp.spmatrix, X: np.ndarray, dist: sp.spmatrix | None,
                prop: float = PROP, seed: int = SEED) -> Nhoods:
    """Milo's refined sampling on graph ``conn`` (pertpy ``make_nhoods``): the
    same cells for the same seed. ``X`` is the graph's representation."""
    adj = sp.csr_matrix(conn, dtype=np.float32, copy=True)
    adj.eliminate_zeros()
    adj.data[:] = 1
    n = adj.shape[0]
    n_ix = int(np.round(n * prop))
    if n_ix < 1:
        raise MiloError("Too few cells to sample neighbourhoods from.")
    rng = random.Random(seed)  # = random.seed(seed); random.sample(...) in pertpy
    sampled = sorted(rng.sample(range(n), k=n_ix))
    X = np.asarray(X, dtype=np.float64)
    refined = np.empty(n_ix, dtype=np.int64)
    for j, v in enumerate(sampled):
        nbrs = adj.indices[adj.indptr[v]:adj.indptr[v + 1]]
        if nbrs.size == 0:  # an isolated cell keeps itself
            refined[j] = v
            continue
        pos = X[nbrs]
        med = np.median(pos, axis=0)
        refined[j] = nbrs[int(np.argmin(((pos - med) ** 2).sum(axis=1)))]
    index = np.unique(refined)
    membership = sp.csc_matrix(adj[:, index], dtype=np.int32)
    if dist is not None:
        d = sp.csr_matrix(dist)[index]
        kth = np.asarray(d.max(axis=1).todense()).ravel().astype(np.float64)
    else:
        kth = np.ones(len(index))
    return Nhoods(index=index, membership=membership, kth_distance=kth)


def spatial_fdr(pvals: np.ndarray, kth_distance: np.ndarray) -> np.ndarray:
    """Weighted Benjamini-Hochberg with weights 1 / k-th neighbour distance
    (Milo's ``graphSpatialFDR``, cydar's spatial FDR). NaN p-values stay NaN
    and do not count."""
    p = np.asarray(pvals, dtype=np.float64)
    with np.errstate(divide="ignore"):
        w = 1.0 / np.asarray(kth_distance, dtype=np.float64)
    w[~np.isfinite(w)] = 0.0
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    o = ok[np.argsort(p[ok], kind="stable")]
    cw = np.cumsum(w[o])
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = w[o].sum() * p[o] / cw
    adj[~np.isfinite(adj)] = 1.0  # leading zero weights: nothing to spend yet
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[o] = np.minimum(adj, 1.0)
    return out


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@dataclass
class MiloResult:
    condition_key: str
    sample_key: str
    label_key: str | None
    conditions: list[str]  # [reference, tested]
    design: str  # "paired" | "unpaired"
    covariates: list[str]
    formula: str
    representation: str
    prop: float
    n_cells: int  # cells in the chosen conditions and samples
    n_samples: dict[str, int]
    index_cells: list[str]
    index: np.ndarray
    sizes: np.ndarray
    kth_distance: np.ndarray
    counts: dict[str, np.ndarray]  # condition -> cells per neighbourhood
    log_fc: np.ndarray
    pval: np.ndarray
    spatial_fdr: np.ndarray
    labels: list[str] | None = None
    label_fraction: np.ndarray | None = None
    edges: np.ndarray | None = None  # (i, j, shared cells)
    n_edges_total: int = 0
    membership: sp.csc_matrix | None = None  # kept server side (cell selection)
    alpha: float = ALPHA
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped_samples: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def significant(self) -> np.ndarray:
        return np.nan_to_num(self.spatial_fdr, nan=1.0) < self.alpha

    def label_summary(self) -> list[dict[str, Any]]:
        """Per label: neighbourhoods, how many go up / down, median logFC."""
        if self.labels is None:
            return []
        df = pd.DataFrame({"label": self.labels, "lfc": self.log_fc, "sig": self.significant})
        rows = []
        for label, g in df.groupby("label", sort=False):
            lfc = g["lfc"].to_numpy()
            fin = lfc[np.isfinite(lfc)]
            rows.append({
                "label": str(label), "n_nhoods": int(len(g)),
                "n_up": int((g["sig"] & (g["lfc"] > 0)).sum()),
                "n_down": int((g["sig"] & (g["lfc"] < 0)).sum()),
                "median_logfc": float(np.median(fin)) if fin.size else None,
            })
        return sorted(rows, key=lambda r: (r["median_logfc"] is None, r["median_logfc"] or 0.0))

    def to_dict(self, layouts: dict[str, np.ndarray] | None = None) -> dict:
        sig = self.significant
        fin = np.isfinite(self.log_fc)
        return {
            "condition_key": self.condition_key, "sample_key": self.sample_key,
            "label_key": self.label_key, "conditions": self.conditions,
            "reference": self.conditions[0], "design": self.design,
            "covariates": self.covariates, "formula": self.formula,
            "representation": self.representation, "prop": self.prop, "seed": SEED,
            "n_cells": self.n_cells, "n_samples": self.n_samples,
            "n_nhoods": int(len(self.index)),
            "n_tested": int(np.isfinite(self.pval).sum()),
            "mean_nhood_size": float(self.sizes.mean()) if len(self.sizes) else 0.0,
            "alpha": self.alpha, "mixed_fraction": MIXED_FRACTION,
            "n_significant": int(sig.sum()),
            "n_up": int((sig & fin & (self.log_fc > 0)).sum()),
            "n_down": int((sig & fin & (self.log_fc < 0)).sum()),
            "nhoods": {
                "index_cell": self.index_cells,
                "cell": self.index.astype(int).tolist(),
                "size": self.sizes.astype(int).tolist(),
                "label": self.labels,
                "label_fraction": _floats(self.label_fraction),
                "log_fc": _floats(self.log_fc),
                "pval": _floats(self.pval),
                "spatial_fdr": _floats(self.spatial_fdr),
                "counts": {k: v.astype(int).tolist() for k, v in self.counts.items()},
            },
            "labels": self.label_summary(),
            "edges": [] if self.edges is None else self.edges.astype(int).tolist(),
            "n_edges_total": self.n_edges_total,
            "layouts": {k: _floats(v, 4) for k, v in (layouts or {}).items()},
            "warnings": self.warnings, "notes": self.notes,
            "dropped_samples": self.dropped_samples,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }


def _floats(a: np.ndarray | None, digits: int = 6) -> list | None:
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float64)
    r = np.round(arr, digits)
    if arr.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in r]
    return [[float(v) if np.isfinite(v) else None for v in row] for row in r]


@dataclass
class Plan:
    """What a run would do: checked before any heavy step."""
    condition_key: str
    sample_key: str
    conditions: list[str]
    design: str
    covariates: list[str]
    formula: str
    n_cells: int
    n_samples: dict[str, int]
    prop: float
    n_nhoods_estimate: int
    estimate_seconds: float
    representation: str
    keep: np.ndarray  # cells counted
    sample_codes: np.ndarray  # per kept cell: its sample
    sample_cond: np.ndarray  # per sample: condition
    sample_donor: np.ndarray  # per sample: donor value
    sample_cov: dict[str, np.ndarray]  # per covariate: per-sample value
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "condition_key": self.condition_key, "sample_key": self.sample_key,
            "conditions": self.conditions, "reference": self.conditions[0],
            "design": self.design, "covariates": self.covariates, "formula": self.formula,
            "n_cells": self.n_cells, "n_samples": self.n_samples, "prop": self.prop,
            "n_nhoods_estimate": self.n_nhoods_estimate,
            "estimate_seconds": round(self.estimate_seconds, 1),
            "representation": self.representation,
            "warnings": self.warnings, "notes": self.notes,
            "dropped_samples": self.dropped_samples,
        }


def estimate_seconds(n_nhoods: int, n_samples: int) -> float:
    return SECONDS_BASE + SECONDS_PER_CELL_OF_TABLE * n_nhoods * max(n_samples, 1)


def plan(adata, condition_key: str, sample_key: str | None, *,
         conditions: list[str] | None = None, covariates: list[str] | None = None,
         paired: bool | None = None, prop: float | None = None,
         batch_key: str | None = None) -> Plan:
    """Check the request against the dataset and work out the design. Cheap:
    no neighbourhoods are built. ``conditions`` is [reference, tested]
    (default: the column's first two values)."""
    obs = adata.obs
    covariates = [c for c in dict.fromkeys(covariates or []) if c]
    for col in (condition_key, sample_key, *covariates):
        if col and col not in obs.columns:
            raise MiloError(f"Column '{col}' not found.")
    if not sample_key:
        raise MiloError(
            "Milo needs a sample or donor column: the samples are its replicates. Pick the "
            "column that names each donor or sample.")
    if sample_key == condition_key:
        raise MiloError("The condition and sample columns must be different.")
    if condition_key in covariates or sample_key in covariates:
        raise MiloError("A covariate cannot be the condition or the sample column.")
    if adata.n_obs > MAX_CELLS:
        raise MiloError(f"Milo runs on at most {MAX_CELLS:,} cells; this dataset has "
                        f"{adata.n_obs:,}.")
    graph(adata)
    rep, _X = representation(adata)
    if prop is None:
        prop = default_prop(adata.n_obs)
    if not MIN_PROP <= prop <= MAX_PROP:
        raise MiloError(f"The sampled share must be between {MIN_PROP} and {MAX_PROP}.")

    cond = _labels(obs, condition_key)
    available = _order(obs[condition_key].where(cond.notna()))
    chosen = list(dict.fromkeys(conditions)) if conditions else available[:2]
    unknown = [c for c in chosen if c not in available]
    if unknown:
        raise MiloError(f"'{unknown[0]}' is not a value of '{condition_key}'.")
    if len(chosen) != 2:
        raise MiloError(f"Milo compares two values of '{condition_key}': pick a reference "
                        "and the condition to test against it.")

    sample = _labels(obs, sample_key)
    keep = (cond.isin(chosen) & sample.notna()).to_numpy()
    for c in covariates:
        keep &= obs[c].notna().to_numpy()
    if not keep.any():
        raise MiloError("No cells have all of the chosen columns set.")
    c_vals = cond.to_numpy()[keep]
    s_vals = sample.to_numpy()[keep]
    ids = pd.Series(c_vals, dtype=object) + "\x1f" + pd.Series(s_vals, dtype=object)
    codes, names = pd.factorize(ids, sort=True)
    if len(names) > MAX_SAMPLES:
        raise MiloError(
            f"{len(names)} samples is too many (at most {MAX_SAMPLES}); '{sample_key}' looks "
            "like a per-cell column rather than a donor or sample.")
    s_cond = np.array([x.split("\x1f", 1)[0] for x in names])
    s_donor = np.array([x.split("\x1f", 1)[1] for x in names])
    n_per = np.bincount(codes, minlength=len(names))

    # Small samples are left out, as in the composition test.
    small = n_per < MIN_CELLS_PER_SAMPLE
    dropped = [f"{s_donor[i]} ({s_cond[i]}, {n_per[i]} cells)" for i in np.flatnonzero(small)]
    if small.any():
        ok_cells = ~small[codes]
        idx = np.flatnonzero(keep)
        keep = keep.copy()
        keep[idx[~ok_cells]] = False
        remap = -np.ones(len(names), dtype=np.int64)
        remap[~small] = np.arange(int((~small).sum()))
        codes = remap[codes[ok_cells]]
        s_cond, s_donor, n_per = s_cond[~small], s_donor[~small], n_per[~small]
    n_samples = {k: int((s_cond == k).sum()) for k in chosen}
    few = [k for k in chosen if n_samples[k] < MIN_SAMPLES]
    if few:
        raise MiloError(
            f"{', '.join(f'{k}: {n_samples[k]}' for k in few)} sample(s) of '{sample_key}' "
            f"(with {MIN_CELLS_PER_SAMPLE}+ cells): Milo needs at least {MIN_SAMPLES} per "
            "condition, because the samples are its replicates. One sample per condition "
            "cannot separate a condition effect from that sample's own make-up.")

    # Covariates must be one value per sample.
    obs_kept = obs.loc[keep]
    sample_cov: dict[str, np.ndarray] = {}
    for c in covariates:
        vals = obs_kept[c].astype(str).to_numpy()
        per = pd.DataFrame({"s": codes, "v": vals}).groupby("s")["v"].nunique()
        if (per > 1).any():
            raise MiloError(
                f"'{c}' varies within a sample, so it cannot be a covariate of the sample-level "
                "test. Pick a column with one value per sample (a batch, a sex, a site).")
        first = pd.DataFrame({"s": codes, "v": vals}).groupby("s")["v"].first()
        sample_cov[c] = first.reindex(range(len(s_cond))).to_numpy()

    donors = sorted(set(s_donor))
    complete = all(set(s_cond[s_donor == d]) == set(chosen) for d in donors)
    if paired and not complete:
        raise MiloError(
            f"Pairing needs every '{sample_key}' value in both conditions; some are missing.")
    design = "paired" if (complete if paired is None else paired) else "unpaired"
    notes: list[str] = []
    if paired is None and not complete and any(
            len(set(s_cond[s_donor == d])) > 1 for d in donors):
        notes.append(f"Some '{sample_key}' values occur in both conditions but not all do, so "
                     "the samples are treated as independent (unpaired).")
    terms = ([sample_key] if design == "paired" else []) + covariates + [condition_key]
    formula = "~ " + " + ".join(terms)
    X = design_matrix(s_cond, chosen, s_donor if design == "paired" else None,
                      [sample_cov[c] for c in covariates])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise MiloError(
            "The covariates are confounded with the condition or the donors (a covariate that "
            "is constant within each donor is already absorbed by pairing), so the condition "
            "effect cannot be estimated. Drop the covariate or turn pairing off.")
    if X.shape[0] - X.shape[1] < 1:
        raise MiloError("Too few samples for the design: no residual degrees of freedom.")

    warnings = design_warnings(adata, condition_key, keep, batch_key)
    n_s = int(len(s_cond))
    if min(n_samples.values()) < 3:
        notes.append("With two samples in a condition the replicate variance is barely "
                     "estimable: treat the results as a screen.")
    n_nh = min(int(np.round(adata.n_obs * prop)), MAX_NHOODS)
    return Plan(
        condition_key=condition_key, sample_key=sample_key, conditions=chosen, design=design,
        covariates=covariates, formula=formula, n_cells=int(keep.sum()), n_samples=n_samples,
        prop=float(prop), n_nhoods_estimate=n_nh,
        estimate_seconds=estimate_seconds(n_nh, n_s), representation=rep, keep=keep,
        sample_codes=codes, sample_cond=s_cond, sample_donor=s_donor, sample_cov=sample_cov,
        warnings=warnings, notes=notes, dropped_samples=dropped,
    )


def design_matrix(s_cond: np.ndarray, chosen: list[str], donors: np.ndarray | None,
                  covariates: list[np.ndarray]) -> np.ndarray:
    """Treatment-coded design over the samples: intercept, donors, covariates,
    then the tested condition (last column)."""
    cols = [np.ones(len(s_cond))]
    for factor in ([donors] if donors is not None else []) + covariates:
        levels = sorted(set(factor))
        cols += [(factor == lv).astype(float) for lv in levels[1:]]
    cols.append((s_cond == chosen[1]).astype(float))
    return np.column_stack(cols)


def design_warnings(adata, condition_key: str, keep: np.ndarray,
                    batch_key: str | None) -> list[str]:
    """Where the design, not the test, limits what a neighbourhood shift means."""
    obs = adata.obs
    out: list[str] = []
    extra = [batch_key] if batch_key and batch_key != condition_key else []
    hit = for_comparison(obs, condition_key, keep, extra)
    if hit is not None:
        out.append(f"{hit.message} A difference in abundance is then also a difference between "
                   "batches (capture, dissociation, sequencing run).")
    if batch_key and batch_key in obs.columns:
        same = batch_key == condition_key
        if same or check(obs, batch_key, condition_key, keep) is not None:
            how = "is the condition" if same else "lines up with the condition"
            out.append(
                f"The neighbour graph was built on an embedding batch-corrected for "
                f"'{batch_key}', which {how}. Correction pulls the conditions into the same "
                "neighbourhoods, which can hide abundance shifts (it also keeps a change of "
                "state from reading as one).")
    return out


def _deseq2(counts: np.ndarray, s_cond: np.ndarray, chosen: list[str],
            donors: np.ndarray | None, covariates: list[np.ndarray],
            ) -> tuple[np.ndarray, np.ndarray]:
    """(log2 fold change, p) per row of ``counts`` (neighbourhoods x samples):
    DESeq2's Wald test of the condition, as pertpy's pydeseq2 solver runs it."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    # Safe level and column names: formulaic parses the design.
    meta = pd.DataFrame(index=[f"s{i}" for i in range(len(s_cond))])
    terms = []
    for name, factor in ([("donor", donors)] if donors is not None else []) + [
            (f"cov{i}", f) for i, f in enumerate(covariates)]:
        levels = {v: f"l{j}" for j, v in enumerate(sorted(set(factor)))}
        meta[name] = pd.Categorical([levels[v] for v in factor])
        terms.append(name)
    meta["condition"] = pd.Categorical(np.where(s_cond == chosen[1], "test", "ref"),
                                       categories=["ref", "test"])
    terms.append("condition")
    df = pd.DataFrame(counts.T.astype(np.int64), index=meta.index,
                      columns=[f"n{i}" for i in range(counts.shape[0])])
    inference = DefaultInference(n_cpus=N_CPUS)
    dds = DeseqDataSet(counts=df, metadata=meta, design="~" + "+".join(terms),
                       refit_cooks=True, inference=inference, quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["condition", "test", "ref"], inference=inference,
                     quiet=True)
    res.summary()
    table = res.results_df
    return (table["log2FoldChange"].to_numpy(dtype=np.float64),
            table["pvalue"].to_numpy(dtype=np.float64))


def _one_condition_share(counts: dict[str, np.ndarray], n_cells: dict[str, int]) -> float:
    """Share of neighbourhoods whose cells come 95%+ from one condition, after
    scaling the conditions to the same size."""
    a, b = (counts[k] / max(n_cells[k], 1) for k in counts)
    tot = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.maximum(a, b) / tot
    ok = tot > 0
    return float((share[ok] >= ONE_CONDITION_SHARE).mean()) if ok.any() else 0.0


def iter_milo(adata, p: Plan, *, label_key: str | None = None,
              stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Run Milo, yielding progress events (``{"step", "message"}``) and, last,
    ``{"result": MiloResult}``. ``adata`` is not changed. ``stop`` is checked
    between steps."""
    stop = stop or (lambda: False)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    if label_key is not None and label_key not in adata.obs.columns:
        raise MiloError(f"Column '{label_key}' not found.")

    yield {"step": "nhoods", "message": "Sampling neighbourhoods on the cell graph"}
    conn, dist = graph(adata)
    rep, X = representation(adata)
    nh = make_nhoods(conn, X, dist, p.prop, SEED)
    if len(nh.index) > MAX_NHOODS:
        raise MiloError(f"{len(nh.index):,} neighbourhoods is too many (at most "
                        f"{MAX_NHOODS:,}); sample a smaller share of the cells.")
    timings["nhoods"] = time.perf_counter() - t0
    if stop():
        return

    yield {"step": "counts", "message": f"Counting cells of {len(p.sample_cond)} samples in "
           f"{len(nh.index):,} neighbourhoods"}
    t = time.perf_counter()
    kept = np.flatnonzero(p.keep)
    onehot = sp.csr_matrix(
        (np.ones(len(kept), dtype=np.int32), (kept, p.sample_codes)),
        shape=(adata.n_obs, len(p.sample_cond)))
    counts = np.asarray((nh.membership.T @ onehot).todense(), dtype=np.int64)
    by_cond = {k: counts[:, p.sample_cond == k].sum(axis=1) for k in p.conditions}
    timings["counts"] = time.perf_counter() - t
    if stop():
        return

    n_s = counts.shape[1]
    yield {"step": "test", "message": f"Fitting a negative binomial model per neighbourhood "
           f"(DESeq2, about {estimate_seconds(len(nh.index), n_s):.0f} s)"}
    t = time.perf_counter()
    tested = counts.sum(axis=1) > 0
    log_fc = np.full(len(nh.index), np.nan)
    pval = np.full(len(nh.index), np.nan)
    donors = p.sample_donor if p.design == "paired" else None
    covs = [p.sample_cov[c] for c in p.covariates]
    if tested.sum() < 2:
        raise MiloError("Fewer than two neighbourhoods hold cells of these samples.")
    lfc_t, p_t = _deseq2(counts[tested], p.sample_cond, p.conditions, donors, covs)
    log_fc[tested], pval[tested] = lfc_t, p_t
    fdr = spatial_fdr(pval, nh.kth_distance)
    timings["test"] = time.perf_counter() - t
    if stop():
        return

    yield {"step": "annotate", "message": "Labelling neighbourhoods and building their graph"}
    t = time.perf_counter()
    labels = frac = None
    if label_key is not None:
        labels, frac = annotate(adata.obs[label_key], nh.membership)
    adjacency = sp.triu(nh.membership.T @ nh.membership, k=1).tocoo()
    n_edges = int(adjacency.nnz)
    order = np.argsort(-adjacency.data, kind="stable")[:MAX_EDGES]
    edges = np.column_stack([adjacency.row[order], adjacency.col[order], adjacency.data[order]])
    timings["annotate"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    n_cells = {k: int(np.isin(p.sample_codes, np.flatnonzero(p.sample_cond == k)).sum())
               for k in p.conditions}
    warnings = list(p.warnings)
    share = _one_condition_share(by_cond, n_cells)
    if share >= ONE_CONDITION_WARN:
        warnings.append(
            f"{share:.0%} of the neighbourhoods hold cells of almost only one condition (95%+, "
            "adjusted for condition size): the conditions separate on this embedding, so "
            "Milo reads a change of state (a stimulation response, an uncorrected batch) as "
            "abundance shifts nearly everywhere. Interpret direction with care, or build the "
            "graph on an embedding where matching cell states mix.")
    notes = list(p.notes)
    n_s = len(p.sample_cond)
    mean_size = float(nh.sizes.mean())
    if mean_size < CELLS_PER_SAMPLE * n_s:
        notes.append(
            f"Neighbourhoods hold {mean_size:.0f} cells on average, for {n_s} samples: Milo "
            f"suggests at least {CELLS_PER_SAMPLE} per sample ({CELLS_PER_SAMPLE * n_s}), so "
            "counts are small and only clear shifts reach significance. A neighbour graph with a "
            "larger k (Data Assessment, neighbours step) gives bigger neighbourhoods.")
    if labels is not None:
        notes.append(f"Neighbourhoods whose commonest '{label_key}' label holds under "
                     f"{MIXED_FRACTION:.0%} of their cells are labelled {MIXED}.")
    untested = int(tested.sum() - np.isfinite(pval).sum())
    if untested:
        notes.append(f"{untested} neighbourhoods were not tested (DESeq2 flagged an outlying "
                     "count, or all their counts are zero).")
    names = adata.obs_names
    yield {"result": MiloResult(
        condition_key=p.condition_key, sample_key=p.sample_key, label_key=label_key,
        conditions=p.conditions, design=p.design, covariates=p.covariates, formula=p.formula,
        representation=rep, prop=p.prop, n_cells=p.n_cells, n_samples=p.n_samples,
        index_cells=[str(names[i]) for i in nh.index], index=nh.index, sizes=nh.sizes,
        kth_distance=nh.kth_distance, counts=by_cond, log_fc=log_fc, pval=pval,
        spatial_fdr=fdr, labels=labels, label_fraction=frac, edges=edges,
        n_edges_total=n_edges, membership=nh.membership, warnings=warnings, notes=notes,
        dropped_samples=p.dropped_samples, timings=timings,
    )}


def run(adata, p: Plan, *, label_key: str | None = None) -> MiloResult:
    """:func:`iter_milo` to the end."""
    for item in iter_milo(adata, p, label_key=label_key):
        if "result" in item:
            return item["result"]
    raise MiloError("Milo stopped before it finished.")


def annotate(values: pd.Series, membership: sp.csc_matrix) -> tuple[list[str], np.ndarray]:
    """Per neighbourhood: its commonest label (``Mixed`` below MIXED_FRACTION)
    and that label's share of its cells (pertpy ``annotate_nhoods``)."""
    lab = _labels(values.to_frame("v"), "v")
    cats = _order(values.where(lab.notna()))
    if len(cats) > MAX_LABELS:
        raise MiloError(f"The label column has {len(cats)} values; pick one with at most "
                        f"{MAX_LABELS} (cell types, clusters).")
    codes = pd.Categorical(lab, categories=cats).codes
    ok = codes >= 0
    onehot = sp.csr_matrix((np.ones(int(ok.sum()), dtype=np.int32),
                            (np.flatnonzero(ok), codes[ok])),
                           shape=(len(values), max(len(cats), 1)))
    tab = np.asarray((membership.T @ onehot).todense(), dtype=np.float64)
    total = tab.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = tab.max(axis=1) / total
    top = tab.argmax(axis=1)
    labels = [MIXED if not (f >= MIXED_FRACTION) else cats[t] for t, f in zip(top, frac)]
    return labels, np.where(total > 0, frac, np.nan)


def layouts(adata, index: np.ndarray, max_dims: int = 2) -> dict[str, np.ndarray]:
    """Index-cell positions on every 2-D-capable embedding (UMAP, t-SNE, ...;
    PCA by its first two components)."""
    out: dict[str, np.ndarray] = {}
    for key in adata.obsm.keys():
        if not str(key).startswith("X_"):
            continue
        arr = np.asarray(adata.obsm[key])
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] == adata.n_obs:
            out[str(key)] = np.asarray(arr[index, :max_dims], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Co-pilot grounding and cell selection: the last run on each loaded dataset
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: MiloResult) -> None:
    """Keep the last result (for the co-pilot and for selecting a
    neighbourhood's cells) as long as the loaded dataset is."""
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        pass


def last_result(adaptor) -> MiloResult | None:
    try:
        with _last_lock:
            r = _last.get(adaptor)
    except TypeError:
        return None
    if r is None:
        return None
    obs = adaptor.adata.obs if hasattr(adaptor, "adata") else None
    if obs is not None:
        if r.membership is not None and r.membership.shape[0] != len(obs):
            return None  # cells filtered since
        if any(k and k not in obs.columns for k in (r.condition_key, r.sample_key, r.label_key)):
            return None
    return r


def nhood_cells(result: MiloResult, nhoods: list[int]) -> np.ndarray:
    """Cells (row indices) in any of the given neighbourhoods."""
    if result.membership is None:
        return np.array([], dtype=np.int64)
    idx = [i for i in nhoods if 0 <= i < result.membership.shape[1]]
    if not idx:
        return np.array([], dtype=np.int64)
    sub = result.membership[:, idx]
    return np.unique(sub.indices).astype(np.int64)


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last Milo run's summary and its labels with the
    most shifting neighbourhoods."""
    r = last_result(adaptor)
    if r is None:
        return [], None
    d = r.to_dict()
    ref, test = r.conditions
    reps = ", ".join(f"{k} {n}" for k, n in r.n_samples.items())
    detail = (f"{test} vs {ref}, samples from '{r.sample_key}' ({reps}; {r.design}), "
              f"{d['n_nhoods']} neighbourhoods (mean {d['mean_nhood_size']:.0f} cells): "
              f"{d['n_significant']} at spatial FDR < {r.alpha} ({d['n_up']} more abundant in "
              f"{test}, {d['n_down']} less)")
    lines = ["\n## Milo neighbourhood differential abundance (last run; samples as replicates)",
             f"- {detail} [result:milo]."]
    rows = [x for x in d["labels"] if x["n_up"] + x["n_down"] > 0]
    rows.sort(key=lambda x: -(x["n_up"] + x["n_down"]))
    for x in rows[:6]:
        lines.append(f"- {x['label']}: {x['n_up']} up, {x['n_down']} down of {x['n_nhoods']} "
                     f"neighbourhoods (median log2FC {x['median_logfc']:+.2f})")
    for w in r.warnings[:2]:
        lines.append(f"- Caution: {w}")
    return lines, detail
