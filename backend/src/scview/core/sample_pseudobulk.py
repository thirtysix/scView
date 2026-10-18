"""Sample-level QC on pseudobulk profiles: what separates the samples, and
which samples stand apart.

Cell-level QC can look clean while one sample is off as a whole (a failed
library, a swapped or degraded sample, a donor unlike the rest), and the
largest difference between samples may be a batch rather than the condition.
Both show when each sample is one profile, as bulk RNA-seq QC looks at them:

1. **Profiles.** Each sample's raw counts are summed (one profile per sample, or
   per sample x condition when the samples carry several conditions, as Kang's
   donors carry CTRL and STIM), optionally within one cell type. A profile
   from fewer than ``MIN_CELLS`` cells is dropped and listed, as in pseudobulk
   DE and the per-donor plots.
2. **PCA.** log2(CPM + 1) of the genes with at least 10 counts in 2+ profiles,
   the ``N_TOP`` most variable of them (as DESeq2's plotPCA), centred, not
   scaled.
3. **What drives each PC.** Every sample-level covariate (an obs column that is
   constant within each profile, the profile's cells and library size, and the
   sex inferred from expression) against each of the first PCs: one-way ANOVA
   for a categorical one (R^2 = eta squared; Kruskal-Wallis p alongside) and
   Spearman for a numeric one (R^2 = rho^2). A PC's driver is its covariate
   with the highest R^2 at p < 0.05.
4. **Outliers.** A distance on the first ``OUTLIER_PCS`` PCs: each score minus
   the median of its condition's profiles (so the condition difference itself is
   not "outlying"), then in SDs of the *other* profiles (leave one out, so an
   outlier does not widen the spread it is judged by); a profile whose squared
   distance exceeds the chi-square 99.9 % quantile is flagged. A profile of few
   cells is noisier, so its reason says so. Chosen over a median/MAD distance
   on simulations of 11 profiles: as sensitive to a planted outlier (40 of 40)
   with fewer false flags (2-6 of 40 runs against 14-18), since the MAD of so few
   profiles is itself noisy.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from scview.core import cluster_qc, qc_groups
from scview.core.confounding import condition_columns
from scview.core.pseudobulk import MIN_GENE_COUNT, PseudobulkError, raw_counts

MIN_CELLS = 10  # per profile, as core/donor_expression.py and pseudobulk DE
MIN_PROFILES = 4  # fewer and a PCA of the profiles says nothing
MIN_FOR_OUTLIERS = 5
MAX_PROFILES = 300  # as pseudobulk DE
N_TOP = 500  # most variable genes, as DESeq2 plotPCA's ntop
N_PCS = 5  # PCs returned and tested against the covariates
OUTLIER_PCS = 3
OUTLIER_QUANTILE = 0.999
P_DRIVER = 0.05
MIN_RESIDUAL_DF = 2  # profiles minus levels: fewer and a categorical covariate is not tested
MAX_LEVELS = 50  # a categorical covariate with more values is an id, not a covariate
SHARE_CONSTANT = 0.9  # a profile's cells must agree this much on a column's value
N_LOADINGS = 6  # top genes each way for PC1 / PC2

_CELLTYPE_RE = re.compile(r"cell_?type|celltype|annotation|^labels?$|^cluster$", re.I)
_SEP = " | "


class SampleQcError(ValueError):
    """A request this cannot serve (unknown column or cell type): a 4xx."""


def default_celltype_column(obs: pd.DataFrame) -> str | None:
    """A cell-type column for the per-cell-type option, else the clustering."""
    for col in obs.columns:
        name = str(col)
        if (not name.startswith("scview_") and _CELLTYPE_RE.search(name)
                and 2 <= obs[col].nunique(dropna=True) <= 100):
            return name
    return None


def _crossed(obs: pd.DataFrame, sample: str, cond: str) -> bool:
    """True when some sample has MIN_CELLS+ cells in two levels of ``cond``."""
    tab = pd.crosstab(obs[sample].astype(str), obs[cond].astype(str))
    return bool(((tab >= MIN_CELLS).sum(axis=1) >= 2).any())


def auto_split(obs: pd.DataFrame, sample: str) -> str | None:
    """The condition column the samples cross (each sample holds several of its
    levels), so profiles are sample x condition; None when every sample holds
    one level (the condition is then a sample-level covariate)."""
    for cond in condition_columns(obs):
        if cond != sample and _crossed(obs, sample, cond):
            return cond
    return None


def _genes_per_cell(M) -> np.ndarray:
    if sp.issparse(M):
        return np.asarray((M > 0).sum(axis=1), dtype=np.float64).ravel()
    return (np.asarray(M) > 0).sum(axis=1).astype(np.float64)


def _sum_rows(M, codes: np.ndarray, k: int) -> np.ndarray:
    S = sp.csr_matrix((np.ones(codes.size), (codes, np.arange(codes.size))), shape=(k, codes.size))
    out = S @ (M if sp.issparse(M) else np.asarray(M))
    return np.asarray(out.toarray() if sp.issparse(out) else out, dtype=np.float64)


def _unit_covariates(obs: pd.DataFrame, rows: np.ndarray, codes: np.ndarray, k: int,
                     skip: set[str]) -> dict[str, dict]:
    """Obs columns constant within every profile: name -> {kind, values}."""
    out: dict[str, dict] = {}
    for col in obs.columns:
        name = str(col)
        if name in skip or name.startswith("_"):
            continue
        s = obs[col].iloc[rows]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if numeric and s.nunique(dropna=True) > MAX_LEVELS:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            frame = pd.DataFrame({"c": codes, "v": v}).groupby("c")["v"]
            mean, spread = frame.mean().reindex(range(k)), frame.std(ddof=0).reindex(range(k))
            overall = float(np.nanstd(v))
            if not (overall > 0) or not np.isfinite(mean).all():
                continue
            if float(np.nanmax(spread.to_numpy())) <= 0.01 * overall:  # e.g. age per donor
                out[name] = {"kind": "numeric", "values": mean.to_numpy().tolist()}
            continue
        if s.nunique(dropna=True) > MAX_LEVELS or s.nunique(dropna=True) < 2:
            continue
        lab = s.astype(str).where(s.notna()).to_numpy()
        frame = pd.DataFrame({"c": codes, "v": lab}).dropna()
        if frame.empty:
            continue
        top = frame.groupby("c")["v"].agg(lambda x: x.value_counts().index[0])
        share = frame.groupby("c")["v"].agg(lambda x: x.value_counts().iloc[0] / len(x))
        top, share = top.reindex(range(k)), share.reindex(range(k))
        if share.isna().any() or float(share.min()) < SHARE_CONSTANT:
            continue
        values = top.astype(str).tolist()
        n_levels = len(set(values))
        if 2 <= n_levels < k:
            if numeric and n_levels >= 5:  # e.g. age: ranked, not grouped
                out[name] = {"kind": "numeric", "values": [float(x) for x in values]}
            else:
                out[name] = {"kind": "categorical", "values": values}
    return out


def _merge_duplicates(covs: dict[str, dict], prefer: str | None) -> dict[str, dict]:
    """One entry per distinct categorical split of the profiles: Kang's orig_ident
    is stim under another name. The ``prefer`` column (the condition the profiles
    are split by) keeps its name; the others are listed in its ``same_as``."""
    out: dict[str, dict] = {}
    seen: dict[tuple, str] = {}
    for name in sorted(covs, key=lambda n: (n != prefer, bool(covs[n].get("derived")))):
        cov = covs[name]
        if cov["kind"] == "categorical":
            key = tuple(pd.factorize(pd.Series(cov["values"]))[0])
            if key in seen:
                out[seen[key]].setdefault("same_as", []).append(name)
                continue
            seen[key] = name
        out[name] = dict(cov)
    return {n: out[n] for n in covs if n in out}  # back in the columns' own order


def _association(scores: np.ndarray, cov: dict) -> list[dict]:
    """R^2 and p of one covariate against each PC's scores."""
    out = []
    values = cov["values"]
    for j in range(scores.shape[1]):
        y = scores[:, j]
        if cov["kind"] == "numeric":
            x = np.asarray(values, dtype=np.float64)
            ok = np.isfinite(x)
            if ok.sum() < 3 or np.ptp(x[ok]) == 0:
                out.append({"pc": j + 1, "r2": None, "p": None, "p_rank": None})
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            out.append({"pc": j + 1, "r2": round(float(rho) ** 2, 4), "p": float(p),
                        "rho": round(float(rho), 4), "p_rank": None})
            continue
        lab = np.asarray(values, dtype=object)
        groups = [y[lab == lv] for lv in pd.unique(lab)]
        sst = float(((y - y.mean()) ** 2).sum())
        ssb = float(sum(g.size * (g.mean() - y.mean()) ** 2 for g in groups))
        r2 = ssb / sst if sst > 0 else 0.0
        if len(groups) < 2 or len(y) - len(groups) < MIN_RESIDUAL_DF:
            # As many levels as profiles, nearly: any PC "fits" it perfectly.
            out.append({"pc": j + 1, "r2": None, "p": None, "p_rank": None})
            continue
        p = float(stats.f_oneway(*groups).pvalue)
        try:
            p_rank = float(stats.kruskal(*groups).pvalue)
        except ValueError:  # all values identical
            p_rank = None
        out.append({"pc": j + 1, "r2": round(r2, 4), "p": p if np.isfinite(p) else None,
                    "p_rank": p_rank})
    return out


def _outliers(scores: np.ndarray, split: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str]:
    """(squared distance, PC index with the largest |z|) per profile, and the rule."""
    m = min(OUTLIER_PCS, scores.shape[1])
    S = scores[:, :m].copy()
    centred_by = "the median of all profiles"
    if split is not None:
        levels, inv = np.unique(split, return_inverse=True)
        if all((inv == i).sum() >= 3 for i in range(levels.size)):
            for i in range(levels.size):
                S[inv == i] -= np.median(S[inv == i], axis=0)
            centred_by = "the median of its condition's profiles"
        else:
            S -= np.median(S, axis=0)
    else:
        S -= np.median(S, axis=0)
    # Each profile against the others only (leave one out), so an outlier does not
    # widen the spread it is judged by.
    n = S.shape[0]
    Z = np.zeros_like(S)
    for i in range(n):
        others = np.delete(S, i, axis=0)
        sd = others.std(axis=0, ddof=1)
        Z[i] = (S[i] - others.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    d2 = (Z ** 2).sum(axis=1)
    rule = (f"Distance on PC1-{m}: each score minus {centred_by}, then against the mean and "
            f"SD of the other profiles (leave one out); flagged above the chi-square "
            f"{OUTLIER_QUANTILE:.1%} quantile ({m} df).")
    return d2, np.abs(Z).argmax(axis=1), rule


def pseudobulk_qc(adata, sample_column: str | None = None, split_by: str | None = None, *,
                  celltype_column: str | None = None, celltype: str | None = None,
                  sex_calls: dict[str, str] | None = None) -> dict:
    """PCA of the samples' pseudobulk profiles, PC-covariate association and outliers.

    ``sample_column`` None detects the samples (``qc_groups.sample_grouping``);
    ``split_by`` None picks a condition the samples cross ("" = none);
    ``celltype`` limits the profiles to that value of ``celltype_column``;
    ``sex_calls`` maps sample -> inferred sex (core/sex_check.py), a covariate.
    A dataset without samples or counts is not an error: the result says why."""
    obs = adata.obs
    for col in (sample_column, split_by, celltype_column):
        if col and col not in obs.columns:
            raise SampleQcError(f"Column '{col}' not found.")
    reason = "requested" if sample_column else None
    if not sample_column:
        sample_column, reason = qc_groups.sample_grouping(adata)
    ct_col = celltype_column or default_celltype_column(obs) or cluster_qc.default_grouping(adata)
    out: dict = {
        "sample_column": sample_column, "reason": reason, "split_by": None,
        "celltype_column": ct_col, "celltype": celltype or None, "celltype_levels": [],
        "n_cells": int(adata.n_obs), "n_cells_used": 0, "min_cells": MIN_CELLS,
        "profiles": [], "dropped": [], "n_genes": 0, "n_top": N_TOP,
        "explained_variance": [], "covariates": [], "association": [], "drivers": {},
        "loadings": {}, "outliers": [], "outlier_rule": None, "counts_source": None,
        "notes": [], "message": None,
    }
    if ct_col and ct_col in obs.columns:
        vc = obs[ct_col].astype(str).where(obs[ct_col].notna()).value_counts()
        order = qc_groups.group_codes(obs[ct_col])[0]
        out["celltype_levels"] = [{"name": n, "n_cells": int(vc.get(n, 0))} for n in order][:100]
    if celltype and not (ct_col and celltype in set(obs[ct_col].astype(str))):
        raise SampleQcError(f"'{celltype}' is not a value of '{ct_col}'.")
    if not sample_column:
        out["message"] = ("No sample, donor or batch column was found, so there are no samples to "
                          "compare: pseudobulk QC needs one profile per sample.")
        return out
    if split_by is None:
        split_by = auto_split(obs, sample_column)
    elif split_by == "" or split_by == sample_column:
        split_by = None
    out["split_by"] = split_by

    try:
        M, gene_names, source = raw_counts(adata)
    except PseudobulkError:
        out["message"] = ("Pseudobulk profiles sum raw counts, and this dataset has none (not in "
                          "layers['counts'], .raw or X).")
        return out
    out["counts_source"] = source

    keep = obs[sample_column].notna().to_numpy()
    if split_by:
        keep &= obs[split_by].notna().to_numpy()
    if celltype:
        keep &= (obs[ct_col].astype(str) == celltype).to_numpy()
    rows = np.flatnonzero(keep)
    if rows.size == 0:
        out["message"] = "No cells have the chosen columns set."
        return out
    samp = obs[sample_column].astype(str).to_numpy()[rows]
    spl = obs[split_by].astype(str).to_numpy()[rows] if split_by else None
    key = pd.Series(samp) if spl is None else pd.Series(samp) + _SEP + pd.Series(spl)
    codes, labels = pd.factorize(key, sort=True)
    k_all = labels.size
    if k_all > MAX_PROFILES:
        raise SampleQcError(f"{k_all} profiles is too many (at most {MAX_PROFILES}); "
                            f"'{sample_column}' looks like a per-cell column.")
    n_cells = np.bincount(codes, minlength=k_all)
    Msub = M[rows] if rows.size < adata.n_obs else M
    genes = _genes_per_cell(Msub)
    P = _sum_rows(Msub, codes, k_all)

    ok = n_cells >= MIN_CELLS
    for i in np.flatnonzero(~ok):
        s, _, c = str(labels[i]).partition(_SEP)
        out["dropped"].append({"sample": s, "split": c or None, "n_cells": int(n_cells[i])})
    if out["dropped"]:
        out["notes"].append(f"{len(out['dropped'])} profile(s) from fewer than {MIN_CELLS} cells "
                            "were left out: their sums are mostly noise.")
    idx = np.flatnonzero(ok)
    out["n_cells_used"] = int(n_cells[idx].sum())
    if idx.size < MIN_PROFILES:
        out["message"] = (f"Only {idx.size} profile(s) with at least {MIN_CELLS} cells"
                          f"{' of ' + celltype if celltype else ''}: a PCA needs {MIN_PROFILES} "
                          "or more.")
        return out

    # Remap codes onto the kept profiles; cells of dropped ones get -1.
    remap = np.full(k_all, -1)
    remap[idx] = np.arange(idx.size)
    kcodes = remap[codes]
    P = P[idx]
    k = idx.size
    lib = P.sum(axis=1)
    cpm = P / np.where(lib > 0, lib, 1)[:, None] * 1e6
    logcpm = np.log2(cpm + 1)
    expressed = (P >= MIN_GENE_COUNT).sum(axis=0) >= 2
    gidx = np.flatnonzero(expressed)
    if gidx.size < 2:
        out["message"] = f"Fewer than 2 genes have {MIN_GENE_COUNT}+ counts in 2+ profiles."
        return out
    var = logcpm[:, gidx].var(axis=0)
    top = gidx[np.argsort(-var, kind="stable")[:N_TOP]]
    X = logcpm[:, top] - logcpm[:, top].mean(axis=0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    n_pcs = min(N_PCS, k - 1, top.size)
    signs = np.sign(Vt[np.arange(n_pcs), np.abs(Vt[:n_pcs]).argmax(axis=1)])
    signs[signs == 0] = 1
    scores = U[:, :n_pcs] * s[:n_pcs] * signs
    loadings = Vt[:n_pcs] * signs[:, None]
    total_var = float((s ** 2).sum())
    out["n_genes"] = int(top.size)
    out["explained_variance"] = [round(float(v), 4) for v in (s[:n_pcs] ** 2 / total_var)]
    names = np.asarray(gene_names)[top]
    for j in range(min(2, n_pcs)):
        order = np.argsort(loadings[j])
        out["loadings"][f"PC{j + 1}"] = {
            "positive": [str(g) for g in names[order[::-1][:N_LOADINGS]]],
            "negative": [str(g) for g in names[order[:N_LOADINGS]]],
        }

    # Sample-level covariates.
    kept_rows = rows[kcodes >= 0]
    kc = kcodes[kcodes >= 0]
    covs = _unit_covariates(obs, kept_rows, kc, k, skip={ct_col} if celltype else set())
    if split_by is None:
        covs.pop(sample_column, None)  # one profile per sample: an id
    lab_s = [str(labels[i]).partition(_SEP)[0] for i in idx]
    lab_c = [str(labels[i]).partition(_SEP)[2] or None for i in idx]
    med_genes = pd.Series(genes[kcodes >= 0]).groupby(kc).median().reindex(range(k)).to_numpy()
    derived = {
        "n_cells": {"kind": "numeric", "values": n_cells[idx].astype(float).tolist(),
                    "derived": True},
        "log10 total counts": {"kind": "numeric", "values": np.log10(np.maximum(lib, 1)).tolist(),
                               "derived": True},
        "median genes per cell": {"kind": "numeric", "values": med_genes.tolist(),
                                  "derived": True},
    }
    if sex_calls:
        sexes = [sex_calls.get(sm) if sex_calls.get(sm) in ("female", "male") else "unknown"
                 for sm in lab_s]
        if {"female", "male"} <= set(sexes):
            derived["inferred sex"] = {"kind": "categorical", "values": sexes, "derived": True}
    covs.update(derived)
    covs = _merge_duplicates(covs, prefer=split_by)

    covariates, association = [], []
    for name, cov in covs.items():
        covariates.append({"name": name, "kind": cov["kind"], "derived": bool(cov.get("derived")),
                           "same_as": cov.get("same_as", []),
                           "values": [round(v, 4) if isinstance(v, float) else v
                                      for v in cov["values"]]})
        tests = _association(scores, cov)
        association.append({
            "covariate": name, "kind": cov["kind"], "same_as": cov.get("same_as", []),
            "test": "Spearman" if cov["kind"] == "numeric" else "ANOVA (Kruskal-Wallis p_rank)",
            "pcs": tests,
        })
    out["covariates"], out["association"] = covariates, association
    for j in range(min(2, n_pcs)):
        best = None
        for a in association:
            t = a["pcs"][j]
            if t["r2"] is not None and t["p"] is not None and t["p"] < P_DRIVER:
                if best is None or t["r2"] > best["r2"]:
                    best = {"covariate": a["covariate"], "r2": t["r2"], "p": t["p"]}
        out["drivers"][f"PC{j + 1}"] = best

    out["profiles"] = [
        {"id": str(labels[i]), "sample": lab_s[n], "split": lab_c[n], "n_cells": int(n_cells[i]),
         "total_counts": float(lib[n]), "pcs": [round(float(v), 4) for v in scores[n]]}
        for n, i in enumerate(idx)
    ]
    if k >= MIN_FOR_OUTLIERS:
        d2, main, rule = _outliers(scores, np.asarray(lab_c, dtype=object) if split_by else None)
        cut = float(stats.chi2.ppf(OUTLIER_QUANTILE, min(OUTLIER_PCS, n_pcs)))
        out["outlier_rule"] = rule
        typical = float(np.median(n_cells[idx]))
        for n, p in enumerate(out["profiles"]):
            p["distance"] = round(float(np.sqrt(d2[n])), 3)
            p["outlier"] = bool(d2[n] > cut)
            if p["outlier"]:
                few = p["n_cells"] < 0.5 * typical
                out["outliers"].append({
                    "id": p["id"], "distance": p["distance"], "cutoff": round(float(np.sqrt(cut)), 3),
                    "pc": int(main[n]) + 1, "few_cells": few,
                    "reason": (f"{p['id']} ({p['n_cells']:,} cells) sits {p['distance']:.1f} "
                               f"SDs from the others (cut-off {np.sqrt(cut):.1f}), mostly along "
                               f"PC{int(main[n]) + 1}."
                               + (f" It has few cells (median {typical:,.0f} per profile), so "
                                  "sampling noise alone can set it apart." if few else "")),
                })
        if split_by:
            flagged: dict[str, int] = {}
            for o in out["outliers"]:
                s_name = o["id"].partition(_SEP)[0]
                flagged[s_name] = flagged.get(s_name, 0) + 1
            for s_name, n_flag in flagged.items():
                if n_flag >= 2 and n_flag == lab_s.count(s_name):
                    out["notes"].append(
                        f"Every profile of {s_name} stands apart: the {sample_column} itself "
                        "differs from the others (another donor, tissue or handling), rather "
                        "than one bad library.")
    else:
        out["notes"].append(f"Outliers need {MIN_FOR_OUTLIERS}+ profiles; there are {k}.")
    return out
