"""Cell-cell communication: LIANA's consensus rank aggregate of ligand-receptor scores.

For every sender group (source), receiver group (target) and ligand-receptor pair
in LIANA's consensus resource (``core/lr_resource.py``), five methods score the
pair from the log-normalised expression of the ligand in the sender and the
receptor in the receiver, as ``li.mt.rank_aggregate`` runs them (liana-py 1.10;
Dimitrov et al. 2022, PMID 35680885):

- CellPhoneDB: the mean of the two means (magnitude) and a permutation p-value,
  the share of label shuffles whose mean is at least as high (specificity).
- Connectome: the product of the means (magnitude) and the mean of the two genes'
  z-scored group means (specificity).
- log2FC: the mean of the two genes' log2 fold changes, group vs the rest, on
  expm1 of the values (specificity).
- NATMI: the product of the means (magnitude) and each mean's share of that gene's
  summed group means, multiplied (specificity).
- SingleCellSignalR: sqrt(L R) / (sqrt(L R) + the matrix mean) (magnitude).

A pair is scored only when both genes are detected in at least ``expr_prop`` of
the sender's and receiver's cells; a complex counts by its least detected subunit
for that filter and by its least expressed subunit for the scores. Each method's
scores are ranked over all scored rows and the ranks combined with robust rank
aggregation (Kolde et al. 2012): ``magnitude_rank`` from the magnitudes,
``specificity_rank`` from the specificities, lower is stronger. The two methods
whose magnitude is the same product (Connectome and NATMI) count once, as LIANA
intends. liana-py 1.10 ranks that shared column twice in place (Connectome's rank,
then NATMI's rank of the negated rank), which turns it upside down: there the
highest product counts as the weakest. ``liana_compat=True`` reproduces that, for
validation only; the specificity rank is unaffected.

LIANA itself is not a dependency: its pandas joins over every group pair copy a
long table once per method and its permutations keep a perms x groups x genes
cube. Here the group statistics are one sparse product, the permutations run in
batches with the same random generator and seed (so the p-values are LIANA's),
and only the scored rows are ever built. Groups with fewer than ``MIN_CELLS``
cells are dropped and cells without a label left out, before anything else.
The agreement with liana-py on Kang and paul15 is in the C040 report.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import beta, rankdata

from scview.core import results_store

EXPR_PROP = 0.1  # LIANA's default: share of cells expressing each gene (subunit)
MIN_CELLS = 5  # LIANA's default: smaller groups are dropped
N_PERMS = 1000  # LIANA's default
SEED = 1337  # LIANA's default
MAX_GROUPS = 60
_PERM_BATCH = 25  # permutations per sparse product

MAGNITUDE = {"lr_means": False, "expr_prod": False, "lrscore": False}  # column: ascending
SPECIFICITY = {"cellphone_pvals": True, "scaled_weight": False, "lr_logfc": False, "spec_weight": False}
METHOD_COLUMNS = {
    "CellPhoneDB": ("lr_means", "cellphone_pvals"),
    "Connectome": ("expr_prod", "scaled_weight"),
    "log2FC": (None, "lr_logfc"),
    "NATMI": ("expr_prod", "spec_weight"),
    "SingleCellSignalR": ("lrscore", None),
}


class CCCError(ValueError):
    """The inputs cannot be scored (too few groups, genes not in the resource)."""


@dataclass
class GroupStats:
    """Per group statistics of the genes any interaction uses."""

    groups: list[str]
    sizes: np.ndarray  # cells per group
    genes: list[str]  # the entity genes, columns of the arrays below
    means: np.ndarray  # groups x genes
    props: np.ndarray  # share of cells with a non-zero value
    zscores: np.ndarray  # (group mean - mean) / sd over all cells (ddof 1), as sc.pp.scale
    logfc: np.ndarray  # log2(mean expm1 in group + 1) - log2(mean expm1 in the rest + 1)
    mat_mean: float  # mean over every cell and every expressed gene (SingleCellSignalR)
    n_cells: int
    n_genes: int  # expressed genes


@dataclass
class Prepared:
    """Cells, labels and the interactions that can be scored on them."""

    X: sp.csr_matrix  # cells x entity genes (float32)
    codes: np.ndarray  # each cell's group
    stats: GroupStats
    interactions: pd.DataFrame  # ligand_complex, receptor_complex, lig_idx, rec_idx (subunit lists)
    n_resource: int
    n_resource_genes_found: int
    dropped_groups: dict[str, int] = field(default_factory=dict)


def _csr32(M) -> sp.csr_matrix:
    M = sp.csr_matrix(M) if not sp.issparse(M) else M.tocsr()
    return M.astype(np.float32, copy=False)


def _group_onehot(codes: np.ndarray, n_groups: int, values: np.ndarray | None = None,
                  cols: np.ndarray | None = None) -> sp.csr_matrix:
    n = len(codes)
    return sp.csr_matrix(
        (np.ones(n) if values is None else values, (codes, np.arange(n) if cols is None else cols)),
        shape=(n_groups, n),
    )


def explode(resource: pd.DataFrame) -> pd.DataFrame:
    """Unique interactions with their subunits listed (``A_B`` is the complex A + B)."""
    res = resource[["ligand", "receptor"]].astype(str).drop_duplicates()
    return pd.DataFrame({
        "ligand_complex": res["ligand"].to_numpy(),
        "receptor_complex": res["receptor"].to_numpy(),
        "ligand_units": [s.split("_") for s in res["ligand"]],
        "receptor_units": [s.split("_") for s in res["receptor"]],
    })


def resource_coverage(resource: pd.DataFrame, genes) -> float:
    """Share of the resource's genes (subunits) found among ``genes``."""
    ex = explode(resource)
    units = {u for us in ex["ligand_units"] for u in us} | {u for us in ex["receptor_units"] for u in us}
    found = set(map(str, genes))
    return len(units & found) / max(1, len(units))


def _sum(M, axis=None):
    out = M.sum(axis=axis, dtype=np.float64)
    return float(out) if axis is None else np.asarray(out).ravel()


def prepare(M, genes, labels: pd.Series, resource: pd.DataFrame, min_cells: int = MIN_CELLS) -> Prepared:
    """Restrict to labelled cells in groups of ``min_cells`` or more, find the
    interactions whose every subunit is an expressed gene, and compute each group's
    statistics for those genes. ``M`` is cells x genes, log-normalised, sparse or
    dense; only the interactions' genes are ever copied out of it."""
    genes = pd.Index(np.asarray(genes, dtype=str))
    if M.shape != (len(labels), len(genes)):
        raise CCCError("Expression matrix and labels do not match.")
    if sp.issparse(M):
        M = M.tocsr()
    lab = labels.astype(str).where(labels.notna())
    keep = lab.notna().to_numpy()
    # Expressed genes are judged on the labelled cells before small groups go, as
    # LIANA drops empty features before it drops small groups.
    if not keep.all():
        M = M[keep]
    lab = lab[keep]
    expressed = _sum(M, axis=0) != 0
    counts = lab.value_counts()
    small = counts[counts < min_cells]
    big = lab.isin(counts.index[counts >= min_cells]).to_numpy()
    present_groups = set(lab[big])
    if isinstance(labels.dtype, pd.CategoricalDtype):
        order = [str(c) for c in labels.cat.categories if str(c) in present_groups]
    else:
        order = sorted(present_groups)
    if len(order) < 2:
        raise CCCError(f"Need at least 2 groups with {min_cells} or more cells; found {len(order)}.")
    if len(order) > MAX_GROUPS:
        raise CCCError(f"{len(order)} groups: communication is scored between at most {MAX_GROUPS}.")
    if not big.all():
        M = M[big]
    codes = pd.Index(order).get_indexer(lab[big])
    n_cells = M.shape[0]
    n_expressed = int(expressed.sum())
    mat_mean = _sum(M) / (n_cells * n_expressed)

    # Unique gene names: the first column of a duplicated name is used.
    first = ~genes.duplicated()
    present = set(genes[first & expressed])
    ex = explode(resource)
    n_resource = len(ex)
    units = {u for us in ex["ligand_units"] for u in us} | {u for us in ex["receptor_units"] for u in us}
    n_found = len(units & set(genes[first]))
    ok = [all(u in present for u in lu) and all(u in present for u in ru)
          for lu, ru in zip(ex["ligand_units"], ex["receptor_units"])]
    ex = ex[np.asarray(ok, dtype=bool)].reset_index(drop=True)
    if ex.empty:
        raise CCCError("None of the resource's ligand-receptor pairs has all its genes expressed here: "
                       "is it the right species?")
    entity = sorted({u for us in ex["ligand_units"] for u in us} | {u for us in ex["receptor_units"] for u in us})
    cols = np.flatnonzero(first)[pd.Index(genes[first]).get_indexer(entity)]
    X = _csr32(M[:, cols])
    X.sort_indices()
    epos = {g: i for i, g in enumerate(entity)}
    ex["lig_idx"] = [[epos[u] for u in us] for us in ex["ligand_units"]]
    ex["rec_idx"] = [[epos[u] for u in us] for us in ex["receptor_units"]]

    stats = group_stats(X, codes, order, mat_mean=mat_mean, n_genes=n_expressed)
    stats.genes = entity
    return Prepared(X=X, codes=codes, stats=stats, interactions=ex, n_resource=n_resource,
                    n_resource_genes_found=n_found,
                    dropped_groups={str(k): int(v) for k, v in small.items()})


def group_stats(X: sp.csr_matrix, codes: np.ndarray, groups: list[str], mat_mean: float,
                n_genes: int) -> GroupStats:
    n_groups = len(groups)
    n_cells = X.shape[0]
    sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)
    onehot = _group_onehot(codes, n_groups)
    X64 = X.astype(np.float64)
    sums = np.asarray((onehot @ X64).todense())
    means = sums / sizes[:, None]
    nz = X64.copy()
    nz.data = (nz.data != 0).astype(np.float64)
    props = np.asarray((onehot @ nz).todense()) / sizes[:, None]
    total = sums.sum(axis=0)
    mu = total / n_cells
    sq = np.asarray(X64.multiply(X64).sum(axis=0)).ravel()
    var = np.maximum(sq - n_cells * mu**2, 0.0) / (n_cells - 1)
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0  # sc.pp.scale leaves a constant gene at 0
    zscores = (means - mu) / sd
    E = X64.copy()
    E.data = np.expm1(E.data)
    esums = np.asarray((onehot @ E).todense())
    etotal = esums.sum(axis=0)
    rest_n = n_cells - sizes
    with np.errstate(invalid="ignore", divide="ignore"):
        rest_mean = (etotal - esums) / rest_n[:, None]
    logfc = np.log2(esums / sizes[:, None] + 1) - np.log2(rest_mean + 1)
    return GroupStats(groups=list(groups), sizes=sizes.astype(int), genes=[], means=means, props=props,
                      zscores=zscores, logfc=logfc, mat_mean=mat_mean, n_cells=n_cells, n_genes=n_genes)


def _side(stats: GroupStats, idx_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per interaction and group: the least detected subunit's share of cells, and
    the least expressed subunit (its gene column): (interactions x groups) each."""
    n_int, n_groups = len(idx_lists), len(stats.groups)
    min_prop = np.empty((n_int, n_groups))
    chosen = np.empty((n_int, n_groups), dtype=np.int64)
    for i, idx in enumerate(idx_lists):
        if len(idx) == 1:
            min_prop[i] = stats.props[:, idx[0]]
            chosen[i] = idx[0]
        else:
            idx_a = np.asarray(idx)
            min_prop[i] = stats.props[:, idx_a].min(axis=1)
            chosen[i] = idx_a[np.argmin(stats.means[:, idx_a], axis=1)]  # first on a tie
    return min_prop, chosen


def candidate_rows(prep: Prepared, expr_prop: float = EXPR_PROP) -> pd.DataFrame:
    """The (source, target, interaction) rows that pass the expression filter, with
    the subunit each side is scored by."""
    st = prep.stats
    lp, lc = _side(st, prep.interactions["lig_idx"].tolist())
    rp, rc = _side(st, prep.interactions["rec_idx"].tolist())
    ok_l = lp >= expr_prop
    ok_r = rp >= expr_prop
    i, s, t = np.nonzero(ok_l[:, :, None] & ok_r[:, None, :])
    return pd.DataFrame({"int": i, "s": s, "t": t, "lig": lc[i, s], "rec": rc[i, t]})


def permutation_counts(X: sp.csr_matrix, codes: np.ndarray, n_groups: int, rows: pd.DataFrame,
                       observed: np.ndarray, n_perms: int = N_PERMS, seed: int = SEED,
                       progress: Callable[[int, int], None] | None = None,
                       stop: Callable[[], bool] | None = None) -> np.ndarray:
    """For each row, how many label shuffles give a ligand-receptor mean at least as
    high as ``observed`` (CellPhoneDB's null). The shuffles are LIANA's: one
    ``default_rng(seed)`` and ``rng.permutation`` per shuffle, in order."""
    rng = np.random.default_rng(seed=seed)
    n_cells = X.shape[0]
    idx = np.arange(n_cells)
    sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)
    inv = (1.0 / sizes)[codes]
    X64 = X.astype(np.float64)
    s, lig = rows["s"].to_numpy(), rows["lig"].to_numpy()
    t, rec = rows["t"].to_numpy(), rows["rec"].to_numpy()
    n_genes = X.shape[1]
    lflat, rflat = s * n_genes + lig, t * n_genes + rec
    counts = np.zeros(len(rows), dtype=np.int64)
    # A batch holds batch x rows shuffled means: keep it near 2e6 values.
    batch = int(max(1, min(_PERM_BATCH, 2_000_000 // max(1, len(rows)))))
    done = 0
    while done < n_perms:
        b = min(batch, n_perms - done)
        perm_cols = np.concatenate([rng.permutation(idx) for _ in range(b)])
        # Shuffle k gives row k of the shuffled matrix cell k's label: group means
        # over X[perm[k]] for every k in the group.
        grp = np.concatenate([codes + j * n_groups for j in range(b)])
        G = sp.csr_matrix((np.tile(inv, b), (grp, perm_cols)), shape=(b * n_groups, n_cells))
        P = np.asarray((G @ X64).todense()).reshape(b, n_groups * n_genes)
        perm_lr = (P[:, lflat] + P[:, rflat]) / 2.0
        counts += (perm_lr >= observed[None, :]).sum(axis=0)
        done += b
        if progress is not None:
            progress(done, n_perms)
        if stop is not None and stop():
            raise InterruptedError("stopped")
    return counts


def robust_rank_aggregate(rmat: np.ndarray) -> np.ndarray:
    """RRA rho (Kolde et al. 2012) per row of a rows x methods rank matrix, as LIANA
    computes it: ranks over each column's maximum, sorted per row, the smallest
    beta-order-statistic p, times the number of methods (capped at 1)."""
    r = rmat / np.max(rmat, axis=0)
    r = np.sort(r, axis=1)
    k = r.shape[1]
    a = np.arange(1, k + 1)[None, :]
    p = beta.cdf(r, a, k - a + 1).min(axis=1)
    return np.clip(p * k, 0, 1)


def _ranks(df: pd.DataFrame, spec: dict[str, bool], twice: tuple[str, ...] = ()) -> np.ndarray:
    cols = []
    for col, ascending in spec.items():
        v = df[col].to_numpy(dtype=np.float64)
        r = rankdata(v if ascending else -v, method="average")
        if col in twice:  # liana-py 1.10: a column shared by two methods is re-ranked
            r = rankdata(-r, method="average")
        cols.append(r)
    return np.column_stack(cols)


def score(prep: Prepared, expr_prop: float = EXPR_PROP, n_perms: int = N_PERMS, seed: int = SEED,
          progress: Callable[[int, int], None] | None = None,
          stop: Callable[[], bool] | None = None, liana_compat: bool = False) -> pd.DataFrame:
    """Every method's score and the two aggregate ranks for each passing row,
    strongest magnitude first (LIANA's order)."""
    st = prep.stats
    rows = candidate_rows(prep, expr_prop)
    if rows.empty:
        raise CCCError(f"No ligand-receptor pair is expressed in {expr_prop:.0%} of both a sender's "
                       "and a receiver's cells.")
    s, t, lig, rec = (rows[c].to_numpy() for c in ("s", "t", "lig", "rec"))
    lm, rm = st.means[s, lig], st.means[t, rec]
    lr_means = (lm + rm) / 2.0
    lr_means[(lm == 0) | (rm == 0)] = 0.0
    sum_l, sum_r = st.means.sum(axis=0)[lig], st.means.sum(axis=0)[rec]
    sqrt_lr = np.sqrt(lm) * np.sqrt(rm)
    ints = prep.interactions
    out = pd.DataFrame({
        "source": np.asarray(st.groups, dtype=object)[s],
        "target": np.asarray(st.groups, dtype=object)[t],
        "ligand_complex": ints["ligand_complex"].to_numpy()[rows["int"]],
        "receptor_complex": ints["receptor_complex"].to_numpy()[rows["int"]],
        "ligand": np.asarray(st.genes, dtype=object)[lig],
        "receptor": np.asarray(st.genes, dtype=object)[rec],
        "ligand_means": lm, "receptor_means": rm,
        "ligand_props": st.props[s, lig], "receptor_props": st.props[t, rec],
        "lr_means": lr_means,
        "expr_prod": lm * rm,
        "scaled_weight": (st.zscores[s, lig] + st.zscores[t, rec]) / 2.0,
        "lr_logfc": (st.logfc[s, lig] + st.logfc[t, rec]) / 2.0,
        "spec_weight": (lm / sum_l) * (rm / sum_r),
        "lrscore": sqrt_lr / (sqrt_lr + st.mat_mean),
    })
    if n_perms > 0:
        counts = permutation_counts(prep.X, prep.codes, len(st.groups), rows, lr_means,
                                    n_perms=n_perms, seed=seed, progress=progress, stop=stop)
        out["cellphone_pvals"] = counts / n_perms
        spec = SPECIFICITY
    else:
        out["cellphone_pvals"] = np.nan
        spec = {k: v for k, v in SPECIFICITY.items() if k != "cellphone_pvals"}
    twice = ("expr_prod",) if liana_compat else ()
    out["magnitude_rank"] = robust_rank_aggregate(_ranks(out, MAGNITUDE, twice))
    out["specificity_rank"] = robust_rank_aggregate(_ranks(out, spec))
    return out.sort_values(["magnitude_rank", "specificity_rank"], kind="stable").reset_index(drop=True)


def estimate_seconds(n_cells: int, n_groups: int, n_perms: int = N_PERMS) -> float:
    """A rough run time, mostly the permutations. Measured on 2 cores: Kang 13.8k
    cells x 13 groups 2.4 s, x 60 groups 4.5 s, five copies (69k cells) 15.7 s."""
    return round(0.5 + n_perms / 1000 * (n_cells * 2.0e-4 + n_groups * n_groups * 6.0e-4), 1)


# ---------------------------------------------------------------------------
# A run's result: the group-pair summary, kept results (CSV) and co-pilot facts
# ---------------------------------------------------------------------------

SPECIFIC = 0.05  # specificity_rank at or below this: counted between two groups
KEEP = 4  # results held for CSV download, a few MB each
KIND = "ccc"  # in the results store


@dataclass
class Result:
    """One run: its settings, the full table and the per group-pair summary."""

    id: str
    dataset_id: str
    groupby: str
    subset: tuple[str, str] | None  # (column, value) the cells were restricted to
    resource: str
    species: str
    matrix_source: str
    groups: list[str]
    group_sizes: list[int]
    dropped_groups: dict[str, int]
    expr_prop: float
    n_perms: int
    n_resource: int
    table: pd.DataFrame
    elapsed_seconds: float
    counts: np.ndarray = field(init=False)  # groups x groups: specific interactions
    strength: np.ndarray = field(init=False)  # their summed ligand-receptor mean

    def __post_init__(self) -> None:
        self.counts, self.strength = pair_summary(self.table, self.groups)

    @property
    def n_cells(self) -> int:
        return int(sum(self.group_sizes))

    def roles(self) -> pd.DataFrame:
        """Per group: specific interactions it sends (as source) and receives (as
        target), autocrine ones counted on both sides."""
        return pd.DataFrame({
            "group": self.groups,
            "outgoing": self.counts.sum(axis=1).astype(int),
            "incoming": self.counts.sum(axis=0).astype(int),
            "outgoing_strength": self.strength.sum(axis=1),
            "incoming_strength": self.strength.sum(axis=0),
        })

    def where(self) -> str:
        cells = f"{self.subset[0]} = {self.subset[1]}" if self.subset else "all cells"
        return f"'{self.groupby}' groups, {cells}"


def pair_summary(table: pd.DataFrame, groups: list[str], threshold: float = SPECIFIC
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(count, summed lr_means) of interactions with specificity_rank <= threshold,
    for each source x target."""
    n = len(groups)
    sig = table[table["specificity_rank"] <= threshold]
    idx = pd.Index(groups)
    s, t = idx.get_indexer(sig["source"]), idx.get_indexer(sig["target"])
    counts = np.zeros((n, n))
    strength = np.zeros((n, n))
    np.add.at(counts, (s, t), 1)
    np.add.at(strength, (s, t), sig["lr_means"].to_numpy(dtype=np.float64))
    return counts, strength


_results: dict[str, Result] = {}
_order: list[str] = []
_results_lock = threading.Lock()
_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def keep(result: Result) -> None:
    with _results_lock:
        _results[result.id] = result
        _order.append(result.id)
        while len(_order) > KEEP:
            _results.pop(_order.pop(0), None)


def recall(dataset_id: str, result_id: str) -> Result | None:
    with _results_lock:
        r = _results.get(result_id)
    if r is None:
        r = results_store.recall(dataset_id, result_id, KIND)
    return r if r is not None and r.dataset_id == dataset_id else None


def remember(adaptor, result: Result) -> None:
    """Keep the last result for the co-pilot, as long as the loaded dataset is."""
    try:
        with _results_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    scope = f" within {result.subset[0]} = {result.subset[1]}" if result.subset else ""
    results_store.keep(adaptor, KIND, result, result_id=result.id,
                       key=f"{result.groupby}|{'|'.join(result.subset or ())}",
                       label=f"Ligand-receptor between '{result.groupby}' groups{scope}")


def _last_result(adaptor, obs: pd.DataFrame) -> Result | None:
    r = results_store.remembered(_last, _results_lock, adaptor, KIND)
    if r is None or r.groupby not in obs.columns or (r.subset and r.subset[0] not in obs.columns):
        return None
    if not set(r.groups) <= set(obs[r.groupby].dropna().astype(str)):
        return None  # groups renamed or re-clustered since
    return r


def _pair(row) -> str:
    return f"{row.source} -> {row.target}: {row.ligand_complex} -> {row.receptor_complex}"


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's top interactions and strongest sender and
    receiver, with a citable detail line."""
    r = _last_result(adaptor, obs)
    if r is None:
        return [], None
    tab = r.table
    n_spec = int((tab["specificity_rank"] <= SPECIFIC).sum())
    detail = (f"LIANA consensus rank aggregate ({r.resource}) between {len(r.groups)} {r.where()}: "
              f"{len(tab):,} source-target-pair rows scored, {n_spec:,} with specificity rank <= {SPECIFIC}")
    lines = ["\n## Cell-cell communication (last run; ligand-receptor, LIANA consensus)",
             f"- {detail} [result:communication]."]
    # The strongest row of each ligand-receptor pair: broadly expressed pairs
    # (TIMP1 -> CD63) would otherwise fill every line with one pair.
    top = tab[tab["specificity_rank"] <= SPECIFIC]
    if top.empty:
        top = tab
    top = top.drop_duplicates(["ligand_complex", "receptor_complex"]).head(8)
    for row in top.itertuples():
        lines.append(f"- {_pair(row)} (magnitude rank {row.magnitude_rank:.2g}, "
                     f"specificity rank {row.specificity_rank:.2g})")
    roles = r.roles()
    if n_spec:
        snd = roles.sort_values("outgoing", ascending=False).iloc[0]
        rcv = roles.sort_values("incoming", ascending=False).iloc[0]
        lines.append(f"- Strongest sender: {snd.group} ({snd.outgoing} specific interactions out); "
                     f"strongest receiver: {rcv.group} ({rcv.incoming} in).")
    lines.append("- These are expression-based predictions of possible signalling, not measured "
                 "interactions; a condition comparison needs a run per condition.")
    return lines, detail
