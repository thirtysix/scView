"""Learned gates: a small classifier trained on a selection, applied to any dataset (N19).

A positional gate (``frontend/src/lib/gates.ts``) is a list of cell positions: it means
nothing in another dataset. A learned gate is the expression rule behind a selection:
the user lassoes cells, scView trains an L1 logistic regression of "selected" against
the cells around them, and the model (a few dozen genes, ~5 KB of JSON) is kept in the
library, exported, imported and applied to other datasets, where it writes an in/out
column and a score.

The recipe follows CellTypist's custom models (Dominguez Conde et al. 2022, PMID
35549406), which ``label_transfer.lt_celltypist`` already uses: log1p CP10k values, a
scaler fitted on the training cells, values clipped at +-10, a logistic regression.

- Positives: the selection. Negatives: the unselected cells FROM THE SAMPLES THAT HOLD
  >= 5% OF THE SELECTION, a sample being one value of the sample column crossed with
  the condition column when both exist (Kang: donor x stim). A lasso drawn on control
  cells then has control negatives only, so a stimulation programme (interferon genes)
  never separates the two classes and the gate still finds the type in stimulated
  cells. Negatives from all cells (``negatives="all"``) are kept as the control the
  validation compares with.
- Candidates: up to 200 genes up and 100 down by log2 fold change, detected in >= 10%
  of the higher group, without ribosomal, mitochondrial and haemoglobin genes and, by
  default, the cell-cycle (Tirosh 2016) and dissociation-stress genes.
- Model: ``LogisticRegression`` with an L1 penalty (liblinear), class-balanced, over a
  grid of 6 C values, 5-fold cross-validation (folds grouped by sample when there are
  >= 3); the sparsest C within one standard error of the best out-of-fold log-loss; at
  most 50 genes; refitted on every training cell. The in/out threshold maximises the
  out-of-fold F1.
- The "gate score" is the model's probability under class-balanced weights: it ranks
  cells and is not a calibrated frequency.

Applying a gate to another dataset matches its genes by name (Ensembl ids against
symbols through ``label_transfer.match_genes``), through one-to-one HCOP orthologs when
the species differ (``core/orthologs.py``); a gene the target lacks takes the training
mean, so it adds nothing. It refuses when fewer than 70% of the gate's genes, or 80% of
its weight, are found; it warns below 95%, when the target's scores bunch in the
undecided middle far more than the training cells' did (a shift of depth, chemistry or
condition), and when a quarter of the weight sits on interferon-response genes.

The functions in ``REPLAY_CODE`` use numpy, scipy and pandas only: the replay script
copies them in with the gate inline (``core/replay_script.py``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

FORMAT = "scview-learned-gate"
VERSION = 1
TAG = "result:lgate"
KIND = "learned_gate_apply"
TRAIN_KIND = "learned_gate_train"
STEP_TRAIN = "learned_gate_train"
STEP_APPLY = "learned_gate_apply"
UNS_KEY = "scview_learned_gates"  # {column: {gate_id, name, score_column, applied}}
DIR_NAME = "learned_gates"

MIN_LABEL_CELLS = 20
WARN_CELLS = 50
MAX_SELECTED_SHARE = 0.9  # of the candidate cells (selection + negative pool)
SAMPLE_SHARE = 0.05  # a sample holding this share of the selection gives negatives
MAX_POS = 10_000
MAX_NEG = 20_000
SEED = 0
N_UP = 200
N_DOWN = 100
MIN_DETECT = 0.10
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_FOLDS = 5
MAX_GENES = 50
MAX_GATE_GENES = 200  # import cap
MAX_FILE_BYTES = 256 * 1024
CLIP = 10.0
TARGET_SUM = 1e4
N_TOP = 10
MAX_ITER = 1000
# Verdict bars (research-n19 section 2)
GOOD_F1, GOOD_AUROC = 0.9, 0.97
CHECK_F1 = 0.75
CLUSTER_JACCARD = 0.9
# Applying: the share of the gate's genes / weight found
REFUSE_GENES, REFUSE_WEIGHT, WARN_COVERAGE = 0.70, 0.80, 0.95
MID_LOW, MID_HIGH = 0.2, 0.8  # the undecided middle of the score
SHIFT_RATIO, SHIFT_MIN = 2.0, 0.05
ISG_SHARE = 0.25
ISG_SETS = ("HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE")
MAX_DISAGREE_INDICES = 20_000

SCORE_NOTE = ("The gate score is the model's probability that a cell is in the gate, with both classes "
              "weighted equally in training: it ranks cells, it is not the chance or frequency of being in.")
EXCLUDE_RULES = ("ribosomal", "mitochondrial", "haemoglobin")
CC_STRESS_RULES = ("cell cycle (Tirosh 2016)", "dissociation stress")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class LearnedGateError(RuntimeError):
    """A gate that cannot be trained or applied as asked (``status`` is the HTTP code)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class LearnedGateCancelled(RuntimeError):
    """The client went away: training stopped and nothing was kept."""


# ---------------------------------------------------------------------------
# The gate file (pydantic, versioned, capped; never pickle)
# ---------------------------------------------------------------------------


def _finite(v: float) -> float:
    if not math.isfinite(float(v)):
        raise ValueError("must be a finite number")
    return float(v)


class GateGene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    gene: str = Field(max_length=100)
    weight: float
    mean_in: float = 0.0
    mean_out: float = 0.0
    pct_in: float = 0.0
    pct_out: float = 0.0

    _f = field_validator("weight", "mean_in", "mean_out", "pct_in", "pct_out")(_finite)


class GateCV(BaseModel):
    model_config = ConfigDict(extra="ignore")
    auroc: float
    precision: float
    recall: float
    f1: float
    n_pos: int = Field(ge=0)
    n_neg: int = Field(ge=0)
    folds: int = Field(ge=0, le=20)
    grouped: bool = False
    C: float
    mid_share: float = 0.0  # share of training cells with an out-of-fold score in [0.2, 0.8]

    _f = field_validator("auroc", "precision", "recall", "f1", "C", "mid_share")(_finite)


class GateSource(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dataset_id: str = Field(default="", max_length=100)
    name: str = Field(default="", max_length=300)
    cells_fingerprint: str = Field(default="", max_length=100)
    n_cells: int = Field(default=0, ge=0)
    selection_hash: str = Field(default="", max_length=100)
    n_selected: int = Field(default=0, ge=0)
    sample_column: str | None = Field(default=None, max_length=200)
    condition_column: str | None = Field(default=None, max_length=200)
    samples: list[str] = Field(default_factory=list, max_length=1000)
    negatives: str = Field(default="own_samples", max_length=40)


class LearnedGate(BaseModel):
    """One gate file (``DATA_DIR/learned_gates/{id}.json``, ~5 KB)."""

    model_config = ConfigDict(extra="ignore")
    format: Literal["scview-learned-gate"]
    version: Literal[1]
    id: str = Field(max_length=64)
    name: str = Field(min_length=1, max_length=60)
    created: str = Field(default="", max_length=40)
    scview_version: str = Field(default="", max_length=40)
    species: str = Field(default="unknown", max_length=20)
    gene_kind: str = Field(default="symbol", max_length=20)
    genes: list[str] = Field(min_length=1, max_length=MAX_GATE_GENES)
    gene_ids: list[str] | None = Field(default=None, max_length=MAX_GATE_GENES)
    coef: list[float] = Field(max_length=MAX_GATE_GENES)
    intercept: float
    mean: list[float] = Field(max_length=MAX_GATE_GENES)
    std: list[float] = Field(max_length=MAX_GATE_GENES)
    clip: float = CLIP
    target_sum: float = TARGET_SUM
    threshold: float
    cv: GateCV
    top_for: list[GateGene] = Field(default_factory=list, max_length=50)
    top_against: list[GateGene] = Field(default_factory=list, max_length=50)
    excluded_rules: list[str] = Field(default_factory=list, max_length=20)
    isg_weight_share: float | None = None
    source: GateSource = Field(default_factory=GateSource)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("id must be letters, digits, '-' or '_'")
        return v

    @field_validator("coef", "mean", "std")
    @classmethod
    def _finite_list(cls, v: list[float]) -> list[float]:
        return [_finite(x) for x in v]

    @field_validator("genes")
    @classmethod
    def _gene_names(cls, v: list[str]) -> list[str]:
        if any(not isinstance(g, str) or not g or len(g) > 100 for g in v):
            raise ValueError("gene names must be 1-100 characters")
        if len(set(v)) != len(v):
            raise ValueError("gene names must be unique")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "LearnedGate":
        n = len(self.genes)
        if not (len(self.coef) == len(self.mean) == len(self.std) == n):
            raise ValueError("genes, coef, mean and std must have equal lengths")
        if self.gene_ids is not None and len(self.gene_ids) != n:
            raise ValueError("gene_ids must have one id per gene")
        if any(s <= 0 for s in self.std):
            raise ValueError("std must be positive")
        for name in ("intercept", "clip", "target_sum", "threshold"):
            _finite(getattr(self, name))
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 < self.clip <= 100.0 or self.target_sum <= 0:
            raise ValueError("clip or target_sum out of range")
        return self


def parse_gate(raw: bytes | str | dict) -> LearnedGate:
    """A gate from its JSON (text, bytes or a parsed dict), capped and checked;
    ``LearnedGateError`` (422) with the reason otherwise."""
    if isinstance(raw, (bytes, str)):
        if len(raw) > MAX_FILE_BYTES:
            raise LearnedGateError(f"A gate file is at most {MAX_FILE_BYTES // 1024} KB.", 413)
        try:
            raw = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            raise LearnedGateError(f"Not a JSON file: {e}") from e
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise LearnedGateError("Not a scView learned gate file (its 'format' is not 'scview-learned-gate').")
    if raw.get("version") != VERSION:
        raise LearnedGateError(f"Gate file version {raw.get('version')!r}; this scView reads version {VERSION}.")
    try:
        return LearnedGate.model_validate(raw)
    except Exception as e:  # pydantic's ValidationError: shown as one line
        first = str(e).splitlines()
        raise LearnedGateError(f"The gate file is not valid: {' '.join(first[:3])[:300]}") from e


# ---------------------------------------------------------------------------
# Library (DATA_DIR/learned_gates/)
# ---------------------------------------------------------------------------


def gates_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / DIR_NAME


def _path(data_dir: str | Path, gate_id: str) -> Path:
    if not _ID_RE.match(gate_id or ""):
        raise LearnedGateError("Unknown gate.", 404)
    return gates_dir(data_dir) / f"{gate_id}.json"


def save_gate(data_dir: str | Path, gate: LearnedGate) -> Path:
    """Write a gate file (beside it first, then swapped in)."""
    d = gates_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = _path(data_dir, gate.id)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(gate.model_dump_json(indent=1))
    tmp.replace(path)
    return path


def load_gate(data_dir: str | Path, gate_id: str) -> LearnedGate:
    path = _path(data_dir, gate_id)
    if not path.is_file():
        raise LearnedGateError("That learned gate is not in the library.", 404)
    return parse_gate(path.read_bytes())


def delete_gate(data_dir: str | Path, gate_id: str) -> bool:
    path = _path(data_dir, gate_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def list_gates(data_dir: str | Path) -> list[dict[str, Any]]:
    """Every gate in the library, newest first (an unreadable file is listed with its error)."""
    d = gates_dir(data_dir)
    out: list[dict[str, Any]] = []
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            out.append(summary(parse_gate(p.read_bytes())))
        except LearnedGateError as e:
            out.append({"id": p.stem, "name": p.stem, "error": str(e), "created": ""})
    return sorted(out, key=lambda g: g.get("created") or "", reverse=True)


def import_gate(data_dir: str | Path, raw: bytes | str | dict) -> LearnedGate:
    """Check and keep an imported gate; a new id when one with this id but other
    content is in the library already."""
    gate = parse_gate(raw)
    path = _path(data_dir, gate.id)
    if path.is_file():
        try:
            same = load_gate(data_dir, gate.id).model_dump() == gate.model_dump()
        except LearnedGateError:
            same = False
        if not same:
            gate = gate.model_copy(update={"id": new_id()})
    save_gate(data_dir, gate)
    return gate


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def verdict(cv: GateCV | dict) -> dict[str, str]:
    """good / check / warn from the out-of-fold F1 and AUROC, with the sentence shown."""
    f1 = cv.f1 if isinstance(cv, GateCV) else float(cv["f1"])
    auroc = cv.auroc if isinstance(cv, GateCV) else float(cv["auroc"])
    if f1 >= GOOD_F1 and auroc >= GOOD_AUROC:
        return {"level": "good", "text": "Expression separates these cells well: the gate should carry to other data."}
    if f1 >= CHECK_F1:
        return {"level": "check", "text": ("Expression mostly separates these cells; check the cells the model "
                                           "and the outline disagree on.")}
    return {"level": "warn", "text": ("Expression does not separate these cells; the outline may cut a "
                                      "continuum, so the gate will not carry well.")}


def summary(gate: LearnedGate) -> dict[str, Any]:
    """A library row: what the gate is, how well it fit, where it came from."""
    return {"id": gate.id, "name": gate.name, "created": gate.created, "species": gate.species,
            "gene_kind": gate.gene_kind, "n_genes": len(gate.genes), "threshold": gate.threshold,
            "cv": gate.cv.model_dump(), "verdict": verdict(gate.cv),
            "top_for": [g.gene for g in gate.top_for[:5]], "top_against": [g.gene for g in gate.top_against[:5]],
            "source": {"dataset_id": gate.source.dataset_id, "name": gate.source.name,
                       "n_selected": gate.source.n_selected, "sample_column": gate.source.sample_column,
                       "condition_column": gate.source.condition_column, "negatives": gate.source.negatives},
            "scview_version": gate.scview_version}


def column_names(name: str) -> tuple[str, str]:
    """The obs columns a gate writes: ``gate_<slug>`` and ``gate_<slug>_score``."""
    from scview.core.label_transfer import slug

    s = slug(name)
    return f"gate_{s}", f"gate_{s}_score"


# ---------------------------------------------------------------------------
# The computation copied into the replay script (numpy, scipy, pandas only)
# ---------------------------------------------------------------------------


def lg_expression(adata, genes, target_sum=1e4):
    """Dense float32 log1p CP10k values of ``genes`` (columns; NaN for a gene missing),
    from counts in layers['counts'], .raw or X; else log-normalised .raw or X as it is."""
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
    M = sp.csr_matrix(M, dtype=np.float32)
    pos = pd.Index(np.asarray(names).astype(str)).get_indexer([str(g) for g in genes])
    out = np.full((M.shape[0], len(genes)), np.nan, dtype=np.float32)
    have = np.flatnonzero(pos >= 0)
    if have.size:
        sub = M[:, pos[have]].toarray()
        if is_counts:
            totals = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
            scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
            sub = np.log1p(sub * scale[:, None].astype(np.float32))
        out[:, have] = sub
    return out


def lg_apply(X, gate):
    """The gate's score (float32) and in/out (bool) per cell from ``X`` (cells x the
    gate's genes, log-normalised; NaN = the gene is missing, which takes the training
    mean and adds nothing)."""
    mean = np.asarray(gate["mean"], dtype=np.float32)
    std = np.asarray(gate["std"], dtype=np.float32)
    coef = np.asarray(gate["coef"], dtype=np.float64)
    X = np.array(X, dtype=np.float32)
    miss = np.isnan(X)
    if miss.any():
        X[miss] = np.broadcast_to(mean, X.shape)[miss]
    X = (X - mean) / std
    np.clip(X, -gate["clip"], gate["clip"], out=X)
    z = X.astype(np.float64) @ coef + float(gate["intercept"])
    score = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return score.astype(np.float32), score >= float(gate["threshold"])


REPLAY_CODE = (lg_expression, lg_apply)


# ---------------------------------------------------------------------------
# Training inputs
# ---------------------------------------------------------------------------


def _usable(obs: pd.DataFrame, col: str, max_unique: int = 200) -> bool:
    s = obs[col]
    if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool
            or pd.api.types.is_string_dtype(s)):
        return False
    n = s.nunique(dropna=True)
    return 1 < n <= max_unique and n < len(s)


def pick_sample_column(obs: pd.DataFrame) -> str | None:
    """The column that names samples (donor, patient, sample ...): of those matching,
    the one with most values, so "Donor" (8) wins over "orig_ident" (CTRL/STIM)."""
    from scview.core.selection_facts import _SAMPLE_RE

    cands = [str(c) for c in obs.columns if not str(c).startswith("_") and _SAMPLE_RE.search(str(c))
             and _usable(obs, str(c))]
    if not cands:
        return None
    return max(cands, key=lambda c: (obs[c].nunique(dropna=True), -cands.index(c)))


def pick_condition_column(obs: pd.DataFrame, sample_column: str | None) -> str | None:
    """The condition column (stim, treatment, disease ...), when it is not the sample column."""
    from scview.core.selection_facts import _CONDITION_RE

    for c in obs.columns:
        name = str(c)
        if name != sample_column and not name.startswith("_") and _CONDITION_RE.search(name) \
                and _usable(obs, name, 20) and not name.startswith("gate_"):
            return name
    return None


def sample_keys(obs: pd.DataFrame, sample_column: str | None, condition_column: str | None) -> np.ndarray | None:
    """Per cell, its sample (sample x condition when both are given), or None."""
    cols = [c for c in (sample_column, condition_column) if c]
    if not cols:
        return None
    parts = [obs[c].astype(object).where(obs[c].notna(), "NA").astype(str) for c in cols]
    key = parts[0].reset_index(drop=True)
    for p in parts[1:]:
        key = key.str.cat(p.reset_index(drop=True), sep=" | ")
    return key.to_numpy(dtype=object)


def _subsample(rows: np.ndarray, groups: np.ndarray | None, cap: int, rng: np.random.Generator) -> np.ndarray:
    """At most ``cap`` of ``rows``, each group keeping its share (sorted)."""
    if rows.size <= cap:
        return rows
    if groups is None:
        return np.sort(rng.choice(rows, cap, replace=False))
    g = groups[rows]
    keys, counts = np.unique(g, return_counts=True)
    quota = np.floor(cap * counts / rows.size).astype(int)
    rest = cap - int(quota.sum())
    if rest > 0:
        order = np.argsort(-(cap * counts / rows.size - quota), kind="stable")[:rest]
        quota[order] += 1
    out = [rng.choice(rows[g == k], min(q, c), replace=False) for k, q, c in zip(keys, quota, counts) if q > 0]
    return np.sort(np.concatenate(out)) if out else rows[:0]


def excluded_genes(names: pd.Index, species: str, exclude_cc_stress: bool) -> np.ndarray:
    """Mask over ``names`` of the genes never used as candidates."""
    from scview.core.gene_filters import is_ribo_mito
    from scview.core.haemoglobin import hb_mask

    mask = np.fromiter((is_ribo_mito(str(g)) for g in names), dtype=bool, count=len(names))
    mask |= hb_mask(names, species)
    if exclude_cc_stress:
        from scview.core.pipeline import CELL_CYCLE_ALIASES, CELL_CYCLE_G2M_GENES, CELL_CYCLE_S_GENES
        from scview.core.species import match_genes
        from scview.core.stress import STRESS_GENES

        drop = set(match_genes(CELL_CYCLE_S_GENES + CELL_CYCLE_G2M_GENES, names, CELL_CYCLE_ALIASES))
        drop |= set(match_genes(STRESS_GENES, names))
        mask |= np.fromiter((str(g) in drop for g in names), dtype=bool, count=len(names))
    return mask


def _col_stats(M: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n = max(M.shape[0], 1)
    mean = np.asarray(M.sum(axis=0), dtype=np.float64).ravel() / n
    detect = np.asarray((M > 0).sum(axis=0), dtype=np.float64).ravel() / n
    return mean, detect


def candidates(X: sp.csr_matrix, pos: np.ndarray, neg: np.ndarray, excl: np.ndarray,
               n_up: int = N_UP, n_down: int = N_DOWN, min_detect: float = MIN_DETECT) -> dict[str, np.ndarray]:
    """Candidate genes (column positions, up then down) by log2 fold change of the
    means (as scanpy computes it), detected in >= ``min_detect`` of the higher group."""
    mp, dp = _col_stats(X[pos])
    mn, dn = _col_stats(X[neg])
    fc = np.log2((np.expm1(mp) + 1e-9) / (np.expm1(mn) + 1e-9))
    up = np.flatnonzero((fc > 0) & (dp >= min_detect) & ~excl)
    up = up[np.argsort(-fc[up], kind="stable")][:n_up]
    down = np.flatnonzero((fc < 0) & (dn >= min_detect) & ~excl)
    down = down[np.argsort(fc[down], kind="stable")][:n_down]
    cols = np.concatenate([up, down]).astype(np.int64)
    return {"cols": cols, "fc": fc, "mean_in": mp, "mean_out": mn, "pct_in": dp, "pct_out": dn,
            "n_up": np.int64(up.size), "n_down": np.int64(down.size)}


@dataclass
class TrainInput:
    """Everything training needs, copied out of the dataset (it can change meanwhile)."""

    X: np.ndarray  # training cells x candidate genes, log-normalised, float32
    y: np.ndarray  # 1 = selected
    rows: np.ndarray  # the training cells' positions in the dataset
    groups: np.ndarray | None  # their sample keys
    genes: list[str]
    gene_ids: list[str] | None
    stats: dict[str, np.ndarray]  # per candidate: fc, mean_in/out, pct_in/out
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def prepare(adata, cells: Any, *, expression: Callable[[Any], tuple[Any, pd.DataFrame, str]],
            sample_column: str | None = None, condition_column: str | None = None,
            exclude_cc_stress: bool = True, negatives: str = "own_samples") -> TrainInput:
    """Decode the selection, pick the negatives and the candidate genes, and copy out the
    training matrix. ``sample_column`` / ``condition_column``: None picks one, "" none.
    ``expression(adata)`` gives the log-normalised matrix, its var and where it came from
    (``label_transfer.expression_of``)."""
    from scview.core.selection_facts import decode_cells, selection_hash
    from scview.core.species import detect_species

    if negatives not in ("own_samples", "all"):
        raise LearnedGateError(f"Unknown negatives '{negatives}'.", 400)
    n_obs = int(adata.n_obs)
    try:
        sel = decode_cells(cells, n_obs)
    except ValueError as e:
        raise LearnedGateError(str(e), 409 if "now has" in str(e) else 400) from e
    obs = adata.obs
    for col in (sample_column, condition_column):
        if col and col not in obs.columns:
            raise LearnedGateError(f"No column '{col}' in this dataset.", 404)
    scol = pick_sample_column(obs) if sample_column is None else (sample_column or None)
    ccol = pick_condition_column(obs, scol) if condition_column is None else (condition_column or None)
    keys = sample_keys(obs, scol, ccol)
    in_sel = np.zeros(n_obs, dtype=bool)
    in_sel[sel] = True
    n_pos_all = int(in_sel.sum())
    if n_pos_all < MIN_LABEL_CELLS:
        raise LearnedGateError(f"The selection has {n_pos_all} cells; a learned gate needs at least "
                               f"{MIN_LABEL_CELLS} (and better {WARN_CELLS} or more).")
    samples: list[str] = []
    if keys is not None and negatives == "own_samples":
        vals, counts = np.unique(keys[in_sel], return_counts=True)
        keep = vals[counts >= SAMPLE_SHARE * n_pos_all]
        samples = sorted(map(str, keep))
        pool = ~in_sel & np.isin(keys, keep)
    else:
        pool = ~in_sel
    n_pool = int(pool.sum())
    if n_pool < MIN_LABEL_CELLS:
        raise LearnedGateError(
            f"Only {n_pool} unselected cells are in the selection's samples; a learned gate needs at least "
            f"{MIN_LABEL_CELLS} cells outside the selection to learn what it is not.")
    if n_pos_all > MAX_SELECTED_SHARE * (n_pos_all + n_pool):
        raise LearnedGateError(
            f"The selection is {100 * n_pos_all / (n_pos_all + n_pool):.0f}% of the cells in its samples; a "
            f"learned gate needs cells outside it (at most {MAX_SELECTED_SHARE:.0%} selected).")
    warnings = []
    if n_pos_all < WARN_CELLS or n_pool < WARN_CELLS:
        warnings.append(f"Few cells ({n_pos_all} selected, {n_pool} outside): the metrics are uncertain.")
    rng = np.random.default_rng(SEED)
    pos = _subsample(np.flatnonzero(in_sel), keys, MAX_POS, rng)
    neg = _subsample(np.flatnonzero(pool), keys, MAX_NEG, rng)
    X, var, source = expression(adata)
    X = sp.csr_matrix(X)
    if X.shape[0] != n_obs:
        raise LearnedGateError("The expression matrix does not match the cells.", 500)
    names = pd.Index(var.index.astype(str))
    species = detect_species(adata).species
    excl = excluded_genes(names, species, exclude_cc_stress)
    cand = candidates(X, pos, neg, excl)
    cols = cand["cols"]
    if cols.size < 2:
        raise LearnedGateError("No gene differs between the selection and the cells around it (detected in "
                               f">= {MIN_DETECT:.0%} of either): there is nothing to learn the gate from.")
    rows = np.concatenate([pos, neg])
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])[order]
    Xt = X[rows][:, cols].toarray().astype(np.float32)
    gene_ids = None
    for c in ("gene_ids", "gene_id", "ensembl_id"):
        if c in var.columns:
            gene_ids = [str(v) for v in var[c].to_numpy()[cols]]
            break
    gene_kind = "ensembl" if names.size and all(str(g).startswith("ENS") for g in names[cols][:20]) else "symbol"
    stats = {k: np.asarray(cand[k])[cols] for k in ("fc", "mean_in", "mean_out", "pct_in", "pct_out")}
    rules = list(EXCLUDE_RULES) + (list(CC_STRESS_RULES) if exclude_cc_stress else [])
    return TrainInput(
        X=Xt, y=y, rows=rows, groups=None if keys is None else keys[rows], genes=[str(g) for g in names[cols]],
        gene_ids=gene_ids, stats=stats, warnings=warnings,
        info={"n_selected": n_pos_all, "n_pool": n_pool, "n_pos": int(pos.size), "n_neg": int(neg.size),
              "sample_column": scol, "condition_column": ccol, "samples": samples, "negatives": negatives,
              "selection_hash": selection_hash(sel, n_obs), "n_cells": n_obs, "species": species,
              "gene_kind": gene_kind, "source": source, "excluded_rules": rules,
              "n_up": int(cand["n_up"]), "n_down": int(cand["n_down"]), "selection": sel})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _model(C: float):
    """L1 logistic regression, class-balanced, liblinear (one thread). sklearn >= 1.8 takes
    ``l1_ratio=1`` and has dropped ``n_jobs``; older versions ``penalty="l1"``, ``n_jobs=1``."""
    import sklearn
    from sklearn.linear_model import LogisticRegression

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    kw: dict[str, Any] = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1", "n_jobs": 1}
    return LogisticRegression(solver="liblinear", class_weight="balanced", C=C, max_iter=MAX_ITER,
                              random_state=SEED, **kw)


def _fit_capped(Xs: np.ndarray, y: np.ndarray, C: float, max_genes: int = MAX_GENES):
    """A fit with at most ``max_genes`` nonzero coefficients (refitted on the largest
    ones when it had more): (coef over all columns, intercept)."""
    m = _model(C).fit(Xs, y)
    coef = m.coef_.ravel().copy()
    nz = np.flatnonzero(coef)
    if nz.size > max_genes:
        keep = nz[np.argsort(-np.abs(coef[nz]), kind="stable")[:max_genes]]
        m2 = _model(C).fit(Xs[:, keep], y)
        coef = np.zeros_like(coef)
        coef[keep] = m2.coef_.ravel()
        return coef, float(m2.intercept_[0])
    return coef, float(m.intercept_[0])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _folds(y: np.ndarray, groups: np.ndarray | None) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
    """5 stratified folds, grouped by sample when there are >= 3 samples and every fold
    then still holds both classes on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n_min = int(min((y == 1).sum(), (y == 0).sum()))
    k = max(2, min(N_FOLDS, n_min))
    if groups is not None and len(np.unique(groups)) >= 3:
        kg = min(k, len(np.unique(groups)))
        try:
            folds = list(StratifiedGroupKFold(n_splits=kg, shuffle=True, random_state=SEED).split(
                np.zeros(y.size), y, groups))
            if all(len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2 for tr, te in folds):
                return folds, True
        except ValueError:
            pass
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(np.zeros(y.size), y)), False


def _balanced_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pos, neg = y == 1, y == 0
    return float(-0.5 * (np.mean(np.log(p[pos])) + np.mean(np.log(1 - p[neg]))))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The threshold that maximises F1 (0.5 when F1 is flat)."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    f1s = np.array([_prf(y, p >= t)[2] for t in grid])
    if np.ptp(f1s) < 1e-9:
        return 0.5
    best = np.flatnonzero(f1s >= f1s.max() - 1e-12)
    return float(grid[best[len(best) // 2]])  # the middle of a tie


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _encode_cells(idx: np.ndarray, n_obs: int) -> dict[str, Any]:
    """A selection payload as the frontend's session files keep one (``lib/session.ts``):
    the indices, or a base64 bitmask when that is shorter."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    list_chars = int(sum(len(str(int(i))) + 1 for i in idx[:MAX_DISAGREE_INDICES + 1]))
    mask_chars = ((n_obs + 7) // 8 + 2) // 3 * 4
    if idx.size <= MAX_DISAGREE_INDICES and list_chars <= mask_chars:
        return {"n_cells": int(n_obs), "indices": [int(i) for i in idx]}
    bits = np.zeros(n_obs, dtype=np.uint8)
    bits[idx] = 1
    return {"n_cells": int(n_obs), "bitmask": base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode()}


def iter_train(inp: TrainInput, *, name: str, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """Fit the gate, yielding ``progress`` after each C, then ``trained`` with the gate and
    its report. ``stop()`` true between fits raises ``LearnedGateCancelled``."""
    from scview.core.label_transfer import lt_scale, lt_scale_fit

    t0 = time.time()
    mean, std = lt_scale_fit(inp.X)
    Xs = lt_scale(inp.X, mean, std, CLIP)
    y = inp.y.astype(np.int64)
    folds, grouped = _folds(y, inp.groups)
    oof_by_c: list[np.ndarray] = []
    losses: list[list[float]] = []
    for i, C in enumerate(C_GRID):
        oof = np.zeros(y.size, dtype=np.float64)
        fold_loss = []
        for tr, te in folds:
            if stop():
                raise LearnedGateCancelled()
            coef, b = _fit_capped(Xs[tr], y[tr], C)
            oof[te] = _sigmoid(Xs[te].astype(np.float64) @ coef + b)
            fold_loss.append(_balanced_logloss(y[te], oof[te]))
        oof_by_c.append(oof)
        losses.append(fold_loss)
        yield {"type": "progress", "step": "fit", "done": i + 1, "total": len(C_GRID), "C": C,
               "log_loss": round(float(np.mean(fold_loss)), 4), "elapsed_s": round(time.time() - t0, 1)}
    means = np.array([np.mean(v) for v in losses])
    ses = np.array([np.std(v, ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0 for v in losses])
    best = int(np.argmin(means))
    within = np.flatnonzero(means <= means[best] + ses[best])
    pick = int(within.min())  # C_GRID ascending: the smallest C is the sparsest
    C = C_GRID[pick]
    oof = oof_by_c[pick]
    if stop():
        raise LearnedGateCancelled()
    coef_all, intercept = _fit_capped(Xs, y, C)
    nz = np.flatnonzero(coef_all)
    if nz.size == 0:
        raise LearnedGateError("The model kept no gene at any penalty: expression does not separate the "
                               "selection from the cells around it.")
    threshold = best_threshold(y, oof)
    pred = oof >= threshold
    prec, rec, f1 = _prf(y, pred)
    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(y, oof))
    mid = float(np.mean((oof >= MID_LOW) & (oof <= MID_HIGH)))
    cv = GateCV(auroc=auroc, precision=prec, recall=rec, f1=f1, n_pos=int((y == 1).sum()),
                n_neg=int((y == 0).sum()), folds=len(folds), grouped=grouped, C=C, mid_share=mid)

    def gene_row(j: int) -> GateGene:
        return GateGene(gene=inp.genes[j], weight=round(float(coef_all[j]), 5),
                        mean_in=round(float(inp.stats["mean_in"][j]), 4),
                        mean_out=round(float(inp.stats["mean_out"][j]), 4),
                        pct_in=round(float(inp.stats["pct_in"][j]), 4),
                        pct_out=round(float(inp.stats["pct_out"][j]), 4))

    order = nz[np.argsort(-coef_all[nz], kind="stable")]
    top_for = [gene_row(int(j)) for j in order if coef_all[j] > 0][:N_TOP]
    top_against = [gene_row(int(j)) for j in order[::-1] if coef_all[j] < 0][:N_TOP]
    info = inp.info
    missed = inp.rows[(y == 1) & ~pred]
    extra = inp.rows[(y == 0) & pred]
    gate = LearnedGate(
        format=FORMAT, version=VERSION, id=new_id(), name=name.strip()[:60] or "learned gate",
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scview_version=_version(),
        species=info.get("species") or "unknown", gene_kind=info.get("gene_kind") or "symbol",
        genes=[inp.genes[j] for j in nz], gene_ids=None if inp.gene_ids is None else [inp.gene_ids[j] for j in nz],
        coef=[float(coef_all[j]) for j in nz], intercept=intercept, mean=[float(mean[j]) for j in nz],
        std=[float(std[j]) for j in nz], clip=CLIP, target_sum=TARGET_SUM, threshold=threshold, cv=cv,
        top_for=top_for, top_against=top_against, excluded_rules=list(info.get("excluded_rules") or []),
        source=GateSource(n_cells=int(info["n_cells"]), selection_hash=str(info["selection_hash"]),
                          n_selected=int(info["n_selected"]), sample_column=info.get("sample_column"),
                          condition_column=info.get("condition_column"),
                          samples=[str(s) for s in info.get("samples") or []][:1000],
                          negatives=str(info.get("negatives") or "own_samples")))
    yield {"type": "trained", "gate": gate, "report": {
        "verdict": verdict(cv), "n_candidates": len(inp.genes), "n_up": info.get("n_up"),
        "n_down": info.get("n_down"), "n_genes": int(nz.size), "C": C, "c_grid": list(C_GRID),
        "log_loss": [round(float(v), 4) for v in means], "n_selected": info["n_selected"],
        "n_pool": info["n_pool"], "n_pos": info["n_pos"], "n_neg": info["n_neg"],
        "sample_column": info.get("sample_column"), "condition_column": info.get("condition_column"),
        "samples": info.get("samples") or [], "negatives": info.get("negatives"),
        "expression": info.get("source"), "warnings": list(inp.warnings),
        "disagree": {"n_missed": int(missed.size), "n_extra": int(extra.size),
                     "n_training": int(y.size),
                     "cells": _encode_cells(np.concatenate([missed, extra]), int(info["n_cells"]))},
        "seconds": round(time.time() - t0, 1), "score_note": SCORE_NOTE}}


def train(inp: TrainInput, *, name: str) -> tuple[LearnedGate, dict[str, Any]]:
    """``iter_train`` run to its end (tests, scripts)."""
    for item in iter_train(inp, name=name):
        if item["type"] == "trained":
            return item["gate"], item["report"]
    raise LearnedGateError("Training ended without a gate.", 500)  # pragma: no cover


def isg_share(gate: LearnedGate, isg_genes: set[str] | None) -> float | None:
    """The share of the gate's |weight| on interferon-response genes (case ignored)."""
    if not isg_genes:
        return None
    wanted = {g.upper() for g in isg_genes}
    w = np.abs(np.asarray(gate.coef))
    total = float(w.sum())
    if total == 0:
        return None
    return float(sum(wi for g, wi in zip(gate.genes, w) if g.upper() in wanted) / total)


def cluster_overlap(obs: pd.DataFrame, sel: np.ndarray, n_obs: int) -> dict[str, Any] | None:
    """The cluster or cell-type group most like the selection (Jaccard), and a note when it
    is >= 0.9: the selection is just that group."""
    from scview.core.selection_facts import _CLUSTER_RE, _LABEL_RE

    in_sel = np.zeros(n_obs, dtype=bool)
    in_sel[sel] = True
    best: dict[str, Any] | None = None
    for c in obs.columns:
        name = str(c)
        if name.startswith(("_", "gate_")) or not (_CLUSTER_RE.search(name) or _LABEL_RE.search(name)):
            continue
        s = obs[c]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object) or s.nunique() > 300:
            continue
        vals = s.astype(object).where(s.notna(), None).to_numpy()
        for g in pd.unique(vals[in_sel]):
            if g is None:
                continue
            grp = vals == g
            inter = int((grp & in_sel).sum())
            jac = inter / int((grp | in_sel).sum())
            if best is None or jac > best["jaccard"]:
                best = {"column": name, "group": str(g), "jaccard": round(jac, 3)}
    if best is None:
        return None
    best["is_cluster"] = best["jaccard"] >= CLUSTER_JACCARD
    best["note"] = (f"This selection is almost exactly {best['column']} = {best['group']} (Jaccard "
                    f"{best['jaccard']:.2f}). To carry those labels to other data, reference mapping "
                    "(Label from a reference dataset) may serve better; the gate is still kept."
                    if best["is_cluster"] else "")
    return best


def _version() -> str:
    from scview.core import provenance

    return provenance._scview_version()


# ---------------------------------------------------------------------------
# Check and apply
# ---------------------------------------------------------------------------


def _target_var(adata) -> pd.DataFrame:
    raw = getattr(adata, "raw", None)
    if raw is not None and raw.n_vars > adata.n_vars:
        return raw.var
    return adata.var


@dataclass
class GeneMap:
    target: list[str | None]  # per gate gene, the target's gene name, or None
    via: str  # "name", "name (re-indexed ...)", "orthologs human -> mouse"
    species: str
    gate_species: str
    n_orthologs_missing: int = 0


def map_genes(gate: LearnedGate, target_var: pd.DataFrame, target_species: str,
              ortholog_table: pd.DataFrame | None = None) -> GeneMap:
    """Each gate gene's name in the target: through one-to-one orthologs when the species
    differ (a gene with several orthologs, or two gate genes landing on one, is missing),
    then by name (``label_transfer.match_genes``: Ensembl ids against symbols)."""
    from scview.core import orthologs
    from scview.core.label_transfer import match_genes

    names: list[str | None] = list(gate.genes)
    ids = gate.gene_ids
    via = "name"
    n_orth_missing = 0
    cross = gate.species in ("human", "mouse") and target_species in ("human", "mouse") \
        and gate.species != target_species
    if cross:
        if gate.gene_kind == "ensembl":
            raise LearnedGateError("The gate names its genes by Ensembl id, which orthologs are not mapped "
                                   "from; train it on a dataset with gene symbols.")
        try:
            fn = orthologs.human_to_mouse if gate.species == "human" else orthologs.mouse_to_human
            m = fn(gate.genes, table=ortholog_table)
        except orthologs.OrthologsUnavailable as e:
            raise LearnedGateError(str(e), 503) from e
        hits = {g: v[0] for g, v in m.items() if len(v) == 1}
        counts = pd.Series(list(hits.values())).value_counts() if hits else pd.Series(dtype=int)
        names = [hits.get(g) if g in hits and counts[hits[g]] == 1 else None for g in gate.genes]
        n_orth_missing = sum(n is None for n in names)
        ids = None
        via = f"one-to-one orthologs, {gate.species} -> {target_species} (HCOP)"
    present = [(i, n) for i, n in enumerate(names) if n is not None]
    gvar = pd.DataFrame(index=pd.Index([n for _, n in present], dtype=str))
    if ids is not None:
        gvar["gene_ids"] = [ids[i] for i, _ in present]
    target: list[str | None] = [None] * len(gate.genes)
    if present:
        match = match_genes(gvar, target_var)
        by_ref = dict(zip(match.ref_names, match.shared))
        for i, n in present:
            target[i] = by_ref.get(str(n))
        if match.renamed:
            via += f" (re-indexed {match.renamed})"
    return GeneMap(target=target, via=via, species=target_species, gate_species=gate.species,
                   n_orthologs_missing=n_orth_missing)


def coverage(gate: LearnedGate, gm: GeneMap) -> dict[str, Any]:
    w = np.abs(np.asarray(gate.coef))
    found = np.array([t is not None for t in gm.target])
    gene_share = float(found.mean()) if found.size else 0.0
    weight_share = float(w[found].sum() / w.sum()) if w.sum() > 0 else 0.0
    missing = [gate.genes[i] for i in np.argsort(-w) if not found[i]][:10]
    return {"n_genes": int(found.size), "n_found": int(found.sum()), "gene_share": round(gene_share, 4),
            "weight_share": round(weight_share, 4), "missing_top": missing}


def check(gate: LearnedGate, adata, *, dataset_id: str | None = None, ortholog_table: pd.DataFrame | None = None,
          isg_genes: set[str] | None = None, target_var: pd.DataFrame | None = None) -> dict[str, Any]:
    """Whether the gate can be applied to ``adata``: species, gene mapping, how much of the
    gate is found, refusals and warnings. Writes nothing."""
    from scview.core.species import detect_species

    species = detect_species(adata).species
    refusal = None
    try:
        gm = map_genes(gate, target_var if target_var is not None else _target_var(adata), species, ortholog_table)
    except LearnedGateError as e:
        return {"ok": False, "refusal": str(e), "warnings": [], "species": species, "gate_species": gate.species,
                "via": None, "coverage": None}
    cov = coverage(gate, gm)
    warnings: list[str] = []
    if cov["gene_share"] < REFUSE_GENES or cov["weight_share"] < REFUSE_WEIGHT:
        refusal = (f"Only {cov['n_found']} of the gate's {cov['n_genes']} genes ({cov['gene_share']:.0%}, "
                   f"{cov['weight_share']:.0%} of its weight) are in this dataset; a gate needs "
                   f"{REFUSE_GENES:.0%} of its genes and {REFUSE_WEIGHT:.0%} of its weight to carry. "
                   + ("Most missing genes have no one-to-one ortholog." if gm.n_orthologs_missing else
                      "The dataset may be a pre-filtered gene panel, or name genes another way."))
    elif cov["gene_share"] < WARN_COVERAGE or cov["weight_share"] < WARN_COVERAGE:
        warnings.append(f"{cov['n_genes'] - cov['n_found']} of the gate's genes are missing here "
                        f"({cov['weight_share']:.0%} of its weight found); they take the training mean, so the "
                        "scores lean on fewer genes.")
    if gm.via.startswith("one-to-one"):
        warnings.append(f"Genes mapped by {gm.via}: the gate was trained on {gate.species} cells, so expect "
                        "partial success.")
    share = gate.isg_weight_share if gate.isg_weight_share is not None else isg_share(gate, isg_genes)
    if share is not None and share >= ISG_SHARE:
        other = dataset_id != gate.source.dataset_id
        if not other and gate.source.sample_column and gate.source.samples:
            keys = sample_keys(adata.obs, gate.source.sample_column if gate.source.sample_column in adata.obs
                               else None, gate.source.condition_column if gate.source.condition_column
                               in adata.obs else None)
            other = keys is not None and bool(np.any(~np.isin(keys, gate.source.samples)))
        if other:
            warnings.append(f"{share:.0%} of the gate's weight is on interferon-response genes (MSigDB Hallmark): "
                            "on cells in another condition (stimulated, infected, treated) its scores may shift.")
    return {"ok": refusal is None, "refusal": refusal, "warnings": warnings, "species": species,
            "gate_species": gate.species, "via": gm.via, "coverage": cov,
            "mapped": [{"gene": g, "target": t} for g, t in zip(gate.genes, gm.target)][:MAX_GATE_GENES],
            "isg_weight_share": None if share is None else round(share, 3),
            "columns": dict(zip(("column", "score_column"), column_names(gate.name)))}


def gate_dict(gate: LearnedGate) -> dict[str, Any]:
    return json.loads(gate.model_dump_json())


def score_cells(gate: LearnedGate, X: sp.csr_matrix, names: pd.Index, gm: GeneMap) -> tuple[np.ndarray, np.ndarray]:
    """The gate on a log-normalised matrix (cells x ``names``): score and in/out."""
    pos = names.get_indexer([t if t is not None else "\0missing" for t in gm.target])
    out = np.full((X.shape[0], len(gate.genes)), np.nan, dtype=np.float32)
    have = np.flatnonzero(pos >= 0)
    if have.size:
        out[:, have] = sp.csr_matrix(X)[:, pos[have]].toarray()
    return lg_apply(out, gate_dict(gate))


def per_sample(obs: pd.DataFrame, inside: np.ndarray, column: str | None) -> list[dict[str, Any]]:
    if not column or column not in obs.columns:
        return []
    vals = obs[column].astype(object).where(obs[column].notna(), "NA").astype(str).to_numpy()
    df = pd.DataFrame({"s": vals, "in": inside})
    g = df.groupby("s", sort=False)["in"].agg(["size", "sum"])
    rows = [{"sample": str(s), "n_cells": int(r["size"]), "n_in": int(r["sum"]),
             "share": round(float(r["sum"]) / float(r["size"]), 4) if r["size"] else 0.0} for s, r in g.iterrows()]
    return sorted(rows, key=lambda r: _natural(r["sample"]))


def _natural(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def apply(adata, gate: LearnedGate, *, X: sp.csr_matrix, var: pd.DataFrame, source: str,
          dataset_id: str | None, sample_column: str | None = None, ortholog_table: pd.DataFrame | None = None,
          isg_genes: set[str] | None = None) -> dict[str, Any]:
    """Score every cell and write obs ``gate_<slug>`` ("in"/"out") and ``gate_<slug>_score``,
    record the step, and return the result (counts, per sample, coverage, warnings).
    ``X``/``var``: the dataset's log-normalised values (``label_transfer.expression_of``)."""
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    rep = check(gate, adata, dataset_id=dataset_id, ortholog_table=ortholog_table, isg_genes=isg_genes,
                target_var=var)
    if not rep["ok"]:
        raise LearnedGateError(rep["refusal"] or "The gate cannot be applied here.")
    gm = map_genes(gate, var, rep["species"], ortholog_table)
    names = pd.Index(var.index.astype(str))
    score, inside = score_cells(gate, X, names, gm)
    col, score_col = column_names(gate.name)
    record = dict(adata.uns.get(UNS_KEY) or {})
    for c in (col, score_col):
        if c in adata.obs.columns and col not in record:
            raise LearnedGateError(f"This dataset already has a column '{c}' that no learned gate wrote; "
                                   "rename the gate (or the column) first.", 409)
    replaced = col in record
    if replaced:
        forget_column_results(adata, col)
    adata.obs[col] = pd.Categorical(np.where(inside, "in", "out"), categories=["in", "out"])
    adata.obs[score_col] = score.astype(np.float32)
    record[col] = {"gate_id": gate.id, "name": gate.name, "score_column": score_col,
                   "applied": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    adata.uns[UNS_KEY] = record  # replaced, not mutated (write_derived rewrites it)
    warnings = list(rep["warnings"])
    mid = float(np.mean((score >= MID_LOW) & (score <= MID_HIGH))) if score.size else 0.0
    if mid > SHIFT_RATIO * gate.cv.mid_share and mid > SHIFT_MIN:
        warnings.append(f"{mid:.0%} of the cells here score in the undecided middle (0.2-0.8), against "
                        f"{gate.cv.mid_share:.0%} of the training cells: the data may be shifted (depth, chemistry, "
                        "condition or batch), so check the in/out split before using it.")
    scol = sample_column if sample_column is not None else pick_sample_column(adata.obs)
    rows = per_sample(adata.obs, inside, scol or None)
    n_in = int(inside.sum())
    result = {
        "gate_id": gate.id, "name": gate.name, "column": col, "score_column": score_col, "replaced": replaced,
        "n_cells": int(adata.n_obs), "n_in": n_in, "share": round(n_in / max(int(adata.n_obs), 1), 4),
        "threshold": gate.threshold, "mid_share": round(mid, 4), "training_mid_share": round(gate.cv.mid_share, 4),
        "sample_column": scol or None, "per_sample": rows, "coverage": rep["coverage"], "via": rep["via"],
        "species": rep["species"], "gate_species": gate.species, "warnings": warnings, "expression": source,
        "cv": gate.cv.model_dump(), "verdict": verdict(gate.cv),
        "source": {"dataset_id": gate.source.dataset_id, "name": gate.source.name},
        "same_dataset": dataset_id is not None and dataset_id == gate.source.dataset_id,
        "score_note": SCORE_NOTE,
    }
    provenance.record_step(adata, step=STEP_APPLY, tool="scview.learned_gate", params={
        "gate": gate_dict(gate), "column": col, "score_column": score_col,
        "target_genes": list(gm.target), "via": gm.via, "expression": source,
    }, note=f"Learned gate '{gate.name}': {n_in:,} of {int(adata.n_obs):,} cells in")
    return result


def record_training(adata, gate: LearnedGate, report: dict[str, Any]) -> None:
    """The training step in the source dataset's provenance (the selection itself is kept in
    the results store, it is too large for a recipe entry)."""
    from scview.core import provenance

    provenance.record_step(adata, step=STEP_TRAIN, tool="scview.learned_gate", params={
        "gate_id": gate.id, "name": gate.name, "n_selected": gate.source.n_selected,
        "selection_hash": gate.source.selection_hash, "sample_column": gate.source.sample_column,
        "condition_column": gate.source.condition_column, "negatives": gate.source.negatives,
        "excluded_rules": gate.excluded_rules, "C": gate.cv.C, "n_genes": len(gate.genes),
        "auroc": round(gate.cv.auroc, 4), "f1": round(gate.cv.f1, 4), "threshold": gate.threshold,
    }, note=f"Learned gate '{gate.name}' trained on {gate.source.n_selected:,} selected cells")


def selection_digest(cells: Any) -> str:
    return hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


def last_apply(adaptor) -> dict[str, Any] | None:
    from scview.core import results_store

    try:
        return results_store.latest(adaptor, KIND)
    except Exception:  # pragma: no cover - defensive
        return None


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last learned gate applied to this dataset."""
    r = last_apply(adaptor)
    if not r:
        return [], None
    cv = r.get("cv") or {}
    detail = (f"learned gate '{r['name']}' applied: {r['n_in']:,} of {r['n_cells']:,} cells in "
              f"({100 * r['share']:.1f}%), column {r['column']}")
    lines = [f"\n## Learned gate (last applied) [{TAG}]",
             f"- {detail} [{TAG}].",
             f"- Trained on '{(r.get('source') or {}).get('name') or 'a selection'}': out-of-fold AUROC "
             f"{cv.get('auroc', float('nan')):.3f}, F1 {cv.get('f1', float('nan')):.2f}, verdict "
             f"{(r.get('verdict') or {}).get('level', '?')} [{TAG}].",
             f"- Genes found here: {r['coverage']['n_found']} of {r['coverage']['n_genes']} "
             f"({100 * r['coverage']['weight_share']:.0f}% of the weight), via {r.get('via')} [{TAG}]."]
    shares = ", ".join(f"{s['sample']} {100 * s['share']:.1f}%" for s in (r.get("per_sample") or [])[:12])
    if shares:
        lines.append(f"- In-gate share per {r.get('sample_column')}: {shares} [{TAG}].")
    for w in (r.get("warnings") or [])[:3]:
        lines.append(f"- Caution: {w} [{TAG}].")
    lines.append(f"- The gate score is a class-balanced model probability, not a frequency [{TAG}].")
    return lines, detail
