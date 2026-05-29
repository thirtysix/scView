"""Ingestion engine — multi-sample merge (§3a).

Combine several loaded units into one dataset with a ``sample`` label. The merge
is **opt-in** and the engine makes the safe choice obvious:

* Gene-axis alignment defaults to **intersection** (``join="inner"`` — no
  fabricated zeros); **union** is offered with a warning.
* The classic silent disaster — merging an Ensembl-indexed sample with a
  symbol-indexed one (near-zero overlap) — is actively guarded against:
  :func:`plan_merge` flags a suspiciously small intersection (< 50 % of the
  smaller var set), diagnoses identifier-mismatch vs a genuine subset, and for a
  mismatch proposes resetting a sample's ``var_names`` to a column that shares
  the other's basis (preferring Ensembl). Index resets are never applied
  silently — the caller (the wizard) confirms.

``plan_merge`` is read-only analysis for the UI; ``merge_units`` applies a
resolved plan. See ``docs/INGESTION_ENGINE.md`` §3a.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

import anndata as ad
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Ensembl gene ids (human ENSG, mouse ENSMUSG, other species ENS<sp>G).
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}", re.IGNORECASE)

# Fraction of the smaller var set below which the intersection is "suspicious".
_SUSPICIOUS_OVERLAP = 0.5
# Soft cell-count threshold above which we warn about interaction sluggishness.
_LARGE_MERGE_CELLS = 200_000


class VarBasis(str, Enum):
    ensembl = "ensembl"
    symbol = "symbol"
    mixed = "mixed"
    unknown = "unknown"


class MergeJoin(str, Enum):
    inner = "inner"  # intersection — default
    outer = "outer"  # union, missing filled with 0


class VarReset(BaseModel):
    """Proposal to re-index one sample's genes onto a shared basis."""

    sample: str
    via_column: str
    from_basis: VarBasis


class Reconciliation(BaseModel):
    needed: bool = False
    feasible: bool = False
    target_basis: VarBasis = VarBasis.unknown
    resets: list[VarReset] = Field(default_factory=list)
    overlap_before: int = 0
    overlap_after: int = 0
    message: str = ""


class MergePlan(BaseModel):
    samples: list[str] = Field(default_factory=list)
    per_sample_genes: dict[str, int] = Field(default_factory=dict)
    bases: dict[str, VarBasis] = Field(default_factory=dict)
    intersection: int = 0
    union: int = 0
    suspicious_low_overlap: bool = False
    reconciliation: Reconciliation | None = None
    recommended_join: MergeJoin = MergeJoin.inner
    est_cells: int = 0
    est_genes: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Basis classification
# ---------------------------------------------------------------------------


def classify_var_basis(names) -> VarBasis:
    """Classify a gene-name iterable as Ensembl-id vs symbol by pattern."""
    sample = [str(n) for n in list(names)[:500]]
    if not sample:
        return VarBasis.unknown
    ens = sum(1 for n in sample if _ENSEMBL_RE.match(n)) / len(sample)
    if ens >= 0.7:
        return VarBasis.ensembl
    if ens <= 0.1:
        return VarBasis.symbol
    return VarBasis.mixed


def _find_basis_column(adata: ad.AnnData, target: VarBasis) -> str | None:
    """Find a .var column whose values are in the target basis (for re-indexing)."""
    for col in adata.var.columns:
        if classify_var_basis(adata.var[col].astype(str).tolist()) == target:
            return col
    return None


# ---------------------------------------------------------------------------
# Planning (read-only)
# ---------------------------------------------------------------------------


def plan_merge(adatas: dict[str, ad.AnnData]) -> MergePlan:
    """Analyse a set of loaded units and produce a MergePlan for the UI."""
    labels = list(adatas)
    var_sets = {lab: set(map(str, a.var_names)) for lab, a in adatas.items()}
    per_sample = {lab: len(s) for lab, s in var_sets.items()}
    bases = {lab: classify_var_basis(list(a.var_names)) for lab, a in adatas.items()}

    inter = set.intersection(*var_sets.values()) if var_sets else set()
    union = set.union(*var_sets.values()) if var_sets else set()
    smaller = min(per_sample.values()) if per_sample else 0
    suspicious = smaller > 0 and len(inter) < _SUSPICIOUS_OVERLAP * smaller

    recon = _plan_reconciliation(adatas, bases, len(inter)) if suspicious else None

    est_cells = sum(a.n_obs for a in adatas.values())
    inner_genes = recon.overlap_after if (recon and recon.feasible) else len(inter)

    warnings: list[str] = []
    if est_cells > _LARGE_MERGE_CELLS:
        warnings.append(
            f"This merge has ~{est_cells:,} cells. Large merged datasets can make selection "
            "and recoloring feel sluggish."
        )
    if len({_counts_like(a) for a in adatas.values()}) > 1:
        warnings.append(
            "Some samples look like raw counts and others like normalized values — merging "
            "them mixes incompatible scales. Use samples processed the same way."
        )

    return MergePlan(
        samples=labels,
        per_sample_genes=per_sample,
        bases=bases,
        intersection=len(inter),
        union=len(union),
        suspicious_low_overlap=suspicious,
        reconciliation=recon,
        recommended_join=MergeJoin.inner,
        est_cells=est_cells,
        est_genes=inner_genes,
        warnings=warnings,
    )


def _plan_reconciliation(
    adatas: dict[str, ad.AnnData], bases: dict[str, VarBasis], overlap_before: int
) -> Reconciliation:
    distinct = {b for b in bases.values() if b in (VarBasis.ensembl, VarBasis.symbol)}

    # Same basis but low overlap → a genuine subset (DEG-filtered, coding-only),
    # not an identifier problem. Inform, don't reconcile.
    if len(distinct) < 2:
        return Reconciliation(
            needed=False,
            feasible=False,
            overlap_before=overlap_before,
            overlap_after=overlap_before,
            message=(
                "These samples use the same gene-label style but share few genes — one is "
                "probably pre-filtered (e.g. DEGs or protein-coding only). The merge keeps the "
                "shared genes; nothing to reconcile."
            ),
        )

    # Different bases → re-index everything onto Ensembl where possible.
    target = VarBasis.ensembl
    resets: list[VarReset] = []
    new_sets: dict[str, set[str]] = {}
    feasible = True
    for lab, a in adatas.items():
        if bases[lab] == target:
            new_sets[lab] = set(map(str, a.var_names))
            continue
        col = _find_basis_column(a, target)
        if col is None:
            feasible = False
            new_sets[lab] = set(map(str, a.var_names))
        else:
            resets.append(VarReset(sample=lab, via_column=col, from_basis=bases[lab]))
            new_sets[lab] = set(a.var[col].astype(str))

    overlap_after = len(set.intersection(*new_sets.values())) if new_sets else 0
    basis_desc = ", ".join(f"{lab}: {bases[lab].value}" for lab in adatas)

    if not feasible:
        return Reconciliation(
            needed=True,
            feasible=False,
            target_basis=target,
            resets=resets,
            overlap_before=overlap_before,
            overlap_after=overlap_after,
            message=(
                f"Samples label genes differently ({basis_desc}) and only {overlap_before} "
                "matched. At least one sample doesn't carry the others' gene IDs, so they can't "
                "be matched automatically — re-export with a shared identifier, or merge with "
                "the union option."
            ),
        )

    switched = ", ".join(f"{r.sample} → its '{r.via_column}' column" for r in resets)
    return Reconciliation(
        needed=True,
        feasible=True,
        target_basis=target,
        resets=resets,
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        message=(
            f"Samples label genes differently ({basis_desc}); only {overlap_before} matched. "
            f"Switching {switched} lifts the overlap to {overlap_after} shared genes. Use "
            f"{target.value} IDs for the merge?"
        ),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def merge_units(
    adatas: dict[str, ad.AnnData],
    *,
    join: MergeJoin = MergeJoin.inner,
    sample_label: str = "sample",
    reconciliation: Reconciliation | None = None,
) -> ad.AnnData:
    """Merge loaded units into one AnnData, applying a resolved reconciliation."""
    if not adatas:
        raise ValueError("No samples to merge.")

    work = {lab: a.copy() for lab, a in adatas.items()}

    if reconciliation and reconciliation.needed and reconciliation.feasible:
        for reset in reconciliation.resets:
            a = work[reset.sample]
            a.var_names = a.var[reset.via_column].astype(str)
            a.var_names_make_unique()

    # fill_value=0: for a union join, a gene absent from a sample means zero
    # expression there, not NaN (which would break downstream analysis).
    merged = ad.concat(
        work, join=join.value, label=sample_label, index_unique=":", fill_value=0
    )
    merged.obs[sample_label] = merged.obs[sample_label].astype("category")
    return merged


def _counts_like(adata: ad.AnnData) -> bool:
    """Cheap heuristic: do the first values look like integer counts?"""
    X = adata.X
    vals = X.data[:1000] if hasattr(X, "data") else np.asarray(X).ravel()[:1000]
    if len(vals) == 0:
        return True
    return bool(np.all(np.mod(vals, 1) == 0))
