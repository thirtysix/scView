"""How well a CellTypist model suits a dataset, and how sure its calls were.

Three checks around the CellTypist annotation step (``pipeline._annotate_celltypist``):

1. Species: each catalog model is human, mouse or another species (read from its
   name and description). A model of the other species shares almost no gene
   symbols with the data (human CD3E, mouse Cd3e), so the UI lists the dataset's
   species first and warns on a mismatch.
2. Gene overlap: CellTypist matches gene names exactly and predicts from the
   model genes the dataset has, dropping the rest of the model's weights. With
   under half of them (``MIN_OVERLAP``) the scores rest on a minority of what the
   model learned, and labels drift towards types whose genes happen to be present.
3. Prediction confidence per label: ``{target}_confidence`` is each cell's highest
   class probability. CellTypist's probabilities are one-vs-rest (a sigmoid per
   class), and 0.5 (``LOW_CONFIDENCE``) is its own cut for calling a class in
   probability-match mode, so a cell below it fits no class of the model well. A
   label whose median cell is below it is flagged, and a dataset whose median cell
   is below it gets a "model may not match this tissue" caution.

The run's model, species and overlap are kept in uns ``scview_celltypist`` (a JSON
registry by column); the per-label table is computed from obs on request, so it
follows later renames of the labels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

RUNS_KEY = "scview_celltypist"
MIN_OVERLAP = 0.5  # share of the model's genes the dataset must have
LOW_CONFIDENCE = 0.5  # CellTypist's own probability cut (p_thres)

ModelSpecies = Literal["human", "mouse", "other"]

_MOUSE_RE = re.compile(r"mouse|\bmice\b|murine", re.IGNORECASE)
_OTHER_RE = re.compile(r"macaque|monkey|\bpig\b|zebrafish|\brat\b", re.IGNORECASE)

OVERLAP_NOTE = (
    "CellTypist predicts from the model genes the dataset has and drops the rest of "
    "the model's weights, so with under half of them the labels rest on a minority of "
    "what the model learned."
)
CONFIDENCE_NOTE = (
    "Confidence is each cell's highest class probability. CellTypist scores every class "
    "on its own, and below 0.5 (its own cut) a cell fits no class of the model well."
)


def model_species(name: str, description: str = "") -> ModelSpecies:
    """Which species a catalog model was trained on, from its name and description.
    Every model in the catalog that names no species is a human one."""
    text = f"{name.replace('_', ' ')} {description}"
    if _MOUSE_RE.search(text):
        return "mouse"
    if _OTHER_RE.search(text):
        return "other"
    return "human"


def species_warning(model: ModelSpecies | str, dataset: str) -> str:
    """A sentence when the model's species is not the dataset's ("" when it is, or
    when either is unknown)."""
    if dataset not in ("human", "mouse") or model == dataset:
        return ""
    cells = "cells of another species" if model == "other" else f"{model} cells"
    return (
        f"This looks like a {dataset} dataset, but the model was trained on {cells}: "
        "gene symbols are matched exactly (human CD3E, mouse Cd3e), so few genes will be "
        "shared and the labels are unreliable."
    )


def gene_overlap(model_genes: Iterable[str], dataset_genes: Iterable[str]) -> dict:
    """How many of the model's genes the dataset has, matched exactly as CellTypist
    matches them."""
    genes = pd.Index([str(g) for g in model_genes]).unique()
    shared = int(genes.isin(pd.Index([str(g) for g in dataset_genes])).sum())
    n = int(len(genes))
    return {"n_model_genes": n, "n_shared_genes": shared,
            "gene_overlap": round(shared / n, 4) if n else 0.0}


def overlap_caution(
    overlap: float | None, n_shared: int | None = None, n_model: int | None = None
) -> str:
    if overlap is None or overlap >= MIN_OVERLAP:
        return ""
    share = f"{overlap:.0%}" if overlap >= 0.01 else f"{overlap:.2%}"
    has = (f"{n_shared:,} of the model's {n_model:,} genes ({share})"
           if n_shared is not None and n_model else f"{share} of the model's genes")
    return (
        f"The dataset has only {has}, under {MIN_OVERLAP:.0%}. "
        f"{OVERLAP_NOTE} Check the species, the gene naming (symbols, not Ensembl ids) "
        "and whether the matrix holds all genes or only the variable ones."
    )


def _registry(adata) -> dict:
    raw = adata.uns.get(RUNS_KEY)
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def record_run(adata, column: str, entry: dict) -> None:
    """Keep what a CellTypist run into ``column`` used (model, species, overlap).
    Replaces the uns entry rather than editing it, as obs_writes needs."""
    reg = _registry(adata)
    reg[column] = entry
    adata.uns[RUNS_KEY] = json.dumps(reg)


def drop_run(adata, column: str) -> None:
    """Forget the record of ``column`` (it was annotated again another way)."""
    reg = _registry(adata)
    if reg.pop(column, None) is not None:
        adata.uns[RUNS_KEY] = json.dumps(reg)


def run_columns(adata) -> list[str]:
    """Columns with a recorded CellTypist run whose labels are still in obs."""
    return [c for c in _registry(adata) if c in adata.obs.columns]


def summarize(adata, column: str) -> dict | None:
    """The run's record plus a per-label confidence table, or None when ``column``
    has no CellTypist confidence beside it."""
    conf_col = f"{column}_confidence"
    if column not in adata.obs.columns or conf_col not in adata.obs.columns:
        return None
    entry = _registry(adata).get(column)
    entry = entry if isinstance(entry, dict) else {}
    labels = adata.obs[column].astype(object).where(adata.obs[column].notna(), None)
    conf = pd.to_numeric(adata.obs[conf_col], errors="coerce").to_numpy(dtype=float)
    percell_col = f"{column}_percell"
    percell = (adata.obs[percell_col].astype(str).to_numpy()
               if percell_col in adata.obs.columns else None)
    lab = labels.to_numpy()
    col = adata.obs[column]
    order = ([str(c) for c in col.cat.categories] if isinstance(col.dtype, pd.CategoricalDtype)
             else sorted({str(v) for v in lab if v is not None}))

    rows = []
    lab_str = np.array([str(v) if v is not None else "" for v in lab], dtype=object)
    for name in order:
        mask = lab_str == name
        n = int(mask.sum())
        if n == 0:
            continue
        c = conf[mask]
        c = c[np.isfinite(c)]
        median = float(np.median(c)) if c.size else None
        row = {
            "label": name,
            "n_cells": n,
            "median_confidence": round(median, 4) if median is not None else None,
            "pct_below": round(float((c < LOW_CONFIDENCE).mean() * 100), 1) if c.size else None,
            "low_confidence": median is not None and median < LOW_CONFIDENCE,
        }
        if percell is not None:
            # Cells whose own best class is the label the majority vote gave them.
            row["pct_agree"] = round(float((percell[mask] == name).mean() * 100), 1)
        rows.append(row)

    finite = conf[np.isfinite(conf)]
    overall = float(np.median(finite)) if finite.size else None
    cautions: list[str] = []
    for text in (
        species_warning(entry.get("model_species", ""), entry.get("dataset_species", "")),
        overlap_caution(entry.get("gene_overlap"), entry.get("n_shared_genes"),
                        entry.get("n_model_genes")),
    ):
        if text:
            cautions.append(text)
    if overall is not None and overall < LOW_CONFIDENCE:
        cautions.append(
            f"The median cell's confidence is {overall:.2f}, under {LOW_CONFIDENCE}: most cells "
            "fit no class of this model well. The model may not match this tissue; try one "
            "trained on it, or the AI or marker-score method."
        )
    n_low = sum(r["low_confidence"] for r in rows)
    return {
        "column": column,
        "confidence_column": conf_col,
        "percell_column": percell_col if percell is not None else None,
        "model": entry.get("model"),
        "model_species": entry.get("model_species"),
        "dataset_species": entry.get("dataset_species"),
        "gene_overlap": entry.get("gene_overlap"),
        "n_model_genes": entry.get("n_model_genes"),
        "n_shared_genes": entry.get("n_shared_genes"),
        "n_cells": int(finite.size),
        "median_confidence": round(overall, 4) if overall is not None else None,
        "pct_below": round(float((finite < LOW_CONFIDENCE).mean() * 100), 1) if finite.size else None,
        "n_low_labels": int(n_low),
        "labels": rows,
        "cautions": cautions,
        "low_confidence": LOW_CONFIDENCE,
        "min_overlap": MIN_OVERLAP,
        "note": CONFIDENCE_NOTE,
    }
