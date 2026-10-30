"""What changes when the genes are a targeted panel (Xenium, CosMx, MERSCOPE).

``spatial.targeted_panel`` says whether a dataset is one. A panel holds a few hundred genes
chosen for a tissue, and a median of about 166 transcripts per cell on the Xenium breast
sample, so several defaults made for whole-transcriptome data do not fit. These are the
defaults of a run on a panel; a parameter the user set always wins (``PipelineParams.explicit``),
and History and the replay script record what ran:

* Normalisation: ``target_sum`` = the median total counts of the cells, not 1e4, which would
  scale 166 counts up 60-fold and inflate log1p's separation of zero from one count.
* Variable genes: a panel of up to ``ALL_GENES_MAX`` genes keeps all of them; the panel was
  selected already, and a dispersion ranking of 300 chosen genes has little left to pick from.
* PCA: at most half the genes, so components never reach the panel's rank.
* QC: MAD stays the mode, with a floor of ``COUNTS_FLOOR`` counts per cell in both modes (the
  squidpy Xenium tutorial's value); a panel cannot satisfy ``min_genes`` = 200, so fixed mode's
  gene floor is 0.
* Doublets: skipped, since Scrublet's simulated doublets assume whole-transcriptome data.
* Cell cycle, dissociation stress: skipped with a sentence when under half of their gene list
  is on the panel. Ribosomal and mitochondrial shares are not called missing either.
* Enrichment: the background is the panel, and each answer carries ``enrichment_caution``:
  a panel chosen for a tissue biases enrichment. It is not refused.
* Ambient RNA: not applicable, there are no empty droplets to measure a soup in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from scview.core import spatial

COUNTS_FLOOR = 10.0
ALL_GENES_MAX = 2000
HIGH_CONTROL_PCT = 5.0  # a cell's share of control counts worth pointing at
HIGH_CONTROL_SHARE = 0.05  # ... when this share of a section's cells is above it

NO_MITO_NOTE = ("This targeted panel has no mitochondrial (MT-) genes, as expected, so % mito is not "
                "part of its QC.")
NO_RIBO_NOTE = ("This targeted panel has no ribosomal-protein genes, as expected, so the "
                "ribosomal-share states do not apply to it.")
DOUBLET_SKIP = ("Scrublet's simulated doublets assume whole-transcriptome data, so doublet detection "
                "does not apply to a targeted panel")
AMBIENT_SKIP = ("Ambient RNA correction does not apply to a targeted panel: it measures the soup in "
                "the empty droplets of a raw matrix, and imaging data has none")
ENRICHMENT_CAUTION = ("Targeted panel: the background is the panel's genes, and a panel chosen for a "
                      "tissue biases enrichment towards what it was designed to measure. Read the "
                      "terms as a description of the panel's genes, not as an unbiased survey.")


def n_genes(adata) -> int:
    """The panel's size: what the import recorded, else the genes measured."""
    rec = spatial._recorded(adata)
    n = rec.get("n_panel_genes")
    try:
        return int(np.asarray(n).ravel()[0]) if n is not None else int(adata.n_vars)
    except (TypeError, ValueError, IndexError):
        return int(adata.n_vars)


def defaults(n: int, explicit: Iterable[str] = (), mode: str = "mad") -> dict[str, Any]:
    """The run's parameters on a panel of ``n`` genes, leaving out those in ``explicit``
    (the user's). ``mode``: the QC threshold mode in force."""
    set_by_user = set(explicit)
    out: dict[str, Any] = {}
    if "target_sum" not in set_by_user:
        out["target_sum_median"] = True
    if "min_genes" not in set_by_user:
        out["min_genes"] = 0
    # A fixed-mode min_counts the user gave is the floor they want.
    if "qc_counts_floor" not in set_by_user and not (mode == "fixed" and "min_counts" in set_by_user):
        out["qc_counts_floor"] = COUNTS_FLOOR
    selection = {"n_top_genes", "hvg_flavor", "hvg_batch_key", "hvg_all_genes"}
    if n <= ALL_GENES_MAX and not selection & set_by_user:
        out["hvg_all_genes"] = True
    if "n_comps" not in set_by_user and n // 2 < 50:
        out["n_comps"] = max(2, n // 2)
    return out


def for_run(adata, params):
    """``params`` with the panel's defaults where the user set nothing; the same object for
    any other dataset."""
    if not spatial.targeted_panel(adata):
        return params
    return replace(params, **defaults(n_genes(adata), params.explicit, params.qc_threshold_mode))


def median_total(totals: np.ndarray, sparse: bool = True) -> float:
    """The target sum of a panel's cells, as ``sc.pp.normalize_total(target_sum=None)`` picks
    it (and ``chunked.normalize_target``): the median over all cells of a sparse matrix, over
    the cells with counts of a dense one; 1 when that is 0."""
    totals = np.asarray(totals, dtype=np.float64).ravel()
    used = totals if sparse else totals[totals > 0]
    return (float(np.median(used)) if used.size else 0.0) or 1.0


def _under_half(present: int, total: int) -> bool:
    return present * 2 < total


def gene_list_skip(what: str, present: int, total: int) -> str:
    return (f"Only {present} of the {total} {what} genes are on this targeted panel, under half of the "
            "list, so no score was written")


def enrichment_caution(adata) -> str | None:
    """The caption for any enrichment of this dataset, or None when it is no panel."""
    return ENRICHMENT_CAUTION if spatial.targeted_panel(adata) else None


def prompt_line(adata) -> str:
    """The sentence for the LLM annotator's prompt, or "" when it is no panel."""
    if not spatial.targeted_panel(adata):
        return ""
    return (f"These cells were measured with a targeted panel of {n_genes(adata)} genes, so only the "
            "genes on the panel can appear as markers: a lineage marker that is missing was probably "
            "not measured, which is no evidence against that cell type.")


def scoring_gene_names(adata) -> list[str]:
    raw = getattr(adata, "raw", None)
    return [str(g) for g in (raw.var_names if raw is not None else adata.var_names)]


def cell_cycle_skip(adata) -> str | None:
    """Why the cell-cycle step does not apply to this panel, or None when it does (or it is no
    panel)."""
    if not spatial.targeted_panel(adata):
        return None
    from scview.core.pipeline import CELL_CYCLE_ALIASES, CELL_CYCLE_G2M_GENES, CELL_CYCLE_S_GENES
    from scview.core.species import match_genes

    names = scoring_gene_names(adata)
    total = len(CELL_CYCLE_S_GENES) + len(CELL_CYCLE_G2M_GENES)
    present = len(match_genes(CELL_CYCLE_S_GENES, names, CELL_CYCLE_ALIASES)) + len(
        match_genes(CELL_CYCLE_G2M_GENES, names, CELL_CYCLE_ALIASES))
    return gene_list_skip("S and G2M", present, total) if _under_half(present, total) else None


def stress_skip(adata) -> str | None:
    """Why the dissociation-stress score does not apply to this panel, or None."""
    if not spatial.targeted_panel(adata):
        return None
    from scview.core import stress

    present = len(stress.signature_genes(scoring_gene_names(adata)))
    total = len(stress.STRESS_GENES)
    return gene_list_skip("dissociation-stress", present, total) if _under_half(present, total) else None


def control_summary(obs) -> dict | None:
    """The share of control counts (``pct_control``): its median, and a sentence when many
    cells of a section have a high one; None without the column."""
    if "pct_control" not in obs.columns:
        return None
    pct = np.nan_to_num(np.asarray(obs["pct_control"], dtype=np.float64))
    high = pct > HIGH_CONTROL_PCT
    out: dict[str, Any] = {"median_pct": round(float(np.median(pct)), 3) if pct.size else 0.0,
                           "share_above": round(float(high.mean()), 4) if pct.size else 0.0,
                           "above_pct": HIGH_CONTROL_PCT, "flag": None}
    libs = obs[spatial.LIBRARY_COLUMN].astype(str).to_numpy() if spatial.LIBRARY_COLUMN in obs.columns \
        else np.full(pct.size, "all", dtype=object)
    flagged = [str(lib) for lib in np.unique(libs) if high[libs == lib].mean() >= HIGH_CONTROL_SHARE]
    if flagged:
        out["flag"] = (f"Many cells of {', '.join(flagged[:4])} have over {HIGH_CONTROL_PCT:g}% control "
                       "counts (negative-control probes and blank codewords): check the run's quality.")
    return out


def assessment(adata) -> dict | None:
    """What the Data Assessment says about the panel path; None when it is no panel."""
    if not spatial.targeted_panel(adata):
        return None
    n = n_genes(adata)
    rec = spatial._recorded(adata)
    tech = spatial.technology(adata)["technology"] if spatial.has_spatial(adata) else "cells"
    obs = adata.obs
    totals = None
    for col in ("total_counts", "transcript_counts"):
        if col in obs.columns:
            totals = obs[col].to_numpy(dtype=np.float64)
            break
    target = round(median_total(totals)) if totals is not None else None
    used = defaults(n)
    lines = [f"Targeted panel of {n:,} genes: the run uses the panel's own settings unless you set them."]
    lines.append("Normalisation to the median total counts of the cells"
                 + (f" (about {target:,})" if target else "") + ", not 10,000.")
    lines.append("All genes go into PCA, without variable-gene selection." if used.get("hvg_all_genes")
                 else f"Variable genes are still selected, from {n:,}.")
    lines.append(f"QC keeps MAD outliers as the rule, with a floor of {COUNTS_FLOOR:g} counts per cell; "
                 "a gene floor does not fit a panel.")
    lines.append(f"{DOUBLET_SKIP}.")
    return {"n_genes": n, "technology": tech, "panel_name": str(rec.get("panel_name") or ""),
            "sentences": lines, "control": control_summary(obs),
            # The steps' form shows these (only what the user changes is sent).
            "defaults": {"target_sum": target, "min_genes": 0, "min_counts": COUNTS_FLOOR,
                         "qc_counts_floor": COUNTS_FLOOR,
                         **({"n_comps": used["n_comps"]} if "n_comps" in used else {})},
            "all_genes": bool(used.get("hvg_all_genes"))}
