"""Dissociation-stress score per cell, and whether a cluster looks driven by it.

Digesting tissue with enzymes at 37 °C switches on immediate-early and heat-shock
genes within minutes, so a population can form around the handling rather than a
cell state (van den Brink et al. 2017, Nat Methods 14:935, PMID 28960196). Each
cell gets a score (scanpy's ``score_genes``: the signature's mean expression minus
that of expression-matched control genes) in ``obs['dissociation_stress_score']``.

A cluster is flagged in the per-cluster QC (``core/cluster_qc.py``) only when both
signs agree: its median score is an outlier against the other clusters, and its own
top markers are signature genes. The score alone is not enough: monocytes and
dendritic cells express FOS, JUNB and DUSP1 at baseline. On Kang 2018 PBMCs, which
were never digested, they score highest (median 0.50-0.54 against 0.22 for the
typical cell type, about 2 robust SDs), and no signature gene is among their top
20 markers.
"""

from __future__ import annotations

import logging

from scview.core.gene_filters import top_markers
from scview.core.species import match_genes

logger = logging.getLogger(__name__)

# A subset of van den Brink et al.'s Supplementary Table 5 (140 mouse genes up in
# the dissociation-affected clusters; all 22 below are on it, checked against the
# supplement): the AP-1, EGR and NR4A immediate-early transcription factors, other
# early-response genes, and the heat-shock chaperones. The full table also holds
# broadly expressed genes (Actg1, Eif1, Ubc, Ddx5) and muscle genes (Des, Ankrd1)
# from the muscle stem cells it was measured in, which would dilute a score in
# other tissues. Human symbols; each mouse ortholog is the title-cased symbol
# (Fos, Hspa1a), so matching ignores case.
STRESS_GENES = (
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "NR4A1",
    "IER2", "IER3", "ZFP36", "DUSP1", "KLF6", "SOCS3", "BTG2", "PPP1R15A",
    "HSPA1A", "HSPA1B", "HSPA8", "HSPB1", "HSP90AA1", "DNAJB1",
)
SCORE_COLUMN = "dissociation_stress_score"
MIN_GENES = 8  # fewer present (a targeted panel, another species): no score

# The per-cluster flag. The score is in log-expression units.
Z_OUTLIER = 3.0  # median score above the typical cluster's by 3 robust SDs
MIN_DIFF = 0.1  # and by at least this much, so near-identical clusters don't flag
SCALE_FLOOR = 0.05  # floor on the robust spread of the cluster medians
TOP_MARKERS = 20  # top markers looked at, after dropping ribosomal/mito genes
MIN_SIGNATURE_MARKERS = 3

FLAG_HELP = (
    "Possibly driven by dissociation stress: immediate-early and heat-shock genes "
    "(FOS, JUN, HSPA1A ...), induced when tissue is digested, score far above the other "
    "clusters and lead its markers. It may be an artifact of sample handling rather than "
    "a cell type; activated immune cells can express these genes too."
)


def signature_genes(names) -> list[str]:
    """The signature genes present, in the dataset's own spelling."""
    return match_genes(STRESS_GENES, names)


def signature_markers(adata, column: str) -> dict[str, list[str]] | None:
    """Per group: the signature genes among its top markers, best first.

    None when ``column`` has no stored markers (nothing to check the score against).
    """
    top = top_markers(adata, column, TOP_MARKERS)
    if top is None:
        return None
    wanted = {g.upper() for g in STRESS_GENES}
    return {g: [x for x in genes if x.upper() in wanted] for g, genes in top.items()}
