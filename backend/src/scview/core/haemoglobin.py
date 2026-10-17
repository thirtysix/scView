"""Haemoglobin RNA per cluster: real red cells, or ambient spill-over.

Red cells and their precursors are full of haemoglobin mRNA. When red cells lyse
during sample preparation that RNA floats free and lands in every droplet
(ambient RNA), so other cells show a little haemoglobin too. The same number
means different things, so a cluster is read against its own markers
(``core/cluster_qc.py``):

- Erythroid: at least MIN_ERYTHROID_MARKERS of its top markers are haemoglobin
  or other red-cell-lineage genes, or its median cell is at least
  ERYTHROID_MEDIAN_PCT haemoglobin (no other cell gets there). Its haemoglobin is
  its own and is never called contamination.
- Ambient in one cluster (flag ``hb_ambient``): no erythroid markers, and a mean
  haemoglobin share far above the other non-erythroid clusters.
- Ambient across the data (a dataset note): the typical non-erythroid cluster
  carries haemoglobin RNA.

Per cell: haemoglobin counts / total counts (%). Per cluster the MEAN over its
cells, since most non-erythroid cells have no haemoglobin read and the median is
0. Measured on scView's own processing, 2026-10-17:

- Kang 2018 PBMC (13 cell types): Eryth median 52% and mean 48%, top markers
  HBB, HBA2, HBA1, HBM, HBD; the 12 other clusters 0.008-0.015% (Mk 0.015%).
- paul15 mouse marrow (19 clusters; Hba-a2 and Hbb-b1 are the only haemoglobin
  genes in its panel): 1Ery mean 0.48% with both among its top markers; 2Ery-4Ery
  0.01-0.06% with no haemoglobin marker but Klf1, Rhd, Ermap, Cpox or Car1 in
  their top 20; non-erythroid clusters at most 0.009%. Haemoglobin level alone
  cannot find erythroid progenitors; their markers can.
"""

from __future__ import annotations

import numpy as np

from scview.core.gene_filters import top_markers
from scview.core.species import match_genes

HUMAN_HB = ("HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBE1", "HBZ", "HBM", "HBQ1")
# Current mouse names and the older ones some annotations still use
# (Hbb-b1 / Hbb-b2 for Hbb-bs / Hbb-bt, as in paul15).
MOUSE_HB = ("Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-b1", "Hbb-b2", "Hbb-y",
            "Hbb-bh1", "Hba-x", "Hbq1a", "Hbq1b")
# Red-cell-lineage genes besides haemoglobin. Left out on purpose: GATA1 and TFRC
# (also megakaryocytes, mast cells, any dividing cell) and CA2, PRDX2, BLVRB
# (broadly expressed), so a megakaryocyte or platelet cluster, where ambient
# haemoglobin often shows, is not taken for erythroid.
ERYTHROID_GENES = ("ALAS2", "GYPA", "GYPB", "KLF1", "CA1", "RHD", "RHAG", "ERMAP", "CPOX",
                   "HEMGN", "AHSP", "SLC4A1", "EPB42", "SPTA1", "KEL", "TRIM58")
ERYTHROID_ALIASES = {"CA1": ("Car1",)}  # the mouse symbol differs

TOP_MARKERS = 20  # top markers looked at, after dropping ribosomal/mito genes
MIN_ERYTHROID_MARKERS = 2
# Kang's red cells: median 52%. Ambient RNA is a small share of a droplet's
# reads, and red cells only part of it, so no other cell type reaches 20%.
ERYTHROID_MEDIAN_PCT = 20.0

# The one-cluster flag: all three must hold. 1% is about 70 times the highest
# non-erythroid cluster in the clean Kang and paul15 data.
AMBIENT_CLUSTER_PCT = 1.0
AMBIENT_FOLD = 3.0  # times the typical non-erythroid cluster
Z_OUTLIER = 3.0  # robust z against the other non-erythroid clusters
SCALE_FLOOR = 0.1  # floor on their robust spread, in percentage points
# The dataset note: the typical non-erythroid cluster at 0.5% (30-60 times the
# clean data) means haemoglobin RNA is everywhere, not in one population.
AMBIENT_WIDE_PCT = 0.5

FLAG_HELP = (
    "Haemoglobin RNA without erythroid markers: far more haemoglobin than the other "
    "non-erythroid clusters, while its own markers are not red-cell genes. Likely ambient "
    "RNA from lysed red cells or red-cell doublets, not a cell type; it can blur markers "
    "and DE."
)
ERYTHROID_NOTE = (
    "Erythroid: red-cell lineage by its own markers. Its haemoglobin is real expression, "
    "not ambient contamination."
)


def hb_genes(names, species: str = "unknown") -> list[str]:
    """The haemoglobin genes present, in the dataset's own spelling, for ``species``
    (``core/species.py``); both lists when it is unknown (they never collide)."""
    pool = HUMAN_HB if species == "human" else MOUSE_HB if species == "mouse" else HUMAN_HB + MOUSE_HB
    return match_genes(pool, names)


def hb_mask(names, species: str = "unknown") -> np.ndarray:
    """Boolean mask over ``names`` of the haemoglobin genes (see ``hb_genes``)."""
    present = set(hb_genes(names, species))
    return np.fromiter((str(n) in present for n in names), dtype=bool, count=len(names))


def erythroid_markers(adata, column: str) -> dict[str, list[str]] | None:
    """Per group: the haemoglobin and erythroid genes among its top markers.

    None when ``column`` has no stored markers (nothing to tell erythroid cells by)."""
    top = top_markers(adata, column, TOP_MARKERS)
    if top is None:
        return None
    wanted = {g.upper() for g in (*HUMAN_HB, *MOUSE_HB, *ERYTHROID_GENES)}
    wanted |= {a.upper() for names in ERYTHROID_ALIASES.values() for a in names}
    return {g: [x for x in genes if x.upper() in wanted] for g, genes in top.items()}


def is_erythroid(markers: list[str] | None, median_pct) -> bool:
    """Erythroid by its markers, or by a red-cell level of haemoglobin."""
    if markers is not None and len(markers) >= MIN_ERYTHROID_MARKERS:
        return True
    try:
        return float(median_pct) >= ERYTHROID_MEDIAN_PCT
    except (TypeError, ValueError):
        return False


def ambient_outlier(z: float, mean_pct, typical) -> bool:
    """One non-erythroid cluster with far more haemoglobin than the others."""
    try:
        mean, typ = float(mean_pct), float(typical)
    except (TypeError, ValueError):
        return False
    return z >= Z_OUTLIER and mean >= AMBIENT_CLUSTER_PCT and mean >= AMBIENT_FOLD * typ
