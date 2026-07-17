"""Filter ribosomal and mitochondrial genes out of an ORA query.

In single-cell data the top marker genes of a cluster are frequently dominated by
ribosomal-protein and mitochondrial transcripts (a well-known technical/ambient
artifact). Because those genes are shared across almost every cell-type signature
and pathway, they drive two failure modes in over-representation analysis:

  1. a generic "translation / ribosome" program outranks the real biology, and
  2. cell-type-signature collections (MSigDB C8) light up for the wrong tissue,
     because ribosomal genes are common to all of them.

Removing ribosomal/mitochondrial genes from the *query* (never from the
background universe, which still represents what was measured) fixes both.
Measured on the bundled ovary dataset (17 clusters, default collections incl. C8):
off-tissue #1 labels 3 -> 0, the translation artifact 3 -> 0, and correct ovary
labels 11 -> 17. Dropping C8 instead removed the off-tissue labels but also the
correct annotation (11 -> 1) and made the translation artifact worse.

Only the query is filtered, and it is opt-out (default on) so a study of ribosome
biogenesis can keep them.
"""

from __future__ import annotations

import re

# Cytoplasmic ribosomal proteins (RPL*/RPS*, including RPLP*/RPSA and species
# lowercase) and mitochondrial ribosomal proteins (MRPL*/MRPS*). The RPS6K*
# kinases are NOT ribosomal proteins and are excluded from the match. Mitochondrially
# encoded transcripts are MT-* (the hyphen keeps metallothioneins like MT1A out).
_RIBO = re.compile(r"^(?!RPS6K)(RP[SL]|MRP[SL])", re.IGNORECASE)
_MITO = re.compile(r"^MT-", re.IGNORECASE)


def is_ribo_mito(gene: str) -> bool:
    """True if the gene symbol is a ribosomal-protein or mitochondrial transcript."""
    return bool(_RIBO.match(gene) or _MITO.match(gene))


def select_query_genes(
    ranked_genes: list[str], n: int, *, exclude_ribo_mito: bool = True
) -> list[str]:
    """Top ``n`` query genes from a rank-ordered marker list.

    With ``exclude_ribo_mito`` (the default), ribosomal/mitochondrial genes are
    dropped *before* taking the top ``n``, so the query still holds ``n`` genes of
    real biology rather than being padded out by artifacts.
    """
    genes = ranked_genes
    if exclude_ribo_mito:
        genes = [g for g in genes if not is_ribo_mito(g)]
    return genes[:n]
