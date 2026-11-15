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

import gzip
import hashlib
import re
from functools import lru_cache
from pathlib import Path

# Cytoplasmic ribosomal proteins (RPL*/RPS*, including RPLP*/RPSA and species
# lowercase) and mitochondrial ribosomal proteins (MRPL*/MRPS*). The RPS6K*
# kinases are NOT ribosomal proteins and are excluded from the match. Mitochondrially
# encoded transcripts are MT-* (the hyphen keeps metallothioneins like MT1A out).
_RIBO = re.compile(r"^(?!RPS6K)(RP[SL]|MRP[SL])", re.IGNORECASE)
_MITO = re.compile(r"^MT-", re.IGNORECASE)


def is_ribo_mito(gene: str) -> bool:
    """True if the gene symbol is a ribosomal-protein or mitochondrial transcript."""
    return bool(_RIBO.match(gene) or _MITO.match(gene))


# Genes with no name, only an identifier: a bare Ensembl id (ENSG00000289963, a gene
# GENCODE has not named), a clone-based id (AC012345.1, AL..., AP..., the GenBank
# accession of the clone the gene was found on) or an old GENCODE clone name
# (RP11-34P13.7, CTD-2245F17.3). They tell a reader, a database lookup or a model
# nothing. LINC01857 and ADAM7-AS1 are names (HGNC's) and stay.
_ENSEMBL = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")
_CLONE = re.compile(r"^[A-Z]{1,2}\d{5,6}\.\d+$")
_OLD_CLONE = re.compile(r"^(RP\d{1,2}|CT[ABCD]|XXbac|XXyac|KB|CH\d{2}|LA16c|WI2|GS1)-[\w.-]+$")


def is_unnamed(gene: str) -> bool:
    """True if the gene has an identifier but no name (see ``_ENSEMBL``/``_CLONE``)."""
    g = str(gene)
    return bool(_ENSEMBL.match(g) or _CLONE.match(g) or _OLD_CLONE.match(g))


# Pseudogenes (FTH1P20, ACSL3P1, MTND1P23, Rpl13a-ps1): named, but by gene type, which no regex
# tells (MMP9, SERP1 also end in P + digits). A shipped list of 24,374 human + mouse symbols, every
# locus of which is a pseudogene in GENCODE v32 / v44 / vM23 / vM33 (the 10x references), none
# coding in any (backend/scripts/make_pseudogenes.py; 63 kB). Shipped, not the GENCODE table
# gene_annotation fetches, so marker lists never wait on the network and are the same offline.
PSEUDOGENES = Path(__file__).resolve().parents[1] / "resources" / "pseudogenes.txt.gz"
PSEUDOGENES_SHA256 = "350937ad48586d3a8c601c280ded4b23ef583b34815888d479f9783449c60d6c"


@lru_cache(maxsize=1)
def pseudogenes() -> frozenset[str]:
    """The shipped pseudogene symbols (case as written: human upper case, mouse title case)."""
    raw = PSEUDOGENES.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PSEUDOGENES_SHA256:
        raise RuntimeError(f"{PSEUDOGENES.name} is not the pinned file; rebuild it with "
                           "backend/scripts/make_pseudogenes.py")
    lines = gzip.decompress(raw).decode().splitlines()
    return frozenset(x for x in lines if x and not x.startswith("#"))


def is_pseudogene(gene: str) -> bool:
    """True if the symbol is a pseudogene in GENCODE (see ``PSEUDOGENES``)."""
    return str(gene) in pseudogenes()


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


def top_markers(adata, column: str, n: int) -> dict[str, list[str]] | None:
    """Per group of ``column``: its top ``n`` stored markers, best first, with
    ribosomal/mitochondrial genes dropped (as for the ORA query). Used to check a
    cluster's identity against a signature (stress, erythroid genes).

    None when ``column`` has no stored markers."""
    from scview.core.ora_background import resolve_markers_key

    key = resolve_markers_key(adata.uns, column)
    if key is None:
        return None
    try:
        names = adata.uns[key]["names"]
        groups = list(names.dtype.names)
    except (KeyError, TypeError, AttributeError):
        return None
    out: dict[str, list[str]] = {}
    for g in groups:
        top: list[str] = []
        for x in names[g]:  # the full ranked list: stop once the top are in
            if len(top) == n:
                break
            if not is_ribo_mito(str(x)):
                top.append(str(x))
        out[str(g)] = top
    return out
