"""Experiment-specific background (universe) for over-representation analysis.

gseapy's ``enrich(background=None)`` falls back to "all genes in your gene_sets",
i.e. the union of every gene named by the selected MSigDB collections. That
universe contains genes which were never measured in the experiment, so the
hypergeometric test is not conditioned on what the assay could actually detect.

The genes ranked by ``rank_genes_groups`` are exactly the genes that were
testable, so they are the universe this module recovers.

Measured on the bundled Kang IFN-beta sample data (13 clusters, Hallmark +
GO:BP, top-100 markers by score): the implicit universe is 18,301 genes against
14,053 tested. The net count of significant terms barely moves (3026 -> 3018),
but 13.4% of the calls flip (206 lost, 198 gained), because both the universe
and each term's effective size shrink together and partly cancel. The fix is
therefore about validity, not about shrinking the term list; expect a larger
effect on shallower datasets where fewer genes are detected.

This module is additive: it writes only ``ora_meta__*`` sidecar keys and never
modifies or removes the pre-existing ``enrichment__*`` keys or their payload
shape. Deleting this module and reverting its call sites restores the previous
behaviour, leaving any orphaned sidecars inert.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bump when a change invalidates cached enrichment payloads. Entries whose
# sidecar is missing (pre-fix) or older than this are recomputed.
# v3: ribosomal/mitochondrial genes are excluded from the query by default, so
# every pre-v3 payload was computed from a different query gene list.
ORA_VERSION = 3

META_PREFIX = "ora_meta__"


def meta_key(column: str, group: str) -> str:
    """Sidecar key holding provenance for ``enrichment__{column}__{group}``.

    Deliberately does NOT start with ``enrichment__``: assessor, assistant and
    pipeline all scan that prefix and parse the obs column out of the key, so a
    sidecar sharing it would be miscounted as a result.
    """
    return f"{META_PREFIX}{column}__{group}"


def resolve_markers_key(uns: Any, column: str = "") -> str | None:
    """Locate the rank_genes_groups uns key backing ``column``."""
    keyed = f"rank_genes_groups__{column}" if column else "rank_genes_groups"
    if keyed in uns:
        return keyed
    if "rank_genes_groups" in uns:
        rgg = uns["rank_genes_groups"]
        if not column:
            return "rank_genes_groups"
        try:
            if rgg.get("params", {}).get("groupby") == column:
                return "rank_genes_groups"
        except AttributeError:
            pass
    return None


def get_background_genes(
    adata: Any, column: str = "", group: str = ""
) -> list[str] | None:
    """Genes that were actually testable, for use as the ORA universe.

    Prefers the full ranked gene list from ``rank_genes_groups`` (literally the
    genes the DE test considered). Falls back to the measured var_names, which
    for a raw-backed AnnData is what scanpy ranks by default (``use_raw=True``).
    Returns None when neither is recoverable, in which case callers should leave
    gseapy's default alone rather than pass a wrong universe.
    """
    uns = getattr(adata, "uns", {})
    markers_key = resolve_markers_key(uns, column)

    if markers_key is not None:
        try:
            names = uns[markers_key]["names"]
            fields = getattr(getattr(names, "dtype", None), "names", None)
            if fields:
                field = group if group in fields else fields[0]
                genes = [str(g) for g in names[field]]
                if genes:
                    return genes
        except (KeyError, TypeError, IndexError) as exc:
            logger.debug("Could not read background from %s: %s", markers_key, exc)

    # Fallback: every measured gene.
    try:
        raw = getattr(adata, "raw", None)
        var_names = raw.var_names if raw is not None else adata.var_names
        genes = [str(g) for g in var_names]
        if genes:
            logger.debug("Background fell back to var_names (%d genes)", len(genes))
            return genes
    except AttributeError as exc:
        logger.debug("Could not read background from var_names: %s", exc)

    return None


def _fingerprint(collections: list[str] | None) -> list[str]:
    """Normalise the gene-set selection so it compares stably across requests."""
    return sorted(str(c) for c in (collections or []))


def write_meta(
    adata: Any,
    column: str,
    group: str,
    *,
    background_n: int | None,
    n_genes: int,
    collections: list[str] | None = None,
    exclude_ribo_mito: bool = True,
) -> None:
    """Record how an enrichment payload was computed, so stale caches are detectable."""
    if not column:
        return
    adata.uns[meta_key(column, group)] = json.dumps(
        {
            "ora_version": ORA_VERSION,
            "background_n": background_n,
            "n_genes": n_genes,
            "collections": _fingerprint(collections),
            "exclude_ribo_mito": bool(exclude_ribo_mito),
        }
    )


def cache_is_current(
    adata: Any,
    column: str,
    group: str,
    *,
    n_genes: int | None = None,
    collections: list[str] | None = None,
    exclude_ribo_mito: bool | None = None,
) -> bool:
    """True when a cached enrichment payload can be served for this request.

    Payloads written before the background fix have no sidecar and were computed
    against gseapy's implicit universe, so they are never current.

    ``pipeline.py`` and the two enrichment endpoints all write the same
    ``enrichment__{column}__{group}`` key but may use different collections or a
    different n_genes, so whichever ran last used to win and its result would be
    served for every later request regardless of parameters. Passing n_genes /
    collections here compares them against the cached run and forces a recompute
    on mismatch. Omit both to check only that the payload postdates the fix (used
    for the UI's "already computed" badge, which has no request to compare to).
    """
    if not column:
        return False
    raw = getattr(adata, "uns", {}).get(meta_key(column, group))
    if raw is None:
        return False
    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
        if int(meta.get("ora_version", 0)) < ORA_VERSION:
            return False
        if n_genes is not None and int(meta.get("n_genes", -1)) != int(n_genes):
            return False
        if collections is not None and list(meta.get("collections", [])) != _fingerprint(
            collections
        ):
            return False
        if exclude_ribo_mito is not None and bool(
            meta.get("exclude_ribo_mito", False)
        ) != bool(exclude_ribo_mito):
            return False
        return True
    except (ValueError, TypeError, AttributeError):
        return False
