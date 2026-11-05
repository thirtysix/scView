"""Gene symbols as the gene index of a CELLxGENE file, at import.

A CELLxGENE Discover .h5ad indexes its genes by Ensembl id and keeps the symbols in
``var["feature_name"]``. Read as it is, every feature that works by gene name finds
nothing: gene search ("XIST"), marker databases, gene sets, CellTypist, % mito (``MT-``),
the sex and cell-cycle checks. The Discover import (``core/public_import.py``) has always
made the symbols the index; this does the same for an .h5ad uploaded directly or through
the import wizard, with the same rule, so one file behaves the same whichever way it came
in: the index becomes ``feature_name`` (an empty symbol keeps its id, repeats get ``-1``,
``-2`` ... as ``anndata.utils.make_index_unique`` does), and the ids go to
``var["gene_ids"]``. A table that already has ``gene_ids`` is left alone (the rule the
replay script and the Discover import share), and so is one without ``feature_name``.
X and .raw are handled each from its own table.

Where: on the dataset's working copy (``ingested/{id}/`` for an upload, whose original
stays byte-identical and read-only; the loaded AnnData in the wizard). The rename is
recorded as ``gene_symbols`` in the ingest step's params, and the replay script repeats
it (``replay_script._ingest``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import pandas as pd

logger = logging.getLogger(__name__)

SYMBOL_COLUMN = "feature_name"
ID_COLUMN = "gene_ids"
# Kept on a loaded AnnData until the ingest provenance records it (session._record_ingest_provenance).
PENDING_KEY = "scview_gene_symbols_pending"


def applies(var: pd.DataFrame) -> bool:
    """Whether ``var`` gets symbols as its index (``feature_name`` and no ``gene_ids``)."""
    return SYMBOL_COLUMN in var.columns and ID_COLUMN not in var.columns


def to_symbols(var: pd.DataFrame) -> tuple[pd.DataFrame, dict] | None:
    """(``var`` with symbols as its index and the ids in ``gene_ids``, a summary), or None
    when the rule does not apply."""
    from scview.core.public_import import unique_symbols

    if not applies(var):
        return None
    ids = [str(v) for v in var.index]
    raw_symbols = [str(v) for v in var[SYMBOL_COLUMN].astype(str)]
    names = unique_symbols(raw_symbols, ids)
    out = var.copy()
    out[ID_COLUMN] = ids
    out.index = pd.Index(names)
    kept_id = sum(1 for s in raw_symbols if not s or s.lower() == "nan")
    made_unique = sum(1 for s, n in zip(raw_symbols, names) if s and s.lower() != "nan" and n != s)
    return out, {"from": f"var['{SYMBOL_COLUMN}']", "ids_in": f"var['{ID_COLUMN}']",
                 "n_genes": len(ids), "renamed": sum(a != b for a, b in zip(ids, names)),
                 "kept_id": kept_id, "made_unique": made_unique}


def apply_in_memory(adata) -> dict | None:
    """Symbols as the index of ``adata.var`` and of ``adata.raw.var``, each where the rule
    applies; the summary (also kept in ``uns[PENDING_KEY]``), or None when neither did."""
    import anndata as ad

    summary: dict = {}
    got = to_symbols(adata.var)
    if got is not None:
        adata.var = got[0]
        summary["var"] = got[1]
    if adata.raw is not None:
        got = to_symbols(adata.raw.var)
        if got is not None:
            raw = adata.raw
            adata.raw = ad.AnnData(X=raw.X, obs=pd.DataFrame(index=adata.obs_names), var=got[0],
                                   varm=dict(raw.varm) if raw.varm is not None else None)
            summary["raw_var"] = got[1]
    if summary:
        adata.uns[PENDING_KEY] = summary
        return summary
    return None


def rewrite_group(group: h5py.Group) -> dict | None:
    """One ``var`` table of an open .h5ad (``/var`` or ``/raw/var``) to symbols in place;
    its summary, or None when the rule does not apply."""
    from anndata.io import read_elem, write_elem

    var = read_elem(group)
    got = to_symbols(var)
    if got is None:
        return None
    parent, name = group.parent, group.name.rsplit("/", 1)[-1]
    del parent[name]
    write_elem(parent, name, got[0])
    return got[1]


def file_applies(path: Path) -> bool:
    """Whether the .h5ad at ``path`` has a var or raw.var table the rule applies to (reads
    the var column names only)."""
    def cols(g) -> set[str]:
        order = g.attrs.get("column-order", [])
        return {c.decode() if isinstance(c, bytes) else str(c) for c in order}

    with h5py.File(path, "r") as f:
        tables = [f.get("var")]
        if isinstance(f.get("raw"), h5py.Group):
            tables.append(f["raw"].get("var"))
        for g in tables:
            if isinstance(g, h5py.Group):
                c = cols(g)
                if SYMBOL_COLUMN in c and ID_COLUMN not in c:
                    return True
    return False


def rewrite_file(path: Path) -> dict | None:
    """``var`` and ``raw/var`` of the .h5ad at ``path`` (writable) to symbols in place."""
    summary: dict = {}
    with h5py.File(path, "r+") as f:
        if isinstance(f.get("var"), h5py.Group):
            got = rewrite_group(f["var"])
            if got:
                summary["var"] = got
        if isinstance(f.get("raw"), h5py.Group) and isinstance(f["raw"].get("var"), h5py.Group):
            got = rewrite_group(f["raw"]["var"])
            if got:
                summary["raw_var"] = got
    return summary or None


def record_in_file(path: Path, summary: dict, original_filename: str) -> None:
    """An ingest step with ``gene_symbols`` in the provenance of the .h5ad at ``path``
    (writable), after the feature split's step when that ran; a source block first when
    the file has none (a plain upload)."""
    import anndata as ad
    from anndata.io import read_elem, write_elem

    from scview.core import provenance

    with h5py.File(path, "r+") as f:
        uns = f["uns"] if isinstance(f.get("uns"), h5py.Group) else f.create_group("uns")
        holder = ad.AnnData()
        if provenance.UNS_KEY in uns:
            holder.uns[provenance.UNS_KEY] = read_elem(uns[provenance.UNS_KEY])
        # A no-op when the feature split already wrote the source block.
        provenance.init_source(holder, origin="uploaded", original_filename=original_filename,
                               fmt="h5ad")
        provenance.record_step(holder, step="ingest", tool="scview.gene_symbols",
                               params={"formats": ["h5ad"], "gene_symbols": summary})
        if provenance.UNS_KEY in uns:
            del uns[provenance.UNS_KEY]
        write_elem(uns, provenance.UNS_KEY, holder.uns[provenance.UNS_KEY])
