"""Spatially variable genes for the co-pilot: the fact (core/spatial_autocorr.py
grounding_lines), its source in the chat context and its citation family."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scview.core import assistant, claims
from scview.core import spatial_autocorr as sa

_spec = importlib.util.spec_from_file_location(
    "spatial_synthetic", Path(__file__).parent / "fixtures" / "spatial_autocorr" / "synthetic.py")
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)


class _Adaptor:
    """Weak-referenceable stand-in holding an AnnData."""

    def __init__(self, adata):
        self.adata = adata
        self.h5ad_path = None


def test_grounding_lists_the_top_genes_and_resolves_its_tag():
    ada = _Adaptor(synthetic.visium())
    assert sa.grounding_lines(ada) == ([], None)
    r = sa.run(ada.adata)
    sa.remember(ada, r)
    lines, detail = sa.grounding_lines(ada)
    text = "\n".join(lines)
    assert sa.TAG in text and r.table["gene"].iloc[0] in text and "Moran's I" in detail
    lines2: list[str] = []
    sources: list = []
    assistant._add_spatial_genes(ada, lines2, sources)
    assert [s.ref for s in sources] == [sa.TAG]
    assert claims.resolve_tag("spatial_genes", [sa.TAG, "result:markers:B"]) == [sa.TAG]
    assert claims.resolve_tag("result:spatial", [sa.TAG]) == [sa.TAG]
