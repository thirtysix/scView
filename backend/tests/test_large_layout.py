"""The large-dataset file layout: sidecar names, the uns marker, the counts identity check."""

from __future__ import annotations

import h5py
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata.io import write_elem

from scview.core import large_layout


def _counts_file(path, n=6, m=4):
    with h5py.File(path, "w") as f:
        write_elem(f, large_layout.COUNTS_KEY, sp.random(n, m, density=0.5, format="csr", random_state=0))
    return pd.Index([f"c{i}" for i in range(n)]), pd.Index([f"g{j}" for j in range(m)])


def test_sidecars_sit_beside_the_h5ad_and_resolve_from_uns(tmp_path):
    h5ad = tmp_path / "derived" / "pbmc.h5ad"
    counts, genes = large_layout.sidecar_paths(h5ad)
    assert (counts.name, genes.name) == ("pbmc.counts.h5", "pbmc.genes.h5")
    uns = {}
    assert not large_layout.is_large(uns)
    large_layout.record(uns, h5ad)
    assert large_layout.is_large(uns)
    moved = tmp_path / "elsewhere" / "pbmc.h5ad"  # a copied folder keeps working
    assert large_layout.counts_path(moved, uns) == moved.with_name("pbmc.counts.h5")
    assert large_layout.genes_path(moved, uns) == moved.with_name("pbmc.genes.h5")


def test_a_plain_dataset_has_no_sidecars():
    with pytest.raises(large_layout.LargeLayoutError):
        large_layout.counts_path("x.h5ad", {})


def test_counts_identity_catches_other_cells_and_genes(tmp_path):
    path = tmp_path / "d.counts.h5"
    assert "missing" in large_layout.counts_mismatch(path, [], [])
    cells, genes = _counts_file(path)
    large_layout.stamp_counts(path, cells, genes)
    assert large_layout.counts_mismatch(path, cells, genes) is None
    assert "other cells" in large_layout.counts_mismatch(path, cells[::-1], genes)
    assert "other genes" in large_layout.counts_mismatch(path, cells, genes[:3])


def test_remove_sidecars_deletes_both(tmp_path):
    h5ad = tmp_path / "d.h5ad"
    counts, genes = large_layout.sidecar_paths(h5ad)
    _counts_file(counts)
    genes.write_bytes(b"")
    large_layout.remove_sidecars(h5ad)
    assert not counts.exists() and not genes.exists()
    large_layout.remove_sidecars(h5ad)  # nothing there: no error
