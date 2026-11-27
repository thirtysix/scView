"""Names with "/" in an h5ad (core/h5_names.py).

A cell-type label "T/NK" beside a label "T" made every save of the dataset fail: the
marker ``pts`` table has one COLUMN per label, HDF5 reads "/" as a path, and "T" was
already a dataset ("Incompatible object (Dataset) already exists"). A "/" key in a dict
(uns, layers, obsm) was written and read back as a nested dict.
"""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import h5_names

ODD = ["T/NK", "/T", "T/", "T//NK", "/", "//", ".", "..", "", "a/.", "a/..", "T/NK/x", "100%", "%2F", "%2f", "T%2fNK",
       "%25", "%252f", "%2525", "%2e", "%2E", "%00", "%", "%/", "/%2f", "a%2", "é/ü", " / "]


@pytest.mark.parametrize("name", ODD + ["T", "plain name", "pct_counts_mt", "a.b", "a\\b"])
def test_escape_round_trip_and_one_key(name):
    key = h5_names.escape(name)
    assert h5_names.unescape(key) == name
    assert "/" not in key and key not in ("", ".")
    if not h5_names.needs_escape(name):
        assert key == name


def test_escape_is_one_to_one():
    names = ODD + ["T", "NK", "T%252fNK", "%2f%2f"]
    assert len({h5_names.escape(n) for n in names}) == len(set(names))
    assert h5_names.escape("T/NK") == "T%2fNK"  # what an outside reader sees
    assert h5_names.escape("100% T/NK") == "100% T%2fNK"  # a plain "%" stays


def test_names_of_earlier_versions_pass_through(tmp_path):
    """core/uns_keys.py writes "/" as "%2F" (upper case) inside marker and enrichment keys,
    and every file since holds them: they are names, not escapes, on disk and in memory."""
    from scview.core import uns_keys

    key = uns_keys.markers_key("cell/type")
    assert key == "rank_genes_groups__cell%2Ftype"
    assert h5_names.escape(key) == key and h5_names.unescape(key) == key
    a = _base()
    a.uns[key] = {"params": {"groupby": "cell/type"}}
    a.uns[uns_keys.enrichment_key("cell/type", "T/NK")] = "{}"
    a.write_h5ad(tmp_path / "x.h5ad")
    with h5py.File(tmp_path / "x.h5ad", "r") as f:
        assert sorted(f["uns"].keys()) == ["enrichment__cell%2Ftype__T%2FNK", key]  # as 0.85 wrote them
    b = ad.read_h5ad(tmp_path / "x.h5ad")
    assert sorted(b.uns) == ["enrichment__cell%2Ftype__T%2FNK", key]
    assert uns_keys.markers_column(key) == "cell/type"


def test_installed_and_wraps_the_registry():
    """anndata's registry is private: a version that moves it must fail here, not at a save."""
    from anndata._io.specs.registry import _REGISTRY

    assert h5_names.installed()
    writers = [k for k, f in _REGISTRY.write.items() if getattr(f, "_scview_names", False)]
    kinds = {k[1] for k in writers}
    assert dict in kinds and pd.DataFrame in kinds
    assert {type(k[0]).__name__ and k[0] for k in writers} >= {h5py.Group}
    readers = {k[1].encoding_type for k, f in _REGISTRY.read.items() if getattr(f, "_scview_names", False)}
    assert readers == {"dict", "dataframe"}


def test_installs_when_anndata_is_imported_later():
    """A process that imports scview first (the server, a child) is covered the moment it
    imports anndata, and one that never does is not made to import it."""
    code = ("import sys, scview\n"
            "assert 'anndata' not in sys.modules, 'scview imported anndata'\n"
            "import scview.core.h5_names as h\n"
            "assert not h.installed()\n"
            "import anndata\n"
            "assert h.installed()\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "ok", out.stderr[-2000:]


def _base(n: int = 24) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, (n, 6)).astype("float32")))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{i}" for i in range(6)]
    return a


def test_file_written_before_the_escape_still_opens(tmp_path):
    """An older scView (or anyone) wrote "A" + "T/NK" columns nested: read as before, and
    the helpers find the nested entry by its name."""
    path = tmp_path / "old.h5ad"
    a = _base()
    a.write_h5ad(path)
    with h5py.File(path, "a") as f:
        obs = f["obs"]
        obs.create_dataset("T/NK", data=np.arange(a.n_obs))  # h5py nests: group "T", dataset "NK"
        obs.attrs["column-order"] = ["T/NK"]
        g = f["uns"].create_group("pts")
        g.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index",
                        "column-order": ["A", "T/NK"]})
        g.create_dataset("_index", data=np.array(["G0", "G1"], dtype=h5py.string_dtype()))
        g.create_dataset("A", data=[0.1, 0.2])
        g.create_dataset("T/NK", data=[0.3, 0.4])
        assert isinstance(obs["T"], h5py.Group)
    b = ad.read_h5ad(path)
    assert list(b.obs.columns) == ["T/NK"] and list(b.uns["pts"].columns) == ["A", "T/NK"]
    assert b.uns["pts"]["T/NK"].tolist() == [0.3, 0.4]
    with h5py.File(path, "a") as f:
        assert h5_names.has(f["obs"], "T/NK") and not h5_names.has(f["obs"], "T/other")
        assert h5_names.child(f["obs"], "T/NK")[:3].tolist() == [0, 1, 2]
        h5_names.write_elem(f["uns"], "pts", b.uns["pts"])  # saved again: escaped, not nested
        assert sorted(f["uns/pts"].keys()) == ["A", "T%2fNK", "_index"]
        assert h5_names.drop(f["obs"], "T/NK") and not h5_names.has(f["obs"], "T/NK")
    b.write_h5ad(path)
    assert list(ad.read_h5ad(path).obs.columns) == ["T/NK"]


def test_group_helpers(tmp_path):
    with h5py.File(tmp_path / "g.h5", "w") as f:
        g = f.create_group("uns")
        for i, name in enumerate(["T", "T/NK", "", ".", "markers__CD4/CD8", "%2f"]):
            h5_names.write_elem(g, name, np.arange(2) + i)
        assert sorted(h5_names.names(g)) == sorted(["T", "T/NK", "", ".", "markers__CD4/CD8", "%2f"])
        assert h5_names.child(g, "T/NK")[:].tolist() == [1, 2]
        assert h5_names.path("uns", "markers__CD4/CD8") == "uns/markers__CD4%2fCD8"
        assert f[h5_names.path("uns", "markers__CD4/CD8")][:].tolist() == [4, 5]
        assert h5_names.has(g, "") and h5_names.has(g, ".") and not h5_names.has(g, "NK")
        h5_names.write_elem(g, "T/NK", np.array([9]))  # replaces
        assert h5_names.child(g, "T/NK")[:].tolist() == [9]
        assert h5_names.drop(g, "T/NK") and not h5_names.drop(g, "T/NK")
        assert h5_names.child(g, "T")[:].tolist() == [0, 1]
        with pytest.raises(KeyError):
            h5_names.child(g, "T/NK")
