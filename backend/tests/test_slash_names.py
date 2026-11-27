"""Names with "/" through the real write and read paths (core/h5_names.py does the work;
nothing here imports it, so these tests show the behaviour itself).

The reported failure: labels "T" and "T/NK" in one column, the pipeline's marker step,
then the save: "Incompatible object (Dataset) already exists" after every step had run.
"""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import pipeline


def _base(n: int = 24) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, (n, 6)).astype("float32")))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{i}" for i in range(6)]
    return a


SETS = [["A", "T/NK"], ["T", "T/NK"], ["T/NK", "T"], ["T/NK", "T/NK/x"], ["T", "T/"], ["/T", "T"], ["T//NK", "T"],
        [".", "..", "x"], ["", "x"], ["a/.", "a"], ["T%2fNK", "T/NK"], ["T%2FNK", "T/NK"], ["%25", "%", "/"]]


def _fill(a: ad.AnnData, where: str, names: list[str]) -> None:
    n = a.n_obs
    if where == "uns_dict":
        a.uns["d"] = {k: np.arange(3) + i for i, k in enumerate(names)}
    elif where == "uns_nested":
        a.uns["colors"] = {"col/umn": {k: f"#00000{i}" for i, k in enumerate(names)}}
    elif where == "uns_key":
        for i, k in enumerate(names):
            a.uns[k] = np.arange(3) + i
    elif where == "uns_frame":
        a.uns["df"] = pd.DataFrame({k: np.arange(4.0) + i for i, k in enumerate(names)}, index=list("wxyz"))
    elif where == "obs_columns":
        for i, k in enumerate(names):
            a.obs[k] = np.arange(n) + i
    elif where == "var_columns":
        for i, k in enumerate(names):
            a.var[k] = np.arange(a.n_vars) + i
    elif where == "obsm_frame":
        a.obsm["m"] = pd.DataFrame({k: np.arange(n) + i for i, k in enumerate(names)}, index=a.obs_names)
    elif where == "obsm_keys":
        for i, k in enumerate(names):
            a.obsm[k] = np.full((n, 2), float(i))
    elif where == "layers":
        for i, k in enumerate(names):
            a.layers[k] = a.X * (i + 1)
    elif where == "obsp_keys":
        for i, k in enumerate(names):
            a.obsp[k] = sp.identity(n, format="csr") * (i + 1)
    elif where == "categories":
        a.obs["c"] = pd.Categorical(np.array(names)[np.arange(n) % len(names)], categories=names)


def _same(x, y) -> None:
    if isinstance(x, pd.DataFrame):
        assert sorted(x.columns) == sorted(y.columns)
        for c in x.columns:
            assert np.array_equal(np.asarray(x[c]), np.asarray(y[c])), c
    elif isinstance(x, dict) or hasattr(x, "keys"):
        assert sorted(x.keys()) == sorted(y.keys())
        for k in x.keys():
            _same(x[k], y[k])
    elif sp.issparse(x):
        assert (x != y).nnz == 0
    else:
        assert np.array_equal(np.asarray(x), np.asarray(y))


WHERE = ["uns_dict", "uns_nested", "uns_key", "uns_frame", "obs_columns", "var_columns", "obsm_frame", "obsm_keys",
         "layers", "obsp_keys", "categories"]


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("names", SETS, ids=lambda s: "+".join(repr(n) for n in s))
def test_odd_names_write_and_read_back(tmp_path, where, names):
    a = _base()
    _fill(a, where, names)
    path = tmp_path / "x.h5ad"
    a.write_h5ad(path)
    for b in (ad.read_h5ad(path), ad.read_h5ad(path, backed="r")):
        _same(dict(a.uns), dict(b.uns))
        _same(a.obs, b.obs)
        _same(a.var, b.var)
        for slot in ("obsm", "layers", "obsp"):
            _same(dict(getattr(a, slot)), dict(getattr(b, slot)))
        if where == "categories":
            assert list(b.obs["c"].cat.categories) == names
        if b.isbacked:
            b.file.close()
    with h5py.File(path, "r") as f:  # one key per name: nothing nested
        depth: list[str] = []
        f.visit(lambda p: depth.append(p) if p.endswith(("/NK", "/NK/x")) else None)
        assert not depth


def test_outside_reader_sees_escaped_names(tmp_path):
    """What plain anndata (no scView) reads from a file scView wrote: the same data under
    "T%2fNK"."""
    a = _base()
    a.obs["CD4/CD8"] = np.arange(a.n_obs)
    a.uns["pts"] = pd.DataFrame({"T": [0.5, 0.1], "T/NK": [0.2, 0.9]}, index=["G0", "G1"])
    a.uns["by_label"] = {"T/NK": "x"}
    path = tmp_path / "x.h5ad"
    a.write_h5ad(path)
    code = ("import sys, anndata as ad\n"
            "assert 'scview' not in sys.modules\n"
            f"b = ad.read_h5ad({str(path)!r})\n"
            "print(sorted(b.obs.columns), sorted(b.uns['pts'].columns), sorted(b.uns['by_label']), "
            "float(b.uns['pts']['T%2fNK'].iloc[1]))\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "['CD4%2fCD8'] ['T', 'T%2fNK'] ['T%2fNK'] 0.9", out.stderr[-2000:]


def test_zarr_write_takes_slash_names(tmp_path):
    """zarr refused "/" in a key outright (the Vitessce export writes zarr)."""
    a = _base()
    a.uns["pts"] = pd.DataFrame({"T": [0.5], "T/NK": [0.2]}, index=["G0"])
    a.obs["CD4/CD8"] = 1.0
    a.write_zarr(tmp_path / "x.zarr")
    b = ad.read_zarr(tmp_path / "x.zarr")
    assert list(b.uns["pts"].columns) == ["T", "T/NK"] and "CD4/CD8" in b.obs.columns


# --- the reported failure: pipeline markers step, save, reopen --------------------------


def _labelled(labels: list[str], n: int = 300, g: int = 50) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (n, g)).astype("float32")
    which = rng.integers(0, len(labels), n)
    for i in range(len(labels)):
        X[which == i, i * 5:(i + 1) * 5] += 4
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = pd.Categorical(np.array(labels)[which])
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


@pytest.mark.parametrize("labels", [["T", "T/NK", "B"], ["A", "T/NK", "B"], ["T/NK", "T/NK/x", "T"]],
                         ids=["prefix", "plain", "deeper"])
def test_markers_step_saves_and_reopens_with_slash_labels(tmp_path, labels):
    a = _labelled(labels)
    out = tmp_path / "derived.h5ad"
    params = pipeline.PipelineParams(marker_columns=["cell_type"])
    a, result = pipeline.run_pipeline(a, ["marker_genes"], params, output_path=str(out))
    assert result.status == "complete", (result.errors, result.notes)
    assert out.exists() and not (tmp_path / "derived.h5ad.partial").exists()
    key = next(k for k in a.uns if k.startswith("rank_genes_groups") and "cell_type" in k)
    b = ad.read_h5ad(out)
    assert list(b.obs["cell_type"].cat.categories) == sorted(labels)
    got = b.uns[key]
    assert set(got["names"].dtype.names) == set(labels)
    for table in ("pts", "pts_rest"):
        assert set(got[table].columns) == set(labels)  # the label as typed, not "T" holding "NK"
        for label in labels:
            assert np.allclose(got[table][label].to_numpy(), a.uns[key][table][label].to_numpy())
    top = pd.DataFrame(got["names"]).iloc[0]
    for i, label in enumerate(labels):  # each label kept its own genes
        assert top[label] in {f"G{j}" for j in range(i * 5, (i + 1) * 5)}


def test_save_failure_says_what_happened(tmp_path):
    """A save that fails after every step ran says so in words, and leaves the old file."""
    a = _labelled(["A", "B"])
    out = tmp_path / "derived.h5ad"
    out.write_bytes(b"previous")

    def write(adata, path):
        raise TypeError("Incompatible object (Dataset) already exists")

    with pytest.raises(Exception) as err:  # noqa: B017, PT011 - the text is the test
        pipeline.run_pipeline(a, ["marker_genes"], pipeline.PipelineParams(marker_columns=["cell_type"]),
                              output_path=str(out), write=write)
    text = str(err.value)
    assert "could not be saved" in text and "as it was before the run" in text
    assert "Incompatible object" in text
    assert out.read_bytes() == b"previous" and not (tmp_path / "derived.h5ad.partial").exists()


# --- the saves that write one element at a time ------------------------------------------

ODD_KEYS = ["by/slash", "by", "by/slash/deeper", "/lead", "trail/", ".", "", "rank_genes_groups__cell%2Ftype"]


def _edited(a: ad.AnnData) -> None:
    """What an edit leaves in memory: uns keys and an obsm name with "/" in them, a table
    with one column per label, a dict keyed by label."""
    for i, key in enumerate(ODD_KEYS):
        a.uns[key] = np.arange(3) + i
    a.uns["pts"] = pd.DataFrame({"T": [0.5, 0.1], "T/NK": [0.2, 0.9]}, index=["G0", "G1"])
    a.uns["mapping"] = {"T": "T cells", "T/NK": "T or NK cells"}
    a.obsm["X_umap/alt"] = np.ones((a.n_obs, 2))
    a.obs["label"] = pd.Categorical(np.array(["T", "T/NK"])[np.arange(a.n_obs) % 2])
    a.obs["CD4/CD8"] = np.arange(a.n_obs)
    a.obs["CD4"] = 1


def _check_edited(b: ad.AnnData) -> None:
    for i, key in enumerate(ODD_KEYS):
        assert b.uns[key].tolist() == (np.arange(3) + i).tolist(), key
    assert sorted(b.uns["pts"].columns) == ["T", "T/NK"] and b.uns["pts"]["T/NK"].tolist() == [0.2, 0.9]
    assert dict(b.uns["mapping"]) == {"T": "T cells", "T/NK": "T or NK cells"}
    assert b.obsm["X_umap/alt"].shape == (b.n_obs, 2)
    assert {"CD4/CD8", "CD4", "label"} <= set(b.obs.columns)
    assert b.obs["CD4/CD8"].tolist() == list(range(b.n_obs))
    assert set(b.uns) == {*ODD_KEYS, "pts", "mapping", "kept"}


def test_edit_save_into_its_own_file(tmp_path):
    """``write_derived``: the copy-and-rewrite save of every label edit and score."""
    from scview.core.obs_writes import uns_snapshot, write_derived

    path = str(tmp_path / "d.h5ad")
    a = _base()
    a.uns["kept"] = "x"
    a.uns["by/slash"] = np.array([7])  # in the file already, replaced by the edit
    a.write_h5ad(path)
    a = ad.read_h5ad(path)
    before = uns_snapshot(a)
    _edited(a)
    write_derived(a, path, path, before, obsm_keys=("X_umap/alt",))
    _check_edited(ad.read_h5ad(path))
    a = ad.read_h5ad(path)  # a second edit removes one and keeps the rest
    before = uns_snapshot(a)
    del a.uns["by/slash"]
    a.uns["by"] = np.array([5])
    write_derived(a, path, path, before)
    b = ad.read_h5ad(path)
    assert "by/slash" not in b.uns and b.uns["by"].tolist() == [5]
    assert b.uns["by/slash/deeper"].tolist() == [2, 3, 4]


def test_large_dataset_save_in_place(tmp_path):
    """``large_mode.write_in_place``: obs, uns keys and obsm written into a backed file."""
    import threading

    from scview.core import large_mode

    path = tmp_path / "large.h5ad"
    a = _base()
    a.uns["kept"] = "x"
    a.uns["by/slash"] = np.array([7])
    a.write_h5ad(path)
    a = ad.read_h5ad(path, backed="r")
    _edited(a)
    large_mode.write_in_place(a, path, threading.RLock(), obs=True, uns_keys=[*ODD_KEYS, "pts", "mapping"],
                              obsm_keys=["X_umap/alt"])
    a.file.close()
    _check_edited(ad.read_h5ad(path))
    a = ad.read_h5ad(path, backed="r")
    del a.uns["by/slash"]
    large_mode.write_in_place(a, path, threading.RLock(), obs=False, uns_removed=["by/slash"])
    a.file.close()
    b = ad.read_h5ad(path)
    assert "by/slash" not in b.uns and b.uns["by"].tolist() == [1, 2, 3]
    assert b.uns["by/slash/deeper"].tolist() == [2, 3, 4]


def test_replay_script_output_saves_with_slash_labels(tmp_path):
    """The replay script's own ``write_h5ad`` runs without scView: it escapes the names itself."""
    from scview.core import replay_script

    code = replay_script._HELPERS["h5_names"]
    a = _base()
    src, out = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    script = (code + "\nimport sys, anndata as ad, pandas as pd\nassert 'scview' not in sys.modules\n"
              f"adata = ad.read_h5ad({str(src)!r})\n"
              "adata.uns['pts'] = pd.DataFrame({'T': [0.5], 'T/NK': [0.2]}, index=['G0'])\n"
              "adata.obs['CD4'] = 1\nadata.obs['CD4/CD8'] = 2\n"
              "print(sorted(h5_safe_names(adata)))\n"
              f"adata.write_h5ad({str(out)!r})\n")
    got = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr[-2000:]
    assert got.stdout.strip() == "['CD4/CD8', 'T/NK']"
    b = ad.read_h5ad(out)
    assert sorted(b.uns["pts"].columns) == ["T", "T/NK"] and "CD4/CD8" in b.obs.columns
