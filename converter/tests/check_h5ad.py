"""check_h5ad.py <out_dir>: the converted fixtures against expected.json.

Runs in the converter image (python3 + anndata; no pytest there): every check
is a plain assert, collected per fixture, and the exit code says whether all
passed. The expected values come from make_fixtures.R, computed from the R
objects, so a check never compares the converter with itself.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def _sum(m) -> float:
    return float(np.asarray(m.sum(dtype=np.float64)).sum())


def _close(a: float, b: float, what: str, rtol: float = 1e-5) -> None:
    assert np.isclose(a, b, rtol=rtol, atol=1e-6), f"{what}: {a} != {b}"


def _encoding(path: Path, key: str) -> str:
    with h5py.File(path, "r") as f:
        return f[key].attrs["encoding-type"]


def check_fixture(name: str, exp: dict, res: dict, out: Path) -> None:
    if "error" in exp:
        assert not res["success"], f"{name}: converted, but should have been refused"
        for part in exp["error"]:
            assert part in res["error"], f"{name}: error {res['error']!r} lacks {part!r}"
        assert res["progress"]["stage"] == "failed", res["progress"]
        return

    assert res["success"], f"{name}: {res.get('error')}"
    assert res["progress"]["stage"] == "done", res["progress"]
    path = out / f"{name}.h5ad"
    assert path.exists() and not (out / f"{name}.h5ad.partial").exists()
    a = ad.read_h5ad(path)

    assert a.shape == (exp["n_obs"], exp["n_vars"]), f"shape {a.shape}"
    _close(_sum(a.X), exp["x_sum"], "X sum")
    assert _encoding(path, "X") == exp["x_format"], _encoding(path, "X")
    assert set(a.layers.keys()) == set(exp["layers"]), sorted(a.layers.keys())
    for layer, total in exp["layers"].items():
        _close(_sum(a.layers[layer]), total, f"layer {layer} sum")
    for layer, fmt in exp.get("layer_formats", {}).items():
        assert _encoding(path, f"layers/{layer}") == fmt, (layer, _encoding(path, f"layers/{layer}"))
    assert a.obs_names.is_unique and a.var_names.is_unique

    if "obs_names_head" in exp:
        assert list(a.obs_names[: len(exp["obs_names_head"])]) == exp["obs_names_head"], list(a.obs_names[:4])
    if "var_names_head" in exp:
        assert list(a.var_names[: len(exp["var_names_head"])]) == exp["var_names_head"], list(a.var_names[:6])
    if "gene_ids_head" in exp:
        assert list(a.var["gene_ids"][:3]) == exp["gene_ids_head"]

    for key, shape in exp.get("obsm", {}).items():
        assert key in a.obsm, f"obsm {key} missing: {list(a.obsm.keys())}"
        assert tuple(np.shape(a.obsm[key])) == tuple(shape), (key, np.shape(a.obsm[key]))
    for key, total in exp.get("obsm_sums", {}).items():
        _close(float(np.asarray(a.obsm[key], dtype=np.float64).sum()), total, f"obsm {key} sum")

    for key, slot in (("protein", "protein_counts"), ("hto", "hto_counts")):
        if key in exp:
            m = a.obsm[slot]
            assert isinstance(m, pd.DataFrame), type(m)
            assert list(m.columns) == exp[key]["columns"], list(m.columns)
            assert list(m.index) == list(a.obs_names)
            _close(float(m.to_numpy().sum()), exp[key]["sum"], f"{slot} sum")

    for col, dtype in exp.get("obs_dtypes", {}).items():
        assert col in a.obs, f"obs {col} missing: {list(a.obs.columns)}"
        assert str(a.obs[col].dtype) == dtype, (col, str(a.obs[col].dtype))
    for col, cats in exp.get("categories", {}).items():
        assert list(a.obs[col].cat.categories) == cats, (col, list(a.obs[col].cat.categories))
    for col in exp.get("ordered", []):
        assert a.obs[col].cat.ordered, col
    for col, n in exp.get("n_missing", {}).items():
        assert int(a.obs[col].isna().sum()) == n, (col, int(a.obs[col].isna().sum()))
    for col in exp.get("obs_absent", []):
        assert col not in a.obs, col
    for col in exp.get("var_columns", []):
        assert col in a.var, f"var {col} missing: {list(a.var.columns)}"

    if "pca_variance_ratio" in exp:
        np.testing.assert_allclose(a.uns["pca"]["variance_ratio"], exp["pca_variance_ratio"])
        np.testing.assert_allclose(a.uns["pca"]["variance"], exp["pca_variance"])
        pcs = np.asarray(a.varm["PCs"])
        assert pcs.shape == (a.n_vars, len(exp["pca_variance"]))
        assert list(np.flatnonzero(np.any(pcs != 0, axis=1))) == exp["pcs_rows"]
        _close(float(pcs.sum()), exp["pcs_sum"], "PCs sum")
    for key in exp.get("uns_metadata", []):
        assert key in a.uns["r_metadata"], (key, list(a.uns["r_metadata"]))

    report = a.uns.get("scview_conversion")
    if exp.get("source_class") == "Seurat":
        assert report is None  # the Seurat output is the original one, untouched
        assert res["report"]["source_class"] == "Seurat"
        return
    assert report is not None, "uns scview_conversion missing"
    assert report["x"] == exp["x"], report["x"]
    assert list(report["layers"]) == [k for k in a.layers.keys()] or set(report["layers"]) == set(a.layers.keys())
    assert int(report["n_cells"]) == exp["n_obs"]
    assert res["report"]["x"] == exp["x"]
    kept = " | ".join(map(str, report["kept"]))
    skipped = " | ".join(map(str, report["skipped"]))
    for part in exp.get("report_kept", []):
        assert part in kept, f"kept lacks {part!r}: {kept}"
    for part in exp.get("report_skipped", []):
        assert part in skipped, f"skipped lacks {part!r}: {skipped}"
    if "object_name" in exp:
        assert report["object_name"] == exp["object_name"]


def main(out: Path) -> int:
    expected = json.loads((out / "expected.json").read_text())
    results = json.loads((out / "results.json").read_text())
    failed = 0
    for name, exp in expected.items():
        try:
            assert name in results, f"{name}: not converted"
            check_fixture(name, exp, results[name], out)
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001 - report every fixture, then fail
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(expected) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
