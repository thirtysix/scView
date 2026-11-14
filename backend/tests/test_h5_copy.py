"""core/h5_copy.py: object references stay valid in a copied group."""

from __future__ import annotations

import h5py
import numpy as np

from scview.core import h5_copy


def test_references_point_into_the_new_file(tmp_path):
    with h5py.File(tmp_path / "a.h5", "w") as f:
        g = f.create_group("obs")
        cats = g.create_group("__categories")
        cats["c"] = np.asarray([b"x", b"y"])
        g["c"] = np.asarray([0, 1, 1], dtype="i1")
        g["c"].attrs["categories"] = cats["c"].ref
        f["elsewhere"] = np.arange(3)
        g.attrs["many"] = np.asarray([cats["c"].ref, f["elsewhere"].ref], dtype=h5py.ref_dtype)
        g.attrs["plain"] = "kept"
    with h5py.File(tmp_path / "a.h5", "r") as fi, h5py.File(tmp_path / "b.h5", "w") as fo:
        fo["elsewhere"] = np.arange(3)
        h5_copy.copy(fi["obs"], fo, "renamed")
    with h5py.File(tmp_path / "b.h5", "r") as f:
        g = f["renamed"]
        assert f[g["c"].attrs["categories"]].name == "/renamed/__categories/c"
        assert [f[r].name for r in g.attrs["many"]] == ["/renamed/__categories/c", "/elsewhere"]
        assert g.attrs["plain"] == "kept"


def test_a_plain_copy_breaks_them(tmp_path):
    """The flaw this module exists for: h5py's own copy keeps the old file's reference."""
    with h5py.File(tmp_path / "a.h5", "w") as f:
        f.create_group("g")["t"] = np.arange(2)
        f["g"].attrs["r"] = f["g/t"].ref
    with h5py.File(tmp_path / "a.h5", "r") as fi, h5py.File(tmp_path / "b.h5", "w") as fo:
        fi.copy(fi["g"], fo, name="g")
    with h5py.File(tmp_path / "b.h5", "r") as f:
        try:
            ok = f[f["g"].attrs["r"]].name == "/g/t"
        except (ValueError, KeyError, OSError):
            ok = False
    assert not ok
