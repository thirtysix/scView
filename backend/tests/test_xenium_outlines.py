"""Xenium cell and nucleus outlines: read from boundaries files, kept in the outline store at
commit, attached later through the raw-body route, refused when they do not fit."""

from __future__ import annotations

import asyncio
import gzip
import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from scview.core import cell_outlines, xenium
from scview.core.ingestion.bundling import build_bundle
from scview.core.ingestion.paths import junk_reason
from scview.core.ingestion.session import IngestOptions, IngestSessionManager
from scview.core.ingestion.validation import validate_bundle
from tests.test_hvg_sensitivity import _register
from tests.test_xenium import EXPERIMENT, N_CELLS, _bundle, _files, _ids

API = "/api/v1"


def _rings(ids: list, *, closed: bool = True) -> pd.DataFrame:
    """A square of side 2 around each cell's table position (``_position``)."""
    rows = []
    for i, cid in enumerate(ids):
        x, y = _position(i)
        ring = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]
        rows += [(cid, vx, vy) for vx, vy in (ring + ring[:1] if closed else ring)]
    return pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"]).astype(
        {"vertex_x": np.float32, "vertex_y": np.float32})


def _position(i: int) -> tuple[float, float]:
    """Where test_xenium._cells puts cell i."""
    return float(i * 10.0), float(i * 5.0 + 1.0)


def _write_parquet(df: pd.DataFrame, path: Path, row_group_size: int = 7) -> Path:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=row_group_size)
    return path


def _positions(n: int) -> np.ndarray:
    return np.array([_position(i) for i in range(n)])


@pytest.mark.parametrize("kind", ["int", "str", "bytes"])
def test_ids_of_every_xoa_version_and_a_ring_split_over_row_groups(tmp_path, kind):
    raw = [i + 1 for i in range(5)]
    ids = {"int": raw, "str": [f"cell-{i}" for i in raw], "bytes": [f"cell-{i}".encode() for i in raw]}[kind]
    # 5 vertices a ring in row groups of 7: every ring crosses a row group.
    names, offsets, xy = xenium.read_boundaries(_write_parquet(_rings(ids), tmp_path / "cell_boundaries.parquet"))
    want = [str(i) for i in raw] if kind == "int" else [f"cell-{i}" for i in raw]
    assert list(names) == want
    assert offsets.tolist() == [0, 4, 8, 12, 16, 20]  # the closing vertex is dropped
    assert xy[:4].tolist() == [[-1, 0], [1, 0], [1, 2], [-1, 2]]


def test_an_open_ring_keeps_its_vertices_and_a_file_out_of_order_is_joined(tmp_path):
    df = _rings([1, 2], closed=False)
    df = pd.concat([df.iloc[:2], df.iloc[4:], df.iloc[2:4]], ignore_index=True)  # cell 1, cell 2, rest of cell 1
    names, offsets, xy = xenium.read_boundaries(_write_parquet(df, tmp_path / "cell_boundaries.parquet", 100))
    assert list(names) == ["1", "2"] and offsets.tolist() == [0, 4, 8]
    assert xy[:4].tolist() == [[-1, 0], [1, 0], [1, 2], [-1, 2]]  # cell 1 in file order


def test_csv_is_read_when_there_is_no_parquet(tmp_path):
    path = tmp_path / "nucleus_boundaries.csv.gz"
    with gzip.open(path, "wt") as fh:
        _rings(["aaa-1", "bbb-1"]).to_csv(fh, index=False)
    names, offsets, _ = xenium.read_boundaries(path)
    assert list(names) == ["aaa-1", "bbb-1"] and offsets.tolist() == [0, 4, 8]


def test_a_file_without_the_columns_is_a_sentence_naming_it(tmp_path):
    path = tmp_path / "cell_boundaries.parquet"
    pq.write_table(pa.table({"cell_id": [1], "x": [1.0]}), path)
    with pytest.raises(xenium.XeniumError, match="cell_boundaries.parquet lacks the column"):
        xenium.read_boundaries(path)
    path.write_bytes(b"PAR1 not really")
    with pytest.raises(xenium.XeniumError, match="could not be read"):
        xenium.read_boundaries(path)


def _with_boundaries(root: Path, n: int = N_CELLS, kind: str = "int", csv: bool = False) -> None:
    ids = _ids(n, kind)
    if csv:
        with gzip.open(root / "cell_boundaries.csv.gz", "wt") as fh:
            _rings(ids).to_csv(fh, index=False)
    else:
        _write_parquet(_rings(ids), root / "cell_boundaries.parquet")
    _write_parquet(_rings(ids), root / "nucleus_boundaries.parquet")


def _commit(tmp_path: Path, src: Path, **opts) -> tuple[str, Path]:
    data = tmp_path / "data"
    mgr = IngestSessionManager(str(data))
    sid = mgr.create_session()
    for p in _files(src):
        dest = mgr.files_dir(sid) / p.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    if opts:
        mgr.set_options(sid, IngestOptions(**opts))

    class DM:
        async def load_dataset(self, did):
            return None

    return asyncio.run(mgr.commit(sid, DM())), data


def test_boundaries_are_kept_not_skipped_and_parquet_beats_csv(tmp_path):
    assert junk_reason("run/outs/cell_boundaries.parquet") is None
    assert junk_reason("run/outs/nucleus_boundaries.csv.gz") is None
    assert junk_reason("run/outs/transcripts.parquet")  # still left out of the folder
    root = tmp_path / "outs"
    _bundle(root)
    _with_boundaries(root)
    _with_boundaries(root, csv=True)
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert len(b.units) == 1 and b.complete
    parts = {f.spatial_part: f.name for f in b.units[0].side_files}
    assert parts["cell_boundaries"] == "cell_boundaries.parquet"
    assert parts["nucleus_boundaries"] == "nucleus_boundaries.parquet"
    assert [s.reason for s in b.skipped if "cell_boundaries.csv.gz" in s.path] == [
        "the same outlines as cell_boundaries.parquet, which is read"]


@pytest.mark.parametrize("mode,kind", [("off", "int"), ("on", "str")])
def test_commit_writes_the_store_in_memory_and_in_blocks(tmp_path, mode, kind):
    src = tmp_path / "src" / "outs"
    _bundle(src, kind=kind, shuffle=True)
    _with_boundaries(src, kind=kind)
    did, data = _commit(tmp_path, tmp_path / "src", streamed=mode)
    assert cell_outlines.kinds_of(data, did) == ["cell", "nucleus"]
    names = ad.read_h5ad(next((data / "ingested" / did).glob("*.h5ad")), backed="r").obs_names
    offsets, xy = cell_outlines.read_for(cell_outlines.store_path(data, did, "cell"), names)
    assert offsets[-1] == 4 * N_CELLS and (np.diff(offsets) == 4).all()
    a = ad.read_h5ad(next((data / "ingested" / did).glob("*.h5ad")))
    assert "cell" not in str(list(a.obsm)) and not any("outline" in k for k in a.uns)  # never in the h5ad
    centre = np.array([xy[offsets[i]:offsets[i + 1]].mean(axis=0) for i in range(N_CELLS)])
    assert np.allclose(centre, a.obsm["spatial"], atol=1e-3)


def test_a_bundle_without_them_loads_as_before(tmp_path):
    src = tmp_path / "src" / "outs"
    _bundle(src)
    did, data = _commit(tmp_path, tmp_path / "src")
    assert cell_outlines.kinds_of(data, did) == []


def test_boundaries_of_another_run_leave_a_note_not_a_refused_dataset(tmp_path):
    src = tmp_path / "src" / "outs"
    _bundle(src)
    df = _rings(_ids(N_CELLS, "int"))
    df["vertex_x"] += 5000.0  # ids 1..n match any XOA 1.0 run: the position says it is another
    _write_parquet(df, src / "cell_boundaries.parquet")
    _write_parquet(_rings(_ids(N_CELLS, "int")), src / "nucleus_boundaries.parquet")
    did, data = _commit(tmp_path, tmp_path / "src")
    assert cell_outlines.kinds_of(data, did) == ["nucleus"]
    a = ad.read_h5ad(next((data / "ingested" / did).glob("*.h5ad")))
    assert "sit inside their outline" in xenium.outlines_note(a)


def test_merged_samples_have_no_outlines_and_the_wizard_says_so(tmp_path):
    src = tmp_path / "src"
    _bundle(src / "a" / "outs", seed=1)
    _bundle(src / "b" / "outs", experiment={**EXPERIMENT, "region_name": ""}, seed=2)
    _with_boundaries(src / "a" / "outs")
    b = build_bundle(_files(src), root=src)
    notes = [i for i in validate_bundle(b).issues if i.code == "xenium_merge_no_outlines"]
    assert len(notes) == 1 and notes[0].severity == "info" and "no cell outlines" in notes[0].message
    did, data = _commit(tmp_path, src, name="two")
    assert cell_outlines.kinds_of(data, did) == []
    lone = build_bundle(_files(src / "a"), root=src / "a")
    assert not [i for i in validate_bundle(lone).issues if i.code == "xenium_merge_no_outlines"]


# --- attach to an existing dataset ---------------------------------------------------------


def _adata(n: int = N_CELLS, technology: str = "xenium") -> ad.AnnData:
    a = ad.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)),
                   obs=pd.DataFrame(index=_ids(n, "int")), var=pd.DataFrame(index=["A", "B", "C"]))
    a.obsm["spatial"] = _positions(n)
    a.obs["library_id"] = pd.Categorical(["r1"] * n)
    a.uns["spatial"] = {"r1": {"images": {}, "scalefactors": {}, "metadata": {}}}
    a.uns["scview_spatial"] = {"technology": technology, "unit": "cell"}
    return a


def _bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


async def _post(client, did, name, body, **kw):
    return await client.post(f"{API}/datasets/{did}/spatial/outlines", params={"name": name}, content=body, **kw)


async def test_attach_then_the_info_and_outlines_routes_show_them(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "xa")
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["spatial"]["outlines"] == [] and info["spatial"]["transcripts"] is False
    r = await _post(client, did, "cell_boundaries.parquet", _bytes(_rings(_ids(N_CELLS, "int"))))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "cell" and body["n_matched"] == N_CELLS and body["outlines"] == ["cell"]
    assert (await client.get(f"{API}/datasets/{did}")).json()["spatial"]["outlines"] == ["cell"]
    r = await client.get(f"{API}/datasets/{did}/spatial/outlines", params={"kind": "cell"})
    head = np.frombuffer(r.content, "<i4", 2)
    assert head.tolist() == [N_CELLS, 4 * N_CELLS]
    leftovers = [p.name for p in cell_outlines.store_dir(dataset_manager.data_dir, did).iterdir()]
    assert leftovers == ["cell.h5"]  # the upload is gone
    r = await _post(client, did, "nucleus_boundaries.parquet", _bytes(_rings(_ids(N_CELLS, "int"))))
    assert r.json()["outlines"] == ["cell", "nucleus"]


async def test_attach_refusals_are_sentences_with_the_counts(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "xb")
    few = _ids(N_CELLS, "int")[: N_CELLS // 2 - 1]
    r = await _post(client, did, "cell_boundaries.parquet", _bytes(_rings(few)))
    assert r.status_code == 422 and f"matches {len(few)} of this dataset's {N_CELLS} cells" in r.json()["detail"]
    moved = _rings(_ids(N_CELLS, "int"))
    moved["vertex_y"] += 1000.0
    r = await _post(client, did, "cell_boundaries.parquet", _bytes(moved))
    assert r.status_code == 422 and "sit inside their outline" in r.json()["detail"]
    r = await _post(client, did, "cells.parquet", b"x")
    assert r.status_code == 422 and "not a Xenium boundaries file" in r.json()["detail"]
    r = await _post(client, did, "cell_boundaries.parquet", b"")
    assert r.status_code == 422 and "empty" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/spatial/outlines", params={"name": "cell_boundaries.parquet"},
                          files={"file": ("cell_boundaries.parquet", b"x")})
    assert r.status_code == 415
    r = await _post(client, "nope", "cell_boundaries.parquet", b"x")
    assert r.status_code == 404
    folder = cell_outlines.store_dir(dataset_manager.data_dir, did)
    assert not list(folder.iterdir())
    plain = ad.AnnData(X=sp.csr_matrix(np.ones((4, 2), dtype=np.float32)))
    other = _register(dataset_manager, plain, "xc")
    r = await _post(client, other, "cell_boundaries.parquet", b"x")
    assert r.status_code == 422 and "no spatial coordinates" in r.json()["detail"]


async def test_removing_the_dataset_removes_the_store(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "xd")
    await _post(client, did, "cell_boundaries.parquet", _bytes(_rings(_ids(N_CELLS, "int"))))
    assert cell_outlines.store_dir(dataset_manager.data_dir, did).exists()
    assert dataset_manager.remove_dataset(did)
    assert not cell_outlines.store_dir(dataset_manager.data_dir, did).exists()
