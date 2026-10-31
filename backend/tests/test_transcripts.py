"""The Xenium transcript store: what is kept, the check against the matrix, view queries, the
routes (raw-body upload, index stream, binary GET) and the co-pilot's fact."""

from __future__ import annotations

import io
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from scview.core import assistant, claims, cell_outlines
from scview.core import transcripts as tr
from scview.core.rag import router as rag_router
from tests.test_hvg_sensitivity import _events, _register

API = "/api/v1"
BOUNDS = (0.0, 0.0, 640.0, 640.0)


def _table(n: int = 5000, seed: int = 0, *, cell_ids: str = "int", names_as: str = "binary", is_gene: bool = False) -> pd.DataFrame:
    """A transcripts table not in gene or spatial order: three genes and every kind of control."""
    rng = np.random.default_rng(seed)
    names = rng.choice(["ERBB2", "KRT7", "RARE", "NegControlProbe_1", "BLANK_1", "antisense_X", "Unassigned_2",
                        "DeprecatedCodeword_1", "Intergenic_Region_3"], size=n, p=[.5, .3, .02, .05, .05, .02, .02, .02, .02])
    ints = rng.integers(-1, 50, size=n)
    df = pd.DataFrame({
        "transcript_id": np.arange(n, dtype=np.uint64),
        "cell_id": ints.astype(np.int32) if cell_ids == "int" else
        [("UNASSIGNED" if i < 0 else f"cell-{i}") for i in ints],
        "overlaps_nucleus": rng.integers(0, 2, size=n).astype(np.uint8),
        "feature_name": [s.encode() for s in names] if names_as == "binary" else names,
        "x_location": rng.uniform(0, 640, size=n).astype(np.float32),
        "y_location": rng.uniform(0, 640, size=n).astype(np.float32),
        "z_location": rng.uniform(0, 20, size=n).astype(np.float32),
        "qv": rng.choice([10.0, 19.9, 20.0, 40.0], size=n).astype(np.float32),
    })
    if is_gene:
        df["is_gene"] = ~pd.Series(names).str.contains("_").to_numpy()
    return df


def _parquet(df: pd.DataFrame, path: Path, row_group_size: int = 700) -> Path:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=row_group_size)
    return path


def _build(tmp_path: Path, df: pd.DataFrame, **kw) -> tuple[Path, dict]:
    source = _parquet(df, tmp_path / "t.parquet")
    dest = tmp_path / "store" / "transcripts.h5"
    events = list(tr.iter_index(source, dest, BOUNDS, **kw))
    assert events[-1]["type"] == "result" and all(e["type"] == "progress" for e in events[:-1])
    return dest, events[-1]["summary"]


def _matrix(path: Path, totals: dict[str, int], controls: dict[str, int] | None = None) -> Path:
    """A 10x matrix with one cell holding each feature's total."""
    controls = controls or {}
    names = [*totals, *controls]
    kinds = [b"Gene Expression"] * len(totals) + [b"Negative Control Probe"] * len(controls)
    data = np.array([*totals.values(), *controls.values()], dtype=np.int32)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"], m["indices"] = data, np.arange(len(names), dtype=np.int64)
        m["indptr"] = np.array([0, len(names)], dtype=np.int64)
        m["shape"] = np.array([len(names), 1], dtype=np.int32)
        m["barcodes"] = np.array([b"1"])
        fe = m.create_group("features")
        fe["name"], fe["feature_type"] = np.array([n.encode() for n in names]), np.array(kinds)
    return path


def _kept(df: pd.DataFrame) -> pd.DataFrame:
    gene = df["feature_name"].map(lambda v: v.decode() if isinstance(v, bytes) else v).isin(["ERBB2", "KRT7", "RARE"])
    return df[gene & (df["qv"] >= 20)].assign(gene=lambda d: d["feature_name"].map(
        lambda v: v.decode() if isinstance(v, bytes) else v))


def test_the_store_keeps_q20_genes_says_what_it_dropped_and_leaves_no_part(tmp_path):
    df = _table()
    dest, s = _build(tmp_path, df)
    kept = _kept(df)
    gene_rows = df["feature_name"].map(lambda v: v.decode()).isin(["ERBB2", "KRT7", "RARE"])
    assert s["n_rows"] == len(df) and s["n_kept"] == len(kept)
    assert s["n_control"] == int((~gene_rows).sum()) and s["n_low_quality"] == int(gene_rows.sum()) - len(kept)
    assert s["n_kept"] + s["n_control"] + s["n_low_quality"] == s["n_rows"]
    assert s["min_qv"] == 20 and s["n_genes"] == 3
    assert [p.name for p in dest.parent.iterdir()] == ["transcripts.h5"]
    with h5py.File(dest) as f:
        assert list(f["genes"].asstr()[:]) == ["ERBB2", "KRT7", "RARE"]
        offsets = f["gene_offsets"][:]
        assert np.diff(offsets).tolist() == [int((kept["gene"] == g).sum()) for g in ("ERBB2", "KRT7", "RARE")]
        assert f["x"].dtype == np.dtype("<f4") and f["qv"].dtype == np.uint8 and f["x"].chunks is None
        qv = f["qv"][:]
    assert qv.min() >= 20 and set(np.unique(qv)) <= {20, 40}


def test_every_point_is_where_it_was_and_in_coarse_spatial_order(tmp_path):
    df = _table(seed=3)
    dest, _ = _build(tmp_path, df)
    kept = _kept(df)
    with h5py.File(dest) as f:
        offsets, tiles = f["gene_offsets"][:], f["tile_offsets"][:]
        x, y = f["x"][:], f["y"][:]
    grid = tr._Grid(BOUNDS)
    for i, g in enumerate(("ERBB2", "KRT7", "RARE")):
        gx, gy = x[offsets[i]:offsets[i + 1]], y[offsets[i]:offsets[i + 1]]
        want = kept[kept["gene"] == g]
        assert sorted(zip(gx.tolist(), gy.tolist())) == sorted(zip(want["x_location"].tolist(), want["y_location"].tolist()))
        at = grid.tile(gx, gy)
        assert (np.diff(at) >= 0).all()  # tile by tile
        assert (np.bincount(at, minlength=tr.TILES**2) == np.diff(tiles[i])).all()


@pytest.mark.parametrize("kw", [{"cell_ids": "text"}, {"names_as": "string"}, {"is_gene": True}])
def test_newer_xoa_columns(tmp_path, kw):
    """String cell ids (UNASSIGNED for none), string feature names, the is_gene column."""
    df = _table(**kw)
    dest, s = _build(tmp_path, df)
    kept = _kept(df)
    assert s["n_kept"] == len(kept)
    assigned = kept["cell_id"] >= 0 if "cell_ids" not in kw else kept["cell_id"] != "UNASSIGNED"
    assert s["n_assigned"] == int(assigned.sum())


def test_is_gene_false_leaves_even_a_name_that_looks_like_a_gene(tmp_path):
    df = _table()
    df["is_gene"] = df["feature_name"].map(lambda v: v == b"ERBB2")
    _, s = _build(tmp_path, df)
    assert s["n_genes"] >= 1
    assert s["n_kept"] == int(((df["feature_name"] == b"ERBB2") & (df["qv"] >= 20)).sum())


def test_the_check_against_the_matrix_counts_q20_with_a_cell_id(tmp_path):
    df = _table()
    kept = _kept(df)
    totals = {g: int(((kept["gene"] == g) & (kept["cell_id"] >= 0)).sum()) for g in ("ERBB2", "KRT7", "RARE")}
    matrix = _matrix(tmp_path / "m.h5", totals, {"NegControlProbe_1": 99})
    _, s = _build(tmp_path, df, matrix=matrix)
    c = s["check"]
    assert c["available"] and c["n_genes"] == 3 and c["n_equal"] == 3 and c["max_abs_diff"] == 0 and c["median_abs_diff"] == 0
    assert c["r"] == pytest.approx(1.0) and c["only_in_transcripts"] == [] and c["only_in_matrix"] == []


def test_the_check_says_when_they_differ(tmp_path):
    df = _table()
    kept = _kept(df)
    totals = {g: int(((kept["gene"] == g) & (kept["cell_id"] >= 0)).sum()) for g in ("ERBB2", "KRT7")}
    totals["KRT7"] += 7
    totals["OTHER"] = 3
    _, s = _build(tmp_path, df, matrix=_matrix(tmp_path / "m.h5", totals))
    c = s["check"]
    assert c["n_genes"] == 2 and c["max_abs_diff"] == 7 and c["max_diff_gene"] == "KRT7"
    assert c["only_in_transcripts"] == ["RARE"] and c["only_in_matrix"] == ["OTHER"]


def test_no_matrix_is_said_not_failed(tmp_path):
    _, s = _build(tmp_path, _table())
    assert s["check"]["available"] is False and "cell_feature_matrix.h5" in s["check"]["note"]


def test_a_footer_with_the_wrong_row_count_is_not_believed(tmp_path):
    """XOA 1.0.1's footer says num_rows = 1,000,000 of 42 M: the row groups are added up."""
    source = _parquet(_table(), tmp_path / "t.parquet")
    plan = tr.read_plan(source)
    assert plan["rows"] == 5000 and plan["row_groups"] == 8 and plan["largest_row_group"] == 700


def test_a_stop_leaves_no_store_and_no_part(tmp_path):
    source = _parquet(_table(), tmp_path / "t.parquet")
    dest = tmp_path / "s" / "transcripts.h5"
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 10  # into the second pass

    with pytest.raises(tr.TranscriptsCancelled):
        list(tr.iter_index(source, dest, BOUNDS, stop=stop))
    assert not dest.exists() and not list(dest.parent.iterdir())


def test_a_build_holds_one_row_group_in_memory(tmp_path):
    source = _parquet(_table(), tmp_path / "t.parquet")
    sizes = []
    original = tr._read_block

    def spy(pf, i, columns, code_of):
        block = original(pf, i, columns, code_of)
        sizes.append(block.n_rows)
        return block

    tr._read_block = spy
    try:
        list(tr.iter_index(source, tmp_path / "s.h5", BOUNDS))
    finally:
        tr._read_block = original
    assert max(sizes) == 700 and len(sizes) == 16  # two passes of 8 row groups


def test_no_gene_with_q20_is_a_sentence(tmp_path):
    df = _table()
    df["qv"] = np.float32(5)
    source = _parquet(df, tmp_path / "t.parquet")
    with pytest.raises(tr.TranscriptsError, match="no gene transcripts"):
        list(tr.iter_index(source, tmp_path / "s.h5", BOUNDS))
    assert not (tmp_path / "s.h5").exists()


# --- view queries --------------------------------------------------------------------------


def test_a_view_gives_exactly_the_points_inside(tmp_path):
    df = _table(20000)
    dest, _ = _build(tmp_path, df)
    kept = _kept(df)
    box = (100.0, 50.0, 180.5, 333.0)
    v = tr.query(dest, ["KRT7", "ERBB2"], box)
    for j, g in enumerate(["KRT7", "ERBB2"]):
        want = kept[(kept["gene"] == g) & kept["x_location"].between(box[0], box[2]) & kept["y_location"].between(box[1], box[3])]
        got = v.gene == j
        assert v.per_gene[j] == len(want)
        assert sorted(zip(v.x[got].tolist(), v.y[got].tolist())) == sorted(zip(want["x_location"].tolist(), want["y_location"].tolist()))
    assert v.total == sum(v.per_gene) == len(v.gene)


def test_a_point_outside_the_tile_grid_is_found(tmp_path):
    df = _table()
    df.loc[df.index[:50], "x_location"] = np.float32(5000.0)  # past the section's bounds, in the edge tile
    df.loc[df.index[:50], "feature_name"] = b"KRT7"
    df.loc[df.index[:50], "qv"] = np.float32(40.0)
    dest, _ = _build(tmp_path, df)
    assert tr.query(dest, ["KRT7"], (4000.0, 0.0, 6000.0, 640.0)).total == 50


def test_over_the_limit_every_gene_is_thinned_by_the_same_share_evenly(tmp_path):
    df = _table(20000)
    dest, _ = _build(tmp_path, df)
    whole = tr.query(dest, ["ERBB2", "KRT7"])
    v = tr.query(dest, ["ERBB2", "KRT7"], limit=1000)
    assert v.total == whole.total and v.per_gene == whole.per_gene
    assert len(v.gene) <= 1000
    shares = [(v.gene == j).sum() / n for j, n in enumerate(whole.per_gene)]
    assert shares[0] == pytest.approx(shares[1], abs=0.002)
    # Evenly over the section: each quarter of it keeps about a quarter of what is shown.
    quarter = (v.x < 320) & (v.y < 320)
    assert 0.2 < quarter.mean() < 0.3


def test_a_view_refuses_what_it_cannot_answer(tmp_path):
    dest, _ = _build(tmp_path, _table())
    with pytest.raises(tr.TranscriptsError, match="1 to 8 genes"):
        tr.query(dest, [f"G{i}" for i in range(9)])
    with pytest.raises(tr.TranscriptsError, match="not among this run's transcripts") as e:
        tr.query(dest, ["NOPE"])
    assert e.value.status == 404
    v = tr.query(dest, ["ERBB2"], (-50.0, -50.0, -10.0, -10.0))
    assert v.total == 0 and len(v.gene) == 0


def test_wire_format():
    v = tr.View(np.array([0, 0, 1], np.uint8), np.array([1, 2, 3], np.float32), np.array([4, 5, 6], np.float32), 9, [2, 7], 0.0)
    raw = tr.encode(v)
    assert np.frombuffer(raw, "<i4", 1)[0] == 3
    assert np.frombuffer(raw, np.uint8, 3, offset=4).tolist() == [0, 0, 1]
    assert len(raw) == 4 + 4 + 24  # three gene bytes, one of padding
    assert np.frombuffer(raw, "<f4", offset=8).tolist() == [1, 2, 3, 4, 5, 6]


def test_estimates_say_what_a_build_needs():
    est = tr.estimate(42_638_083, 1_000_000)
    # Measured on the breast Rep1 run: 312 MB on disk, 12.6 s on two cores.
    assert 0.8 * 312e6 < est["disk_bytes"] < 1.2 * 312e6 and est["seconds_range"][0] <= 12.6 <= est["seconds_range"][1]
    for_size = tr.estimate_for_size(669_769_801)
    assert abs(for_size["rows"] - 42_638_083) / 42_638_083 < 0.05 and for_size["upload_bytes"] == 669_769_801


# --- routes --------------------------------------------------------------------------------


def _adata(n: int = 40, libraries: int = 1) -> ad.AnnData:
    a = ad.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)), obs=pd.DataFrame(index=[str(i) for i in range(n)]),
                   var=pd.DataFrame(index=["ERBB2", "KRT7", "RARE"]))
    a.obsm["spatial"] = np.column_stack((np.linspace(0, 640, n), np.linspace(0, 640, n)))
    libs = [f"r{i % libraries}" for i in range(n)]
    a.obs["library_id"] = pd.Categorical(libs)
    a.uns["spatial"] = {lib: {"images": {}, "scalefactors": {}, "metadata": {}} for lib in sorted(set(libs))}
    a.uns["scview_spatial"] = {"technology": "xenium", "unit": "cell"}
    return a


def _bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, row_group_size=700)
    return buf.getvalue()


async def _attach(client, did, df=None):
    r = await client.post(f"{API}/datasets/{did}/spatial/transcripts", params={"name": "transcripts.parquet"},
                          content=_bytes(_table() if df is None else df))
    assert r.status_code == 200, r.text
    return r.json()


async def test_upload_index_query_and_delete_through_the_routes(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "ta")
    data = dataset_manager.data_dir
    st = (await client.get(f"{API}/datasets/{did}/spatial/transcripts/status", params={"size": 669_769_801})).json()
    assert st["attached"] is False and st["estimate_for_size"]["disk_bytes"] > 0 and st["disk_free"] > 0
    assert (await client.get(f"{API}/datasets/{did}/spatial/transcripts", params={"genes": "ERBB2"})).status_code == 404

    st = await _attach(client, did)
    assert st["uploaded"] is True and st["attached"] is False and st["estimate"]["rows"] == 5000
    assert tr.upload_path(data, did).is_file()
    assert not list(tr.store_dir(data, did).glob("*.part"))

    r = await client.post(f"{API}/datasets/{did}/spatial/transcripts/index-stream")
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "progress" in kinds
    assert events[0][1]["estimate"]["peak_bytes"] > 0
    assert not tr.upload_path(data, did).exists()  # the parquet is gone once indexed
    summary = events[-2][1]
    assert summary["n_genes"] == 3 and summary["check"]["available"] is False

    st = (await client.get(f"{API}/datasets/{did}/spatial/transcripts/status")).json()
    assert st["attached"] and [g["name"] for g in st["genes"]] == ["ERBB2", "KRT7", "RARE"] and st["running"] is False
    assert (await client.get(f"{API}/datasets/{did}")).json()["spatial"]["transcripts"] is True

    r = await client.get(f"{API}/datasets/{did}/spatial/transcripts",
                         params={"genes": "KRT7,ERBB2", "x0": 0, "x1": 320, "y0": 0, "y1": 320, "limit": 50})
    assert r.status_code == 200
    n = np.frombuffer(r.content, "<i4", 1)[0]
    assert int(r.headers["x-transcripts-shown"]) == n <= 50
    assert int(r.headers["x-transcripts-total"]) == sum(map(int, r.headers["x-transcripts-per-gene"].split(",")))
    assert int(r.headers["x-transcripts-total"]) > n

    assert (await client.delete(f"{API}/datasets/{did}/spatial/transcripts")).json() == {"deleted": True}
    assert not tr.store_dir(data, did).exists()


async def test_route_refusals_are_sentences(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "tb")
    post = lambda name, body, **kw: client.post(  # noqa: E731
        f"{API}/datasets/{did}/spatial/transcripts", params={"name": name}, content=body, **kw)
    r = await post("transcripts.csv.gz", b"x")
    assert r.status_code == 422 and "transcripts.parquet" in r.json()["detail"]
    r = await post("transcripts.parquet", b"not parquet at all")
    assert r.status_code == 422 and "not parquet" in r.json()["detail"]
    r = await post("transcripts.parquet", b"")
    assert r.status_code == 422 and "empty" in r.json()["detail"]
    r = await post("transcripts.parquet", _bytes(pd.DataFrame({"a": [1]})))
    assert r.status_code == 422 and "lacks the column" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/spatial/transcripts", params={"name": "transcripts.parquet"},
                          files={"file": ("transcripts.parquet", b"x")})
    assert r.status_code == 415
    assert not list(tr.store_dir(dataset_manager.data_dir, did).glob("*"))
    r = await client.post(f"{API}/datasets/{did}/spatial/transcripts/index-stream")
    assert r.status_code == 422 and "send transcripts.parquet first" in r.json()["detail"]
    await _attach(client, did)
    await client.post(f"{API}/datasets/{did}/spatial/transcripts/index-stream")
    r = await client.get(f"{API}/datasets/{did}/spatial/transcripts", params={"genes": ",".join(f"G{i}" for i in range(9))})
    assert r.status_code == 422 and "1 to 8 genes" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/spatial/transcripts", params={"genes": "NOPE"})
    assert r.status_code == 404 and "NOPE" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/spatial/transcripts", params={"genes": "ERBB2", "x0": 1})
    assert r.status_code == 422 and "all of x0" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/spatial/transcripts",
                         params={"genes": "ERBB2", "x0": 5, "x1": 1, "y0": 0, "y1": 1})
    assert r.status_code == 422


async def test_a_merged_dataset_is_refused_with_a_sentence(client, dataset_manager):
    did = _register(dataset_manager, _adata(libraries=2), "tc")
    r = await client.post(f"{API}/datasets/{did}/spatial/transcripts", params={"name": "transcripts.parquet"},
                          content=_bytes(_table()))
    assert r.status_code == 422 and "merges 2 sections" in r.json()["detail"]
    plain = _register(dataset_manager, ad.AnnData(X=sp.csr_matrix(np.ones((4, 2), dtype=np.float32))), "td")
    r = await client.post(f"{API}/datasets/{plain}/spatial/transcripts", params={"name": "transcripts.parquet"},
                          content=_bytes(_table()))
    assert r.status_code == 422 and "no spatial coordinates" in r.json()["detail"]


async def test_removing_the_dataset_removes_its_transcripts(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "te")
    await _attach(client, did)
    assert tr.store_dir(dataset_manager.data_dir, did).exists()
    assert dataset_manager.remove_dataset(did)
    assert not tr.store_dir(dataset_manager.data_dir, did).exists()


def test_the_upload_routes_pass_the_json_body_cap():
    from scview.main import _UPLOAD_PATH_HINTS

    for path in ("/api/v1/datasets/x/spatial/transcripts", "/api/v1/datasets/x/spatial/outlines"):
        assert any(h in path for h in _UPLOAD_PATH_HINTS)
    conf = (Path(__file__).parents[2] / "frontend" / "nginx.conf")
    if conf.exists():  # not mounted in a backend-only container
        assert "spatial/(outlines|transcripts)" in conf.read_text()


# --- co-pilot ------------------------------------------------------------------------------


class _Adaptor:
    def __init__(self, path):
        self.h5ad_path = str(path)


def test_the_fact_says_what_is_attached_and_cites_its_tag(tmp_path):
    data = tmp_path / "data"
    h5ad = data / "ingested" / "d1" / "x.h5ad"
    lines: list[str] = []
    sources: list = []
    assistant._add_imaging_store(_Adaptor(h5ad), lines, sources)
    assert lines == [] and sources == []  # nothing attached: nothing said

    cell_outlines.write(cell_outlines.store_path(data, "d1", "cell"), ["a"], np.array([0, 3]), np.zeros((3, 2)))
    assistant._add_imaging_store(_Adaptor(h5ad), lines, sources)
    assert "cell outlines" in "\n".join(lines) and "no transcripts" in "\n".join(lines)

    source = _parquet(_table(), tmp_path / "t.parquet")
    df = _table()
    kept = _kept(df)
    totals = {g: int(((kept["gene"] == g) & (kept["cell_id"] >= 0)).sum()) for g in ("ERBB2", "KRT7", "RARE")}
    list(tr.iter_index(source, tr.store_path(data, "d1"), BOUNDS, matrix=_matrix(tmp_path / "m.h5", totals)))
    lines, sources = [], []
    assistant._add_imaging_store(_Adaptor(h5ad), lines, sources)
    text = "\n".join(lines)
    assert tr.TAG in text and f"{len(kept):,} kept" in text and "r = 1.0000" in text and "q 20" in text
    assert [s.ref for s in sources] == [tr.TAG] and sources[0].detail and "r = 1.0000" in sources[0].evidence
    for word in ("transcripts", "outlines"):
        assert claims.resolve_tag(word, [tr.TAG, "result:markers:B"]) == [tr.TAG]


def test_router_words_keep_the_fact_on_a_classified_route():
    for facet_words in ("transcripts", "outlines", "segmentation"):
        assert any(facet_words in words for words in rag_router._RESULT_FACET_HINTS.values())
