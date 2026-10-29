"""The per-dataset results store (core/results_store.py) and the modules that
keep their results in it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import composition, de_results, provenance, results_store
from tests.test_composition import T2, T2_SAMPLES, _obs


@dataclass
class _Nested:
    name: str
    values: np.ndarray


@dataclass
class _Payload:
    table: pd.DataFrame
    matrix: sp.csc_matrix
    pair: tuple[str, str]
    by_key: dict
    nested: list[_Nested] = field(default_factory=list)
    note: str | None = None


class FakeAdaptor:
    """What the store reads from an adaptor: its file path, adata and cells hash."""

    def __init__(self, root: Path, dataset_id: str, adata: ad.AnnData) -> None:
        self.h5ad_path = str(root / "derived" / dataset_id / "data.h5ad")
        self.adata = adata

    def cells_fingerprint(self) -> str:
        return hashlib.sha256("\n".join(self.adata.obs_names).encode()).hexdigest()[:16]


def _adata(n: int = 20) -> ad.AnnData:
    return ad.AnnData(np.zeros((n, 3), dtype=np.float32),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]))


def _stored(dataset_id: str, kind: str, value, **kw):
    fut = results_store.keep(dataset_id, kind, value, **kw)
    assert fut is not None
    fut.result()
    return results_store.entries(dataset_id, kind)


def test_round_trip_keeps_structure_and_p_values(tmp_path, monkeypatch):
    monkeypatch.setattr(results_store, "TRUSTED_MODULES", ("scview.", "tests."))
    table = pd.DataFrame({"gene": ["A", "B", "C"], "logfoldchange": [1.5, -2.25, 0.1],
                          "pval": [1e-300, 0.5, 2e-40], "pvals_adj": [1e-298, 0.6, 3e-39],
                          "n": np.array([1, 2, 300], dtype=np.int64)})
    payload = _Payload(table=table, matrix=sp.random(30, 4, density=0.2, format="csc", random_state=0),
                       pair=("leiden", "3"), by_key={("a", 1): "x"},
                       nested=[_Nested("n1", np.arange(5, dtype=np.float64))])
    [e] = _stored("ds1", "test", payload, key="k", label="a test")
    entry, back = results_store.load("ds1", e["id"])
    assert entry["label"] == "a test" and isinstance(back, _Payload)
    assert back.pair == ("leiden", "3") and back.by_key == {("a", 1): "x"}
    assert back.nested[0].name == "n1" and np.array_equal(back.nested[0].values, np.arange(5))
    assert (back.matrix != payload.matrix).nnz == 0 and back.matrix.format == "csc"
    # p-values stay float64 and exact below float32's range; the rest is narrowed.
    assert back.table["pval"].dtype == np.float64 and back.table["pval"].iloc[0] == 1e-300
    assert back.table["pvals_adj"].dtype == np.float64 and back.table["pval"].iloc[2] == 2e-40
    assert back.table["logfoldchange"].dtype == np.float32
    assert back.table["n"].dtype == np.uint16
    assert list(back.table["gene"]) == ["A", "B", "C"]


def test_p_value_column_names():
    for name in ("pval", "pvals_adj", "padj", "p_value", "spatial_fdr", "cellphone_pvals", "qval", "FDR"):
        assert results_store.is_p_column(name), name
    for name in ("logfoldchange", "score", "pct_in", "lr_means"):
        assert not results_store.is_p_column(name), name


def test_refuses_classes_outside_scview(tmp_path):
    [e] = _stored("ds1", "test", {"x": 1})
    import gzip
    path = results_store.root() / "ds1" / e["id"] / results_store.RESULT
    doc = gzip.decompress(path.read_bytes()).decode().replace('{"x": 1}', '{"__dc__": "os:PathLike", "f": {}}')
    path.write_bytes(gzip.compress(doc.encode()))
    assert results_store.load("ds1", e["id"]) is None


def test_unsafe_ids_are_ignored():
    assert results_store.keep("../x", "test", {}) is None
    assert results_store.load("ds1", "../../etc") is None
    assert results_store.entries("a/b") == []


def test_dataset_id_from_adaptor_path(tmp_path):
    assert results_store.dataset_id_of(FakeAdaptor(tmp_path, "abc-1", _adata())) == "abc-1"
    up = type("A", (), {"h5ad_path": str(tmp_path / "uploads" / "u9" / "folder" / "x.h5ad")})()
    assert results_store.dataset_id_of(up) == "u9"
    assert results_store.dataset_id_of(object()) is None


def test_superseded_kept_last_three(tmp_path):
    for i in range(6):
        _stored("ds1", "markers", {"i": i}, key="leiden")
    rows = results_store.entries("ds1", "markers")
    assert [e["status"] for e in rows] == ["primary"] + ["superseded"] * results_store.KEEP_SUPERSEDED
    assert results_store.load("ds1", rows[0]["id"])[1] == {"i": 5}
    on_disk = {p.name for p in (results_store.root() / "ds1").iterdir() if p.is_dir()}
    assert on_disk == {e["id"] for e in rows}


def test_primaries_capped_per_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(results_store, "KEEP_PER_KIND", 3)
    for i in range(5):
        _stored("ds1", "de", {"i": i}, key=f"k{i}")
    assert [results_store.load("ds1", e["id"])[1]["i"] for e in results_store.entries("ds1", "de")] == [4, 3, 2]


def test_byte_cap_drops_oldest_unpinned_first(tmp_path, monkeypatch):
    big = pd.DataFrame({"x": np.random.default_rng(0).random(20_000)})
    first = _stored("ds1", "t", big, key="a")[0]
    size = first["bytes"]
    monkeypatch.setattr(results_store, "MAX_BYTES", int(size * 2.5))
    _stored("ds1", "t", big, key="b")
    _stored("ds1", "t", big, key="c")
    keys = [e["key"] for e in results_store.entries("ds1", "t")]
    assert keys == ["c", "b"]


def test_stale_when_cells_or_recipe_change(tmp_path):
    a = FakeAdaptor(tmp_path, "ds2", _adata())
    results_store.keep(a, "test", {"v": 1}).result()
    assert results_store.entries("ds2", adaptor=a)[0]["stale"] is False
    assert results_store.latest(a, "test") == {"v": 1}

    provenance.record_step(a.adata, step="normalize", tool="scanpy")
    assert results_store.entries("ds2", adaptor=a)[0]["stale"] is True
    assert results_store.latest(a, "test") is None
    assert results_store.latest(a, "test", current_only=False) == {"v": 1}

    b = FakeAdaptor(tmp_path, "ds2", _adata(10))  # cells filtered
    assert results_store.entries("ds2", adaptor=b)[0]["stale"] is True
    assert results_store.entries("ds2")[0]["stale"] is None  # no adaptor: unknown


def test_a_step_that_only_adds_a_result_leaves_others_current(tmp_path):
    a = FakeAdaptor(tmp_path, "ds2b", _adata())
    results_store.keep(a, "test", {"v": 1}).result()
    provenance.record_step(a.adata, step="spatial_autocorr", tool="scview.spatial_autocorr")
    provenance.record_step(a.adata, step="gene_programs", tool="scview.gene_programs")
    assert results_store.entries("ds2b", adaptor=a)[0]["stale"] is False
    provenance.record_step(a.adata, step="clustering", tool="scanpy")
    assert results_store.entries("ds2b", adaptor=a)[0]["stale"] is True


def test_an_entry_from_before_data_steps_still_counts_every_step(tmp_path):
    a = FakeAdaptor(tmp_path, "ds2c", _adata())
    fp = results_store.fingerprint(a)
    old = {k: v for k, v in fp.items() if k != "n_data_steps"}
    assert results_store._is_stale(old, fp) is False
    provenance.record_step(a.adata, step="spatial_autocorr", tool="scview.spatial_autocorr")
    now = results_store.fingerprint(a)
    assert results_store._is_stale(old, now) is True  # written by an older version: every step counts
    assert results_store._is_stale(fp, now) is False


def test_delete_and_delete_dataset(tmp_path):
    [e] = _stored("ds1", "test", {"v": 1})
    assert results_store.delete("ds1", e["id"]) is True
    assert results_store.delete("ds1", e["id"]) is False
    assert results_store.entries("ds1") == []
    _stored("ds1", "test", {"v": 2})
    results_store.delete_dataset("ds1")
    assert not (results_store.root() / "ds1").exists()


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class _Gene:
    def __init__(self, gene, lfc, p, padj, stat):
        self.gene, self.logfoldchange, self.pval, self.pval_adj, self.stat = gene, lfc, p, padj, stat


def test_de_result_survives_a_restart(tmp_path):
    a = FakeAdaptor(tmp_path, "ds3", _adata())
    genes = [_Gene("CD14", 2.0, 1e-200, 1e-198, 30.5), _Gene("LYZ", -1.0, 0.01, 0.02, -2.5)]
    rid = de_results.remember("ds3", genes, method="wilcoxon",
                              comparison={"mode": "groups", "group_a": "STIM", "group_b": "CTRL"}, adaptor=a)
    results_store.flush()
    de_results.clear()  # a restart

    stored = de_results.recall("ds3", rid)
    assert stored is not None and stored.genes == ["CD14", "LYZ"]
    assert stored.pvals[0] == 1e-200 and stored.label.endswith("STIM vs CTRL (Wilcoxon)")
    assert de_results.recall("other", rid) is None

    de_results.clear()
    listed = de_results.list_results("ds3")
    assert [r for r, _ in listed] == [rid]


def test_composition_grounding_after_the_dataset_is_reopened(tmp_path):
    obs = _obs(T2_SAMPLES, T2)
    adata = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obs_names = [f"c{i}" for i in range(len(obs))]
    result = composition.composition_test(adata.obs, "cond", "donor", "group")
    composition.remember(FakeAdaptor(tmp_path, "ds4", adata), result)
    results_store.flush()

    reopened = FakeAdaptor(tmp_path, "ds4", adata)  # a new object: nothing in memory
    lines, detail = composition.grounding_lines(reopened, adata.obs)
    assert detail and any("g4" in line for line in lines)
    back = composition._last_result(reopened, adata.obs)
    assert [g.group for g in back.groups] == [g.group for g in result.groups]
    assert back.to_dict() == result.to_dict()

    provenance.record_step(adata, step="filter_cells", tool="scanpy")  # computed on an earlier version now
    assert composition._last_result(FakeAdaptor(tmp_path, "ds4", adata), adata.obs) is None


@pytest.mark.asyncio
async def test_results_api(client, dataset_manager, tmp_data_dir):
    did = "abc123"
    (tmp_data_dir / "uploads" / did).mkdir(parents=True)
    (tmp_data_dir / "uploads" / did / "metadata.json").write_text('{"name": "x"}')
    _stored(did, "ppi", [{"label": "q"}], label="Protein interactions")

    r = await client.get(f"/api/v1/datasets/{did}/results")
    assert r.status_code == 200
    [row] = r.json()["results"]
    assert row["kind"] == "ppi" and row["label"] == "Protein interactions" and row["stale"] is None

    r = await client.get(f"/api/v1/datasets/{did}/results/{row['id']}")
    assert r.status_code == 200 and r.json()["params"] == {}
    assert (await client.delete(f"/api/v1/datasets/{did}/results/{row['id']}")).status_code == 200
    assert (await client.get(f"/api/v1/datasets/{did}/results/{row['id']}")).status_code == 404
    assert (await client.get("/api/v1/datasets/nope/results")).status_code == 404

    _stored(did, "ppi", [{"label": "q"}])
    dataset_manager.remove_dataset(did)
    assert not (results_store.root() / did).exists()


# ---------------------------------------------------------------------------
# Phase 2: superseded markers / enrichment (core/uns_history.py)
# ---------------------------------------------------------------------------

def _clustered(n: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, (n, 30)).astype(np.float32),
                   obs=pd.DataFrame({"leiden": pd.Categorical(np.repeat(["0", "1", "2"], n // 3))},
                                    index=[f"c{i}" for i in range(n)]))
    a.var_names = [f"g{j}" for j in range(30)]
    a.X[: n // 3, :5] += 6  # group 0 has markers
    return a


def _markers(a: ad.AnnData, method: str) -> dict:
    import scanpy as sc
    from scview.core.uns_keys import markers_key
    sc.tl.rank_genes_groups(a, "leiden", method=method, key_added=markers_key("leiden"))
    return a.uns[markers_key("leiden")]


def test_before_save_keeps_what_a_rerun_replaced(tmp_path):
    from scview.core import uns_history
    from scview.core.uns_keys import enrichment_key, markers_key

    a = _clustered()
    _markers(a, "t-test")
    a.uns[enrichment_key("leiden", "0")] = '{"terms": [1]}'
    a.uns["ora_meta__leiden__0"] = '{"ora_version": 5}'
    uns_history.capture("ds5", a)

    same = a.uns[markers_key("leiden")]
    a.uns[markers_key("leiden")] = dict(reversed(list(same.items())))  # same content, as read back
    assert uns_history.before_save("ds5", a) == 0

    _markers(a, "wilcoxon")
    del a.uns[enrichment_key("leiden", "0")]
    assert uns_history.before_save("ds5", a) == 2
    results_store.flush()
    rows = results_store.entries("ds5", adaptor=FakeAdaptor(tmp_path, "ds5", a))
    by_kind = {e["kind"]: e for e in rows}
    assert by_kind["markers"]["label"] == "Markers of 'leiden' (t-test)"
    assert by_kind["markers"]["status"] == "superseded" and by_kind["markers"]["stale"] is False
    assert by_kind["enrichment"]["key"] == "leiden|0"
    _, value = results_store.load("ds5", by_kind["enrichment"]["id"])
    assert value["value"] == '{"terms": [1]}' and value["meta"] == '{"ora_version": 5}'
    _, markers = results_store.load("ds5", by_kind["markers"]["id"])
    assert markers["value"]["params"]["method"] == "t-test"
    assert list(markers["value"]["names"].dtype.names) == ["0", "1", "2"]
    assert uns_history.before_save("ds5", a) == 0  # the new ones are the saved ones now


def test_archived_markers_stale_after_relabel_not_after_reorder(tmp_path):
    from scview.core import uns_history

    a = _clustered()
    _markers(a, "t-test")
    uns_history.capture("ds6", a)
    _markers(a, "wilcoxon")
    uns_history.before_save("ds6", a)
    results_store.flush()
    fake = FakeAdaptor(tmp_path, "ds6", a)
    a.obs["leiden"] = a.obs["leiden"].cat.reorder_categories(["2", "0", "1"])
    assert results_store.entries("ds6", adaptor=fake)[0]["stale"] is False
    a.obs["leiden"] = pd.Categorical(np.tile(["0", "1", "2"], 20))
    assert results_store.entries("ds6", adaptor=fake)[0]["stale"] is True


def test_three_superseded_per_column(tmp_path):
    from scview.core import uns_history

    a = _clustered()
    _markers(a, "t-test")
    uns_history.capture("ds7", a)
    for method in ("wilcoxon", "t-test_overestim_var", "logreg", "t-test", "wilcoxon"):
        _markers(a, method)
        uns_history.before_save("ds7", a)
    results_store.flush()
    rows = results_store.entries("ds7")
    assert len(rows) == results_store.KEEP_SUPERSEDED and {e["status"] for e in rows} == {"superseded"}


@pytest.mark.asyncio
async def test_restore_api(client, dataset_manager):
    from scview.core.obs_writes import uns_snapshot, write_derived
    from scview.core.uns_keys import enrichment_key, markers_key
    from tests.test_hvg_sensitivity import _register

    a = _clustered()
    _markers(a, "t-test")
    a.uns[enrichment_key("leiden", "0")] = '{"terms": []}'
    did = _register(dataset_manager, a, "rs")
    adaptor = await dataset_manager.get_or_load_dataset(did)

    _markers(adaptor.adata, "wilcoxon")  # a re-run, saved
    assert dataset_manager.persist_uns(did, adaptor, [markers_key("leiden")])
    results_store.flush()
    r = await client.get(f"/api/v1/datasets/{did}/results?kind=markers")
    [old] = r.json()["results"]
    assert old["label"].endswith("(t-test)") and old["stale"] is False

    assert (await client.post(f"/api/v1/datasets/{did}/results/nope/restore")).status_code == 404
    r = await client.post(f"/api/v1/datasets/{did}/results/{old['id']}/restore")
    assert r.status_code == 200, r.text
    results_store.flush()
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.uns[markers_key("leiden")]["params"]["method"] == "t-test"
    assert enrichment_key("leiden", "0") not in adaptor.adata.uns  # computed from other markers
    rows = (await client.get(f"/api/v1/datasets/{did}/results")).json()["results"]
    assert {(e["kind"], e["label"]) for e in rows} == {("markers", "Markers of 'leiden' (wilcoxon)"),
                                                        ("enrichment", "Enrichment of 'leiden' = 0")}
    dataset_manager.evict(did)  # from disk: the restore was saved
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.uns[markers_key("leiden")]["params"]["method"] == "t-test"

    wil = next(e for e in rows if e["kind"] == "markers")
    bad = await client.post(f"/api/v1/datasets/{did}/results/"
                            f"{next(e for e in rows if e['kind'] == 'enrichment')['id']}/restore")
    assert bad.status_code == 200  # enrichment comes back too
    adata = adaptor.adata
    before = uns_snapshot(adata)
    adata.obs["leiden"] = pd.Categorical(np.tile(["0", "1", "2"], 20))  # re-clustered
    write_derived(adata, adaptor.h5ad_path, adaptor.h5ad_path, before)
    r = await client.post(f"/api/v1/datasets/{did}/results/{wil['id']}/restore")
    assert r.status_code == 409 and "earlier groups" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Phase 3: exports, pin, compare (core/results_export.py)
# ---------------------------------------------------------------------------

def _tsv(data: bytes) -> pd.DataFrame:
    import gzip
    import io
    return pd.read_csv(io.StringIO(gzip.decompress(data).decode()), sep="\t")


def test_tsv_formats_p_values_and_floats():
    from scview.core import results_export as rx
    df = pd.DataFrame({"gene": ["A", "B"], "logfoldchange": np.array([0.1, 2.5], dtype=np.float32),
                       "pval_adj": [1e-300, np.nan]})
    text = __import__("gzip").decompress(rx.tsv_gz(df)).decode().splitlines()
    assert text[0] == "gene\tlogfoldchange\tpval_adj"
    assert text[1] == "A\t0.1\t1.000e-300" and text[2] == "B\t2.5\t"  # no float32 noise, NaN empty


def test_tables_per_kind(tmp_path):
    from scview.core import results_export as rx
    a = _clustered()
    m = _markers(a, "t-test")
    t = rx.table({"kind": "markers"}, {"uns_key": "k", "value": m})
    assert list(t.columns[:4]) == ["group", "rank", "gene", "scores"] and len(t) == 3 * 30
    assert t[t.group == "0"].gene.iloc[0] in {f"g{j}" for j in range(5)}

    genes = [_Gene("CD14", 2.0, 1e-200, 1e-198, 30.5)]
    rid = de_results.remember("ds8", genes, comparison={"mode": "groups", "group_a": "a", "group_b": "b"})
    t = rx.table({"kind": "de"}, de_results.recall("ds8", rid))
    assert list(t.columns) == ["gene", "logfoldchange", "pval", "pval_adj", "stat"]

    t = rx.table({"kind": "enrichment"}, {"value": '[{"term": "T1", "adjusted_pvalue": 0.01, "genes": ["A", "B"]}]'})
    assert t.genes.iloc[0] == "A;B"

    obs = _obs(T2_SAMPLES, T2)
    t = rx.table({"kind": "composition"}, composition.composition_test(obs, "cond", "donor", "group"))
    assert "group" in t.columns and len(t) == 5 and {"props_A", "props_B"} <= set(t.columns)

    t = rx.table({"kind": "cytokine"}, {"result": {"groups": ["g1"], "cytokines": ["IFN1", "IL6"],
                                                    "z": [[3.0, 0.1]], "p": [[0.001, 0.9]], "coef": [[1, 0]]}})
    assert list(t.cytokine) == ["IFN1", "IL6"]
    t = rx.table({"kind": "planner"}, {"kind": "de", "needed": [{"log2fc": 1, "donors_needed": 4}] * 3})
    assert len(t) == 3  # fallback: the longest list of records
    assert rx.table({"kind": "x"}, {"a": 1}) is None


def test_results_zip_and_manifest(tmp_path):
    import zipfile
    from scview.core import results_export as rx
    _stored("ds9", "ppi", [{"label": "q", "n_links": 3}], label="Protein interactions")
    _stored("ds9", "x", {"a": 1}, label="no table")
    out = tmp_path / "r.zip"
    assert rx.results_zip("ds9", out) == 2
    with zipfile.ZipFile(out) as z:
        manifest = json.loads(z.read("manifest.json"))
        files = [r["file"] for r in manifest["results"]]
        assert sum(f is not None for f in files) == 1
        member = next(f for f in files if f)
        assert member.startswith("results/Protein interactions_") and _tsv(z.read(member)).n_links[0] == 3


def test_pinned_survives_retention(tmp_path):
    first = _stored("ds10", "markers", {"i": 0}, key="leiden")[0]
    assert results_store.set_pinned("ds10", first["id"], True)
    for i in range(1, 7):
        _stored("ds10", "markers", {"i": i}, key="leiden")
    rows = results_store.entries("ds10")
    assert first["id"] in {e["id"] for e in rows}
    assert next(e for e in rows if e["id"] == first["id"])["status"] == "pinned"
    assert results_store.set_pinned("ds10", first["id"], False)
    assert first["id"] not in {e["id"] for e in results_store.entries("ds10")}  # the oldest: let go
    assert results_store.set_pinned("ds10", "nope", True) is False


def test_compare_markers_and_enrichment():
    from scview.core import results_export as rx
    a = _clustered()
    old = dict(_markers(a, "t-test"))
    cur = dict(_markers(a, "wilcoxon"))
    out = rx.compare_markers(old, cur, n=5)
    g0 = next(g for g in out["groups"] if g["group"] == "0")
    assert out["earlier"]["method"] == "t-test" and out["current"]["method"] == "wilcoxon"
    assert g0["shared"] >= 4  # both find the planted g0..g4
    e = rx.compare_enrichment('[{"term": "A", "adjusted_pvalue": 0.01}, {"term": "B", "adjusted_pvalue": 0.02}]',
                              '[{"term": "A", "adjusted_pvalue": 0.001}, {"term": "C", "adjusted_pvalue": 0.5}]')
    assert e["shared"] == 1 and e["only_earlier"] == ["B"] and e["n_sig_current"] == 1


@pytest.mark.asyncio
async def test_phase3_api(client, dataset_manager):
    import zipfile
    from scview.core.uns_keys import markers_key
    from tests.test_hvg_sensitivity import _register

    a = _clustered()
    _markers(a, "t-test")
    did = _register(dataset_manager, a, "p3")
    adaptor = await dataset_manager.get_or_load_dataset(did)
    _markers(adaptor.adata, "wilcoxon")
    dataset_manager.persist_uns(did, adaptor, [markers_key("leiden")])
    results_store.flush()
    [old] = (await client.get(f"/api/v1/datasets/{did}/results")).json()["results"]

    r = await client.get(f"/api/v1/datasets/{did}/results/{old['id']}/table.tsv.gz")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert len(_tsv(r.content)) == 90
    r = await client.get(f"/api/v1/datasets/{did}/results/{old['id']}/compare?n=5")
    assert r.status_code == 200 and r.json()["earlier"]["method"] == "t-test"
    r = await client.post(f"/api/v1/datasets/{did}/results/{old['id']}/pin", json={"pinned": True})
    assert r.status_code == 200
    rows = (await client.get(f"/api/v1/datasets/{did}/results")).json()["results"]
    assert rows[0]["status"] == "pinned"

    r = await client.get(f"/api/v1/datasets/{did}/results.zip")
    with zipfile.ZipFile(__import__("io").BytesIO(r.content)) as z:
        assert "manifest.json" in z.namelist() and len(z.namelist()) == 2
    r = await client.get(f"/api/v1/datasets/{did}/export/h5ad?with_results=true")
    assert r.status_code == 200
    with zipfile.ZipFile(__import__("io").BytesIO(r.content)) as z:
        names = z.namelist()
        assert any(n.endswith(".h5ad") for n in names) and "manifest.json" in names
