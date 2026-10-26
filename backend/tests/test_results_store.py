"""The per-dataset results store (core/results_store.py) and the modules that
keep their results in it."""

from __future__ import annotations

import hashlib
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
    rdir = results_store.root() / "ds1" / e["id"]
    doc = (rdir / "result.json").read_text().replace('{"x": 1}', '{"__dc__": "os:PathLike", "f": {}}')
    (rdir / "result.json").write_text(doc)
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
