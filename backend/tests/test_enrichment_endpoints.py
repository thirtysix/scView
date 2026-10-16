"""The enrichment groups list and local (MSigDB) enrichment endpoints.

Runs gseapy offline against a two-term MSigDB collection written to a temp
directory, so the hypergeometric test, the cache and its "already computed"
badge are exercised end to end without the real MSigDB files.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.config import Settings
from scview.core.uns_keys import markers_key

API = "/api/v1"
GENES = [f"G{i}" for i in range(60)]
# T cells up-regulate G0-G9, B cells G10-G19 (see _register).
T_SET = GENES[0:10]
B_SET = GENES[10:20]


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_T_PROGRAM": {"geneSymbols": T_SET},
        "HALLMARK_B_PROGRAM": {"geneSymbols": B_SET},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    """conftest's settings plus the temp MSigDB directory."""
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


def _register(dm, did: str = "enr", labels: tuple[str, str, str] = ("T", "B", "NK"),
              column: str = "cell_type", markers: bool = True) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, len(GENES))).astype("float32")
    X[:30, 0:10] += 8
    X[30:60, 10:20] += 8
    a = ad.AnnData(X)
    a.var_names = GENES
    a.obs_names = [f"C{i}" for i in range(90)]
    a.obs[column] = pd.Categorical(
        [labels[0]] * 30 + [labels[1]] * 30 + [labels[2]] * 30, categories=list(labels)
    )
    if markers:
        sc.tl.rank_genes_groups(a, column, method="wilcoxon", pts=True,
                                key_added=markers_key(column))
    a.write_h5ad(d / "data.h5ad")
    return did


def _local(column: str = "cell_type", group: str = "T", **kw) -> dict:
    return {"column": column, "group": group, "n_genes": 10, "collections": ["h.all"], **kw}


# ---------------------------------------------------------------------------
# GET /enrichment/groups
# ---------------------------------------------------------------------------


async def test_groups_lists_marker_groups_none_computed_yet(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/enrichment/groups", params={"column": "cell_type"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "cell_type"
    assert set(body["groups"]) == {"T", "B", "NK"}
    assert body["enrichment_computed"] == {"T": False, "B": False, "NK": False}


async def test_groups_marks_a_group_computed_after_compute_local(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    assert r.status_code == 200, r.text
    r = await client.get(f"{API}/datasets/{did}/enrichment/groups", params={"column": "cell_type"})
    assert r.json()["enrichment_computed"] == {"T": True, "B": False, "NK": False}


async def test_groups_finds_markers_under_the_bare_key(client, dataset_manager):
    # scanpy's default key with params.groupby naming the column, as files often ship.
    did = _register(dataset_manager, markers=False)
    path = Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad"
    a = ad.read_h5ad(path)
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon")
    a.write_h5ad(path)
    r = await client.get(f"{API}/datasets/{did}/enrichment/groups", params={"column": "cell_type"})
    assert r.status_code == 200, r.text
    assert set(r.json()["groups"]) == {"T", "B", "NK"}


async def test_groups_column_without_markers_is_404(client, dataset_manager):
    did = _register(dataset_manager, markers=False)
    r = await client.get(f"{API}/datasets/{did}/enrichment/groups", params={"column": "cell_type"})
    assert r.status_code == 404


async def test_groups_missing_dataset_is_404(client, dataset_manager):
    r = await client.get(f"{API}/datasets/ghost/enrichment/groups", params={"column": "x"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /enrichment/compute-local
# ---------------------------------------------------------------------------


async def test_compute_local_finds_the_group_program(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "msigdb_local"
    assert body["group"] == "T" and body["groupby"] == "cell_type"
    assert body["n_genes_used"] == 10
    top = body["results"][0]
    assert top["term"] == "HALLMARK_T_PROGRAM"
    assert top["collection"] == "h.all"
    assert top["adjusted_pvalue"] < 0.01
    assert set(top["genes"]) <= set(T_SET)


async def test_compute_local_second_call_is_served_from_cache(client, dataset_manager):
    did = _register(dataset_manager)
    first = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    again = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    assert again.status_code == 200, again.text
    assert again.json()["source"] == "precomputed"
    assert [r["term"] for r in again.json()["results"]] == [
        r["term"] for r in first.json()["results"]
    ]


async def test_compute_local_other_params_recompute(client, dataset_manager):
    did = _register(dataset_manager)
    await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local",
                          json=_local(n_genes=20))
    assert r.json()["source"] == "msigdb_local"
    assert r.json()["n_genes_used"] == 20


async def test_compute_local_unknown_group_is_404(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local",
                          json=_local(group="nope"))
    assert r.status_code == 404


async def test_compute_local_column_without_markers_is_404(client, dataset_manager):
    did = _register(dataset_manager, markers=False)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    assert r.status_code == 404


async def test_compute_local_collection_not_on_disk_is_400(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local",
                          json=_local(collections=["c8.all"]))
    assert r.status_code == 400


async def test_compute_local_without_msigdb_dir_is_500(client, dataset_manager, test_settings):
    did = _register(dataset_manager)
    test_settings.MSIGDB_DIR = ""
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local())
    assert r.status_code == 500
    assert "MSigDB" in r.json()["detail"]


# ---------------------------------------------------------------------------
# "/" in a group name (CellTypist labels)
# ---------------------------------------------------------------------------

TCM = "Tcm/Naive helper T cells"


async def test_slash_label_enrichment_survives_a_write_and_reload(client, dataset_manager):
    # h5py read "/" as a path separator, so this result came back nested under
    # "enrichment__cell_type__Tcm" and was never found again.
    did = _register(dataset_manager, labels=(TCM, "B", "NK"))
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local(group=TCM))
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "msigdb_local"
    assert r.json()["results"][0]["term"] == "HALLMARK_T_PROGRAM"

    # Renaming another label writes the derived file and reloads it.
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename",
                          json={"old": "NK", "new": "NK cells"})
    assert r.status_code == 200, r.text
    assert dataset_manager.get_dataset(did).h5ad_path.split("/")[-3] == "derived"

    r = await client.get(f"{API}/datasets/{did}/enrichment/groups", params={"column": "cell_type"})
    assert r.json()["enrichment_computed"][TCM] is True
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local(group=TCM))
    assert r.json()["source"] == "precomputed"
    assert r.json()["results"][0]["term"] == "HALLMARK_T_PROGRAM"


async def test_slash_label_rename_carries_its_cached_enrichment(client, dataset_manager):
    did = _register(dataset_manager, labels=(TCM, "B", "NK"))
    await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=_local(group=TCM))
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename",
                          json={"old": TCM, "new": "CD4 T/naive"})
    assert r.status_code == 200, r.text
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local",
                          json=_local(group="CD4 T/naive"))
    assert r.json()["source"] == "precomputed"
