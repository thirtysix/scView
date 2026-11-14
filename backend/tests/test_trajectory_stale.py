"""One staleness verdict for PAGA, a pseudotime and a velocity run (QA round 5: after a
batch correction rebuilt the graph, GET /trajectory, /trajectory/paga and /velocity said
``"stale": null`` while the results list marked the same velocity run stale)."""

from __future__ import annotations

import json

from scview.core import provenance, results_store
from scview.core import velocity_input as vi
from scview.core.trajectory import (
    PAGA_KEY,
    PARAMS_KEY,
    compute_dpt,
    compute_paga,
    graph_stale,
    paga_view,
    public_params,
    read_params,
)
from tests.test_trajectory_dpt import _register
from tests.test_trajectory_paga import _with_graph
from tests.test_velocity import _run, dataset_with_store

API = "/api/v1"


def test_a_graph_step_stales_paga_and_pseudotime_a_clustering_does_not():
    a, _ = _with_graph()
    compute_paga(a, "stage")
    compute_dpt(a, "stage", "early")
    assert paga_view(a)["stale"] is None and graph_stale(a, read_params(a)) is None
    assert "fingerprint" not in public_params(a, read_params(a))

    provenance.record_step(a, step="clustering", tool="leiden", params={})  # labels only
    provenance.record_step(a, step="marker_genes", tool="scanpy", params={})  # reads no graph
    provenance.record_step(a, step="velocity", tool="scview", params={})  # additive
    assert paga_view(a)["stale"] is None and public_params(a, read_params(a))["stale"] is None

    provenance.record_step(a, step="batch_correction", tool="harmony", params={})
    provenance.record_step(a, step="neighbors", tool="scanpy", params={})
    assert paga_view(a)["stale"] == "Batch correction rerun, neighbours rebuilt after this result. Compute PAGA again."
    assert public_params(a, read_params(a))["stale"] == "Batch correction rerun, neighbours rebuilt after this result."

    # The recipe's own pseudotime entry carries no fingerprint.
    assert "fingerprint" not in provenance.read_provenance(a)["history"][-6]["params"]


def test_a_record_from_before_fingerprints_compares_the_graph():
    a, _ = _with_graph()
    compute_paga(a, "stage")
    record = json.loads(a.uns[PAGA_KEY])
    del record["fingerprint"]
    a.uns[PAGA_KEY] = json.dumps(record)
    assert paga_view(a)["stale"] is None
    a.uns["neighbors"]["params"] = {"use_rep": "X_pca_harmony", "n_neighbors": 30}
    assert paga_view(a)["stale"] == ("Computed on the existing neighbour graph (use_rep=X_pca, k=15); the dataset "
                                     "now has the existing neighbour graph (use_rep=X_pca_harmony, k=30). "
                                     "Compute PAGA again.")


async def test_the_listing_reports_stale_runs_and_attached_counts(client, dataset_manager):
    a, _ = _with_graph()
    did, _ = _register(dataset_manager, a)
    for root in ("early", "late"):
        r = await client.post(f"{API}/datasets/{did}/trajectory/dpt", json={"root_column": "stage", "root_value": root})
        assert r.status_code == 200 and "fingerprint" not in r.json()
    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert listed["computed"]["root_value"] == "late" and listed["computed"]["stale"] is None
    assert set(listed["runs"]) == {"scview_dpt_early", "scview_dpt_late"}
    assert {"scview_dpt_early", "scview_dpt_late"} <= set(listed["pseudotime_columns"])
    assert listed["velocity_layers"] is False

    # Counts in the modality store (not in layers) count as attached.
    vi.store_path(dataset_manager.data_dir, did).parent.mkdir(parents=True, exist_ok=True)
    vi.store_path(dataset_manager.data_dir, did).write_bytes(b"")
    assert (await client.get(f"{API}/datasets/{did}/trajectory")).json()["velocity_layers"] is True

    adaptor = await dataset_manager.get_or_load_dataset(did)
    provenance.record_step(adaptor.adata, step="neighbors", tool="scanpy", params={})
    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert listed["computed"]["stale"] == "Neighbours rebuilt after this result."
    assert listed["runs"]["scview_dpt_early"]["stale"] == "Neighbours rebuilt after this result."
    assert PARAMS_KEY in adaptor.adata.uns


async def test_velocity_status_says_what_the_results_list_says(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "velstale")
    await _run(client, did)
    results_store.flush()
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["stale"] is None
    adaptor = await dataset_manager.get_or_load_dataset(did)
    provenance.record_step(adaptor.adata, step="neighbors", tool="scanpy", params={})
    listed = (await client.get(f"{API}/datasets/{did}/results")).json()
    entry = next(e for e in (listed["results"] if isinstance(listed, dict) else listed) if e["kind"] == "velocity")
    assert entry["stale"] is True
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["stale"] == f"{entry['stale_reason'][0].upper()}{entry['stale_reason'][1:]}: run velocity again."
