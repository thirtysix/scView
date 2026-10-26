"""The Perturbation endpoints: guide assignment saved, Mixscape streamed and saved."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scview.core import crispr, mixscape, provenance, results_store
from tests.test_hvg_sensitivity import _events, _register
from tests.test_mixscape import planted

API = "/api/v1"


def _with_guides():
    """The planted screen with guide counts that say each cell's guide."""
    a, ko = planted()
    labels = a.obs["perturbation"].astype(str).to_numpy()
    del a.obs["perturbation"]
    guides = ["Ag1", "Ag2", "Bg1", "NTg1", "NTg2"]
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 3, (a.n_obs, len(guides))).astype(float)
    for i, lab in enumerate(labels):
        g = {"A": "Ag1" if i % 2 else "Ag2", "B": "Bg1", "NT": "NTg1" if i % 2 else "NTg2"}[lab]
        counts[i, guides.index(g)] = 30
    counts[5, guides.index("Bg1")] = 30  # an NT cell with a B guide too: "multiple"
    crispr.attach_counts(a, counts, guides)
    return a, labels, ko


async def test_assign_then_mixscape_saves_and_is_served(client, dataset_manager):
    a, labels, ko = _with_guides()
    did = _register(dataset_manager, a, "ps1")
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert ds["perturbation"] == {"n_guides": 5, "cellranger": False, "assigned": False, "columns": []}
    info = (await client.get(f"{API}/datasets/{did}/perturbation")).json()
    assert info["suggested"]["source"] == "counts" and info["call"] is None and info["mixscape"] is None

    prev = (await client.post(f"{API}/datasets/{did}/perturbation/preview", json={})).json()
    assert prev["control"] == "NT" and prev["targets"] == ["A", "B"] and prev["counts"]["multiple"] == 1
    r = await client.post(f"{API}/datasets/{did}/perturbation/assign", json={})
    assert r.status_code == 200, r.text
    assert r.json()["call"]["counts"] == {"assigned": a.n_obs - 1, "control": 149, "multiple": 1, "none": 0}

    r = await client.post(f"{API}/datasets/{did}/perturbation/mixscape/run-stream", json={"split_by": "rep"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "progress" in kinds
    start = events[0][1]
    assert start["labels"] == "perturbation" and start["control"] == "NT" and start["n_cells"] == a.n_obs - 1
    result = events[-2][1]
    by = {t["target"]: t for t in result["targets"]}
    assert by["A"]["n_ko"] == int(ko.sum()) and by["B"]["n_ko"] == 0

    # Saved: the reopened dataset has the columns, the record and the history step.
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    obs = adaptor.adata.obs
    assert obs["guide_call"].astype(str).iloc[151] == "Ag1" and obs["n_guides"].iloc[5] == 2
    assert obs["perturbation"].astype(str).iloc[5] == "multiple"
    assert pd.isna(obs[mixscape.CLASS].iloc[5])
    assert (obs[mixscape.GLOBAL].astype(str).to_numpy()[ko] == "KO").all()
    steps = [h["step"] for h in provenance.read_provenance(adaptor.adata)["history"]]
    assert steps[-2:] == ["crispr_assign", "mixscape"]

    # After a reload the last run comes from the results store.
    results_store.flush()
    got = (await client.get(f"{API}/datasets/{did}/perturbation/mixscape")).json()
    assert got["n_targets"] == 2 and got["targets"][0]["posterior_hist"]["edges"][-1] == 1.0
    info = (await client.get(f"{API}/datasets/{did}/perturbation")).json()
    assert info["call"]["control"] == "NT" and info["mixscape"]["summary"]["n_ko"] == int(ko.sum())
    entries = results_store.entries(did, "mixscape")
    assert entries and entries[0]["label"] == "Mixscape: 2 targets vs NT"

    from scview.core.assistant import build_grounding_context
    from scview.core.claims import resolve_tag

    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "CRISPR screen" in context and "KO fraction per target: A" in context
    assert any(s.ref == mixscape.TAG for s in sources)
    assert resolve_tag("mixscape", [mixscape.TAG, "result:milo"]) == [mixscape.TAG]
    assert resolve_tag("perturbation", [mixscape.TAG]) == [mixscape.TAG]


async def test_refusals(client, dataset_manager):
    a, _ = planted()
    del a.obsm["X_pca"]
    did = _register(dataset_manager, a, "ps2")
    # A guide-like column is found; nothing is assigned yet.
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert ds["perturbation"]["columns"] == ["perturbation"]
    r = await client.post(f"{API}/datasets/{did}/perturbation/mixscape/run-stream", json={})
    assert r.status_code == 400 and "Assign the guides" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/perturbation/mixscape/run-stream",
                          json={"labels": "perturbation", "control": "NT"})
    assert r.status_code == 422 and "PCA" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/perturbation/assign", json={"source": "counts"})
    assert r.status_code == 400 and "guide counts" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/perturbation/assign", json={"control": "zzz"})
    assert r.status_code == 400
    assert (await client.get(f"{API}/datasets/{did}/perturbation/mixscape")).status_code == 404
    plain = _register(dataset_manager, planted()[0][:, :5].copy(), "ps3")
    p = (await client.get(f"{API}/datasets/{plain}")).json()["perturbation"]
    assert p["columns"] == ["perturbation"]


async def test_assign_from_a_column(client, dataset_manager):
    a, _ = planted()
    a.obs["gene"] = a.obs["perturbation"].astype(str).replace({"NT": "non-targeting"})
    did = _register(dataset_manager, a, "ps4")
    r = await client.post(f"{API}/datasets/{did}/perturbation/assign",
                          json={"source": "column", "column": "gene"})
    assert r.status_code == 200, r.text
    call = r.json()["call"]
    # The dataset's own 'perturbation' column is kept; scView's goes beside it.
    assert call["control"] == "non-targeting" and call["columns"]["perturbation"] == "perturbation_scview"
    r = await client.post(f"{API}/datasets/{did}/perturbation/mixscape/run-stream", json={"lda": False})
    kinds = [e for e, _ in _events(r.text)]
    assert kinds[-2:] == ["result", "complete"]
