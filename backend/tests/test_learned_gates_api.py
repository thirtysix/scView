"""Learned gate endpoints (api/v1/learned_gates.py): train (SSE) -> save -> library ->
check -> apply, the refusals, and a client going away mid-training."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from scview.api.v1 import learned_gates as api
from scview.core import learned_gate as lg
from scview.core import provenance, results_store
from tests.test_learned_gate import _cells

API = "/api/v1"


def _register(dm, did: str, seed: int = 0) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"Data {did}", "status": "ready"}))
    _cells(seed=seed).write_h5ad(d / "data.h5ad")
    return did


def _events(text: str) -> list[tuple[str, dict]]:
    out, event = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: ") and event:
            out.append((event, json.loads(line[6:])))
    return out


def _a_ctrl() -> dict:
    a = _cells()
    mask = ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy()
    return {"n_cells": int(a.n_obs), "indices": [int(i) for i in np.flatnonzero(mask)]}


async def _train(client, did: str, **kw) -> dict:
    r = await client.post(f"{API}/datasets/{did}/learned-gates/train-stream",
                          json={"name": "A cells", "cells": _a_ctrl(), **kw})
    assert r.status_code == 200, r.text
    ev = _events(r.text)
    names = [e for e, _ in ev]
    assert names[0] == "start" and names[-1] == "complete", names
    assert names.count("progress") == len(lg.C_GRID)
    return dict(ev)["result"]


async def test_train_save_list_check_apply(client, dataset_manager):
    did = _register(dataset_manager, "lg1")
    res = await _train(client, did)
    gate = res["gate"]
    assert res["saved"] is False and res["report"]["verdict"]["level"] == "good"
    assert gate["source"]["dataset_id"] == did and gate["source"]["name"] == "Data lg1"
    assert res["overlap"]["group"] == "A" and not res["overlap"]["is_cluster"]
    # not in the library until saved
    assert (await client.get(f"{API}/learned-gates")).json()["gates"] == []
    r = await client.post(f"{API}/learned-gates/{gate['id']}/save")
    assert r.status_code == 200, r.text
    lib = (await client.get(f"{API}/learned-gates")).json()["gates"]
    assert [g["id"] for g in lib] == [gate["id"]] and lib[0]["verdict"]["level"] == "good"
    r = await client.get(f"{API}/learned-gates/{gate['id']}")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert json.loads(r.text)["format"] == lg.FORMAT
    # the training is recorded on the source, its selection kept in the results store
    adaptor = await dataset_manager.get_or_load_dataset(did)
    steps = [h["step"] for h in provenance.read_provenance(adaptor.adata)["history"]]
    assert steps[-1] == lg.STEP_TRAIN
    results_store.flush()
    kept = results_store.entries(did, lg.TRAIN_KIND)
    assert kept and kept[0]["key"] == gate["id"]

    other = _register(dataset_manager, "lg2", seed=3)
    r = await client.post(f"{API}/datasets/{other}/learned-gates/{gate['id']}/check")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["ok"] and rep["coverage"]["gene_share"] == 1.0 and rep["sample_column"] == "donor"
    assert rep["columns"] == {"column": "gate_a_cells", "score_column": "gate_a_cells_score"}
    r = await client.post(f"{API}/datasets/{other}/learned-gates/{gate['id']}/apply", json={})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["column"] == "gate_a_cells" and len(out["per_sample"]) == 4 and not out["same_dataset"]
    adaptor = await dataset_manager.get_or_load_dataset(other)
    assert "gate_a_cells" in adaptor.adata.obs and "gate_a_cells_score" in adaptor.adata.obs
    truth = (adaptor.adata.obs["cluster"] == "A").to_numpy()
    pred = (adaptor.adata.obs["gate_a_cells"] == "in").to_numpy()
    assert (pred & truth).sum() / truth.sum() > 0.9
    results_store.flush()
    assert results_store.latest(adaptor, lg.KIND)["n_in"] == out["n_in"]
    lines, _ = lg.grounding_lines(adaptor)
    assert lines and lg.TAG in lines[0]
    # the co-pilot cites it, and the claims check resolves the tag the model writes
    from scview.core.claims import resolve_tag

    r = await client.post(f"{API}/datasets/{other}/assistant/chat",
                          json={"query": "Which cell groups and clusters does this dataset have?"})
    assert r.status_code == 200, r.text
    assert lg.TAG in [s["ref"] for s in r.json()["sources"]]
    assert resolve_tag("lgate", [lg.TAG]) == [lg.TAG]
    assert resolve_tag("result:learned_gate", [lg.TAG]) == [lg.TAG]
    from scview.core import feature_guide

    assert "Gates > Learned gates" in feature_guide.compact_guide()
    # applying again replaces the gate's own columns
    r = await client.post(f"{API}/datasets/{other}/learned-gates/{gate['id']}/apply")
    assert r.status_code == 200 and r.json()["replaced"]
    r = await client.post(f"{API}/datasets/{other}/learned-gates/{gate['id']}/check")
    assert r.json()["replaces"]
    # export, delete, import back
    text = (await client.get(f"{API}/learned-gates/{gate['id']}")).text
    assert (await client.delete(f"{API}/learned-gates/{gate['id']}")).status_code == 200
    r = await client.post(f"{API}/learned-gates/import", content=text)
    assert r.status_code == 200 and r.json()["id"] == gate["id"]


async def test_refusals(client, dataset_manager):
    did = _register(dataset_manager, "lg3")
    url = f"{API}/datasets/{did}/learned-gates/train-stream"
    tiny = {"n_cells": 960, "indices": list(range(5))}
    r = await client.post(url, json={"name": "x", "cells": tiny})
    assert r.status_code == 422 and "at least 20" in r.json()["detail"]
    r = await client.post(url, json={"name": "x", "cells": {"n_cells": 10, "indices": [1]}})
    assert r.status_code == 409
    r = await client.post(url, json={"name": "x", "cells": _a_ctrl(), "sample_column": "nope"})
    assert r.status_code == 404
    r = await client.post(url, json={"name": "", "cells": _a_ctrl()})
    assert r.status_code == 422
    r = await client.post(f"{API}/datasets/missing/learned-gates/train-stream", json={"name": "x", "cells": tiny})
    assert r.status_code == 404
    api._running.add(did)
    try:
        assert (await client.post(url, json={"name": "x", "cells": _a_ctrl()})).status_code == 409
    finally:
        api._running.discard(did)
    assert (await client.post(f"{API}/learned-gates/nope/save")).status_code == 404
    assert (await client.get(f"{API}/learned-gates/nope")).status_code == 404
    assert (await client.delete(f"{API}/learned-gates/nope")).status_code == 404
    assert (await client.post(f"{API}/datasets/{did}/learned-gates/nope/check")).status_code == 404
    r = await client.post(f"{API}/learned-gates/import", content=b'{"format": "other"}')
    assert r.status_code == 422
    r = await client.post(f"{API}/learned-gates/import", content=b" " * (lg.MAX_FILE_BYTES + 1))
    assert r.status_code == 413
    # a column of the gate's name that no gate wrote: check says so, apply refuses
    res = await _train(client, did)
    gid = res["gate"]["id"]
    assert (await client.post(f"{API}/learned-gates/{gid}/save")).status_code == 200
    adaptor = await dataset_manager.get_or_load_dataset(did)
    adaptor.adata.obs["gate_a_cells"] = "mine"
    rep = (await client.post(f"{API}/datasets/{did}/learned-gates/{gid}/check")).json()
    assert not rep["ok"] and "no learned gate wrote" in rep["refusal"]
    r = await client.post(f"{API}/datasets/{did}/learned-gates/{gid}/apply")
    assert r.status_code == 409
    r = await client.post(f"{API}/datasets/{did}/learned-gates/{gid}/apply", json={"sample_column": "nope"})
    assert r.status_code == 404


async def test_a_client_going_away_stops_training(dataset_manager, test_settings, monkeypatch):
    did = _register(dataset_manager, "lg4")
    seen = {"stop": None, "steps": 0}

    def slow(inp, *, name, stop):
        seen["stop"] = stop
        while not stop():
            seen["steps"] += 1
            yield {"type": "progress", "step": "fit", "done": seen["steps"], "total": 6}
        raise lg.LearnedGateCancelled()

    monkeypatch.setattr(lg, "iter_train", slow)
    body = api.TrainRequest(name="A cells", cells=_a_ctrl())
    resp = await api.train_stream(did, body, dm=dataset_manager, settings=test_settings)
    it = resp.body_iterator
    first = await it.__anext__()
    assert "event: start" in first and did in api._running
    second = await it.__anext__()
    assert "event: progress" in second
    await it.aclose()  # the client went away
    for _ in range(50):
        if did not in api._running:
            break
        await asyncio.sleep(0.02)
    assert seen["stop"]() is True and did not in api._running
    assert lg.list_gates(dataset_manager.data_dir) == []
