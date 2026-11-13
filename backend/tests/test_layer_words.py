"""Captions and colour bars say what a layer's values are, not its key: after a run
``.raw`` holds log-normalised values ("Log-normalized (raw)"), in a CELLxGENE file raw
counts ("Counts (raw)"). Emont 2022 QA: the dot plot said "mean (raw)" over log values."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp
from mcp import Client

from scview.mcp import plots
from scview.mcp.server import SERVER

API = "/api/v1"
GENES = ["CD3E", "MS4A1", "LYZ", "NKG7"]


def _register(dm, did: str, raw_counts: bool) -> str:
    """60 cells; .raw holds counts (``raw_counts``) or log-normalised values, X log values."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(60, len(GENES))).astype(np.float32)
    logn = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(logn))
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(60)]
    a.obs["cluster"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
    full = ad.AnnData(sp.csr_matrix(counts if raw_counts else logn))
    full.var_names = GENES
    a.raw = full
    a.obsm["X_umap"] = rng.normal(size=(60, 2)).astype(np.float32)
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def _tool(name: str, args: dict):
    async with Client(SERVER) as c:
        r = await c.call_tool(name, args)
    assert not r.is_error, r.content
    return r


@pytest.mark.parametrize("raw_counts,label", [(False, "Log-normalized (raw)"), (True, "Counts (raw)")])
async def test_by_group_heatmap_and_mcp_say_what_raw_holds(client, dataset_manager, monkeypatch, raw_counts, label):
    did = _register(dataset_manager, f"lw{int(raw_counts)}", raw_counts)
    r = await client.get(f"{API}/datasets/{did}/expression/by-group",
                         params={"genes": ["CD3E"], "groupby": "cluster", "layer": "raw"})
    assert r.status_code == 200 and r.json()["layer_label"] == label
    r = await client.get(f"{API}/datasets/{did}/expression/heatmap-cells",
                         params={"genes": "CD3E", "groupby": "cluster", "layer": "raw"})
    meta = json.loads(pa.ipc.open_stream(r.content).read_all().schema.metadata[b"scview"])
    assert meta["layer"] == "raw" and meta["layer_label"] == label

    r = await _tool("expression_by_group", {"dataset_id": did, "genes": ["CD3E"], "groupby": "cluster",
                                            "layer": "raw"})
    out = json.loads(r.content[0].text)
    assert out["layer"] == "raw" and out["layer_label"] == label

    seen: dict[str, str] = {}
    monkeypatch.setattr(plots, "violin", lambda v, g, title, ylabel, caption: seen.update(y=ylabel) or b"\x89PNG")
    monkeypatch.setattr(plots, "embedding",
                        lambda xy, v, title, label_, note, focus, tissue=False: seen.update(c=label_) or b"\x89PNG")
    await _tool("plot", {"dataset_id": did, "kind": "violin", "gene": "CD3E", "groupby": "cluster", "layer": "raw"})
    await _tool("plot", {"dataset_id": did, "kind": "embedding", "gene": "CD3E", "layer": "raw"})
    assert seen == {"y": f"CD3E, {label}", "c": f"CD3E, {label}"}
    r = await _tool("plot", {"dataset_id": did, "kind": "dotplot", "genes": ["CD3E"], "groupby": "cluster",
                             "layer": "raw"})
    assert any(f"layer {label}" in getattr(b, "text", "") for b in r.content)


def test_dotplot_colour_bar_names_the_values(monkeypatch):
    import matplotlib.figure

    labels: list[str] = []
    real = matplotlib.figure.Figure.colorbar

    def spy(self, *a, **kw):
        labels.append(kw.get("label"))
        return real(self, *a, **kw)

    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", spy)
    plots.dotplot({"genes": ["A"], "groups": ["0"], "mean": [[1.0]], "fraction": [[0.5]],
                   "layer": "raw", "layer_label": "Log-normalized (raw)"}, "t", "c")
    assert labels == ["mean, Log-normalized (raw)"]
