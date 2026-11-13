"""plot on tissue sections and get_dataset's spatial fields (MCP), called through the SDK's client
in process as test_mcp.py does."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scview.mcp import plots, tools
from tests.test_hvg_sensitivity import _register
from tests.test_mcp import PNG, _call, _png_size, msigdb_dir, test_settings  # noqa: F401 — fixtures
from tests.test_spatial_domains import synthetic


def _two_sections(n_side: int = 12, tech: str = "visium"):
    """Two sections at the same pixel positions (as two slides are), random expression."""
    a = synthetic.tissue(tech, n_side=n_side, sections=2)
    rng = np.random.default_rng(1)
    x = rng.poisson(1.0, size=(a.n_obs, 3)).astype(np.float32)
    x[:, 0] += (a.obs["stripe"].cat.codes.to_numpy() * 3).astype(np.float32)  # G0 follows the stripes
    a.X = np.log1p(x)
    return a


def _text(content) -> str:
    return next(b.text for b in content if b.type == "text")


def _png(content) -> bytes:
    data = base64.b64decode(next(b for b in content if b.type == "image").data)
    assert data.startswith(PNG) and len(data) < 300_000 and max(_png_size(data)) <= 1600
    return data


# --- get_dataset and plot ----------------------------------------------------------------------

async def test_get_dataset_lists_spatial_its_sections_and_the_off_tissue_record(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(), "gsp")
    meta = Path(dataset_manager.data_dir) / "uploads" / did / "metadata.json"
    rec = json.loads(meta.read_text())
    rec["off_tissue"] = {"removed": 713, "kept": 4279, "column": "in_tissue"}
    rec["public_source"] = {"about_spatial_note": "Discover lists a spatial assay (Visium), but no positions"}
    meta.write_text(json.dumps(rec))
    d = await _call("get_dataset", {"dataset_id": did})
    assert "spatial" in d["available_embeddings"] and "spatial" in d["modalities"]
    assert d["spatial"]["n_sections"] == 2 and d["spatial"]["sections"] == [["s0", 144], ["s1", 144]]
    assert d["spatial"]["unit"] == "spot" and d["spatial"]["library_column"] == "library_id"
    assert d["off_tissue"] == {"removed": 713, "kept": 4279, "column": "in_tissue"}
    assert "no positions" in d["spatial_note"]


async def test_plot_draws_sections_side_by_side_or_one_alone(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(), "psec")
    both = await _call("plot", {"dataset_id": did, "embedding": "spatial", "color_by": "stripe"})
    _png(both)
    assert "288 spots; 2 of 2 sections (library_id)" in _text(both)
    one = await _call("plot", {"dataset_id": did, "embedding": "spatial", "gene": "G0", "section": "s1"})
    _png(one)
    assert "144 spots" in _text(one) and "section s1, one of 2" in _text(one)  # the other's spots left out
    # A subgroup within the sections drawn: greyed within them, the other section still left out.
    sub = await _call("plot", {"dataset_id": did, "embedding": "spatial", "color_by": "stripe", "section": "s0",
                               "within_column": "stripe", "within_value": "t0"})
    assert "144 spots" in _text(sub) and "drawn with stripe = t0" in _text(sub)
    by_col = await _call("plot", {"dataset_id": did, "embedding": "spatial", "section_column": "stripe",
                                  "section": ["t0", "t1"]})
    assert "2 of 4 sections (stripe)" in _text(by_col)
    err = await _call("plot", {"dataset_id": did, "embedding": "spatial", "section": "s9"}, error=True)
    assert "No section 's9' in library_id; its sections: s0, s1" in err
    err = await _call("plot", {"dataset_id": did, "embedding": "X_tsne"}, error=True)
    assert "spatial" in err


def test_the_sections_figure_caps_its_panels_and_says_so(monkeypatch):
    class Adaptor:
        def __init__(self, obs):
            self.adata = type("A", (), {"obs": obs})()

    libs = pd.Categorical([f"s{i}" for i in range(15) for _ in range(3)], categories=[f"s{i}" for i in range(15)])
    labels, order, note = tools._sections(Adaptor(pd.DataFrame({"library_id": libs})), None, None)
    assert order == [f"s{i}" for i in range(plots.MAX_PANELS)]
    assert note == f"; {plots.MAX_PANELS} of 15 sections (library_id); pass section for the others"
    xy = np.random.default_rng(0).uniform(0, 100, (45, 2))
    png = plots.sections(xy, pd.Series(np.arange(45) % 4).astype(str), labels, order, "t", "c", note)
    assert png.startswith(PNG)
    png = plots.sections(xy, np.arange(45, dtype=float), labels, order[:2], "t", "c", note,
                         focus=np.arange(45) % 2 == 0)
    assert png.startswith(PNG)
    one = pd.DataFrame({"library_id": pd.Categorical(["a"] * 4)})
    assert tools._sections(Adaptor(one), None, None) == (None, [], "")
