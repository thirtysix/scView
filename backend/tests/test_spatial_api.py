"""Spatial API (api/v1/spatial.py, GET /datasets/{id} ``spatial``) and the co-pilot's
fact about spatial data."""

from __future__ import annotations

import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
from PIL import Image

from scview.core import spatial
from tests.test_spatial import _squidpy_like

API = "/api/v1"


# ---------------------------------------------------------------------------
# API and the co-pilot's fact
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": "visium", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_dataset_info_and_image_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _squidpy_like(), "vis")
    r = await client.get(f"{API}/datasets/{did}")
    assert r.status_code == 200, r.text
    s = r.json()["spatial"]
    assert s["technology"] == "visium" and s["unit"] == "spot" and s["library_column"] == "library_id"
    (lib,) = s["libraries"]
    assert lib == {"id": "V1_LN", "n_obs": 12, "spot_diameter_fullres": 50.0,
                   "images": {"lowres": {"width": 7, "height": 5, "scalef": 0.1}},
                   "bounds": [0.0, 0.0, 110.0, 55.0]}
    assert "spatial" not in r.json()["available_embeddings"]

    r = await client.get(f"{API}/datasets/{did}/spatial/V1_LN/image", params={"res": "lowres"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "private" in r.headers["cache-control"]
    img = np.asarray(Image.open(io.BytesIO(r.content)))
    assert img.shape == (5, 7, 3) and int(img[0, 0, 0]) == 200
    etag = r.headers["etag"]
    r = await client.get(f"{API}/datasets/{did}/spatial/V1_LN/image", headers={"If-None-Match": etag})
    assert r.status_code == 304
    r = await client.get(f"{API}/datasets/{did}/spatial/V1_LN/image", params={"res": "hires"})
    assert r.status_code == 404 and "hires" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/spatial/nope/image")
    assert r.status_code == 404
    r = await client.get(f"{API}/datasets/{did}/embeddings/spatial")
    assert r.status_code == 200


async def test_plain_dataset_has_no_spatial(client, dataset_manager):
    a = _squidpy_like(lib=None)
    del a.obsm["spatial"]
    did = _register(dataset_manager, a, "plain")
    r = await client.get(f"{API}/datasets/{did}")
    assert r.json()["spatial"] is None
    r = await client.get(f"{API}/datasets/{did}/spatial/x/image")
    assert r.status_code == 404


def test_copilot_fact():
    a = _squidpy_like()
    spatial.adopt(a)
    lines, detail = spatial.grounding_lines(a)
    assert lines[0].startswith("- Spatial: Visium, 1 section, 12 spots (V1_LN 12)")
    assert "doublet detection does not apply" in lines[0] and lines[0].endswith(f"[{spatial.TAG}]")
    from scview.core.claims import _FAMILY_PREFIX

    assert _FAMILY_PREFIX["tissue"] == spatial.TAG
