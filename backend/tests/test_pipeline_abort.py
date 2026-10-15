"""A pipeline run that stops part-way must not leave unsaved steps in the cache.

Steps change the cached AnnData in place and the file is written only at the
end. A stream abandoned mid-run (tab closed, proxy timeout) used to leave the
cache log-normalised while the disk still held counts, so re-running the same
steps log-transformed twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from scview.api.v1.assessment import RunPipelineRequest, run_assessment_pipeline_stream
from scview.core.dataset_manager import DatasetManager


def _counts_dataset(data_dir: Path) -> None:
    d = data_dir / "uploads" / "ds"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (200, 100)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(100)]
    a.write_h5ad(d / "data.h5ad")


async def test_abandoned_stream_drops_the_half_run_cache(tmp_path):
    _counts_dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))
    await dm.load_dataset("ds")
    resp = await run_assessment_pipeline_stream(
        "ds", RunPipelineRequest(steps=["normalization", "log_transform", "pca"]), dm
    )
    it = resp.body_iterator
    seen = False
    async for chunk in it:
        if "step_done" in chunk and '"log_transform"' in chunk:
            seen = True
            break  # the client goes away here
    await it.aclose()
    assert seen

    assert dm.get_dataset("ds") is None  # evicted, so the next access reads the disk
    reloaded = await dm.get_or_load_dataset("ds")
    assert float(reloaded.adata.X.max()) == float(np.asarray(reloaded.adata.X).round().max())  # still counts
