"""A joint embedding of many datasets (``MAX_INTEGRATE_DATASETS`` 32): the plan reads 28 small
units, its estimate counts each unit, a set that cannot fit the limit is refused, and the
default name stays readable."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.config import Settings
from scview.core import integration, preflight
from scview.core.dataset_manager import DatasetManager

API = "/api/v1"
MiB = 1 << 20


def _unit(seed: int, n: int = 60, genes: int = 80) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, genes)).astype(np.float32))
    obs = pd.DataFrame({"donor": pd.Categorical([f"d{seed}"] * n)}, index=[f"u{seed}_c{i}" for i in range(n)])
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(genes)]))


def _add(data_dir: Path, n: int) -> list[str]:
    ids = []
    for i in range(n):
        did = f"u{i:02d}"
        up = data_dir / "uploads" / did
        up.mkdir(parents=True)
        (up / "metadata.json").write_text(json.dumps({"id": did, "name": f"slide {i}", "status": "ready"}))
        _unit(i).write_h5ad(up / "data.h5ad")
        ids.append(did)
    return ids


def test_the_cap_is_32():
    assert Settings().MAX_INTEGRATE_DATASETS == 32


def test_28_units_are_read_and_estimated_unit_by_unit(tmp_path):
    dm = DatasetManager(data_dir=str(tmp_path))
    ids = _add(tmp_path, 28)
    checks, units = integration.load_units(dm, ids)
    plan = integration.build_plan(checks, units, max_datasets=32, max_cells=10**6)
    assert plan.ok and len(units) == 28 and plan.est_cells == 28 * 60
    est = integration.estimate_joint(units, integration.joint_genes(plan))
    # Each unit's counts, obs and gene names are held: the estimate counts all 28.
    one = integration._unit_bytes(next(iter(units.values())).adata)
    assert est.units_bytes == sum(integration._unit_bytes(u.adata) for u in units.values()) >= 28 * one * 0.9
    assert one > next(iter(units.values())).adata.X.data.nbytes  # more than the counts alone
    assert integration.read_need(dm, ids) > 0


def test_a_set_that_cannot_fit_the_limit_is_refused(tmp_path):
    dm = DatasetManager(data_dir=str(tmp_path))
    ids = _add(tmp_path, 28)
    checks, units = integration.load_units(dm, ids)
    plan = integration.build_plan(checks, units, max_datasets=32, max_cells=10**6)
    est = integration.estimate_joint(units, integration.joint_genes(plan))
    small = preflight.Available(limit_bytes=est.peak_bytes // 3, used_bytes=est.units_bytes,
                                available_bytes=est.peak_bytes // 3 - est.units_bytes, limited=True,
                                source="cgroup")
    out = integration.joint_preflight(est, small, preflight.Cached())
    assert out["blocking"] and "integrate fewer or smaller datasets" in out["blocking"]
    roomy = preflight.Available(limit_bytes=64 * 1024 * MiB, used_bytes=est.units_bytes,
                                available_bytes=60 * 1024 * MiB, limited=True, source="cgroup")
    assert integration.joint_preflight(est, roomy, preflight.Cached())["blocking"] is None


def test_the_default_name_stays_short():
    assert integration.default_name(["a", "b"]) == "joint: a + b"
    labels = [f"slide {i}" for i in range(28)]
    assert integration.default_name(labels) == "joint: slide 0 + slide 1 + slide 2 + 25 more"


async def test_the_plan_takes_28_datasets_and_refuses_33(client, dataset_manager):
    ids = _add(Path(dataset_manager.data_dir), 28)
    r = await client.post(f"{API}/integrate/plan", json={"dataset_ids": ids})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["datasets"]) == 28 and not any("At most" in b for b in body["blocking"])
    r = await client.post(f"{API}/integrate/plan", json={"dataset_ids": [f"x{i}" for i in range(33)]})
    assert r.status_code == 400 and "At most 32" in r.json()["detail"]
