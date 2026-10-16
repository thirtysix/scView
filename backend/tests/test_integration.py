"""Joint multi-dataset integration (core/integration.py + /integrate endpoints)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.merge import MergeJoin
from scview.core.integration import (
    LABEL_COLUMN,
    IntegrationError,
    batch_mixing,
    build_plan,
    build_unit,
    carried_obs,
    joint_steps,
    load_units,
    looks_like_counts,
    merge_joint,
    read_batch_mixing,
    recover_counts,
    run_integration_streamed,
    unique_labels,
)
from scview.core.uns_keys import BATCH_MIXING_KEY

_TYPES = ["T", "B", "Mono"]


def _counts_dataset(n_per_type: int = 60, n_genes: int = 120, batch_shift: float = 1.0,
                    seed: int = 0) -> ad.AnnData:
    """Three cell types with distinct marker blocks; ``batch_shift`` scales a
    non-marker gene block to mimic a technical batch effect."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(42).uniform(0.5, 2.0, n_genes)  # shared biology
    rows, labels = [], []
    for i, t in enumerate(_TYPES):
        mu = base.copy()
        mu[i * 10:(i + 1) * 10] *= 8
        mu[40:80] *= batch_shift
        rows.append(rng.poisson(mu, size=(n_per_type, n_genes)))
        labels += [t] * n_per_type
    n = len(labels)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(labels), "barcode": [f"bc{i}" for i in range(n)]},
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    return ad.AnnData(sp.csr_matrix(np.vstack(rows).astype(np.float32)), obs=obs, var=var)


def _lognorm(adata: ad.AnnData) -> ad.AnnData:
    """A processed copy whose counts are gone (X and raw both log-normalised)."""
    import scanpy as sc

    p = adata.copy()
    sc.pp.normalize_total(p, target_sum=1e4)
    sc.pp.log1p(p)
    p.raw = p
    return p


def _add_dataset(data_dir: Path, did: str, name: str, adata: ad.AnnData,
                 subdir: str = "uploads") -> None:
    up = data_dir / "uploads" / did
    up.mkdir(parents=True, exist_ok=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    d = data_dir / subdir / did
    d.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(d / f"{name}.h5ad")


# ---------------------------------------------------------------------------
# Counts recovery
# ---------------------------------------------------------------------------


def test_looks_like_counts():
    a = _counts_dataset()
    assert looks_like_counts(a.X)
    assert not looks_like_counts(_lognorm(a).X)
    assert not looks_like_counts(np.array([[1.0, -2.0]]))
    assert not looks_like_counts(sp.csr_matrix((3, 3)))  # all zeros: no signal


def test_recover_counts_prefers_counts_layer():
    a = _lognorm(_counts_dataset())
    a.layers["counts"] = _counts_dataset().X
    rc = recover_counts(a, [])
    assert rc.source == "layers['counts']"
    assert looks_like_counts(rc.X)


def test_recover_counts_falls_back_to_source_aligned_by_barcode(tmp_path):
    original = _counts_dataset()
    src = tmp_path / "uploads" / "d1" / "orig.h5ad"
    src.parent.mkdir(parents=True)
    original.write_h5ad(src)

    # Processed: counts gone, cells filtered + reordered.
    processed = _lognorm(original)[[5, 2, 100, 0]].copy()
    rc = recover_counts(processed, [str(src)])
    assert rc.source == "X of the original upload"
    np.testing.assert_array_equal(rc.X.toarray(), original.X[[5, 2, 100, 0]].toarray())


def test_recover_counts_errors_when_nothing_has_counts():
    with pytest.raises(IntegrationError, match="no raw counts"):
        recover_counts(_lognorm(_counts_dataset()), [])


# ---------------------------------------------------------------------------
# Annotations, labels, merge
# ---------------------------------------------------------------------------


def test_carried_obs_keeps_annotations_and_prefixes_source_clusters():
    a = _counts_dataset()
    a.obs["scview_leiden_r0.5"] = pd.Categorical(["0", "1"] * (a.n_obs // 2))
    a.obs["leiden"] = a.obs["scview_leiden_r0.5"]
    a.obs["n_genes"] = 5  # numeric: recomputed by the joint QC, not carried
    a.uns["scview_active_clustering"] = "scview_leiden_r0.5"
    obs = carried_obs(a, "A")
    assert "cell_type" in obs.columns
    assert "barcode" not in obs.columns  # too many levels
    assert not {"leiden", "scview_leiden_r0.5", "n_genes"} & set(obs.columns)
    assert set(obs["source_cluster"]) == {"A:0", "A:1"}


def test_unique_labels():
    assert unique_labels(["a", "a", "a_2", ""]) == ["a", "a_2", "a_2_2", "dataset"]


def test_merge_joint_labels_cells_and_keeps_unshared_columns():
    a, b = _counts_dataset(seed=1), _counts_dataset(seed=2)
    a.obs["condition"] = "ctrl"  # only A has it
    units = {
        "A": build_unit(a, dataset_id="a", name="A", label="A", fallback_paths=[]),
        "B": build_unit(b, dataset_id="b", name="B", label="B", fallback_paths=[]),
    }
    joint = merge_joint(units)
    assert joint.n_obs == a.n_obs + b.n_obs
    assert joint.obs_names.is_unique  # same barcodes in both datasets
    assert list(joint.obs[LABEL_COLUMN].cat.categories) == ["A", "B"]
    assert "condition" in joint.obs.columns
    assert joint.obs.loc[joint.obs[LABEL_COLUMN] == "B", "condition"].isna().all()


def test_plan_blocks_single_dataset_and_cell_cap(tmp_path):
    dm = DatasetManager(str(tmp_path))
    _add_dataset(tmp_path, "a", "A", _counts_dataset())
    _add_dataset(tmp_path, "b", "B", _counts_dataset(seed=3))
    checks, units = load_units(dm, ["a"])
    assert not build_plan(checks, units, max_datasets=8, max_cells=10**6).ok

    checks, units = load_units(dm, ["a", "b"])
    plan = build_plan(checks, units, max_datasets=8, max_cells=10**6)
    assert plan.ok, plan.blocking
    assert plan.merge.intersection == 120
    assert "cell_type" in plan.shared_columns

    capped = build_plan(checks, units, max_datasets=8, max_cells=100)
    assert not capped.ok and "limit" in capped.blocking[0]


def test_plan_reports_dataset_without_counts(tmp_path):
    dm = DatasetManager(str(tmp_path))
    _add_dataset(tmp_path, "a", "A", _counts_dataset())
    _add_dataset(tmp_path, "b", "B", _lognorm(_counts_dataset()))
    checks, units = load_units(dm, ["a", "b"])
    plan = build_plan(checks, units, max_datasets=8, max_cells=10**6)
    assert not plan.ok
    assert checks[1].error and "no raw counts" in checks[1].error


def test_joint_steps():
    assert "batch_correction" in joint_steps("harmony")
    assert "uncorrected_embedding" in joint_steps("harmony")
    assert "batch_correction" not in joint_steps("none")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _cross_batch_fraction(emb: np.ndarray, batch: np.ndarray, k: int = 15) -> float:
    """Mean fraction of each cell's k nearest neighbours from another batch."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)
    return float((batch[idx[:, 1:]] != batch[:, None]).mean())


def test_integration_end_to_end_mixes_batches(tmp_path):
    dm = DatasetManager(str(tmp_path))
    _add_dataset(tmp_path, "a", "A", _counts_dataset(seed=1))
    # A moderate effect: one so extreme that every Harmony cluster holds a single
    # batch (e.g. 5x here) is beyond what Harmony can correct.
    _add_dataset(tmp_path, "b", "B", _counts_dataset(seed=2, batch_shift=2.0))
    _, units = load_units(dm, ["a", "b"])

    events = list(run_integration_streamed(
        units, dataset_id="joint1", name="joint", data_dir=tmp_path,
        output_path_for=lambda src: dm.derived_h5ad_path("joint1", src),
        join=MergeJoin.inner, overrides={"n_top_genes": 100, "n_comps": 20},
    ))
    kinds = [e for e, _ in events]
    assert kinds[:3] == ["stage", "stage", "created"]
    assert "step_error" not in kinds, [d for e, d in events if e == "step_error"]
    assert kinds[-1] == "pipeline_complete"

    # Merged counts are the immutable source; processing landed in derived/.
    assert (tmp_path / "ingested" / "joint1" / "joint.h5ad").exists()
    out = ad.read_h5ad(tmp_path / "derived" / "joint1" / "joint.h5ad")
    for key in ("X_pca_harmony", "X_umap", "X_umap_uncorrected"):
        assert key in out.obsm
    assert out.uns["scview_active_clustering"] in out.obs.columns
    assert set(out.obs[LABEL_COLUMN]) == {"A", "B"}

    batch = out.obs[LABEL_COLUMN].astype(str).to_numpy()
    before = _cross_batch_fraction(out.obsm["X_pca"], batch)
    after = _cross_batch_fraction(out.obsm["X_pca_harmony"], batch)
    assert after > before + 0.15, (before, after)

    # The same before/after mixing, scored at integration time and saved with it.
    assert ("substep", {"step": "batch_correction", "message": "Scoring batch mixing"}) in events
    mixing = read_batch_mixing(out)
    assert mixing["uncorrected"] == pytest.approx(
        _cross_batch_fraction(out.obsm["X_pca"], batch, 30), abs=1e-4
    )
    assert mixing["corrected"] == pytest.approx(
        _cross_batch_fraction(out.obsm["X_pca_harmony"], batch, 30), abs=1e-4
    )
    assert mixing["ideal"] == pytest.approx(180 / 359, abs=1e-4)
    ct = out.obs["cell_type"].astype(str).to_numpy()
    assert _cross_batch_fraction(out.obsm["X_pca_harmony"], ct) < 0.05  # biology kept

    meta = json.loads((tmp_path / "uploads" / "joint1" / "metadata.json").read_text())
    assert meta["source"] == "integrated"
    assert [d["id"] for d in meta["integrated_from"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_integrate_endpoints(client, tmp_data_dir, dataset_manager):
    _add_dataset(tmp_data_dir, "a", "A", _counts_dataset(seed=1))
    _add_dataset(tmp_data_dir, "b", "B", _counts_dataset(seed=2, batch_shift=3.0))

    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"]})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = await client.post("/api/v1/integrate/run-stream",
                          json={"dataset_ids": ["a"], "name": "x"})
    assert r.status_code == 400  # one dataset is not an integration

    r = await client.post(
        "/api/v1/integrate/run-stream",
        json={"dataset_ids": ["a", "b"], "name": "AB", "labels": ["ctrl", "stim"],
              "params": {"n_top_genes": 100, "n_comps": 20}},
    )
    assert r.status_code == 200
    events = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert events[-1] == "complete", r.text[-500:]
    new_id = json.loads(r.text.strip().splitlines()[-1][6:])["dataset_id"]

    info = (await client.get(f"/api/v1/datasets/{new_id}")).json()
    assert info["name"] == "AB"
    assert "X_umap_uncorrected" in info["available_embeddings"]
    assert {"dataset", "cell_type"} <= {c["name"] for c in info["obs_columns"]}

    # The batch-mixing score is part of the joint dataset.
    r = await client.get(f"/api/v1/datasets/{new_id}/integration/summary")
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["joint"] is True
    assert summary["label_column"] == "dataset"
    assert summary["cluster_column"] == info["active_clustering"]
    assert summary["mixing"]["corrected"] is not None
    assert [d["label"] for d in summary["mixing"]["datasets"]] == ["ctrl", "stim"]

    r = await client.get("/api/v1/datasets/a/integration/summary")
    assert r.json() == {"dataset_id": "a", "joint": False}
    r = await client.get("/api/v1/datasets/ghost/integration/summary")
    assert r.status_code == 404

    # A joint dataset built before the score existed is scored on first view, and saved.
    import h5py

    derived = next((tmp_data_dir / "derived" / new_id).glob("*.h5ad"))
    with h5py.File(derived, "a") as f:
        del f["uns"][BATCH_MIXING_KEY]
    dataset_manager.evict(new_id)
    r = await client.get(f"/api/v1/datasets/{new_id}/integration/summary")
    assert r.json()["mixing"] == summary["mixing"]
    assert BATCH_MIXING_KEY in ad.read_h5ad(derived).uns


# ---------------------------------------------------------------------------
# Batch mixing
# ---------------------------------------------------------------------------


def _two_datasets(n_a: int, n_b: int, *, separated: bool, seed: int = 0) -> ad.AnnData:
    """Random 10-d points labelled A/B; ``separated`` moves B far away (no mixing)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_a + n_b, 10)).astype(np.float32)
    if separated:
        X[n_a:] += 100
    a = ad.AnnData(np.zeros((n_a + n_b, 1), dtype=np.float32))
    a.obs[LABEL_COLUMN] = pd.Categorical(["A"] * n_a + ["B"] * n_b)
    a.obsm["X_pca"] = X
    return a


def test_batch_mixing_separated_vs_mixed():
    a = _two_datasets(200, 200, separated=True)
    a.obsm["X_pca_harmony"] = _two_datasets(200, 200, separated=False).obsm["X_pca"]
    m = batch_mixing(a)
    assert m["k"] == 30 and m["n_scored"] == 400
    assert m["uncorrected"] == 0.0
    assert m["ideal"] == pytest.approx(200 / 399, abs=1e-4)
    assert m["corrected"] == pytest.approx(m["ideal"], abs=0.05)
    assert m["spaces"] == {"uncorrected": "X_pca", "corrected": "X_pca_harmony"}


def test_batch_mixing_ideal_follows_the_dataset_proportions():
    # 300 A + 100 B: an A cell expects 100/399 others, a B cell 300/399.
    m = batch_mixing(_two_datasets(300, 100, separated=False))
    rows = {d["label"]: d for d in m["datasets"]}
    assert rows["A"]["ideal"] == pytest.approx(100 / 399, abs=1e-4)
    assert rows["B"]["ideal"] == pytest.approx(300 / 399, abs=1e-4)
    assert m["ideal"] == pytest.approx((300 * 100 + 100 * 300) / 399 / 400, abs=1e-4)
    assert m["corrected"] is None, "no corrected space, no corrected score"


def test_batch_mixing_excludes_the_cell_itself_among_duplicates():
    # All points identical: the neighbour order is arbitrary, but with k = n - 1
    # every cell's neighbours are exactly all the other cells.
    a = _two_datasets(25, 25, separated=False)
    a.obsm["X_pca"] = np.zeros_like(a.obsm["X_pca"])
    m = batch_mixing(a, k=49)
    assert m["uncorrected"] == pytest.approx(25 / 49, abs=1e-4)


def test_batch_mixing_scores_a_subset_of_a_large_dataset():
    m = batch_mixing(_two_datasets(300, 300, separated=True), max_scored=50)
    assert m["n_scored"] == 50 and m["n_cells"] == 600
    assert m["uncorrected"] == 0.0


def test_batch_mixing_needs_two_datasets_and_a_pca():
    assert batch_mixing(_two_datasets(30, 0, separated=False)) is None
    no_pca = _two_datasets(30, 30, separated=False)
    del no_pca.obsm["X_pca"]
    assert batch_mixing(no_pca) is None
    assert batch_mixing(ad.AnnData(np.zeros((5, 1), dtype=np.float32))) is None


def test_redoing_pca_drops_the_mixing_score():
    from scview.core.pipeline import PipelineParams, run_pipeline

    a = _counts_dataset()
    a.uns[BATCH_MIXING_KEY] = "{}"
    out, res = run_pipeline(a, ["normalization", "log_transform", "pca"], PipelineParams(n_comps=10))
    assert res.errors == {}
    assert BATCH_MIXING_KEY not in out.uns


def test_reset_to_counts_drops_the_mixing_score():
    from scview.core.pipeline import PipelineParams, _run_reset_to_counts

    a = _counts_dataset()
    a.layers["counts"] = a.X.copy()
    a.uns[BATCH_MIXING_KEY] = "{}"
    assert BATCH_MIXING_KEY not in _run_reset_to_counts(a, PipelineParams()).uns
