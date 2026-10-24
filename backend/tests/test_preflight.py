"""Preflight estimates: calibration against measured peaks, monotonicity, the
verdict and its suggestions, the memory and CPU readings, and the endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import preflight as pf
from scview.core.pipeline import ALL_STEPS, PipelineParams

GiB = 1 << 30
API = "/api/v1"
REPROCESS = ["reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "normalization",
             "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
             "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress"]


# --- calibration: the measured peaks the coefficients were fitted to -----------------------
# Headers as read from the test files; measured = cgroup anon at the peak minus the process
# before loading (scanpy imported), "Reprocess from scratch" steps, 2 CPUs, a fresh process.
# Re-measured 2026-10-22 since doublet detection scores the counts alone (core/scrublet_lean.py)
# and filtering cuts each matrix once (pipeline.subset_lean); pbmc is the 2026-10-21 run.
def _h(fmt, n, m, mats, obsm=0, obsp=0, varm=0, uns=0, obs_cols=0, var_cols=0, emb=False):
    # mats: (where, sparse, nnz, dtype[, non-zeros of a dense one])
    h = pf.Header(fmt, n, m, [pf.Matrix(mat[0], n, m, mat[1], mat[2], mat[3], "int32",
                                        nonzero=mat[4] if len(mat) > 4 else None) for mat in mats],
                  obsm={"X_pca": [n, 50]} if emb else {}, obsm_bytes=obsm, obsp_bytes=obsp,
                  varm_bytes=varm, uns_bytes=uns, obs_columns=obs_cols, var_columns=var_cols)
    return h


CALIBRATION = {
    # name: (header, measured resident after load, measured run peak, marker columns[, steps])
    "pbmc 10x h5": (_h("10x_h5", 1222, 33538, [("X", True, 2498225, "float32")]),
                    0.04, 0.63, []),
    # all dense: the first 64 rows say 27% of the counts are non-zero (read_header)
    "paul15 (dense)": (_h("h5ad", 2730, 3451, [("X", False, 9421230, "float32", 9421230),
                                               ("layers/counts", False, 9421230, "float32", 2578058),
                                               ("raw/X", False, 9421230, "float32", 2578058)],
                          546000, 990600, 1380400, 7497385, 10, 11, emb=True),
                       0.13, 0.63, ["paul15_clusters", "scview_leiden_r0.5"]),
    "Kang (sparse)": (_h("h5ad", 13836, 14053, [("X", False, 194437308, "float64"),
                                                ("layers/counts", True, 9655137, "float64"),
                                                ("raw/X", True, 9655137, "float64")],
                         8301600, 4574264, 5621200, 31893321, 15, 11, emb=True),
                      1.76, 2.65, ["cluster", "scview_leiden_r0.5"]),
    "ovary 10k": (_h("h5ad", 10000, 20082, [("X", True, 12109310, "float32"),
                                            ("layers/counts", False, 200820000, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  0.91, 4.36, ["cluster", "Cluster"]),
    "ovary 20k": (_h("h5ad", 20000, 20082, [("X", True, 24099492, "float32"),
                                            ("layers/counts", False, 401640000, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  1.75, 8.07, ["cluster", "Cluster"]),
    # the whole run fits in 16 GiB now (it was killed in filtering, and at 18 GiB before that
    # in doublet detection)
    "ovary 37k": (_h("h5ad", 37104, 20082, [("X", True, 44738263, "float32"),
                                            ("layers/counts", False, 745122528, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  3.22, 13.90, ["cluster", "Cluster"]),
    # the hashing PBMC filtered again (doublets by either source, per-type mito with the
    # floor) after scaling: a dense float32 X of every gene. Filtering alone.
    "hashing PBMC re-filter": (
        _h("h5ad", 20238, 22628, [("X", False, 457945464, "float32"),
                                  ("layers/counts", True, 18121931, "float32"),
                                  ("raw/X", True, 18121931, "float32")],
           5990448, 6870392, 9051200, 972622, 12, 12, emb=True),
        2.05, 3.36, [], ["filtering"]),
}


@pytest.mark.parametrize("name", list(CALIBRATION))
def test_estimates_are_within_the_stated_band_of_the_measured_peaks(name):
    h, load_m, peak_m, cols, *steps = CALIBRATION[name]
    load = pf.resident_bytes(h) / GiB
    run = pf.estimate_run(h, steps=steps[0] if steps else REPROCESS,
                          params=PipelineParams(marker_columns=cols), warm=False)
    assert abs(run.peak_bytes / GiB / peak_m - 1) <= pf.ERROR_BAND
    assert abs(load - load_m) <= max(pf.ERROR_BAND * load_m, 0.03)


def test_doublet_detection_no_longer_copies_the_dense_counts_layer():
    # Scrublet scored a copy of the whole dataset: ovary's dense X (after the reset) and its
    # dense counts layer, 1.5 GiB each at 20k cells, made the step need more than 5 of them.
    # It scores the counts alone as CSR now (core/scrublet_lean.py): its z-scoring sets the
    # need, not the layers.
    h = CALIBRATION["ovary 20k"][0]
    run = pf.estimate_run(h, steps=REPROCESS, warm=True)
    dense = 20000 * 20082 * 4
    dbl = next(e for e in run.steps if e.step == "doublet_detection")
    # (a header alone: all cells as one channel, the variable genes as a share of the genes)
    assert dbl.transient_bytes == int(pf.DBL_DENSE_PER_VALUE * 20000 * 20082) + pf.DBL_OUT_PER_CELL * 20000
    assert dbl.transient_bytes < 3.5 * dense
    # the dense counts are sized by their non-zeros: those of the sparse X beside them
    no_x = pf.Header("h5ad", 20000, 20082, [m for m in h.matrices if m.where != "X"])
    assert pf._State.from_header(no_x).layers["counts"].nonzero is None
    assert abs(pf._State.from_header(h).layers["counts"].nonzero - h.matrices[0].nnz) <= 1


# Filtering's own peak (held at its start + its transient) in those runs. It removed 11% of
# ovary's cells and 22% of its genes, none of Kang's or paul15's cells, and 23% of the
# hashing PBMC's (3% of its genes).
FILTERING_PEAK = {"ovary 10k": 2.38, "ovary 20k": 4.53, "ovary 37k": 8.28, "Kang (sparse)": 0.50,
                  "hashing PBMC re-filter": 3.36}


@pytest.mark.parametrize("name", list(FILTERING_PEAK))
def test_filtering_estimates_are_within_the_band(name):
    # A processed dataset was taken to keep every cell (+0.34 GiB for the hashing PBMC, which
    # needed +4.3 and was killed at 6 GB); filtering now cuts each matrix once, and data
    # filtered before can lose cells again.
    h, _, _, cols, *steps = CALIBRATION[name]
    run = pf.estimate_run(h, steps=steps[0] if steps else REPROCESS,
                          params=PipelineParams(marker_columns=cols), warm=False)
    step = next(e for e in run.steps if e.step == "filtering")
    assert abs(step.peak_bytes / GiB / FILTERING_PEAK[name] - 1) <= pf.ERROR_BAND


def test_filtering_frees_the_matrices_it_cut():
    # The dataset's own copy gives its matrices up as they are cut: ovary holds less after
    # filtering (4.41 GiB measured) than before it (5.82).
    h, _, _, cols = CALIBRATION["ovary 37k"]
    run = pf.estimate_run(h, steps=REPROCESS, params=PipelineParams(marker_columns=cols), warm=False)
    i = next(i for i, e in enumerate(run.steps) if e.step == "filtering")
    assert run.steps[i + 1].resident_bytes < run.steps[i].resident_bytes


def test_reset_to_counts_frees_the_dense_x_before_the_later_steps():
    h = CALIBRATION["Kang (sparse)"][0]
    run = pf.estimate_run(h, steps=REPROCESS, warm=False)
    x_dense = 13836 * 14053 * 8
    counts = 9655137 * (8 + 4) + 13837 * 4
    reset, qc = run.steps[0], run.steps[1]
    assert qc.resident_bytes <= reset.resident_bytes - (x_dense - counts)
    # so the run's peak is well under "what it holds now + the largest step"
    biggest = max(e.transient_bytes for e in run.steps)
    assert run.peak_bytes < run.start_bytes + biggest - GiB


# --- monotonic in cells and stored values ----------------------------------------------------


def test_run_estimate_grows_with_cells_and_values():
    peaks = [pf.estimate_run((n, 15000), nnz=n * 1000, steps=REPROCESS, warm=True).peak_bytes
             for n in (1000, 5000, 20000, 80000)]
    assert peaks == sorted(peaks) and len(set(peaks)) == 4
    by_nnz = [pf.estimate_run((10000, 15000), nnz=v, steps=REPROCESS, warm=True).peak_bytes
              for v in (1_000_000, 10_000_000, 50_000_000)]
    assert by_nnz == sorted(by_nnz) and len(set(by_nnz)) == 3
    dense = pf.estimate_run((10000, 15000), dense=True, steps=REPROCESS, warm=True).peak_bytes
    assert dense > by_nnz[-1]


def test_load_estimate_grows_with_cells_and_values():
    def res(n, nnz):
        return pf.resident_bytes(pf.Header("h5ad", n, 20000, [pf.Matrix("X", n, 20000, True, nnz)]))

    assert res(1000, 10**6) < res(10000, 10**6) < res(10000, 10**7)


def test_a_dense_matrix_takes_cells_x_genes_x_itemsize():
    assert pf.Matrix("X", 100, 50, False, 0, "float64").nbytes == 100 * 50 * 8


# --- estimate_step_peak: the pure function for a pre-step hook ------------------------------


def test_step_peak_scaling_holds_no_dense_copy_of_every_gene():
    # It used to make X dense float32 and z-score it (2 x n x m x 4); now it only works out
    # the variable genes' statistics (core/scaling.py): a chunk of a sparse X's values, or a
    # block of a dense X's genes.
    n, m = 100_000, 28_000
    sparse = pf.estimate_step_peak("scaling", (n, m), 200_000_000, False)
    assert sparse == pf.SPARSE_STATS_CHUNK * 33 + m * 24 and sparse < n * m * 4 / 50
    dense = pf.estimate_step_peak("scaling", (n, m), 0, True)
    assert dense == 3 * n * pf.STAT_BLOCK * 4


@pytest.mark.parametrize("cells, genes, nnz, step, measured_gib", [
    # Each step alone on the state before it, warm, 2 CPUs (2026-10-23, process RSS):
    # Kang and the 100k-neuron subset of 10x's 1.3M (after filtering), 2,000 variable genes.
    # Markers re-measured 2026-10-24 by blocks of genes (core/rank_genes.py) on the states
    # a live in-memory run saved; scanpy's own took 0.220 and 2.851.
    (13_815, 13_908, 9_643_068, "pca", 0.240),
    (13_815, 13_908, 9_643_068, "marker_genes", 0.105),
    (88_640, 19_824, 186_795_611, "pca", 1.431),
    (88_640, 19_824, 186_795_611, "marker_genes", 0.329),
])
def test_pca_and_marker_estimates_are_within_the_error_band_of_measured_runs(
        cells, genes, nnz, step, measured_gib):
    mats = [pf.Matrix(w, cells, genes, True, nnz, "float32") for w in ("X", "layers/counts", "raw/X")]
    h = pf.Header("h5ad", cells, genes, mats, n_hvg=2000)
    t, _, _ = pf._STEP_MODELS[step](pf._State.from_header(h), PipelineParams())
    assert abs(t / pf.GiB / measured_gib - 1) <= pf.ERROR_BAND


def test_scaling_sets_raw_to_x_and_pca_reads_the_variable_genes_densely():
    n, m = 50_000, 20_000
    h = pf.Header("h5ad", n, m, [pf.Matrix("X", n, m, True, 100_000_000, "float32")])
    run = pf.estimate_run(h, steps=["normalization", "log_transform", "highly_variable_genes",
                                    "scaling", "pca"], params=PipelineParams(n_top_genes=2000), warm=True)
    by = {e.step: e for e in run.steps}
    # .raw is X itself during the run: scaling adds no matrix to what is held.
    assert by["pca"].resident_bytes - by["scaling"].resident_bytes < 50 * pf.MiB
    # PCA: about two copies of cells x variable genes in float32, not of every gene.
    assert 2 * n * 2000 * 4 <= by["pca"].transient_bytes <= 3 * n * 2000 * 4
    # Saved and reopened, .raw is its own copy of the log values.
    assert run.end_bytes > by["pca"].resident_bytes + 0.9 * h.x.nbytes


def test_step_peak_doublets_ignores_the_layers_and_reads_the_non_zeros():
    n, m = 4000, 10000
    base = pf.estimate_step_peak("doublet_detection", (n, m), 0, True, nonzero=n * m // 10)
    # Scrublet's dense z-scoring, and the per-cell results
    assert base == int(pf.DBL_DENSE_PER_VALUE * n * m) + pf.DBL_OUT_PER_CELL * n
    # the layers and .raw are not copied any more (they were: +other_bytes)
    assert pf.estimate_step_peak("doublet_detection", (n, m), 0, True, nonzero=n * m // 10,
                                 other_bytes=n * 1000) == base
    # a dense X of unknown non-zeros is taken as full: six float32 CSR copies of every value
    assert pf.estimate_step_peak("doublet_detection", (n, m), 0, True) == \
        int(pf.DBL_SPARSE_COPIES * (n * m * 8 + (n + 1) * 4)) + pf.DBL_OUT_PER_CELL * n


# Doublet detection as the pipeline runs it on a dataset in memory (Scrublet per capture
# channel, pipeline._run_doublet_detection), measured 2026-10-23: peak RSS over the process
# before the step (VmHWM reset, after a warm-up run), 2 CPUs, a fresh container each; the
# plan's variable genes as ``doublet_plan`` counted them on that data.
DOUBLET_MEASURED = {
    # name: (cells, genes, X sparse, stored values, plan channels (the largest), plan hvg, GiB)
    "paul15, one channel": (2730, 3451, False, 2386248, [2730], 657, 0.262),
    "Kang, 2 lanes from the barcodes": (13836, 14053, True, 9655137, [6573, 7263], 1522, 1.470),
    "Kang, all cells as one": (13836, 14053, True, 9655137, [13836], 1506, 2.590),
    "Kang, its 8 donors": (13836, 14053, True, 9655137, [3159], 1472, 0.672),
    "brain 100k, 133 gem_well": (100000, 27998, True, 201003839, [914], 1744, 0.261),
    "brain 100k, 133 from the barcodes": (100000, 27998, True, 201003839, [914], 1744, 0.238),
    "brain 30k subset, all cells as one": (30000, 27998, True, 60189617, [30000], 1391, 5.223),
    # main's run: one channel of every cell (the 2.5x over-estimate of the Roihu test pass)
    "brain 100k, all cells as one": (100000, 27998, True, 201003839, [100000], 1362, 16.542),
}


@pytest.mark.parametrize("name", list(DOUBLET_MEASURED))
def test_doublet_estimate_follows_the_largest_channel_within_the_band(name):
    n, m, sparse, nnz, channels, hvg, measured = DOUBLET_MEASURED[name]
    t = pf.estimate_step_peak("doublet_detection", (n, m), nnz, not sparse,
                              nonzero=None if sparse else nnz,
                              doublets=pf.DoubletPlan(channels, hvg))
    assert abs(t / GiB / measured - 1) <= pf.ERROR_BAND


def test_doublet_time_follows_each_channel_and_its_neighbour_search():
    # measured: the brain's 133 wells (largest 914) 43 s; one channel of its 100k cells 688 s
    def est(plan, n):
        state = pf._State(n, 27998, pf.Matrix("X", n, 27998, True, n * 2000), {}, None, 0)
        return pf._step_doublet(state, None, plan)[2]

    wells = est(pf.DoubletPlan([752] * 133, 1744), 100000)
    whole = est(pf.DoubletPlan([100000], 1362), 100000)
    assert 43 / 2 <= wells <= 43 * 2 and 688 / 2 <= whole <= 688 * 2


def test_doublet_estimate_leaves_out_channels_too_small_to_score():
    plan = pf.DoubletPlan([5000, 80, 120], 1500)
    t = pf.estimate_step_peak("doublet_detection", (5200, 20000), 10_000_000, False, doublets=plan)
    assert t == pf.DBL_PER_CELL_HVG * 5000 * 1500 + pf.DBL_OUT_PER_CELL * 5200
    tiny = pf.estimate_step_peak("doublet_detection", (200, 20000), 400_000, False,
                                 doublets=pf.DoubletPlan([80, 120], None))
    assert tiny == int(pf.DBL_DENSE_PER_VALUE * 120 * 20000) + pf.DBL_OUT_PER_CELL * 200
    none = pf.estimate_step_peak("doublet_detection", (160, 20000), 320_000, False,
                                 doublets=pf.DoubletPlan([80, 80], None))
    assert none == pf.DBL_OUT_PER_CELL * 160  # nothing scored: the results alone


def test_doublet_plan_counts_scrublets_own_variable_genes_in_the_largest_channel(monkeypatch):
    import scanpy as sc

    from scview.core import scrublet_lean
    from tests.test_doublet_channels import _dataset

    a = _dataset([150, 260, 200])  # three gem_well channels, w2 the largest
    plan = pf.doublet_plan(a, PipelineParams())
    assert plan.channels == [150, 260, 200]
    seen = []
    real = sc.pp.scrublet

    def spy(adata, *args, **kw):
        seen.append(adata.n_vars)
        return real(adata, *args, **kw)

    monkeypatch.setattr(sc.pp, "scrublet", spy)
    labels = a.obs["gem_well"].astype(str).to_numpy()
    scrublet_lean.scrublet_counts(a.X[labels == "w2"], 0.01, use_approx_neighbors=False)
    assert plan.hvg == seen[0]  # the genes Scrublet scored in that channel
    per = pf.step_peak_for(a, "doublet_detection", PipelineParams())
    whole = pf.step_peak_for(a, "doublet_detection", PipelineParams(doublet_channel_key="__all__"))
    assert per == pf.DBL_PER_CELL_HVG * 260 * plan.hvg + pf.DBL_OUT_PER_CELL * a.n_obs
    assert whole > per


def test_doublet_plan_of_a_file_reads_its_obs_alone(tmp_path):
    from tests.test_doublet_channels import _dataset

    a = _dataset([150, 260, 200])
    p = tmp_path / "f.h5ad"
    a.write_h5ad(p)
    plan = pf._file_doublet_plan(p, PipelineParams())
    assert plan.channels == [150, 260, 200] and plan.hvg is None  # the matrix is not read
    assert pf._file_doublet_plan(p, PipelineParams(doublet_channel_key="__all__")).channels == [610]
    assert pf._file_doublet_plan(tmp_path / "missing.h5ad", PipelineParams()) is None


def test_doublet_estimate_without_the_data_takes_one_channel_and_a_share_of_the_genes():
    h = pf.Header("h5ad", 20000, 20082, [pf.Matrix("X", 20000, 20082, True, 24_000_000)])
    one = pf.estimate_run(h, steps=["doublet_detection"], warm=True).steps[0].transient_bytes
    assert one == int(pf.DBL_DENSE_PER_VALUE * 20000 * 20082) + pf.DBL_OUT_PER_CELL * 20000
    plan = pf.DoubletPlan([5000] * 4)  # e.g. read from the file's obs
    four = pf.estimate_run(h, steps=["doublet_detection"], warm=True, doublets=plan).steps[0]
    assert four.transient_bytes == int(pf.DBL_DENSE_PER_VALUE * 5000 * 20082) + \
        pf.DBL_OUT_PER_CELL * 20000


def test_step_peak_unknown_step_and_params():
    assert pf.estimate_step_peak("umap", (100, 100), 100, False) == 0
    small = pf.estimate_step_peak("pca", (10000, 15000), 10**6, False,
                                  PipelineParams(n_top_genes=500))
    large = pf.estimate_step_peak("pca", (10000, 15000), 10**6, False,
                                  PipelineParams(n_top_genes=4000))
    assert small < large


def test_step_peak_for_reads_the_live_anndata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.random(300, 200, density=0.1, format="csr", dtype=np.float32, random_state=rng))
    without = pf.step_peak_for(a, "filtering"), pf.step_peak_for(a, "doublet_detection")
    a.layers["counts"] = a.X.copy()
    # cut after X, whose old copy is let go first: a layer no larger does not raise the peak
    assert pf.step_peak_for(a, "filtering") == without[0]
    a.layers["dense"] = a.X.toarray()  # a larger one does
    assert pf.step_peak_for(a, "filtering") > without[0]
    assert pf.step_peak_for(a, "doublet_detection") == without[1]  # Scrublet gets X alone
    assert pf.step_peak_for(a, "no_such_step") == 0


def test_live_dense_counts_are_sized_by_their_sampled_non_zeros():
    # A reset puts ovary's dense counts in X: as every value, six CSR copies made 20 GiB of
    # a step that needs 8.6; their non-zeros (6%) are counted on a sample of rows.
    rng = np.random.default_rng(0)
    X = (rng.random((3000, 4000)) < 0.06).astype(np.float32) * rng.integers(1, 9, (3000, 4000))
    h = pf.header_from_adata(ad.AnnData(X=X))
    assert abs(h.x.nonzero / np.count_nonzero(X) - 1) < 0.05
    sparse = pf.step_peak_for(ad.AnnData(X=sp.csr_matrix(X)), "doublet_detection")
    assert pf.step_peak_for(ad.AnnData(X=X), "doublet_detection") == pytest.approx(sparse, rel=0.05)


# --- verdict ------------------------------------------------------------------------------


def _avail(free_gb=5.0, limit_gb=8.0, limited=True):
    return pf.Available(int(limit_gb * GiB), int((limit_gb - free_gb) * GiB), int(free_gb * GiB),
                        limited, "container limit" if limited else "machine")


@pytest.mark.parametrize("need_share, level", [(0.0, "fits"), (0.69, "fits"), (0.70, "tight"),
                                               (1.0, "tight"), (1.01, "wont_fit")])
def test_verdict_thresholds(need_share, level):
    avail = 10 * GiB
    assert pf._level(int(need_share * avail), avail) == level


def test_unknown_memory_and_nothing_free():
    assert pf._level(GiB, None) == "unknown"
    assert pf._level(GiB, 0) == "wont_fit"
    assert pf._level(-GiB, 0) == "fits"


def _verdict(need_gb, cached=pf.Cached(), alternatives=(), avail=None):
    return pf.make_verdict(need=int(need_gb * GiB), peak=int(need_gb * GiB), stage="doublet detection",
                           seconds=(60, 240), avail=avail or _avail(), cached=cached,
                           lead=f"About {need_gb} GB more than now at its peak (doublet detection)",
                           breakdown=[], notes=[], alternatives=list(alternatives))


def test_wont_fit_suggestions_in_order_and_only_those_that_help():
    v = _verdict(7.0, cached=pf.Cached(2, int(1.9 * GiB)),
                 alternatives=[("Skip doublet detection", int(3 * GiB)),
                               ("Use 1,000 variable genes instead of 2,000", int(6.99 * GiB)),
                               ("Subsample the cells: half of them", int(3.5 * GiB))])
    assert v["level"] == "wont_fit"
    s = v["suggestions"]
    assert s[0].startswith("Raise SCVIEW_MEMORY_GB in the .env file in the scView folder from 8.0 to at least")
    # used 3 GB + 7 GB / 0.7 = 13 GB, so the estimate's error still fits
    assert "at least 13 GB" in s[0]
    assert s[1].startswith("Close the 2 other open datasets (about 1.9 GB)")
    assert s[2] == "Skip doublet detection (about 3.0 GB more at its peak)."
    assert s[3] == "Subsample the cells: half of them (about 3.5 GB more at its peak)."
    assert len(s) == 4  # fewer variable genes would not help: left out


def test_the_sentence_and_the_cached_datasets():
    v = _verdict(1.0, cached=pf.Cached(1, int(0.5 * GiB)))
    assert v["level"] == "fits" and v["suggestions"] == []
    assert v["sentence"] == ("About 1.0 GB more than now at its peak (doublet detection), roughly "
                             "1 to 4 min. scView has 5.0 GB free of its 8.0 GB limit. 1 other open "
                             "dataset holds about 0.5 GB.")
    machine = _verdict(1.0, avail=_avail(limited=False))
    assert "of the 8.0 GB this machine gives it (no limit is set)" in machine["sentence"]


def test_closing_others_frees_enough():
    # 4.5 GB needed, 5 free: tight; with 3 GB held by others it would fit under 70%
    v = _verdict(4.5, cached=pf.Cached(2, 3 * GiB))
    assert v["level"] == "tight" and v["cached"]["frees_enough"]
    assert v["suggestions"][1].endswith("that would free enough.")


def test_tiny_amounts_do_not_read_as_zero():
    v = _verdict(1.0, cached=pf.Cached(1, 10 << 20))
    assert v["sentence"].endswith("1 other open dataset holds under 0.1 GB.")


@pytest.mark.parametrize("secs, text", [((2, 8), "a few seconds"), ((20, 80), "roughly 20 s to 1 min"),
                                        ((59.7, 240), "roughly 1 to 4 min"),
                                        ((120, 360), "roughly 2 to 6 min"),
                                        ((1800, 7200), "roughly 30 min to 2.0 h")])
def test_time_text(secs, text):
    assert pf.time_text(secs) == text


# --- memory and CPUs the container has ----------------------------------------------------


def test_file_cache_the_kernel_can_drop_is_not_counted_as_used(tmp_path):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{8 * GiB}\n")
    (cg / "memory.current").write_text(f"{3 * GiB}\n")
    (cg / "memory.stat").write_text(f"anon {GiB}\nfile {2 * GiB}\ninactive_file {GiB}\n")
    a = pf.available_memory(cg, tmp_path / "meminfo")
    assert (a.used_bytes, a.available_bytes, a.limited) == (2 * GiB, 6 * GiB, True)


def test_cpu_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert pf.cpu_quota(tmp_path) == 2.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert pf.cpu_quota(tmp_path) >= 1.0
    # fewer CPUs than the calibration: slower; more: faster, by less than the ratio
    assert pf._cpu_factor(1) > 1 > pf._cpu_factor(8) >= 0.5


# --- endpoints --------------------------------------------------------------------------------


def _adata(n=40, m=30) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(0.5, (n, m)).astype(np.float32)))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(m)]
    a.obs["cluster"] = pd.Categorical(["a", "b"] * (n // 2))
    return a


def _register(dm, did="pf1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return did


async def test_dataset_preflight_unknown_dataset_is_404(client):
    r = await client.get(f"{API}/datasets/nope/preflight", params={"steps": "pca"})
    assert r.status_code == 404


async def test_dataset_preflight_unknown_step_is_422_listing_the_valid_names(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": "qc_metrics,umap"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail.startswith("Unknown step: umap.")
    assert all(s in detail for s in ALL_STEPS)
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": " , "})
    assert r.status_code == 422 and r.json()["detail"].startswith("No steps given.")


async def test_dataset_preflight_from_the_file_header(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/preflight",
                         params={"steps": "pca,qc_metrics,doublet_detection", "n_top_genes": 20})
    assert r.status_code == 200
    body = r.json()
    assert not body["in_memory"]  # read from the header, not loaded
    assert dataset_manager.get_dataset(did) is None
    assert [s["step"] for s in body["steps"]] == ["qc_metrics", "doublet_detection", "pca", "save"]
    v = body["verdict"]
    assert v["level"] in ("fits", "tight", "wont_fit", "unknown")
    assert "GB" in v["sentence"] and v["error_band"] == pf.ERROR_BAND
    assert body["header"]["n_cells"] == 40


async def test_dataset_preflight_takes_the_doublet_channels(client, dataset_manager):
    # Per GEM well the step needs its largest well; all cells as one, the whole dataset.
    from tests.test_doublet_channels import _dataset

    d = Path(dataset_manager.data_dir) / "uploads" / "pf2"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "pf2", "name": "pf2", "status": "ready"}))
    _dataset([150, 260, 200]).write_h5ad(d / "data.h5ad")
    for loaded in (False, True):  # from the file's obs, then from the open dataset
        if loaded:
            await dataset_manager.load_dataset("pf2")
        per, whole = [
            next(s for s in (await client.get(f"{API}/datasets/pf2/preflight", params={
                "steps": "doublet_detection", **({"doublet_channel_key": k} if k else {})})).json()["steps"]
                 if s["step"] == "doublet_detection")["transient_bytes"]
            for k in ("", "__all__")]
        assert per < whole, loaded


async def test_dataset_preflight_uses_the_cached_dataset(client, dataset_manager):
    did = _register(dataset_manager)
    await dataset_manager.load_dataset(did)
    body = (await client.get(f"{API}/datasets/{did}/preflight", params={"steps": "scaling"})).json()
    assert body["in_memory"]
    # what the dataset holds now is not needed again
    assert body["verdict"]["need_bytes"] < body["verdict"]["peak_bytes"]


@pytest.fixture()
async def ingest_client(client, tmp_data_dir):
    from scview.core.ingestion.session import IngestSessionManager
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield client, mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


async def test_ingest_preflight_unknown_session_is_404(ingest_client):
    client, _ = ingest_client
    assert (await client.get(f"{API}/ingest/session/abc123/preflight")).status_code == 404


async def test_ingest_preflight_per_unit_and_merge(ingest_client):
    client, mgr = ingest_client
    sid = mgr.create_session()
    _adata(40, 30).write_h5ad(mgr.files_dir(sid) / "one.h5ad")
    body = (await client.get(f"{API}/ingest/session/{sid}/preflight")).json()
    assert [u["label"] for u in body["units"]] == ["one"] and body["merge"] is None
    v = body["units"][0]["verdict"]
    assert v["sentence"].startswith("40 cells x 30 genes: Loading takes under 0.1 GB at its peak")
    assert body["units"][0]["header"]["matrices"][0]["nnz"] > 0
    _adata(20, 30).write_h5ad(mgr.files_dir(sid) / "two.h5ad")
    body = (await client.get(f"{API}/ingest/session/{sid}/preflight")).json()
    assert len(body["units"]) == 2
    assert body["merge"]["sentence"].startswith("Merging the 2 samples takes ")
    assert "GB at its peak and" in body["merge"]["sentence"]



def test_doublets_on_a_processed_dataset_are_estimated_on_the_counts_alone():
    # pipeline._run_doublet_detection hands Scrublet a dataset of the counts when X
    # is processed (scaled, dense): the dense X and .raw are not copied. Measured on
    # Kang: +2.66 GiB, estimated +2.48 (the whole-object model said +6.0).
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    counts = sp.random(3000, 2000, density=0.1, format="csr", random_state=0, dtype=np.float32)
    counts.data = np.ceil(counts.data * 5)
    proc = ad.AnnData(X=rng.normal(size=(3000, 2000)))  # scaled, dense float64
    proc.layers["counts"] = counts
    proc.obsm["X_pca"] = rng.normal(size=(3000, 20))
    raw = ad.AnnData(X=counts.copy())
    lean = pf.step_peak_for(proc, "doublet_detection")
    whole = pf.step_peak_for(raw, "doublet_detection")
    assert lean < whole + 2 * counts.data.nbytes  # no copy of the 48 MB dense X
    dense_x = 3000 * 2000 * 8
    proc_no_counts = ad.AnnData(X=rng.normal(size=(3000, 2000)))
    proc_no_counts.obsm["X_pca"] = proc.obsm["X_pca"]
    assert pf.step_peak_for(proc_no_counts, "doublet_detection") > lean + dense_x


# --- Leiden's time and the fast-Leiden offer ----------------------------------------------
# Measured on Roihu on the same saved graph of the brain data, resolution 0.5 (seconds), and
# the laptop's live 100k run (253 s at 88,640 cells, 2 CPUs; Roihu 234 s there).
ROIHU_FULL = {88_640: 234, 221_828: 1_586, 443_946: 8_738}
ROIHU_FAST = {88_640: 17, 221_828: 44, 443_946: 138, 1_214_593: 446}
LAPTOP = 253 / 234


@pytest.mark.parametrize(("iterations", "measured"), [(-1, ROIHU_FULL), (2, ROIHU_FAST)])
def test_leiden_time_follows_the_measured_runs(iterations, measured):
    for n, secs in measured.items():
        est = pf.leiden_seconds(n, iterations)
        assert abs(est / (secs * LAPTOP) - 1) <= 0.2, (n, est, secs)
        lo, hi = pf._range(est, 2.0)
        assert lo <= secs * LAPTOP <= hi  # the range shown holds it
    assert abs(pf.leiden_seconds(88_640) / 253 - 1) <= 0.05  # the laptop's own run


def test_fast_leiden_grows_far_slower_and_small_data_keeps_its_line():
    assert pf.leiden_seconds(1_200_000, 2) < 600 < pf.leiden_seconds(1_200_000, -1) / 50
    assert pf.leiden_seconds(13_836) == pytest.approx(0.5 + 4e-4 * 13_836)  # Kang: the old line
    assert pf.leiden_seconds(13_836, 2) < pf.leiden_seconds(13_836)


def test_the_run_estimate_uses_the_runs_setting_and_no_cpu_speed_up():
    def clustering(params, cpus):
        est = pf.estimate_run((300_000, 20_000), 300_000_000, steps=["clustering"], params=params,
                              cpus=cpus)
        return next(e for e in est.steps if e.step == "clustering").seconds

    full, fast = clustering(PipelineParams(), 2.0), clustering(PipelineParams(leiden_iterations=2), 2.0)
    assert fast[1] < full[0]
    assert clustering(PipelineParams(), 8.0) == full  # Leiden uses one CPU
    louvain = clustering(PipelineParams(clustering_method="louvain"), 2.0)
    assert louvain == pf._range(0.5 + 4e-4 * 300_000, 2.0)  # igraph's multilevel: the old line
    sh = pf.ChunkedShape(300_000, 20_000, 300_000_000)
    assert pf.chunked_step("clustering", sh, PipelineParams(leiden_iterations=2))[2] == \
        pytest.approx(pf.leiden_seconds(300_000, 2))


def test_leiden_choice_offers_fast_leiden_over_ten_minutes(monkeypatch):
    monkeypatch.delenv("SCVIEW_LEIDEN_OFFER_SECONDS", raising=False)
    steps = ["neighbors", "clustering"]
    big = pf.leiden_choice(300_000, steps, PipelineParams())
    assert big["offer_fast"] and big["full_seconds"] > 600 > big["fast_seconds"]
    assert big["iterations"] == -1 and big["n_cells"] == 300_000 and big["offer_seconds"] == 600
    assert big["full_text"].startswith("roughly") and big["fast_text"].startswith("roughly")
    assert pf.leiden_choice(300_000, steps, PipelineParams(leiden_iterations=2))["iterations"] == 2
    small = pf.leiden_choice(88_640, steps, PipelineParams())
    assert not small["offer_fast"]
    # Filtering's share of the cells comes off first
    assert pf.leiden_choice(100_000, ["filtering", *steps], PipelineParams())["n_cells"] == 90_000
    # The threshold can be lowered for a check (.env)
    monkeypatch.setenv("SCVIEW_LEIDEN_OFFER_SECONDS", "120")
    assert pf.leiden_choice(88_640, steps, PipelineParams())["offer_fast"]
    assert pf.leiden_choice(300_000, ["pca"], PipelineParams()) is None  # no clustering
    assert pf.leiden_choice(300_000, steps, PipelineParams(clustering_method="louvain")) is None


async def test_dataset_preflight_reports_the_leiden_choice(client, dataset_manager, monkeypatch):
    monkeypatch.setenv("SCVIEW_LEIDEN_OFFER_SECONDS", "0")  # 40 cells: any time offers it
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/preflight"
    body = (await client.get(url, params={"steps": "neighbors,clustering"})).json()
    assert body["leiden"]["offer_fast"] and body["leiden"]["iterations"] == -1
    body = (await client.get(url, params={"steps": "neighbors,clustering",
                                          "leiden_iterations": 2})).json()
    assert body["leiden"]["iterations"] == 2
    body = (await client.get(url, params={"steps": "neighbors,clustering",
                                          "clustering_method": "louvain"})).json()
    assert body["leiden"] is None
    r = await client.get(url, params={"steps": "clustering", "leiden_iterations": 3})
    assert r.status_code == 422 and "Leiden iterations" in r.json()["detail"]


def test_umap_3d_choice_in_memory_and_in_large_dataset_mode():
    steps = ["neighbors", "embeddings"]
    assert pf.umap_3d_choice(300_000, ["pca"], PipelineParams(), False) is None  # no UMAP
    mem = pf.umap_3d_choice(50_000, steps, PipelineParams(), chunked=False)
    assert mem["on"] and mem["default_on"] and mem["setting"] == "auto" and not mem["offer"]
    big = pf.umap_3d_choice(1_160_000, steps, PipelineParams(), chunked=True)
    assert not big["on"] and not big["default_on"] and big["offer"]
    # About half the UMAP step's time at 1.16M cells (2,941 s measured on Roihu with both)
    assert 1000 < big["extra_seconds"] < 2000 and big["extra_text"].startswith("roughly")
    assert pf.umap_3d_choice(1_160_000, steps, PipelineParams(umap_3d="on"), True)["on"]
    assert not pf.umap_3d_choice(50_000, steps, PipelineParams(umap_3d="off"), False)["on"]
    assert pf.umap_3d_choice(100_000, ["filtering", *steps], PipelineParams(), False)["n_cells"] \
        == 90_000


def test_umap_time_without_the_3d_umap_is_about_half():
    both, two_d = pf.umap_seconds(1_000_000, True), pf.umap_seconds(1_000_000, False)
    assert two_d == pytest.approx(0.5 + (both - 0.5) * (1 - pf.UMAP_3D_SHARE))
    s = pf._State(1_000_000, 2000, None, {}, None, 0)
    assert pf._step_embeddings(s, PipelineParams(umap_3d="off"))[2] == pytest.approx(two_d)
    sh = pf.ChunkedShape(1_000_000, 20_000, 1_000_000_000)
    assert pf.chunked_step("embeddings", sh, pf._large_params(PipelineParams()))[2] == \
        pytest.approx(two_d)


async def test_dataset_preflight_reports_the_umap_3d_choice(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/preflight"
    body = (await client.get(url, params={"steps": "neighbors,embeddings"})).json()
    assert body["umap_3d"]["on"] and body["umap_3d"]["setting"] == "auto"
    body = (await client.get(url, params={"steps": "neighbors,embeddings", "umap_3d": "off"})).json()
    assert not body["umap_3d"]["on"]
    body = (await client.get(url, params={"steps": "neighbors,embeddings", "chunked": "on"})).json()
    assert not body["umap_3d"]["on"] and body["umap_3d"]["offer"]
    r = await client.get(url, params={"steps": "embeddings", "umap_3d": "maybe"})
    assert r.status_code == 422 and "umap_3d" in r.json()["detail"]
