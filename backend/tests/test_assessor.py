"""Tests for preprocessing-state assessment — raw-counts detection + provenance overlay."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scview.core import provenance
from scview.core.assessor import assess_preprocessing

_RNG = np.random.default_rng(0)


def _counts(n_obs=60, n_var=40):
    return ad.AnnData(X=_RNG.poisson(3.0, (n_obs, n_var)).astype("float32"))


def test_raw_counts_not_normalized_scaled_or_logged():
    # 10x-style counts: integer-valued but float-typed.
    s = assess_preprocessing(_counts())
    for step in (s.normalization, s.log_transform, s.scaling):
        assert step.done is False
        assert step.confidence == "high"
    assert "raw" in s.normalization.details.lower()


def test_scaled_data_is_detected():
    # z-scored data: non-integer, has negatives, mean~0 std~1.
    a = ad.AnnData(X=_RNG.standard_normal((200, 50)).astype("float32"))
    assert assess_preprocessing(a).scaling.done is True


def test_lognorm_data_not_flagged_as_raw():
    a = ad.AnnData(X=np.log1p(_RNG.poisson(3.0, (60, 40)).astype("float32")))
    s = assess_preprocessing(a)
    assert s.log_transform.done is True  # values <= ~log1p(max), consistent with log1p
    assert s.scaling.done is False  # non-negative → not scaled


def test_provenance_overlay_overrides_heuristic():
    # Data still looks like raw counts, but scView recorded normalizing it →
    # the recorded fact wins.
    a = _counts()
    provenance.record_step(a, step="normalization", tool="sc.pp.normalize_total", params={})
    s = assess_preprocessing(a)
    assert s.normalization.done is True
    assert "recorded" in s.normalization.details.lower()
    # a step NOT recorded stays not-done
    assert s.scaling.done is False


def _norm_log(target, n_obs=80, n_var=60, hvg_of=None):
    c = _RNG.poisson(3.0, (n_obs, n_var)).astype("float64") + 1
    x = np.log1p(c / c.sum(1, keepdims=True) * target).astype("float32")
    return ad.AnnData(X=x)


def test_cpm_scale_warns():
    s = assess_preprocessing(_norm_log(1e6))
    d = s.normalization.details
    assert "CPM" in d and "1e6" in d
    assert "No raw counts" in d


def test_1e4_scale_no_warning():
    d = assess_preprocessing(_norm_log(1e4)).normalization.details
    assert "CPM" not in d and "WARNING" not in d
    assert "scanpy/Seurat default" in d


def test_counts_layer_reported_recoverable():
    a = _norm_log(1e6)
    a.layers["counts"] = _RNG.poisson(3.0, a.shape).astype("float32")
    assert "recoverable from layers['counts']" in assess_preprocessing(a).normalization.details


def test_hvg_subset_softens_message():
    a = _norm_log(1e4, n_var=60)
    full = _norm_log(1e4, n_var=60)[:, :10].copy()
    full.raw = a
    d = assess_preprocessing(full).normalization.details
    assert "gene subset" in d or "gene subsetting" in d
    assert "recoverable" not in d  # raw here is normalised, not counts


def test_raw_counts_have_no_scale_note():
    d = assess_preprocessing(_counts()).normalization.details
    assert "Median per-cell total" not in d


def test_sample_totals_in_chunks_equal_the_whole_sample():
    # _scale_note reads its 2,000-cell sample in chunks of rows (the whole sample
    # dense, with its copies, took 1 GB on ovary's 20k genes).
    import scipy.sparse as sp

    from scview.core import assessor

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, (600, 50)).astype(np.float64)
    logx = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    for X in (logx, sp.csr_matrix(logx)):
        logged, totals = assessor._sample_totals(X, 600)  # 3 chunks, the last one partial
        assert logged and np.allclose(totals, np.expm1(logx).sum(axis=1))
    logged, totals = assessor._sample_totals(counts * 50, 600)
    assert not logged and np.allclose(totals, (counts * 50).sum(axis=1))
    assert assessor._sample_totals(logx - 1.0, 600) is None  # scaled: negative values


def test_seurat_qc_columns_are_named_but_the_step_still_runs():
    a = _counts()
    a.obs["nCount_RNA"] = np.asarray(a.X.sum(axis=1)).ravel()
    a.obs["nFeature_RNA"] = np.asarray((a.X > 0).sum(axis=1)).ravel()
    a.obs["percent.mt"] = 1.0
    q = assess_preprocessing(a).qc_metrics
    assert q.done is False  # scanpy's own columns are what other steps read
    assert "nCount_RNA, nFeature_RNA, percent.mt" in q.details and "Seurat" in q.details


def test_no_qc_columns_says_so():
    q = assess_preprocessing(_counts()).qc_metrics
    assert q.done is False and q.details == "No standard QC metric columns found in obs"


def test_mito_columns_under_other_tools_names_are_found():
    from scview.core.assessor import mito_columns

    names = ["pct_counts_mt", "percent.mt", "mitochondrial_percent", "subsets_Mito_percent", "mito_frac",
             "pct_counts_mito", "percent_mito", "pct_mt", "n_genes", "mt_genes", "pct_counts_in_top_50_genes"]
    assert mito_columns(names) == sorted(names[:8])
    a = _counts()
    a.obs["n_genes"] = 5
    a.obs["mitochondrial_percent"] = 0.5  # CELLxGENE (Emont 2022 mouse fat)
    q = assess_preprocessing(a).qc_metrics
    assert q.done and "mitochondrial_percent" in q.details and "no MT%" not in q.details


def test_cellxgene_cell_cycle_columns_named_and_step_still_runs():
    a = _counts()
    a.obs["cell_cycle_s_score"] = 0.2
    a.obs["cell_cycle_g2m_score"] = -0.1
    a.obs["cell_cycle_phase"] = "S"
    c = assess_preprocessing(a).cell_cycle
    assert c.done is False  # regress-out reads S_score / G2M_score, which the step adds
    assert "cell_cycle_g2m_score, cell_cycle_phase, cell_cycle_s_score" in c.details
    assert "No cell cycle scores" not in c.details
    # Seurat's names too; scanpy's own columns still mean done
    b = _counts()
    b.obs["S.Score"], b.obs["G2M.Score"], b.obs["Phase"] = 0.1, 0.1, "G1"
    assert "S.Score" in assess_preprocessing(b).cell_cycle.details
    b.obs["S_score"], b.obs["G2M_score"] = 0.1, 0.1
    assert assess_preprocessing(b).cell_cycle.done is True


def test_cellxgene_cell_cycle_scores_feed_the_cycling_check():
    from scview.core import cell_cycle

    a = _counts()
    a.obs["cell_cycle_s_score"] = np.linspace(-0.2, 0.4, a.n_obs)
    a.obs["cell_cycle_g2m_score"] = 0.0
    calls = cell_cycle.cycling_cells(a.obs)
    assert calls is not None and calls.sum() == int((a.obs["cell_cycle_s_score"] >= 0.1).sum())


def test_author_harmony_embedding_counts_as_batch_correction():
    a = _counts()
    a.obsm["X_harmony"] = _RNG.standard_normal((a.n_obs, 5))
    b = assess_preprocessing(a).batch_correction
    assert b.done is True and b.confidence == "medium" and "X_harmony with 5 dimensions" in b.details
    assert assess_preprocessing(_counts()).batch_correction.done is False


def test_doublet_id_column_is_read_as_doublet_calls():
    from scview.core import doublet_source

    a = _counts()
    a.obs["doublet_id"] = ["Singlet"] * a.n_obs  # Wilson 2022 kidney: doublets removed upstream
    d = assess_preprocessing(a).doublet_detection
    assert d.done is True and "'doublet_id' (0 of 60 cells" in d.details and "removed" in d.details
    a.obs["doublet_id"] = ["Doublet"] * 3 + ["Singlet"] * (a.n_obs - 3)
    assert "(3 of 60 cells called doublets)" in assess_preprocessing(a).doublet_detection.details
    used = doublet_source.resolve(a.obs)
    assert used is not None and used.n == 3 and used.source == "doublet_id"
