"""chromVAR per-cell activity (core/atac_chromvar.py): background peaks, deviations against the definition."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import atac_chromvar as cv
from scview.core import atac_motifs as am
from scview.core import atac_import, modality_store


def _counts(n_cells=60, n_peaks=200, seed=0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.5, 0.5, n_peaks) + 0.02
    depth = rng.uniform(0.5, 2.0, n_cells)
    m = rng.poisson(np.outer(depth, base)).astype(np.float32)
    m[:, m.sum(axis=0) == 0] = 1
    return sp.csr_matrix(m)


def _hits(n_peaks=200, n_motifs=5, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.random((n_peaks, n_motifs)) < 0.2
    h[:, 4] = False  # a motif with no peak
    h[:, 3] = False
    h[17, 3] = True  # a motif with one peak
    return h


def _lists(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c = np.nonzero(h)
    ptr = np.zeros(h.shape[0] + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(r, minlength=h.shape[0]))
    return ptr, c.astype(np.int32)


def _definition(counts: np.ndarray, hits: np.ndarray, bg: np.ndarray, threshold: float = 1.0):
    """compute_deviations_single, peaks x cells in the code's own terms, one motif at a time."""
    C = counts.T  # peaks x cells
    fragments = C.sum(axis=1)
    expectation = fragments / fragments.sum()
    per_cell = C.sum(axis=0)
    n_iter = bg.shape[1]
    devs, zs = [], []
    for m in range(hits.shape[1]):
        peaks = np.flatnonzero(hits[:, m])
        if peaks.size == 0:
            devs.append(np.full(C.shape[1], np.nan))
            zs.append(np.full(C.shape[1], np.nan))
            continue
        observed = C[peaks].sum(axis=0)
        expected = expectation[peaks].sum() * per_cell
        obs_dev = (observed - expected) / expected
        sampled = np.zeros((n_iter, C.shape[1]))
        sampled_expected = np.zeros((n_iter, C.shape[1]))
        for t in range(n_iter):
            drawn = bg[peaks, t]
            sampled[t] = C[drawn].sum(axis=0)  # a peak drawn twice counts twice
            sampled_expected[t] = expectation[drawn].sum() * per_cell
        d = (sampled - sampled_expected) / sampled_expected
        norm = obs_dev - d.mean(axis=0)
        z = norm / d.std(axis=0, ddof=1)
        bad = expected < threshold
        norm[bad] = np.nan
        z[bad] = np.nan
        devs.append(norm)
        zs.append(z)
    return np.array(devs).T, np.array(zs).T


def test_kernel_matches_compute_deviations_single_with_its_nans():
    m = _counts().toarray()
    m[3] = 0  # a cell with no counts
    m[0, m.sum(axis=0) == 0] = 1
    counts = sp.csr_matrix(m)
    hits = _hits()
    fragments = np.asarray(counts.sum(axis=0)).ravel()
    bg = cv.background_peaks(fragments, np.random.default_rng(0).uniform(0.3, 0.6, 200), iterations=20)
    ptr, idx = _lists(hits)
    model = cv.make_model(ptr, idx, hits.shape[1], fragments, bg)
    dev, z = cv.deviations(model, counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        want_dev, want_z = _definition(counts.toarray().astype(np.float64), hits, bg)
    assert np.allclose(z, want_z, equal_nan=True, rtol=1e-5, atol=1e-5)
    assert np.allclose(dev, want_dev, equal_nan=True, rtol=1e-5, atol=1e-5)
    assert np.isnan(z[:, 4]).all() and np.isnan(z[3]).all()  # no peak; a cell with no counts
    assert np.isfinite(z[0, 0])


def test_block_size_does_not_change_the_answer():
    counts = _counts()
    hits = _hits()
    fragments = np.asarray(counts.sum(axis=0)).ravel()
    bg = cv.background_peaks(fragments, np.full(200, 0.4) + np.arange(200) * 1e-3, iterations=10)
    ptr, idx = _lists(hits)
    model = cv.make_model(ptr, idx, hits.shape[1], fragments, bg)
    _d, whole = cv.deviations(model, counts)
    parts = np.vstack([cv.deviations(model, counts[i:i + 7])[1] for i in range(0, counts.shape[0], 7)])
    assert np.allclose(whole, parts, equal_nan=True)


def test_background_peaks_match_on_counts_and_gc_with_the_given_seed():
    rng = np.random.default_rng(0)
    n = 4000
    fragments = np.exp(rng.normal(3, 1, n))
    gc = rng.uniform(0.3, 0.7, n)
    bg = cv.background_peaks(fragments, gc, seed=5)
    assert bg.shape == (n, cv.N_ITERATIONS) and bg.min() >= 0 and bg.max() < n
    assert np.array_equal(bg, cv.background_peaks(fragments, gc, seed=5))
    assert not np.array_equal(bg, cv.background_peaks(fragments, gc, seed=6))
    # a peak's draws sit near it in (log10 counts, GC), far nearer than random peaks
    near_gc = np.abs(gc[bg] - gc[:, None]).mean()
    near_counts = np.abs(np.log10(fragments)[bg] - np.log10(fragments)[:, None]).mean()
    far = np.abs(gc[rng.integers(0, n, n)] - gc).mean()
    assert near_gc < far / 2 and near_counts < 0.5


def test_background_peaks_use_the_grid_and_width_of_the_source():
    """With two tight clouds far apart on the whitened grid, the 0.1-wide kernel keeps draws in their own cloud."""
    rng = np.random.default_rng(1)
    fragments = np.concatenate([np.full(300, 10.0), np.full(300, 10000.0)]) * rng.uniform(0.95, 1.05, 600)
    gc = np.concatenate([np.full(300, 0.4), np.full(300, 0.6)]) + rng.normal(0, 0.002, 600)
    bg = cv.background_peaks(fragments, gc, iterations=30, seed=2)
    assert (bg[:300] < 300).mean() > 0.99 and (bg[300:] >= 300).mean() > 0.99


def test_background_peaks_refuse_a_peak_without_counts():
    with pytest.raises(ValueError, match="at least one count"):
        cv.background_peaks(np.array([1.0, 0.0, 2.0]), np.array([0.4, 0.5, 0.6]))


def test_wilcoxon_by_group_matches_scipy_and_skips_nan_and_unlabelled():
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 300)
    codes = rng.integers(0, 3, 300)
    v[codes == 1] += 0.5
    v[5] = np.nan
    codes[6] = -1
    means, rest, p = cv.wilcoxon_by_group(v, codes, 3)
    ok = np.isfinite(v) & (codes >= 0)
    want = mannwhitneyu(v[ok & (codes == 1)], v[ok & (codes != 1)], method="asymptotic", use_continuity=False).pvalue
    assert p[1] == pytest.approx(want, rel=1e-6)
    assert means[1] == pytest.approx(np.nanmean(v[ok & (codes == 1)]))
    assert rest[1] == pytest.approx(np.nanmean(v[ok & (codes != 1)]))


def test_variability_is_the_sd_of_z_with_chromvars_p():
    from scipy.stats import chi2

    z = np.array([0.0, 2.0, np.nan, -2.0, 1.0])
    sd, p = cv.variability(z, 5)
    assert sd == pytest.approx(np.nanstd(z, ddof=1))
    assert p == pytest.approx(chi2.sf(4 * sd * sd, 4))
    assert np.isnan(cv.variability(np.full(3, np.nan), 3)[0])


def _run_on_store(tmp_path, with_unknown=True) -> tuple[dict, am.PeakMotifTable, list[str]]:
    counts = _counts()
    # a peak with no count at all
    m = counts.toarray()
    m[:, 9] = 0
    counts = sp.csr_matrix(m)
    names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(200)]
    barcodes = [f"c{i}" for i in range(counts.shape[0])]
    store = modality_store.store_path(tmp_path, "d", atac_import.STORE)
    modality_store.write(store, {atac_import.MATRIX: counts}, barcodes, names, None, {"genome": "GRCh38"})
    hits = _hits()
    gc = np.random.default_rng(3).uniform(30, 60, 200)
    if with_unknown:
        gc[11] = np.nan
    table = am.PeakMotifTable(genome="GRCh38", key="k", motif_ids=np.array([f"M{i}" for i in range(5)]),
                              motif_names=np.array(["SPI1", "CEBPA", "PAX5", "ONE", "NONE"]), lengths=np.full(5, 8), thresholds=np.zeros(5),
                              gc_motif=np.array([0.4, 0.8, 0.4, 0.4, 0.4]), packed=np.packbits(hits, axis=1), gc=gc, background=np.full(4, 0.25))
    out = tmp_path / "staging"
    summary = cv.run(store, np.arange(counts.shape[0]), table, out, seed=1)
    return summary, table, barcodes


def test_run_writes_a_store_by_barcode_and_leaves_out_empty_peaks(tmp_path):
    summary, table, barcodes = _run_on_store(tmp_path)
    assert summary["n_peaks_used"] == 198 and summary["n_peaks_no_counts"] == 1 and summary["n_peaks_left_out"] == 1
    assert summary["n_cells"] == 60 and len(summary["variability"]) == 5
    with modality_store.ModalityStore(tmp_path / "staging" / f"{cv.STORE}.h5") as s:
        assert s.barcodes.tolist() == barcodes and s.features.tolist() == table.motif_ids.tolist()
        assert s.feature_attr("name").tolist() == table.motif_names.tolist()
        z = s.read(cv.Z, np.arange(60)).toarray()
        dev = s.read(cv.DEVIATIONS, np.arange(60)).toarray()
    by_motif = np.load(tmp_path / "staging" / cv.BY_MOTIF)
    assert by_motif.shape == (5, 60)
    assert np.allclose(z.T[:3], by_motif[:3], equal_nan=True)
    assert dev.shape == z.shape
    assert summary["n_nan"][4] == 60


def test_publish_reads_and_group_stats(tmp_path):
    summary, table, barcodes = _run_on_store(tmp_path)
    summary = {**summary, "cells_fingerprint": "f"}
    cv.publish(tmp_path / "staging", tmp_path, "d", summary)
    assert cv.read_summary(tmp_path, "d")["cells_fingerprint"] == "f"
    # the dataset's cells reordered and filtered: read by barcode
    obs = barcodes[10:20][::-1]
    v = cv.motif_values(tmp_path, "d", summary, "M0", obs)
    z = np.load(modality_store.store_dir(tmp_path, "d") / cv.BY_MOTIF)[0]
    assert np.allclose(v, z[10:20][::-1], equal_nan=True)
    with pytest.raises(cv.ChromVarError, match="not in the chromVAR run"):
        cv.motif_values(tmp_path, "d", summary, "M0", ["somebody else"])
    with pytest.raises(cv.ChromVarError, match="is not in the chromVAR run"):
        cv.motif_values(tmp_path, "d", summary, "MX", barcodes)
    codes = np.array([0, 1] * 30)
    out = cv.group_stats(tmp_path, "d", summary, "col", codes, ["A", "B"], barcodes, n_heatmap=3, n_rows=2)
    assert out["groups"] == ["A", "B"] and len(out["heatmap"]["motifs"]) == 3 and len(out["heatmap"]["mean_z"][0]) == 2
    assert len(out["rows"]["A"]) == 2
    a = out["rows"]["A"][0]
    j = summary["motif_ids"].index(a["motif_id"])
    zj = np.load(modality_store.store_dir(tmp_path, "d") / cv.BY_MOTIF)[j]
    assert a["mean_diff"] == pytest.approx(np.nanmean(zj[codes == 0]) - np.nanmean(zj[codes == 1]), abs=1e-3)
    assert a["tfs"]
    # kept per run and column, cleared by a new run
    assert cv.group_stats(tmp_path, "d", summary, "col", codes, ["A", "B"], barcodes) is out


def test_ranking_is_by_variability_and_skips_motifs_without_peaks(tmp_path):
    summary, _t, _b = _run_on_store(tmp_path)
    ranked = cv.ranking(summary, 10)
    assert "M4" not in [r["motif_id"] for r in ranked]
    v = [r["variability"] for r in ranked]
    assert v == sorted(v, reverse=True)


def test_a_run_with_too_few_peaks_says_so(tmp_path):
    counts = sp.csr_matrix(np.ones((3, 2), dtype=np.float32))
    store = modality_store.store_path(tmp_path, "d", atac_import.STORE)
    modality_store.write(store, {atac_import.MATRIX: counts}, ["a", "b", "c"], ["p1", "p2"], None, {})
    table = am.PeakMotifTable(genome="GRCh38", key="k", motif_ids=np.array(["M"]), motif_names=np.array(["M"]), lengths=np.array([8]),
                              thresholds=np.zeros(1), gc_motif=np.zeros(1), packed=np.zeros((2, 1), np.uint8),
                              gc=np.array([40.0, np.nan]), background=np.full(4, 0.25))
    with pytest.raises(cv.ChromVarError, match="Fewer than 2 peaks"):
        cv.run(store, np.arange(3), table, tmp_path / "o")


def test_matrix_name_is_the_stores():
    assert cv.MATRIX == atac_import.MATRIX


def test_estimates_are_within_20_percent_of_what_was_measured():
    # Through a child at 2 cores: mouse E18 240 s (41.7M values, 117.7 hits per peak), see the module.
    assert abs(cv.estimate_seconds(41_686_530, 50, 117.7) / 240 - 1) < 0.2


def test_child_process_writes_its_result(tmp_path):
    summary, table, _b = _run_on_store(tmp_path)
    table_file = tmp_path / "t.npz"
    am.save_table(table, table_file)
    store = modality_store.store_path(tmp_path, "d", atac_import.STORE)
    events = list(cv.iter_run(store, np.arange(60), table_file, tmp_path / "again", seed=1))
    assert events[-1]["type"] == "result" and events[0]["type"] == "stage" and events[0]["total"] == 3
    got = events[-1]["summary"]
    assert got["variability"] == summary["variability"]  # the same seed, the same numbers
    assert got["peak_mb"] > 0
