"""ATAC analysis from the peak store (core/atac.py): each step against its definition."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

from scview.core import atac, atac_import, gene_annotation, modality_store, wnn
from tests import multiome_fixture as mf


def _store(tmp_path: Path, peaks: sp.csr_matrix, barcodes: list[str], table: pd.DataFrame, genome: str = "GRCh38"
           ) -> modality_store.ModalityStore:
    path = atac.store_path(tmp_path, "d1")
    modality_store.write(path, {atac.MATRIX: peaks}, barcodes, table["name"].tolist(),
                         atac_import.feature_attrs(table["name"].tolist()), {"genome": genome})
    return modality_store.ModalityStore(path)


@pytest.fixture()
def world(tmp_path):
    g, p, barcodes, table = mf.matrices(n_per=40)
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, tmp_path, "GRCh38")
    a = ad.AnnData(X=g, obs=pd.DataFrame({"group": pd.Categorical(["A"] * 40 + ["B"] * 40)}, index=barcodes),
                   var=pd.DataFrame({"gene_ids": [x[1] for x in mf.GENES]}, index=[x[0] for x in mf.GENES]))
    atac_import.write_info(a, {"n_peaks": len(table), "genome": "GRCh38", "nnz": int(p.nnz), "computed": ["peak counts"]})
    store = _store(tmp_path, p, barcodes, table)
    yield {"dir": tmp_path, "adata": a, "peaks": p, "store": store, "table": table, "rows": store.rows_of(barcodes)}
    store.close()
    gene_annotation.clear_cache()


def _manual_tfidf(p: sp.csr_matrix) -> np.ndarray:
    x = p.toarray().astype(np.float64)
    cell = x.sum(axis=1, keepdims=True)
    peak = x.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.log1p(x / cell * 1e4 * (x.shape[0] / peak)))


def test_top_peaks_is_findtopfeatures_q5():
    total = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=float)
    # ecdf(0) = 2/20 = 0.10 > 0.05: with ties at the bottom even the lowest peaks can pass
    assert atac.top_peaks(total).size == 20
    assert atac.top_peaks(total, 10.0).tolist() == list(range(2, 20))
    many = np.arange(100, dtype=float)
    assert atac.top_peaks(many).tolist() == list(range(5, 100))  # ecdf(4) = 0.05 is not above it


def test_tfidf_is_signac_method_1_and_uses_totals_over_every_peak(world):
    store, rows = world["store"], world["rows"]
    stats = atac.peak_stats(store, rows)
    assert np.allclose(stats.total, np.asarray(world["peaks"].sum(axis=0)).ravel())
    assert np.allclose(stats.cell_total, np.asarray(world["peaks"].sum(axis=1)).ravel())
    selected = np.arange(0, store.n_vars, 2)  # a selection never changes the totals used
    op = atac.TfIdf(store, rows, stats, selected)
    got = sp.vstack([m for _pos, m in op.iter_blocks()]).toarray()
    assert np.allclose(got, _manual_tfidf(world["peaks"])[:, selected])


def test_blocked_products_equal_the_matrix(world):
    store, rows = world["store"], world["rows"]
    stats = atac.peak_stats(store, rows)
    selected = atac.top_peaks(stats.total)
    op = atac.TfIdf(store, rows, stats, selected, blocks=100)  # many blocks
    dense = _manual_tfidf(world["peaks"])[:, selected]
    rng = np.random.default_rng(0)
    right, left = rng.normal(size=(selected.size, 3)), rng.normal(size=(rows.size, 3))
    assert np.allclose(op.dot(right), dense @ right)
    assert np.allclose(op.tdot(left), dense.T @ left)


def test_svd_finds_the_exact_top_components(world):
    store, rows = world["store"], world["rows"]
    stats = atac.peak_stats(store, rows)
    selected = atac.top_peaks(stats.total)
    op = atac.TfIdf(store, rows, stats, selected)
    dense = _manual_tfidf(world["peaks"])[:, selected]
    u0, s0, vt0 = np.linalg.svd(dense, full_matrices=False)
    u, s, vt = atac.block_lanczos_svd(op, 5)
    assert np.allclose(s, s0[:5], rtol=1e-5)
    for i in range(5):
        assert abs(np.corrcoef(u[:, i], u0[:, i])[0, 1]) > 0.999
        assert abs(np.dot(vt[i], vt0[i])) > 0.999
    # The same answer from the same seed; each component's largest loading is positive.
    again = atac.block_lanczos_svd(op, 5)
    assert np.array_equal(again[0], u)
    assert (vt[np.arange(5), np.abs(vt).argmax(axis=1)] > 0).all()


def test_embedding_and_depth_correlation():
    rng = np.random.default_rng(1)
    depth = rng.uniform(10, 100, 200)
    u = np.column_stack([depth * 0.01 + rng.normal(0, 0.001, 200), rng.normal(size=200)])
    e = atac.scaled_embedding(u)
    assert np.allclose(e.mean(axis=0), 0) and np.allclose(e.std(axis=0, ddof=1), 1)
    r = atac.depth_correlation(e, depth)
    assert r[0] > 0.99 and abs(r[0]) > atac.DEPTH_LIMIT and abs(r[1]) < 0.3
    assert np.allclose(r[1], np.corrcoef(e[:, 1], depth)[0, 1])


def test_first_component_is_dropped_only_past_the_limit(world):
    store, rows = world["store"], world["rows"]
    stats = atac.peak_stats(store, rows)
    p = atac.plan(world["adata"], world["dir"], "d1", n_lsi=8, dims_to=6)
    res = atac.run_lsi(store, rows, p, stats)
    r0 = abs(res.depth_r[0])
    assert res.dropped_first == (r0 > atac.DEPTH_LIMIT)
    assert res.dims_used[0] == (2 if res.dropped_first else 1) and res.dims_used[-1] == 6
    p.drop_first = "no"
    assert atac.run_lsi(store, rows, p, stats).dims_used[0] == 1
    p.drop_first = "yes"
    assert atac.run_lsi(store, rows, p, stats).dims_used[0] == 2


def test_a_stopped_run_keeps_nothing_and_leaves_no_files(world):
    store, rows = world["store"], world["rows"]
    p = atac.plan(world["adata"], world["dir"], "d1", n_lsi=8, dims_to=6)
    genes = atac.choose_genes(world["adata"], "GRCh38", world["dir"])
    asked = {"n": 0}

    def stop_after_a_while() -> bool:
        asked["n"] += 1
        return asked["n"] > 12

    with pytest.raises(atac.AtacCancelled):
        for _item in atac.iter_run(store, rows, p, data_dir=world["dir"], dataset_id="d1", genes=genes,
                                   stop=stop_after_a_while):
            pass
    left = sorted(f.name for f in store.path.parent.iterdir())
    assert left == [f"{atac.STORE}.h5"]


def test_the_spool_gives_the_same_blocks_and_is_skipped_without_disk(world, monkeypatch, tmp_path):
    import shutil

    store, rows = world["store"], world["rows"]
    stats = atac.peak_stats(store, rows)
    selected = atac.top_peaks(stats.total)
    direct = atac.TfIdf(store, rows, stats, selected, blocks=100)
    spooled = atac.TfIdf(store, rows, stats, selected, blocks=100)
    assert spooled.spool(tmp_path, int(stats.n_cells[selected].sum()))
    rng = np.random.default_rng(0)
    right = rng.normal(size=(selected.size, 3))
    assert np.allclose(spooled.dot(right), direct.dot(right))
    no_disk = atac.TfIdf(store, rows, stats, selected)

    class Usage:
        free = 0

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: Usage)
    assert no_disk.spool(tmp_path, 10) is False and no_disk._spooled is None


def test_gene_regions_are_strand_aware():
    table = pd.DataFrame({"start": [10_000, 50_000, 500], "end": [20_000, 60_000, 900], "strand": ["+", "-", "+"]})
    start, end = atac.gene_regions(table)
    assert start.tolist() == [8_000, 50_000, 0]  # + extends left, floored at 0
    assert end.tolist() == [20_000, 62_000, 900]  # - extends right


def test_overlap_matrix_counts_body_and_upstream_only(world):
    genes = atac.choose_genes(world["adata"], "GRCh38", world["dir"])
    assert genes.by_id == 4 and genes.by_name == 0 and genes.names == ["GENEA", "GENEB", "GENEC", "GENED"]
    chrom, start, end = atac.peak_positions(world["store"])
    m = atac.overlap_matrix(chrom, start, end, genes).toarray()
    t = world["table"]
    for j, name in enumerate(genes.names):
        hit = set(t.loc[m[:, j] > 0, "kind"])
        assert (t.loc[m[:, j] > 0, "gene"] == name).all()
        assert hit == {"promoter", "body"}  # never the distal peak


def test_genes_match_by_name_when_ids_are_absent(world):
    a = world["adata"].copy()
    del a.var["gene_ids"]
    genes = atac.choose_genes(a, "GRCh38", world["dir"])
    assert genes.by_id == 0 and genes.by_name == 4


def test_missing_annotation_is_a_reason_not_a_failure(world):
    genes = atac.choose_genes(world["adata"], "hg19", world["dir"])
    assert genes.names == [] and "not 'hg19'" in genes.reason


def test_activity_by_cell_and_by_gene_agree_and_mark_the_right_cells(world):
    store = world["store"]
    genes = atac.choose_genes(world["adata"], "GRCh38", world["dir"])
    summary, totals = atac.write_activity(store, genes, world["dir"], "d1")
    assert summary["available"] and summary["n_genes"] == 4 and summary["n_genes_with_peaks"] == 4
    chrom, start, end = atac.peak_positions(store)
    expected = (world["peaks"] @ atac.overlap_matrix(chrom, start, end, genes)).toarray()
    with modality_store.ModalityStore(atac.store_path(world["dir"], "d1", atac.ACTIVITY_STORE)) as s:
        assert np.array_equal(s.read(atac.ACTIVITY_MATRIX, np.arange(s.n_obs)).toarray(), expected)
        assert s.meta["kind"] == "peak-based"
    with modality_store.ModalityStore(atac.store_path(world["dir"], "d1", atac.ACTIVITY_BY_GENE)) as s:
        assert list(s.barcodes) == genes.names
        assert np.array_equal(s.read(atac.ACTIVITY_MATRIX, np.arange(4)).toarray().T, expected)
    assert np.allclose(totals, expected.sum(axis=1))
    a = world["adata"]
    a.obs[atac.OBS_ACTIVITY_TOTAL] = totals.astype(np.float32)
    out = atac.activity_gene(a, world["dir"], "d1", "GENEA")
    assert np.array_equal(out["counts"], expected[:, 0])
    assert out["values"][:40].mean() > 2 * out["values"][40:].mean()
    assert np.allclose(out["values"], np.log1p(expected[:, 0] / totals * 1e4), atol=1e-5)
    with pytest.raises(atac.AtacError, match="no gene activity"):
        atac.activity_gene(a, world["dir"], "d1", "NOT_A_GENE")


def test_activity_follows_filtered_cells_by_barcode(world):
    genes = atac.choose_genes(world["adata"], "GRCh38", world["dir"])
    _summary, totals = atac.write_activity(world["store"], genes, world["dir"], "d1")
    a = world["adata"]
    a.obs[atac.OBS_ACTIVITY_TOTAL] = totals.astype(np.float32)
    whole = atac.activity_gene(a, world["dir"], "d1", "GENEC")
    keep = np.arange(a.n_obs)[::3]
    sub = a[keep].copy()
    assert np.array_equal(atac.activity_gene(sub, world["dir"], "d1", "GENEC")["values"], whole["values"][keep])
    rows = world["store"].rows_of(sub.obs_names)
    assert np.array_equal(rows, keep)


def test_activity_without_positions_says_so(tmp_path):
    p = sp.csr_matrix(np.eye(4, dtype=np.float32))
    path = atac.store_path(tmp_path, "x")
    modality_store.write(path, {atac.MATRIX: p}, list("abcd"), list("wxyz"), atac_import.feature_attrs(list("wxyz")), {})
    genes = atac.GeneChoice(["G"], ["E"], np.array(["chr1"], dtype=object), np.array([0]), np.array([9]))
    with modality_store.ModalityStore(path) as s:
        summary, _t = atac.write_activity(s, genes, tmp_path, "x")
    assert not summary["available"] and "no positions" in summary["reason"]


def test_peak_annotation_classes_and_distance():
    t = pd.DataFrame({"gene_id": ["E1", "E2", "E3"], "gene_name": ["PLUS", "MINUS", "PSEUDO"],
                      "chrom": ["chr1", "chr1", "chr1"], "start": [10_000, 50_000, 100_000],
                      "end": [20_000, 60_000, 100_500], "strand": ["+", "-", "+"],
                      "gene_type": ["protein_coding", "protein_coding", "processed_pseudogene"],
                      "tss": [10_000, 59_999, 100_000]})
    chrom = np.array(["chr1"] * 6 + ["chr9"], dtype=object)
    start = np.array([9_900, 8_000, 15_000, 30_000, 61_900, 100_100, 5])
    end = np.array([10_100, 8_500, 15_500, 30_500, 62_400, 100_200, 10])
    out = atac.annotate_peaks(chrom, start, end, t)
    assert out["peak_class"].tolist() == ["promoter", "promoter", "gene body", "distal", "promoter", "distal", "distal"]
    assert out["nearest_gene"].tolist()[:5] == ["PLUS", "PLUS", "PLUS", "PLUS", "MINUS"]
    assert out["nearest_gene"].iloc[5] == "MINUS"  # the pseudogene never counts as nearest
    d = out["distance_to_tss"].to_numpy()
    assert d[0] == 0  # the TSS is inside the peak
    assert d[1] == -1_501  # 8,000-8,499 is upstream of a + TSS at 10,000
    assert d[2] == 5_000  # downstream on +
    assert d[4] == -(61_900 - 59_999)  # 61,900 is upstream of a - TSS at 59,999
    assert np.isnan(d[6]) and out["nearest_gene"].iloc[6] == ""  # a contig without genes


def test_invalid_peak_positions_have_no_context():
    t = pd.DataFrame({"gene_id": ["E1"], "gene_name": ["G"], "chrom": ["chr1"], "start": [100], "end": [200],
                      "strand": ["+"], "gene_type": ["protein_coding"], "tss": [100]})
    out = atac.annotate_peaks(np.array(["chr1"], dtype=object), np.array([-1]), np.array([-1]), t)
    assert out["nearest_gene"].iloc[0] == "" and out["peak_class"].iloc[0] == ""


def test_wilcoxon_matches_scipy_with_zero_ties_and_ties_among_values():
    rng = np.random.default_rng(3)
    n = 200
    codes = np.array([0] * 40 + [1] * 60 + [2] * 100, dtype=np.int32)
    x = np.zeros(n)
    on = rng.random(n) < 0.3
    x[on] = np.round(rng.gamma(2, 2, on.sum()), 1) + 0.1  # rounded: ties among the values too
    x[:40][rng.random(40) < 0.5] = 3.3
    cells = np.flatnonzero(x)
    n_per = np.bincount(codes)
    z, p = atac.wilcoxon_sparse(x[cells], cells, codes, n_per, n)
    for g in range(3):
        ref = mannwhitneyu(x[codes == g], x[codes != g], method="asymptotic", use_continuity=False)
        assert p[g] == pytest.approx(ref.pvalue, rel=1e-9), g


def test_wilcoxon_leaves_out_cells_without_a_label():
    x = np.array([5.0, 0, 3, 0, 9, 1], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, -1, -1], dtype=np.int32)
    cells = np.flatnonzero(x)
    z, p = atac.wilcoxon_sparse(x[cells], cells, codes, np.array([2, 2]), 4)
    ref = mannwhitneyu([5.0, 0], [3.0, 0], method="asymptotic", use_continuity=False)
    assert p[0] == pytest.approx(ref.pvalue, rel=1e-9)


def _da(world, **kw) -> dict:
    a = world["adata"]
    codes, every, chosen = atac.group_codes(a, "group", kw.pop("groups", None))
    out = None
    for item in atac.iter_differential(world["store"], world["rows"], codes, every, chosen, **kw):
        if item["type"] == "result":
            out = item
    return out


def test_differential_finds_each_groups_peaks_and_its_bookkeeping(world):
    out = _da(world)
    f = out["frame"]
    t = world["table"]
    top_a = f[(f["group"] == "A") & (f["log2fc"] > 0)].sort_values("pval").head(6)
    assert set(t.iloc[top_a["peak_index"].to_numpy()]["gene"]) == {"GENEA", "GENEB"}
    top_b = f[(f["group"] == "B") & (f["log2fc"] > 0)].sort_values("pval").head(6)
    assert set(t.iloc[top_b["peak_index"].to_numpy()]["gene"]) == {"GENEC", "GENED"}
    assert out["n_significant"]["A"] >= 6 and out["n_tested"]["A"] <= t.shape[0]
    row = f[f["group"] == "A"].iloc[0]
    assert row["pval_adj"] >= row["pval"] and 0 <= row["pct_in"] <= 1 and row["pct_rest"] <= 1
    # Matches scipy on one peak, on TF-IDF values.
    x = _manual_tfidf(world["peaks"])[:, int(row["peak_index"])]
    labels = world["adata"].obs["group"].to_numpy()
    ref = mannwhitneyu(x[labels == "A"], x[labels != "A"], method="asymptotic", use_continuity=False)
    assert row["pval"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_differential_batches_do_not_change_the_answer(world):
    one = _da(world)["frame"]
    many = _da(world, batch_values=50)["frame"]
    pd.testing.assert_frame_equal(one.reset_index(drop=True), many.reset_index(drop=True))


def test_differential_prefilter_and_chosen_groups(world):
    out = _da(world, groups=["B"], min_pct=0.9)
    assert set(out["frame"]["group"]) == {"B"}
    assert out["n_tested"]["B"] < world["table"].shape[0]


def test_differential_context_columns(world):
    out = _da(world)
    table, note = atac.attach_context(out["frame"].head(20), world["store"], "GRCh38", world["dir"])
    assert note is None
    assert {"peak", "nearest_gene", "distance_to_tss", "peak_class"} <= set(table.columns)
    assert table["peak"].iloc[0] in set(world["table"]["name"])
    table2, note2 = atac.attach_context(out["frame"].head(5), world["store"], "hg19", world["dir"])
    assert "not 'hg19'" in note2 and (table2["nearest_gene"] == "").all()


def test_group_codes_refuses_what_cannot_be_tested(world):
    a = world["adata"]
    with pytest.raises(atac.AtacError, match="not found"):
        atac.group_codes(a, "nope")
    with pytest.raises(atac.AtacError, match="no group 'Z'"):
        atac.group_codes(a, "group", ["Z"])
    b = a.copy()
    b.obs["one"] = pd.Categorical(["x"] * b.n_obs)
    with pytest.raises(atac.AtacError, match="fewer than 2"):
        atac.group_codes(b, "one")


def test_cells_outside_the_store_are_refused(world):
    a = world["adata"].copy()
    a.obs_names = [f"other{i}" for i in range(a.n_obs)]
    with pytest.raises(atac.AtacError, match="not in its ATAC store"):
        atac.cell_rows(world["store"], a)


def test_no_atac_and_missing_store_sentences(tmp_path):
    a = ad.AnnData(X=np.zeros((60, 2), dtype=np.float32))
    with pytest.raises(atac.AtacError, match="no ATAC peaks"):
        atac.plan(a, tmp_path, "none")
    atac_import.write_info(a, {"n_peaks": 3})
    with pytest.raises(atac.AtacError, match="store of this dataset is missing"):
        atac.plan(a, tmp_path, "none")


def test_plan_limits(world):
    a = world["adata"]
    p = atac.plan(a, world["dir"], "d1")
    assert p.n_lsi == min(50, world["table"].shape[0] - 1) and p.dims_to == min(30, p.n_lsi) and p.drop_first == "auto"
    d = p.to_dict()
    assert d["estimate_seconds"] > 0 and d["estimate_bytes"] > 0
    with pytest.raises(atac.AtacError, match="Resolution"):
        atac.plan(a, world["dir"], "d1", resolution=20)
    few = a[:10].copy()
    with pytest.raises(atac.AtacError, match="at least 50 cells"):
        atac.plan(few, world["dir"], "d1")


def test_plan_nnz_scales_with_the_cells_left(world):
    a = world["adata"]
    whole = atac.plan(a, world["dir"], "d1")
    half = atac.plan(a[:60].copy(), world["dir"], "d1")
    assert half.nnz == pytest.approx(whole.nnz * 60 / 80, rel=0.01)


def test_wnn_stage_name_follows_the_second_modality_and_protein_is_unchanged():
    rng = np.random.default_rng(0)
    rna, other = rng.normal(size=(30, 5)), rng.normal(size=(30, 4))
    protein_steps = wnn.wnn_steps(rna, other, 5, 20)
    assert [s for s in _names(protein_steps)] == ["knn_rna", "knn_protein", "bandwidth", "weights", "joint_knn"]
    lsi_steps = wnn.wnn_steps(rna, other, 5, 20, second="lsi")
    assert [s for s in _names(lsi_steps)] == ["knn_rna", "knn_lsi", "bandwidth", "weights", "joint_knn"]
    a = wnn.wnn_neighbors(rna, other, 5, 20)
    b = _final(wnn.wnn_steps(rna, other, 5, 20, second="lsi"))
    assert np.array_equal(a["rna_weight"], b["rna_weight"]) and np.array_equal(a["idx"], b["idx"])


def _names(gen):
    try:
        while True:
            yield next(gen)
    except StopIteration:
        return


def _final(gen):
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def test_the_replay_script_repeats_the_run_from_the_original_upload(tmp_path):
    import gzip
    import subprocess
    import sys

    from scview.core import replay_script
    from scview.core.ingestion.session import IngestSessionManager

    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    (m.files_dir(sid) / h5.name).write_bytes(h5.read_bytes())
    did = m._build_dataset(sid)
    gene_annotation.seed_from(gtf, m.data_dir, "GRCh38")
    a = ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))
    a.obs["group"] = pd.Categorical(["A"] * 40 + ["B"] * 40)
    p = atac.plan(a, m.data_dir, did, n_lsi=8, dims_to=6, resolution=0.7)
    with atac.open_store(a, m.data_dir, did) as store:
        rows = atac.cell_rows(store, a)
        genes = atac.choose_genes(a, p.genome, m.data_dir)
        res = None
        for item in atac.iter_run(store, rows, p, data_dir=m.data_dir, dataset_id=did, genes=genes):
            if item["type"] == "result":
                res = item["result"]
    atac.save_run(a, p, res)
    a.obsm["X_pca"] = np.asarray(a.X.todense(), dtype=np.float32)[:, :4] + np.random.default_rng(0).normal(0, 0.1, (80, 4)).astype(np.float32)
    a.uns["neighbors"] = {"params": {"use_rep": "X_pca", "n_neighbors": 15}}
    wp = atac.plan_wnn(a, resolution=0.7)
    assert wp.lsi_dims == list(range(2, 9))
    steps = [h["step"] for h in atac.provenance_history(a)]
    assert steps[-1] == "atac_lsi"
    described = {h["step"]: d["status"] for h, d in zip(atac.provenance_history(a), replay_script.describe(a))}
    assert described["atac_lsi"] == "exact"

    src = tmp_path / "input" / h5.name
    src.parent.mkdir()
    src.write_bytes(h5.read_bytes())
    with gzip.open(src.parent / "gencode.v32.basic.annotation.gtf.gz", "wt") as fh:
        fh.write(mf.gtf_text())
    script, dst = tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    script.write_text(replay_script.build_script(a, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    b = ad.read_h5ad(dst)
    assert list(b.var_names) == [x[0] for x in mf.GENES]  # no peak among the genes
    assert list(b.obs_names) == list(a.obs_names)
    for i in range(8):
        assert abs(np.corrcoef(b.obsm[atac.OBSM_LSI][:, i], a.obsm[atac.OBSM_LSI][:, i])[0, 1]) > 0.999
    assert b.obs[atac.OBS_CLUSTERS].astype(str).tolist() == a.obs[atac.OBS_CLUSTERS].astype(str).tolist()
    # The SVD's sums run in another block order here, so the seeded layout is not the same to
    # the digit: the same picture, the two groups as far apart.
    labels = a.obs["group"].to_numpy()
    for u in (a.obsm[atac.OBSM_UMAP], b.obsm[atac.OBSM_UMAP]):
        apart = np.linalg.norm(u[labels == "A"].mean(axis=0) - u[labels == "B"].mean(axis=0))
        assert apart > 2 * max(u[labels == "A"].std(axis=0).max(), u[labels == "B"].std(axis=0).max())
    g = b.uns["atac_gene_activity_genes"].tolist()
    assert g == [x[0] for x in mf.GENES]
    assert np.allclose(b.obs[atac.OBS_ACTIVITY_TOTAL], a.obs[atac.OBS_ACTIVITY_TOTAL])
    gene_annotation.clear_cache()


def test_the_wnn_replay_step_is_recorded_as_exact():
    from scview.core import replay_script

    a = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    from scview.core import provenance

    provenance.record_step(a, step="atac_wnn", tool="t", params={"rep": "X_pca", "n_pcs_rna": 3, "lsi_dims": [2, 3], "k": 2})
    assert replay_script.describe(a)[-1]["status"] == "exact"
    provenance.record_step(a, step="atac_wnn", tool="t", params={})
    assert replay_script.describe(a)[-1]["status"] == "not_replayable"
    provenance.record_step(a, step="atac_lsi", tool="t", params={})
    assert replay_script.describe(a)[-1]["status"] == "not_replayable"


def test_copilot_grounds_on_the_run_with_its_tag_and_the_router_finds_it(world):
    from scview.core.claims import _FAMILY_PREFIX
    from scview.core.rag import router

    a = world["adata"]
    a.obs[atac_import.OBS_TOTAL] = np.asarray(world["peaks"].sum(axis=1), dtype=np.float32).ravel()
    info = atac_import.read_info(a)
    info["lsi"] = {"depth_r": [-0.8], "dropped_first": True, "dims_used": [2, 30], "n_selected": 70, "n_clusters": 5}
    info["activity"] = {"available": True, "n_genes": 4, "n_genes_with_peaks": 4}
    atac_import.write_info(a, info)
    a.obs[atac.OBS_CLUSTERS] = pd.Categorical(["0"] * a.n_obs)
    a.obsm[atac.OBSM_LSI] = np.zeros((a.n_obs, 3), dtype=np.float32)
    lines, _detail = atac.grounding_lines(a)
    text = " ".join(lines)
    assert "r = -0.80" in text and "left out" in text and "peak-based" in text and "[result:atac]" in text
    assert _FAMILY_PREFIX["atac"] == atac.TAG
    assert any("atac" == w or "peaks" == w for w in router._RESULT_FACET_HINTS["groups"])


def test_export_writes_cells_by_peaks_for_the_cells_left(world):
    a = world["adata"][::2].copy()
    path = atac.export_h5ad(a, world["dir"], "d1")
    out = ad.read_h5ad(path)
    assert out.shape == (40, world["table"].shape[0])
    assert list(out.obs_names) == list(a.obs_names)
    keep = np.arange(0, 80, 2)
    assert np.array_equal(out.X.toarray(), world["peaks"][keep].toarray())
    assert list(out.var["chrom"]) == world["table"]["chrom"].tolist()
    assert out.var["start"].tolist() == world["table"]["start"].tolist()
    assert (out.var["feature_types"] == "Peaks").all() and out.obs["group"].notna().all()
    assert atac.export_h5ad(a, world["dir"], "d1") == path  # kept until the cells change
    other = atac.export_h5ad(world["adata"][:60].copy(), world["dir"], "d1")
    assert other != path and not path.exists()


def test_a_reset_to_counts_leaves_no_result_in_the_summary_or_the_facts(world):
    a = world["adata"]
    info = atac_import.read_info(a)
    info["lsi"] = {"depth_r": [-0.8], "dropped_first": True, "dims_used": [2, 30], "n_selected": 70, "n_clusters": 5}
    info["wnn"] = {"mean_rna_weight": 0.5, "n_clusters": 3}
    atac_import.write_info(a, info)
    a.obs[atac.OBS_CLUSTERS] = pd.Categorical(["0"] * a.n_obs)
    a.obs[atac.OBS_WNN_WEIGHT] = 0.5
    # The columns stay after a reset; its embeddings do not.
    assert atac.summary(a)["lsi"] is None and atac.summary(a)["wnn"] is None
    text = " ".join(atac.grounding_lines(a)[0])
    assert "ATAC LSI" not in text and "WNN" not in text
    a.obsm[atac.OBSM_LSI] = np.zeros((a.n_obs, 3), dtype=np.float32)
    a.obsm[atac.OBSM_WNN] = np.zeros((a.n_obs, 2), dtype=np.float32)
    assert atac.summary(a)["lsi"]["n_clusters"] == 5 and atac.summary(a)["wnn"]["n_clusters"] == 3


def test_grounding_lines_say_nothing_without_atac_and_tag_their_facts(world):
    assert atac.grounding_lines(ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))) == ([], "")
    lines, detail = atac.grounding_lines(world["adata"])
    assert "72 peaks" in detail and all(atac.TAG in line for line in lines[1:])
