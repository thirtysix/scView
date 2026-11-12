"""Peak-gene links (core/atac_links.py, api/v1/atac_links.py): Signac's numbers on a small genome."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import atac as atac_api
from scview.core import atac, atac_import, atac_links, gene_annotation, genome_seq, modality_store, provenance, results_store
from scview.core.modality_store import ModalityStore
from tests.genome_fixture import install
from tests.test_hvg_sensitivity import _register

API = "/api/v1"
N_CELLS = 240

# name, gene id, chrom, GTF start (1-based), end, strand
GENES = [("GENEA", "ENSG01", "chr1", 100_001, 110_000, "+"),
         ("GENEB", "ENSG02", "chr1", 600_001, 610_000, "-"),
         ("GENEC", "ENSG03", "chr2", 50_001, 60_000, "+"),
         ("RARE", "ENSG04", "chr2", 80_001, 90_000, "+"),
         ("LONELY", "ENSG05", "chr3", 900_001, 900_500, "+"),
         ("NOPOS", "ENSG99", "chrZ", 1, 100, "+")]


def _gtf() -> str:
    lines = []
    for name, gid, chrom, start, end, strand in GENES[:-1]:
        lines.append(f'{chrom}\ttest\tgene\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gid}.1"; gene_type "protein_coding"; gene_name "{name}";')
    return "\n".join(lines) + "\n"


def _world(seed: int = 3):
    """Counts of peaks every 10 kb on chr1 and every 5 kb on chr2 and chr3's start, a peak near GENEA and one near GENEC
    open with their gene, three peaks on a contig the genome does not name."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(100):
        s = 5_000 + i * 10_000
        rows.append(("chr1", s, s + 300 + (i % 40) * 3))
    for i in range(60):
        s = 2_000 + i * 5_000
        rows.append(("chr2", s, s + 300 + (i % 30) * 3))
    for i in range(60):
        s = 2_000 + i * 5_000
        rows.append(("chr3", s, s + 300 + (i % 30) * 3))
    rows.append(("chr3", 899_000, 899_400))  # LONELY's only peak within 500 kb is too far from these
    rows += [("KI270728.1", 1_000 + 2_000 * i, 1_500 + 2_000 * i) for i in range(3)]
    table = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    names = [f"{c}:{s}-{e}" for c, s, e in rows]
    state = rng.random(N_CELLS) < 0.5
    lam = rng.gamma(2.0, 0.3, len(rows))
    peaks = rng.poisson(lam, (N_CELLS, len(rows))).astype(np.float32)
    table["name"] = names
    driven_a = int(np.flatnonzero((table["chrom"] == "chr1") & (table["start"] == 105_000))[0])
    driven_c = int(np.flatnonzero((table["chrom"] == "chr2") & (table["start"] == 52_000))[0])
    peaks[:, driven_a] = rng.poisson(0.1 + 3.0 * state)
    peaks[:, driven_c] = rng.poisson(0.1 + 3.0 * ~state)
    genes = rng.poisson(0.3, (N_CELLS, len(GENES))).astype(np.float32)
    genes[:, 0] += rng.poisson(6.0 * state)
    genes[:, 2] += rng.poisson(6.0 * ~state)
    genes[:, 3] = 0
    genes[:4, 3] = 2  # RARE: in 4 cells
    x = np.log1p(genes / np.maximum(genes.sum(axis=1, keepdims=True), 1) * 1e4).astype(np.float32)
    barcodes = [f"c{i}" for i in range(N_CELLS)]
    obs = pd.DataFrame({"group": pd.Categorical(np.where(state, "A", "B"))}, index=barcodes)
    var = pd.DataFrame({"gene_ids": [g[1] for g in GENES], "highly_variable": [True, True, True, True, True, False]},
                       index=[g[0] for g in GENES])
    a = ad.AnnData(X=sp.csr_matrix(x), obs=obs, var=var)
    a.obs[atac_import.OBS_TOTAL] = peaks.sum(axis=1)
    a.obs[atac_import.OBS_N_PEAKS] = (peaks > 0).sum(axis=1).astype(np.float32)
    atac_import.write_info(a, {"n_peaks": len(rows), "genome": "GRCh38", "nnz": int((peaks > 0).sum()), "computed": ["peak counts"]})
    return a, sp.csr_matrix(peaks), table, x


def _sequences(seed: int = 5) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    out = {}
    for chrom, size in (("chr1", 1_020_000), ("chr2", 310_000), ("chr3", 910_000)):
        out[chrom] = "".join(np.array(list("ACGT"))[rng.integers(0, 4, size)].tolist())
    return out


@pytest.fixture(scope="module")
def sequences():
    return _sequences()


@pytest.fixture()
def seeded(dataset_manager, tmp_path, monkeypatch, sequences):
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(_gtf())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, dataset_manager.data_dir, "GRCh38")
    install(monkeypatch, dataset_manager.data_dir, "GRCh38", sequences)
    yield dataset_manager
    gene_annotation.clear_cache()
    genome_seq.clear_cache()


def _seed_store(data_dir, did, p, barcodes, table):
    modality_store.write(atac.store_path(data_dir, did), {atac.MATRIX: p}, barcodes, table["name"].tolist(),
                         atac_import.feature_attrs(table["name"].tolist()), {"genome": "GRCh38"})


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def _core(tmp_path, seed=atac_links.SEED, names=None, mask=None, gc=None, n_sample=atac_links.N_SAMPLE):
    """iter_links on the world directly, GC from the feature's position hash so no genome is needed."""
    a, p, table, x = _world()
    store_file = tmp_path / "peaks.h5"
    _seed_store_file(store_file, p, list(a.obs_names), table)
    chrom, start, end = table["chrom"].to_numpy(dtype=object), table["start"].to_numpy(), table["end"].to_numpy()
    if gc is None:
        gc = 30.0 + (start % 37) * 0.7
        gc[chrom == "KI270728.1"] = np.nan
    gene_table = pd.DataFrame([(g[0], g[1], g[2], g[3] - 1 if g[5] == "+" else g[4] - 1, g[5]) for g in GENES if g[0] != "NOPOS"],
                              columns=["name", "gene_id", "chrom", "tss", "strand"])
    if names is not None:
        gene_table = gene_table[gene_table["name"].isin(names)]
    gene_table = atac_links.order_genes(gene_table)
    index = pd.Index([g[0] for g in GENES])

    def read(wanted):
        return x[:, index.get_indexer(wanted)]

    folder = tmp_path / "spool"
    folder.mkdir(exist_ok=True)
    rows = np.arange(N_CELLS) if mask is None else np.flatnonzero(mask)
    with ModalityStore(store_file) as store:
        items = list(atac_links.iter_links(store, rows, mask, read, gene_table, store.features, chrom, start, end, gc, folder, seed=seed,
                                               n_sample=n_sample))
    return items[-1], a, p, table, x


def _seed_store_file(path, p, barcodes, table):
    modality_store.write(path, {atac.MATRIX: p}, barcodes, table["name"].tolist(), atac_import.feature_attrs(table["name"].tolist()),
                         {"genome": "GRCh38"})


# ---------------------------------------------------------------------------
# MatchRegionStats
# ---------------------------------------------------------------------------


def test_density_is_rs_density_with_bandwidth_one():
    # R 4.6.1: d <- density(c(10, 11, 13, 40), kernel = "gaussian", bw = 1); round(d$y[c(1, 50, 100, 300, 512)], 12)
    x, y = atac_links.r_density(np.array([10.0, 11.0, 13.0, 40.0]))
    assert x[0] == 7.0 and x[-1] == 43.0 and x.size == 512
    np.testing.assert_allclose(y[[0, 49, 99, 299, 511]], R_DENSITY_Y, atol=2e-12)


# R 4.6.1, the same call; see the test above. Weights are checked on real peaks against R in the batch's report.
R_DENSITY_Y = [0.00115694415112865, 0.17959571214859, 0.0632709046184478, 0.0, 0.00112285002679575]


def test_weights_are_the_product_of_each_features_query_over_all():
    features = np.array([[40.0, 100.0, 500.0], [41.0, 101.0, 501.0], [60.0, 100.0, 500.0], [40.0, 5000.0, 500.0], [40.0, 100.0, 900.0]])
    s = atac_links.BackgroundSampler(features, np.array([1, 2, 3, 4]))
    w = s.weights(0)
    v = features[[1, 2, 3, 4]]
    want = np.ones(4)
    for f in range(3):
        gx, gy = atac_links.r_density(v[:, f])
        qx, qy = atac_links.r_density(features[[0], f])
        q = np.interp(v[:, f], qx, qy, left=atac_links.OUTSIDE, right=atac_links.OUTSIDE)
        want *= q / np.interp(v[:, f], gx, gy, left=atac_links.OUTSIDE, right=atac_links.OUTSIDE)
    np.testing.assert_allclose(w, want, rtol=1e-9)
    assert w[0] > 1e3 * w[1:].max()  # within +-3 of all three features beats anything outside


def test_a_draw_follows_the_weights_and_its_seed_alone():
    rng = np.random.default_rng(0)
    features = np.column_stack([rng.normal(45, 5, 2000), rng.integers(10, 400, 2000).astype(float), rng.integers(200, 900, 2000).astype(float)])
    others = np.arange(1, 2000)
    s = atac_links.BackgroundSampler(features, others)
    one = s.draw(0, 777, seed=0)
    assert one.size == atac_links.N_SAMPLE and np.unique(one).size == one.size and 0 not in one
    np.testing.assert_array_equal(one, s.draw(0, 777, seed=0))
    assert not np.array_equal(one, s.draw(0, 777, seed=1))
    assert not np.array_equal(one, s.draw(0, 778, seed=0))
    # The draw leans on the weights: what it took is far heavier than what it left.
    w = s.weights(0)
    taken = np.isin(others, one)
    assert np.log(w[taken]).mean() > np.log(w[~taken]).mean() + 1


def test_a_draw_never_takes_a_zero_weight_and_fewer_candidates_are_all_taken():
    features = np.array([[40.0, 1.0, 5.0], [40.0, 1.0, 5.0], [41.0, 2.0, 5.0], [42.0, 2.0, 6.0]])
    s = atac_links.BackgroundSampler(features, np.array([1, 2, 3]))
    assert sorted(s.draw(0, 0).tolist()) == [1, 2, 3]
    s.base[1] = 0.0
    assert sorted(s.draw(0, 0).tolist()) == [1, 3]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_scores_are_pearson_with_signacs_defaults(tmp_path):
    result, _a, p, table, x = _core(tmp_path)
    t, c = result["frame"], result["counts"]
    assert {"GENEA", "GENEC"} <= set(t["gene"])
    dense = p.toarray().astype(np.float64)
    where = {n: i for i, n in enumerate(table["name"])}
    genes = {g[0]: i for i, g in enumerate(GENES)}
    for row in t.itertuples():
        want = np.corrcoef(dense[:, where[row.peak]], x[:, genes[row.gene]].astype(np.float64))[0, 1]
        assert row.score == pytest.approx(want, abs=1e-12)
    assert abs(t["score"]).min() > 0.05
    assert (t["p"] < 0.05).all()
    a_top = t[t["gene"] == "GENEA"].sort_values("score", ascending=False).iloc[0]
    assert a_top["peak"] == "chr1:105000-105330" and a_top["score"] > 0.5 and a_top["z"] > 5
    c_top = t[t["gene"] == "GENEC"].sort_values("score", ascending=False).iloc[0]
    assert c_top["peak"] == "chr2:52000-52330" and c_top["distance_to_tss"] == 2000.0
    assert c["n_genes_asked"] == 5 and c["n_genes_few_cells"] == 1  # RARE: in 4 cells, not more than 10
    assert c["n_genes_few_peaks"] == 1  # LONELY: one peak within 500 kb
    assert c["n_peaks_unknown"] == 3
    assert c["n_pairs_tested"] >= c["n_links"] > 0 and c["n_pairs_cis"] >= c["n_pairs_tested"]
    assert not t["peak"].str.startswith("KI270728").any()


def test_z_is_the_backgrounds_mean_and_sd_and_bh_covers_every_tested_pair(tmp_path):
    result, _a, p, table, x = _core(tmp_path)
    t = result["frame"]
    row = t[t["gene"] == "GENEA"].sort_values("score", ascending=False).iloc[0]
    dense = p.toarray().astype(np.float64)
    keep = np.flatnonzero(((dense > 0).sum(axis=0) > 10) & ~table["chrom"].eq("KI270728.1").to_numpy())
    start, end = table["start"].to_numpy(), table["end"].to_numpy()
    gc = 30.0 + (start % 37) * 0.7
    features = np.column_stack([gc[keep], dense[:, keep].sum(axis=0), (end - start + 1)[keep].astype(float)])
    chrom = table["chrom"].to_numpy(dtype=object)[keep]
    s = atac_links.BackgroundSampler(features, np.flatnonzero(chrom != "chr1"))
    pos = int(np.flatnonzero(keep == int(row["peak_index"]))[0])
    bg = s.draw(pos, int(row["peak_index"]))
    r_bg = np.array([np.corrcoef(dense[:, keep[b]], x[:, 0].astype(np.float64))[0, 1] for b in bg])
    assert row["z"] == pytest.approx((row["score"] - r_bg.mean()) / r_bg.std(ddof=1), rel=1e-9)
    from scipy.special import ndtr

    assert row["p"] == pytest.approx(float(ndtr(-abs(row["z"]))))
    # BH is over the pairs tested, not over the links kept: the adjusted p is never below the raw one by rank among links alone.
    assert (t["padj"] >= t["p"] - 1e-15).all()
    assert result["counts"]["n_links_padj"] == int((t["padj"] < 0.05).sum())


def test_a_genes_z_does_not_depend_on_the_other_genes_and_a_new_seed_only_moves_z(tmp_path):
    # Fewer than the world's 120 other-chromosome peaks: with all of them no seed could matter.
    full = _core(tmp_path, n_sample=30)[0]["frame"]
    alone = _core(tmp_path, names=["GENEA"], n_sample=30)[0]["frame"]
    both = full[full["gene"] == "GENEA"].set_index("peak").loc[alone["peak"]]
    np.testing.assert_array_equal(both["z"].to_numpy(), alone["z"].to_numpy())
    np.testing.assert_array_equal(both["score"].to_numpy(), alone["score"].to_numpy())
    other = _core(tmp_path, seed=9, n_sample=30)[0]["frame"]
    shared = full.merge(other, on=["gene", "peak"], suffixes=("", "_b"))
    np.testing.assert_array_equal(shared["score"].to_numpy(), shared["score_b"].to_numpy())
    assert not np.array_equal(shared["z"].to_numpy(), shared["z_b"].to_numpy())


def test_window_is_signacs_overlap_of_a_one_based_peak(tmp_path):
    """TSS 100,001 (1-based): the window reaches 600,001, a peak named with that start is in, 600,002 is out; on the
    left it reaches -399,999, so test the right edge. Peak names parse as 1-based inclusive."""
    a, p, table, x = _world()
    edge = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [600_001, 600_002], "end": [600_300, 600_300]})
    edge["name"] = [f"chr1:{s}-{e}" for s, e in zip(edge["start"], edge["end"])]
    keep = table["chrom"].isin(["chr2", "chr3"]).to_numpy()
    use = pd.concat([edge.iloc[[0]], edge.iloc[[1]], table[keep]], ignore_index=True)
    rng = np.random.default_rng(1)
    gene = x[:, 0]
    open_cells = rng.poisson(0.5 + 2 * (gene > 0), (2, N_CELLS)).astype(np.float32).T
    peaks = sp.hstack([sp.csr_matrix(open_cells), p[:, keep]]).tocsr()
    store_file = tmp_path / "edge.h5"
    _seed_store_file(store_file, peaks, list(a.obs_names), use)
    gc = 30.0 + (use["start"].to_numpy() % 37) * 0.7
    genes = pd.DataFrame([("GENEA", "ENSG01", "chr1", 100_000, "+")], columns=["name", "gene_id", "chrom", "tss", "strand"])
    folder = tmp_path / "s"
    folder.mkdir()
    with ModalityStore(store_file) as store:
        out = list(atac_links.iter_links(store, np.arange(N_CELLS), None, lambda _w: x[:, [0]], genes, store.features,
                                         use["chrom"].to_numpy(dtype=object), use["start"].to_numpy(), use["end"].to_numpy(), gc, folder))
    c = out[-1]["counts"]
    # Only one candidate on chr1 is within the window, so the gene has fewer than 2 and is skipped, as Signac does.
    assert c["n_genes_few_peaks"] == 1 and c["n_genes_tested"] == 0


def test_gc_uses_signacs_parse_of_the_10x_name(tmp_path):
    from tests.genome_fixture import write_2bit

    path = write_2bit(tmp_path / "t.2bit", {"chr1": "A" * 9 + "G" + "A" * 10, "chr2": "C" * 20})
    two = genome_seq.TwoBit(path)
    chrom = np.array(["chr1", "chr2", "KI270728.1", "chr1"], dtype=object)
    start, end = np.array([10, 2, 2, -1]), np.array([14, 3, 3, -1])
    gc = atac_links.peak_gc(two, chrom, start, end)
    # chr1:10-14 is bases 9..13 0-based half open: the G and four A's, 5 bases.
    assert gc[0] == pytest.approx(20.0) and gc[1] == 100.0 and np.isnan(gc[2]) and np.isnan(gc[3])
    assert atac_links.peak_widths(start[0], end[0]) == 5
    two.close()


def test_a_chosen_group_is_the_only_cells_counted(tmp_path):
    a = _world()[0]
    mask = (a.obs["group"] == "A").to_numpy()
    result, _a, p, table, x = _core(tmp_path, mask=mask)
    assert result["counts"]["n_cells"] == int(mask.sum())
    dense = p.toarray().astype(np.float64)[mask]
    where = {n: i for i, n in enumerate(table["name"])}
    genes = {g[0]: i for i, g in enumerate(GENES)}
    for row in result["frame"].head(20).itertuples():
        want = np.corrcoef(dense[:, where[row.peak]], x[mask, genes[row.gene]].astype(np.float64))[0, 1]
        assert row.score == pytest.approx(want, abs=1e-12)


def test_a_stop_ends_the_run_and_leaves_nothing_in_memory_maps(tmp_path):
    a, p, table, x = _world()
    path = tmp_path / "peaks.h5"
    _seed_store_file(path, p, list(a.obs_names), table)
    gc = np.full(len(table), 40.0)
    genes = pd.DataFrame([("GENEA", "ENSG01", "chr1", 100_000, "+")], columns=["name", "gene_id", "chrom", "tss", "strand"])
    with ModalityStore(path) as store, pytest.raises(atac.AtacCancelled):
        list(atac_links.iter_links(store, np.arange(N_CELLS), None, lambda _w: x[:, [0]], genes, store.features,
                                   table["chrom"].to_numpy(dtype=object), table["start"].to_numpy(), table["end"].to_numpy(), gc,
                                   tmp_path, stop=lambda: True))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _register_world(dm, did: str = "lk"):
    a, p, table, x = _world()
    _register(dm, a, did)
    _seed_store(dm.data_dir, did, p, list(a.obs_names), table)
    return did, a


async def _run(client, did, **body):
    r = await client.post(f"{API}/datasets/{did}/atac/links", json=body)
    assert r.status_code == 200, r.text
    return _events(r.text)


async def test_estimate_says_what_a_run_needs_and_a_first_download(client, seeded):
    did, _a = _register_world(seeded)
    r = await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["n_genes"] == 5 and e["n_cells"] == N_CELLS and e["estimate_seconds"] > 0 and e["estimate_bytes"] > 0
    assert e["estimate_disk_bytes"] > 0 and e["genome"]["downloaded"] is True
    (Path(seeded.data_dir) / genome_seq.SUBDIR / "hg38.2bit").unlink()
    late = (await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={})).json()
    assert late["genome"]["downloaded"] is False and late["estimate_seconds"] > e["estimate_seconds"]
    r = await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={"genes": "list", "gene_list": ["GENEA", "NOTHING"]})
    assert r.json()["missing"] == ["NOTHING"] and r.json()["n_genes"] == 1
    r = await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={"cells_column": "group", "cells_value": ["A"]})
    assert r.json()["n_cells"] == int((_a.obs["group"] == "A").sum())


async def test_run_kept_paged_by_gene_csv_locus_and_a_replay_step(client, seeded):
    did, _a = _register_world(seeded)
    events = await _run(client, did)
    assert [e for e, _ in events][0] == "start" and [e for e, _ in events][-1] == "complete", events[-1]
    assert any(e == "progress" and d.get("step") == "links" for e, d in events)
    result = next(d for e, d in events if e == "result")
    rid = result["id"]
    assert result["counts"]["n_links"] > 0 and result["counts"]["n_peaks_unknown"] == 3 and result["stale"] is False
    assert "never proof" in result["caption"] and "cell types" in result["caption"]
    assert result["seconds"] >= 0 and result["params"]["seed"] == atac_links.SEED

    page = (await client.get(f"{API}/datasets/{did}/atac/links/{rid}", params={"gene": "genea"})).json()
    assert page["total"] >= 1 and {r["gene"] for r in page["rows"]} == {"GENEA"}
    assert set(page["rows"][0]) == {"gene", "peak", "distance_to_tss", "score", "z", "p", "padj"}
    # A p between 1e-15 and 1e-10 survives the JSON (pandas' default 10 decimals made it 0).
    res = atac_links.recall(did, rid)
    tiny = atac_links.LinksResult("t", res.table.assign(p=3.2e-12), res.counts, res.params, "m", "c", "h", "t", 1.0)
    assert atac_links.payload(tiny)["rows"][0]["p"] == pytest.approx(3.2e-12)
    ranked = (await client.get(f"{API}/datasets/{did}/atac/links/{rid}", params={"sort": "abs_score", "descending": "true", "limit": 3})).json()
    assert abs(ranked["rows"][0]["score"]) >= abs(ranked["rows"][-1]["score"]) and ranked["total"] == result["counts"]["n_links"]
    bad = await client.get(f"{API}/datasets/{did}/atac/links/{rid}", params={"sort": "nope"})
    assert bad.status_code == 422 and "cannot be sorted" in bad.json()["detail"]
    found = (await client.get(f"{API}/datasets/{did}/atac/links/{rid}", params={"search": "chr2:52000"})).json()
    assert all("chr2:52000" in r["peak"] for r in found["rows"]) and found["total"] >= 1

    csv = await client.get(f"{API}/datasets/{did}/atac/links/{rid}/csv")
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("gene,peak,chrom,start,end,tss,distance_to_tss,score,z,p,padj")
    assert len(csv.text.splitlines()) == result["counts"]["n_links"] + 1

    locus = (await client.get(f"{API}/datasets/{did}/atac/links/{rid}/locus", params={"gene": "GENEA"})).json()
    assert locus["gene"] == "GENEA" and locus["tss"] == 100_000 and locus["chrom"] == "chr1"
    assert locus["start"] <= min(a["middle"] for a in locus["arcs"]) and locus["end"] >= max(a["middle"] for a in locus["arcs"])
    assert locus["start"] <= 100_000 <= locus["end"] and any(p["start"] == 105_000 for p in locus["peaks"])
    assert any(g["name"] == "GENEA" for g in locus["genes"])
    none = await client.get(f"{API}/datasets/{did}/atac/links/{rid}/locus", params={"gene": "RARE"})
    assert none.status_code == 404 and "no links for 'RARE'" in none.json()["detail"]

    latest = (await client.get(f"{API}/datasets/{did}/atac/links")).json()
    assert latest["result"]["id"] == rid
    adaptor = await seeded.get_or_load_dataset(did)
    step = provenance.read_provenance(adaptor.adata)["history"][-1]
    assert step["step"] == "atac_links" and step["params"]["result_id"] == rid
    from scview.core import replay_script

    assert replay_script.describe(adaptor.adata)[-1]["status"] == "exact"


async def test_a_new_cell_set_makes_a_result_stale_and_a_relabel_too(client, seeded):
    did, _a = _register_world(seeded)
    events = await _run(client, did, cells_column="group", cells_value=["A"])
    rid = next(d for e, d in events if e == "result")["id"]
    assert (await client.get(f"{API}/datasets/{did}/atac/links")).json()["result"]["id"] == rid
    # A relabel records no step, but these are no longer the cells of 'A'.
    adaptor = await seeded.get_or_load_dataset(did)
    labels = adaptor.adata.obs["group"].astype(str).to_numpy().copy()
    labels[:20] = np.where(labels[:20] == "A", "B", "A")
    adaptor.adata.obs["group"] = pd.Categorical(labels)
    body = (await client.get(f"{API}/datasets/{did}/atac/links")).json()
    assert body["result"] is None
    assert (await client.get(f"{API}/datasets/{did}/atac/links/{rid}")).json()["stale"] is True
    # A pipeline step that is not additive: the kept run stays reachable, marked.
    adaptor.adata.obs["group"] = pd.Categorical(_a.obs["group"].astype(str).to_numpy())
    provenance.record_step(adaptor.adata, step="clustering", tool="t", params={})
    body = (await client.get(f"{API}/datasets/{did}/atac/links")).json()
    assert body["result"] is None and body["runs"][0]["stale"] is True
    assert (await client.get(f"{API}/datasets/{did}/atac/links/{rid}")).json()["stale"] is True


async def test_the_links_step_leaves_earlier_results_current():
    assert "atac_links" in results_store.ADDITIVE_STEPS


async def test_errors_are_one_sentence(client, seeded, monkeypatch):
    did, _a = _register_world(seeded)
    r = await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={"cells_column": "nope", "cells_value": ["A"]})
    assert r.status_code == 404 and "'nope'" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/atac/links", json={"cells_column": "group"})
    assert r.status_code == 422 and "Pick which values" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/atac/links", json={"genes": "list", "gene_list": ["NOTHING"]})
    assert r.status_code == 404 and "NOTHING" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/atac/links", json={"genes": "list", "gene_list": []})
    assert r.status_code == 422
    missing = await client.get(f"{API}/datasets/{did}/atac/links/ffffffff")
    assert missing.status_code == 404 and "gone" in missing.json()["detail"]

    from scview.config import get_settings

    (Path(seeded.data_dir) / genome_seq.SUBDIR / "hg38.2bit").unlink()
    genome_seq.clear_cache()
    monkeypatch.setattr(get_settings(), "GENOME_AUTO_DOWNLOAD", False)
    events = await _run(client, did)
    err = next(d for e, d in events if e == "error")["error"]
    assert "GENOME_AUTO_DOWNLOAD" in err and "\n" not in err


async def test_rna_that_is_not_log_normalised_is_refused(client, seeded):
    a, p, table, _x = _world()
    a.X = sp.csr_matrix(np.rint(a.X.toarray() * 3).astype(np.float32))
    _register(seeded, a, "cnt")
    _seed_store(seeded.data_dir, "cnt", p, list(a.obs_names), table)
    r = await client.post(f"{API}/datasets/cnt/atac/links", json={})
    assert r.status_code == 422 and "log-normalised" in r.json()["detail"]


async def test_no_variable_genes_says_so(client, seeded):
    a, p, table, _x = _world()
    del a.var["highly_variable"]
    _register(seeded, a, "nohv")
    _seed_store(seeded.data_dir, "nohv", p, list(a.obs_names), table)
    r = await client.post(f"{API}/datasets/nohv/atac/links", json={})
    assert r.status_code == 422 and "variable genes" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/nohv/atac/links", json={"genes": "all"})
    assert [e for e, _ in _events(r.text)][-1] == "complete"


async def test_one_run_at_a_time(client, seeded):
    did, _a = _register_world(seeded)
    atac_api._running[did] = "differential"
    try:
        r = await client.post(f"{API}/datasets/{did}/atac/links", json={})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
    finally:
        atac_api._running.pop(did, None)


async def test_a_run_reserves_what_it_said_and_lets_go(client, seeded, monkeypatch):
    did, _a = _register_world(seeded)
    seen = {}
    real = seeded.reserve_run

    def spy(dataset_id, need, remaining=None):
        seen["need"] = need
        return real(dataset_id, need, remaining)

    monkeypatch.setattr(seeded, "reserve_run", spy)
    await _run(client, did)
    est = (await client.post(f"{API}/datasets/{did}/atac/links/estimate", json={})).json()
    assert seen["need"] == est["estimate_bytes"] and seeded._runs.get(did) is None and did not in atac_api._running


async def test_cells_changing_mid_run_saves_nothing(client, seeded, monkeypatch):
    did, _a = _register_world(seeded)
    real = atac_links.iter_links

    def shrink(store, rows, mask, read, *args, **kwargs):
        # A pipeline filtered the cells while the run read genes.
        return real(store, rows, mask, lambda names: read(names)[:-3], *args, **kwargs)

    monkeypatch.setattr(atac_links, "iter_links", shrink)
    events = await _run(client, did)
    err = next(d for e, d in events if e == "error")["error"]
    assert "cells changed" in err
    assert (await client.get(f"{API}/datasets/{did}/atac/links")).json()["result"] is None


async def test_a_large_dataset_reads_genes_from_its_gene_store_and_matches(client, seeded, monkeypatch):
    from scview.core import large_mode
    from tests.large_fixture import register

    a, p, table, _x = _world()
    # The in-memory twin gives the reference numbers.
    counts_genes = sp.csr_matrix(np.expm1(a.X.toarray()))
    counts = ad.AnnData(X=sp.csr_matrix(np.rint(counts_genes.toarray()).astype(np.float32)), obs=a.obs.copy(), var=a.var.drop(columns="highly_variable"))
    h5ad = register(seeded.data_dir, "lg", counts)
    assert h5ad
    _seed_store(seeded.data_dir, "lg", p, list(a.obs_names), table)
    adaptor = await seeded.get_or_load_dataset("lg")
    assert adaptor.is_large
    adaptor.adata.uns[atac_import.UNS_KEY] = a.uns[atac_import.UNS_KEY]
    adaptor.adata.var["highly_variable"] = True

    def boom(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("X was read whole")

    for name in ("to_memory", "toarray"):
        if hasattr(large_mode.OnDiskMatrix, name):
            monkeypatch.setattr(large_mode.OnDiskMatrix, name, boom)
    events = await _run(client, "lg")
    assert [e for e, _ in events][-1] == "complete", events[-1]
    res = next(d for e, d in events if e == "result")
    dense = p.toarray().astype(np.float64)
    where = {n: i for i, n in enumerate(table["name"])}
    adaptor = await seeded.get_or_load_dataset("lg")
    assert adaptor.is_large
    for row in res["rows"][:10]:
        y = adaptor.get_expression([row["gene"]])[:, 0].astype(np.float64)
        want = np.corrcoef(dense[:, where[row["peak"]]], y)[0, 1]
        assert row["score"] == pytest.approx(want, abs=1e-9)


# ---------------------------------------------------------------------------
# Co-pilot, guide, replay
# ---------------------------------------------------------------------------


async def test_copilot_grounds_on_links_with_its_tag_and_the_router_finds_it(client, seeded):
    from scview.core.claims import _FAMILY_PREFIX, resolve_tag
    from scview.core.rag import router

    did, _a = _register_world(seeded)
    await _run(client, did)
    results_store.flush()
    adaptor = await seeded.get_or_load_dataset(did)
    lines = atac_links.grounding_lines(adaptor)
    text = " ".join(lines)
    assert "[result:atac_links]" in text and "never proof" in text and "GENEA" in text
    assert _FAMILY_PREFIX["links"] == atac_links.TAG
    assert resolve_tag("result:atac_links", [atac_links.TAG]) == [atac_links.TAG]
    assert resolve_tag("links", [atac_links.TAG]) == [atac_links.TAG]
    assert any(w == "peak-gene" for w in router._RESULT_FACET_HINTS["groups"])
    assert atac_links.grounding_lines(type("T", (), {"h5ad_path": None})()) == []
    gene = str(atac_links.latest(adaptor).table["gene"].iloc[0])
    named = atac_links.named_gene_lines(adaptor, f"which peaks link to {gene.lower()}?")
    assert named and named[0][0] == gene and "candidates" in named[0][1]
    assert atac_links.named_gene_lines(adaptor, "which peaks link?") == []
    assert atac_links.named_gene_lines(adaptor, "") == []


def test_the_full_guide_has_the_card_and_the_compact_one_does_not():
    from scview.core import feature_guide

    assert "Peak-gene links" in feature_guide.full_guide()
    assert "Peak-gene links" not in feature_guide.compact_guide()


def test_an_old_result_without_the_newer_fields_still_opens():
    frame = pd.DataFrame({"gene": ["A"], "peak": ["chr1:1-2"], "distance_to_tss": [5.0], "score": [0.2], "z": [3.0], "p": [0.001], "padj": [0.01]})
    res = atac_links.LinksResult("x", frame, {"n_links": 1}, {}, "m", "c", "h", "t", 1.0)
    out = atac_links.payload(res)
    assert out["rows"][0]["gene"] == "A" and out["missing"] == [] and out["total"] == 1


def test_a_step_without_settings_is_not_replayed():
    from scview.core import replay_script

    a = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    provenance.record_step(a, step="atac_links", tool="t", params={})
    assert replay_script.describe(a)[-1]["status"] == "not_replayable"


def _write_h5(path, genes, peaks, barcodes, peak_names):
    """A Cell Ranger ARC style .h5 of these matrices: genes first, then peaks."""
    full = sp.hstack([genes, peaks]).tocsr()
    full.sort_indices()
    names = [g[0] for g in GENES[:-1]]
    ids = [g[1] for g in GENES[:-1]]
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.array(barcodes, dtype="S")
        m["data"] = full.data.astype(np.int32)
        m["indices"] = full.indices.astype(np.int64)
        m["indptr"] = full.indptr.astype(np.int64)
        m["shape"] = np.array([full.shape[1], full.shape[0]], dtype=np.int32)
        fe = m.create_group("features")
        fe["_all_tag_keys"] = np.array([b"genome", b"interval"])
        fe["feature_type"] = np.array(["Gene Expression"] * len(names) + ["Peaks"] * len(peak_names), dtype="S")
        fe["genome"] = np.array(["GRCh38"] * (len(names) + len(peak_names)), dtype="S")
        fe["id"] = np.array(ids + peak_names, dtype="S")
        fe["interval"] = np.array([f"{g[2]}:{g[3] - 1}-{g[3]}" for g in GENES[:-1]] + peak_names, dtype="S")
        fe["name"] = np.array(names + peak_names, dtype="S")
    return Path(path)


def test_the_replay_script_reproduces_scores_and_z_from_the_input(tmp_path, sequences, monkeypatch):
    """The script's own lognorm and genome read, from the original upload: scores bitwise, z from the same seed."""
    import gzip

    from scview.core import replay_script
    from scview.core.ingestion.session import IngestSessionManager
    from tests.genome_fixture import write_2bit

    a_world, p, table, _x = _world()
    raw = sp.csr_matrix(np.random.default_rng(2).poisson(0.3, (N_CELLS, len(GENES) - 1)).astype(np.float32))
    genes = raw.toarray()
    state = (a_world.obs["group"] == "A").to_numpy()
    genes[:, 0] += np.random.default_rng(3).poisson(6.0 * state)
    genes[:, 2] += np.random.default_rng(4).poisson(6.0 * ~state)
    h5 = _write_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5", sp.csr_matrix(genes), p, list(a_world.obs_names), table["name"].tolist())
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    (m.files_dir(sid) / h5.name).write_bytes(h5.read_bytes())
    did = m._build_dataset(sid)
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(_gtf())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, m.data_dir, "GRCh38")
    install(monkeypatch, m.data_dir, "GRCh38", sequences)
    a = ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))
    a.layers["counts"] = a.X.copy()
    import scanpy as sc

    provenance.record_step(a, step="normalization", tool="scanpy", params={"target_sum": 1e4, "method": "log1p"})
    sc.pp.normalize_total(a, target_sum=1e4)
    provenance.record_step(a, step="log_transform", tool="scanpy", params={})
    sc.pp.log1p(a)
    a.var["highly_variable"] = True

    class Adaptor:
        adata = a
        is_large = False
        h5ad_path = None

        def default_expression_layer(self):
            return "X"

        def gene_names(self):
            return list(a.var_names)

        def get_expression(self, names):
            return np.asarray(a[:, names].X.toarray(), dtype=np.float32)

    adaptor = Adaptor()
    plan = atac_links.plan_links(adaptor, m.data_dir, genome="GRCh38", mode="list", gene_list=["GENEA", "GENEB", "GENEC"])
    with atac.open_store(a, m.data_dir, did) as store:
        items = list(atac_links.iter_run(adaptor, store, plan, data_dir=m.data_dir))
    mine = items[-1]["frame"]
    assert len(mine) > 0
    res = atac_links.build_result(plan, items[-1], ["GENEA", "GENEB", "GENEC"])
    atac_links.record(a, res)
    assert replay_script.describe(a)[-1]["status"] == "exact"
    src = tmp_path / "input" / h5.name
    src.parent.mkdir()
    src.write_bytes(h5.read_bytes())
    with gzip.open(src.parent / "gencode.v32.basic.annotation.gtf.gz", "wt") as fh:
        fh.write(_gtf())
    write_2bit(src.parent / "hg38.2bit", sequences)
    script, dst = tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    script.write_text(replay_script.build_script(a, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "not the ones the run used" not in proc.stdout
    b = ad.read_h5ad(dst)
    got = b.uns[f"atac_links_{res.result_id}"]
    assert got["gene"].astype(str).tolist() == mine["gene"].tolist() and got["peak"].astype(str).tolist() == mine["peak"].tolist()
    np.testing.assert_array_equal(got["score"].to_numpy(dtype=np.float64), mine["score"].to_numpy())
    np.testing.assert_array_equal(got["z"].to_numpy(dtype=np.float64), mine["z"].to_numpy())
    gene_annotation.clear_cache()
    genome_seq.clear_cache()


def test_module_file_has_no_scview_imports_in_the_replayed_code():
    import inspect

    for obj in atac_links.REPLAY_CODE:
        assert "scview" not in inspect.getsource(obj).replace("scView", "")
