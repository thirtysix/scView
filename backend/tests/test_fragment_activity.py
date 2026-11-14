"""Fragment-based gene activity (core/fragment_activity.py = Signac GeneActivity) and its use in
the ATAC run (core/atac.py, api/v1/atac.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac, atac_fragments, fragment_activity as fa, fragment_file as ff, gene_annotation
from tests import multiome_fixture as mf
from tests.fragments_fixture import write_fragments

API = "/api/v1"


def _brute(lines, cells, chrom, start, end):
    """Signac's FeatureMatrix by hand: each fragment line of a listed cell, 1 per region it overlaps."""
    out = np.zeros((len(cells), len(chrom)), dtype=np.int64)
    where = {c: i for i, c in enumerate(cells)}
    for c, s, e, bc, _n in lines:
        if bc not in where:
            continue
        for g in range(len(chrom)):
            if chrom[g] == c and s < end[g] and e > start[g]:
                out[where[bc], g] += 1
    return out


def test_region_hits_is_every_overlap_with_nested_and_long_regions():
    rng = np.random.default_rng(0)
    gs = np.sort(rng.integers(0, 100_000, 60))
    ge = gs + rng.integers(1, 30_000, 60)
    ge[5] = gs[5] + 90_000  # one long region holds many others
    reach = np.maximum.accumulate(ge)
    fs = np.sort(rng.integers(0, 120_000, 2_000))
    fe = fs + rng.integers(1, 600, 2_000)
    frag, region = fa.region_hits(fs, fe, gs, ge, reach)
    got = set(zip(frag.tolist(), region.tolist()))
    want = {(i, j) for i in range(fs.size) for j in range(gs.size) if fs[i] < ge[j] and fe[i] > gs[j]}
    assert got == want and len(got) > 2_000
    # touching ends do not overlap (half open), as tabix on a BED-style file
    f, r = fa.region_hits(np.array([100, 200]), np.array([200, 300]), np.array([200]), np.array([250]), np.array([250]))
    assert f.tolist() == [1] and r.tolist() == [0]


def test_signac_keeps_protein_coding_genes_under_500_kb():
    start = np.array([0, 0, 0, 10])
    end = np.array([1_000, 500_000, 499_999, 20])
    keep = fa.signac_genes(start, end, np.array(["protein_coding", "protein_coding", "protein_coding", "lncRNA"], dtype=object))
    assert keep.tolist() == [True, False, True, False]


def _world(tmp_path, block_size=300):
    rng = np.random.default_rng(3)
    cells = [f"BC{i:03d}-1" for i in range(30)]
    lines = []
    for chrom, n in (("chr1", 900), ("chr3", 400), ("chrUn_X", 30)):
        for _ in range(n):
            s = int(rng.integers(0, 60_000))
            lines.append((chrom, s, s + int(rng.integers(40, 400)), f"BC{int(rng.integers(0, 33)):03d}-1" if rng.random() > 0.05
                          else "OTHER-1", 1))
    lines.sort(key=lambda x: ({"chr1": 0, "chr3": 1, "chrUn_X": 2}[x[0]], x[1]))
    path, index = write_fragments(tmp_path / "f.tsv.gz", lines, block_size=block_size)
    # genes: overlapping and nested on chr1, chr2 (no fragments), chr3, chrX (not in the file)
    genes = pd.DataFrame([("A", "chr1", 1_000, 9_000), ("B", "chr1", 2_000, 40_000), ("C", "chr1", 5_000, 6_000),
                          ("D", "chr1", 50_000, 59_000), ("E", "chr2", 0, 5_000), ("F", "chr3", 10_000, 30_000),
                          ("G", "chrX", 0, 1_000), ("H", "chr3", 70_000, 80_000)], columns=["name", "chrom", "start", "end"])
    return cells, lines, path, index, genes


def _run(path, index, cells, genes, stop=lambda: False):
    order = fa.row_order(genes["chrom"].to_numpy(), genes["start"].to_numpy(), ff.load_index(index).names)
    g = genes.iloc[order].reset_index(drop=True)
    parts = []
    out = list(fa.iter_counts(path, cells, g["chrom"].to_numpy(), g["start"].to_numpy(), g["end"].to_numpy(),
                              parts.append, stop=stop))
    return g, sp.vstack(parts).tocsr(), out[-1]


def test_counts_are_signacs_feature_matrix_across_blocks_and_chromosomes(tmp_path, monkeypatch):
    cells, lines, path, index, genes = _world(tmp_path)
    monkeypatch.setattr(fa, "FLUSH_VALUES", 7)  # many small writes
    real = ff.iter_text

    def small(p, block=ff.TEXT_BLOCK):  # pieces of a few lines: genes span pieces
        for done, text in real(p):
            lines_ = text.splitlines(keepends=True)
            for i in range(0, len(lines_), 37):
                yield done, b"".join(lines_[i:i + 37])

    monkeypatch.setattr(ff, "iter_text", small)
    g, m, result = _run(path, index, cells, genes)
    assert g["name"].tolist() == ["A", "B", "C", "D", "F", "H", "E", "G"]  # file's order (chr1, chr3), then the rest
    want = _brute(lines, cells, g["chrom"].tolist(), g["start"].to_numpy(), g["end"].to_numpy())
    assert m.shape == (8, len(cells)) and np.array_equal(m.toarray().T, want)
    assert result["totals"].tolist() == want.sum(axis=1).tolist() and result["n_counted"] == int(want.sum())
    assert result["n_genes_with_fragments"] == int((want.sum(axis=0) > 0).sum())
    assert result["n_lines"] == len(lines)
    # the replay's pandas reader gives the same matrix
    replay, seen = fa.fragment_gene_counts(path, cells, g["chrom"].to_numpy(dtype=object), g["start"].to_numpy(),
                                           g["end"].to_numpy(), chunk_lines=50)
    assert np.array_equal(replay.toarray(), want) and seen == {"chr1", "chr3", "chrUn_X"}


def test_an_unsorted_file_and_a_stop_say_so(tmp_path):
    cells, lines, path, index, genes = _world(tmp_path)
    shuffled = lines[:5] + [lines[400]] + lines[5:400] + lines[401:]
    bad, bad_index = write_fragments(tmp_path / "bad.tsv.gz", shuffled)
    with pytest.raises(fa.FragmentActivityError, match="not sorted"):
        _run(bad, bad_index, cells, genes)
    with pytest.raises(fa.FragmentActivityCancelled):
        _run(path, index, cells, genes, stop=lambda: True)


# ---------------------------------------------------------------------------
# In the ATAC run
# ---------------------------------------------------------------------------


@pytest.fixture()
def annotated(dataset_manager, tmp_path):
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, dataset_manager.data_dir, "GRCh38")
    yield
    gene_annotation.clear_cache()


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        got = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in got:
            out.append((got["event"], json.loads(got["data"])))
    return out


def _fragments_for(barcodes, n_per):
    """Group A's fragments over GENEA (chr1 10,001-20,000, +) and none over GENEC; B the other way."""
    rng = np.random.default_rng(5)
    lines = []
    for i, bc in enumerate(barcodes):
        a = i < n_per
        for _ in range(int(rng.integers(20, 60))):
            if a:
                s = int(rng.integers(8_500, 20_000))  # includes the 2 kb upstream of GENEA
                lines.append(("chr1", s, s + 150, bc, 1))
            else:
                s = int(rng.integers(8_500, 20_000))
                lines.append(("chr2", s, s + 150, bc, 1))
    return sorted(lines, key=lambda x: (x[0], x[1]))


async def test_the_run_uses_the_fragments_when_attached_and_says_so(client, dataset_manager, annotated, tmp_path):
    from tests.test_atac_api import _register_multiome

    did, _p, _t, a = _register_multiome(dataset_manager, "fa1")
    state = (await client.get(f"{API}/datasets/{did}/atac")).json()
    assert state["plan"]["activity_source"] == "peaks" and state["plan"]["fragment_bytes"] is None
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 10, "dims_to": 8, "activity_source": "fragments"})
    assert r.status_code == 409 and "needs the fragments file" in r.json()["detail"]

    lines = _fragments_for(list(a.obs_names), 40)
    frag, idx = write_fragments(tmp_path / "x.tsv.gz", lines)
    folder = atac_fragments.fragments_path(dataset_manager.data_dir, did).parent
    folder.mkdir(parents=True, exist_ok=True)
    frag.replace(atac_fragments.fragments_path(dataset_manager.data_dir, did))
    idx.replace(atac_fragments.index_path(dataset_manager.data_dir, did))
    state = (await client.get(f"{API}/datasets/{did}/atac")).json()
    assert state["plan"]["activity_source"] == "fragments" and state["plan"]["fragment_bytes"] > 0
    peaks_plan = atac.plan(a, dataset_manager.data_dir, did, activity_source="peaks").to_dict()
    assert state["plan"]["estimate_bytes"] > peaks_plan["estimate_bytes"]  # the pass over the file holds more

    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 10, "dims_to": 8})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert events[0][1]["activity_source"] == "fragments"
    result = next(d for e, d in events if e == "result")
    act = result["activity"]
    assert act["kind"] == "fragment-based" and act["n_genes"] == 4 and act["biotypes"] == ["protein_coding"]
    assert "GeneActivity" in act["caption"] and act["n_fragments_read"] == len(lines)
    # the counts are the fragments over body + 2 kb upstream, Signac's scale factor the median cell total
    gene_a = next(g for g in mf.GENES if g[0] == "GENEA")
    lo, hi = gene_a[3] - 1 - 2_000, gene_a[4]
    want = np.zeros(a.n_obs)
    where = {b: i for i, b in enumerate(a.obs_names)}
    for c, s, e, bc, _n in lines:
        if c == "chr1" and s < hi and e > lo:
            want[where[bc]] += 1
    body = (await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA", "groupby": "group"})).json()
    adaptor = await dataset_manager.get_or_load_dataset(did)
    total = adaptor.adata.obs[atac.OBS_ACTIVITY_TOTAL].to_numpy(dtype=np.float64)
    assert act["scale_factor"] == pytest.approx(np.median(total))
    expect = np.log1p(np.divide(want * act["scale_factor"], total, out=np.zeros_like(want), where=total > 0))
    np.testing.assert_allclose(body["values"], expect, atol=1e-3)
    assert "GeneActivity" in body["caption"]
    assert np.mean(body["violin"]["A"]) > np.mean(body["violin"]["B"])
    lines, _detail = atac.grounding_lines(adaptor.adata, dataset_manager.data_dir)
    assert any("fragment-based (Signac's GeneActivity" in x for x in lines)
    # peaks on request: the old caption, 10,000 as scale factor
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 10, "dims_to": 8, "activity_source": "peaks"})
    act = next(d for e, d in _events(r.text) if e == "result")["activity"]
    assert act["kind"] == "peak-based" and "scale_factor" not in act
    body = (await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA"})).json()
    assert "Peak-based" in body["caption"]


def test_the_replay_counts_the_fragments_beside_the_input(tmp_path):
    import gzip
    import subprocess
    import sys

    import anndata as ad

    from scview.core import atac_import, modality_store, replay_script
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
    lines = _fragments_for(list(a.obs_names), 40)
    frag, idx = write_fragments(tmp_path / "pbmc_atac_fragments.tsv.gz", lines)
    modality_store.store_dir(m.data_dir, did).mkdir(parents=True, exist_ok=True)
    atac_fragments.fragments_path(m.data_dir, did).write_bytes(frag.read_bytes())
    atac_fragments.index_path(m.data_dir, did).write_bytes(idx.read_bytes())
    p = atac.plan(a, m.data_dir, did, n_lsi=8, dims_to=6, resolution=0.7)
    assert p.activity_source == "fragments" and p.fragments_file == atac_fragments.FRAGMENTS_FILE
    with atac.open_store(a, m.data_dir, did) as store:
        rows = atac.cell_rows(store, a)
        genes = atac.choose_genes(a, p.genome, m.data_dir)
        res = [i for i in atac.iter_run(store, rows, p, data_dir=m.data_dir, dataset_id=did, genes=genes)
               if i["type"] == "result"][0]["result"]
    atac.save_run(a, p, res)
    mine = {g: atac.activity_gene(a, m.data_dir, did, g)["counts"] for g in ("GENEA", "GENEC")}
    src = tmp_path / "input" / h5.name
    src.parent.mkdir()
    src.write_bytes(h5.read_bytes())
    with gzip.open(src.parent / "gencode.v32.basic.annotation.gtf.gz", "wt") as fh:
        fh.write(mf.gtf_text())
    (src.parent / atac_fragments.FRAGMENTS_FILE).write_bytes(frag.read_bytes())
    script, dst = tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    text = replay_script.build_script(a, dataset_id="t", dataset_name="test")
    assert "Fragment-based gene activity (Signac's GeneActivity)" in text
    script.write_text(text)
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    b = ad.read_h5ad(dst)
    got = b.uns["atac_gene_activity_genes"].tolist()
    for g, counts in mine.items():
        np.testing.assert_array_equal(np.asarray(b.obsm["atac_gene_activity"][:, got.index(g)].todense()).ravel(), counts)
    np.testing.assert_allclose(b.obs[atac.OBS_ACTIVITY_TOTAL], a.obs[atac.OBS_ACTIVITY_TOTAL])
    saved = atac_import.read_info(a)["activity"]
    assert saved["kind"] == "fragment-based"
    assert b.uns["atac_gene_activity_scale_factor"] == pytest.approx(saved["scale_factor"])
    gene_annotation.clear_cache()
