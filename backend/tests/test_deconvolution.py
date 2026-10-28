"""Bulk deconvolution (core/deconvolution.py, api/v1/deconvolution.py) on synthetic data
whose cell-type fractions are known."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from scview.core import claims, deconvolution as dc, provenance, replay_script, results_store

API = "/api/v1"
TYPES = ["B cell", "CD4.T", "Mono", "NK"]
SIZES = np.array([1500.0, 1000.0, 4000.0, 2000.0])  # counts per cell: the types differ in RNA content


def _profiles(rng, n_genes: int) -> np.ndarray:
    base = rng.gamma(0.6, 1.0, size=(n_genes, len(TYPES)))
    for k in range(len(TYPES)):
        base[20 + k * 25:20 + (k + 1) * 25, k] *= 25  # 25 markers per type
    return base / base.sum(axis=0)


def _donor_scale(rng, n_genes: int, noisy: np.ndarray, sd_noisy: float) -> np.ndarray:
    sd = np.where(noisy, sd_noisy, 0.05)
    return np.exp(rng.normal(0, sd)[:, None] + rng.normal(0, 0.02, size=(n_genes, len(TYPES))))


def synthetic(n_donors: int = 4, cells: int = 60, n_genes: int = 400, sd_noisy: float = 0.05, seed: int = 0,
              noisy_share: float = 0.4):
    """(AnnData of counts with cell_type + donor, a function bulk(fractions, donor_seed) -> counts)."""
    rng = np.random.default_rng(seed)
    theta = _profiles(rng, n_genes)
    noisy = rng.random(n_genes) < noisy_share
    names = [f"G{i}" for i in range(n_genes - 4)] + ["MT-CO1", "RPL3", "XIST", "RPS4Y1"]
    blocks, ct, donor = [], [], []
    for d in range(n_donors):
        prof = theta * _donor_scale(rng, n_genes, noisy, sd_noisy)
        prof /= prof.sum(axis=0)
        for k, t in enumerate(TYPES):
            n = cells + 10 * k
            blocks.append(rng.poisson(prof[:, k] * SIZES[k], size=(n, n_genes)))
            ct += [t] * n
            donor += [f"D{d + 1}"] * n
    X = sp.csr_matrix(np.vstack(blocks).astype(np.float32))
    obs = pd.DataFrame({"cell_type": pd.Categorical(ct), "donor": pd.Categorical(donor)},
                       index=[f"c{i}" for i in range(X.shape[0])])
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=names))
    a.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]

    def bulk(fractions, donor_seed: int, n_cells: int = 5000) -> np.ndarray:
        r = np.random.default_rng(donor_seed)
        prof = theta * _donor_scale(r, n_genes, noisy, sd_noisy)
        prof /= prof.sum(axis=0)
        n_k = np.asarray(fractions) * n_cells
        return r.poisson(prof @ (n_k * SIZES)).astype(np.float64)

    return a, bulk, names


def _tsv(names, columns: dict[str, np.ndarray], index_label: str = "gene") -> bytes:
    df = pd.DataFrame(columns, index=names)
    return df.to_csv(sep="\t", index_label=index_label).encode()


TRUTH = {"s1": [0.4, 0.3, 0.2, 0.1], "s2": [0.1, 0.1, 0.5, 0.3], "s3": [0.25, 0.25, 0.25, 0.25]}


def _bulk_file(bulk, names, truth=TRUTH) -> bytes:
    return _tsv(names, {s: bulk(p, 100 + i) for i, (s, p) in enumerate(truth.items())})


def _wide(r: dc.Result, col: str = "cell_fraction") -> pd.DataFrame:
    return r.fractions.pivot(index="sample", columns="cell_type", values=col).loc[r.samples, TYPES]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["weighted", "nnls"])
def test_recovers_known_fractions(method):
    a, bulk, names = synthetic()
    r = dc.run(a, _bulk_file(bulk, names), "bulk.tsv", celltype_column="cell_type", donor_column="donor",
               method=method)
    assert r.method == method and r.samples == list(TRUTH)
    est = _wide(r)
    np.testing.assert_allclose(est.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(est.to_numpy(), np.array(list(TRUTH.values())), atol=0.05 if method == "weighted"
                               else 0.08)
    # RNA fractions weigh each type by its cell size.
    rna = _wide(r, "rna_fraction")
    S = np.asarray(r.cell_size)
    expect = est.to_numpy() * S / (est.to_numpy() * S).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(rna.to_numpy(), expect, atol=1e-9)
    assert rna.loc["s3", "Mono"] > est.loc["s3", "Mono"]  # big cells, more of the RNA
    assert (r.fit["r2"] > 0.9).all()
    assert r.input_kind == "counts" and r.gene_share == 1.0


def test_weighted_beats_nnls_when_donors_differ():
    a, bulk, names = synthetic(n_donors=6, sd_noisy=1.2, seed=3)
    raw = _bulk_file(bulk, names)
    truth = np.array(list(TRUTH.values()))
    err = {}
    for method in ("weighted", "nnls"):
        r = dc.run(a, raw, "bulk.tsv", celltype_column="cell_type", donor_column="donor", method=method)
        err[method] = float(np.sqrt(((_wide(r).to_numpy() - truth) ** 2).mean()))
    assert err["weighted"] < 0.05
    assert err["weighted"] < 0.7 * err["nnls"], err


def test_genes_left_out_of_the_reference():
    a, _, names = synthetic()
    ref = dc.build_reference(a, "cell_type", "donor")
    left = set(ref.genes[~ref.usable])
    assert {"MT-CO1", "RPL3", "XIST", "RPS4Y1"} <= left


def test_auto_without_donors_is_nnls_and_says_why():
    a, bulk, names = synthetic()
    r = dc.run(a, _bulk_file(bulk, names), "bulk.tsv", celltype_column="cell_type", donor_column="", method="auto")
    assert r.method == "nnls" and "no donor column" in r.method_reason
    assert r.marker_source and "fold change" in r.marker_source  # no stored markers here
    with pytest.raises(dc.DeconvolutionError, match="weighted method needs"):
        dc.run(a, _bulk_file(bulk, names), "bulk.tsv", celltype_column="cell_type", donor_column="",
               method="weighted")


def test_auto_finds_the_columns_and_runs_weighted():
    a, bulk, names = synthetic()
    r = dc.run(a, _bulk_file(bulk, names), "bulk.tsv")
    assert r.celltype_column == "cell_type" and r.donor_column == "donor" and r.method == "weighted"
    assert r.method_reason.startswith("Auto: weighted")


def test_stored_markers_are_the_nnls_genes():
    import scanpy as sc

    a, bulk, names = synthetic()
    b = a.copy()
    sc.pp.normalize_total(b)
    sc.pp.log1p(b)
    sc.tl.rank_genes_groups(b, "cell_type", method="wilcoxon", pts=True)
    a.uns["rank_genes_groups"] = b.uns["rank_genes_groups"]
    r = dc.run(a, _bulk_file(bulk, names), "bulk.tsv", celltype_column="cell_type", donor_column="donor",
               method="nnls")
    assert "stored markers" in r.marker_source and r.n_genes_used <= dc.MARKERS_PER_TYPE * len(TYPES)


def test_a_type_too_small_is_left_out_with_a_note():
    a, bulk, names = synthetic()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.obs.iloc[:5, a.obs.columns.get_loc("cell_type")] = "Rare"
    ref = dc.build_reference(a, "cell_type", "donor")
    assert "Rare" not in ref.types and any("Rare" in n for n in ref.notes)


def test_no_counts_is_refused():
    a, _, _ = synthetic()
    a.X = a.X.multiply(0.37).tocsr()
    with pytest.raises(dc.DeconvolutionError, match="raw counts"):
        dc.build_reference(a, "cell_type", "donor")


# ---------------------------------------------------------------------------
# The bulk file
# ---------------------------------------------------------------------------


def _ref():
    a, bulk, names = synthetic()
    return a, bulk, names, dc.build_reference(a, "cell_type", "donor")


def test_gct_with_versioned_ensembl_ids():
    a, bulk, names, ref = _ref()
    ids = [f"ENSG{i:011d}.{i % 7 + 1}" for i in range(len(names))]
    values = {s: bulk(p, 100 + i) for i, (s, p) in enumerate(TRUTH.items())}
    body = pd.DataFrame({"Name": ids, "Description": names, **values}).to_csv(sep="\t", index=False)
    gct = f"#1.2\n{len(names)}\t{len(values)}\n{body}".encode()
    b = dc.parse_bulk(gzip.compress(gct), "x.gct.gz", ref)
    assert b.id_kind == "ensembl" and b.samples == list(TRUTH) and b.n_matched == len(names)
    r = dc.run(a, gzip.compress(gct), "x.gct.gz", celltype_column="cell_type", donor_column="donor")
    np.testing.assert_allclose(_wide(r).to_numpy(), np.array(list(TRUTH.values())), atol=0.05)


def test_samples_in_rows_are_turned():
    a, bulk, names, ref = _ref()
    df = pd.DataFrame({s: bulk(p, 100 + i) for i, (s, p) in enumerate(TRUTH.items())}, index=names).T
    raw = df.to_csv(index_label="sample").encode()
    b = dc.parse_bulk(raw, "x.csv", ref)
    assert b.transposed and b.samples == list(TRUTH) and b.n_matched == len(names)


def test_log_values_are_refused():
    a, bulk, names, ref = _ref()
    raw = _tsv(names, {"s1": np.log2(bulk(TRUTH["s1"], 1) + 1)})
    with pytest.raises(dc.DeconvolutionError, match="log-scaled"):
        dc.parse_bulk(raw, "x.tsv", ref)


def test_negative_and_missing_values_are_refused():
    a, bulk, names, ref = _ref()
    v = bulk(TRUTH["s1"], 1)
    v[3] = -5
    with pytest.raises(dc.DeconvolutionError, match="negative"):
        dc.parse_bulk(_tsv(names, {"s1": v}), "x.tsv", ref)
    text = _tsv(names, {"s1": bulk(TRUTH["s1"], 1)}).decode().splitlines()
    text[5] = text[5].split("\t")[0] + "\tNA"
    with pytest.raises(dc.DeconvolutionError, match="missing"):
        dc.parse_bulk("\n".join(text).encode(), "x.tsv", ref)


def test_tpm_is_accepted_with_a_warning():
    a, bulk, names, ref = _ref()
    cols = {}
    for i, (s, p) in enumerate(TRUTH.items()):
        v = bulk(p, 100 + i)
        cols[s] = v / v.sum() * 1e6 + 0.001
    b = dc.parse_bulk(_tsv(names, cols), "x.tsv", ref)
    assert b.input_kind == "tpm" and any("TPM" in w and "relative" in w for w in b.warnings)


def test_too_few_shared_genes_is_refused():
    a, bulk, names, ref = _ref()
    keep = names[:150]
    raw = _tsv(keep, {"s1": bulk(TRUTH["s1"], 1)[:150]})
    with pytest.raises(dc.DeconvolutionError, match="Only 3[0-9]% of this dataset's genes"):
        dc.run(a, raw, "x.tsv", celltype_column="cell_type", donor_column="donor")


def test_star_rows_duplicates_and_case():
    a, bulk, names, ref = _ref()
    v = bulk(TRUTH["s1"], 1)
    rows = ["N_unmapped", "N_multimapping", *[n.lower() for n in names], "g0"]
    vals = [10, 20, *v, 7]
    raw = pd.DataFrame({"gene": rows, "s1": vals}).to_csv(sep="\t", index=False).encode()
    b = dc.parse_bulk(raw, "x.tsv", ref)
    assert b.n_rows == len(names) + 1 and b.n_matched == len(names)
    assert any("summary rows" in n for n in b.notes) and any("summed" in n for n in b.notes)
    i = int(np.flatnonzero(ref.genes[b.positions] == "G0")[0])
    assert b.values[i, 0] == v[0] + 7


def test_gdc_star_counts_use_unstranded():
    a, bulk, names, ref = _ref()
    v = bulk(TRUTH["s1"], 1)
    df = pd.DataFrame({"gene_id": [f"ENSG{i:011d}.3" for i in range(len(names))], "gene_name": names,
                       "gene_type": "protein_coding", "unstranded": v, "stranded_first": v // 2,
                       "tpm_unstranded": v / v.sum() * 1e6})
    raw = ("# gene-model: GENCODE v36\n" + df.to_csv(sep="\t", index=False)).encode()
    b = dc.parse_bulk(raw, "abc.rna_seq.augmented_star_gene_counts.tsv", ref)
    assert b.samples == ["abc"] and b.input_kind == "counts"
    np.testing.assert_array_equal(b.values[:, 0], v[b.positions])


def test_not_a_table_and_too_big_are_refused(monkeypatch):
    a, _, _, ref = _ref()
    with pytest.raises(dc.DeconvolutionError, match="not a text table"):
        dc.parse_bulk(b"\x00\x01\x02binary", "x.tsv", ref)
    monkeypatch.setattr(dc, "MAX_BULK_SAMPLES", 2)
    with pytest.raises(dc.DeconvolutionError, match="at most 2"):
        dc.parse_bulk(_tsv(["G1", "G2"], {"a": [100, 1], "b": [100, 2], "c": [100, 3]}), "x.tsv", ref)


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------


def test_self_check_holds_out_each_donor():
    a, _, _ = synthetic(n_donors=4, sd_noisy=0.8, seed=5)
    events = list(dc.iter_check(a, celltype_column="cell_type", donor_column="donor", n_mixtures=8, seed=1))
    progress = [d for e, d in events if e == "progress"]
    assert progress[0]["done"] == 0 and progress[-1]["done"] == 4
    c = events[-1][1]
    assert isinstance(c, dc.CheckResult) and c.methods == ["weighted", "nnls"]
    assert len(c.points) == 2 * 4 * (8 + 1) * len(TYPES)
    overall = {o["method"]: o for o in c.overall}
    assert overall["weighted"]["r"] > 0.95 and overall["weighted"]["rmse"] < 0.05
    assert set(c.per_type["cell_type"]) == set(TYPES)
    assert abs(overall["weighted"]["bias"]) < 0.01  # fractions sum to 1: mean error over types ~ 0
    # Same seed, same mixtures.
    again = dc.check(a, celltype_column="cell_type", donor_column="donor", n_mixtures=8, seed=1)
    pd.testing.assert_frame_equal(again.points, c.points)


def test_self_check_needs_three_donors():
    a, _, _ = synthetic(n_donors=2)
    with pytest.raises(dc.DeconvolutionError, match="3\\+ donors"):
        dc.check(a, celltype_column="cell_type", donor_column="donor")


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def test_reference_zip_contents(monkeypatch):
    monkeypatch.setattr(dc, "CIBERSORTX_CELLS", 100)
    a, _, names = synthetic()
    data = b"".join(dc.iter_reference_zip(a, celltype_column="cell_type", donor_column="donor", block_cells=97))
    z = zipfile.ZipFile(io.BytesIO(data))
    assert set(z.namelist()) == {
        "matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz", "cells.tsv", "cibersortx_reference.txt",
        "signature_cpm.tsv", "pseudobulk_counts.tsv", "pseudobulk_meta.tsv", "cell_size.tsv", "README.txt"}
    m = scipy.io.mmread(io.BytesIO(gzip.decompress(z.read("matrix.mtx.gz"))))
    assert m.shape == (len(names), a.n_obs)
    np.testing.assert_array_equal(sp.csr_matrix(m).T.toarray(), a.X.toarray())
    feats = pd.read_csv(io.BytesIO(gzip.decompress(z.read("features.tsv.gz"))), sep="\t", header=None)
    assert list(feats[1]) == names and feats[0][0] == "ENSG00000000000" and feats[2][0] == "Gene Expression"
    barcodes = gzip.decompress(z.read("barcodes.tsv.gz")).decode().split()
    assert barcodes == list(a.obs_names)
    cells = pd.read_csv(io.BytesIO(z.read("cells.tsv")), sep="\t")
    assert list(cells.columns) == ["cell_id", "cell_type", "cell_state", "donor"]
    assert list(cells["cell_type"]) == list(a.obs["cell_type"].astype(str))
    # CIBERSORTx: row 1 = each cell's label, no periods or spaces; genes in rows; counts.
    cx = z.read("cibersortx_reference.txt").decode().splitlines()
    head = cx[0].split("\t")
    assert head[0] == "GeneSymbol" and 3 * len(TYPES) <= len(head) - 1 <= 100
    assert set(head[1:]) == {"B_cell", "CD4_T", "Mono", "NK"}
    row = cx[1].split("\t")
    assert row[0] in names and all(v.isdigit() for v in row[1:])
    sig = pd.read_csv(io.BytesIO(z.read("signature_cpm.tsv")), sep="\t", index_col=0)
    assert list(sig.columns) == TYPES
    np.testing.assert_allclose(sig.sum(axis=0), 1e6, rtol=1e-4)
    pb = pd.read_csv(io.BytesIO(z.read("pseudobulk_counts.tsv")), sep="\t", index_col=0)
    meta = pd.read_csv(io.BytesIO(z.read("pseudobulk_meta.tsv")), sep="\t")
    assert pb.shape == (len(names), 4 * len(TYPES)) and list(meta["column"]) == list(pb.columns)
    assert int(pb.to_numpy().sum()) == int(a.X.sum())
    size = pd.read_csv(io.BytesIO(z.read("cell_size.tsv")), sep="\t")
    np.testing.assert_allclose(size["mean_counts_per_cell"], SIZES, rtol=0.1)
    assert "CIBERSORTx" in z.read("README.txt").decode()


def test_cibersortx_label():
    assert dc.cibersortx_label("CD4.T cells (naive)") == "CD4_T_cells_naive"


# ---------------------------------------------------------------------------
# Co-pilot, replay
# ---------------------------------------------------------------------------


def test_claims_tag_family_resolves():
    assert claims.resolve_tag("deconvolution", [dc.TAG]) == [dc.TAG]
    assert claims.resolve_tag("result:deconv", [dc.TAG]) == [dc.TAG]


def test_replay_names_the_step_as_not_replayed():
    a, _, _ = synthetic()
    provenance.record_step(a, step=dc.STEP, tool="test", params={"bulk_file": "x.tsv", "method": "weighted"})
    text = replay_script.build_script(a)
    assert "Bulk deconvolution" in text and "not replayed" in text
    assert dc.STEP in results_store.ADDITIVE_STEPS


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if not ln.startswith(":")]
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if ev:
            out.append((ev, json.loads(data or "{}")))
    return out


async def test_api_run_get_csv_rerun_and_saved(client, dataset_manager):
    a, bulk, names = synthetic()
    did = _register(dataset_manager, a, "dc1")
    url = f"{API}/datasets/{did}/deconvolution"
    assert (await client.get(url)).status_code == 404
    d = (await client.get(f"{url}/defaults")).json()
    assert d["celltype_column"] == "cell_type" and d["donor_column"] == "donor" and d["n_donors"] == 4

    raw = gzip.compress(_bulk_file(bulk, names))
    r = await client.post(f"{url}/upload", files={"file": ("my bulk.tsv.gz", raw, "application/gzip")},
                          data={"options": json.dumps({"method": "auto"})})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "weighted" and body["n_samples"] == 3 and body["types"] == TYPES
    s1 = body["samples"][0]
    assert s1["sample"] == "s1" and abs(sum(s1["cell_fractions"].values()) - 1) < 1e-3
    assert abs(s1["cell_fractions"]["B cell"] - 0.4) < 0.05 and s1["converged"]

    got = (await client.get(url)).json()
    assert got["result_id"] == body["result_id"] and got["samples"] == body["samples"]
    csv = await client.get(f"{url}/{body['result_id']}.csv")
    assert csv.status_code == 200
    lines = csv.text.strip().splitlines()
    assert lines[0] == "sample,fraction,B cell,CD4.T,Mono,NK" and len(lines) == 1 + 2 * 3
    files = list((Path(dataset_manager.data_dir) / "uploads" / did / "deconvolution").iterdir())
    assert len(files) == 1 and files[0].name.endswith("my_bulk.tsv.gz")

    # A re-run without a file uses the kept one; another method.
    r2 = await client.post(f"{url}/upload", data={"options": json.dumps({"method": "nnls"})})
    assert r2.status_code == 200, r2.text
    assert r2.json()["method"] == "nnls" and r2.json()["filename"] == "my bulk.tsv.gz"

    # Saved with the dataset: a History step; GET after a reload.
    results_store.flush()
    dataset_manager.evict(did)
    dc._last.clear()
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    steps = [h for h in provenance.read_provenance(adata)["history"] if h["step"] == dc.STEP]
    assert len(steps) == 2 and steps[0]["params"]["bulk_file"] == "my bulk.tsv.gz"
    again = (await client.get(url)).json()
    assert again["result_id"] == r2.json()["result_id"]
    lines, detail = dc.grounding_lines(await dataset_manager.get_or_load_dataset(did))
    assert detail and any(f"[{dc.TAG}]" in ln for ln in lines) and any("s2: Mono" in ln for ln in lines)


async def test_api_refusals(client, dataset_manager):
    a, bulk, names = synthetic()
    did = _register(dataset_manager, a, "dc2")
    url = f"{API}/datasets/{did}/deconvolution/upload"
    r = await client.post(url, data={"options": "{}"})
    assert r.status_code == 422 and "Choose a bulk" in r.json()["detail"]
    r = await client.post(url, files={"file": ("b.tsv", _tsv(names, {"s": np.full(len(names), 1.5)}))})
    assert r.status_code == 422 and "log-scaled" in r.json()["detail"]
    raw = _bulk_file(bulk, names)
    r = await client.post(url, files={"file": ("b.tsv", raw)}, data={"options": json.dumps({"celltype_column": "zz"})})
    assert r.status_code == 422 and "not found" in r.json()["detail"]
    r = await client.post(url, files={"file": ("b.tsv", raw)}, data={"options": "not json"})
    assert r.status_code == 422
    r = await client.post(url, files={"file": ("b.tsv", raw)}, data={"options": json.dumps({"method": "best"})})
    assert r.status_code == 422
    r = await client.post(url, files={"file": ("b.tsv", raw)},
                          data={"options": json.dumps({"donor_column": "", "method": "weighted"})})
    assert r.status_code == 422 and "weighted method needs" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/{did}/deconvolution/nope.csv")).status_code == 404
    assert (await client.get(f"{API}/datasets/missing/deconvolution")).status_code == 404


async def test_api_self_check_stream_and_last(client, dataset_manager):
    a, _, _ = synthetic(seed=2)
    did = _register(dataset_manager, a, "dc3")
    url = f"{API}/datasets/{did}/deconvolution"
    assert (await client.get(f"{url}/check")).status_code == 404
    r = await client.post(f"{url}/validate", json={"n_mixtures": 5, "seed": 3})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "progress" and kinds[-2:] == ["result", "complete"] and kinds.count("progress") == 5
    res = dict(events)["result"]
    assert res["methods"] == ["weighted", "nnls"] and res["n_mixtures"] == 5
    assert len(res["points"]["weighted"]["truth"]) == 4 * 6 * len(TYPES)
    assert {o["method"] for o in res["overall"]} == {"weighted", "nnls"}
    got = (await client.get(f"{url}/check")).json()
    assert got["result_id"] == res["result_id"]
    a2, _, _ = synthetic(n_donors=2)
    did2 = _register(dataset_manager, a2, "dc4")
    r = await client.post(f"{API}/datasets/{did2}/deconvolution/validate", json={})
    assert r.status_code == 422 and "3+ donors" in r.json()["detail"]
    r = await client.post(f"{url}/validate", json={"n_mixtures": 1000})
    assert r.status_code == 422


async def test_api_reference_zip(client, dataset_manager):
    a, _, _ = synthetic()
    did = _register(dataset_manager, a, "dc5")
    r = await client.get(f"{API}/datasets/{did}/deconvolution/reference.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "scview_reference_dc5.zip" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert "cibersortx_reference.txt" in z.namelist() and z.testzip() is None
    r = await client.get(f"{API}/datasets/{did}/deconvolution/reference.zip", params={"celltype_column": "zz"})
    assert r.status_code == 422
