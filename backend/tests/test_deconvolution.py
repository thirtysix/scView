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
# A reference of two conditions (Kang: CTRL bulk against CTRL + STIM cells)
# ---------------------------------------------------------------------------

CTYPES = ["B", "Mk", "Mono", "T"]  # the reference's (natural) order
CSIZES = {"B": 1500.0, "T": 1000.0, "Mono": 4000.0, "Mk": 1500.0}
CTRUTH = {"s1": {"B": 0.4, "T": 0.3, "Mono": 0.28, "Mk": 0.02}, "s2": {"B": 0.2, "T": 0.5, "Mono": 0.28, "Mk": 0.02},
          "s3": {"B": 0.3, "T": 0.3, "Mono": 0.38, "Mk": 0.02}}


def two_conditions(seed: int = 0, n_genes: int = 400, n_ifn: int = 30, fold: float = 60.0, n_donors: int = 4):
    """STIM raises 30 response genes 60x in B, T and Mono, not in the small, broad Mk type
    (as interferon in Kang); (AnnData with cell_type, donor, stim; bulk(fractions, seed, cond))."""
    rng = np.random.default_rng(seed)
    order = ["B", "T", "Mono", "Mk"]
    theta = rng.gamma(0.6, 1.0, size=(n_genes, 4))
    theta /= theta.sum(axis=0)
    theta[:, 3] = theta[:, :3].mean(axis=1) * np.exp(rng.normal(0, 0.1, n_genes))  # Mk: broad
    for k in range(3):
        theta[20 + k * 25:20 + (k + 1) * 25, k] *= 25
    theta[95:105, 3] *= 3
    theta /= theta.sum(axis=0)
    ifn = np.arange(n_genes - n_ifn, n_genes)
    sizes = np.array([CSIZES[t] for t in order])

    def profile(r, cond):
        prof = theta * np.exp(r.normal(0, 0.05, size=(n_genes, 1)))
        if cond == "STIM":
            prof[np.ix_(ifn, [0, 1, 2])] *= fold
        return prof / prof.sum(axis=0)

    n_of = {"B": 60, "T": 80, "Mono": 70, "Mk": 6}
    blocks, ct, donor, stim = [], [], [], []
    for d in range(n_donors):
        for cond in ("CTRL", "STIM"):
            prof = profile(rng, cond)
            for k, t in enumerate(order):
                blocks.append(rng.poisson(prof[:, k] * sizes[k], size=(n_of[t], n_genes)))
                ct += [t] * n_of[t]
                donor += [f"D{d}"] * n_of[t]
                stim += [cond] * n_of[t]
    X = sp.csr_matrix(np.vstack(blocks).astype(np.float32))
    obs = pd.DataFrame({"cell_type": pd.Categorical(ct), "donor": pd.Categorical(donor),
                        "stim": pd.Categorical(stim)}, index=[f"c{i}" for i in range(X.shape[0])])
    obs["orig_ident"] = obs["stim"]  # Kang repeats the condition under another name
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))

    def bulk(fr: dict, s: int, cond: str = "CTRL", n_cells: int = 5000) -> np.ndarray:
        r = np.random.default_rng(s)
        n_k = np.array([fr[t] for t in order]) * n_cells
        return r.poisson(profile(r, cond) @ (n_k * sizes)).astype(np.float64)

    return a, bulk, list(a.var_names)


def _ctrl_bulk(bulk, names) -> bytes:
    return _tsv(names, {s: bulk(p, 10 + i) for i, (s, p) in enumerate(CTRUTH.items())})


@pytest.mark.parametrize("method", ["weighted", "nnls"])
def test_condition_mismatch_is_flagged_and_matching_cells_fix_it(method):
    a, bulk, names = two_conditions()
    raw = _ctrl_bulk(bulk, names)
    mixed = dc.run(a, raw, "b.tsv", celltype_column="cell_type", donor_column="donor", method=method)
    est = mixed.fractions.pivot(index="sample", columns="cell_type", values="cell_fraction")
    assert (est["Mk"] >= 0.2).all()  # truth 0.02: Mk absorbs the missing response
    assert {f["cell_type"] for f in mixed.flags} == {"Mk"} and len(mixed.flags) == 3
    assert mixed.flags[0]["reference_share"] == pytest.approx(6 / 216, abs=1e-6)
    warn = [w for w in mixed.warnings if w.startswith("Mk is ")]
    assert len(warn) == 1 and "in 3 of 3 samples but 3% of the reference cells" in warn[0]
    assert "Try restricting the reference cells" in warn[0]
    assert mixed.reference_cells is None

    ctrl = dc.run(a, raw, "b.tsv", celltype_column="cell_type", donor_column="donor", method=method,
                  cells_column="stim", cells_value="CTRL")
    assert ctrl.reference_cells == {"column": "stim", "value": "CTRL", "n_cells": 4 * 216}
    assert ctrl.n_reference_cells == 4 * 216 and not ctrl.flags
    est = ctrl.fractions.pivot(index="sample", columns="cell_type", values="cell_fraction").loc[list(CTRUTH), CTYPES]
    truth = pd.DataFrame(CTRUTH).T.loc[list(CTRUTH), CTYPES]
    np.testing.assert_allclose(est.to_numpy(), truth.to_numpy(), atol=0.05)


def test_flags_need_both_a_large_fraction_and_a_large_ratio():
    share = np.array([0.5, 0.02, 0.1])
    P = np.array([[0.6, 0.15, 0.25], [0.2, 0.3, 0.5]])
    flags = dc.absorber_flags(["a", "b"], ["X", "Y", "Z"], P, share)
    # a: X 0.6 is only 1.2x its share, Y 0.15 under 20%; b: Y 0.3 = 15x, Z 0.5 = 5x.
    assert [(f["sample"], f["cell_type"]) for f in flags] == [("b", "Y"), ("b", "Z")]
    assert dc.absorber_warnings(flags[:1], 2, restricted=True)[0].startswith("Y is 30% in b but 2% of the reference")


def test_reference_columns_offer_the_conditions_once():
    a, _, _ = two_conditions()
    cols = dc.reference_columns(a, exclude=("cell_type", "donor"))
    assert [c["column"] for c in cols] == ["stim"]  # orig_ident repeats it
    assert {v["value"]: v["n_cells"] for v in cols[0]["values"]} == {"CTRL": 864, "STIM": 864}


def test_reference_cells_must_exist():
    a, _, _ = two_conditions()
    with pytest.raises(dc.DeconvolutionError, match="No cells have stim = LPS"):
        dc.build_reference(a, "cell_type", "donor", "stim", "LPS")
    with pytest.raises(dc.DeconvolutionError, match="not found"):
        dc.build_reference(a, "cell_type", "donor", "nope", "x")
    with pytest.raises(dc.DeconvolutionError, match="Pick which value"):
        dc.build_reference(a, "cell_type", "donor", "stim", "")


def test_self_check_and_export_use_the_reference_cells():
    a, _, _ = two_conditions()
    c = dc.check(a, celltype_column="cell_type", donor_column="donor", n_mixtures=3, cells_column="stim",
                 cells_value="CTRL")
    assert c.reference_cells["value"] == "CTRL"
    assert any("condition or tissue the reference does not match is not tested" in n for n in c.notes)
    data = b"".join(dc.iter_reference_zip(a, celltype_column="cell_type", donor_column="donor",
                                          cells_column="stim", cells_value="CTRL"))
    z = zipfile.ZipFile(io.BytesIO(data))
    cells = pd.read_csv(io.BytesIO(z.read("cells.tsv")), sep="\t")
    assert len(cells) == 864 and set(cells["cell_id"]) == set(a.obs_names[a.obs["stim"] == "CTRL"])
    assert "with stim = CTRL" in z.read("README.txt").decode()


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


async def test_api_reference_cells_option_flags_and_fact(client, dataset_manager):
    a, bulk, names = two_conditions()
    did = _register(dataset_manager, a, "dc6")
    url = f"{API}/datasets/{did}/deconvolution"
    d = (await client.get(f"{url}/defaults")).json()
    assert [c["column"] for c in d["reference_columns"]] == ["stim"]
    raw = _ctrl_bulk(bulk, names)
    body = (await client.post(f"{url}/upload", files={"file": ("b.tsv", raw)},
                              data={"options": json.dumps({"method": "weighted"})})).json()
    assert body["reference_cells"] is None and all(s["flagged"] == ["Mk"] for s in body["samples"])
    assert body["flags"][0]["cell_type"] == "Mk" and any(w.startswith("Mk is ") for w in body["warnings"])
    lines, _ = dc.grounding_lines(await dataset_manager.get_or_load_dataset(did))
    assert any(ln.startswith("- Doubtful estimates: Mk is ") for ln in lines)

    opts = {"method": "weighted", "cells_column": "stim", "cells_value": "CTRL"}
    r = await client.post(f"{url}/upload", data={"options": json.dumps(opts)})  # the kept file again
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference_cells"] == {"column": "stim", "value": "CTRL", "n_cells": 864}
    assert not body["flags"] and all(not s["flagged"] for s in body["samples"])
    assert all(s["cell_fractions"]["Mk"] < 0.06 for s in body["samples"])
    lines, detail = dc.grounding_lines(await dataset_manager.get_or_load_dataset(did))
    assert "with stim = CTRL" in detail and any("Reference cells: only those with stim = CTRL" in ln for ln in lines)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    step = [h for h in provenance.read_provenance(adata)["history"] if h["step"] == dc.STEP][-1]
    assert step["params"]["reference_cells"] == {"column": "stim", "value": "CTRL"}

    r = await client.post(f"{url}/validate", json={"n_mixtures": 2, "cells_column": "stim", "cells_value": "STIM"})
    res = dict(_sse(r.text))["result"]
    assert res["reference_cells"]["value"] == "STIM"
    r = await client.get(f"{url}/reference.zip", params={"cells_column": "stim", "cells_value": "CTRL"})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert r.status_code == 200 and len(pd.read_csv(io.BytesIO(z.read("cells.tsv")), sep="\t")) == 864
    r = await client.post(f"{url}/upload", data={"options": json.dumps({"cells_column": "stim", "cells_value": "X"})})
    assert r.status_code == 422 and "No cells have stim = X" in r.json()["detail"]


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


def test_flags_use_the_largest_share_in_one_donor():
    # Kang CTRL: CD8 T is 5% of the pooled cells but 12% in one donor; NNLS on a matched
    # reference put it near 30% in two samples. 5x the pooled share flagged it, 5x 12% does not.
    share = np.array([0.05, 0.02])
    share_max = np.array([0.12, 0.03])
    P = np.array([[0.3, 0.1], [0.1, 0.4]])
    assert [(f["sample"], f["cell_type"]) for f in dc.absorber_flags(["a", "b"], ["CD8", "Mk"], P, share)] == [
        ("a", "CD8"), ("b", "Mk")]
    flags = dc.absorber_flags(["a", "b"], ["CD8", "Mk"], P, share, share_max)
    assert [(f["sample"], f["cell_type"]) for f in flags] == [("b", "Mk")]
    assert flags[0]["reference_share"] == pytest.approx(0.02)


def test_largest_donor_share_skips_small_donors():
    ref = dc.Reference(
        genes=np.array(["G0"]), gene_ids=None, usable=np.array([True]), types=["A", "B"], donors=["d0", "d1", "d2"],
        P=np.ones((5, 1)), group_type=np.array([0, 1, 0, 1, 0]), group_donor=np.array([0, 0, 1, 1, 2]),
        group_cells=np.array([90.0, 10.0, 80.0, 20.0, 5.0]), totals=np.ones(5), celltype_column="t",
        donor_column="d", donor_reason=None, counts_source="X", cell_type=np.zeros(1, dtype=int),
        cell_donor=np.zeros(1, dtype=int))
    # d2 (5 cells, all A) is left out: A at most 90% (d0), B at most 20% (d1).
    np.testing.assert_allclose(ref.type_share_max, [0.9, 0.2])
