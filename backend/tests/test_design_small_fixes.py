"""Small study-design fixes from the v0.75 QA round (Wilson 2022 kidney, Emont 2022 fat):
the default layer on a CELLxGENE file, Milo labels with k, a numeric age covariate, named
per-donor sides, the cell-level fold-change note, and compare_de_results in API terms."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi import HTTPException

from scview.api.v1.de import differential_expression_compare
from scview.core import de_methods, paired_design
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.test_mcp import _call, _register, msigdb_dir, test_settings  # noqa: F401 - fixtures
from tests.test_pseudobulk_covariates import A, B, _req
from tests.test_pseudobulk_voom import _DM, _counts


def _adaptor(a: ad.AnnData) -> AnnDataAdaptor:
    adaptor = AnnDataAdaptor("unused.h5ad")
    adaptor._adata = a
    return adaptor


def _cellxgene(raw_counts: bool = True) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, (50, 20)).astype(np.float32)
    logn = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(logn))
    a.var_names = [f"G{i}" for i in range(20)]
    a.raw = ad.AnnData(sp.csr_matrix(counts if raw_counts else logn), var=a.var.copy())
    return a


def test_default_layer_is_log_normalised_when_raw_holds_counts():
    # CELLxGENE: X log-normalised, .raw counts. Plots on counts track depth: default to X.
    cx = _adaptor(_cellxgene())
    assert cx.default_expression_layer() == "X"
    assert next(x for x in cx.available_expression_layers() if x["key"] == "raw")["label"] == "Counts (raw)"
    # scanpy / scView: .raw holds log values: it stays the default.
    sc_like = _adaptor(_cellxgene(raw_counts=False))
    assert sc_like.default_expression_layer() == "raw"
    assert next(x for x in sc_like.available_expression_layers()
                if x["key"] == "raw")["label"] == "Log-normalized (raw)"
    # Counts in both, or an HVG-only X: keep .raw.
    both = _cellxgene()
    both.X = both.raw.X.copy()
    assert _adaptor(both).default_expression_layer() == "raw"
    hvg = _cellxgene()[:, :5].copy()
    assert _adaptor(hvg).default_expression_layer() == "raw"


def test_milo_runs_with_another_k_or_covariate_get_their_own_label():
    from scview.core import milo

    def result(k, covs):
        r = milo.MiloResult.__new__(milo.MiloResult)
        r.n_neighbors, r.representation, r.covariates = k, "X_harmony", covs
        return r

    assert milo.run_settings(result(30, [])) == "k=30, X_harmony"
    assert milo.run_settings(result(30, ["assay"])) == "k=30, X_harmony, + assay"
    assert milo.run_settings(result(None, [])) == "k=?, X_harmony"


def _with_age(a):
    ages = {d: 50 + 3 * i for i, d in enumerate(A + B)}
    a.obs["development_stage"] = a.obs["donor"].astype(str).map(
        lambda d: f"{ages[d]}-year-old stage").astype("category")
    return a


@pytest.mark.asyncio
async def test_an_age_held_as_text_is_one_slope_when_asked():
    a = _with_age(_counts(A, B, seed=12))
    # As a factor: 12 levels for 12 samples, no residual df, and the hint says what to do.
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(covariates=["development_stage"]), _DM(a))
    assert "pass it in numeric_covariates" in e.value.detail
    res = await differential_expression_compare(
        "x", _req(numeric_covariates=["development_stage"]), _DM(a))
    assert res.design == "~development_stage + condition" and res.covariates == ["development_stage"]
    assert res.genes
    a.obs["development_stage"] = a.obs["development_stage"].astype(str).where(
        a.obs["donor"] != "a0", "unknown")
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(numeric_covariates=["development_stage"]), _DM(a))
    assert "'unknown' holds no number" in e.value.detail


def test_unpaired_tests_name_their_sides():
    points = [{"group": "g", "condition": c, "sample": f"{c}{i}", "value": float(i + (c == "HFD"))}
              for c in ("chow", "HFD") for i in range(4)]
    t = paired_design.group_unpaired_tests(points, ["chow", "HFD"], ["g"])["g"]
    assert t["n_donors"] == {"chow": 4, "HFD": 4} and t["n_a"] == 4


def test_one_sided_fold_changes_are_explained():
    assert de_methods.one_sided_note([1.0, -2.5, None]) is None
    note = de_methods.one_sided_note([25.3, -27.0, 1.0])
    assert note.startswith("2 genes have |log2FC| of 20 or more") and "1e-9" in note
    # The scanpy formula gives such values for a gene absent on one side.
    M = sp.csr_matrix(np.array([[2.0, 1.0], [1.5, 1.0], [0.0, 1.0], [0.0, 1.2]]))
    lfc = de_methods.logfoldchanges(M, np.array([True, True, False, False]), {})
    assert lfc[0] > de_methods.ONE_SIDED_LFC


async def test_compare_de_results_speaks_in_api_terms(client, dataset_manager):
    did = _register(dataset_manager)
    scope = {"dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL",
             "within_column": "cluster", "within_value": "A", "replicate_key": "donor_id"}
    cell = await _call("differential_expression", scope)
    pb = await _call("differential_expression", {**scope, "method": "pseudobulk"})
    r = await _call("compare_de_results", {"dataset_id": did, "result_a": cell["result_id"],
                                           "result_b": pb["result_id"]})
    assert not any("tick" in n for n in r["notes"])
    r = await _call("compare_de_results", {"dataset_id": did, "result_a": cell["result_id"],
                                           "result_b": pb["result_id"], "common_tested_only": True})
    assert any(n.startswith("Only the") and "every result tested" in n for n in r["notes"])


def test_api_words_replaces_the_checkbox():
    from scview.mcp.tools import _api_words

    note = ('The results tested different genes (10 to 12; 9 in all): tick "only genes tested in all" '
            "to count only those.")
    assert _api_words(note).endswith("pass common_tested_only=true to count only those.")
