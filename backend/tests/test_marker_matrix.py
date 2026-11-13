"""Markers are ranked on log-normalised values, never the raw counts a file keeps in .raw.

A CELLxGENE file keeps raw COUNTS in .raw beside a log-normalised X. Markers ranked .raw
whenever it was there: kidney MAGI2 log2FC 437 (research/v075-qa). The choice is now the gene
set scoring's (core/marker_matrix.py), each table records it, and a stored table ranked on
counts is stale: computed again on the next request.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import marker_matrix, rank_genes, replay_script
from scview.core.ora_background import resolve_markers_key
from scview.core.uns_keys import markers_key

API = "/api/v1"


def _counts(n_per: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.5, size=(3 * n_per, 40)).astype("float32")
    X[:n_per, :8] += rng.poisson(80, size=(n_per, 8))  # deep marker genes: tens of counts
    X[n_per:2 * n_per, 8:16] += rng.poisson(80, size=(n_per, 8))
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs_names = [f"C{i}" for i in range(3 * n_per)]
    a.obs["cell_type"] = pd.Categorical(np.array(["T", "B", "NK"]).repeat(n_per))
    return a


def _lognorm(counts: ad.AnnData):
    from scview.core.normalization import shifted_log

    return shifted_log(counts.X).astype(np.float32)


def cellxgene() -> ad.AnnData:
    """X log-normalised, .raw the raw counts (the CELLxGENE layout)."""
    a = _counts()
    raw = a.copy()
    a.X = _lognorm(a)
    a.raw = raw
    return a


def classic() -> ad.AnnData:
    """.raw log-normalised, X scaled (scanpy's tutorial layout)."""
    a = _counts()
    a.X = _lognorm(a)
    a.raw = a.copy()
    X = a.X.toarray()
    a.X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return a


def _lfc(rgg) -> np.ndarray:
    return np.asarray(rgg["logfoldchanges"].tolist(), dtype=float)


def _top(rgg, group: str, n: int = 3) -> list[str]:
    return [str(g) for g in rgg["names"][group][:n]]


def test_cellxgene_layout_ranks_x_not_the_counts_in_raw():
    a = cellxgene()
    label = marker_matrix.rank(a, "cell_type")
    rgg = a.uns["rank_genes_groups"]
    assert label == marker_matrix.X_LOG and rgg["params"]["matrix"] == label
    assert np.nanmax(np.abs(_lfc(rgg))) < 15  # log values: a few units, not hundreds
    # the same table as ranking X directly
    b = a.copy()
    rank_genes.rank_genes_groups(b, "cell_type", method="wilcoxon", pts=True, use_raw=False)
    np.testing.assert_array_equal(_lfc(rgg), _lfc(b.uns["rank_genes_groups"]))
    # what used to happen: the counts in .raw, fold changes far past any log value
    c = a.copy()
    rank_genes.rank_genes_groups(c, "cell_type", method="wilcoxon", pts=True, use_raw=True)
    assert marker_matrix.on_counts(c.uns["rank_genes_groups"])


def test_classic_layout_still_ranks_raw_bit_for_bit():
    a = classic()
    b = a.copy()
    assert marker_matrix.rank(a, "cell_type") == marker_matrix.RAW_LOG
    rank_genes.rank_genes_groups(b, "cell_type", method="wilcoxon", pts=True, specificity=True)
    for f in ("logfoldchanges", "pvals", "scores"):
        np.testing.assert_array_equal(np.asarray(a.uns["rank_genes_groups"][f].tolist()),
                                      np.asarray(b.uns["rank_genes_groups"][f].tolist()))
    assert _top(a.uns["rank_genes_groups"], "T") == _top(b.uns["rank_genes_groups"], "T")


def test_counts_only_ranks_their_shifted_log():
    a = _counts()
    label = marker_matrix.rank(a, "cell_type")
    assert label == "shifted log of the counts in X"
    ref = cellxgene()
    marker_matrix.rank(ref, "cell_type")
    # the shifted log of the counts is X of the CELLxGENE layout: the same markers
    assert _top(a.uns["rank_genes_groups"], "T") == _top(ref.uns["rank_genes_groups"], "T")
    np.testing.assert_allclose(_lfc(a.uns["rank_genes_groups"]), _lfc(ref.uns["rank_genes_groups"]),
                               rtol=1e-5, atol=1e-5)
    assert set(_top(a.uns["rank_genes_groups"], "T", 8)) == {f"G{i}" for i in range(8)}


def test_on_counts_reads_the_fold_changes_and_the_matrix_named():
    a = cellxgene()
    rank_genes.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True, use_raw=True)
    rgg = a.uns["rank_genes_groups"]
    assert marker_matrix.on_counts(rgg)
    rgg["params"]["matrix"] = "log-normalised X"  # a table that names its matrix is trusted
    assert not marker_matrix.on_counts(rgg)
    # sane fold changes, but it says it ranked .raw and .raw holds counts
    b = cellxgene()
    rank_genes.rank_genes_groups(b, "cell_type", method="wilcoxon", pts=True, use_raw=False)
    legacy = b.uns["rank_genes_groups"]
    legacy["params"]["use_raw"] = True
    assert not marker_matrix.on_counts(legacy)  # the table alone cannot tell
    assert marker_matrix.on_counts(legacy, b)
    # an infinite fold change (expm1 of a count past 709) is counts; NaN alone is not judged
    def table(values):
        return {"params": {"use_raw": True},
                "logfoldchanges": np.rec.fromarrays([np.array(values)], names=["T"])}

    assert marker_matrix.on_counts(table([np.inf, 1.0]))
    assert not marker_matrix.on_counts(table([np.nan, 1.0]))


def test_resolve_markers_key_skips_a_counts_table():
    a = cellxgene()
    rank_genes.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True, use_raw=True,
                                 key_added=markers_key("cell_type"))
    a.uns["rank_genes_groups"] = a.uns[markers_key("cell_type")]
    assert resolve_markers_key(a.uns, "cell_type") is None
    assert resolve_markers_key(a.uns, "") is None
    marker_matrix.rank(a, "cell_type", key_added=markers_key("cell_type"))
    assert resolve_markers_key(a.uns, "cell_type") == markers_key("cell_type")


def test_the_pipeline_marker_step_ranks_the_log_values():
    from scview.core import pipeline

    a = cellxgene()
    pipeline._wilcoxon(a, "cell_type", key_added=markers_key("cell_type"))
    rgg = a.uns[markers_key("cell_type")]
    assert rgg["params"]["matrix"] == marker_matrix.X_LOG
    assert np.nanmax(np.abs(_lfc(rgg))) < 15


def test_the_raw_layer_is_labelled_counts_when_it_holds_counts():
    from scview.core.anndata_adaptor import AnnDataAdaptor

    for adata, label in ((cellxgene(), "Counts (raw)"), (classic(), "Log-normalized (raw)")):
        adaptor = AnnDataAdaptor("unused.h5ad")
        adaptor._adata = adata
        assert adaptor.layer_label("raw") == label
        out = adaptor.expression_by_group(["G0"], "cell_type")
        assert out["layer"] == "raw" and out["layer_label"] == label


def test_the_replay_helper_skips_counts_in_raw():
    import scanpy as sc

    ns = {"np": np, "sp": sp, "sc": sc}
    for name in ("looks_like_counts", "markers_key"):
        exec(replay_script._HELPERS[name], ns)
    assert "looks_like_counts" in replay_script._HELPER_NEEDS["markers_key"]
    a = cellxgene()
    ns["markers_for"](a, "cell_type")
    assert a.uns[markers_key("cell_type")]["params"]["use_raw"] is False
    b = classic()
    ns["markers_for"](b, "cell_type")
    assert b.uns[markers_key("cell_type")]["params"]["use_raw"] is True


# --- through the markers route -------------------------------------------------------------


def _register(dm, did: str, adata: ad.AnnData) -> None:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")


async def _json(client, did: str, **params) -> dict:
    r = await client.get(f"{API}/datasets/{did}/markers", params={"format": "json", **params})
    assert r.status_code == 200, r.text
    return r.json()


async def test_route_on_a_cellxgene_layout_gives_log_fold_changes(client, dataset_manager):
    _register(dataset_manager, "cxg", cellxgene())
    d = await _json(client, "cxg", groupby_column="cell_type")
    assert d["matrix"] == marker_matrix.X_LOG
    lfc = [r["logfoldchange"] for rs in d["markers"].values() for r in rs]
    assert max(abs(v) for v in lfc if v is not None) < 15
    assert [r["gene"] for r in d["markers"]["T"][:8]] and {
        r["gene"] for r in d["markers"]["T"][:8]} == {f"G{i}" for i in range(8)}


async def test_route_recomputes_a_stored_counts_table(client, dataset_manager):
    a = cellxgene()
    rank_genes.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True, use_raw=True,
                                 key_added=markers_key("cell_type"))
    a.uns["rank_genes_groups"] = a.uns[markers_key("cell_type")].copy()
    _register(dataset_manager, "stale", a)
    d = await _json(client, "stale", groupby_column="cell_type")
    assert d["matrix"] == marker_matrix.X_LOG
    assert max(abs(r["logfoldchange"]) for rs in d["markers"].values() for r in rs
               if r["logfoldchange"] is not None) < 15
    # the bare request finds the default table stale too and computes it for its column
    bare = await _json(client, "stale")
    assert bare.get("matrix") == marker_matrix.X_LOG and "reason" not in bare
    adaptor = await dataset_manager.get_or_load_dataset("stale")
    for key in (markers_key("cell_type"), "rank_genes_groups"):  # the default one replaced too
        assert adaptor.adata.uns[key]["params"]["matrix"] == marker_matrix.X_LOG
