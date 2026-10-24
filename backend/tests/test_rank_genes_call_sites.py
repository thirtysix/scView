"""Every marker / DE call site, by blocks of genes, against scanpy's own for the same call.

Each call site is run twice on equal data: once with ``rank_genes`` made to fall back,
so scanpy's ``rank_genes_groups`` runs with exactly the arguments the call site passes
(what the code called before), and once on the blocks path with scanpy's function made
to fail. Every ``uns`` entry the call writes is captured both times and must match bit
for bit (names in order, scores, p-values, adjusted p-values, fold changes, pts), and so
must what the call site returns. DE between groups is also checked against scanpy's
``reference=`` mode on the whole matrix.

Data as the pipeline leaves it: Kang (sparse .raw, log-normalised; X scaled variable
genes) and paul15 (dense), from tests/fixtures/sparse_markers.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.api.v1 import de as de_api
from scview.api.v1 import markers as markers_api
from scview.api.v1 import metadata as metadata_api
from scview.core import de_methods, marker_db, merge_check, pipeline, rank_genes
from scview.core import marker_specificity as ms
from scview.core.marker_selection import group_markers
from scview.core.uns_keys import markers_key
from tests.test_sparse_markers import _kang, _paul15

FIELDS = ("names", "scores", "pvals", "pvals_adj", "logfoldchanges")


def _processed(name: str):
    """(adata, cluster column) as after a pipeline run: .raw log-normalised (every gene),
    X the scaled variable genes."""
    a, col = (_kang(), "cluster") if name == "kang" else (_paul15(), "paul15_clusters")
    a.obs[col] = a.obs[col].astype(str).astype("category")
    a.raw = a.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=60, flavor="seurat")
    a = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(a, max_value=10)
    return a, col


class Runner:
    """Runs a call site on the scanpy path or the blocks path, capturing every entry
    ``rank_genes.rank_genes_groups`` writes."""

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        self.real = rank_genes.rank_genes_groups
        self.real_unsupported = rank_genes.unsupported
        self.real_sc = sc.tl.rank_genes_groups

    def run(self, fn, *, blocks: bool):
        captured = []

        def capture(adata, groupby, **kw):
            path = self.real(adata, groupby, **kw)
            assert path == (rank_genes.SPARSE if blocks else rank_genes.SCANPY)
            captured.append(copy.copy(adata.uns[kw.get("key_added") or "rank_genes_groups"]))
            return path

        def fail(*a, **k):
            raise AssertionError("scanpy's rank_genes_groups was called")

        self.mp.setattr(rank_genes, "rank_genes_groups", capture)
        if blocks:
            self.mp.setattr(sc.tl, "rank_genes_groups", fail)
        else:
            self.mp.setattr(rank_genes, "unsupported", lambda *a, **k: "forced for the test")
        try:
            out = fn()
        finally:
            self.mp.setattr(rank_genes, "rank_genes_groups", self.real)
            self.mp.setattr(rank_genes, "unsupported", self.real_unsupported)
            self.mp.setattr(sc.tl, "rank_genes_groups", self.real_sc)
        assert captured, "the call site did not rank genes"
        return out, captured

    def both(self, make, fn):
        """``fn(data)`` on two equal copies made by ``make()``: (scanpy's, ours)."""
        ref = self.run(lambda: fn(make()), blocks=False)
        got = self.run(lambda: fn(make()), blocks=True)
        return ref, got


@pytest.fixture()
def runner(monkeypatch):
    return Runner(monkeypatch)


def _same_entry(ref: dict, got: dict) -> None:
    assert [k for k in got if k != ms.SPECIFICITY_FIELD] == list(ref)
    assert got["params"] == ref["params"]
    for field in FIELDS:
        if field not in ref:
            continue
        assert got[field].dtype == ref[field].dtype
        for g in ref[field].dtype.names:
            np.testing.assert_array_equal(got[field][g], ref[field][g], err_msg=f"{field} {g}")
    for field in ("pts", "pts_rest"):
        if field in ref:
            pd.testing.assert_frame_equal(got[field], ref[field], check_exact=True)


def _same_captures(ref_caps, got_caps) -> None:
    assert len(ref_caps) == len(got_caps)
    for r, g in zip(ref_caps, got_caps):
        _same_entry(r, g)


@pytest.mark.parametrize("name", ["kang", "paul15"])
def test_pipeline_marker_step(runner, name):
    a, col = _processed(name)
    params = pipeline.PipelineParams(marker_columns=[col, "Donor"] if name == "kang" else [col])

    def step(adata):
        pipeline._run_marker_genes(adata, params)
        return adata

    (ref, ref_caps), (got, got_caps) = runner.both(lambda: a.copy(), step)
    _same_captures(ref_caps, got_caps)
    for key in [k for k in ref.uns if str(k).startswith("rank_genes_groups")]:
        _same_entry({k: v for k, v in ref.uns[key].items() if k != ms.SPECIFICITY_FIELD},
                    got.uns[key])
    # tau as marker_specificity attaches it on scanpy's path
    rgg, ref_rgg = got.uns[markers_key(col)], ref.uns[markers_key(col)]
    pd.testing.assert_frame_equal(rgg[ms.SPECIFICITY_FIELD], ref_rgg[ms.SPECIFICITY_FIELD])
    assert got.uns["rank_genes_groups"] is not None


def test_markers_for_a_new_column_and_the_llm_markers(runner):
    a, _ = _processed("kang")
    a.obs["stimulated"] = (a.obs["Donor"].astype(str).str[-1] < "5").to_numpy()  # bool column

    def new_column(adata):
        pipeline._markers_for_column(adata, "stimulated")
        return adata

    (ref, ref_caps), (got, got_caps) = runner.both(lambda: a.copy(), new_column)
    _same_captures(ref_caps, got_caps)
    assert ms.has_specificity(got.uns[markers_key("stimulated")])

    (ref, ref_caps), (got, got_caps) = runner.both(
        lambda: a.copy(), lambda adata: pipeline._top_markers_per_group(adata, "cluster"))
    _same_captures(ref_caps, got_caps)
    assert ref == got


@pytest.mark.parametrize("name", ["kang", "paul15"])
def test_on_demand_markers_endpoint(runner, name):
    a, col = _processed(name)
    a.uns["log1p"] = {"base": None}
    a.obs["sub"] = a.obs[col].astype(str).where(np.arange(a.n_obs) % 3 > 0, "other")

    (ref, ref_caps), (got, got_caps) = runner.both(
        lambda: a, lambda adata: markers_api._compute_markers(adata, "sub"))
    _same_captures(ref_caps, got_caps)
    _same_entry(ref, got)
    assert got["params"]["use_raw"] is True
    # what the pipeline would store for that column: scanpy on the dataset itself
    b = a.copy()
    b.obs["sub"] = b.obs["sub"].astype("category")
    sc.tl.rank_genes_groups(b, "sub", method="wilcoxon", pts=True)
    _same_entry(b.uns["rank_genes_groups"], got)


def test_on_demand_markers_follow_the_log_base():
    a, col = _processed("kang")
    a.uns["log1p"] = {"base": 2.0}
    got = markers_api._compute_markers(a, col)
    b = a.copy()
    sc.tl.rank_genes_groups(b, col, method="wilcoxon", pts=True)  # the pipeline's own
    _same_entry(b.uns["rank_genes_groups"], got)


def test_merging_two_categories_recomputes_markers(runner):
    a, col = _processed("kang")
    sc.tl.rank_genes_groups(a, col, method="wilcoxon", pts=True, key_added=markers_key(col))
    a.uns["rank_genes_groups"] = a.uns[markers_key(col)]
    cats = list(a.obs[col].cat.categories)

    def merge(adata):
        c = adata.obs[col].copy()
        c[c == cats[1]] = cats[0]
        adata.obs[col] = c.cat.remove_unused_categories()
        metadata_api._follow_rename(adata, col, cats[1], cats[0], True)
        return adata

    (ref, ref_caps), (got, got_caps) = runner.both(lambda: a.copy(), merge)
    _same_captures(ref_caps, got_caps)
    assert got.uns["rank_genes_groups"] is got.uns[markers_key(col)]
    assert cats[1] not in got.uns[markers_key(col)]["names"].dtype.names
    assert ms.has_specificity(got.uns[markers_key(col)])


def test_marker_database_check_markers(runner):
    a, col = _processed("kang")
    labels = a.obs[col].astype(str).to_numpy().copy()
    labels[[3, 400]] = "pair"  # two cells: left out (fewer than 3)
    labels[10:25] = None
    a.obs["g"] = pd.Categorical(labels)
    groups = marker_db._group_names(a, "g")
    assert "pair" in groups

    (ref, ref_caps), (got, got_caps) = runner.both(
        lambda: a, lambda adata: marker_db._computed(adata, "g", groups))
    _same_captures(ref_caps, got_caps)
    assert "pair" not in got_caps[0]["names"].dtype.names
    assert group_markers(got, "g", n=25) == group_markers(ref, "g", n=25)


@pytest.mark.parametrize("name", ["kang", "paul15"])
def test_merge_check_pair_test(runner, name):
    a, col = _processed(name)
    log = a.raw.X
    codes = a.obs[col].cat.codes.to_numpy()
    in_a, in_b = np.flatnonzero(codes == 0), np.flatnonzero(codes == 1)

    (ref, ref_caps), (got, got_caps) = runner.both(
        lambda: None, lambda _: merge_check.test_pair(log, a.raw.var_names, in_a, in_b))
    _same_captures(ref_caps, got_caps)
    assert got_caps[0]["params"]["reference"] == "b"
    assert ref == got and got["n_tested_genes"] > 0


def _de_sides(a, col):
    codes = a.obs[col].cat.codes.to_numpy()
    sel = codes == 0
    return [("selection vs rest", sel, ~sel), ("between groups", sel, codes == 2)]


@pytest.mark.parametrize("name", ["kang", "paul15"])
@pytest.mark.parametrize("method", ["wilcoxon", "t-test"])
def test_cell_level_de(runner, name, method):
    a, col = _processed(name)
    for label, mask_a, mask_b in _de_sides(a, col):
        for min_pct in (0.0, 0.1):
            test, _pre = de_api._prepare(a, mask_a, mask_b, min_pct)
            (ref, ref_caps), (got, got_caps) = runner.both(
                lambda: test.copy(), lambda t: de_methods.run(t, method))
            _same_captures(ref_caps, got_caps)
            for field in ("genes", "scores", "logfoldchanges", "pvals", "pvals_adj"):
                np.testing.assert_array_equal(getattr(got, field), getattr(ref, field),
                                              err_msg=f"{label} {min_pct} {field}")


@pytest.mark.parametrize("name", ["kang", "paul15"])
@pytest.mark.parametrize("method", ["wilcoxon", "t-test"])
def test_de_between_groups_is_scanpys_reference_mode(name, method, monkeypatch):
    # A vs B on the two groups' cells equals scanpy's reference= mode on every cell
    a, col = _processed(name)
    cats = list(a.obs[col].cat.categories)
    codes = a.obs[col].cat.codes.to_numpy()
    test, _pre = de_api._prepare(a, codes == 0, codes == 2, 0.0)  # every gene tested
    real = sc.tl.rank_genes_groups
    monkeypatch.setattr(sc.tl, "rank_genes_groups", None)  # the blocks path only
    got = de_methods.run(test, method)
    monkeypatch.setattr(sc.tl, "rank_genes_groups", real)
    b = a.copy()
    sc.tl.rank_genes_groups(b, col, groups=[cats[0]], reference=cats[2], method=method)
    rgg = b.uns["rank_genes_groups"]
    names = np.asarray(rgg["names"][cats[0]]).astype(str)
    idx = pd.Index(got.genes).get_indexer(names)
    assert (idx >= 0).all() and idx.size == got.genes.size
    np.testing.assert_array_equal(got.scores[idx].astype(np.float32), rgg["scores"][cats[0]])
    np.testing.assert_array_equal(got.pvals[idx], rgg["pvals"][cats[0]])
    np.testing.assert_array_equal(got.pvals_adj[idx], rgg["pvals_adj"][cats[0]])
    np.testing.assert_array_equal(got.logfoldchanges[idx].astype(np.float32),
                                  rgg["logfoldchanges"][cats[0]])
