"""Scrublet per capture channel (core/doublets.py, scrublet_lean.scrublet_by_channel).

Which cells share a channel (an explicit choice, a channel column, the barcodes, the
donors as a guess, else all cells), each channel scored alone with its own rate and
threshold, cells left unscored when their channel is too small, and every consumer of the
calls (summary, threshold, doublet source, filtering, QC report, co-pilot, replay; the
preflight's model is in tests/test_preflight.py). The expected scores are Scrublet's own
on each channel's rows, run by hand.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import doublet_source, doublets, provenance, qc_filter, scrublet_lean
from scview.core import pipeline as pl

FIXTURES = Path(__file__).parent / "fixtures" / "doublets"
LETTERS = np.array(list("ACGT"))


def _barcodes(n: int, length: int = 16, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(x) for x in LETTERS[rng.integers(0, 4, (n, length))]]


def _counts(sizes: list[int], n_genes: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Counts of channels of ``sizes`` cells, each with three populations (their own gene
    blocks, so Scrublet has variable genes) and 5% doublets of two of them, and each
    cell's channel index; rows shuffled so no channel is one block."""
    rng = np.random.default_rng(seed)
    blocks = []
    for n in sizes:
        X = rng.poisson(0.4, size=(n, n_genes)).astype(np.float32)
        per = n // 3
        for p, lo in enumerate((0, 60, 120)):
            rows = slice(p * per, (p + 1) * per if p < 2 else n)
            X[rows, lo:lo + 60] += rng.poisson(4.0, size=(X[rows].shape[0], 60))
        d = max(2, n // 20)
        X[n - d:] = X[:d] + X[per:per + d]  # doublets within this channel
        blocks.append(X)
    X = np.vstack(blocks)
    codes = np.repeat(np.arange(len(sizes)), sizes)
    perm = np.random.default_rng(seed + 1).permutation(X.shape[0])
    return X[perm], codes[perm]


def _dataset(sizes: list[int], seed: int = 0, column: str | None = "gem_well",
             names: str = "plain") -> ad.AnnData:
    X, codes = _counts(sizes, seed=seed)
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    if names == "suffix":  # Cell Ranger aggr: -1, -2, ...
        a.obs_names = [f"{b}-{c + 1}" for b, c in zip(_barcodes(a.n_obs, seed=seed), codes)]
    else:
        a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    if column:
        a.obs[column] = pd.Categorical([f"w{c + 1}" for c in codes])
    return a


def _obs(names, **columns) -> ad.AnnData:
    a = ad.AnnData(obs=pd.DataFrame(index=pd.Index([str(n) for n in names])))
    for k, v in columns.items():
        a.obs[k] = pd.Categorical(v)
    return a


# --- which cells share a channel --------------------------------------------------------


def test_kang_barcodes_are_its_two_lanes_not_its_pooled_donors():
    # Kang 2018: CTRL_AAACATACATTTCC.1 / STIM_...: two lanes, each holding 8 donors that
    # demuxlet told apart. The donors were pooled in one droplet emulsion per lane, so a
    # doublet can be two donors' cells: the channel is the lane.
    with gzip.open(FIXTURES / "kang_obs_names.txt.gz", "rt") as f:
        names = f.read().split()
    assert len(names) == 13836
    donors = [f"d{i % 8}" for i in range(len(names))]
    ch = doublets.channels(_obs(names, Donor=donors))
    assert (ch.column, ch.reason, ch.derived) == ("capture_channel", "barcode_prefix", True)
    assert ch.names == ["CTRL", "STIM"]
    assert ch.sizes.tolist() == [6573, 7263]
    assert all(n.startswith(lab) for n, lab in zip(names, np.asarray(ch.names)[ch.codes]))


def test_the_barcode_suffix_is_the_gem_well_and_sorts_as_numbers():
    codes = np.repeat(np.arange(12), 120)
    names = [f"{b}-{c + 1}" for b, c in zip(_barcodes(codes.size), codes)]
    ch = doublets.channels(_obs(names))
    assert (ch.column, ch.reason) == ("capture_channel", "barcode_suffix")
    assert ch.names == [str(i) for i in range(1, 13)]  # 10 after 9, not after 1
    assert np.array_equal(ch.codes, codes)


@pytest.mark.parametrize(("names", "want"), [
    # Seurat merge of two aggr runs: prefix and suffix both vary
    (["S1_" + b + "-1" for b in _barcodes(150, seed=1)] + ["S1_" + b + "-2" for b in _barcodes(150, seed=2)]
     + ["S2_" + b + "-1" for b in _barcodes(150, seed=3)], ["S1-1", "S1-2", "S2-1"]),
    # anndata concat after Cell Ranger: -1-0, -1-1
    ([b + "-1-0" for b in _barcodes(150, seed=1)] + [b + "-1-1" for b in _barcodes(150, seed=2)],
     ["1-0", "1-1"]),
    # v1 chemistry, 14 nt, a ':' prefix
    (["A:" + b for b in _barcodes(150, 14, 1)] + ["B:" + b for b in _barcodes(150, 14, 2)], ["A", "B"]),
])
def test_prefixes_and_suffixes_of_other_merges(names, want):
    found = doublets.barcode_channels(names)
    assert found is not None and found[0] == want
    assert np.bincount(found[1]).tolist() == [150] * len(want)


@pytest.mark.parametrize("names", [
    [b + "-1" for b in _barcodes(300)],  # one GEM well: nothing to split
    [f"cell{i}" for i in range(300)],  # not 10x barcodes
    [b + "-1" for b in _barcodes(299)] + ["oddname-2"],  # one name that is not a barcode
    [b[:15] + "-" + str(i % 2) for i, b in enumerate(_barcodes(300))],  # 15 nt: not 10x
])
def test_names_that_do_not_carry_channels(names):
    assert doublets.barcode_channels(names) is None


def test_an_explicit_choice_wins_and_all_cells_is_one_channel():
    names = [f"{b}-{1 + i % 2}" for i, b in enumerate(_barcodes(400))]
    a = _obs(names, lane=["L1"] * 200 + ["L2"] * 200, gem_well=["g"] * 400)
    assert doublets.channel_grouping(a, "lane") == ("lane", "requested")
    ch = doublets.channels(a, doublets.ALL_CELLS)
    assert (ch.column, ch.reason, ch.names) == (None, "requested_all", ["all cells"])
    assert (ch.codes == 0).all()
    # the barcodes' channels can be asked for by the column they are written to
    assert doublets.channel_grouping(a, "capture_channel") == ("capture_channel", "barcode_suffix")
    with pytest.raises(doublets.DoubletError, match="not found"):
        doublets.channels(a, "nope")


def test_a_column_named_for_the_channel_comes_before_the_barcodes_in_any_case():
    names = [f"{b}-{1 + i % 2}" for i, b in enumerate(_barcodes(400))]
    a = _obs(names, **{"GEM.Well": ["a"] * 100 + ["b"] * 300, "Donor": ["d1", "d2"] * 200})
    assert doublets.channel_grouping(a) == ("GEM.Well", "channel_column")
    a = _obs(names, Lane=["x"] * 400)  # one lane: one channel, from its column
    ch = doublets.channels(a)
    assert (ch.column, ch.reason, ch.sizes.tolist()) == ("Lane", "channel_column", [400])


def test_donors_then_batches_are_a_guess_and_have_no_group_cap():
    # Past the 50 groups a violin can show: the run has no such cap.
    a = _obs([f"c{i}" for i in range(60 * 120)], donor=[f"d{i // 120}" for i in range(60 * 120)])
    ch = doublets.channels(a)
    assert (ch.column, ch.reason, len(ch.names)) == ("donor", "donor", 60)
    assert ch.note == doublets.GUESS_NOTE
    b = _obs([f"c{i}" for i in range(400)], batch=["b1", "b2"] * 200)
    assert doublets.channel_grouping(b) == ("batch", "batch")
    assert doublets.channel_grouping(_obs([f"c{i}" for i in range(400)])) == (None, "none")


def test_a_grouping_too_fine_for_scrublet_is_passed_over():
    # Plate data: one cell per well. Scrublet cannot run on one cell.
    a = _obs([f"c{i}" for i in range(400)], well=[f"A{i}" for i in range(400)])
    ch = doublets.channels(a)
    assert (ch.column, ch.reason) == (None, "none")
    assert len(ch.skipped) == 1 and "'well'" in ch.skipped[0] and "median of 1 cells" in ch.skipped[0]
    s = doublets.suggest_rate(a)
    assert s["skipped"] == list(ch.skipped) and s["per_channel"] is False


def test_the_suggestion_gives_each_channel_its_own_rate():
    a = _obs([f"c{i}" for i in range(8050)], gem_well=["a"] * 2000 + ["b"] * 6000 + ["tiny"] * 50)
    s = doublets.suggest_rate(a)
    assert (s["column"], s["reason"], s["per_channel"], s["n_channels"]) == ("gem_well", "channel_column", True, 3)
    assert [(c["name"], c["n_cells"], c["rate"], c["scored"]) for c in s["channels"]] == [
        ("a", 2000, 0.016, True), ("b", 6000, 0.048, True), ("tiny", 50, doublets.MIN_RATE, False)]
    assert s["n_unscored"] == 50 and s["largest_channel"] == 6000
    assert s["rate"] == pytest.approx((2000 * 0.016 + 6000 * 0.048 + 50 * 0.004) / 8050, abs=1e-4)


# --- scoring each channel alone ----------------------------------------------------------


@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_one_channel_of_every_cell_is_bitwise_one_scrublet_run(kind):
    a = _dataset([360], column=None)
    if kind == "csr":
        a.X = sp.csr_matrix(a.X)
    X = a.X.copy()
    want_s, want_c, want_u = scrublet_lean.scrublet_counts(
        X, expected_doublet_rate=doublets.rate_for_cells(360), use_approx_neighbors=False)
    found = doublets.run_scrublet(a, a.X)
    assert np.array_equal(a.obs["doublet_score"].to_numpy(), want_s, equal_nan=True)
    scored = np.isfinite(want_s)
    assert np.array_equal(a.obs["predicted_doublet"].to_numpy()[scored], want_c[scored].astype(bool))
    u = a.uns["scrublet"]
    assert np.array_equal(u["doublet_scores_sim"], want_u["doublet_scores_sim"])
    assert u["threshold"] == want_u["threshold"] and doublets.scrublet_threshold(a) == want_u["threshold"]
    assert found["expected_doublet_rate"] == doublets.rate_for_cells(360)
    assert found["channel_column"] is None and found["channel_reason"] == "none"


def test_each_channel_is_scrublet_on_its_own_rows_with_its_own_rate():
    sizes = [150, 260, 60, 200]  # the third is under MIN_CHANNEL_CELLS
    a = _dataset(sizes)
    a.obs.loc[a.obs_names[:7], "gem_well"] = np.nan  # cells with no channel
    X = a.X.copy()
    found = doublets.run_scrublet(a, a.X)
    labels = a.obs["gem_well"].astype(str).where(a.obs["gem_well"].notna()).to_numpy()
    score = a.obs["doublet_score"].to_numpy()
    call = a.obs["predicted_doublet"].to_numpy()
    assert a.obs["predicted_doublet"].dtype == bool
    rows = {r["name"]: r for r in found["channels"]}
    for name in ("w1", "w2", "w4"):
        idx = np.flatnonzero(labels == name)
        rate = doublets.rate_for_cells(idx.size)
        s, c, u = scrublet_lean.scrublet_counts(X[idx], expected_doublet_rate=rate, use_approx_neighbors=False)
        assert np.array_equal(score[idx], s, equal_nan=True), name
        assert np.array_equal(call[idx], np.where(np.isfinite(s), c, False).astype(bool)), name
        assert rows[name]["expected_doublet_rate"] == rate and rows[name]["status"] == "scored"
        assert rows[name]["threshold"] == pytest.approx(float(u["threshold"]))
        assert rows[name]["n_called"] == int(call[idx].sum())
    small = np.flatnonzero(labels == "w3")
    assert np.isnan(score[small]).all() and not call[small].any() and rows["w3"]["status"] == "too_small"
    none = np.flatnonzero(pd.isna(labels))
    assert none.size == 7 and np.isnan(score[none]).all() and not call[none].any()
    assert found["n_unscored"] == int(np.isnan(score).sum()) >= small.size + 7
    # One run over all of them would not give these scores (other neighbours, one rate).
    whole, _, _ = scrublet_lean.scrublet_counts(X, expected_doublet_rate=0.05, use_approx_neighbors=False)
    w1 = np.flatnonzero(labels == "w1")
    assert not np.allclose(whole[w1], score[w1], equal_nan=True)
    # A rate set by the user is every channel's.
    found = doublets.run_scrublet(a, a.X, rate=0.07)
    assert {r["expected_doublet_rate"] for r in found["channels"]} == {0.07}
    idx = np.flatnonzero(labels == "w2")
    s, _, _ = scrublet_lean.scrublet_counts(X[idx], expected_doublet_rate=0.07, use_approx_neighbors=False)
    assert np.array_equal(a.obs["doublet_score"].to_numpy()[idx], s, equal_nan=True)


def test_exact_neighbours_up_to_the_cap_per_channel(monkeypatch):
    seen = []

    def fake(counts, expected_doublet_rate, use_approx_neighbors=None, owned=False):
        seen.append((counts.shape[0], use_approx_neighbors))
        return np.zeros(counts.shape[0]), np.zeros(counts.shape[0], dtype=bool), {"threshold": 0.5}

    monkeypatch.setattr(scrublet_lean, "scrublet_counts", fake)
    monkeypatch.setattr(doublets, "EXACT_KNN_MAX_CELLS", 150)
    a = _obs([f"c{i}" for i in range(320)], gem_well=["a"] * 120 + ["b"] * 200)
    found = doublets.run_scrublet(a, sp.csr_matrix((320, 5), dtype=np.float32))
    assert sorted(seen) == [(120, False), (200, True)]
    assert [c["neighbors"] for c in found["channels"]] == ["exact", "approximate"]
    assert found["neighbors"] == "mixed"


def test_a_channel_scrublet_fails_in_is_left_unscored(monkeypatch):
    def fake(counts, expected_doublet_rate, use_approx_neighbors=None, owned=False):
        if counts.shape[0] == 130:
            raise ValueError("n_components=30 must be strictly less than min(n_samples, n_features)")
        return np.full(counts.shape[0], 0.1), np.ones(counts.shape[0], dtype=bool), {"threshold": 0.05}

    monkeypatch.setattr(scrublet_lean, "scrublet_counts", fake)
    a = _obs([f"c{i}" for i in range(330)], gem_well=["a"] * 130 + ["b"] * 200)
    found = doublets.run_scrublet(a, sp.csr_matrix((330, 5), dtype=np.float32))
    assert [c["status"] for c in found["channels"]] == ["failed", "scored"]
    assert np.isnan(a.obs["doublet_score"].to_numpy()[:130]).all()
    assert int(a.obs["predicted_doublet"].sum()) == 200


@pytest.mark.parametrize("kind", ["csr", "dense"])
def test_rows_read_from_a_backed_file_score_the_same(tmp_path, kind):
    a = _dataset([150, 200])
    if kind == "csr":
        a.X = sp.csr_matrix(a.X)
    p = tmp_path / "b.h5ad"
    a.write_h5ad(p)
    codes = a.obs["gem_well"].cat.codes.to_numpy().astype(np.int64)
    rates, approx = [0.01, 0.02], [False, False]
    want = scrublet_lean.scrublet_by_channel(a.X, codes, rates, approx)
    b = ad.read_h5ad(p, backed="r")
    got = scrublet_lean.scrublet_by_channel(b.X, codes, rates, approx)
    b.file.close()
    assert np.array_equal(got[0], want[0], equal_nan=True) and np.array_equal(got[1], want[1])
    assert [r["threshold"] for r in got[2]] == [r["threshold"] for r in want[2]]


def test_barcode_channels_are_written_found_by_name_next_time_and_saved(tmp_path):
    a = _dataset([150, 210], column=None, names="suffix")
    found = doublets.run_scrublet(a, a.X)
    assert (found["channel_column"], found["channel_reason"]) == ("capture_channel", "barcode_suffix")
    col = a.obs["capture_channel"]
    assert isinstance(col.dtype, pd.CategoricalDtype) and list(col.cat.categories) == ["1", "2"]
    assert [str(n).rsplit("-", 1)[1] for n in a.obs_names] == col.astype(str).tolist()
    assert doublets.channel_grouping(a) == ("capture_channel", "channel_column")  # the column now
    p = tmp_path / "c.h5ad"
    a.write_h5ad(p)  # bool calls and NaN scores save (an object column of NaN calls did not)
    b = ad.read_h5ad(p)
    rec = doublets.channel_record(b)
    assert rec["column"] == "capture_channel" and rec["reason"] == "barcode_suffix"
    assert [(r["name"], r["n_cells"], r["status"]) for r in rec["rows"]] == [("1", 150, "scored"), ("2", 210, "scored")]
    assert doublets.scrublet_threshold(b) is None  # one per channel, none overall


def test_more_channels_than_a_violin_shows(monkeypatch):
    def fake(counts, expected_doublet_rate, use_approx_neighbors=None, owned=False):
        n = counts.shape[0]
        return np.linspace(0, 1, n), np.linspace(0, 1, n) > 0.9, {"threshold": 0.9}

    monkeypatch.setattr(scrublet_lean, "scrublet_counts", fake)
    codes = np.repeat(np.arange(133), 110)
    a = _obs([f"{b}-{c + 1}" for b, c in zip(_barcodes(codes.size), codes)])
    doublets.run_scrublet(a, sp.csr_matrix((codes.size, 5), dtype=np.float32))
    s = doublets.summary(a)
    assert len(s["channels"]) == 133 > 50 and s["per_channel"] and s["threshold"] is None
    assert s["n_called_now"] == sum(r["n_called"] for r in s["channels"])
    assert [r["name"] for r in s["channels"][:3]] == ["1", "2", "3"]


# --- the cells left unscored, in every view ---------------------------------------------


def _scored_with_a_small_channel() -> ad.AnnData:
    a = _dataset([160, 220, 40])
    doublets.run_scrublet(a, a.X)
    return a


def test_summary_counts_the_unscored_cells_and_the_calls_per_channel():
    a = _scored_with_a_small_channel()
    s = doublets.summary(a)
    scores = a.obs["doublet_score"].to_numpy()
    assert s["n_unscored"] == int(np.isnan(scores).sum()) >= 40 and s["n_unscored_small_channels"] == 40
    assert sum(s["hist"]["counts"]) == s["n_scored"] == a.n_obs - s["n_unscored"]
    assert s["threshold"] is None and s["n_called"] is None  # each channel had its own
    by = {r["name"]: r for r in s["channels"]}
    assert by["w3"]["status"] == "too_small" and by["w3"]["n_scored"] == 0 and by["w3"]["n_called"] == 0
    assert sum(r["n_called"] for r in s["channels"]) == s["n_called_now"] == int(a.obs["predicted_doublet"].sum())
    assert by["w1"]["expected_rate"] == round(doublets.rate_for_cells(160), 4)
    at = doublets.summary(a, threshold=0.1)
    assert at["n_called"] == int((np.nan_to_num(scores, nan=-1) > 0.1).sum())
    assert sum(r["n_called"] for r in at["channels"]) == at["n_called"]
    # one threshold over every channel's scores; the unscored stay uncalled; Scrublet's back
    original = a.obs["predicted_doublet"].to_numpy().copy()
    doublets.apply_threshold(a, 0.1)
    assert not a.obs["predicted_doublet"].to_numpy()[np.isnan(scores)].any()
    doublets.apply_threshold(a, None)
    assert np.array_equal(a.obs["predicted_doublet"].to_numpy(), original)


def test_doublet_source_filtering_and_the_report_keep_and_say_the_unscored():
    a = _scored_with_a_small_channel()
    k = int(np.isnan(a.obs["doublet_score"].to_numpy()).sum())
    used = doublet_source.resolve(a.obs)
    assert used.source == "scrublet" and int(used.unscored.sum()) == k
    info = doublet_source.describe(a.obs)
    assert info["scrublet"]["n_unscored"] == k
    assert f"{k:,} cells not scored by Scrublet" in doublet_source.text(info)
    t = qc_filter.Thresholds.from_params(pl.PipelineParams(qc_threshold_mode="fixed", min_genes=0,
                                                           min_cells=0, drop_doublets=True))
    assert qc_filter.doublet_choices(a.obs, t)["unscored"] == k
    out, found = pl._run_filtering(a, pl.PipelineParams(qc_threshold_mode="fixed", min_genes=0,
                                                        min_cells=0, drop_doublets=True))
    assert out.n_obs == a.n_obs - int(a.obs["predicted_doublet"].sum())
    assert found["doublets_unscored_kept"] == k


def test_the_report_and_the_copilot_say_the_channels_in_one_line(tmp_path):
    from scview.core import assistant, qc_report
    from scview.core.anndata_adaptor import AnnDataAdaptor

    a = _scored_with_a_small_channel()
    p = tmp_path / "r.h5ad"
    a.write_h5ad(p)
    facts = qc_report.build_facts(AnnDataAdaptor(str(p)))
    text = next(f.detail for f in facts.quality if f.ref == "qc:doublets")
    assert "Scrublet ran in 3 channels by gem_well, 40-220 cells each" in text
    assert "1 channel under 100 cells not scored (40 cells)" in text
    rows = [{"name": str(i), "n_cells": 700, "expected_doublet_rate": 0.0056, "n_called": 3,
             "status": "scored"} for i in range(133)]
    line = assistant._param_text("channels", rows)
    assert line == ("Scrublet per capture channel: 133 channels, 700-700 cells each, "
                    "expected 0.6-0.6 %, 399 called")


# --- the pipeline param, provenance, replay ----------------------------------------------


def test_the_channel_choice_is_a_pipeline_param_recorded_with_each_channel():
    assert pl.PipelineParams().doublet_channel_key == ""
    a = _dataset([150, 200])
    a.obs["lane"] = pd.Categorical(["L1"] * a.n_obs)
    out, res = pl.run_pipeline(a, ["doublet_detection"], pl.PipelineParams(doublet_channel_key="lane"))
    assert res.errors == {}
    p = provenance.recipe(out)[-1]["params"]
    assert (p["doublet_channel_key"], p["channel_column"], p["channel_reason"]) == ("lane", "lane", "requested")
    assert [(c["name"], c["n_cells"]) for c in p["channels"]] == [("L1", 350)]
    assert p["expected_doublet_rate"] == doublets.rate_for_cells(350)  # one channel: its rate
    out, res = pl.run_pipeline(a, ["doublet_detection"], pl.PipelineParams(doublet_channel_key="nope"))
    assert "not found" in res.errors["doublet_detection"]


def test_replay_scores_each_channel_as_the_app_did(tmp_path):
    from tests.test_replay_script import _replay
    from scview.core import replay_script

    inp = _dataset([150, 230, 50], column=None, names="suffix")
    out, res = pl.run_pipeline(inp.copy(), ["doublet_detection"], pl.PipelineParams())
    assert res.errors == {}
    assert replay_script.describe(out)[-1]["status"] == "exact"
    rep = _replay(tmp_path, inp, out)
    assert np.array_equal(rep.obs["doublet_score"].to_numpy(), out.obs["doublet_score"].to_numpy(),
                          equal_nan=True)
    assert np.array_equal(rep.obs["predicted_doublet"].to_numpy(), out.obs["predicted_doublet"].to_numpy())
    assert rep.obs["capture_channel"].astype(str).tolist() == out.obs["capture_channel"].astype(str).tolist()


def test_replay_of_a_channel_column_and_of_all_cells(tmp_path):
    from tests.test_replay_script import _replay

    inp = _dataset([150, 200])
    for key, sub in (("", "column"), (doublets.ALL_CELLS, "all")):
        (tmp_path / sub).mkdir()
        out, res = pl.run_pipeline(inp.copy(), ["doublet_detection"],
                                   pl.PipelineParams(doublet_channel_key=key))
        assert res.errors == {}
        rep = _replay(tmp_path / sub, inp, out)
        assert np.array_equal(rep.obs["doublet_score"].to_numpy(), out.obs["doublet_score"].to_numpy(),
                              equal_nan=True), key
        assert np.array_equal(rep.obs["predicted_doublet"].to_numpy(),
                              out.obs["predicted_doublet"].to_numpy()), key
    assert rep.uns["scrublet"]["threshold"] == out.uns["scrublet"]["threshold"]  # one run: Scrublet's own


def test_the_run_manifest_carries_each_channel_and_says_the_replay_is_exact():
    from scview.core import run_manifest

    a = _dataset([150, 200])
    out, _ = pl.run_pipeline(a, ["doublet_detection"], pl.PipelineParams())
    step = run_manifest.build_manifest(out, dataset_id="d", dataset_name="d")["steps"][-1]
    assert step["step"] == "doublet_detection" and step["replay"]["status"] == "exact"
    assert step["tool"] == "scanpy.pp.scrublet (per capture channel)"
    assert [(c["name"], c["n_cells"], c["expected_doublet_rate"]) for c in step["params"]["channels"]] == [
        ("w1", 150, doublets.rate_for_cells(150)), ("w2", 200, doublets.rate_for_cells(200))]
    assert (step["params"]["channel_column"], step["params"]["channel_reason"]) == ("gem_well", "channel_column")
