"""Upstream TF candidates from promoter motifs (C095): core/tf_motifs.py + /tf-motifs.

Every input here is synthetic (a few PFMs in JASPAR's format, random promoters in
UCSC's upstream FASTA layout, a refGene table), so no test touches the network. The
child-process build runs on a tiny motif set in a couple of seconds.
"""

from __future__ import annotations

import gzip
import itertools
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import stats

from scview.core import provenance, replay_script, results_store
from scview.core import tf_motifs as tm
from scview.core.claims import resolve_tag

API = "/api/v1"

JASPAR_TEXT = """>MA0001.1\tTFA
A  [ 20  0  0  0 20  0  0 20  0  0 ]
C  [  0 20  0  0  0  0 20  0  0 20 ]
G  [  0  0 20  0  0 20  0  0  0  0 ]
T  [  0  0  0 20  0  0  0  0 20  0 ]
>MA0002.1\tTFB::TFC
A  [  1 18  0  0  0 19  2  0 ]
C  [ 17  0  1 19  0  0  0 18 ]
G  [  1  1 18  0 19  0 16  1 ]
T  [  1  1  1  1  1  1  2  1 ]
>MA0003.2\tSHORT(var.2)
A  [ 10  0  0  0 ]
C  [  0 10  0  0 ]
G  [  0  0 10  0 ]
T  [  0  0  0 10 ]
"""
PLANT_A = "ACGTAGCATC"  # the consensus of MA0001.1


def _rc(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# ---------------------------------------------------------------------------
# Parsing


def test_parse_jaspar_reads_ids_names_and_counts():
    motifs = tm.parse_jaspar(JASPAR_TEXT)
    assert [m.id for m in motifs] == ["MA0001.1", "MA0002.1", "MA0003.2"]
    assert [m.name for m in motifs] == ["TFA", "TFB::TFC", "SHORT(var.2)"]
    assert motifs[0].counts.shape == (4, 10) and motifs[1].counts[1, 0] == 17
    assert tm.tf_members("TFB::TFC") == ["TFB", "TFC"]
    assert tm.tf_members("NR2F1(var.2)") == ["NR2F1"] and tm.tf_members("Ahr::Arnt") == ["Ahr", "Arnt"]


@pytest.mark.parametrize("bad", [
    ">M1 X\nA [ 1 2 ]\nC [ 1 2 ]\nG [ 1 2 ]\n",  # no T row
    ">M1 X\nA [ 1 2 ]\nC [ 1 ]\nG [ 1 2 ]\nT [ 1 2 ]\n",  # rows of different lengths
    "A [ 1 2 ]\n",  # no header
    ">M1 X\nA [ 1 x ]\nC [ 1 2 ]\nG [ 1 2 ]\nT [ 1 2 ]\n",
    "",
])
def test_parse_jaspar_refuses_other_text(bad):
    with pytest.raises(ValueError):
        tm.parse_jaspar(bad)


# ---------------------------------------------------------------------------
# Log-odds and the exact threshold


def test_log_odds_uses_the_background_and_pseudocount():
    counts = np.array([[10.0], [0], [0], [0]])
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    w = tm.log_odds(counts, bg, pseudocount=0.8)
    p = (counts[:, 0] + 0.8 * bg) / 10.8
    np.testing.assert_allclose(w[:, 0], np.log2(p / bg))


def test_threshold_matches_brute_force_on_a_short_motif():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 12, size=(4, 5)).astype(float)
    bg = np.array([0.29, 0.21, 0.21, 0.29])
    wint = tm.integer_matrix(tm.log_odds(counts, bg))
    scores, probs = [], []
    for word in itertools.product(range(4), repeat=5):
        scores.append(sum(wint[b, j] for j, b in enumerate(word)))
        probs.append(np.prod([bg[b] for b in word]))
    scores, probs = np.array(scores), np.array(probs)
    lo, dist = tm.score_distribution(wint, bg)
    for s in np.unique(scores):
        assert dist[s - lo] == pytest.approx(probs[scores == s].sum(), rel=1e-9)
    for p in (1e-2, 1e-3, 2e-3):
        t = tm.threshold(wint, bg, p)
        sf = lambda x: probs[scores >= x].sum()  # noqa: E731
        assert sf(t) < p and sf(t - 1) >= p  # the lowest score with P(score >= t) < p


def test_a_motif_that_cannot_reach_p_has_no_threshold():
    counts = np.full((4, 4), 5.0)  # flat: every word equally likely
    assert tm.threshold(tm.integer_matrix(tm.log_odds(counts, np.full(4, 0.25))), np.full(4, 0.25)) is None
    short = tm.parse_jaspar(JASPAR_TEXT)[2]  # 4 bp: its best word has p ~ 0.004
    bg = np.full(4, 0.25)
    assert tm.threshold(tm.integer_matrix(tm.log_odds(short.counts, bg)), bg) is None


def test_gc_flag():
    gc = tm.parse_jaspar(">M X\nA [ 0 0 0 ]\nC [ 9 0 9 ]\nG [ 0 9 0 ]\nT [ 0 0 0 ]\n")[0]
    at = tm.parse_jaspar(">M X\nA [ 9 0 9 ]\nC [ 0 0 0 ]\nG [ 0 0 0 ]\nT [ 0 9 0 ]\n")[0]
    assert tm.gc_content(gc.counts)[0] > tm.GC_RICH and tm.gc_content(gc.counts)[1] == 1.0
    assert tm.gc_content(at.counts)[0] < 0.1


# ---------------------------------------------------------------------------
# The scan


def _random_promoters(n: int, length: int = 200, seed: int = 1) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=length)) for _ in range(n)]


def _brute(seq: str, wint: np.ndarray, t: int) -> bool:
    codes = tm.encode(seq)
    L = wint.shape[1]
    rc = wint[::-1, ::-1]
    for i in range(len(codes) - L + 1):
        win = codes[i:i + L]
        if (win == 4).any():
            continue
        for w in (wint, rc):
            if w[win, np.arange(L)].sum() >= t:
                return True
    return False


def test_scan_finds_planted_motifs_on_both_strands_and_matches_brute_force():
    motifs = tm.parse_jaspar(JASPAR_TEXT)[:2]
    seqs = _random_promoters(40)
    planted = {}
    for i in range(0, 40, 4):  # every 4th: forward; every 4th + 1: reverse strand
        s = seqs[i]
        seqs[i] = s[:50] + PLANT_A + s[60:]
        planted[i] = "+"
        s = seqs[i + 1]
        seqs[i + 1] = s[:120] + _rc(PLANT_A) + s[130:]
        planted[i + 1] = "-"
    codes = np.stack([tm.encode(s) for s in seqs])
    bg = tm.background(codes)
    wints = [tm.integer_matrix(tm.log_odds(m.counts, bg)) for m in motifs]
    thr = np.array([tm.threshold(w, bg) for w in wints])
    hits = tm.scan(codes, wints, thr)
    assert hits.shape == (40, 2)
    assert all(hits[i, 0] for i in planted)
    for i, s in enumerate(seqs):  # exactly the brute-force answer, both motifs
        for j in range(2):
            assert hits[i, j] == _brute(s, wints[j], thr[j]), (i, j)

    # The same promoters shuffled: the planted words are gone.
    rng = np.random.default_rng(5)
    shuffled = ["".join(rng.permutation(list(seqs[i]))) for i in planted]
    sh = tm.scan(np.stack([tm.encode(s) for s in shuffled]), wints, thr)
    assert sh[:, 0].mean() < 0.3


def test_a_window_with_an_unknown_base_is_never_a_hit():
    motif = tm.parse_jaspar(JASPAR_TEXT)[0]
    bg = np.full(4, 0.25)
    w = tm.integer_matrix(tm.log_odds(motif.counts, bg))
    t = tm.threshold(w, bg)
    clean = "T" * 30 + PLANT_A + "T" * 30
    masked = "T" * 30 + PLANT_A[:4] + "N" + PLANT_A[5:] + "T" * 30
    hits = tm.scan(np.stack([tm.encode(clean), tm.encode(masked)]), [w], np.array([t]))
    assert hits[:, 0].tolist() == [True, False]


def test_scan_blocks_give_the_same_answer(monkeypatch):
    motifs = tm.parse_jaspar(JASPAR_TEXT)[:2]
    codes = np.stack([tm.encode(s) for s in _random_promoters(30, seed=9)])
    bg = tm.background(codes)
    wints = [tm.integer_matrix(tm.log_odds(m.counts, bg)) for m in motifs]
    thr = np.array([tm.threshold(w, bg) for w in wints])
    whole = tm.scan(codes, wints, thr)
    monkeypatch.setattr(tm, "_BLOCK_FLOATS", 1)  # one promoter per block
    seen = []
    assert np.array_equal(tm.scan(codes, wints, thr, lambda d, n: seen.append(d)), whole)
    assert seen[-1] == 30 and len(seen) == 30


# ---------------------------------------------------------------------------
# Promoters and the table


def test_one_promoter_per_symbol():
    records = [
        ("NM_000200", "chr1_KI270706v1_random", "A" * 10),  # alt contig: last resort
        ("NM_000300", "chr2", "C" * 10),
        ("NR_000100", "chr2", "G" * 10),  # non-coding: after NM_
        ("NM_000250", "chr3", "T" * 10),  # lowest NM accession on a primary chromosome wins
        ("NM_009999", "chrX", "ACGTNNNNNN"),  # GENE2: more unknown bases than its rival
        ("NM_010000", "chrX", "ACGTACGTAA"),
        ("XM_1", "chr1", "A" * 10),  # no symbol in refGene: left out
    ]
    symbols = {"NM_000200": "GENE1", "NM_000300": "GENE1", "NR_000100": "GENE1", "NM_000250": "GENE1",
               "NM_009999": "GENE2", "NM_010000": "GENE2"}
    names, accs, codes = tm.choose_promoters(records, symbols, length=12)
    assert names == ["GENE1", "GENE2"]
    assert accs == ["NM_000250", "NM_010000"]
    assert codes.shape == (2, 12) and codes[0, :2].tolist() == [4, 4]  # padded at the distal end
    assert codes[0, 2:].tolist() == [3] * 10


def test_parse_upstream_and_refgene(tmp_path):
    fa = tmp_path / "up.fa.gz"
    with gzip.open(fa, "wt") as fh:
        fh.write(">NM_000299_up_1000_chr1_201282452_f chr1:201282452-201283451\nacgt\nACGT\n"
                 ">NR_1_up_1000_chr1_KI270706v1_random_5_r chr1_KI270706v1_random:5-1004\nTTTT\n")
    assert list(tm.parse_upstream(fa)) == [("NM_000299", "chr1", "acgtACGT"),
                                          ("NR_1", "chr1_KI270706v1_random", "TTTT")]
    rg = tmp_path / "refGene.txt.gz"
    with gzip.open(rg, "wt") as fh:
        fh.write("585\tNM_000299\tchr1\t+\t1\t2\t1\t2\t1\t1,\t2,\t0\tPKP1\tcmpl\tcmpl\t0,\n")
    assert tm.parse_refgene(rg) == {"NM_000299": "PKP1"}


def test_bit_packed_table_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    table = _table(rng.random((50, 13)) < 0.3)
    path = tmp_path / "t.npz"
    tm.save_table(table, path)
    back = tm.load_table_file(path)
    assert np.array_equal(back.hits, table.hits) and back.hits.dtype == bool
    assert back.genes.tolist() == table.genes.tolist() and back.motif_names.tolist() == table.motif_names.tolist()
    assert back.meta == table.meta and back.assembly == "hg38"


def test_cache_key_names_the_inputs_and_parameters(monkeypatch):
    k = tm.cache_key("hg38")
    assert k != tm.cache_key("mm10")
    monkeypatch.setattr(tm, "P_THRESHOLD", 1e-5)
    assert tm.cache_key("hg38") != k


def test_child_process_builds_a_tiny_table(tmp_path):
    files = _tiny_inputs(tmp_path)
    events: list[dict] = []
    t0 = time.time()
    final = tm.run_child(files, tmp_path / "out.npz", "hg38", on_event=events.append)
    assert time.time() - t0 < 10
    assert final["n_motifs"] == 3 and final["n_unreachable"] == 1 and final["peak_mb"] > 0
    assert {e["stage"] for e in events} >= {"read", "scan"}
    table = tm.load_table_file(tmp_path / "out.npz")
    assert table.genes.tolist() == sorted(f"GN{i}" for i in range(24))
    assert table.meta["threads"] == "2"
    planted = {f"GN{i}" for i in range(0, 24, 2)}
    col = table.motif_ids.tolist().index("MA0001.1")
    assert {g for g, h in zip(table.genes, table.hits[:, col]) if h} >= planted
    assert not table.hits[:, 2].any() and table.thresholds[2] == -1  # SHORT cannot reach 1e-4


def test_child_process_failure_is_reported(tmp_path):
    files = _tiny_inputs(tmp_path)
    files["jaspar"].write_text("not a motif file\n")
    with pytest.raises(RuntimeError, match="motif scan failed"):
        tm.run_child(files, tmp_path / "out.npz", "hg38")


def test_ensure_files_refuses_a_changed_file_and_respects_the_switch(tmp_path):
    with pytest.raises(tm.MotifTableUnavailable, match="TF_MOTIFS_AUTO_DOWNLOAD is off"):
        tm.ensure_files(tmp_path, "hg38", auto_download=False)
    raw = tm.raw_dir(tmp_path)
    raw.mkdir(parents=True)
    (raw / tm.JASPAR.filename).write_text(JASPAR_TEXT)
    with pytest.raises(tm.MotifTableUnavailable, match="sha256 differs"):
        tm.ensure_files(tmp_path, "hg38", auto_download=False)


def test_download_checks_the_pin(tmp_path):
    import httpx

    src = tm.Source("x", "https://example.test/x.txt", "0" * 64, "x.txt", 10)

    def handler(request):
        return httpx.Response(200, content=b"something else")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="not the pinned"):
            tm.download(tmp_path, src, client=client)
    assert not list(tm.raw_dir(tmp_path).iterdir())


def test_assembly_for():
    a = ad.AnnData(np.zeros((2, 2)))
    assert tm.assembly_for(a, "human") == "hg38" and tm.assembly_for(a, "unknown") is None
    assert tm.assembly_for(a, "mouse") == "mm10"
    a.var["genome"] = ["GRCm39", "GRCm39"]
    assert tm.assembly_for(a, "mouse") == "mm39"


# ---------------------------------------------------------------------------
# The test and its columns


def test_hypergeometric_odds_ratio_and_bh_match_scipy():
    rng = np.random.default_rng(7)
    hits = rng.random((300, 6)) < 0.2
    hits[:40, 0] = True  # motif 0 in every marker promoter
    table = _table(hits)
    markers = {"A": [f"GN{i}" for i in range(40)], "B": [f"GN{i}" for i in range(200, 260)]}
    res = tm.enrich(table, [f"GN{i}" for i in range(300)], markers)
    assert res["N"] == 300
    a = res["groups"][0]
    top = a["rows"][0]
    assert top["motif_id"] == "M0" and top["k"] == 40 and top["n"] == 40 and top["K"] == int(hits[:, 0].sum())
    want = stats.hypergeom.sf(top["k"] - 1, 300, top["K"], 40)
    assert top["p"] == pytest.approx(want, rel=1e-9)
    assert top["p"] == pytest.approx(stats.fisher_exact(
        [[40, 0], [top["K"] - 40, 300 - 40 - top["K"] + 40]], alternative="greater")[1], rel=1e-6)
    # BH across every motif with K > 0 in the group
    ks = hits[:40].sum(axis=0)
    Ks = hits.sum(axis=0)
    ps = stats.hypergeom.sf(ks - 1, 300, Ks, 40)
    padj = stats.false_discovery_control(ps, method="bh")
    by_id = {r["motif_id"]: r for r in a["rows"]}
    for j in range(6):
        if ks[j] >= 3:
            assert by_id[f"M{j}"]["padj"] == pytest.approx(padj[j], rel=1e-9)
    assert all(r["k"] >= 3 for g in res["groups"] for r in g["rows"])
    assert [r["p"] for r in a["rows"]] == sorted(r["p"] for r in a["rows"])
    # odds ratio, Haldane-corrected here (no marker lacks motif 0)
    b, c, d = 0.5, top["K"] - 40 + 0.5, 300 - 40 - top["K"] + 40 + 0.5
    assert top["odds_ratio"] == pytest.approx(round(40.5 * d / (b * c), 3))


def test_bh_against_a_hand_computation():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    want = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = want
    np.testing.assert_allclose(tm.bh(p), out)


def test_a_group_with_too_few_markers_is_said():
    res = tm.enrich(_table(np.ones((10, 2), bool)), [f"GN{i}" for i in range(10)], {"A": ["GN1", "NOPE"]})
    g = res["groups"][0]
    assert g["rows"] == [] and "at least 3" in g["note"]


def test_universe_matches_case_and_counts_one_promoter_once():
    table = _table(np.ones((5, 1), bool))
    res = tm.enrich(table, ["gn0", "GN1", "Gn2", "GN3", "GN4", "extra"], {"A": ["gn0", "GN1", "Gn2"]})
    assert res["N"] == 5 and res["groups"][0]["n"] == 3


def test_tf_expressed_names_the_expressed_member_of_a_dimer():
    stats_ = {"TFB": {"gene": "TFB", "pct": {"A": 0.02}, "logfc": {"A": -0.4}},
              "TFC": {"gene": "TFC", "pct": {"A": 0.6}, "logfc": {}}}
    got = tm.tf_expressed(["TFB", "TFC"], "A", stats_)
    assert got == {"gene": "TFC", "pct": 0.6, "logfc": None, "expressed": True}
    assert tm.tf_expressed(["TFB"], "A", stats_)["expressed"] is False
    assert tm.tf_expressed(["NOPE"], "A", stats_) is None


def test_activity_agrees_is_null_without_activity():
    assert tm.activity_agrees(["TFA"], "A", None) is None
    groups, sources = ["A", "B"], [f"T{i}" for i in range(20)] + ["TFA"]
    means = np.zeros((2, 21))
    means[0, -1], means[1, -1] = 3.0, -3.0
    act = tm.activity_ranks(groups, sources, means, np.array([10, 10]))
    assert tm.activity_agrees(["TFA"], "A", act) is True
    assert tm.activity_agrees(["TFA"], "B", act) is False
    assert tm.activity_agrees(["NOTSCORED"], "A", act) is None
    assert tm.activity_agrees(["NOTSCORED", "TFA"], "A", act) is True


# ---------------------------------------------------------------------------
# Helpers


def _table(hits: np.ndarray) -> tm.MotifTable:
    n, m = hits.shape
    return tm.MotifTable(
        assembly="hg38", genes=np.array([f"GN{i}" for i in range(n)]), accessions=np.array([f"NM_{i}" for i in range(n)]),
        motif_ids=np.array([f"M{j}" for j in range(m)]), motif_names=np.array([f"TF{j}" for j in range(m)]),
        lengths=np.full(m, 10), thresholds=np.full(m, 900), gc=np.linspace(0.3, 0.8, m), hits=hits,
        background=np.array([0.3, 0.2, 0.2, 0.3]), meta={"n_genes": n, "n_motifs": m, "hit_rate": 0.2})


def _tiny_inputs(tmp_path: Path) -> dict[str, Path]:
    seqs = _random_promoters(24, length=300, seed=11)
    for i in range(0, 24, 2):
        seqs[i] = seqs[i][:100] + (PLANT_A if i % 4 == 0 else _rc(PLANT_A)) + seqs[i][110:]
    up = tmp_path / "upstream.fa.gz"
    with gzip.open(up, "wt") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">NM_{i:06d}_up_1000_chr1_{1000 + i}_f chr1:{1000 + i}-{2000 + i}\n{s}\n")
    rg = tmp_path / "refGene.txt.gz"
    with gzip.open(rg, "wt") as fh:
        for i in range(24):
            fh.write(f"0\tNM_{i:06d}\tchr1\t+\t1\t2\t1\t2\t1\t1,\t2,\t0\tGN{i}\tcmpl\tcmpl\t0,\n")
    jp = tmp_path / "jaspar.txt"
    jp.write_text(JASPAR_TEXT)
    return {"jaspar": jp, "upstream": up, "refgene": rg}


# ---------------------------------------------------------------------------
# API

N_GENES = 120


def _dataset() -> ad.AnnData:
    """Human-looking genes; group A over-expresses GN0-GN29 (planted motif), B GN60-GN89."""
    rng = np.random.default_rng(0)
    names = [f"GN{i}" for i in range(N_GENES - 3)] + ["TFA", "TFB", "TFC"]
    n_cells = 80
    X = rng.poisson(0.4, size=(n_cells, N_GENES)).astype(np.float32)
    group = np.array(["A"] * 40 + ["B"] * 40)
    X[:40, :30] += rng.poisson(6, size=(40, 30))
    X[40:, 60:90] += rng.poisson(6, size=(40, 30))
    X[:40, N_GENES - 3] += 5  # TFA expressed in A
    X[:40, N_GENES - 1] += 5  # TFC in A
    a = ad.AnnData(X, var=pd.DataFrame(index=names), obs=pd.DataFrame({"cluster": pd.Categorical(group)},
                                                                       index=[f"c{i}" for i in range(n_cells)]))
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", pts=True)
    return a


def _dataset_table() -> tm.MotifTable:
    names = [f"GN{i}" for i in range(N_GENES - 3)] + ["TFA", "TFB", "TFC"]
    rng = np.random.default_rng(4)
    hits = rng.random((len(names), 3)) < 0.15
    hits[:30, 0] = True  # MA0001.1 (TFA) in every group-A marker promoter
    hits[:30, 1] = True  # MA0002.1 (TFB::TFC) too
    t = tm.MotifTable(
        assembly="hg38", genes=np.array(names), accessions=np.array([f"NM_{i}" for i in range(len(names))]),
        motif_ids=np.array(["MA0001.1", "MA0002.1", "MA0099.1"]), motif_names=np.array(["TFA", "TFB::TFC", "GCX"]),
        lengths=np.array([10, 8, 9]), thresholds=np.array([900, 800, 700]), gc=np.array([0.4, 0.5, 0.8]),
        hits=hits, background=np.array([0.3, 0.2, 0.2, 0.3]), meta={"n_genes": len(names), "n_motifs": 3,
                                                                     "hit_rate": 0.2})
    return t


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _species_human(monkeypatch):
    from scview.api.v1 import tf_motifs as api
    from scview.core.species import SpeciesCall

    monkeypatch.setattr(api, "detect_species", lambda adata: SpeciesCall(species="human", method="symbol_case"))


async def test_api_not_prepared_then_run_and_get(client, dataset_manager, test_settings, monkeypatch):
    _species_human(monkeypatch)
    did = _register(dataset_manager, _dataset(), "tf1")
    st = (await client.get(f"{API}/tf-motifs/status", params={"dataset_id": did})).json()
    assert st["dataset"] == {"species": "human", "assembly": "hg38", "prepared": False}
    assert {a["assembly"] for a in st["assemblies"]} == {"hg38", "mm10", "mm39"}
    assert "JASPAR 2024" in st["citation"] and "candidate" in st["caveats"]
    assert (await client.get(f"{API}/datasets/{did}/tf-motifs")).status_code == 404
    got = await client.get(f"{API}/datasets/{did}/tf-motifs", params={"allow_empty": True})
    assert got.status_code == 200 and got.json() is None

    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})
    assert r.status_code == 409 and "Prepare" in r.json()["detail"]

    dest = Path(test_settings.DATA_DIR) / "tf_motifs"
    tm.save_table(_dataset_table(), tm.table_path(dest, "hg38"))
    st = (await client.get(f"{API}/tf-motifs/status", params={"dataset_id": did})).json()
    assert st["dataset"]["prepared"] is True

    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster", "n_markers": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "cluster" and body["assembly"] == "hg38" and body["N"] == N_GENES
    a = next(g for g in body["groups"] if g["group"] == "A")
    assert a["n_cells"] == 40
    top = {row["motif_id"]: row for row in a["rows"]}
    assert top["MA0001.1"]["padj"] < 0.05 and top["MA0001.1"]["tfs"] == ["TFA"]
    assert top["MA0001.1"]["tf_expressed"]["gene"] == "TFA" and top["MA0001.1"]["tf_expressed"]["expressed"]
    assert top["MA0002.1"]["tf_expressed"]["gene"] == "TFC"  # the expressed member of the dimer
    assert top["MA0001.1"]["activity_agrees"] is None
    assert body["activity"]["available"] is False and "CollecTRI" in body["activity"]["hint"]
    assert all(row["gc_rich"] == (row["motif_id"] == "MA0099.1") for g in body["groups"] for row in g["rows"])

    # Recorded (History) as a step that only adds a result; the last run comes back
    results_store.flush()
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    step = provenance.read_provenance(adaptor.adata)["history"][-1]
    assert step["step"] == tm.STEP and step["params"]["column"] == "cluster"
    assert tm.STEP in results_store.ADDITIVE_STEPS
    got = await client.get(f"{API}/datasets/{did}/tf-motifs")
    assert got.status_code == 200 and got.json()["result_id"] == body["result_id"]

    # The co-pilot fact: the highlighted group's candidates, said as candidates
    lines, detail = tm.grounding_lines(adaptor, {"column": "cluster", "value": "A"})
    text = "\n".join(lines)
    assert "candidates" in text and "TFA (MA0001.1)" in text and f"[{tm.TAG}]" in text
    assert "candidates, B" not in text
    assert resolve_tag("tfmotifs", [tm.TAG, "result:activity"]) == [tm.TAG]
    assert resolve_tag("result:tfmotifs", [tm.TAG]) == [tm.TAG]

    # The replay script says it is a result only
    script = replay_script.build_script(adaptor.adata)
    compile(script, "replay.py", "exec")
    assert replay_script.describe(adaptor.adata)[-1]["status"] == "not_replayable"


async def test_api_422s(client, dataset_manager, test_settings, monkeypatch):
    _species_human(monkeypatch)
    dest = Path(test_settings.DATA_DIR) / "tf_motifs"
    tm.save_table(_dataset_table(), tm.table_path(dest, "hg38"))
    a = _dataset()
    del a.uns["rank_genes_groups"]
    did = _register(dataset_manager, a, "tf2")
    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})
    assert r.status_code == 422 and "marker genes" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "nope"})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster", "n_markers": 3})
    assert r.status_code == 422

    from scview.api.v1 import tf_motifs as api
    from scview.core.species import SpeciesCall

    monkeypatch.setattr(api, "detect_species", lambda adata: SpeciesCall())
    r = await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})
    assert r.status_code == 422 and "species" in r.json()["detail"]


async def test_api_activity_agrees_from_collectri(client, dataset_manager, test_settings, monkeypatch):
    """With the CollecTRI network on disk the agreement column is filled in."""
    _species_human(monkeypatch)
    from scview.api.v1 import tf_motifs as api

    monkeypatch.setattr(api, "_activity", lambda adata, column, settings, species: (
        tm.activity_ranks(["A", "B"], ["TFA", "TFC"] + [f"X{i}" for i in range(10)],
                          np.array([[2.0, -1.0] + [0.0] * 10, [-2.0, 1.0] + [0.0] * 10]), np.array([40, 40])),
        None))
    dest = Path(test_settings.DATA_DIR) / "tf_motifs"
    tm.save_table(_dataset_table(), tm.table_path(dest, "hg38"))
    did = _register(dataset_manager, _dataset(), "tf3")
    body = (await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})).json()
    a = next(g for g in body["groups"] if g["group"] == "A")
    rows = {r["motif_id"]: r for r in a["rows"]}
    assert rows["MA0001.1"]["activity_agrees"] is True
    assert rows["MA0002.1"]["activity_agrees"] is False  # TFC is scored and lower in A; TFB is not scored
    assert body["activity"]["available"] is True


async def test_api_prepare_streams_the_result_or_the_error(client, test_settings):
    dest = Path(test_settings.DATA_DIR) / "tf_motifs"
    test_settings.TF_MOTIFS_AUTO_DOWNLOAD = False
    r = await client.post(f"{API}/tf-motifs/prepare", json={"species": "mouse"})
    assert r.status_code == 200
    assert "event: error" in r.text and "TF_MOTIFS_AUTO_DOWNLOAD is off" in r.text
    tm.save_table(_dataset_table(), tm.table_path(dest, "mm10"))
    r = await client.post(f"{API}/tf-motifs/prepare", json={"species": "mouse"})
    assert "event: result" in r.text and "event: complete" in r.text and '"assembly": "mm10"' in r.text
    r = await client.post(f"{API}/tf-motifs/prepare", json={"species": "human", "assembly": "mm10"})
    assert r.status_code == 422
