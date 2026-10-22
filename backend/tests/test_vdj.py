"""TCR / BCR contigs (core/vdj.py): parsing, the barcode join, chains, clonotypes and
the repertoire statistics. Expected values are worked out by hand on a contig table
built to hold one case per cell, or computed with scipy."""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy

from scview.core import provenance, vdj

# 16-base barcodes, one per cell of the fixture.
BC = ["AAACCTGAGAAACCAT", "AAACCTGAGAAACCGC", "AAACCTGAGAAACCTA", "AAACCTGAGAAACGAG",
      "AAACCTGAGAAACTCA", "AAACCTGAGAAAGTGG", "AAACCTGAGAACAACT", "AAACCTGAGAACAATC",
      "AAACCTGAGAACTCGG", "AAACCTGAGAACTGTA", "AAACCTGAGAAGAAGC"]
COLS = ["barcode", "is_cell", "contig_id", "high_confidence", "length", "chain", "v_gene", "d_gene",
        "j_gene", "c_gene", "full_length", "productive", "cdr3", "cdr3_nt", "reads", "umis",
        "raw_clonotype_id", "raw_consensus_id"]


def _row(cell: int, k: int, chain: str, cdr3: str, umis: int, *, v: str = "", productive: str = "true",
         high: str = "true", is_cell: str = "true", reads: int | None = None) -> dict:
    v = v or {"TRA": "TRAV1-2", "TRB": "TRBV6-5", "IGH": "IGHV3-23", "IGK": "IGKV1-8",
              "IGL": "IGLV2-14"}.get(chain, "")
    return {"barcode": f"{BC[cell]}-1", "is_cell": is_cell, "contig_id": f"{BC[cell]}-1_contig_{k}",
            "high_confidence": high, "length": 500, "chain": chain, "v_gene": v, "d_gene": "",
            "j_gene": chain[:3] + "J1", "c_gene": chain[:3] + "C", "full_length": "true",
            "productive": productive, "cdr3": cdr3, "cdr3_nt": "TGT" + cdr3, "reads": reads or umis * 100,
            "umis": umis, "raw_clonotype_id": "", "raw_consensus_id": ""}


def t_rows() -> list[dict]:
    """One case per cell (indices into BC):
    0, 1: TRA CAVA + TRB CASSA, paired (1 also has a low-confidence TRB with more UMIs);
    2: the same pair plus a weaker second TRA (extra VJ: same clonotype, primary only);
    3: TRB CASSA alone (its TRA has no CDR3); 4: TRB CASSA alone (its TRA is non-productive);
    5: three TRA chains (multichain); 6: a TRA (and an IGH in the B file: TCR + BCR);
    7: only a non-productive contig; 8: no contig; 10: not a cell of the dataset."""
    return [
        _row(0, 1, "TRA", "CAVA", 5), _row(0, 2, "TRB", "CASSA", 7),
        _row(0, 2, "TRB", "CASSA", 7),  # a repeated line
        _row(1, 1, "TRA", "CAVA", 3), _row(1, 2, "TRB", "CASSA", 4),
        _row(1, 3, "TRB", "CASSZ", 50, high="false"),
        _row(2, 1, "TRA", "CAVA", 2), _row(2, 2, "TRB", "CASSA", 9), _row(2, 3, "TRA", "CAVX", 1),
        _row(3, 1, "TRB", "CASSA", 6), _row(3, 2, "TRA", "None", 4),
        _row(4, 1, "TRB", "CASSA", 3), _row(4, 2, "TRA", "CAVQ", 8, productive="false"),
        _row(5, 1, "TRA", "CAVB", 4), _row(5, 2, "TRA", "CAVC", 3), _row(5, 3, "TRA", "CAVD", 2),
        _row(5, 4, "TRB", "CASSB", 5),
        _row(6, 1, "TRA", "CAVE", 4),
        _row(7, 1, "TRB", "CASSN", 2, productive="false"),
        _row(10, 1, "TRA", "CAVZ", 3), _row(10, 2, "TRB", "CASSZZ", 3),
    ]


def b_rows() -> list[dict]:
    """6: IGH (with the TRA above: TCR + BCR); 9: IGH + IGK paired, and an IGL contig
    that Cell Ranger did not call a cell."""
    return [
        _row(6, 2, "IGH", "CARE", 6),
        _row(9, 1, "IGH", "CARX", 20), _row(9, 2, "IGK", "CQQX", 15),
        _row(9, 3, "IGL", "CQSY", 30, is_cell="false"),
    ]


def write_csv(path: Path, rows: list[dict], *, gz: bool = False) -> Path:
    df = pd.DataFrame(rows, columns=COLS)
    if gz:
        with gzip.open(path, "wt") as fh:
            df.to_csv(fh, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def cells_adata(names: list[str] | None = None, extra_obs: dict | None = None) -> ad.AnnData:
    names = names or [f"{b}-1" for b in BC[:10]]
    a = ad.AnnData(np.ones((len(names), 3), dtype=np.float32))
    a.obs_names = names
    a.var_names = ["G1", "G2", "G3"]
    for k, v in (extra_obs or {}).items():
        a.obs[k] = pd.Categorical(v)
    return a


@pytest.fixture()
def tb(tmp_path: Path) -> tuple[Path, Path, Path]:
    t = write_csv(tmp_path / "t_filtered_contig_annotations.csv", t_rows())
    b = write_csv(tmp_path / "b_filtered_contig_annotations.csv", b_rows())
    both = write_csv(tmp_path / "all.csv", t_rows() + b_rows())
    return t, b, both


# ---------------------------------------------------------------------------
# Parsing and filters
# ---------------------------------------------------------------------------


def test_reads_10x_and_drops_each_failing_contig_once(tb):
    df, fmt = vdj.read_contigs(tb[2])
    assert fmt == "10x" and len(df) == len(t_rows()) + len(b_rows())
    contigs, kept, dropped = vdj.keep_chains(df)
    assert dropped == {"duplicate": 1, "no_barcode": 0, "not_cell": 1, "low_confidence": 1,
                       "non_productive": 2, "unknown_locus": 0, "no_cdr3": 1}
    assert len(contigs) == len(df) - 1
    assert len(kept) == len(contigs) - 5
    assert set(kept["locus"]) == {"TRA", "TRB", "IGH", "IGK"}


def test_gzipped_table_reads_the_same(tmp_path, tb):
    gz = write_csv(tmp_path / "t.csv.gz", t_rows(), gz=True)
    a, _ = vdj.read_contigs(gz)
    b, _ = vdj.read_contigs(tb[0])
    pd.testing.assert_frame_equal(a, b)


def test_not_a_contig_table_is_refused(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("gene,cell1,cell2\nCD3E,1,2\n")
    with pytest.raises(vdj.VdjError, match="not a V\\(D\\)J contig table"):
        vdj.read_contigs(p)


def _airr(rows: list[dict], *, tenx_export: bool = False) -> pd.DataFrame:
    out = []
    for r in rows:
        rec = {"cell_id": r["barcode"], "sequence_id": r["contig_id"], "junction_aa": r["cdr3"],
               "junction": r["cdr3_nt"], "v_call": f"{r['v_gene']}*01,{r['v_gene']}*02",
               "d_call": "", "j_call": f"{r['j_gene']}*01", "c_call": r["c_gene"],
               "productive": "T" if r["productive"] == "true" else "F", "consensus_count": r["reads"]}
        if tenx_export:  # 10x's AIRR file: no locus, UMIs in duplicate_count
            rec["duplicate_count"] = r["umis"]
            rec["is_cell"] = "T" if r["is_cell"] == "true" else "F"
        else:
            rec["locus"] = r["chain"]
            rec["umi_count"] = r["umis"]
        out.append(rec)
    return pd.DataFrame(out)


@pytest.mark.parametrize("tenx_export", [False, True])
def test_airr_rearrangement_reads_like_the_10x_table(tmp_path, tenx_export):
    rows = [r for r in t_rows() if r["high_confidence"] == "true"]
    p = tmp_path / "airr.tsv"
    _airr(rows, tenx_export=tenx_export).to_csv(p, sep="\t", index=False)
    airr, fmt = vdj.read_contigs(p)
    tenx, _ = vdj.read_contigs(write_csv(tmp_path / "t.csv", rows))
    assert fmt == "airr"
    for col in ("barcode", "contig_id", "locus", "cdr3", "cdr3_nt", "v", "j", "umis", "reads", "productive"):
        assert airr[col].tolist() == tenx[col].tolist(), col
    assert airr["v"].iloc[0] == "TRAV1-2"  # first call, allele dropped


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------


def test_barcode_core_finds_the_sequence_in_decorated_names():
    seq = BC[0]
    assert vdj.barcode_core(f"{seq}-1") == (seq, "1")
    assert vdj.barcode_core(f"{seq}-1:sampleA") == (seq, "1")
    assert vdj.barcode_core(f"sampleA_{seq}-1") == (seq, "1")
    assert vdj.barcode_core(f"{seq}-1-0") == (seq, "1")
    assert vdj.barcode_core(f"{seq}") == (seq, None)
    assert vdj.barcode_core("cell_12") == (None, None)
    assert list(vdj.library_tags([f"{seq}-1:s1", f"{BC[1]}-1:s2", f"{BC[2]}-2"])) == ["*-1:s1", "*-1:s2", "*-2"]


def test_match_counts_both_ways_and_ambiguous_cores():
    names = [f"{BC[0]}-1:s1", f"{BC[0]}-1:s2", f"{BC[1]}-1:s1", f"{BC[2]}-1"]
    contigs = [f"{BC[0]}-1", f"{BC[1]}-1", f"{BC[2]}-1", f"{BC[3]}-1"]
    cells, exact = vdj.match_barcodes(contigs, names)
    assert cells.tolist() == [-2, 2, 3, -1] and exact.tolist() == [False, False, True, False]
    s2 = np.array([False, True, False, False])
    cells, _ = vdj.match_barcodes(contigs, names, s2)
    assert cells.tolist() == [1, -1, -1, -1]


def test_gem_suffix_decides_between_two_libraries():
    names = [f"{BC[0]}-1", f"{BC[0]}-2"]
    cells, _ = vdj.match_barcodes([f"{BC[0]}-2", f"{BC[0]}-3"], names)
    # -2 matches the -2 cell exactly; -3 has no such suffix and its core is in both
    assert cells.tolist() == [1, -2]


# ---------------------------------------------------------------------------
# Chains, status, clonotypes
# ---------------------------------------------------------------------------

STATUS = ["paired", "paired", "paired + extra chain", "single chain", "single chain", "multichain",
          "ambiguous", "no usable chain", "no contig", "paired"]


def _attached(tb, **kw) -> ad.AnnData:
    a = cells_adata()
    stats = vdj.attach_file(a, tb[2], **kw)
    assert stats["matched"] == 9
    return a


def test_status_clonotypes_and_sizes_by_hand(tb):
    a = _attached(tb)
    assert a.obs[vdj.STATUS_COL].astype(str).tolist() == STATUS
    # Legend order follows the status order, unused ones dropped.
    assert list(a.obs[vdj.STATUS_COL].cat.categories) == [s for s in vdj.STATUS_ORDER if s in STATUS]
    clon = a.obs[vdj.CLONOTYPE_COL].astype(object).where(a.obs[vdj.CLONOTYPE_COL].notna(), None).tolist()
    # Cells 0-2 share TRA CAVA + TRB CASSA (cell 2 by its primary TRA); 3-4 share the lone TRB.
    assert clon == ["TCR1", "TCR1", "TCR1", "TCR2", "TCR2", None, None, None, None, "BCR1"]
    size = a.obs[vdj.SIZE_COL].to_numpy()
    assert size[:5].tolist() == [3, 3, 3, 2, 2] and size[9] == 1 and np.isnan(size[5:9]).all()
    assert a.obs[vdj.EXPANSION_COL].astype(str).tolist() == (
        ["2-5 cells"] * 5 + ["no clonotype"] * 4 + ["1 cell"])
    assert list(a.obs[vdj.EXPANSION_COL].cat.categories) == ["1 cell", "2-5 cells", "no clonotype"]
    assert a.obs["tra_cdr3"].astype(object).tolist()[:3] == ["CAVA", "CAVA", "CAVA"]
    assert pd.isna(a.obs["tra_cdr3"].iloc[3]) and a.obs["trb_cdr3"].iloc[3] == "CASSA"
    assert a.obs["igh_cdr3"].iloc[9] == "CARX" and a.obs["igk_cdr3"].iloc[9] == "CQQX"
    assert "igl_cdr3" not in a.obs.columns  # the only IGL contig was not a cell
    assert a.obs[vdj.RECEPTOR_COL].astype(str).tolist() == (
        ["TCR alpha-beta"] * 6 + ["TCR + BCR", "none", "none", "BCR"])


def test_primary_chain_is_the_one_with_most_umis(tb):
    a = _attached(tb)
    t = vdj.read_table(a)
    assert t.loc[f"{BC[2]}-1", "VJ_1_cdr3"] == "CAVA" and t.loc[f"{BC[2]}-1", "VJ_2_cdr3"] == "CAVX"
    # Multichain: three TRA, the top two kept, the count says three.
    assert t.loc[f"{BC[5]}-1", "n_vj"] == 3
    assert [t.loc[f"{BC[5]}-1", f"VJ_{i}_cdr3"] for i in (1, 2)] == ["CAVB", "CAVC"]
    # The low-confidence TRB (50 UMIs) did not become cell 1's primary.
    assert t.loc[f"{BC[1]}-1", "VDJ_1_cdr3"] == "CASSA" and t.loc[f"{BC[1]}-1", "n_vdj"] == 1


def test_same_v_gene_splits_a_clonotype(tmp_path):
    rows = [_row(0, 1, "TRA", "CAVA", 5), _row(0, 2, "TRB", "CASSA", 5),
            _row(1, 1, "TRA", "CAVA", 5), _row(1, 2, "TRB", "CASSA", 5, v="TRBV7-9")]
    p = write_csv(tmp_path / "v.csv", rows)
    a = cells_adata()
    vdj.attach_file(a, p)
    assert a.obs[vdj.CLONOTYPE_COL].iloc[0] == a.obs[vdj.CLONOTYPE_COL].iloc[1]
    vdj.redefine(a, same_v_gene=True)
    assert a.obs[vdj.CLONOTYPE_COL].iloc[0] != a.obs[vdj.CLONOTYPE_COL].iloc[1]
    assert a.obs[vdj.SIZE_COL].iloc[:2].tolist() == [1, 1]


def test_t_then_b_file_equals_one_combined_file(tb):
    one = _attached(tb)
    two = cells_adata()
    vdj.attach_file(two, tb[0])
    assert two.obs[vdj.STATUS_COL].iloc[6] == "single chain"  # TRA only, before the B file
    vdj.attach_file(two, tb[1])
    for col in (vdj.STATUS_COL, vdj.CLONOTYPE_COL, vdj.SIZE_COL, vdj.EXPANSION_COL, vdj.RECEPTOR_COL):
        assert one.obs[col].astype(str).tolist() == two.obs[col].astype(str).tolist(), col
    pd.testing.assert_frame_equal(vdj.read_table(one), vdj.read_table(two))


def test_replace_drops_the_earlier_chains(tb):
    a = cells_adata()
    vdj.attach_file(a, tb[0])
    vdj.attach_file(a, tb[0], replace=True)
    assert a.obs[vdj.STATUS_COL].astype(str).tolist()[:6] == STATUS[:6]


def test_chain_table_survives_a_round_trip_and_a_cell_subset(tmp_path, tb):
    a = _attached(tb)
    path = tmp_path / "x.h5ad"
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    pd.testing.assert_frame_equal(vdj.read_table(a), vdj.read_table(b))
    sub = b[[0, 1, 3, 9]].copy()
    vdj.redefine(sub)
    assert sub.obs[vdj.SIZE_COL].tolist() == [2, 2, 1, 1]


# ---------------------------------------------------------------------------
# Merged datasets: samples
# ---------------------------------------------------------------------------


def _merged() -> ad.AnnData:
    names = [f"{b}-1:s1" for b in BC[:10]] + [f"{b}-1:s2" for b in BC[:4]]
    return cells_adata(names, {"sample": ["s1"] * 10 + ["s2"] * 4})


def test_several_libraries_need_a_sample(tb):
    a = _merged()
    with pytest.raises(vdj.VdjError, match="2 libraries") as e:
        vdj.attach_file(a, tb[2])
    assert e.value.status == 409 and vdj.CHAINS_KEY not in a.obsm
    stats = vdj.attach_file(a, tb[2], sample_column="sample", sample="s1")
    assert stats["matched"] == 9 and stats["matched_core"] == 9 and stats["n_candidate_cells"] == 10
    assert a.obs[vdj.STATUS_COL].astype(str).tolist() == STATUS + ["not profiled"] * 4
    stats = vdj.attach_file(a, tb[0], sample_column=vdj.SUFFIX_COLUMN, sample="*-1:s2")
    assert stats["matched"] == 4 and stats["cells_without_contigs"] == 0
    # The same clonotype in both samples is one clonotype.
    clon = a.obs[vdj.CLONOTYPE_COL].astype(str).tolist()
    assert clon[10] == clon[0] and a.obs[vdj.SIZE_COL].iloc[0] == 6


def test_names_without_barcodes_match_by_name_and_are_not_libraries(tmp_path):
    a = cells_adata([str(i) for i in range(1, 5)])
    rows = [{**_row(0, 1, "TRB", "CASSA", 3), "barcode": "2", "contig_id": "2_1"}]
    stats = vdj.attach_file(a, write_csv(tmp_path / "n.csv", rows))
    assert stats["matched_exact"] == 1 and vdj.SUFFIX_COLUMN not in [c["name"] for c in vdj.sample_choices(a)]


def test_kang_style_names_are_two_libraries(tb):
    names = [f"CTRL_{b}.1" for b in BC[:5]] + [f"STIM_{b}.1" for b in BC[:5]]
    a = cells_adata(names, {"stim": ["CTRL"] * 5 + ["STIM"] * 5})
    choices = {c["name"]: c["values"] for c in vdj.sample_choices(a)}
    assert choices[vdj.SUFFIX_COLUMN] == ["CTRL_*.1", "STIM_*.1"]
    with pytest.raises(vdj.VdjError, match="2 libraries"):
        vdj.attach_file(a, tb[2])


def test_seurat_style_prefix_is_matched(tb):
    names = [f"donorA_{b}-1" for b in BC[:10]]
    a = cells_adata(names)
    stats = vdj.attach_file(a, tb[2])
    assert stats["matched_core"] == 9 and a.obs[vdj.STATUS_COL].astype(str).tolist() == STATUS


# ---------------------------------------------------------------------------
# The record, provenance and summary
# ---------------------------------------------------------------------------


def test_attach_records_the_file_and_refuses_it_twice(tb):
    a = cells_adata()
    rec = vdj.attach(a, tb[2], file_name="all.csv", sha256="abc")
    assert rec["matched"] == 9 and rec["unmatched"] == 1 and rec["unmatched_usable"] == 1
    assert rec["unmatched_examples"] == [f"{BC[10]}-1"] and rec["cells_without_contigs"] == 1
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "vdj_attach" and step["params"]["sha256"] == "abc"
    with pytest.raises(vdj.VdjError, match="already attached"):
        vdj.attach(a, tb[2], file_name="all.csv", sha256="abc")
    s = vdj.summary(a)
    assert s["has_vdj"] and s["n_clonotypes"] == 3 and s["n_expanded"] == 2 and s["n_cells_expanded"] == 5
    assert s["match"]["cells_with_chain"] == 8 and s["match"]["cells_without_contigs"] == 1
    assert s["expansion"]["2-5 cells"] == {"clonotypes": 2, "cells": 5}
    assert s["status"]["multichain"] == 1 and s["loci"]["TRA"] == 5 and not s["stale"]


def test_foreign_clonotype_column_is_not_overwritten(tb):
    a = cells_adata()
    a.obs["clonotype"] = "x"
    with pytest.raises(vdj.VdjError, match="did not write"):
        vdj.attach(a, tb[2], file_name="all.csv", sha256="abc")
    assert (a.obs["clonotype"] == "x").all()


def test_removed_cells_make_the_sizes_stale_until_recomputed(tb):
    a = cells_adata()
    vdj.attach(a, tb[2], file_name="all.csv", sha256="abc")
    sub = a[[0, 3, 4, 9]].copy()
    assert vdj.summary(sub)["stale"] is True
    vdj.set_definition(sub, same_v_gene=False)
    assert vdj.summary(sub)["stale"] is False and sub.obs[vdj.SIZE_COL].tolist() == [1, 2, 2, 1]


def test_remove_clears_every_column(tb):
    a = cells_adata()
    vdj.attach(a, tb[2], file_name="all.csv", sha256="abc")
    vdj.remove(a)
    assert vdj.CHAINS_KEY not in a.obsm and vdj.UNS_KEY not in a.uns
    assert not [c for c in a.obs.columns if "cdr3" in c or c.startswith(("clon", "vdj_"))]
    assert vdj.summary(a)["has_vdj"] is False


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_diversity_formulas_by_hand_and_scipy():
    c = np.array([3, 1])
    assert vdj.shannon(c) == pytest.approx(entropy(c / 4))
    assert vdj.shannon(c) == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
    assert vdj.simpson(c) == pytest.approx(1 - (0.75 ** 2 + 0.25 ** 2))
    assert vdj.clonality(c) == pytest.approx(1 - entropy(c / 4) / np.log(2))
    assert vdj.clonality([5]) is None
    assert vdj.clonality([1] * 7) == 0.0 and vdj.gini([1] * 7) == 0.0  # not a rounding -1e-16
    assert vdj.gini([1, 9]) == pytest.approx(0.4)  # sum|xi-xj| / (2 n^2 mean) = 16 / 40
    assert vdj.gini([4, 4, 4]) == pytest.approx(0.0)
    assert vdj.gini([0, 0, 0, 1]) == pytest.approx(0.0)  # zeros are not clonotypes


def test_overlap_formulas_by_hand():
    x, y = [2, 1, 0], [1, 1, 1]
    assert vdj.morisita_horn(x, y) == pytest.approx(0.75)  # 2*3 / ((5/9 + 3/9) * 9)
    assert vdj.morisita_horn(x, x) == pytest.approx(1.0)
    assert vdj.morisita_horn([1, 0], [0, 1]) == pytest.approx(0.0)
    assert vdj.jaccard(x, y) == pytest.approx(2 / 3)


def _repertoire() -> ad.AnnData:
    """60 paired T cells in groups A (40 cells) and B (20): A holds clonotype X x10
    and 30 singletons; B holds X x5 and 15 singletons."""
    rows, names, group, k = [], [], [], 0
    rng = np.random.default_rng(1)
    bases = list("ACGT")
    for g, n_x, n_single in (("A", 10, 30), ("B", 5, 15)):
        for i in range(n_x + n_single):
            bc = "".join(rng.choice(bases, 16))
            names.append(f"{bc}-1")
            group.append(g)
            cdr3 = "CASSX" if i < n_x else f"CASS{g}{i}"
            rows += [{**_row(0, 1, "TRA", "CAVX" if i < n_x else f"CAV{g}{i}", 3), "barcode": f"{bc}-1",
                      "contig_id": f"{bc}_1"},
                     {**_row(0, 2, "TRB", cdr3, 4, v="TRBV20-1" if i >= n_x and i % 2 else "TRBV6-5"),
                      "barcode": f"{bc}-1", "contig_id": f"{bc}_2"}]
            k += 1
    return cells_adata(names, {"group": group}), rows


def test_diversity_per_group_and_rarefaction(tmp_path):
    a, rows = _repertoire()
    vdj.attach_file(a, write_csv(tmp_path / "r.csv", rows))
    d = vdj.diversity(a, "group")
    A, B = d["groups"]
    assert (A["n_with_clonotype"], B["n_with_clonotype"]) == (40, 20)
    assert A["richness"] == 31 and B["richness"] == 16
    assert A["shannon"] == pytest.approx(entropy([10] + [1] * 30))
    assert B["simpson"] == pytest.approx(1 - (0.25 ** 2 + 15 * 0.05 ** 2))
    r1 = vdj.diversity(a, "group", rarefy=True, seed=3)
    r2 = vdj.diversity(a, "group", rarefy=True, seed=3)
    assert r1 == r2 and r1["depth"] == 20
    # B has exactly the depth: every draw takes all of its cells.
    assert r1["groups"][1]["rarefied"]["richness"] == 16 and r1["groups"][1]["rarefied"]["richness_sd"] == 0
    assert r1["groups"][0]["rarefied"]["richness"] < 31
    assert "Rarefied" in r1["caption"] and "Not rarefied" in d["caption"]
    big = vdj.diversity(a, "group", rarefy=True, min_cells=25)
    assert big["groups"][1]["richness"] is None and big["depth"] == 40 and big["min_cells"] == 25
    assert big["groups"][0]["rarefied"]["richness"] == 31  # A at its own size: every cell, every draw


def test_overlap_and_usage_per_group(tmp_path):
    a, rows = _repertoire()
    vdj.attach_file(a, write_csv(tmp_path / "r.csv", rows))
    o = vdj.overlap(a, "group")
    assert o["groups"] == ["A", "B"] and o["shared"][0][1] == 1
    # X = 40, Y = 20, sum xy = 10 * 5; Dx = (100 + 30) / 1600, Dy = (25 + 15) / 400
    assert o["matrix"][0][1] == pytest.approx(100 / ((130 / 1600 + 40 / 400) * 800), abs=1e-4)
    assert vdj.overlap(a, "group", "jaccard")["matrix"][0][1] == pytest.approx(1 / 46, abs=1e-4)
    u = vdj.usage(a, "group")
    assert u["chain"] == "TRB" and u["count"] == "clonotypes" and u["n"] == [31, 16]
    # A's clonotypes: X (cell 0, TRBV6-5) + 30 singletons at i = 10..39 (odd i = TRBV20-1: 15).
    row = dict(zip(u["genes"], u["counts"][0]))
    assert row == {"TRBV6-5": 16, "TRBV20-1": 15}
    cells = vdj.usage(a, "group", count="cells")
    assert dict(zip(cells["genes"], cells["counts"][0])) == {"TRBV6-5": 25, "TRBV20-1": 15}


def test_top_clonotypes_table(tmp_path):
    a, rows = _repertoire()
    vdj.attach_file(a, write_csv(tmp_path / "r.csv", rows))
    t = vdj.clonotypes(a, group_by="group", n=3)
    top = t["rows"][0]
    assert top["id"] == "TCR1" and top["n_cells"] == 15 and top["by_group"] == {"A": 10, "B": 5}
    assert top["top_group"] == "A" and top["top_share"] == pytest.approx(10 / 15)
    assert top["expansion"] == "6-20 cells" and len(top["indices"]) == 15
    assert [c["cdr3"] for c in top["chains"]] == ["CAVX", "CASSX"]
    within = vdj.clonotypes(a, group_by="group", group="B", n=2)
    assert within["rows"][0]["n_in_group"] == 5 and within["n_clonotypes"] == 16


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_script_repeats_the_attach_exactly(tmp_path, tb):
    import subprocess
    import sys

    from scview.core import replay_script

    inp = _merged()
    out = inp.copy()
    vdj.attach(out, tb[0], file_name=tb[0].name, sha256="t", sample_column="sample", sample="s1")
    vdj.attach(out, tb[1], file_name=tb[1].name, sha256="b", sample_column="sample", sample="s1")
    vdj.attach(out, tb[0], file_name=tb[0].name, sha256="t", sample_column=vdj.SUFFIX_COLUMN,
               sample="*-1:s2")
    vdj.set_definition(out, same_v_gene=True)
    assert [d["status"] for d in replay_script.describe(out)] == ["exact"] * 4

    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    out.write_h5ad(src)  # an input that already has the columns: the script starts from none
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    for col in vdj.our_columns(out):
        assert rep.obs[col].astype(str).tolist() == out.obs[col].astype(str).tolist(), col
    pd.testing.assert_frame_equal(vdj.read_table(rep), vdj.read_table(out))


# ---------------------------------------------------------------------------
# Co-pilot fact
# ---------------------------------------------------------------------------


def test_copilot_fact_names_the_expanded_clonotypes_and_their_cell_type(tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import build_grounding_context
    from scview.core.claims import resolve_tag

    a, rows = _repertoire()
    a.obs["cell_type"] = pd.Categorical(["CD8 T"] * 50 + ["CD4 T"] * 10)
    vdj.attach_file(a, write_csv(tmp_path / "r.csv", rows))
    p = tmp_path / "x.h5ad"
    a.write_h5ad(p)
    context, sources = build_grounding_context(AnnDataAdaptor(str(p)))
    line = next(ln for ln in context.splitlines() if "[result:vdj]" in ln)
    # X: 10 cells in A (all CD8 T) + 5 in B (cells 40-44: CD8 T) -> 15 of 15 CD8 T.
    assert "TCR1 (15 cells; TRA CAVX/TRB CASSX, 100% in cell_type 'CD8 T')" in line
    assert "46 clonotypes over 60 cells; 1 expanded (2+ cells) holding 15 cells" in line
    refs = [s.ref for s in sources]
    assert "result:vdj" in refs and resolve_tag("vdj", refs) == ["result:vdj"]
    assert resolve_tag("result:vdj:TCR1", refs) == ["result:vdj"]
    plain = tmp_path / "plain.h5ad"
    cells_adata().write_h5ad(plain)
    assert "[result:vdj]" not in build_grounding_context(AnnDataAdaptor(str(plain)))[0]
