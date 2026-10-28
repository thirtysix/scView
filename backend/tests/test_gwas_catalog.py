"""GWAS trait links (core/gwas_catalog.py, api/v1/gwas.py) on tiny synthetic
catalog files in the real column layout; no network."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scview.core import claims, gwas_catalog as gc, orthologs, provenance, replay_script, results_store

API = "/api/v1"

# The 38 columns of gwas-catalog-download-associations-alt-full.tsv (v1.0.2), in order.
COLUMNS = [
    "DATE ADDED TO CATALOG", "PUBMEDID", "FIRST AUTHOR", "DATE", "JOURNAL", "LINK", "STUDY", "DISEASE/TRAIT",
    "INITIAL SAMPLE SIZE", "REPLICATION SAMPLE SIZE", "REGION", "CHR_ID", "CHR_POS", "REPORTED GENE(S)",
    "MAPPED_GENE", "UPSTREAM_GENE_ID", "DOWNSTREAM_GENE_ID", "SNP_GENE_IDS", "UPSTREAM_GENE_DISTANCE",
    "DOWNSTREAM_GENE_DISTANCE", "STRONGEST SNP-RISK ALLELE", "SNPS", "MERGED", "SNP_ID_CURRENT", "CONTEXT",
    "INTERGENIC", "RISK ALLELE FREQUENCY", "P-VALUE", "PVALUE_MLOG", "P-VALUE (TEXT)", "OR or BETA",
    "95% CI (TEXT)", "PLATFORM [SNPS PASSING QC]", "CNV", "MAPPED_TRAIT", "MAPPED_TRAIT_URI",
    "STUDY ACCESSION", "GENOTYPING TECHNOLOGY",
]
EFO = "http://www.ebi.ac.uk/efo/EFO_"


def uri(k: int) -> str:
    return f"{EFO}{k:07d}"


def row(*, chrom="1", pos="1000", mapped="", snp_ids="", up="", down="", intergenic="0", p="1E-10",
        trait="trait", uris=None, pmid="100", reported=None) -> dict:
    p_float = float(p)
    return {"PUBMEDID": pmid, "CHR_ID": chrom, "CHR_POS": pos, "MAPPED_GENE": mapped, "UPSTREAM_GENE_ID": up,
            "DOWNSTREAM_GENE_ID": down, "SNP_GENE_IDS": snp_ids, "INTERGENIC": intergenic, "P-VALUE": p,
            "PVALUE_MLOG": "" if p_float == 0 else repr(-np.log10(p_float)), "MAPPED_TRAIT": trait,
            "MAPPED_TRAIT_URI": uris or uri(1), "DISEASE/TRAIT": reported or trait, "STUDY ACCESSION": "GCST1"}


def tsv_bytes(rows: list[dict]) -> bytes:
    """The TSV as the Catalog writes it: CRLF line ends, empty fields for missing."""
    lines = ["\t".join(COLUMNS)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in COLUMNS))
    return ("\r\n".join(lines) + "\r\n").encode()


def zip_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("gwas-catalog-download-associations-alt-full.tsv", tsv_bytes(rows))
    return buf.getvalue()


def mappings_bytes(traits: dict[str, tuple[str, str]]) -> bytes:
    lines = ["Disease trait\tEFO term\tEFO URI\tParent term\tParent URI"]
    for u, (label, parent) in traits.items():
        lines.append(f"{label} (reported)\t{label}\t{u}\t{parent}\t{EFO}0000001")
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_fixture() -> gc.Catalog:
    rows = [
        # in a gene: Ensembl id from SNP_GENE_IDS, symbol from MAPPED_GENE
        row(mapped="IRF5", snp_ids="ENSG00000128604", pos="128000000", chrom="7", uris=uri(1), trait="lupus"),
        # intergenic: both flanking genes, ids from UPSTREAM/DOWNSTREAM
        row(mapped="KIF21B - CACNA1S", up="ENSG00000116852", down="ENSG00000081248", intergenic="1",
            chrom="1", pos="201037533", uris=uri(1)),
        # two traits on one row (comma-separated URIs); labels from the mappings file
        row(mapped="PTPN22", snp_ids="ENSG00000134242", chrom="1", pos="113834946",
            uris=f"{uri(1)}, {uri(2)}", trait="lupus, rheumatoid arthritis"),
        # extended MHC: dropped
        row(mapped="HLA-DRB1", snp_ids="ENSG00000196126", chrom="6", pos="32600000", uris=uri(1)),
        # not genome-wide significant: dropped
        row(mapped="WEAK1", chrom="2", pos="5000", p="1E-6", uris=uri(1)),
        # p below double precision: kept by PVALUE_MLOG
        dict(row(mapped="TINY1", chrom="3", pos="7000", uris=uri(2)), **{"P-VALUE": "1E-400", "PVALUE_MLOG": "400"}),
        # a haplotype: one record per SNP
        row(mapped="HAPA; HAPB - HAPC", chrom="4;4", pos="1000;900000", uris=uri(3)),
        # a haplotype whose parts do not line up: dropped
        row(mapped="BADA; BADB; BADC", chrom="5;5", pos="1;2", uris=uri(3)),
        # two genes in one row, more symbols than ids: ids resolved through other rows
        row(mapped="IRF5, TNPO3", snp_ids="ENSG00000128604", chrom="7", pos="128100000", uris=uri(3)),
        # no mapped gene at all
        row(mapped="", chrom="8", pos="100", uris=uri(3)),
        # the same SNP for the same trait in a second study, smaller p: one record, the better study
        row(mapped="IRF5", snp_ids="ENSG00000128604", pos="128000000", chrom="7", uris=uri(1), p="1E-30",
            pmid="200"),
        # a newer URI the mappings file does not list: the parent of its reported trait there
        row(mapped="NOD2", snp_ids="ENSG00000167207", chrom="16", pos="50700000", trait="Crohn disease",
            uris="http://purl.obolibrary.org/obo/MONDO_0005011", reported="rheumatoid arthritis (reported)"),
        # a trait missing from the mappings file
        row(mapped="CD14", snp_ids="ENSG00000170458", chrom="5", pos="140000000", uris=uri(9),
            trait="amount of CD14 in blood"),
    ]
    maps = mappings_bytes({uri(1): ("systemic lupus erythematosus", "Immune system disorder"),
                           uri(2): ("rheumatoid arthritis", "Immune system disorder"),
                           uri(3): ("haplotype trait", "Other trait")})
    return gc.build(io.BytesIO(zip_bytes(rows)), io.BytesIO(maps), chunk_rows=4)


def _records(cat: gc.Catalog) -> pd.DataFrame:
    return pd.DataFrame({"trait": cat.trait_uri[cat.trait], "gene": cat.symbols[cat.gene],
                         "chrom": cat.chrom, "pos": cat.pos, "mlog": cat.mlog, "pmid": cat.pmid})


def test_parse_real_layout():
    cat = _parse_fixture()
    rec = _records(cat)
    genes = set(rec["gene"])
    assert {"IRF5", "KIF21B", "CACNA1S", "PTPN22", "TINY1", "HAPA", "HAPB", "HAPC", "TNPO3", "CD14", "NOD2"} == genes
    assert "HLA-DRB1" not in genes and "WEAK1" not in genes and "BADA" not in genes
    # comma-separated URIs count for each trait
    assert set(rec.loc[rec["gene"] == "PTPN22", "trait"]) == {uri(1), uri(2)}
    # intergenic flanks carry their Ensembl ids
    ens = dict(zip(cat.symbols, cat.ensembl))
    assert ens["KIF21B"] == "ENSG00000116852" and ens["CACNA1S"] == "ENSG00000081248"
    assert ens["IRF5"] == "ENSG00000128604" and ens["TNPO3"] == ""
    # haplotype parts go to their own SNP
    hap = rec[rec["gene"].str.startswith("HAP")].set_index("gene")
    assert hap.loc["HAPA", "pos"] == 1000 and hap.loc["HAPC", "pos"] == 900000
    # one record per trait, gene and SNP: the smallest p and its study
    irf = rec[(rec["gene"] == "IRF5") & (rec["trait"] == uri(1))]
    assert len(irf) == 1 and irf["pmid"].iat[0] == 200 and irf["mlog"].iat[0] == pytest.approx(30)
    assert rec.loc[rec["gene"] == "TINY1", "mlog"].iat[0] == pytest.approx(400)
    # labels and parents from the mappings file; a missing trait gets one from its label
    lab = dict(zip(cat.trait_uri, zip(cat.trait_label, cat.trait_parent)))
    assert lab[uri(2)] == ("rheumatoid arthritis", "Immune system disorder")
    assert lab[uri(9)] == ("amount of CD14 in blood", gc.MEASUREMENT_PARENT)
    assert lab["http://purl.obolibrary.org/obo/MONDO_0005011"] == ("Crohn disease", "Immune system disorder")
    assert cat.meta["n_xmhc"] == 1 and cat.meta["n_rows"] == 13 and cat.meta["n_significant"] == 12


def test_row_records_forms():
    assert gc.row_records("1", "10", "A - B", "x", "", "", "1") == [("A", "", 1, 10), ("B", "", 1, 10)]
    assert gc.row_records("1", "10", "A, B", "", "", "ENSG00000000009", "0") == [
        ("", "ENSG00000000009", 1, 10), ("A", "", 1, 10), ("B", "", 1, 10)]
    assert gc.row_records("1", "10", "A - B", "ENSG00000000001", "ENSG00000000002", float("nan"), "1") == [
        ("A", "ENSG00000000001", 1, 10), ("B", "ENSG00000000002", 1, 10)]
    assert gc.row_records("X", "5", "A, B", "", "", "", "0") == [("A", "", 23, 5), ("B", "", 23, 5)]
    assert gc.row_records("10 x 14", "1 x 2", "A - B x C", "", "", "", "") == [
        ("A", "", 10, 1), ("B", "", 10, 1), ("C", "", 14, 2)]
    assert gc.row_records(float("nan"), float("nan"), float("nan"), float("nan"), "", float("nan"), "1") == []


def test_build_refuses_a_file_without_the_columns():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.tsv", "A\tB\r\n1\t2\r\n")
    with pytest.raises(ValueError, match="lacks columns"):
        gc.build(buf, io.BytesIO(mappings_bytes({})))


def test_loci_merge_within_one_megabase():
    trait = np.array([0, 0, 0, 0, 0, 1, 0])
    chrom = np.array([1, 1, 1, 1, 2, 1, 1])
    pos = np.array([0, 500_000, 1_400_000, 3_000_000, 600_000, 500_000, 4_000_000])
    loc = gc.assign_loci(trait, chrom, pos)
    # 0 / 0.5 / 1.4 Mb chain into one locus; 3 Mb and 4 Mb are one more (1 Mb apart); chr2 and trait 1 apart
    assert loc[0] == loc[1] == loc[2]
    assert loc[3] == loc[6] != loc[0]
    assert len({loc[0], loc[3], loc[4], loc[5]}) == 4


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_hypergeometric_odds_ratio_and_bh_match_scipy():
    N, n = 2000, 80
    K = np.array([10, 50, 200, 500])
    k = np.array([3, 5, 12, 0])
    p = gc.hypergeom_sf(k, N, K, n)
    for kk, KK, pp in zip(k, K, p):
        table = [[kk, n - kk], [KK - kk, N - n - KK + kk]]
        assert pp == pytest.approx(stats.fisher_exact(table, alternative="greater")[1], rel=1e-9)
    orr = gc.odds_ratio(k[:3], N, K[:3], n)
    for kk, KK, o in zip(k[:3], K[:3], orr):
        assert o == pytest.approx(stats.fisher_exact([[kk, n - kk], [KK - kk, N - n - KK + kk]])[0])
    assert np.isfinite(gc.odds_ratio(np.array([0]), N, np.array([10]), n)).all()  # Haldane for a zero cell
    raw = np.array([0.001, 0.04, 0.03, 0.5, 0.2, 0.0001])
    np.testing.assert_allclose(gc.bh(raw), multipletests(raw, method="fdr_bh")[1])


def _synthetic_catalog(n_genes=600, n_traits=80, seed=3) -> gc.Catalog:
    """Traits of 15-150 genes, each gene its own locus (separate chromosomes slots),
    gene degree skewed (a few genes in many traits)."""
    rng = np.random.default_rng(seed)
    weight = 1.0 / (1 + np.arange(n_genes)) ** 0.6
    weight /= weight.sum()
    tr, gn = [], []
    for t in range(n_traits):
        size = int(rng.integers(15, 150))
        genes = rng.choice(n_genes, size=size, replace=False, p=weight)
        tr.extend([t] * size)
        gn.extend(genes.tolist())
    trait = np.asarray(tr, np.int32)
    gene = np.asarray(gn, np.int32)
    chrom = (1 + gene % 22).astype(np.int8)
    pos = (gene * 5_000_000 % 200_000_000 + 1_000).astype(np.int32)
    cat = gc.Catalog(
        symbols=np.asarray([f"G{i}" for i in range(n_genes)]), ensembl=np.asarray([""] * n_genes),
        trait_uri=np.asarray([uri(t) for t in range(n_traits)]),
        trait_label=np.asarray([f"trait {t}" for t in range(n_traits)]),
        trait_parent=np.asarray([gc.MEASUREMENT_PARENT if t % 10 == 0 else "Other trait" for t in range(n_traits)]),
        trait=trait, gene=gene, chrom=chrom, pos=pos, mlog=np.full(trait.size, 10, np.float32),
        pmid=np.full(trait.size, 1, np.int32), locus=gc.assign_loci(trait, chrom, pos),
    )
    return cat


def test_trait_filters_size_and_measurements():
    cat = _synthetic_catalog()
    bg = gc.background(cat, [f"G{i}" for i in range(300)])
    assert bg.genes.size == 300
    with_m = gc.eligible_traits(cat, bg, include_measurements=True)
    without = gc.eligible_traits(cat, bg, include_measurements=False)
    assert ((bg.sizes[with_m] >= gc.MIN_TRAIT) & (bg.sizes[with_m] <= gc.MAX_TRAIT)).all()
    assert set(without) < set(with_m)
    assert all(cat.trait_parent[t] == gc.MEASUREMENT_PARENT for t in set(with_m) - set(without))
    small = np.flatnonzero(bg.sizes < gc.MIN_TRAIT)
    assert not set(small) & set(with_m)


def test_random_sets_are_seeded_and_degree_matched():
    cat = _synthetic_catalog()
    bg = gc.background(cat, cat.symbols)
    q = np.arange(0, 600, 13)
    a = gc.random_sets(bg.bins, q, 50, seed=7)
    b = gc.random_sets(bg.bins, q, 50, seed=7)
    assert (a != b).nnz == 0
    assert (a != gc.random_sets(bg.bins, q, 50, seed=8)).nnz > 0
    want = np.bincount(bg.bins[q], minlength=gc.N_BINS)
    for d in range(50):
        got = np.bincount(bg.bins[a[d].indices], minlength=gc.N_BINS)
        assert (got == want).all() and a[d].nnz == q.size
    # deciles are equal-sized and ordered by degree
    sizes = np.bincount(bg.bins)
    assert sizes.max() - sizes.min() <= 1
    assert bg.degree[bg.bins == 0].max() <= bg.degree[bg.bins == gc.N_BINS - 1].min()


def test_group_traits_deterministic_and_finds_a_planted_trait():
    cat = _synthetic_catalog()
    bg = gc.background(cat, cat.symbols)
    t = 5
    members = np.flatnonzero(np.asarray(bg.gt[:, t].todense()).ravel())
    # the trait's rarer genes plus rare genes outside it (the commonest genes are in every trait)
    q = np.sort(np.concatenate([members[-20:], np.setdiff1d(np.arange(600), members)[-30:]]))
    r1 = gc.group_traits(cat, bg, q, include_measurements=True, n_draws=300, refine_draws=1000, seed=1)
    r2 = gc.group_traits(cat, bg, q, include_measurements=True, n_draws=300, refine_draws=1000, seed=1)
    assert r1 == r2
    assert r1["rows"][0]["uri"] == uri(t)
    top = r1["rows"][0]
    assert top["n_overlap"] >= 20 and top["n_loci"] >= 3
    assert top["p_empirical"] < 0.01 and top["n_draws"] == 1000 and top["lead"]
    assert all(r["n_loci"] >= gc.MIN_LOCI for r in r1["rows"])
    assert r1["n_tested"] == gc.eligible_traits(cat, bg, True).size
    # BH across the tested traits
    assert top["p_adj"] >= top["p"]


def test_empirical_p_is_calibrated_on_random_queries():
    cat = _synthetic_catalog(seed=11)
    bg = gc.background(cat, cat.symbols)
    rng = np.random.default_rng(5)
    ps = []
    for i in range(40):
        q = np.sort(rng.choice(bg.genes.size, 40, replace=False))
        r = gc.group_traits(cat, bg, q, include_measurements=True, n_draws=200, refine_draws=0, seed=i,
                            max_rows=10_000, min_loci=1)
        # traits with no overlap have p = 1 and are not listed
        listed = {x["uri"]: x["p_empirical"] for x in r["rows"]}
        ps.extend(listed.get(uri(t), 1.0) for t in gc.eligible_traits(cat, bg, True))
    ps = np.asarray(ps)
    for alpha in (0.05, 0.2, 0.5):
        assert (ps <= alpha).mean() <= alpha + 0.03, alpha
    assert (ps <= 0.5).mean() > 0.3  # not degenerate


def test_small_query_is_not_tested():
    cat = _synthetic_catalog()
    bg = gc.background(cat, cat.symbols)
    r = gc.group_traits(cat, bg, np.arange(3))
    assert r["rows"] == [] and r["n_query"] == 3


# ---------------------------------------------------------------------------
# One gene; mouse
# ---------------------------------------------------------------------------


def test_gene_traits_sorted_with_pubmed_ids():
    cat = _parse_fixture()
    got = gc.gene_traits(cat, "irf5")
    assert got["gene"] == "IRF5" and got["ensembl"] == "ENSG00000128604"
    assert [t["trait"] for t in got["traits"]] == ["systemic lupus erythematosus", "haplotype trait"]
    assert got["traits"][0]["pmids"] == [200] and got["traits"][0]["p"] == pytest.approx(1e-30)
    assert gc.gene_traits(cat, "ENSG00000128604")["gene"] == "IRF5"
    assert gc.gene_traits(cat, "NOPE") is None


def _hcop() -> pd.DataFrame:
    return pd.DataFrame({"human": ["CD3E", "STAT1", "IRF5", "IRF5", "A1", "A2", "A3", "A4", "A5", "A6"],
                         "mouse": ["Cd3e", "Stat1", "Irf5", "Irf5b", "Multi", "Multi", "Multi", "Multi", "Multi",
                                   "Multi"],
                         "evidence": np.array([5, 4, 3, 2, 3, 3, 3, 3, 3, 3], dtype=np.int16)})


def test_mouse_to_human_mirrors_the_rules():
    m = orthologs.mouse_to_human(["Cd3e", "Irf5", "Irf5b", "Multi", "Nope"], table=_hcop())
    assert m == {"Cd3e": ["CD3E"], "Irf5": ["IRF5"], "Irf5b": [], "Multi": [], "Nope": []}


# ---------------------------------------------------------------------------
# A dataset with markers
# ---------------------------------------------------------------------------


def _marker_dataset(prefix: str = "G", n_genes: int = 400, seed: int = 0) -> ad.AnnData:
    """Three groups of 60 cells; group A expresses genes 0-19 highly, B 20-39, C 40-59."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(180, n_genes)).astype(np.float32)
    for g, start in enumerate((0, 20, 40)):
        X[g * 60:(g + 1) * 60, start:start + 20] += rng.poisson(8.0, size=(60, 20))
    a = ad.AnnData(X, obs=pd.DataFrame({"group": pd.Categorical(np.repeat(["A", "B", "C"], 60))},
                                       index=[f"c{i}" for i in range(180)]),
                   var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_genes)]))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", pts=True)
    return a


def _catalog_rows(symbol=lambda i: f"G{i}") -> list[dict]:
    """Trait 1 holds genes 0-14 (group A's markers), each at its own locus, plus 300-309;
    28 filler traits of 25 genes from all 400 give the background its degrees."""
    rng = np.random.default_rng(1)
    rows = [row(mapped=symbol(i), chrom=str(1 + i % 20), pos=str(10_000_000 + i * 3_000_000), uris=uri(1),
                trait="planted") for i in [*range(15), *range(300, 310)]]
    for t in range(2, 30):
        for i in rng.choice(np.arange(400), size=25, replace=False):
            rows.append(row(mapped=symbol(int(i)), chrom=str(1 + int(i) % 20), pos=str(10_000_000 + int(i) * 3_000_000),
                            uris=uri(t), trait=f"filler {t}"))
    return rows


def _place(dest: Path, rows: list[dict], monkeypatch) -> None:
    """Put both files where a user would place them by hand, pinned to their hashes."""
    dest.mkdir(parents=True, exist_ok=True)
    z = zip_bytes(rows)
    m = mappings_bytes({uri(1): ("planted trait", "Immune system disorder"),
                        **{uri(t): (f"filler trait {t}", "Other trait") for t in range(2, 30)}})
    (dest / gc.ASSOC_NAME).write_bytes(z)
    (dest / gc.MAPPINGS_NAME).write_bytes(m)
    monkeypatch.setattr(gc, "ASSOC_SHA256", hashlib.sha256(z).hexdigest())
    monkeypatch.setattr(gc, "MAPPINGS_SHA256", hashlib.sha256(m).hexdigest())


def test_run_on_a_dataset_finds_the_planted_trait(tmp_path, monkeypatch):
    _place(tmp_path, _catalog_rows(), monkeypatch)
    cat = gc.ensure(tmp_path, allow_download=False)
    res = gc.run(cat, _marker_dataset(), "group", n_draws=200, refine_draws=500)
    by = {g["group"]: g for g in res["groups"]}
    assert set(by) == {"A", "B", "C"}
    top = by["A"]["rows"][0]
    assert top["trait"] == "planted trait" and top["n_loci"] >= 10 and top["lead"]
    assert set(top["genes"]) <= {f"G{i}" for i in range(15)} and top["dataset_genes"] == top["genes"]
    assert not any(r["trait"] == "planted trait" for r in by["B"]["rows"])
    assert res["caveat"] == gc.CAVEAT and res["release"] == gc.RELEASE and res["mapping"] is None
    with pytest.raises(KeyError):
        gc.run(cat, _marker_dataset(), "nope")
    bare = _marker_dataset()
    del bare.uns["rank_genes_groups"]
    with pytest.raises(ValueError, match="Markers tab"):
        gc.run(cat, bare, "group")


def test_run_on_mouse_maps_through_orthologs(tmp_path, monkeypatch):
    _place(tmp_path, _catalog_rows(), monkeypatch)
    cat = gc.ensure(tmp_path, allow_download=False)
    table = pd.DataFrame({"human": [f"G{i}" for i in range(400)], "mouse": [f"m{i}" for i in range(400)],
                          "evidence": np.full(400, 5, np.int16)})
    res = gc.run(cat, _marker_dataset(prefix="m"), "group", species="mouse", n_draws=200, refine_draws=0,
                 to_human=lambda s: orthologs.mouse_to_human(s, table=table))
    top = next(g for g in res["groups"] if g["group"] == "A")["rows"][0]
    assert top["trait"] == "planted trait"
    assert top["dataset_genes"] == [g.replace("G", "m") for g in top["genes"]]
    assert "HCOP" in res["mapping"] and res["n_background"] > 100


# ---------------------------------------------------------------------------
# Fetching and preparing
# ---------------------------------------------------------------------------


def test_prepare_from_files_placed_by_hand_and_wrong_files(tmp_path, monkeypatch):
    with pytest.raises(gc.GwasUnavailable, match="GWAS_AUTO_DOWNLOAD is off"):
        gc.ensure(tmp_path, allow_download=False)
    _place(tmp_path, _catalog_rows(), monkeypatch)
    monkeypatch.setattr(gc, "MAPPINGS_SHA256", "0" * 64)
    with pytest.raises(gc.GwasUnavailable, match="delete it to fetch it again"):
        gc.ensure(tmp_path, allow_download=False)
    assert gc.read_meta(tmp_path) is None
    with pytest.raises(gc.GwasUnavailable, match="not downloaded yet"):
        gc.load(tmp_path)


def test_download_checks_sha256_and_reports_progress(tmp_path, monkeypatch):
    z, m = zip_bytes(_catalog_rows()), mappings_bytes({uri(1): ("planted trait", "Immune system disorder")})
    monkeypatch.setattr(gc, "ASSOC_SHA256", hashlib.sha256(z).hexdigest())
    monkeypatch.setattr(gc, "MAPPINGS_SHA256", hashlib.sha256(m).hexdigest())
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200, content=z if req.url.path.endswith(".zip") else m)

    events = []
    cat = gc.ensure(tmp_path, allow_download=True, client=httpx.Client(transport=httpx.MockTransport(handler)),
                    progress=events.append)
    assert seen == [gc.ASSOC_URL, gc.MAPPINGS_URL]
    assert cat.n_traits == 29 and gc.read_meta(tmp_path)["assoc_sha256"] == gc.ASSOC_SHA256
    assert any(e["stage"] == "parse" for e in events)
    assert not list(tmp_path.glob("*.part"))

    other = tmp_path / "bad"
    monkeypatch.setattr(gc, "ASSOC_SHA256", "f" * 64)
    with pytest.raises(gc.GwasUnavailable, match="not the release"):
        gc.ensure(other, allow_download=True, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert not (other / gc.ASSOC_NAME).exists() and not list(other.glob("*.part"))


def test_auto_download_flag(monkeypatch):
    monkeypatch.delenv("GWAS_AUTO_DOWNLOAD", raising=False)
    assert gc.auto_download()
    monkeypatch.setenv("GWAS_AUTO_DOWNLOAD", "false")
    assert not gc.auto_download()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_api_not_downloaded_then_download_run_get(client, dataset_manager, test_settings, monkeypatch):
    monkeypatch.setenv("GWAS_AUTO_DOWNLOAD", "false")
    monkeypatch.setattr(gc, "N_DRAWS", 200)
    did = _register(dataset_manager, _marker_dataset(), "gw1")
    st = (await client.get(f"{API}/gwas/status")).json()
    assert st["downloaded"] is False and st["release"] == gc.RELEASE and st["caveat"] == gc.CAVEAT
    r = await client.post(f"{API}/datasets/{did}/gwas/clusters", json={"column": "group", "species": "human"})
    assert r.status_code == 409 and "not downloaded yet" in r.json()["detail"]
    assert (await client.get(f"{API}/gwas/gene/G1")).status_code == 409

    # Off and nothing on disk: the stream says how to place the files.
    body = (await client.post(f"{API}/gwas/download")).text
    assert "event: error" in body and "GWAS_AUTO_DOWNLOAD is off" in body

    # Placed by hand: prepared without a download.
    _place(Path(test_settings.DATA_DIR) / "gwas", _catalog_rows(), monkeypatch)
    body = (await client.post(f"{API}/gwas/download")).text
    assert "event: progress" in body and "event: complete" in body
    st = (await client.get(f"{API}/gwas/status")).json()
    assert st["downloaded"] and st["n_traits"] == 29

    assert (await client.get(f"{API}/datasets/{did}/gwas/clusters")).status_code == 404
    r = await client.post(f"{API}/datasets/{did}/gwas/clusters", json={"column": "nope", "species": "human"})
    assert r.status_code == 422 and "nope" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/gwas/clusters",
                          json={"column": "group", "species": "human", "n_markers": 50})
    assert r.status_code == 200, r.text
    res = r.json()
    a = next(g for g in res["groups"] if g["group"] == "A")
    assert a["rows"][0]["trait"] == "planted trait" and res["n_markers"] == 50

    got = (await client.get(f"{API}/datasets/{did}/gwas/clusters")).json()
    assert got["result_id"] == res["result_id"]
    assert (await client.get(f"{API}/datasets/{did}/gwas/clusters?column=other")).status_code == 404

    # A History step, saved with the dataset; kept across a reload; additive.
    results_store.flush()
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    step = provenance.read_provenance(adaptor.adata)["history"][-1]
    assert step["step"] == "gwas_traits" and step["params"]["column"] == "group"
    assert "gwas_traits" in results_store.ADDITIVE_STEPS
    gc._last.clear()
    again = (await client.get(f"{API}/datasets/{did}/gwas/clusters?column=group")).json()
    assert again["result_id"] == res["result_id"]

    gene = (await client.get(f"{API}/gwas/gene/g3")).json()
    assert gene["gene"] == "G3" and gene["traits"][0]["trait"] == "planted trait"
    assert (await client.get(f"{API}/gwas/gene/NOPE")).status_code == 404


async def test_api_cluster_run_needs_markers(client, dataset_manager, test_settings, monkeypatch):
    _place(Path(test_settings.DATA_DIR) / "gwas", _catalog_rows(), monkeypatch)
    gc.ensure(Path(test_settings.DATA_DIR) / "gwas", allow_download=False)
    a = _marker_dataset()
    del a.uns["rank_genes_groups"]
    did = _register(dataset_manager, a, "gw2")
    r = await client.post(f"{API}/datasets/{did}/gwas/clusters", json={"column": "group", "species": "human"})
    assert r.status_code == 422 and "Markers tab" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Co-pilot, replay
# ---------------------------------------------------------------------------


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata
        self.h5ad_path = None


def test_grounding_fact_for_the_highlighted_group(tmp_path, monkeypatch):
    _place(tmp_path, _catalog_rows(), monkeypatch)
    cat = gc.ensure(tmp_path, allow_download=False)
    adata = _marker_dataset()
    adaptor = _Adaptor(adata)
    assert gc.grounding_lines(adaptor, {"column": "group", "value": "A"}) == ([], "")
    res = gc.run(cat, adata, "group", n_draws=200, refine_draws=0)
    gc.remember(adaptor, res)
    lines, detail = gc.grounding_lines(adaptor, {"column": "group", "value": "A"})
    text = "\n".join(lines)
    assert "planted trait" in text and "not evidence that this cell type mediates" in text
    assert "[result:gwas:group:A]" in text and detail == "GWAS traits of group A"
    assert gc.grounding_lines(adaptor, {"column": "other", "value": "A"}) == ([], "")
    assert gc.grounding_lines(adaptor, None) == ([], "")
    from scview.core import assistant

    blocks, sources = [], []
    assistant._add_gwas_traits(adaptor, {"highlighted": {"column": "group", "value": "A"}}, blocks, sources)
    assert sources[0].ref == "result:gwas:group:A" and "planted trait" in sources[0].evidence
    refs = ["result:gwas:group:A", "result:ppi"]
    assert claims.resolve_tag("gwas:group:A", refs) == ["result:gwas:group:A"]
    assert claims.resolve_tag("result:traits", refs) == ["result:gwas:group:A"]


def test_replay_notes_the_step_without_code():
    snip = replay_script._snippet({"step": "gwas_traits", "params": {"column": "group", "release": gc.RELEASE}},
                                  None)
    assert snip.status == replay_script.NOT_REPLAYABLE
    assert any("not replayed" in line for line in snip.code)
