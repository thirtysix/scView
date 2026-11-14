"""Getting spliced and unspliced counts in (core/velocity_input.py): names matched across tools,
each file format, the plain and wizard ingests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import modality_store as ms
from scview.core import velocity_input as vi
from tests.velocity_fixture import barcode, legacy_categories_h5ad, legacy_h5ad, splicing_counts, write_loom, write_starsolo


def test_barcode_of_reads_each_tools_convention():
    assert vi.barcode_of("possorted_genome_bam_ABC12:AAACCTGAGAGGGATAx") == ("AAACCTGAGAGGGATA", "possorted genome bam ABC12")
    assert vi.barcode_of("AAACCTGAGAGGGATA-1") == ("AAACCTGAGAGGGATA", "1")
    assert vi.barcode_of("AAACCTGAGAGGGATA-1:s2")[1] == "1 s2"
    assert vi.barcode_of("cell7") == ("cell7", "")


def test_match_cells_by_barcode_then_sample():
    ds = [barcode(i) + "-1" for i in range(5)]
    m = vi.match_cells([f"sampleA:{barcode(i)}x" for i in (4, 1, 9)], ds)
    assert m.n_matched == 2 and m.file_rows.tolist() == [0, 1] and m.dataset_names == [ds[4], ds[1]]
    assert not m.used_sample

    # The same barcode in two samples of a merged dataset: the file's prefix tells them apart.
    ds = [f"{barcode(1)}-1:s1", f"{barcode(1)}-1:s2", f"{barcode(2)}-1:s1"]
    m = vi.match_cells([f"s1:{barcode(1)}x", f"s2:{barcode(1)}x", f"s1:{barcode(2)}x"], ds)
    assert m.dataset_names == ds and m.used_sample
    # A prefix that says nothing: ambiguous, counted, left out.
    m = vi.match_cells([f"lane:{barcode(1)}x", f"lane:{barcode(2)}x"], ds)
    assert m.n_ambiguous == 1 and m.dataset_names == [ds[2]]


def test_refusal_says_how_many_matched():
    m = vi.match_cells([f"s:{barcode(i)}x" for i in range(100, 200)], [barcode(i) + "-1" for i in range(90, 110)] + ["x"] * 80)
    with pytest.raises(vi.VelocityInputError, match=r"matches only 10 of this dataset's 100 cells \(100 cells in the file\)"):
        vi.check_majority(m, "f.loom")
    # One sample of a merged dataset: most of the file matches though few of the dataset do.
    m = vi.match_cells([barcode(i) for i in range(10)], [barcode(i) + "-1" for i in range(100)])
    vi.check_majority(m, "f.loom")


def test_adopt_anndata_moves_the_layers_out(tmp_path):
    a = splicing_counts(50, 60)
    want = a.layers["unspliced"][7].toarray().copy()
    rec = vi.adopt_anndata(a, "ds", tmp_path, file_name="u.h5ad")
    assert not any(k in a.layers for k in vi.LAYERS)
    assert set(rec["matrices"]) == set(vi.LAYERS) and rec["n_cells"] == 50
    assert json.loads(a.uns[vi.UNS_KEY])["source"]["file"] == "u.h5ad"
    with ms.ModalityStore(vi.store_path(tmp_path, "ds")) as st:
        row = st.rows_of([a.obs_names[7]])
        assert (st.read("unspliced", row).toarray() == want).all()
        assert st.feature_attr("gene_id")[0] == "ENSG00000.1"

    plain = ad.AnnData(sp.csr_matrix(np.ones((3, 2), dtype=np.float32)))
    assert vi.adopt_anndata(plain, "none", tmp_path) is None
    assert not vi.has_store(tmp_path, "none") and vi.UNS_KEY not in plain.uns
    one = ad.AnnData(sp.csr_matrix(np.ones((3, 2), dtype=np.float32)), layers={"spliced": sp.csr_matrix(np.ones((3, 2), dtype=np.float32))})
    assert vi.adopt_anndata(one, "one", tmp_path) is None and "spliced" in one.layers


def test_adopt_upload_of_a_legacy_h5ad(tmp_path):
    a = splicing_counts(60, 70)
    src = tmp_path / "legacy.h5ad"
    legacy_h5ad(src, a)
    before = src.read_bytes()
    assert vi.file_has_layers(src)
    out = tmp_path / "out.h5ad"
    rec = vi.adopt_upload(src, out, "ds", tmp_path, file_name="legacy.h5ad")
    assert rec["cells_matched"] == 60 and src.read_bytes() == before
    back = ad.read_h5ad(out)
    assert back.n_obs == 60 and "spliced" not in back.layers and "clusters" in back.obs
    assert json.loads(back.uns[vi.UNS_KEY])["source"]["file"] == "legacy.h5ad"
    with ms.ModalityStore(vi.store_path(tmp_path, "ds")) as st:
        assert st.matrices == ["spliced", "unspliced"]
        assert (st.read("spliced", np.arange(60)).toarray() == a.layers["spliced"].toarray()).all()
        assert st.barcodes[3] == a.obs_names[3]


def test_adopt_upload_keeps_legacy_category_references_valid(tmp_path):
    """anndata 0.7.x categoricals (``categories`` = an HDF5 object reference): a raw h5py copy
    left them pointing into the upload, and the dataset could not be opened (QA round 5)."""
    a = splicing_counts(60, 70)
    src = tmp_path / "erythroid.h5ad"
    legacy_categories_h5ad(src, a)
    want = ad.read_h5ad(src)
    out = tmp_path / "out.h5ad"
    vi.adopt_upload(src, out, "ds", tmp_path, file_name="erythroid.h5ad")
    back = ad.read_h5ad(out)
    assert "spliced" not in back.layers
    assert back.obs["clusters"].tolist() == want.obs["clusters"].tolist()
    assert list(back.obs["stage"].cat.categories) == ["E7", "E8"]
    assert back.var["chrom"].tolist() == want.var["chrom"].tolist()


def test_adopt_upload_refuses_a_copy_that_does_not_open(tmp_path, monkeypatch):
    from scview.core import h5_copy

    a = splicing_counts(30, 40)
    src = tmp_path / "erythroid.h5ad"
    legacy_categories_h5ad(src, a)
    monkeypatch.setattr(h5_copy, "repoint", lambda s, d: 0)  # the old raw copy
    out = tmp_path / "out.h5ad"
    with pytest.raises(Exception, match="(?i)reference"):
        vi.adopt_upload(src, out, "ds", tmp_path, file_name="erythroid.h5ad")
    assert not out.exists() and not list(tmp_path.glob(".out.h5ad*"))


@pytest.mark.parametrize("kind", ["loom", "starsolo", "starsolo_gz", "h5ad"])
def test_attach_each_format_matches_the_same_counts(tmp_path, kind):
    a = splicing_counts(80, 70)
    dataset = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    dataset = dataset[10:].copy()  # filtered: 70 of the file's 80 cells remain
    if kind == "loom":
        write_loom(tmp_path / "v.loom", a)
        paths = [tmp_path / "v.loom"]
    elif kind == "h5ad":
        renamed = a.copy()
        renamed.obs_names = [f"s1:{n.split('-')[0]}x" for n in a.obs_names]
        renamed.write_h5ad(tmp_path / "v.h5ad")
        paths = [tmp_path / "v.h5ad"]
    else:
        paths = write_starsolo(tmp_path / "solo", a, zipped=kind.endswith("gz"))
    rec = vi.attach(dataset, "ds", tmp_path, paths)
    assert rec["cells_matched"] == 70 and rec["cells_without"] == 0 and rec["cells_in_file"] == 80
    assert rec["genes_by_id"] == 70 and set(rec["matrices"]) == set(vi.LAYERS)
    with ms.ModalityStore(vi.store_path(tmp_path, "ds")) as st:
        rows = st.rows_of(dataset.obs_names)
        assert (rows >= 0).all()
        got = st.read("unspliced", rows).toarray()
        assert list(st.features) == list(dataset.var_names)
    assert (got == a.layers["unspliced"][10:].toarray()).all()
    assert not (vi.store_path(tmp_path, "ds").parent / ".velocity-attach").exists()


def test_attach_genes_by_name_when_either_side_has_no_id(tmp_path):
    a = splicing_counts(40, 50)
    write_loom(tmp_path / "v.loom", a)
    dataset = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=pd.DataFrame(index=a.var_names))
    rec = vi.attach(dataset, "ds", tmp_path, [tmp_path / "v.loom"])
    assert rec["genes_by_id"] == 0 and rec["genes_by_name"] == 50


def test_attach_refuses_a_file_of_other_cells_and_changes_nothing(tmp_path):
    a = splicing_counts(40, 50)
    write_loom(tmp_path / "v.loom", a)
    other = ad.AnnData(np.ones((40, 2), dtype=np.float32), obs=pd.DataFrame(index=[barcode(i + 500) + "-1" for i in range(40)]))
    with pytest.raises(vi.VelocityInputError, match="matches only 0 of this dataset's 40 cells"):
        vi.attach(other, "ds", tmp_path, [tmp_path / "v.loom"])
    assert not vi.has_store(tmp_path, "ds") and vi.UNS_KEY not in other.uns
    # Cells match, genes do not.
    named = ad.AnnData(np.ones((40, 2), dtype=np.float32), obs=a.obs.copy(), var=pd.DataFrame(index=["x", "y"]))
    with pytest.raises(vi.VelocityInputError, match="None of"):
        vi.attach(named, "ds", tmp_path, [tmp_path / "v.loom"])
    assert not vi.has_store(tmp_path, "ds")


def test_starsolo_unsorted_matrix_gives_the_same_rows(tmp_path):
    a = splicing_counts(40, 50)
    paths = write_starsolo(tmp_path / "solo", a)
    mtx = tmp_path / "solo" / "spliced.mtx"
    lines = mtx.read_text().splitlines()
    head = [i for i, line in enumerate(lines) if not line.startswith("%")][0] + 1
    body = lines[head:]
    np.random.default_rng(0).shuffle(body)
    mtx.write_text("\n".join(lines[:head] + body) + "\n")
    dataset = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    vi.attach(dataset, "ds", tmp_path, paths)
    with ms.ModalityStore(vi.store_path(tmp_path, "ds")) as st:
        assert (st.read("spliced", st.rows_of(dataset.obs_names)).toarray() == a.layers["spliced"].toarray()).all()


# -- a dataset built apart from the looms: several samples, names in another form ------------------
# Real case: Pollyea 2019 AML (Zenodo 6587903), velocyto CellID "cellsorted_day0_UIB5X:BARCODE"
# against an h5ad of bare barcodes with a "batch" column; 12 barcodes recur across its samples.

_SAMPLES = {"day0": 0, "day2": 50, "day4": 100}  # first barcode of each sample's 60: 50-59 and 100-109 recur


def _sample_runs(n=60, genes=50):
    """Each sample's own counts (one velocyto loom each) with overlapping barcodes."""
    runs = {}
    for k, (name, start) in enumerate(_SAMPLES.items()):
        a = splicing_counts(n, genes, seed=k)
        a.obs_names = [barcode(start + i) + "-1" for i in range(n)]
        runs[name] = a
    return runs


def _merged(runs, keep_every=1, names="bare"):
    """The dataset as an author's h5ad: cells of every sample, bare barcodes (or Cell Ranger
    aggr's -1/-2/-3), a sample column and a QC flag that splits each sample, in another order."""
    parts = []
    for k, (name, a) in enumerate(runs.items()):
        b = a[(k % 2 if keep_every > 1 else 0)::keep_every].copy()  # day2 keeps odd rows: barcodes day0 lost
        code = [n.split("-")[0] for n in b.obs_names]
        b.obs_names = code if names == "bare" else [f"{c}-{k + 1}" for c in code]
        b.obs["batch"] = name
        b.obs["qc_pass"] = np.where(np.arange(b.n_obs) % 10 == 0, "no", "yes")
        parts.append(b)
    for b in parts:
        for k in vi.LAYERS:
            del b.layers[k]
    merged = ad.concat(parts, merge="first")
    merged.obs_names_make_unique()
    order = np.random.default_rng(3).permutation(merged.n_obs)
    return merged[order].copy()


def _truth(runs, dataset):
    """Each dataset cell's own unspliced counts, from its own sample."""
    out = []
    for name, cell in zip(dataset.obs["batch"], dataset.obs_names, strict=True):
        a = runs[name]
        code = cell.split("-")[0]
        out.append(a.layers["unspliced"][[n.split("-")[0] for n in a.obs_names].index(code)].toarray()[0])
    return np.asarray(out)


def test_a_merged_dataset_matches_each_sample_by_its_sample_column():
    runs = _sample_runs()
    dataset = _merged(runs, keep_every=2)
    file_names = [f"cellsorted_{s}_X{k}:{n.split('-')[0]}" for k, (s, a) in enumerate(runs.items()) for n in a.obs_names]
    m = vi.match_cells(file_names, list(dataset.obs_names), vi.sample_columns(dataset.obs))
    assert m.sample_key == "batch" and m.n_matched == dataset.n_obs == 90 and m.n_ambiguous == 0
    assert [(s["file_sample"], s["dataset_sample"], s["cells_matched"]) for s in m.samples] == [
        ("cellsorted day0 X0", "day0", 30), ("cellsorted day2 X1", "day2", 30), ("cellsorted day4 X2", "day4", 30)]
    batch = dict(zip(dataset.obs_names, dataset.obs["batch"], strict=True))
    assert all(batch[d] in file_names[r] for r, d in zip(m.file_rows, m.dataset_names, strict=True))
    # Without the column the names alone cannot tell recurring barcodes apart: left out, counted.
    bare = vi.match_cells(file_names, list(dataset.obs_names))
    assert bare.sample_key is None and bare.n_matched < 90 and bare.n_ambiguous > 0
    # The QC flag splits each sample: never taken for the sample.
    assert "qc_pass" in vi.sample_columns(dataset.obs)
    only_flag = vi.match_cells(file_names, list(dataset.obs_names), {"qc_pass": vi.sample_columns(dataset.obs)["qc_pass"]})
    assert only_flag.sample_key is None


def test_cell_ranger_aggr_suffixes_are_learned_as_samples():
    runs = _sample_runs()
    dataset = _merged(runs, names="aggr")  # BARCODE-1, -2, -3: nothing in common with the loom's sample names
    file_names = [f"run{k}_ABCDE:{n.split('-')[0]}x" for k, a in enumerate(runs.values()) for n in a.obs_names]
    m = vi.match_cells(file_names, list(dataset.obs_names), {})
    assert m.sample_key == "cell names" and m.n_matched == 180 and m.n_ambiguous == 0
    assert [s["dataset_sample"] for s in m.samples] == ["1", "2", "3"]


def test_one_samples_loom_pairs_none_of_another_samples_cells():
    """Old matching paired a barcode found once on each side even across samples: on Pollyea,
    the day0 loom alone gave 58 day2/day4 cells day0 cells' counts."""
    runs = _sample_runs()
    dataset = _merged(runs, keep_every=2)  # day0 keeps even rows, day2 odd: 51-59 are day2's alone
    file_names = [f"cellsorted_day0_X0:{n.split('-')[0]}" for n in runs["day0"].obs_names]
    m = vi.match_cells(file_names, list(dataset.obs_names), vi.sample_columns(dataset.obs))
    batch = dict(zip(dataset.obs_names, dataset.obs["batch"], strict=True))
    assert m.n_matched == 30 and {batch[d] for d in m.dataset_names} == {"day0"}
    assert m.n_other_sample == 5  # 51, 53, ... 59: in the dataset, but day2's cells
    old = vi.match_cells(file_names, list(dataset.obs_names))  # names alone: those 5 paired with day2's cells
    assert old.n_matched == 35 and sum(batch[d] != "day0" for d in old.dataset_names) == 5
    assert m.samples_without == [{"dataset_sample": "day2", "cells": 30}, {"dataset_sample": "day4", "cells": 30}]
    # A third of the dataset, half of the file: refused before, now judged within its own sample (all 30).
    assert m.n_dataset_in_samples == 30
    vi.check_majority(m, "day0.loom")
    with pytest.raises(vi.VelocityInputError, match="matches only 35 of this dataset's 90 cells"):
        vi.check_majority(old, "day0.loom")


def test_attach_one_loom_per_sample_together(tmp_path):
    runs = _sample_runs()
    dataset = _merged(runs, keep_every=2)
    paths = []
    for k, (name, a) in enumerate(runs.items()):
        paths.append(tmp_path / f"stored_{name}.loom")
        write_loom(paths[-1], a, sample=f"cellsorted_{name}_X{k}")
    rec = vi.attach(dataset, "ds", tmp_path, paths, labels=[f"{n}.loom" for n in runs])
    assert rec["file"] == "day0.loom + day2.loom + day4.loom" and rec["kind"] == "loom"
    assert rec["cells_matched"] == 90 and rec["cells_in_file"] == 180 and rec["cells_in_file_unmatched"] == 90
    assert rec["sample_key"] == "batch" and rec["samples_without"] == []
    with ms.ModalityStore(vi.store_path(tmp_path, "ds")) as st:
        got = st.read("unspliced", st.rows_of(dataset.obs_names)).toarray()
    np.testing.assert_array_equal(got, _truth(runs, dataset))
    # Files of other genes are refused before anything changes.
    other = splicing_counts(60, 40)
    write_loom(tmp_path / "other.loom", other, sample="x")
    with pytest.raises(vi.VelocityInputError, match="lists other genes than day0.loom"):
        vi.attach(dataset, "ds2", tmp_path, [paths[0], tmp_path / "other.loom"], labels=["day0.loom", "other.loom"])
    assert not vi.has_store(tmp_path, "ds2")
    # Bare names in several files: each file's name keeps their barcodes apart.
    for k, (name, a) in enumerate(runs.items()):
        b = a.copy()
        b.obs_names = [n.split("-")[0] for n in a.obs_names]
        b.write_h5ad(tmp_path / f"{name}.h5ad")
    rec = vi.attach(dataset, "ds3", tmp_path, [tmp_path / f"{n}.h5ad" for n in runs])
    assert rec["cells_matched"] == 90 and rec["sample_key"] == "batch"
    with ms.ModalityStore(vi.store_path(tmp_path, "ds3")) as st:
        np.testing.assert_array_equal(st.read("unspliced", st.rows_of(dataset.obs_names)).toarray(), _truth(runs, dataset))


# -- the import wizard ---------------------------------------------------------


def _commit(data_dir: Path, path: Path, **opts):
    from scview.core.ingestion import IngestOptions, IngestSessionManager

    mgr = IngestSessionManager(str(data_dir))
    sid = mgr.create_session()
    shutil.copyfile(path, mgr.files_dir(sid) / path.name)
    if opts:
        mgr.set_options(sid, IngestOptions(**opts))
    did = mgr._build_dataset(sid)
    return did, next((data_dir / "ingested" / did).glob("*.h5ad"))


@pytest.mark.parametrize("streamed", ["off", "on"])
def test_wizard_moves_the_layers_of_an_h5ad_read_either_way(tmp_path, streamed):
    a = splicing_counts(60, 70)
    a.layers["other"] = a.layers["spliced"].copy()
    a.write_h5ad(tmp_path / "w.h5ad")
    did, h5 = _commit(tmp_path, tmp_path / "w.h5ad", streamed=streamed)
    back = ad.read_h5ad(h5)
    assert list(back.layers.keys()) == ["other"] and back.n_obs == 60
    assert json.loads(back.uns[vi.UNS_KEY])["source"]["kind"] == "upload"
    with ms.ModalityStore(vi.store_path(tmp_path, did)) as st:
        assert set(st.matrices) == set(vi.LAYERS)
        assert (st.read("unspliced", st.rows_of(back.obs_names)).toarray() == a.layers["unspliced"].toarray()).all()


def test_wizard_reads_a_loom_and_a_legacy_h5ad(tmp_path):
    a = splicing_counts(60, 70)
    write_loom(tmp_path / "v.loom", a)
    did, h5 = _commit(tmp_path, tmp_path / "v.loom")
    back = ad.read_h5ad(h5)
    assert not any(k in back.layers for k in vi.LAYERS)
    with ms.ModalityStore(vi.store_path(tmp_path, did)) as st:
        # The dataset took velocyto's names from the loom: the store is addressed by the same ones.
        assert (st.rows_of(back.obs_names) >= 0).all()
        assert st.feature_attr("gene_id")[0].startswith("ENSG") or st.n_vars == 70

    legacy_h5ad(tmp_path / "old.h5ad", a)
    did, h5 = _commit(tmp_path, tmp_path / "old.h5ad")
    assert "spliced" not in ad.read_h5ad(h5).layers and vi.has_store(tmp_path, did)


def test_wizard_leaves_a_plain_upload_alone(tmp_path):
    a = splicing_counts(60, 70)
    plain = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    plain.layers["counts"] = plain.X.copy()
    plain.write_h5ad(tmp_path / "p.h5ad")
    did, h5 = _commit(tmp_path, tmp_path / "p.h5ad")
    back = ad.read_h5ad(h5)
    assert list(back.layers.keys()) == ["counts"] and vi.UNS_KEY not in back.uns
    assert not ms.store_dir(tmp_path, did).exists()


def test_one_layer_alone_stays_where_it_is(tmp_path):
    a = splicing_counts(60, 70)
    del a.layers["unspliced"]
    a.write_h5ad(tmp_path / "one.h5ad")
    did, h5 = _commit(tmp_path, tmp_path / "one.h5ad", streamed="on")
    assert "spliced" in ad.read_h5ad(h5).layers and not vi.has_store(tmp_path, did)
