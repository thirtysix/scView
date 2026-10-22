"""The plain .h5ad upload splits feature types like the import wizard: antibody
(CITE-seq) rows leave X for obsm["protein_counts"], CellPlex tags go to
obsm["hto_counts"], other types are dropped, and the uploaded file stays
byte-identical. Expected values come from slicing the fixture's own matrix with
scipy and from the CLR formula written out in numpy, not from the split code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import hashing, protein, provenance, replay_script
from scview.core.ingestion import feature_split

API = "/api/v1"

GENES = [f"GENE{i}" for i in range(12)]
ABS = ["CD3_TotalSeqB", "CD4_TotalSeqB", "Mouse_IgG1_isotype_Ctrl_TotalSeqB"]
CMOS = ["CMO301", "CMO302"]
GUIDES = ["sgRNA_A"]
# Interleaved, as Cell Ranger never guarantees order: genes, then antibodies, a guide, tags, genes.
ORDER = GENES[:6] + ABS + GUIDES + CMOS + GENES[6:]
TYPES = (["Gene Expression"] * 6 + ["Antibody Capture"] * 3 + ["CRISPR Guide Capture"]
         + ["Multiplexing Capture"] * 2 + ["Gene Expression"] * 6)


def _cite(n_cells=50, fmt="csr", seed=0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    dense = rng.poisson(1.5, size=(n_cells, len(ORDER))).astype(np.float32)
    dense[:, len(GENES[:6]):len(GENES[:6]) + 3] += rng.poisson(40, size=(n_cells, 3))  # antibodies are deep
    X = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](dense)
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}-1" for i in range(n_cells)]),
                   var=pd.DataFrame({"gene_ids": [f"ID{i}" for i in range(len(ORDER))],
                                     "feature_types": pd.Categorical(TYPES)}, index=ORDER))
    a.layers["counts"] = X.copy()
    a.varm["PCs"] = rng.normal(size=(len(ORDER), 2))
    a.obsm["X_umap"] = rng.normal(size=(n_cells, 2))
    a.obs["donor"] = pd.Categorical(["d1", "d2"] * (n_cells // 2))
    a.uns["note"] = "from the lab"
    a.raw = a.copy()
    return a


def _cols(names: list[str]) -> list[int]:
    return [ORDER.index(n) for n in names]


def _clr(x: np.ndarray) -> np.ndarray:
    """Seurat CLR, margin = 2: log1p(x / exp(mean(log1p(x)))) per cell."""
    x = x.astype(np.float64)
    return np.log1p(x / np.exp(np.log1p(x).mean(axis=1, keepdims=True)))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_split_matches_the_matrix_and_the_wizard(tmp_path, fmt):
    a = _cite(fmt=fmt)
    src = tmp_path / "cite.h5ad"
    a.write_h5ad(src, compression="gzip")
    before = _sha(src)
    assert feature_split.needs_split(src)
    out = tmp_path / "split.h5ad"
    summary = feature_split.split_h5ad(src, out, original_filename="cite.h5ad")
    assert _sha(src) == before
    assert summary["n_genes"] == 12 and summary["n_antibodies"] == 3 and summary["n_multiplexing_tags"] == 2
    assert summary["other_features"] == {"CRISPR Guide Capture": 1}

    b = ad.read_h5ad(out)
    dense = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    gene_cols = _cols(GENES)
    assert list(b.var_names) == GENES and list(b.obs_names) == list(a.obs_names)
    got = b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X)
    np.testing.assert_array_equal(got, dense[:, gene_cols])
    assert type(b.X) is type(a.X) or (fmt == "dense" and isinstance(b.X, np.ndarray))
    np.testing.assert_array_equal(np.asarray(b.layers["counts"].todense() if sp.issparse(b.layers["counts"])
                                             else b.layers["counts"]), dense[:, gene_cols])
    np.testing.assert_array_equal(b.varm["PCs"], a.varm["PCs"][gene_cols])
    assert list(b.raw.var_names) == GENES  # .raw said which of its features are genes
    np.testing.assert_array_equal(b.obsm["X_umap"], a.obsm["X_umap"])
    assert b.uns["note"] == "from the lab" and list(b.obs["donor"]) == list(a.obs["donor"])

    pc = b.obsm[protein.COUNTS_KEY]
    assert list(pc.columns) == ["CD3", "CD4", "Mouse_IgG1_isotype_Ctrl"]
    np.testing.assert_array_equal(pc.to_numpy(), dense[:, _cols(ABS)])
    np.testing.assert_allclose(b.obsm[protein.CLR_KEY].to_numpy(), _clr(dense[:, _cols(ABS)]), rtol=1e-6)
    hto = b.obsm[hashing.HTO_KEY]
    assert list(hto.columns) == CMOS
    np.testing.assert_array_equal(hto.to_numpy(), dense[:, _cols(CMOS)])
    info = b.uns[protein.UNS_KEY]
    assert list(info["controls"]) == ["Mouse_IgG1_isotype_Ctrl"]

    # The import wizard's split of the same data gives the same dataset.
    w, _ = protein.split_features(ad.read_h5ad(src))
    np.testing.assert_array_equal(got, w.X.toarray() if sp.issparse(w.X) else np.asarray(w.X))
    for key in (protein.COUNTS_KEY, protein.CLR_KEY, hashing.HTO_KEY):
        pd.testing.assert_frame_equal(b.obsm[key], w.obsm[key])

    (step,) = provenance.recipe(b)
    assert step["step"] == "ingest" and step["params"]["feature_split"]["n_antibodies"] == 3
    assert provenance.read_provenance(b)["source"]["original_filename"] == "cite.h5ad"


def test_no_split_without_other_feature_types(tmp_path):
    plain = tmp_path / "plain.h5ad"
    ad.AnnData(np.ones((3, 2), dtype=np.float32)).write_h5ad(plain)
    genes_only = tmp_path / "genes.h5ad"
    a = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    a.var["feature_types"] = "Gene Expression"
    a.write_h5ad(genes_only)
    assert not feature_split.needs_split(plain) and not feature_split.needs_split(genes_only)


def test_a_layout_it_cannot_stream_is_refused(tmp_path):
    a = _cite()
    a.layers["frame"] = sp.csr_matrix(a.X)
    src = tmp_path / "x.h5ad"
    a.write_h5ad(src)
    import h5py

    with h5py.File(src, "a") as f:  # a layer stored in an encoding the split does not read
        f["layers"]["frame"].attrs["encoding-type"] = "awkward-array"
    with pytest.raises(feature_split.SplitUnsupported, match="awkward"):
        feature_split.split_h5ad(src, tmp_path / "o.h5ad", original_filename="x.h5ad")


def test_replay_snippet_keeps_the_genes():
    snip = replay_script._ingest({"formats": ["h5ad"], "feature_split": {"n_antibodies": 3}}, None)
    assert snip.status == replay_script.APPROXIMATE
    adata = _cite()
    ns = {"adata": adata}
    exec("\n".join(snip.code), ns)  # noqa: S102 - the script's own lines
    assert list(ns["adata"].var_names) == GENES


# ---------------------------------------------------------------------------
# Through POST /datasets/upload
# ---------------------------------------------------------------------------


async def test_upload_splits_and_keeps_the_original_byte_identical(client, dataset_manager, tmp_data_dir, tmp_path):
    a = _cite(n_cells=60)
    src = tmp_path / "cite_pbmc.h5ad"
    a.write_h5ad(src, compression="gzip")
    sent = src.read_bytes()
    r = await client.post(f"{API}/datasets/upload",
                          files={"file": ("cite_pbmc.h5ad", sent, "application/octet-stream")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    stored = tmp_data_dir / "uploads" / did / "cite_pbmc.h5ad"
    assert stored.read_bytes() == sent  # the upload, as sent
    working = tmp_data_dir / "ingested" / did / "cite_pbmc.h5ad"
    assert working.exists() and dataset_manager.h5ad_candidates(did)[0] == str(working)

    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["n_genes"] == 12 and info["protein"]["names"] == ["CD3", "CD4", "Mouse_IgG1_isotype_Ctrl"]
    meta = json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())
    assert meta["feature_split"]["n_antibodies"] == 3
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert list(adata.var_names) == GENES and hashing.HTO_KEY in adata.obsm
    np.testing.assert_array_equal(adata.obsm[protein.COUNTS_KEY].to_numpy(), a.X.toarray()[:, _cols(ABS)])

    # Deleting the dataset removes its working copy too.
    assert (await client.delete(f"{API}/datasets/{did}")).status_code in (200, 204)
    assert not (tmp_data_dir / "ingested" / did).exists()


async def test_upload_without_feature_types_is_unchanged(client, tmp_data_dir, tmp_path):
    src = tmp_path / "plain.h5ad"
    ad.AnnData(np.ones((4, 3), dtype=np.float32)).write_h5ad(src)
    r = await client.post(f"{API}/datasets/upload", files={"file": ("plain.h5ad", src.read_bytes(), "x")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    assert not (tmp_data_dir / "ingested" / did).exists()
    assert "feature_split" not in json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())


async def test_a_split_that_fails_leaves_the_upload_as_it_was(client, tmp_data_dir, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise feature_split.SplitUnsupported("X is stored as 'awkward-array'")

    monkeypatch.setattr(feature_split, "split_h5ad", boom)
    src = tmp_path / "cite.h5ad"
    _cite().write_h5ad(src)
    r = await client.post(f"{API}/datasets/upload", files={"file": ("cite.h5ad", src.read_bytes(), "x")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    assert not (tmp_data_dir / "ingested" / did).exists()
    meta = json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())
    assert "stay in the expression matrix" in meta["feature_split"]["error"]
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["n_genes"] == len(ORDER)


async def test_a_dataset_uploaded_before_is_not_split_on_open(dataset_manager, tmp_data_dir, tmp_path):
    # Uploaded before this change: uploads/ only. Opening it does not split it silently.
    did = "0123456789abcdef0123456789abcdef"
    up = tmp_data_dir / "uploads" / did
    up.mkdir(parents=True)
    _cite().write_h5ad(up / "old.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": "old", "filename": "old.h5ad",
                                                  "status": "ready"}))
    adaptor = await dataset_manager.load_dataset(did)
    assert adaptor.adata.n_vars == len(ORDER) and not (tmp_data_dir / "ingested" / did).exists()
