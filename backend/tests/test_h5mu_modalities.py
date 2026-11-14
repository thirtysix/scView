"""An .h5mu's protein, hashtag and guide modalities are kept at import (core/ingestion/h5mu_mods.py).

pertpy's papalexi_2021 (rna, adt, hto, gdo) imported as RNA alone with no word (``issues: []``).
Now adt becomes the protein panel, hto the hashing tags, gdo the guide counts; a modality scView
has no place for is named in an import issue; obs columns pulled into MuData's shared table
(``rna:replicate``) or kept per modality come along when the RNA obs lacks them.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import write_elem

from scview.core import crispr, hashing, protein
from scview.core.ingestion import IngestSessionManager, h5mu_mods
from scview.core.ingestion.bundling import build_bundle
from scview.core.ingestion.loading import load_unit
from scview.core.ingestion.validation import validate_bundle

N = 60
CELLS = [f"l1_CELL{i:03d}" for i in range(N)]


def _mod(n_features: int, prefix: str, seed: int, cells=CELLS, obs=None) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sp.csc_matrix(rng.poisson(3.0, (len(cells), n_features)).astype(np.float32))
    return ad.AnnData(X=X, obs=obs.copy() if obs is not None else pd.DataFrame(index=cells),
                      var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_features)]))


def write_papalexi_like(path: Path, *, extra_mod: bool = True, shared_obs: bool = True) -> Path:
    """rna, adt (4), hto (3), gdo (5) and an unknown modality, as mudata writes them."""
    obs = pd.DataFrame({"replicate": pd.Categorical(np.where(np.arange(N) % 2, "rep1", "rep2")),
                        "nCount_RNA": np.arange(N, dtype=float)}, index=CELLS)
    rna = _mod(20, "GENE", 0, obs=obs)
    adt = _mod(4, "AB", 1)
    adt.var_names = ["CD86", "PDL1", "PDL2", "CD366"]
    hto = _mod(3, "rep", 2, cells=CELLS[::-1])  # another row order: aligned by name
    gdo_obs = pd.DataFrame({"guide_ID": [f"g{i % 5}" for i in range(N)]}, index=CELLS)
    gdo = _mod(5, "GUIDEg", 3, obs=gdo_obs)
    shared = pd.DataFrame({"rna:replicate": obs["replicate"].astype(str).to_numpy(),
                           "rna:batch_note": "b1", "gdo:guide_ID": gdo_obs["guide_ID"].to_numpy()},
                          index=CELLS) if shared_obs else pd.DataFrame(index=CELLS)
    with h5py.File(path, "w", userblock_size=512) as f:
        f.attrs["encoding-type"] = "MuData"
        f.attrs["encoding-version"] = "0.1.0"
        write_elem(f, "obs", shared)
        write_elem(f, "var", pd.DataFrame(index=[f"v{i}" for i in range(32 + (7 if extra_mod else 0))]))
        mods = f.create_group("mod")
        write_elem(mods, "rna", rna)
        write_elem(mods, "adt", adt)
        write_elem(mods, "hto", hto)
        write_elem(mods, "gdo", gdo)
        if extra_mod:
            write_elem(mods, "spliced_foo", _mod(7, "F", 4))
    with open(path, "r+b") as fh:
        fh.write(b"MuData (format-version=0.1.0;creator=mudata;creator-version=0.2.0)")
    return path


def test_the_plan_names_what_each_modality_becomes(tmp_path):
    p = write_papalexi_like(tmp_path / "papalexi.h5mu")
    roles = {m.name: m.role for m in h5mu_mods.plan(p)}
    assert roles == {"adt": "protein", "hto": "hashtags", "gdo": "guides", "spliced_foo": None}
    kept, left = h5mu_mods.sentences(p)
    assert "adt (4 features) becomes the protein panel" in kept and "gdo (5 features)" in kept
    assert left.startswith("Not read from this .h5mu: spliced_foo (7 features")


def test_validation_lists_the_modalities_and_the_left_out_one(tmp_path):
    p = write_papalexi_like(tmp_path / "papalexi.h5mu")
    report = validate_bundle(build_bundle([p]))
    codes = {i.code: i for i in report.issues}
    assert report.ok and codes["h5mu_modalities"].severity.value == "info"
    assert "spliced_foo" in codes["h5mu_modality_left_out"].message
    q = write_papalexi_like(tmp_path / "plain.h5mu", extra_mod=False)
    assert "h5mu_modality_left_out" not in {i.code for i in validate_bundle(build_bundle([q])).issues}


def test_the_loaded_dataset_keeps_protein_hashtags_and_guides(tmp_path):
    p = write_papalexi_like(tmp_path / "papalexi.h5mu")
    a = load_unit(build_bundle([p]).units[0])
    assert a.n_vars == 20 and list(a.obs_names) == CELLS
    prot = a.obsm[protein.COUNTS_KEY]
    assert list(prot.columns) == ["CD86", "PDL1", "PDL2", "CD366"] and protein.CLR_KEY in a.obsm
    assert protein.has_protein(a) and prot.shape == (N, 4)
    # Hashtags were written in reverse row order: aligned to the RNA cells by name.
    with h5py.File(p) as f:
        from anndata.io import read_elem

        hto = read_elem(f["mod"]["hto"])
    tags = a.obsm[hashing.HTO_KEY]
    assert tags.shape == (N, 3)
    assert np.array_equal(tags.loc[CELLS[0]].to_numpy(), np.asarray(hto[CELLS[0]].X.toarray()).ravel())
    guides = crispr.guide_counts_frame(a)
    assert guides is not None and guides.shape == (N, 5)
    # obs: the RNA's own columns, plus the guide column the RNA table lacks; rna:replicate is the same.
    assert "replicate" in a.obs and "guide_ID" in a.obs and "batch_note" in a.obs
    assert "rna_replicate" not in a.obs and "gdo_guide_ID" not in a.obs
    assert a.obs["guide_ID"].astype(str).iloc[7] == "g2"


def test_the_commit_writes_a_dataset_with_the_modalities(tmp_path):
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    write_papalexi_like(m.files_dir(sid) / "papalexi.h5mu")
    did = m._build_dataset(sid)
    a = ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))
    assert protein.has_protein(a) and hashing.HTO_KEY in a.obsm and crispr.guide_counts_frame(a) is not None
