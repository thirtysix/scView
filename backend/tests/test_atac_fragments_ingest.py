"""ATAC fragments in the import flow: detected by name, kept as a side file, renamed into the
new dataset's modality folder, refused with the numbers when they are another sample's."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac_fragments as af
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import SideKind, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.session import IngestCommitError
from tests.fragments_fixture import write_fragments

CELLS = [f"BC{i:03d}-1" for i in range(40)]


def _h5ad(path: Path, cells=None) -> Path:
    cells = cells or CELLS
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (len(cells), 30)).astype(np.float32)),
                   obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=[f"G{i}" for i in range(30)]))
    a.write_h5ad(path)
    return path


def _fragments(folder: Path, name: str = "atac_fragments.tsv.gz", barcode_of=lambda i: CELLS[i % 40]):
    lines = [(c, i * 7, i * 7 + 60, barcode_of(i), 1) for c in ("chr1", "chr22") for i in range(300)]
    return write_fragments(folder / name, lines)


def test_detection_by_name_and_magic(tmp_path):
    frag, idx = _fragments(tmp_path)
    found = detect_file(frag)
    assert found.kind == FileKind.atac_fragments and found.details["part"] == "fragments"
    assert detect_file(idx).details["part"] == "index"
    other = tmp_path / "other.tsv.gz"
    other.write_bytes(frag.read_bytes())
    assert detect_file(other).kind != FileKind.atac_fragments


def test_bundling_attaches_the_pair_to_the_only_sample(tmp_path):
    h5 = _h5ad(tmp_path / "pbmc.h5ad")
    frag, idx = _fragments(tmp_path)
    bundle = build_bundle([h5, frag, idx])
    side = bundle.units[0].side_files
    assert {f.spatial_part for f in side} == {"fragments", "index"} and {f.side_kind for f in side} == {SideKind.atac}
    assert not bundle.issues


def test_bundling_says_what_is_missing_or_cannot_be_attached(tmp_path):
    h5 = _h5ad(tmp_path / "pbmc.h5ad")
    frag, idx = _fragments(tmp_path)
    no_index = build_bundle([h5, frag])
    assert len(no_index.units[0].side_files) == 1 and any("Cell Ranger ARC" in i for i in no_index.issues)
    alone = build_bundle([frag, idx])
    assert any("sample's matrix" in i for i in alone.issues)
    two = build_bundle([h5, _h5ad(tmp_path / "other.h5ad"), frag, idx])
    assert any("one sample at a time" in i for i in two.issues) and not any(u.side_files for u in two.units)
    index_only = build_bundle([h5, idx])
    assert any("expected one ATAC fragments file" in i for i in index_only.issues)


@pytest.fixture()
def store(tmp_path) -> tuple[IngestSessionManager, DatasetManager]:
    data = tmp_path / "data"
    return IngestSessionManager(data_dir=str(data)), DatasetManager(data_dir=str(data))


def _session(store: IngestSessionManager, frag_barcode=lambda i: CELLS[i % 40]) -> str:
    sid = store.create_session()
    folder = store.files_dir(sid)
    _h5ad(folder / "pbmc.h5ad")
    _fragments(folder, barcode_of=frag_barcode)
    return sid


async def test_commit_renames_the_files_into_the_dataset(store):
    sessions, dm = store
    sid = _session(sessions)
    inode = (sessions.files_dir(sid) / "atac_fragments.tsv.gz").stat().st_ino
    did = await sessions.commit(sid, dm)
    assert af.fragments_path(sessions.data_dir, did).stat().st_ino == inode  # a rename, not a copy
    st = af.status(sessions.data_dir, did)
    assert st["attached"] and st["barcode_share"] == 1.0 and st["name"] == "atac_fragments.tsv.gz"
    assert st["genome"] == "GRCh38"
    uploads = Path(sessions.data_dir) / "uploads" / did
    assert not list(uploads.rglob("atac_fragments*"))  # the originals do not keep a second copy


async def test_commit_refuses_another_samples_file_and_keeps_it_staged(store):
    sessions, dm = store
    sid = _session(sessions, frag_barcode=lambda i: f"ZZ{i % 40:03d}-1")
    with pytest.raises(IngestCommitError, match="0 of the first 600 fragments"):
        await sessions.commit(sid, dm)
    assert (sessions.files_dir(sid) / "atac_fragments.tsv.gz").exists()
