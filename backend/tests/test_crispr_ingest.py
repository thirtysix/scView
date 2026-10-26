"""CRISPR guides in the import flow: Cell Ranger's protospacer calls detected, kept
from ``crispr_analysis/`` beside their sample's matrix, and joined to its cells;
guide counts kept through a merge."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import crispr
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import FileRole, SideKind, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from tests.test_ingestion_session import _barcode, _stage_h5ad
from tests.test_vdj_ingest import _mex

HEADER = "cell_barcode,num_features,feature_call,num_umis\n"


def _calls(path: Path, rows: list[tuple[int, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(f"{_barcode(i)},{len(c.split('|'))},{c},{u}\n" for i, c, u in rows))
    return path


def test_calls_are_detected(tmp_path):
    p = _calls(tmp_path / "protospacer_calls_per_cell.csv", [(0, "CD86g1", "24")])
    det = detect_file(p)
    assert det.kind == FileKind.crispr_calls and "guide calls" in det.reason


def test_calls_in_crispr_analysis_bundle_with_their_sample(tmp_path):
    outs = tmp_path / "pbmc" / "outs"
    (outs / "filtered_feature_bc_matrix").mkdir(parents=True)
    files = _mex(outs / "filtered_feature_bc_matrix", "")
    calls = _calls(outs / "crispr_analysis" / "protospacer_calls_per_cell.csv", [(0, "NTg1", "9")])
    summary = outs / "crispr_analysis" / "protospacer_umi_thresholds.csv"
    summary.write_text("Protospacer,UMI threshold\nNTg1,3\n")
    bundle = build_bundle(files + [calls, summary], root=tmp_path)
    assert bundle.complete and len(bundle.units) == 1
    side = bundle.units[0].side_files
    assert [(f.name, f.side_kind, f.role) for f in side] == [(calls.name, SideKind.crispr, FileRole.side)]
    # The rest of Cell Ranger's crispr_analysis stays skipped.
    assert [s.path for s in bundle.skipped] == ["pbmc/outs/crispr_analysis/protospacer_umi_thresholds.csv"]


def test_calls_without_a_sample_to_join_are_an_issue(tmp_path):
    calls = _calls(tmp_path / "protospacer_calls_per_cell.csv", [(0, "NTg1", "9")])
    two = build_bundle(_mex(tmp_path, "GSM1_") + _mex(tmp_path, "GSM2_") + [calls])
    assert any("guide calls" in i for i in two.issues)
    assert not [f for u in two.units for f in u.side_files]


@pytest.fixture()
def mgr(tmp_path: Path) -> IngestSessionManager:
    for sub in ("uploads", "converted", "cache"):
        (tmp_path / sub).mkdir()
    return IngestSessionManager(data_dir=str(tmp_path))


async def test_commit_joins_the_calls_to_the_cells(mgr, tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(data_dir=str(tmp_path))
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, fname="screen.h5ad", n_obs=4)
    h5 = mgr.files_dir(sid) / "screen.h5ad"
    a = ad.read_h5ad(h5)
    a.obs_names = [_barcode(i) for i in range(4)]
    a.write_h5ad(h5)
    _calls(mgr.files_dir(sid) / "protospacer_calls_per_cell.csv",
           [(0, "CD86g1", "24"), (1, "CD86g1|NTg2", "3|10"), (3, "NTg1", "40")])
    did = await mgr.commit(sid, dm)
    out = dm.get_dataset(did).adata
    assert out.obs[crispr.CR_CALL].tolist() == ["CD86g1", "CD86g1|NTg2", "", "NTg1"]
    assert out.obs[crispr.CR_N].tolist() == [1, 2, 0, 1]
    p = crispr.preview(out)
    assert p["source"] == "cellranger" and p["counts"] == {"assigned": 2, "control": 1, "multiple": 1, "none": 1}
    from scview.core import seq_metrics

    assert seq_metrics.read(out) is None  # not a sequencing-QC side file


def test_merge_keeps_every_guide():
    a1 = ad.AnnData(X=np.ones((2, 1), dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    a2 = ad.AnnData(X=np.ones((3, 1), dtype=np.float32), obs=pd.DataFrame(index=["c", "d", "e"]))
    a3 = ad.AnnData(X=np.ones((1, 1), dtype=np.float32), obs=pd.DataFrame(index=["f"]))
    crispr.attach_counts(a1, np.array([[1, 2], [3, 4]]), ["g1", "g2"])
    crispr.attach_counts(a2, np.array([[5], [6], [7]]), ["g3"])
    crispr.common_guides([a1, a2, a3])
    for a in (a1, a2, a3):
        assert list(a.obsm[crispr.GUIDE_KEY].columns) == ["g1", "g2", "g3"]
    assert a2.obsm[crispr.GUIDE_KEY]["g3"].tolist() == [5, 6, 7] and a3.obsm[crispr.GUIDE_KEY].sum().sum() == 0
    merged = ad.concat({"s1": a1, "s2": a2, "s3": a3}, label="sample", index_unique=":")
    assert merged.obsm[crispr.GUIDE_KEY].shape == (6, 3)
