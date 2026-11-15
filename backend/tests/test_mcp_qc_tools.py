"""MCP QC tools (v0.80, QA round 6 on Reck 2025's kidney nuclei): every filtering parameter
can be previewed (the nuclei warning's pct_mt_floor), pooled libraries are compared per donor
x library, and get_markers leaves out genes with an id but no name, as the marker choice does."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_library_concordance import _pooled_design
from tests.test_mcp import GENES, _adata, _call, msigdb_dir, test_settings  # noqa: F401 — fixtures

ID, CLONE = "ENSG00000236283", "AC012345.1"


def _register(dm, a, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _named(a):
    """A copy (``_adata`` is cached) whose type A's two top genes get an Ensembl id and a clone
    name, and three genes are mitochondrial."""
    a = a.copy()
    names = list(GENES)
    names[0], names[1] = ID, CLONE
    names[-3:] = ["MT-CO1", "MT-ND1", "MT-CYB"]
    a.var_names = names
    return a


async def test_qc_filter_preview_takes_every_filtering_parameter(client, dataset_manager):
    did = _register(dataset_manager, _named(_adata()), "mcpqc")
    base = await _call("qc_filter_preview", {"dataset_id": did, "sample_column": "donor_id"})
    assert base["has_mt"] is True and base["thresholds"]["pct_mt_floor"] == 2.0  # the default floor
    r = await _call("qc_filter_preview", {
        "dataset_id": did, "sample_column": "donor_id", "pct_mt_floor": 7.5, "mad_upper": True,
        "counts_floor": 5, "max_counts": 1e6, "mito_mode": "per_type", "mito_type_column": "cluster",
        "mito_type_floor": True, "doublet_source": "auto", "drop_hash_negatives": False,
        "min_atac_counts": 0, "max_atac_counts": 0, "max_nucleosome_signal": 0, "min_tss_enrichment": 0})
    t = r["thresholds"]
    assert (t["pct_mt_floor"], t["mad_upper"], t["counts_floor"], t["max_counts"]) == (7.5, True, 5.0, 1e6)
    assert (t["mito_mode"], t["mito_type_column"], t["mito_type_floor"]) == ("per_type", "cluster", True)
    # Every filtering parameter of the step has a preview argument (named without qc_).
    from scview.mcp.server import SERVER
    from mcp import Client

    async with Client(SERVER) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    args = set(tools["qc_filter_preview"].input_schema["properties"])
    step = {"min_genes", "max_genes", "min_counts", "max_counts", "max_pct_mt", "drop_doublets",
            "qc_threshold_mode", "qc_mad_k", "qc_mad_k_mt", "qc_mad_upper", "qc_sample_column", "qc_counts_floor",
            "qc_genes_floor", "qc_pct_mt_floor", "qc_grid_mito", "qc_mito_mode", "qc_mito_type_column",
            "qc_mito_type_floor", "qc_doublet_source", "drop_hash_negatives", "qc_min_atac_counts",
            "qc_max_atac_counts", "qc_max_nucleosome_signal", "qc_min_tss_enrichment"}
    alias = {"qc_threshold_mode": "mode", "qc_sample_column": "sample_column"}
    assert {alias.get(p, p.removeprefix("qc_")) for p in step} <= args


async def test_library_concordance_reads_pooled_libraries(client, dataset_manager):
    a, _ = _pooled_design()
    did = _register(dataset_manager, a, "mcppool")
    out = await _call("library_concordance", {"dataset_id": did, "library_column": "library_id"})
    assert out["status"] == "checked" and out["pooled"] is True and out["flagged"] == ["D0"]
    assert out["note"].startswith("'library_id' pools several samples per library")


async def test_get_markers_leaves_out_genes_without_a_name(client, dataset_manager):
    did = _register(dataset_manager, _named(_adata()), "mcpmk")
    # A's ten markers (G0-G9) hold both: the top ten without them reach past G9.
    r = await _call("get_markers", {"dataset_id": did, "column": "cluster", "n_genes": 10})
    genes_a = [row[0] for row in r["markers"]["A"]]
    assert len(genes_a) == 10 and ID not in genes_a and CLONE not in genes_a
    assert set(r["unnamed_skipped"]["A"]) == {ID, CLONE} and "include_unnamed=true" in r["unnamed_note"]
    r = await _call("get_markers", {"dataset_id": did, "column": "cluster", "n_genes": 10, "include_unnamed": True})
    assert {ID, CLONE} <= {row[0] for row in r["markers"]["A"]} and "unnamed_skipped" not in r
