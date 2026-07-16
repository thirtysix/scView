"""Endpoint wiring for the enrichment network.

The network maths is covered in test_enrichment_network.py; this asserts the route
delegates to compute-local correctly and shapes its response.
"""

from __future__ import annotations

import pytest

from scview.api.v1.enrichment import (
    EnrichmentNetworkRequest,
    EnrichmentResponse,
    EnrichmentResult,
    enrichment_network,
)


class _DM:
    async def get_or_load_dataset(self, _id):
        return object()


def _result(term, genes, padj, gene_count):
    return EnrichmentResult(
        term=term, pvalue=padj, adjusted_pvalue=padj, overlap_count=len(genes),
        gene_count=gene_count, genes=list(genes), collection="c5.go.bp",
    )


@pytest.fixture()
def stub_compute(monkeypatch):
    """Replace compute-local so the route is tested without gseapy or a dataset."""
    calls = {}

    async def fake(dataset_id, body, dm=None, settings=None):
        calls["body"] = body
        return EnrichmentResponse(
            group=body.group, groupby=body.column, n_genes_used=100,
            source="msigdb_local",
            results=[
                _result("A", ["G1", "G2", "G3"], 1e-9, 50),
                _result("B", ["G1", "G2", "G4"], 1e-8, 60),
                _result("C", ["Z1", "Z2", "Z3"], 1e-7, 70),
                _result("NS", ["G1"], 0.9, 50),
            ],
        )

    monkeypatch.setattr("scview.api.v1.enrichment.compute_enrichment_local", fake)
    return calls


@pytest.mark.asyncio
async def test_returns_network_shape(stub_compute):
    out = await enrichment_network(
        "ds1", EnrichmentNetworkRequest(column="cluster", group="B"), dm=_DM(), settings=None
    )
    assert out["dataset_id"] == "ds1"
    assert out["group"] == "B"
    assert out["groupby"] == "cluster"
    assert out["n_significant"] == 3, "the non-significant term must be excluded"
    assert {n["term"] for n in out["nodes"]} == {"A", "B", "C"}
    assert out["params"]["metric"] == "combined"


@pytest.mark.asyncio
async def test_forwards_enrichment_params_to_compute_local(stub_compute):
    await enrichment_network(
        "ds1",
        EnrichmentNetworkRequest(
            column="cluster", group="B", n_genes=250, collections=["h.all"], resolution=0.5
        ),
        dm=_DM(), settings=None,
    )
    body = stub_compute["body"]
    assert body.n_genes == 250, "n_genes must reach the enrichment, not be defaulted"
    assert body.collections == ["h.all"]


@pytest.mark.asyncio
async def test_overlapping_terms_cluster_together(stub_compute):
    out = await enrichment_network(
        "ds1", EnrichmentNetworkRequest(column="cluster", group="B"), dm=_DM(), settings=None
    )
    by_term = {n["term"]: n["cluster"] for n in out["nodes"]}
    assert by_term["A"] == by_term["B"], "A and B share 2 of 4 genes"
    assert by_term["C"] != by_term["A"], "C shares nothing"


@pytest.mark.asyncio
async def test_edges_included_by_default(stub_compute):
    out = await enrichment_network(
        "ds1", EnrichmentNetworkRequest(column="cluster", group="B"), dm=_DM(), settings=None
    )
    assert out["n_edges"] >= 1
    assert len(out["edges"]) == out["n_edges"]


@pytest.mark.asyncio
async def test_include_edges_false_drops_payload_but_keeps_count(stub_compute):
    # The collapsed view renders clusters only; edges are tens of thousands of rows.
    out = await enrichment_network(
        "ds1",
        EnrichmentNetworkRequest(column="cluster", group="B", include_edges=False),
        dm=_DM(), settings=None,
    )
    assert out["edges"] == []
    assert out["n_edges"] >= 1, "the count must survive so the UI can still report it"
    assert out["n_clusters"] >= 1, "clustering must still use the edges internally"


@pytest.mark.asyncio
async def test_bad_metric_is_422_not_500(stub_compute):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await enrichment_network(
            "ds1",
            EnrichmentNetworkRequest(column="cluster", group="B", metric="nope"),
            dm=_DM(), settings=None,
        )
    assert exc.value.status_code == 422
