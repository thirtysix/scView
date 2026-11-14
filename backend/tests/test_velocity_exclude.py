"""Genes a velocity run leaves out (core/velocity.py ``resolve_exclusion``, core/velocity_gene_sets.py):
matching, the run, its record, the replay script and the API."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from scview.core import provenance
from scview.core import velocity as vel
from scview.core import velocity_gene_sets as vgs
from scview.core import velocity_input as vi
from tests.test_velocity import API, _run, _sse, dataset_with_store
from tests.test_velocity_replay import _replay_namespace
from tests.velocity_fixture import Adaptor, splicing_counts


def test_murk_list_is_barile_table_s2():
    genes = vgs.MURK_BARILE_2021
    assert len(genes) == len(set(genes)) == 89
    assert genes[:3] == ("Hba-x", "Smim1", "Dcxr") and "Slc25a37" in genes and "Klf1" in genes
    # The supplement as downloaded: a header, then one gene per row with its slope change.
    assert len(vgs.SOURCE_SHA256) == 64 and vgs.SOURCE_URL.endswith("13059_2021_2414_MOESM3_ESM.csv")
    d = vgs.describe()[0]
    assert d["id"] == "murk_barile2021" and d["n_genes"] == 89 and d["species"] == "mouse"


def test_names_match_as_written_through_a_symbol_column_or_ignoring_case():
    features = ["ENSMUSG01", "ENSMUSG02", "Gata1", "KLF1"]
    var = pd.DataFrame({"gene_symbols": ["Hba-x", "Smim1", "Gata1", "Klf1"]}, index=features)
    rows, rec = vel.resolve_exclusion(features, var, ["Gata1", "Smim1", "klf1", "Nope", ""], [])
    assert rows.tolist() == [1, 2, 3]
    assert rec["n_given"] == 4 and rec["n_matched"] == 3 and rec["unmatched"] == ["Nope"]
    assert rec["genes"] == ["ENSMUSG02", "Gata1", "KLF1"]
    rows, rec = vel.resolve_exclusion(features, var, [], ["murk_barile2021"])
    assert rows.tolist() == [0, 1, 3] and rec["sets"] == ["murk_barile2021"] and rec["n_unmatched"] == 86  # Klf1 too
    assert vel.resolve_exclusion(features, None, [], [])[1] == {}
    with pytest.raises(vel.VelocityError, match="Unknown gene list"):
        vel.resolve_exclusion(features, None, [], ["nope"])
    with pytest.raises(vel.VelocityError, match="at most"):
        vel.resolve_exclusion(features, None, [f"g{i}" for i in range(vel.MAX_EXCLUDE + 1)], [])


def _run_with(tmp_path, adata, **kw):
    params = vel.Params(n_top_genes=200, n_neighbors=20, n_pcs=10, **kw)
    p = vel.prepare(Adaptor(adata), tmp_path, "ds", params)
    events = list(vel.iter_run(p))
    return p, events, next(ev["out"] for ev in events if ev["type"] == "result")


def test_left_out_genes_are_not_fitted_and_are_recorded(tmp_path):
    adata = splicing_counts()
    vi.adopt_anndata(adata, "ds", tmp_path, file_name="x.h5ad")
    drop = ["G0", "G1", "G2", "G255", "Nope"]  # G255 fails the shared-counts filter anyway
    p, events, out = _run_with(tmp_path, adata, exclude_genes=tuple(drop))
    assert not set(drop) & set(out["genes"]) and out["n_excluded_kept"] == 3
    assert any("4 genes left out (3 of them pass" in ev.get("message", "") for ev in events)
    run = vel.save(adata, p, out, {"result_id": "x1"})
    assert run["params"]["exclude_genes"] == ["G0", "G1", "G2", "G255"] and run["params"]["exclude_sets"] == []
    assert run["params"]["exclude_given"] == drop
    lines, _ = vel.grounding_lines(Adaptor(adata))
    assert any("Left out 4 genes before the fit" in x for x in lines)
    assert run["excluded"]["n_matched"] == 4 and run["excluded"]["unmatched"] == ["Nope"]
    assert run["excluded"]["n_passing_filter"] == 3 and "genes" not in run["excluded"]
    step = provenance.read_provenance(adata)["history"][-1]
    assert step["step"] == "velocity" and step["params"]["exclude_genes"] == ["G0", "G1", "G2", "G255"]


def test_leaving_genes_out_is_the_same_as_counts_without_them(tmp_path):
    """Barile's fix by hand (QA round 5): counts written without the genes, attached, run. Each
    cell's total leaves them out too, so the scaling matches."""
    full = splicing_counts()
    drop = ["G0", "G1", "G2", "G7"]
    vi.adopt_anndata(full, "ds", tmp_path / "a", file_name="x.h5ad")
    _p, _e, out = _run_with(tmp_path / "a", full, exclude_genes=tuple(drop))
    trimmed = splicing_counts()[:, [g not in drop for g in full.var_names]].copy()
    vi.adopt_anndata(trimmed, "ds", tmp_path / "b", file_name="x.h5ad")
    _p, _e, ref = _run_with(tmp_path / "b", trimmed)
    assert out["genes"] == ref["genes"]
    assert np.allclose(out["confidence"], ref["confidence"], atol=1e-6)
    assert np.allclose(out["unspliced_fraction"], ref["unspliced_fraction"])


def test_a_run_without_exclusion_records_none(tmp_path):
    adata = splicing_counts()
    vi.adopt_anndata(adata, "ds", tmp_path, file_name="x.h5ad")
    p, _events, out = _run_with(tmp_path, adata)
    run = vel.save(adata, p, out, {"result_id": "x2"})
    assert run["excluded"] is None and run["params"]["exclude_genes"] == []


def test_replay_leaves_out_the_same_genes(tmp_path):
    from scview.core import replay_script

    a = splicing_counts()
    a.write_h5ad(tmp_path / "up.h5ad")
    adata = a.copy()
    vi.adopt_anndata(adata, "ds", tmp_path / "s", file_name="x.h5ad")
    _p, _e, out = _run_with(tmp_path / "s", adata, exclude_genes=("G3", "G4", "G5"))
    rep = a.copy()
    for k in vi.LAYERS:
        del rep.layers[k]
    ns = _replay_namespace(str(tmp_path / "up.h5ad"))
    ns["velocity_run"](rep, None, "stochastic", 200, 20, 10, 20, exclude=["G3", "G4", "G5"])
    assert np.allclose(rep.obs[vel.OBS_CONFIDENCE], out["confidence"], atol=1e-5)
    assert np.isnan(rep.var.loc["G3", vel.VAR_GAMMA])
    snip = replay_script._velocity({"method": "stochastic", "n_top_genes": 200, "n_neighbors": 20, "n_pcs": 10,
                                    "exclude_genes": ["G3", "G4"], "exclude_sets": ["murk_barile2021"]},
                                   replay_script._Ctx(adata=None))
    code = "\n".join(snip.code)
    assert "exclude=['G3', 'G4']" in code and "2 genes left out (lists: murk_barile2021)" in code
    compile(code, "s", "exec")
    plain = "\n".join(replay_script._velocity({"method": "stochastic", "n_top_genes": 200, "n_neighbors": 20, "n_pcs": 10},
                                              replay_script._Ctx(adata=None)).code)
    assert "exclude" not in plain


async def test_api_takes_genes_and_the_murk_list(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "velx")
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert [g["id"] for g in s["gene_sets"]] == ["murk_barile2021"] and s["gene_sets"][0]["n_genes"] == 89
    events = await _run(client, did, exclude_genes=["G0", "G1"], exclude_sets=["murk_barile2021"])
    run = dict(events)["result"]["run"]
    assert run["params"]["exclude_genes"] == ["G0", "G1"] and run["params"]["exclude_sets"] == ["murk_barile2021"]
    assert run["excluded"]["n_matched"] == 2 and run["excluded"]["n_unmatched"] == 89
    genes = (await client.get(f"{API}/datasets/{did}/velocity/genes")).json()["genes"]
    assert not {"G0", "G1"} & {g["gene"] for g in genes}
    r = await client.post(f"{API}/datasets/{did}/velocity/run-stream", json={"exclude_sets": ["nope"]})
    assert r.status_code == 422
    r = await client.post(f"{API}/datasets/{did}/velocity/run-stream",
                          json={"n_top_genes": 200, "n_neighbors": 20, "exclude_genes": [f"G{i}" for i in range(255)]})
    assert _sse(r.text)[-1][0] == "error" and "too few to fit" in _sse(r.text)[-1][1]["error"]


def test_the_list_is_pinned():
    """The supplement's first column in its order (the hash in velocity_gene_sets is of the whole
    downloaded file): an edit that keeps 89 names still fails here."""
    assert hashlib.sha256(",".join(vgs.MURK_BARILE_2021).encode()).hexdigest() == (
        "e0ddb7efca1a4bcb1041c9d156356abe6dfb2fd23bae5e6e14c6334c9a41f336")
