"""Two or more libraries of one sample (core/library_concordance.py, GET /qc/libraries):
a library holding another sample's cells is flagged against the between-sample spread,
random halves of one sample are not, and pooled lanes, one library per sample, a sample
split by tissue, small libraries, too few samples and libraries different by design
each say why nothing was flagged."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import library_concordance as lc

API = "/api/v1"
TYPES = ["A", "B", "C", "D"]
BASE = {"ctrl": [0.4, 0.3, 0.2, 0.1], "stim": [0.25, 0.3, 0.2, 0.25]}
PLANT = [0.05, 0.15, 0.2, 0.6]  # a stim-like mix, D-rich


def _design(n_donors: int = 8, *, plant: bool = False, cells: int = 900, seed: int = 0,
            two_libs: tuple[int, ...] = (0,), lib_cells: int = 600, tissues: bool = False,
            genes: int = 0) -> tuple[ad.AnnData, dict]:
    """Donors D0.. (half ctrl, half stim), one library each, except the donors in
    ``two_libs``: two libraries ``L<d>a`` / ``L<d>b`` of ``lib_cells`` each from the
    donor's own mix (``plant``: the first one's second library from ``PLANT``).
    ``tissues``: those donors' two libraries are fat and liver instead, one each."""
    rng = np.random.default_rng(seed)
    obs, depth = [], []
    for d in range(n_donors):
        cond = "ctrl" if d % 2 == 0 else "stim"
        mix = rng.dirichlet(np.array(BASE[cond]) * 200)
        libs = [(f"L{d}a", lib_cells), (f"L{d}b", lib_cells)] if d in two_libs else [(f"L{d}", cells)]
        for j, (lib, n) in enumerate(libs):
            p = PLANT if (plant and d == two_libs[0] and j == 1) else mix
            types = rng.choice(TYPES, size=n, p=p)
            tissue = ("liver" if j == 1 else "fat") if (tissues and d in two_libs) else "fat"
            for t in types:
                obs.append({"donor": f"D{d}", "library_id": lib, "condition": cond,
                            "cell_type": t, "tissue": tissue})
            depth.append(rng.lognormal(np.log(3000), 0.3, n))
    frame = pd.DataFrame(obs)
    for c in frame.columns:
        frame[c] = frame[c].astype("category")
    frame.index = [f"c{i}" for i in range(len(frame))]
    total = np.concatenate(depth)
    X = sp.csr_matrix(rng.poisson(5, (len(frame), genes)).astype(np.float32)) if genes else \
        sp.csr_matrix((len(frame), 1), dtype=np.float32)
    a = ad.AnnData(X, obs=frame)
    if not genes:
        a.var_names = ["G0"]
    else:
        a.var_names = [f"G{i}" for i in range(genes)]
    cells = {"total_counts": total, "n_genes": total * 0.4, "pct_mt": np.zeros(len(total)), "has_mt": True}
    return a, cells


def test_a_library_with_another_mix_is_flagged_against_the_between_sample_spread():
    a, cells = _design(plant=True)
    r = lc.library_concordance(a, cells)
    assert r["status"] == "checked"
    assert (r["sample_column"], r["library_column"], r["library_source"]) == ("donor", "library_id", "channel_column")
    assert r["celltype_column"] == "cell_type" and r["split_by"] == []
    assert r["reference"]["basis"] == "matched" and r["reference"]["matched_on"] == ["condition"]
    assert r["n_checked"] == 1 and r["flagged"] == ["D0"]
    row = r["samples"][0]
    assert row["flags"] == ["composition"] and row["pair"]["distance"] > r["reference"]["cut"]
    assert row["pair"]["differing"][0]["group"] == "D"  # the planted shift leads
    assert "L0a" in row["message"] and "L0b" in row["message"] and "D " in row["message"]
    assert "alike in condition" in row["message"] and "median UMIs" in row["message"]
    assert row["libraries"][0]["median_counts"] == pytest.approx(3000, rel=0.1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_libraries_of_one_mix_are_not_flagged(seed):
    a, cells = _design(seed=seed, two_libs=(0, 3))
    r = lc.library_concordance(a, cells)
    assert r["n_checked"] == 2 and r["n_flagged"] == 0
    assert all(s["pair"]["distance"] < r["reference"]["cut"] for s in r["samples"])


def test_pooled_lanes_are_not_libraries_of_a_sample():
    """Kang 2018: each lane (here the condition) holds all 8 donors: nothing nests."""
    a, cells = _design()
    a.obs["library_id"] = a.obs["condition"]
    r = lc.library_concordance(a, cells)
    assert r["status"] == "none" and "nested in 'donor'" in r["reason"] and r["samples"] == []


def test_one_library_per_sample_and_no_samples_say_so():
    a, cells = _design(two_libs=())
    r = lc.library_concordance(a, cells)
    assert r["status"] == "none" and r["library_column"] is None
    b = a.copy()
    del b.obs["donor"]
    r = lc.library_concordance(b, cells)
    assert r["status"] == "none" and "No donor or sample column" in r["reason"]


def test_libraries_of_different_tissues_are_not_compared():
    """Emont 2022: a mouse's inguinal and epididymal fat are two libraries of one donor."""
    a, cells = _design(plant=True, tissues=True)
    r = lc.library_concordance(a, cells)
    assert r["status"] == "none" and "'tissue'" in r["reason"]
    # A third library of the same tissue is compared with its twin only.
    a, cells = _design(two_libs=(0, 3), tissues=True)
    extra, extra_cells = _design(n_donors=1, two_libs=(), cells=600, seed=5)
    extra.obs["library_id"] = pd.Categorical(["L0c"] * extra.n_obs)
    extra.obs["tissue"] = pd.Categorical(["fat"] * extra.n_obs)
    both = ad.concat([a, extra], index_unique="-")
    for c in both.obs.columns:
        both.obs[c] = both.obs[c].astype("category")
    cells = {k: (np.concatenate([cells[k], extra_cells[k]]) if k != "has_mt" else True) for k in cells}
    r = lc.library_concordance(both, cells)
    assert r["split_by"] == ["tissue"]
    assert [s["name"] for s in r["samples"]] == ["D0 · fat"]
    assert {lib["name"] for lib in r["samples"][0]["libraries"]} == {"L0a", "L0c"}


def test_a_small_library_is_listed_not_judged():
    a, cells = _design(plant=True, lib_cells=150)
    r = lc.library_concordance(a, cells)
    row = r["samples"][0]
    assert row["status"] == "too_small" and not row["flagged"]
    assert all(lib["too_small"] for lib in row["libraries"]) and "fewer than two" in row["message"]


def test_too_few_samples_fall_back_to_a_fixed_cut():
    a, cells = _design(n_donors=3, plant=True)
    r = lc.library_concordance(a, cells)
    assert r["reference"]["basis"] == "rule" and r["reference"]["cut"] == lc.RULE_CUT
    assert "Too few samples to calibrate" in r["note"] and r["flagged"] == ["D0"]
    assert "fixed cut" in r["samples"][0]["message"]


def test_libraries_different_in_every_sample_read_as_the_design():
    """Sorted fractions: every donor's second library is another mix, so nothing is a fault."""
    rng = np.random.default_rng(3)
    a, cells = _design(two_libs=(0, 1, 2, 3, 4, 5))
    second = a.obs["library_id"].astype(str).str.endswith("b").to_numpy()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.obs.loc[second, "cell_type"] = rng.choice(TYPES, size=int(second.sum()), p=PLANT)
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    r = lc.library_concordance(a, cells)
    assert r["n_flagged"] == 0 and "by design" in r["note"]
    assert {s["status"] for s in r["samples"]} >= {"by_design"}


def test_named_columns_and_a_missing_one():
    a, cells = _design(plant=True)
    a.obs["run"] = a.obs["library_id"]
    r = lc.library_concordance(a, cells, sample_column="donor", library_column="run",
                               celltype_column="cell_type")
    assert r["library_source"] == "requested" and r["flagged"] == ["D0"]
    r = lc.library_concordance(a, cells, library_column="condition")
    assert r["status"] == "none" and "does not nest" in r["reason"]
    with pytest.raises(lc.LibraryConcordanceError, match="Column 'nope' not found"):
        lc.library_concordance(a, cells, library_column="nope")


def test_libraries_from_the_barcode_prefix():
    a, cells = _design(plant=True)
    rng = np.random.default_rng(0)
    a.obs_names = [f"{lib}_{''.join(rng.choice(list('ACGT'), 16))}" for lib in a.obs["library_id"].astype(str)]
    del a.obs["library_id"]
    r = lc.library_concordance(a, cells)
    assert r["library_source"] == "barcodes" and r["flagged"] == ["D0"]


def test_without_depth_or_cell_types():
    a, _ = _design(plant=True)
    r = lc.library_concordance(a, None)
    assert r["flagged"] == ["D0"] and r["samples"][0]["libraries"][0]["median_counts"] is None
    b = a.copy()
    del b.obs["cell_type"]
    r = lc.library_concordance(b, None)
    assert r["celltype_column"] is None and r["n_flagged"] == 0 and "only depth" in r["note"]


def test_report_and_grounding_lines():
    a, cells = _design(plant=True, two_libs=(0, 3))
    r = lc.library_concordance(a, cells)
    line, concerns = lc.report_lines(r)
    assert "'library_id' within 'donor'" in line and "1 with libraries that disagree (D0)" in line
    assert len(concerns) == 1 and concerns[0].startswith("D0: libraries L0a")
    lines, detail = lc.grounding_lines(r)
    assert all(f"[{lc.TAG}]" in ln for ln in lines[:3]) and detail.startswith("1 of 2")
    assert lc.grounding_lines(lc.library_concordance(*_design(two_libs=()))) == ([], "")


def test_qc_report_carries_the_flag(tmp_path: Path):
    from scview.core import qc_report
    from scview.core.anndata_adaptor import AnnDataAdaptor

    a, _ = _design(plant=True, genes=20)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    f = qc_report.build_facts(AnnDataAdaptor(str(p)))
    assert any(s.ref == "qc:libraries" and "1 with libraries that disagree" in s.detail for s in f.overview)
    assert any(c.ref == "qc:libraries" and "L0b" in c.detail for c in f.concerns)
    assert any(s.ref == "qc:libraries" for s in f.next_steps)


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    a, _ = _design(plant=True, genes=20)
    did = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/qc/libraries"
    r = await client.get(url)
    assert r.status_code == 200, r.text
    assert r.json()["flagged"] == ["D0"] and r.json()["samples"][0]["libraries"][0]["median_counts"] > 0
    r = await client.get(url, params={"sample_column": "donor", "library_column": "condition"})
    assert r.status_code == 200 and r.json()["status"] == "none"
    r = await client.get(url, params={"celltype_column": "nope"})
    assert r.status_code == 404 and "Column 'nope' not found" in r.json()["detail"]


async def test_mcp_tool(client, dataset_manager):
    from mcp import Client

    from scview.mcp.server import SERVER

    a, _ = _design(plant=True, genes=20)
    did = _register(dataset_manager, a, "mcplib")
    async with Client(SERVER) as c:
        r = await c.call_tool("library_concordance", {"dataset_id": did})
        bad = await c.call_tool("library_concordance", {"dataset_id": did, "library_column": "library"})
    assert not r.is_error, r.content
    out = json.loads("".join(getattr(b, "text", "") for b in r.content))
    assert out["flagged"] == ["D0"] and out["library_column"] == "library_id"
    assert out["samples"][0]["pair"]["differing"][0]["group"] == "D" and "L0b" in out["samples"][0]["message"]
    assert bad.is_error and "library_id" in "".join(getattr(b, "text", "") for b in bad.content)


def test_router_sends_library_questions_to_the_qc_facts():
    from scview.core.rag.router import named_result_facets

    assert "qc" in named_result_facets("Do the two libraries of NCD07 agree?")
    assert "qc" in named_result_facets("was this sample sequenced twice?")
