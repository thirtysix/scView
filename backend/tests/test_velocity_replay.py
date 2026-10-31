"""RNA velocity in the replay script and the co-pilot (core/replay_script.py, core/assistant.py facts)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import velocity as vel
from scview.core import velocity_input as vi
from tests.test_velocity import run_core
from tests.velocity_fixture import Adaptor, barcode, splicing_counts, write_loom, write_starsolo


def _replay_namespace(script_input: str) -> dict:
    import sys

    import scanpy as sc

    from scview.core import replay_script

    ns = {"sys": sys, "ad": ad, "np": np, "pd": pd, "sc": sc, "sp": sp, "INPUT": script_input}
    exec(replay_script._READ_INPUT, ns)
    exec(replay_script._HELPERS["velocity"], ns)
    return ns


@pytest.mark.parametrize("method", vel.METHODS)
def test_replay_function_gives_what_the_run_gave(tmp_path, method):
    """The script reads the original upload's layers with plain scanpy and numpy."""
    a = splicing_counts()
    a.write_h5ad(tmp_path / "up.h5ad")
    adata, p, out = run_core(tmp_path / "s", a.copy(), method=method)
    rep = ad.read_h5ad(tmp_path / "up.h5ad")
    for k in vi.LAYERS:
        del rep.layers[k]  # nothing in the script's dataset but what it reads from the input
    ns = _replay_namespace(str(tmp_path / "up.h5ad"))
    ns["velocity_run"](rep, None, method, 200, 20, 10)
    assert np.allclose(rep.obs[vel.OBS_CONFIDENCE], out["confidence"], atol=1e-5)
    assert np.allclose(rep.obsm["velocity_umap"], out["arrows"]["X_umap"], atol=1e-5)
    index = {g: i for i, g in enumerate(out["genes"])}
    fit = [g for g in rep.var_names if g in index]
    assert np.allclose(rep.var.loc[fit, vel.VAR_GAMMA].to_numpy(), [out["gamma"][index[g]] for g in fit], rtol=1e-3)
    assert rep.var[vel.VAR_GENE].sum() == out["n_velocity_genes"]


def test_replay_snippet_names_the_attached_file(tmp_path):
    from scview.core import replay_script

    ctx = replay_script._Ctx(adata=None)
    snip = replay_script._velocity({"method": "stochastic", "n_top_genes": 2000, "n_neighbors": 30, "n_pcs": 30,
                                    "source_stored_as": ["abc_v.loom"], "source_files": ["v.loom"]}, ctx)
    assert snip.status == replay_script.EXACT and "velocity" in ctx.helpers
    assert "'abc_v.loom'" in "\n".join(snip.code) or '"abc_v.loom"' in "\n".join(snip.code)
    compile("\n".join(snip.code), "s", "exec")
    bad = replay_script._velocity({"method": "dynamical"}, replay_script._Ctx(adata=None))
    assert bad.status == replay_script.NOT_REPLAYABLE
    compile(replay_script._HELPERS["velocity"], "h", "exec")


def test_replay_reads_a_loom_and_a_starsolo_folder(tmp_path):
    a = splicing_counts(120, 100)
    write_loom(tmp_path / "abc_v.loom", a)
    ns = _replay_namespace(str(tmp_path / "x.h5ad"))
    names, genes, S, U = ns["velocity_counts"]([str(tmp_path / "abc_v.loom")])
    assert names[0].endswith("x") and genes == list(a.var_names)
    assert (U.toarray() == a.layers["unspliced"].toarray()).all()
    paths = write_starsolo(tmp_path / "solo", a, zipped=True)
    names, genes, S, U = ns["velocity_counts"]([str(p) for p in paths])
    assert genes == list(a.var_names) and (S.toarray() == a.layers["spliced"].toarray()).all()
    assert ns["velocity_barcode"](f"sample:{barcode(3)}x") == barcode(3)


def test_copilot_grounding_claims_and_routing(tmp_path):
    from scview.core import claims
    from scview.core.rag import router

    adata, p, out = run_core(tmp_path)
    assert vel.grounding_lines(Adaptor(adata)) == ([], None)  # no saved run yet
    run = vel.save(adata, p, out, {"result_id": "abc"})
    lines, detail = vel.grounding_lines(Adaptor(adata))
    assert f"[{vel.TAG}]" in lines[0] and "median confidence" in detail
    assert any("not a fate claim" in x for x in lines) and any("coherence" in x for x in lines)
    assert run["n_velocity_genes"] == out["n_velocity_genes"]
    assert claims.resolve_tag("velocity", [vel.TAG]) == [vel.TAG]
    assert claims.resolve_tag("result:velocity", [vel.TAG, "result:wnn"]) == [vel.TAG]
    for q in ("what does the rna velocity show", "is the unspliced fraction low", "phase portrait of Ins2"):
        assert "groups" in router.named_result_facets(q)


def test_feature_guide_says_what_is_not_offered():
    from scview.core import feature_guide

    entry = next(f for f in feature_guide.FEATURES if f[0] == "Unified View > Velocity")
    assert "not the dynamical model" in entry[2]
