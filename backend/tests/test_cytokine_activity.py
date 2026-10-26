"""Cytokine response, CytoSig-style (C079): core/cytokine_activity.py + /cytokines/activity.

The signature here is synthetic (random weights for invented cytokines, written in
the signature.centroid layout), with its sha256 pinned for the test, so no test
touches the network. Agreement with CytoSig 0.0.3 on Kang STIM vs CTRL (Spearman of
z 0.9997) is in the module docstring and the item report.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import cytokine_activity as ca
from scview.core.claims import resolve_tag
from tests.large_fixture import register, small_twin

API = "/api/v1"
N_GENES = 700
CYTOKINES = ["IFN1", "IFNG", "TNFA", "IL4", "IL6", "IL10", "TGFB1", "IL2"]


def _signature(genes: list[str], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.3, (len(genes), len(CYTOKINES))), index=genes, columns=CYTOKINES)


def _raw(sig: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    sig.to_csv(buf, sep="\t", float_format="%.6f")
    # CytoSig's header names the cytokines only, no label for the gene column.
    lines = buf.getvalue().splitlines()
    lines[0] = "\t".join(sig.columns)
    return ("\n".join(lines) + "\n").encode()


GENES = [f"G{i}" for i in range(N_GENES)]
SIG = _signature(GENES[:600] + [f"NOTHERE{i}" for i in range(50)])
RAW = _raw(SIG)


@pytest.fixture(autouse=True)
def small_signature(monkeypatch):
    monkeypatch.setattr(ca, "SHA256", hashlib.sha256(RAW).hexdigest())
    monkeypatch.setattr(ca, "MIN_FILE_CYTOKINES", 5)
    monkeypatch.setattr(ca, "MIN_FILE_GENES", 100)
    ca._loaded.clear()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _cells(n_per: int = 60, seed: int = 0, genes: list[str] | None = None,
           sig: pd.DataFrame | None = None) -> ad.AnnData:
    """Log-normalised cells in two cell types x CTRL / STIM; STIM moves every gene along
    its IFN1 signature weight (the positive control)."""
    genes = genes or GENES
    rng = np.random.default_rng(seed)
    n = 4 * n_per
    stim = np.tile(np.repeat([0, 1], n_per), 2).astype(bool)
    ctype = np.repeat(["T", "Mono"], 2 * n_per)
    w = (SIG if sig is None else sig)["IFN1"].reindex([g.upper() for g in genes]).fillna(0).to_numpy()
    mu = np.full((n, len(genes)), 3.0) * rng.gamma(4.0, 0.25, size=(1, len(genes)))
    mu[stim] *= np.exp(2.0 * w)
    mu[ctype == "Mono", :30] *= 3.0  # a cell-type difference unrelated to cytokines
    a = ad.AnnData(sp.csr_matrix(rng.poisson(mu).astype("float32")))
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["stim"] = pd.Categorical(np.where(stim, "STIM", "CTRL"))
    a.obs["cell_type"] = pd.Categorical(ctype, categories=["T", "Mono"])
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# --- the signature file -----------------------------------------------------------


def test_parse_reads_the_centroid_layout_and_refuses_other_files():
    sig = ca.parse(RAW)
    assert list(sig.columns) == CYTOKINES and sig.shape == (650, 8)
    assert sig.loc["G3", "IFNG"] == pytest.approx(SIG.loc["G3", "IFNG"], abs=1e-6)
    with pytest.raises(ValueError):
        ca.parse(b"<html><body>captive portal</body></html>")
    with pytest.raises(ValueError, match="IFN1"):
        ca.parse(_raw(SIG.rename(columns={"IFN1": "X1"})))


def test_download_checks_the_sha256_and_records_its_provenance(tmp_path):
    ca.download(tmp_path, client=_client(lambda r: httpx.Response(200, content=RAW)))
    meta = ca.read_meta(tmp_path)
    assert meta["licence"].startswith("GPL-3.0") and "not redistributed" in meta["licence"]
    assert meta["url"] == ca.URL and ca.COMMIT in meta["url"] and meta["sha256"] == ca.SHA256
    assert meta["n_genes"] == 650 and meta["n_cytokines"] == 8
    assert ca.load(tmp_path, auto_download=False).shape == (650, 8)

    other = tmp_path / "other"
    with pytest.raises(ValueError, match="sha256"):
        ca.download(other, client=_client(lambda r: httpx.Response(200, content=RAW + b"G9\t1\n")))
    assert not any(other.iterdir())


def test_offline_switched_off_and_altered_files_say_what_to_do(tmp_path, monkeypatch):
    def offline(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(ca, "download", offline)
    with pytest.raises(ca.SignatureUnavailable, match="check the internet"):
        ca.load(tmp_path / "a")
    with pytest.raises(ca.SignatureUnavailable, match="CYTOSIG_AUTO_DOWNLOAD"):
        ca.load(tmp_path / "a", auto_download=False)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / ca.FILENAME).write_bytes(RAW.replace(b"G1\t", b"G1x\t", 1))
    with pytest.raises(ca.SignatureUnavailable, match="sha256"):
        ca.load(tmp_path / "b")


# --- the method -------------------------------------------------------------------


def test_ridge_coefficients_are_the_closed_form_and_the_z_finds_the_planted_cytokine():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 6))
    Y = np.column_stack([X[:, 2] * 0.5 + rng.normal(0, 1, 800), rng.normal(0, 1, 800)])
    fit = ca.ridge_z(X, Y, lam=50.0, n_perm=300)
    Xs, Ys = ca.standardise(X), ca.standardise(Y)
    beta = np.linalg.solve(Xs.T @ Xs + 50.0 * np.eye(6), Xs.T @ Ys)
    np.testing.assert_allclose(fit["beta"], beta, rtol=1e-10)
    assert int(np.argmax(fit["z"][:, 0])) == 2 and fit["z"][2, 0] > 8 and fit["p"][2, 0] < 0.01
    assert np.abs(fit["z"][:, 1]).max() < 4  # noise: no cytokine stands out
    again = ca.ridge_z(X, Y, lam=50.0, n_perm=300)
    np.testing.assert_array_equal(fit["z"], again["z"])  # fixed seed


def test_mouse_symbols_are_matched_upper_case_and_ambiguous_ones_dropped():
    sig = ca.parse(RAW)
    genes = ["G1", "G2", "g3", "G3", "Unrelated"]  # G3 twice once upper-cased
    m = ca.match_genes(sig, genes, "mouse")
    assert list(m.signature.index) == ["G1", "G2"] and m.positions.tolist() == [0, 1]
    assert "upper-cased" in m.mapped_by
    h = ca.match_genes(sig, genes, "human")
    assert list(h.signature.index) == ["G1", "G2", "G3"] and h.positions.tolist() == [0, 1, 3]


def _hcop(pairs: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["human", "mouse", "evidence"]).astype({"evidence": "int16"})


def test_mouse_genes_pair_one_to_one_through_hcop_orthologs():
    sig = ca.parse(RAW)
    table = _hcop([("G1", "Gm1", 5), ("G2", "Other2", 4),  # a mouse name that differs
                   ("G3", "Gm3a", 3), ("G3", "Gm3", 9),  # one-to-many: best supported wins
                   ("G4", "Gm1", 5),  # many-to-one: Gm1 is already G1's
                   ("G5", "Gm5", 2)])  # under the evidence cut
    genes = ["Gm1", "Other2", "Gm3a", "Gm3", "Gm5", "G6"]
    m = ca.match_genes(sig, genes, "mouse", table)
    assert list(m.signature.index) == ["G1", "G2", "G3"] and m.positions.tolist() == [0, 1, 3]
    assert "HCOP orthologs" in m.mapped_by
    off = ca.match_genes(sig, genes, "mouse", None, "offline")
    assert "upper-cased" in off.mapped_by and "HCOP orthologs unavailable: offline" in off.mapped_by
    # Human ignores the table: exact symbols, as before.
    h = ca.match_genes(sig, ["G1", "G2"], "human", table)
    assert list(h.signature.index) == ["G1", "G2"] and h.mapped_by == "human gene symbols"


def test_conditions_profile_is_a_minus_b_within_each_group():
    a = _cells(n_per=20)
    d = ca.design_conditions(a.obs, "cell_type", "stim", "STIM", "CTRL")
    sums, sizes = ca.group_sums(a.X, np.arange(5), d.codes, d.k)
    Y, kept, skipped, ns = ca.profiles(d, sums, sizes)
    X = a.X.toarray()[:, :5].astype(np.float64)
    t = (a.obs["cell_type"] == "T").to_numpy()
    s = (a.obs["stim"] == "STIM").to_numpy()
    np.testing.assert_allclose(Y[:, 0], X[t & s].mean(0) - X[t & ~s].mean(0), rtol=1e-6)
    assert kept == [0, 1] and ns == [[20, 20], [20, 20]] and not skipped
    few = ca.profiles(d, sums, sizes, min_cells=25)
    assert few[1] == [] and len(few[2]) == 2


# --- endpoint -----------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


@pytest.fixture()
def signature(tmp_data_dir: Path) -> Path:
    dest = tmp_data_dir / "networks" / "cytosig"
    ca.download(dest, client=_client(lambda r: httpx.Response(200, content=RAW)))
    return dest


async def test_stim_vs_ctrl_puts_ifn1_on_top_and_is_reused(client, dataset_manager, signature):
    did = _register(dataset_manager, _cells(), "kangish")
    body = {"mode": "conditions", "groupby": "cell_type", "condition": "stim", "a": "STIM", "b": "CTRL",
            "n_permutations": 200}
    r = await client.post(f"{API}/datasets/{did}/cytokines/activity", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["groups"] == ["T", "Mono"] and res["group_cells"] == [[60, 60], [60, 60]]
    assert res["n_shared_genes"] == 600 and res["n_signature_genes"] == 600
    assert res["cytokines"] == CYTOKINES and res["licence"].startswith("GPL-3.0")
    for t in res["top"]:
        assert t["cytokines"][0]["cytokine"] == "IFN1" and t["cytokines"][0]["z"] > 5
    assert not res["reused"]
    again = (await client.post(f"{API}/datasets/{did}/cytokines/activity", json=body)).json()
    assert again["reused"] and again["z"] == res["z"]

    # The co-pilot's fact names the comparison and the top cytokine.
    from scview.core.assistant import build_grounding_context

    adaptor = await dataset_manager.get_or_load_dataset(did)
    lines, detail = ca.grounding_lines(adaptor)
    assert any("cytokine response (CytoSig): T STIM vs CTRL top IFN1 z=" in ln for ln in lines)
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "cytokine response (CytoSig)" in context
    refs = [s.ref for s in sources]
    assert ca.TAG in refs and resolve_tag("cytokines", refs) == [ca.TAG]


async def test_groups_vs_the_rest(client, dataset_manager, signature):
    did = _register(dataset_manager, _cells(), "groups")
    r = await client.post(f"{API}/datasets/{did}/cytokines/activity",
                          json={"groupby": "stim", "n_permutations": 200})
    assert r.status_code == 200, r.text
    res = r.json()
    z = dict(zip(res["groups"], res["z"]))
    j = res["cytokines"].index("IFN1")
    assert z["STIM"][j] > 5 and z["CTRL"][j] < -5
    alls = (await client.post(f"{API}/datasets/{did}/cytokines/activity",
                              json={"groupby": "stim", "reference": "all", "n_permutations": 200})).json()
    assert alls["reference"] == "all" and alls["z"][alls["groups"].index("STIM")][j] > 5


async def test_bad_requests_too_few_genes_and_offline(client, dataset_manager, signature, monkeypatch):
    did = _register(dataset_manager, _cells(), "bad")
    url = f"{API}/datasets/{did}/cytokines/activity"
    assert (await client.post(url, json={"mode": "groups"})).status_code == 400
    assert (await client.post(url, json={"groupby": "nope"})).status_code == 404
    r = await client.post(url, json={"mode": "conditions", "condition": "stim", "a": "STIM", "b": "X"})
    assert r.status_code == 400 and "'X'" in r.json()["detail"]
    assert (await client.post(f"{API}/datasets/ghost/cytokines/activity",
                              json={"groupby": "stim"})).status_code == 404

    monkeypatch.setattr(ca, "MIN_SHARED_GENES", 650)
    r = await client.post(url, json={"groupby": "stim", "n_permutations": 100})
    assert r.status_code == 422 and "600 of the 600" in r.json()["detail"]
    monkeypatch.setattr(ca, "MIN_SHARED_GENES", 500)

    ca._loaded.clear()
    (signature / ca.FILENAME).unlink()
    monkeypatch.setattr(ca, "download", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    r = await client.post(url, json={"groupby": "cell_type", "n_permutations": 100})
    assert r.status_code == 503 and "check the internet" in r.json()["detail"]


async def test_mouse_genes_map_by_upper_case(client, dataset_manager, signature, monkeypatch):
    from scview.core import orthologs

    def offline(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(orthologs, "download", offline)  # no HCOP table: the old upper-case match
    genes = [f"Gm{i}" for i in range(N_GENES)]  # mouse-style: title case
    sig = _signature(genes[:600])
    sig.index = [g.upper() for g in sig.index]
    raw = _raw(sig)
    (signature / ca.FILENAME).write_bytes(raw)
    ca._loaded.clear()
    ca.SHA256 = hashlib.sha256(raw).hexdigest()  # restored by the autouse fixture's monkeypatch
    a = _cells(genes=genes, sig=sig)
    did = _register(dataset_manager, a, "mouse")
    r = await client.post(f"{API}/datasets/{did}/cytokines/activity",
                          json={"mode": "conditions", "condition": "stim", "a": "STIM", "b": "CTRL",
                                "n_permutations": 100})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["species"] == "mouse" and "upper-cased" in res["mapped_by"]
    assert "HCOP orthologs unavailable" in res["mapped_by"]
    assert res["groups"] == ["all cells"] and res["top"][0]["cytokines"][0]["cytokine"] == "IFN1"


async def test_a_large_dataset_sums_from_the_gene_store(client, tmp_data_dir, signature, monkeypatch):
    from tests.large_fixture import counts_adata

    counts = counts_adata(n_genes=300)
    names = list(counts.var_names)
    sig = _signature(names)
    raw = _raw(sig)
    (signature / ca.FILENAME).write_bytes(raw)
    ca._loaded.clear()
    monkeypatch.setattr(ca, "SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(ca, "MIN_SHARED_GENES", 100)
    register(tmp_data_dir, "bigC", counts)
    up = tmp_data_dir / "uploads" / "smallC"
    up.mkdir(parents=True)
    small_twin(counts).write_h5ad(up / "twin.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": "smallC", "name": "twin", "status": "ready"}))
    body = {"mode": "conditions", "groupby": "leiden", "condition": "condition", "a": "stim", "b": "ctrl",
            "n_permutations": 100}
    big = await client.post(f"{API}/datasets/bigC/cytokines/activity", json=body)
    small = await client.post(f"{API}/datasets/smallC/cytokines/activity", json=body)
    assert big.status_code == 200, big.text
    assert small.status_code == 200, small.text
    b, s = big.json(), small.json()
    assert b["matrix_source"].endswith("(gene store)")
    assert b["groups"] == s["groups"] and b["n_shared_genes"] == s["n_shared_genes"]
    np.testing.assert_allclose(b["coef"], s["coef"], atol=1e-5)
    np.testing.assert_allclose(b["z"], s["z"], atol=1e-2)


async def test_mouse_genes_map_through_hcop_orthologs(client, dataset_manager, signature, tmp_data_dir):
    """50 signature genes have a mouse name that is not the human one upper-cased:
    HCOP finds them, upper-casing would not."""
    from scview.core import orthologs

    genes = [f"Gm{i}" for i in range(N_GENES)]
    sig = _signature([g.upper() for g in genes[:600]])
    raw = _raw(sig)
    (signature / ca.FILENAME).write_bytes(raw)
    ca._loaded.clear()
    ca.SHA256 = hashlib.sha256(raw).hexdigest()  # restored by the autouse fixture's monkeypatch
    mouse_genes = [f"Renamed{i}" if i < 50 else g for i, g in enumerate(genes)]
    a = _cells(genes=genes, sig=sig)
    a.var_names = mouse_genes
    hcop = _hcop([(f"GM{i}", mouse_genes[i], 5) for i in range(600)])
    dest = tmp_data_dir / "networks"
    hcop.to_parquet(dest / orthologs.TABLE, index=False)
    (dest / orthologs.META).write_text(json.dumps({"pinned": True}))
    orthologs._loaded.clear()
    did = _register(dataset_manager, a, "mouse_hcop")
    r = await client.post(f"{API}/datasets/{did}/cytokines/activity",
                          json={"mode": "conditions", "condition": "stim", "a": "STIM", "b": "CTRL",
                                "n_permutations": 100})
    assert r.status_code == 200, r.text
    res = r.json()
    assert "HCOP orthologs" in res["mapped_by"] and res["n_signature_genes"] == 600
    assert res["top"][0]["cytokines"][0]["cytokine"] == "IFN1"
    adaptor = await dataset_manager.get_or_load_dataset(did)
    lines, _ = ca.grounding_lines(adaptor)
    assert lines[-1].startswith(
        "- Caveat: human signature; mouse genes mapped through HCOP orthologs (600 signature genes mapped")
