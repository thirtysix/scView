"""Reference mapping / label transfer (core/label_transfer.py) and its replay step."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import label_transfer as lt
from scview.core import provenance, replay_script

N_GENES = 900
MARKERS = 40
TYPES = ("A", "B", "C", "D")


def _cells(types: dict[str, int], seed: int, shift: float = 0.0, prefix: str = "c") -> ad.AnnData:
    """Counts: every type has 40 marker genes of its own (type i: genes 40i..40i+39)
    over a Poisson background; ``shift`` multiplies a block of 60 other genes in every
    cell (a batch effect)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for t, n in types.items():
        i = TYPES.index(t)
        lam = np.full(N_GENES, 0.4)
        lam[i * MARKERS:(i + 1) * MARKERS] = 6.0
        if shift:
            lam[600:660] *= shift
        size = rng.uniform(0.6, 1.6, size=(n, 1))
        rows.append(rng.poisson(lam * size))
        labels += [t] * n
    X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    a.obs_names = [f"{prefix}{i}" for i in range(X.shape[0])]
    a.obs["cell_type"] = pd.Categorical(labels)
    return a


def _reference(**kw) -> ad.AnnData:
    ref = _cells({"A": 150, "B": 150, "C": 150}, seed=1, prefix="r", **kw)
    rng = np.random.default_rng(3)
    ref.obsm["X_umap"] = (np.array([[0, 0], [10, 0], [0, 10]])[ref.obs["cell_type"].cat.codes]
                          + rng.normal(0, 0.5, size=(ref.n_obs, 2))).astype(np.float32)
    return ref


def _query(**kw) -> ad.AnnData:
    q = _cells({"A": 120, "B": 120, "C": 120, "D": 80}, seed=2, prefix="q", **kw)
    q.obs["truth"] = q.obs["cell_type"]
    q.obs["cluster"] = pd.Categorical(q.obs["cell_type"].map({"A": "0", "B": "1", "C": "2", "D": "3"}).astype(str))
    del q.obs["cell_type"]
    return q


def _prepare(ref, query, **kw) -> lt.Prepared:
    return lt.prepare(ref, query, label_column="cell_type", ref_info={"id": "ref1", "name": "Ref one",
                                                                      "path": "/data/ref.h5ad", "mtime": 1.0},
                      ref_expression=lambda a: lt.expression_of(a, []),
                      query_expression=lambda a: lt.expression_of(a, []),
                      cluster_column=kw.pop("cluster_column", "cluster"), **kw)


def _map(p: lt.Prepared, method: str = lt.METHOD_PROJECTION, k: int = 15) -> dict:
    events = list(lt.iter_transfer(p, method=method, k=k, n_hvg=300))
    assert [e["type"] for e in events][-1] == "mapped"
    assert {e["step"] for e in events if e["type"] == "progress"} == {"genes", "space"}
    return events[-1]["out"]


def test_lognorm_is_log1p_of_counts_per_10k():
    M = sp.csr_matrix(np.array([[1, 3, 0], [0, 0, 0], [2, 2, 4]], dtype=np.float32))
    X = lt.lt_lognorm(M).toarray()
    want = np.log1p(np.array([[2500, 7500, 0], [0, 0, 0], [2500, 2500, 5000]], dtype=np.float64))
    assert np.allclose(X, want, rtol=1e-6)


@pytest.mark.parametrize("method", [lt.METHOD_PROJECTION, lt.METHOD_INTEGRATION])
def test_known_types_map_and_a_type_the_reference_lacks_is_flagged(method):
    q = _query()
    p = _prepare(_reference(), q)
    assert len(p.genes) == N_GENES and p.ref_log.shape == (450, N_GENES)
    out = _map(p, method)
    truth = q.obs["truth"].astype(str).to_numpy()
    known = truth != "D"
    assert (out["label"][known] == truth[known]).mean() > 0.97
    assert np.median(out["score"][known]) > 0.9
    # D is in no reference cell: most of it is flagged, few of the rest (~1% by chance).
    assert out["unmapped"][~known].mean() > 0.8
    assert out["unmapped"][known].mean() < 0.05
    assert out["umap"].shape == (q.n_obs, 2)
    # A query cell of type B lands on the reference's B island of the UMAP.
    assert np.abs(out["umap"][truth == "B"].mean(axis=0) - [10, 0]).max() < 1.5
    s = lt.summarize(p, out, method=method, prefix="ref_one")
    rows = {r["cluster"]: r for r in s["clusters"]}
    assert rows["3"]["candidate_absent"] is True and rows["0"]["candidate_absent"] is False
    assert s["clusters"][0]["cluster"] == "3"  # most unmapped first
    assert s["n_shared_genes"] == N_GENES and s["k"] == 15
    assert sum(s["score_hist"]["counts"]) == q.n_obs and len(s["score_hist"]["edges"]) == lt.HIST_BINS + 1
    assert {r["label"] for r in s["reference_labels"]} == {"A", "B", "C"}


def test_integration_removes_a_shift_shared_by_every_query_cell():
    ref = _reference()
    q = _query(shift=3.0)  # 60 genes three times higher in every query cell
    p = _prepare(ref, q)
    truth = q.obs["truth"].astype(str).to_numpy()
    known = truth != "D"
    proj = _map(p, lt.METHOD_PROJECTION)
    integ = _map(p, lt.METHOD_INTEGRATION)
    # Projection keeps the shift (it flagged ~9% of known cells before the cut followed
    # the query's median ratio); the cut now absorbs it, and D still flags in both.
    assert proj["shift"] > 1.05 and integ["shift"] < proj["shift"]
    for out in (proj, integ):
        assert out["unmapped"][known].mean() < 0.02
        assert out["unmapped"][~known].mean() > 0.8
    assert (integ["label"][known] == truth[known]).mean() > 0.95


def test_celltypist_trained_on_the_reference_labels_the_cells():
    q = _query()
    out = _map(_prepare(_reference(), q), lt.METHOD_CELLTYPIST)
    truth = q.obs["truth"].astype(str).to_numpy()
    known = truth != "D"
    assert (out["label"][known] == truth[known]).mean() > 0.95
    assert out["unmapped"][~known].mean() > 0.8  # the check is the projection's own


def test_novelty_ratio_is_scale_free():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(300, 5)).astype(np.float32)
    q = rng.normal(size=(100, 5)).astype(np.float32)
    r1, rr1 = lt.lt_novelty(ref, q, 10)
    r2, rr2 = lt.lt_novelty(ref * 7, q * 7, 10)
    assert np.allclose(r1, r2, rtol=1e-4) and np.allclose(rr1, rr2, rtol=1e-4)
    # Drawn from the reference's own distribution: about 1% above its 99th percentile.
    assert (r1 > np.percentile(rr1, 99)).mean() < 0.06


def test_vote_weights_near_neighbours_more():
    idx = np.array([[0, 1, 2]])
    codes = np.array([0, 1, 1])
    w = lt.lt_weights(np.array([[0.1, 2.0, 2.0]], dtype=np.float32))
    best, score = lt.lt_vote(idx, w, codes, 2)
    assert best[0] == 0 and 0.5 < score[0] < 1  # one near neighbour outweighs two far ones


def test_refusals():
    ref, q = _reference(), _query()
    with pytest.raises(lt.LabelTransferError, match="no column"):
        lt.check_labels(ref, "nope")
    ref.obs["score"] = np.arange(ref.n_obs, dtype=float)
    with pytest.raises(lt.LabelTransferError, match="numbers"):
        lt.check_labels(ref, "score")
    ref.obs["one"] = pd.Categorical(["x"] * ref.n_obs)
    with pytest.raises(lt.LabelTransferError, match="at least 2"):
        lt.check_labels(ref, "one")
    assert "pick a mouse reference" in lt.check_species("human", "mouse")
    assert lt.check_species("human", "unknown") is None and lt.check_species("mouse", "mouse") is None
    # Too few shared genes: the query keeps 300 of the reference's genes.
    small = q[:, :300].copy()
    with pytest.raises(lt.LabelTransferError, match="share 300 genes") as e:
        _prepare(_reference(), small)
    assert e.value.status == 422
    # Scaled values only: nothing to log-normalise.
    scaled = q.copy()
    scaled.X = sp.csr_matrix(np.asarray(scaled.X.todense()) - 1.0)
    with pytest.raises(lt.LabelTransferError, match="No raw counts"):
        _prepare(_reference(), scaled)
    with pytest.raises(lt.LabelTransferError, match="k must"):
        list(lt.iter_transfer(_prepare(_reference(), _query()), method="projection", k=1))


def test_species_mismatch_is_refused(monkeypatch):
    from scview.core import species

    calls = iter([species.SpeciesCall(species="human"), species.SpeciesCall(species="mouse")])
    monkeypatch.setattr(species, "detect_species", lambda a: next(calls))
    with pytest.raises(lt.LabelTransferError, match="looks human and this dataset mouse"):
        _prepare(_reference(), _query())


def test_genes_are_matched_across_ensembl_ids_and_symbols():
    ref, q = _reference(), _query()
    ids = [f"ENSG{i:011d}" for i in range(N_GENES)]
    q.var["gene_ids"] = ids
    ref.var_names = ids  # the reference indexed by Ensembl id, the query by symbol
    m = lt.match_genes(ref.var, q.var)
    assert len(m.shared) == N_GENES and m.renamed == "this dataset by its var 'gene_ids'"
    assert m.shared[:2] == ["G0", "G1"] and m.ref_names[:2] == ids[:2]
    p = _prepare(ref, q)
    assert len(p.genes) == N_GENES and p.info["renamed"].startswith("this dataset")
    out = _map(p)
    known = q.obs["truth"].astype(str).to_numpy() != "D"
    assert (out["label"][known] == q.obs["truth"].astype(str).to_numpy()[known]).mean() > 0.97


def test_unlabelled_reference_cells_are_left_out():
    ref = _reference()
    ref.obs["cell_type"] = ref.obs["cell_type"].astype(object)
    ref.obs.iloc[:20, ref.obs.columns.get_loc("cell_type")] = None
    p = _prepare(ref, _query())
    assert p.ref_log.shape[0] == 430 and p.info["n_reference_unlabelled"] == 20
    assert p.ref_umap.shape == (430, 2)


def test_label_columns_lists_annotations_only():
    ref = _reference()
    ref.obs["barcode"] = ref.obs_names.astype(str)
    ref.obs["n_genes"] = 5
    cols = {c["column"]: c["n_labels"] for c in lt.label_columns(ref)}
    assert cols == {"cell_type": 3}


def test_save_writes_columns_summary_and_provenance():
    q = _query()
    p = _prepare(_reference(), q)
    out = _map(p)
    s = lt.summarize(p, out, method="projection", prefix="ref_one")
    lt.save(q, p, out, s, resources={"seconds": 1.0})
    assert q.obs["ref_one_transfer"].dtype.name == "category"
    assert q.obs["ref_one_transfer_score"].dtype == np.float32
    assert q.obs["ref_one_unmapped"].dtype == bool
    assert q.obsm["X_umap_ref_ref_one"].shape == (q.n_obs, 2)
    assert [x["prefix"] for x in lt.saved(q)] == ["ref_one"]
    json.dumps(lt.read_all(q))
    step = provenance.read_provenance(q)["history"][-1]
    assert step["step"] == "label_transfer"
    prm = step["params"]
    assert prm["reference_id"] == "ref1" and prm["reference_file"] == "ref.h5ad"
    assert prm["label_column"] == "cell_type" and prm["method"] == "projection" and prm["k"] == 15
    assert len(prm["genes"]) == 300 and set(prm["genes"]) <= set(prm["novelty_genes"])
    assert prm["outputs"] == ["ref_one_transfer", "ref_one_transfer_score", "ref_one_unmapped",
                              "X_umap_ref_ref_one"]
    assert replay_script.describe(q)[-1]["status"] == replay_script.EXACT


def test_replay_reproduces_the_mapping(tmp_path):
    from tests.test_replay_script import _replay

    ref, q = _reference(), _query()
    inp = q.copy()
    p = _prepare(ref, q)
    out = _map(p, lt.METHOD_INTEGRATION)
    lt.save(q, p, out, lt.summarize(p, out, method="integration", prefix="ref_one"))
    ref.write_h5ad(tmp_path / "ref.h5ad")  # beside the input, by the recorded name
    rep = _replay(tmp_path, inp, q)
    assert (rep.obs["ref_one_transfer"].astype(str).to_numpy() == q.obs["ref_one_transfer"].astype(str).to_numpy()).all()
    assert np.allclose(rep.obs["ref_one_transfer_score"], q.obs["ref_one_transfer_score"], atol=1e-5)
    assert (rep.obs["ref_one_unmapped"].to_numpy() == q.obs["ref_one_unmapped"].to_numpy()).mean() > 0.99
    assert np.allclose(rep.obsm["X_umap_ref_ref_one"], q.obsm["X_umap_ref_ref_one"], atol=1e-3)


def test_replay_without_the_record_is_a_comment():
    q = _query()
    provenance.record_step(q, step="label_transfer", tool="t", params={"method": "projection"})
    assert replay_script.describe(q)[-1]["status"] == replay_script.NOT_REPLAYABLE
    snippet = replay_script._label_transfer({"method": "celltypist", "genes": ["G1"], "novelty_genes": ["G1"],
                                             "label_column": "x", "prefix": "r", "k": 15,
                                             "reference_file": "r.h5ad"}, replay_script._Ctx(adata=q))
    assert snippet.status == replay_script.APPROXIMATE and "CellTypist" in snippet.note
    compile("\n".join(snippet.code), "step.py", "exec")


def test_slug():
    assert lt.slug("Kang PBMC (ctrl)") == "kang_pbmc_ctrl"
    assert lt.slug("  ") == "reference"
