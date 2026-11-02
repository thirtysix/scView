"""Motif routes (api/v1/atac_motifs.py): state, scan, enrichment, logo, chromVAR; co-pilot, replay, large datasets."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scview.api.v1 import atac as atac_api
from scview.core import assistant, atac_chromvar as cv
from scview.core import atac_motifs as am
from scview.core import claims, feature_guide, gene_annotation, provenance, replay_script, results_store, tf_motifs
from scview.core.rag import router as rag_router
from tests import genome_fixture
from tests import multiome_fixture as mf
from tests.test_atac_api import _events, _register_multiome

API = "/api/v1"
JASPAR_TEXT = (
    ">MA0001.1 SPI1\nA [ 9 0 0 0 9 0 0 0 ]\nC [ 0 9 0 0 0 9 0 0 ]\nG [ 0 0 9 0 0 0 9 0 ]\nT [ 0 0 0 9 0 0 0 9 ]\n"
    ">MA0002.1 GENEA::NOPE\nA [ 9 9 1 1 ]\nC [ 1 1 9 1 ]\nG [ 1 1 1 9 ]\nT [ 1 1 1 1 ]\n"
    ">MA0003.1 GENEC\nA [ 1 2 8 1 2 8 ]\nC [ 8 1 1 2 8 1 ]\nG [ 1 8 2 8 1 1 ]\nT [ 2 1 1 1 1 2 ]\n"
)


@pytest.fixture()
def annotated(dataset_manager, tmp_path):
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, dataset_manager.data_dir, "GRCh38")
    yield
    gene_annotation.clear_cache()


@pytest.fixture()
def inputs(monkeypatch, dataset_manager):
    """A tiny genome over the fixture's peaks, and a pinned three-motif JASPAR file."""
    rng = np.random.default_rng(0)
    table = mf.peak_table()
    seqs = {c: "".join("ACGT"[i] for i in rng.integers(0, 4, int(table[table["chrom"] == c]["end"].max()) + 100))
            for c in table["chrom"].unique()}
    genome_fixture.install(monkeypatch, dataset_manager.data_dir, "GRCh38", seqs)
    path = am.jaspar_path(dataset_manager.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(JASPAR_TEXT)
    src = tf_motifs.Source("JASPAR", "http://x/j.txt", hashlib.sha256(JASPAR_TEXT.encode()).hexdigest(), tf_motifs.JASPAR.filename, len(JASPAR_TEXT))
    monkeypatch.setattr(tf_motifs, "JASPAR", src)
    am._tables.clear()
    yield


async def _scan(client, did: str) -> list:
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/scan")
    assert r.status_code == 200, r.text
    return _events(r.text)


async def test_state_says_what_a_scan_needs_before_it_runs(client, dataset_manager, annotated, monkeypatch):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/atac/motifs")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["available"] and s["genome"] == "GRCh38" and not s["scan"]["ready"] and not s["scan"]["running"]
    assert s["files"]["genome"]["downloaded"] is False and s["files"]["genome"]["size"] > 8e8
    assert s["scan"]["estimate_seconds"] > 0 and s["scan"]["estimate_bytes"] > 1e8
    assert s["chromvar"] is None and s["chromvar_estimate"]["estimate_bytes"] > 1e8
    assert s["enrichment_defaults"]["p_max"] == 0.005 and s["enrichment_defaults"]["min_pct"] == 0.2
    assert not list(Path(dataset_manager.data_dir).glob("genome/*.2bit"))


async def test_state_on_a_dataset_without_atac_or_on_another_genome_is_a_sentence(client, dataset_manager):
    import anndata as ad

    from tests.test_hvg_sensitivity import _register

    rna = _register(dataset_manager, ad.AnnData(X=np.ones((5, 3), dtype=np.float32)), "rna")
    body = (await client.get(f"{API}/datasets/{rna}/atac/motifs")).json()
    assert body["available"] is False and "no ATAC peaks" in body["reason"]
    r = await client.post(f"{API}/datasets/{rna}/atac/motifs/scan")
    assert r.status_code == 404


async def test_scan_then_nothing_more_to_do_on_the_same_peaks(client, dataset_manager, annotated, inputs):
    did, _p, table, _a = _register_multiome(dataset_manager)
    events = await _scan(client, did)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-2:] == ["result", "complete"]
    assert any(e == "progress" and d.get("stage") == "scan" for e, d in events)
    result = next(d for e, d in events if e == "result")
    assert result["meta"]["n_motifs"] == 3 and result["meta"]["n_peaks"] == len(table)
    assert result["meta"]["n_left_out"] == 0
    # cached per peak set, under DATA_DIR/atac_motifs/<genome>: a second run costs nothing
    files = list((Path(dataset_manager.data_dir) / "atac_motifs" / "GRCh38").glob("hits_*.npz"))
    assert len(files) == 1
    before = files[0].stat().st_mtime
    again = await _scan(client, did)
    assert [e for e, _ in again] == ["result", "complete"]
    assert files[0].stat().st_mtime == before
    s = (await client.get(f"{API}/datasets/{did}/atac/motifs")).json()
    assert s["scan"]["ready"] and s["scan"]["meta"]["n_motifs"] == 3
    assert not dataset_manager.run_reservations()


async def test_scan_failure_is_one_sentence_and_a_switched_off_download_says_so(client, dataset_manager, annotated, monkeypatch, test_settings):
    did, *_ = _register_multiome(dataset_manager)
    test_settings.GENOME_AUTO_DOWNLOAD = False
    path = am.jaspar_path(dataset_manager.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(JASPAR_TEXT)
    monkeypatch.setattr(tf_motifs, "JASPAR", tf_motifs.Source("J", "http://x/j.txt", hashlib.sha256(JASPAR_TEXT.encode()).hexdigest(), tf_motifs.JASPAR.filename, 1))
    events = await _scan(client, did)
    err = next(d for e, d in events if e == "error")
    assert "GENOME_AUTO_DOWNLOAD is off" in err["error"]
    assert not dataset_manager.run_reservations()


def test_a_second_build_joins_the_running_one_and_cancel_stops_it(tmp_path, monkeypatch):
    calls = []

    def slow_cmd(*a, **k):
        calls.append(a)
        return ["python", "-c", "import time; print('{\"type\": \"progress\", \"stage\": \"scan\"}', flush=True); time.sleep(60)"]

    monkeypatch.setattr(am, "scan_cmd", slow_cmd)
    monkeypatch.setattr(am, "ensure_jaspar", lambda *a, **k: tmp_path / "j")
    monkeypatch.setattr(am.genome_seq, "ensure", lambda *a, **k: None)
    ended = []
    b = am.start_build(tmp_path, "GRCh38", "key1", tmp_path / "s.h5", on_end=lambda: ended.append(1))
    assert am.start_build(tmp_path, "GRCh38", "key1", tmp_path / "s.h5") is b
    assert am.current_build("key1") is b
    deadline = time.time() + 20
    while not b.events and time.time() < deadline:
        time.sleep(0.05)
    b.stop.set()
    while not b.done and time.time() < deadline:
        time.sleep(0.05)
    assert b.done and b.cancelled and b.error is None and ended == [1]
    assert am.current_build("key1") is None
    assert len(calls) == 1


async def test_cancel_route_with_nothing_running(client, dataset_manager, annotated):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/scan/cancel")
    assert r.status_code == 200 and r.json() == {"cancelled": False}


async def _differential(client, did: str, test: str = "wilcoxon") -> dict:
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "test": test})
    return next(d for e, d in _events(r.text) if e == "result")


async def test_enrichment_needs_the_scan_then_a_differential_result(client, dataset_manager, annotated, inputs):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={})
    assert r.status_code == 409 and "Scan peaks first" in r.json()["detail"]
    await _scan(client, did)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={})
    assert r.status_code == 404 and "Differential peaks first" in r.json()["detail"]


async def test_enrichment_uses_a_stored_result_of_either_test(client, dataset_manager, annotated, inputs):
    did, *_ = _register_multiome(dataset_manager)
    await _scan(client, did)
    for test in ("wilcoxon", "lr"):
        da = await _differential(client, did, test)
        r = await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={"da_id": da["id"], "p_max": 0.5, "min_pct": 0.0, "background": 200})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["column"] == "group" and body["da_id"] == da["id"] and body["params"]["da_test"] == test
        a = body["rows"]["A"]
        assert len(a) == 3 and {"motif_id", "motif", "tfs", "observed", "background", "fold_enrichment", "pval", "pval_adj", "gc_rich",
                                "tf_gene", "tf_pct", "tf_expressed"} <= set(a[0])
        assert body["summary"]["A"]["n_query"] > 0 and body["summary"]["A"]["n_background"] > 0
    # the TF's own RNA: GENEA is up in A, in the fixture
    row = next(x for x in a if x["motif"] == "GENEA::NOPE")
    assert row["tf_gene"] == "GENEA" and row["tf_expressed"] is True and row["tf_pct"] > 0.5
    assert next(x for x in a if x["motif"] == "SPI1")["tf_gene"] == ""
    # kept by id; a page, the CSV, the zip with logos
    rid = body["id"]
    page = (await client.get(f"{API}/datasets/{did}/atac/motifs/enrichment/{rid}", params={"group": "B", "limit": 1})).json()
    assert list(page["rows"]) == ["B"] and len(page["rows"]["B"]) == 1
    results_store.flush()
    latest = (await client.get(f"{API}/datasets/{did}/atac/motifs/enrichment", params={"limit": 2})).json()
    assert latest["id"] == rid and len(latest["rows"]["A"]) == 2
    csv = pd.read_csv(io.StringIO((await client.get(f"{API}/datasets/{did}/atac/motifs/enrichment/{rid}/csv")).text))
    assert {"group", "motif_id", "tfs", "fold_enrichment", "pval_adj", "tf_expressed"} <= set(csv.columns) and len(csv) == 6
    z = await client.get(f"{API}/datasets/{did}/atac/motifs/enrichment/{rid}/export")
    assert z.status_code == 200 and "zip" in z.headers["content-type"]
    with zipfile.ZipFile(io.BytesIO(z.content)) as f:
        assert {"motif_enrichment.csv", "README.txt"} <= set(f.namelist())
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/enrichment/nope")).status_code == 404
    assert (await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={"groups": ["Z"]})).status_code == 404
    # a provenance step, additive: the differential result stays current
    steps = [h["step"] for h in provenance.read_provenance(dataset_manager._datasets[did].adata).get("history", [])]
    assert "atac_motif_enrichment" in steps
    assert (await client.get(f"{API}/datasets/{did}/atac/differential")).json() is not None


async def test_logo_route(client, dataset_manager, annotated, inputs):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/atac/motifs/logo/MA0001.1")
    assert r.status_code == 200 and r.json()["motif"] == "SPI1" and r.json()["svg"].startswith("<svg")
    assert r.json()["consensus"] == "ACGTACGT" and r.json()["length"] == 8
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/logo/NOPE")).status_code == 404


async def test_chromvar_runs_in_a_child_keeps_by_barcode_and_reads_back(client, dataset_manager, annotated, inputs):
    did, _p, _t, a = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/chromvar", json={})
    assert r.status_code == 409 and "Scan peaks first" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar")).json() is None
    await _scan(client, did)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/chromvar", json={"seed": 3})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-2:] == ["result", "complete"] and "stage" in names
    start = events[0][1]
    assert start["estimate_seconds"] > 0 and start["estimate_bytes"] > 0 and start["seed"] == 3
    result = next(d for e, d in events if e == "result")
    assert result["n_cells"] == a.n_obs and result["seed"] == 3 and not result["stale"] and result["ranking"]
    assert did not in atac_api._running and not dataset_manager.run_reservations()
    state = (await client.get(f"{API}/datasets/{did}/atac/motifs")).json()
    assert state["chromvar"]["run_id"] == result["run_id"]
    assert not list(cv._folder(dataset_manager.data_dir, did).glob(".chromvar-*"))
    # groups x motifs, a motif on the plot, the ranking
    g = await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/groups", params={"column": "group"})
    assert g.status_code == 200, g.text
    body = g.json()
    assert body["groups"] == ["A", "B"] and len(body["heatmap"]["mean_z"][0]) == 2 and body["rows"]["A"]
    m = await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/motif", params={"motif_id": "MA0001.1", "groupby": "group"})
    assert m.status_code == 200, m.text
    assert len(m.json()["values"]) == a.n_obs and set(m.json()["violin"]) == {"A", "B"}
    csv = (await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/csv")).text
    assert "variability" in csv
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/motif", params={"motif_id": "NOPE"})).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/groups", params={"column": "nope"})).status_code == 404
    steps = [h for h in provenance.read_provenance(dataset_manager._datasets[did].adata).get("history", []) if h["step"] == "atac_chromvar"]
    assert steps and steps[-1]["params"]["seed"] == 3


async def test_chromvar_stale_after_the_cells_change(client, dataset_manager, annotated, inputs):
    did, _p, _t, a = _register_multiome(dataset_manager)
    await _scan(client, did)
    await client.post(f"{API}/datasets/{did}/atac/motifs/chromvar", json={})
    adaptor = dataset_manager._datasets[did]
    summary = cv.read_summary(dataset_manager.data_dir, did)
    assert summary["cells_fingerprint"] == adaptor.cells_fingerprint()
    summary["cells_fingerprint"] = "other cells"
    (cv._folder(dataset_manager.data_dir, did) / cv.SUMMARY).write_text(json.dumps(summary))
    assert (await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar")).json()["stale"] is True
    assert cv.grounding_lines(dataset_manager.data_dir, did, adaptor.cells_fingerprint()) == ([], "")


async def test_a_client_going_away_stops_chromvar_and_keeps_nothing(dataset_manager, annotated, inputs):
    """The stop flag ends the child; ``_events``' closers remove what it wrote."""
    did, *_ = _register_multiome(dataset_manager)
    out = cv.staging(dataset_manager.data_dir, did)
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return True

    store = Path(dataset_manager.data_dir) / "modalities" / did / "atac.h5"
    table = am.PeakMotifTable(genome="GRCh38", key="k", motif_ids=np.array(["M"]), motif_names=np.array(["M"]), lengths=np.array([8]),
                              thresholds=np.zeros(1), gc_motif=np.zeros(1), packed=np.zeros((len(mf.peak_table()), 1), np.uint8),
                              gc=np.full(len(mf.peak_table()), 40.0), background=np.full(4, 0.25))
    table_file = out.parent / "t.npz"
    am.save_table(table, table_file)
    with pytest.raises(am.ScanCancelled):
        list(cv.iter_run(store, np.arange(80), table_file, out, stop=stop))
    assert calls["n"] > 0
    assert not (out / cv.SUMMARY).exists()


async def test_a_large_dataset_tests_motifs_and_runs_chromvar_without_reading_x(client, dataset_manager, annotated, inputs, monkeypatch):
    import anndata as ad

    from scview.core import atac_import, large_layout, large_mode
    from tests.large_fixture import register
    from tests.test_atac_api import _dataset, _seed

    did = "lg"
    a, p, table = _dataset(n_per=60)
    h5ad = register(dataset_manager.data_dir, did, ad.AnnData(X=a.X.copy(), obs=a.obs.copy(), var=a.var.copy()))
    assert large_layout.sidecar_paths(h5ad)
    _seed(dataset_manager.data_dir, did, p, list(a.obs_names), table)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large
    adaptor.adata.uns[atac_import.UNS_KEY] = a.uns[atac_import.UNS_KEY]

    def boom(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("X was read")

    for name in ("to_memory", "toarray"):
        if hasattr(large_mode.OnDiskMatrix, name):
            monkeypatch.setattr(large_mode.OnDiskMatrix, name, boom)
    assert [e for e, _ in await _scan(client, did)][-1] == "complete"
    da = await _differential(client, did)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={"da_id": da["id"], "p_max": 0.5, "min_pct": 0.0, "background": 200})
    assert r.status_code == 200, r.text
    row = next(x for x in r.json()["rows"]["A"] if x["motif"] == "GENEA::NOPE")
    assert row["tf_gene"] == "GENEA" and row["tf_pct"] > 0.5  # read through the gene store
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/chromvar", json={})
    assert [e for e, _ in _events(r.text)][-1] == "complete", r.text[-400:]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large
    assert [h["step"] for h in provenance.read_provenance(adaptor.adata)["history"]][-1] == "atac_chromvar"
    m = await client.get(f"{API}/datasets/{did}/atac/motifs/chromvar/motif", params={"motif_id": "MA0001.1"})
    assert m.status_code == 200 and len(m.json()["values"]) == 120


def test_cost_of_a_step_recorded_is_additive():
    assert {"atac_motif_enrichment", "atac_chromvar"} <= results_store.ADDITIVE_STEPS


def test_copilot_cites_both_and_routes_the_words():
    for word in ("which motifs are enriched in the b cell peaks", "what does chromvar say about SPI1", "jaspar motif"):
        assert any(hint in word for hints in rag_router._RESULT_FACET_HINTS.values() for hint in hints), word
    assert claims.resolve_tag("atacmotifs", ["result:atacmotifs", "result:tfmotifs"]) == ["result:atacmotifs"]
    assert claims.resolve_tag("chromvar", ["result:chromvar"]) == ["result:chromvar"]
    assert hasattr(assistant, "_add_atac_motifs")
    assert "Accessibility > Motifs in peaks" in feature_guide.full_guide()
    assert "Motifs in peaks" not in feature_guide.compact_guide()


async def test_grounding_lines_name_the_group_motif_and_tf(client, dataset_manager, annotated, inputs):
    did, *_ = _register_multiome(dataset_manager)
    await _scan(client, did)
    da = await _differential(client, did)
    await client.post(f"{API}/datasets/{did}/atac/motifs/enrichment", json={"da_id": da["id"], "p_max": 0.5, "min_pct": 0.0, "background": 200})
    results_store.flush()
    adaptor = dataset_manager._datasets[did]
    lines, detail = am.grounding_lines(adaptor)
    assert "GC-matched" in detail and lines[0].startswith("\n## Motifs in differential peaks")
    assert any("Motifs in peaks, A" in x for x in lines) and "[result:atacmotifs]" in lines[1]
    assert "not binding" in lines[-1]


def test_an_old_or_unset_result_still_opens():
    t = pd.DataFrame({"group": ["A"], "motif_id": ["M"], "motif": ["SPI1"], "pval": [0.1], "pval_adj": [0.2], "fold_enrichment": [1.2]})
    res = am.MotifEnrichment("r", "d", "c", ["A", "B"], t, {"A": {"n_query": 1}}, {}, "m", "GRCh38", {}, "now")
    out = am.payload(res)
    assert out["rows"]["B"] == [] and out["rows"]["A"][0]["motif"] == "SPI1"
    assert am.payload(am.MotifEnrichment("r", "d", "c", ["A"], pd.DataFrame(), {}, {}, "m", "GRCh38", {}, "now"))["rows"]["A"] == []


def _step_code(snippet) -> str:
    return "\n".join(snippet.code)


def test_replay_snippets_are_exact_with_their_settings():
    ctx = replay_script._Ctx(None)
    p = {"column": "group", "genome": "GRCh38", "da_test": "lr", "p_column": "pval", "p_max": 0.005, "min_pct": 0.2, "min_log2fc": 0.0,
         "background": 40000, "seed": 0, "groups": None}
    s = replay_script._atac_motif_enrichment(p, ctx)
    assert s.status == replay_script.EXACT and "atac_motif_enrichment" in _step_code(s) and "hg38.2bit" in _step_code(s)
    assert "atac_motifs" in ctx.helpers
    bad = replay_script._atac_motif_enrichment({"column": "group"}, replay_script._Ctx(None))
    assert bad.status == replay_script.NOT_REPLAYABLE
    c = replay_script._atac_chromvar({"seed": 1, "iterations": 50, "w": 0.1, "bs": 50, "genome": "mm10"}, replay_script._Ctx(None))
    assert c.status == replay_script.EXACT and "mm10.2bit" in _step_code(c)
    assert replay_script._atac_chromvar({"genome": "mm10"}, replay_script._Ctx(None)).status == replay_script.NOT_REPLAYABLE


def test_replay_helper_runs_on_its_own_and_gives_the_same_bits(monkeypatch, tmp_path):
    """The helper's text, executed alone (numpy, scipy, pandas, numba), scans as the module does."""
    rng = np.random.default_rng(0)
    seqs = {"chr1": "".join("ACGT"[i] for i in rng.integers(0, 4, 30_000))}
    two_bit = genome_fixture.write_2bit(tmp_path / "hg38.2bit", seqs)
    jaspar = tmp_path / "j.txt"
    jaspar.write_text(JASPAR_TEXT)
    source = "import numpy as np\nimport pandas as pd\nimport scipy.sparse as sp\nINPUT = " + repr(str(tmp_path / "in.h5")) + "\n"
    source += replay_script._HELPERS["atac"].split("class AtacError")[0].split("def top_peaks")[0] + replay_script._HELPERS["atac_motifs"]
    ns: dict = {}
    exec(compile(source, "helper", "exec"), ns)  # noqa: S102
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (two_bit, jaspar)}
    names = ["chr1:100-900", "chr1:1000-1500", "KI270728.1:5-500", "chr1:2000-2800"]
    table = ns["atac_motif_table"](names, {"jaspar_url": f"file://{jaspar}", "jaspar_sha256": sha[jaspar], "two_bit_url": f"file://{two_bit}", "two_bit_sha256": sha[two_bit]})
    motifs = tf_motifs.parse_jaspar(JASPAR_TEXT)
    chrom = np.array(["chr1", "chr1", "KI270728.1", "chr1"], dtype=object)
    mine = am.build(chrom, np.array([100, 1000, 5, 2000]), np.array([900, 1500, 500, 2800]), motifs, am.genome_seq.TwoBit(two_bit), "GRCh38", "k")
    assert np.array_equal(table["hits"], mine.hits(np.arange(4)))
    assert np.array_equal(table["gc"], mine.gc, equal_nan=True)
