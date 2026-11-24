"""Per-cell label tables into a user annotation column (core/label_import.py,
POST /labels/import and /labels/import-file, MCP import_labels).

QA round 7 brought 91 author labels for 46,700 cells in with 91 ``/labels`` calls and
an order join checked by hand (2 cells missing from one donor shifted 400 labels)."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import label_import as li
from scview.core import provenance, replay_script
from scview.core.user_labels import UNASSIGNED, LabelError, user_columns

API = "/api/v1"


def _adata(n: int = 12) -> ad.AnnData:
    a = ad.AnnData(np.ones((n, 3), dtype="float32"))
    a.obs_names = [f"AAAC{i:02d}-1" for i in range(n)]
    a.var_names = ["G0", "G1", "G2"]
    a.obs["leiden"] = pd.Categorical(["0", "1"] * (n // 2))
    return a


def _table(rows: list[tuple[str, str]], header=("barcode", "label"), sep=",") -> bytes:
    return ("\n".join([sep.join(header), *(sep.join(r) for r in rows)]) + "\n").encode()


# --- core -------------------------------------------------------------------------------------

def test_by_name_labels_matched_cells_and_reports_extra_and_left_out():
    a = _adata()
    rows = [(f"AAAC{i:02d}-1", "T cell" if i < 5 else "B cell") for i in range(10)] + [("ZZZ-1", "T cell")]
    t = li.read_table(_table(rows), "labels.csv")
    out = li.import_labels(a, "author", t)
    assert out["match"] == "name" and out["n_rows"] == 11 and out["n_matched"] == 10
    assert out["n_extra"] == 1 and out["extra_examples"] == ["ZZZ-1"]
    assert out["n_unmatched_cells"] == 2 and out["unmatched_examples"] == ["AAAC10-1", "AAAC11-1"]
    assert {r["label"]: r["n_cells"] for r in out["labels"]} == {"T cell": 5, "B cell": 5}
    assert list(a.obs["author"][:5]) == ["T cell"] * 5
    assert a.obs["author"].iloc[11] == UNASSIGNED
    assert out["n_unassigned"] == 2
    assert "author" in user_columns(a)
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "import_labels"
    assert step["params"]["file"] == "labels.csv" and step["params"]["barcode_column"] == "barcode"


def test_refused_when_too_few_rows_match_with_examples_and_a_suffix_hint():
    a = _adata()
    rows = [(f"AAAC{i:02d}", "T") for i in range(12)]  # the table lost Cell Ranger's "-1"
    t = li.read_table(_table(rows), "x.csv")
    with pytest.raises(LabelError) as e:
        li.import_labels(a, "author", t)
    msg = str(e.value)
    assert "Only 0 of 12" in msg and "'AAAC00'" in msg and "'AAAC00-1'" in msg and "strip_suffix" in msg
    assert "author" not in a.obs  # nothing changed
    out = li.import_labels(a, "author", t, strip_suffix=True)
    assert out["n_matched"] == 12 and (a.obs["author"] == "T").all()
    assert out["n_unassigned"] == 0 and out["n_labels"] == 1  # QA-8: no "with Unassigned" when none is


def test_hints_for_an_r_style_suffix_and_a_sample_prefix():
    a = _adata()
    a.obs_names = [f"{'CTRL' if i < 6 else 'STIM'}_AAAC{i:02d}.1" for i in range(12)]
    with pytest.raises(LabelError, match="'.1' suffix removed"):
        li.import_labels(a, "c", li.LabelTable(labels=["T"] * 12, barcodes=[f"CTRL_AAAC{i:02d}" for i in range(12)]))
    with pytest.raises(LabelError, match="prefix is set aside .e.g. 'CTRL_AAAC00.1'"):
        li.import_labels(a, "c", li.LabelTable(labels=["T"] * 12, barcodes=[f"AAAC{i:02d}.1" for i in range(12)]))
    a.obs_names = [f"{'CTRL' if i < 6 else 'STIM'}_AAAC{i % 6:02d}" for i in range(12)]
    with pytest.raises(LabelError, match="in several samples"):
        li.import_labels(a, "c", li.LabelTable(labels=["T"] * 6, barcodes=[f"AAAC{i:02d}" for i in range(6)]))


def test_min_match_lets_a_table_with_other_cells_through():
    a = _adata()
    rows = [(f"AAAC{i:02d}-1", "T") for i in range(6)] + [(f"OTHER{i}", "T") for i in range(4)]
    t = li.read_table(_table(rows), "x.csv")
    with pytest.raises(LabelError, match="6 of 10"):
        li.import_labels(a, "author", t)
    assert li.import_labels(a, "author", t, min_match=0.5)["n_extra"] == 4


def test_by_order_needs_one_row_per_cell():
    a = _adata()
    vals = ["T"] * 6 + ["B"] * 6
    out = li.import_labels(a, "by_order", li.LabelTable(labels=vals))
    assert out["match"] == "order" and list(a.obs["by_order"]) == vals
    with pytest.raises(LabelError, match="14 rows and the dataset 12 cells"):
        li.import_labels(a, "by_order2", li.LabelTable(labels=vals + ["T", "T"]))
    t = li.read_table(_table([(str(i), v) for i, v in enumerate(vals)], ("n", "lab")), "o.tsv",
                      label_column="lab", by_order=True)
    assert t.barcodes is None and t.source["barcode_column"] is None


def _two_samples(n: int = 12) -> ad.AnnData:
    a = _adata(n)
    a.obs["sample"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n // 2))
    return a


def test_by_order_checks_the_rows_sample_by_sample():
    """A table whose samples are in another order (B then A) has the right row count; with a sample
    column it is refused, naming the samples, and in the dataset's order it is checked and kept."""
    a = _two_samples()
    good = [("A", "T")] * 6 + [("B", "M")] * 6
    t = li.read_table(_table(good, ("donor", "label")), "o.csv", label_column="label", by_order=True)
    out = li.import_labels(a, "ok", t)
    assert out["order_check"] == {"table_column": "donor", "dataset_column": "sample", "n_samples": 2,
                                  "lined_up": True}
    assert list(a.obs["ok"]) == ["T"] * 6 + ["M"] * 6
    swapped = [("B", "M")] * 6 + [("A", "T")] * 6
    t = li.read_table(_table(swapped, ("donor", "label")), "o.csv", label_column="label", by_order=True)
    with pytest.raises(LabelError) as e:
        li.import_labels(a, "bad", t)
    msg = str(e.value)
    assert "'donor' does not line up with the dataset's 'sample'" in msg
    assert "2 of 2 samples" in msg and "counts per sample agree" in msg
    assert "A: 6 table rows at 7-12, 6 dataset cells at 1-6" in msg
    assert "bad" not in a.obs.columns
    # rows per sample differ (one A cell missing, one B row extra): still 12 rows in all
    interleaved = _two_samples()
    interleaved.obs["sample"] = pd.Categorical(["A", "B"] * 6)
    t = li.read_table(_table(good, ("donor", "label")), "o.csv", label_column="label", by_order=True)
    with pytest.raises(LabelError, match="A: 6 table rows at 1-6, 6 dataset cells at 1, 3, 5, ..."):
        li.import_labels(interleaved, "bad", t)
    shifted = [("A", "T")] * 5 + [("B", "M")] * 7
    t = li.read_table(_table(shifted, ("donor", "label")), "o.csv", label_column="label", by_order=True)
    with pytest.raises(LabelError, match="rows per sample differ"):
        li.import_labels(a, "bad", t)


def test_by_order_sample_column_named_or_absent():
    a = _two_samples()
    rows = [("A", "x", "T")] * 6 + [("B", "y", "M")] * 6
    t = li.read_table(_table(rows, ("s", "other", "label")), "o.csv", label_column="label", by_order=True,
                      sample_column="s")
    assert list(t.others) == ["s"]
    assert li.import_labels(a, "named", t, dry_run=True)["order_check"]["table_column"] == "s"
    t = li.read_table(_table(rows, ("s", "other", "label")), "o.csv", label_column="label", by_order=True,
                      sample_column="other")
    with pytest.raises(LabelError, match="not all in any column of the dataset"):
        li.import_labels(a, "named", t, dry_run=True)
    with pytest.raises(LabelError, match="No column 'nope'"):
        li.read_table(_table(rows, ("s", "other", "label")), "o.csv", label_column="label", by_order=True,
                      sample_column="nope")
    # no sample column: the plain count check, and the answer says nothing was checked
    t = li.read_table(_table([("x", "T")] * 12, ("other", "label")), "o.csv", label_column="label",
                      by_order=True)
    assert li.import_labels(a, "plain", t, dry_run=True)["order_check"] is None


def test_into_an_existing_column_says_how_many_cells_change():
    a = _adata()
    li.import_labels(a, "c", li.LabelTable(labels=["x"] * 6 + ["y"] * 6))
    out = li.import_labels(a, "c", li.LabelTable(labels=["y", "y", "z"], barcodes=["AAAC00-1", "AAAC07-1",
                                                                                    "AAAC08-1"]),
                           new_column=False, min_match=0)
    assert out["n_matched"] == 3 and out["n_changed"] == 2  # AAAC07 was y already
    assert list(a.obs["c"]) == ["y"] + ["x"] * 5 + ["y", "y", "z"] + ["y"] * 3
    step = provenance.read_provenance(a)["history"][-1]
    assert step["params"]["n_changed"] == 2 and step["params"]["new_column"] is False


def test_blank_labels_and_duplicates():
    a = _adata()
    rows = [("AAAC00-1", "T"), ("AAAC00-1", "T"), ("AAAC01-1", "NA"), ("AAAC02-1", "")]
    out = li.import_labels(a, "c", li.read_table(_table(rows), "x.csv"), min_match=0)
    assert out["n_blank"] == 2 and out["n_duplicates"] == 1 and out["n_matched"] == 1
    assert a.obs["c"].iloc[1] == UNASSIGNED
    with pytest.raises(LabelError, match="more than one label"):
        li.import_labels(a, "d", li.read_table(_table([("AAAC00-1", "T"), ("AAAC00-1", "B")]), "x.csv"))


def test_existing_user_column_keeps_the_cells_the_table_leaves_out():
    a = _adata()
    li.import_labels(a, "c", li.LabelTable(labels=["x"] * 12))
    out = li.import_labels(a, "c", li.LabelTable(labels=["y"], barcodes=["AAAC03-1"]), new_column=False)
    assert out["created"] is False
    assert a.obs["c"].iloc[3] == "y" and (a.obs["c"].drop(a.obs_names[3]) == "x").all()
    with pytest.raises(LabelError, match="not a user annotation column"):
        li.import_labels(a, "leiden", li.LabelTable(labels=["y"], barcodes=["AAAC03-1"]), new_column=False)
    with pytest.raises(LabelError) as e:  # a new column of an existing name
        li.import_labels(a, "c", li.LabelTable(labels=["y"], barcodes=["AAAC03-1"]))
    assert e.value.status == 409


def test_read_table_tsv_gzip_columns_and_dry_run():
    a = _adata()
    data = gzip.compress(_table([(f"AAAC{i:02d}-1", f"s{i % 2}", "T") for i in range(12)],
                                ("cell", "sample", "type"), sep="\t"))
    with pytest.raises(LabelError, match="Name the label column"):
        li.read_table(data, "t.tsv.gz")
    t = li.read_table(data, "t.tsv.gz", label_column="type")
    assert t.source == {"file": "t.tsv.gz", "barcode_column": "cell", "label_column": "type", "sep": "\t"}
    out = li.import_labels(a, "c", t, dry_run=True)
    assert out["dry_run"] is True and out["n_matched"] == 12 and "c" not in a.obs
    with pytest.raises(LabelError, match="No column 'nope'"):
        li.read_table(data, "t.tsv.gz", label_column="nope")


def test_a_label_scview_cannot_store_is_refused_with_its_row():
    a = _adata()
    with pytest.raises(LabelError, match="Row 2: Label is too long"):
        li.import_labels(a, "c", li.LabelTable(labels=["T", "x" * 200], barcodes=["AAAC00-1", "AAAC01-1"]))


# --- replay -----------------------------------------------------------------------------------

def test_replay_reads_the_table_again_beside_the_script(tmp_path):
    inp = _adata()
    a = inp.copy()
    rows = [(f"AAAC{i:02d}", "T cell" if i < 4 else "B cell") for i in range(10)]
    (tmp_path / "authors.csv").write_bytes(_table(rows))
    li.import_labels(a, "author", li.read_table((tmp_path / "authors.csv").read_bytes(), "authors.csv"),
                     strip_suffix=True)
    assert replay_script.describe(a)[-1]["status"] == "approximate"
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "out.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(a, dataset_id="t", dataset_name="t"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = ad.read_h5ad(dst)
    assert list(out.obs["author"].astype(str)) == list(a.obs["author"].astype(str))


def test_inline_import_is_not_replayable_and_says_how_to_join():
    a = _adata()
    li.import_labels(a, "author", li.LabelTable(labels=["T"], barcodes=["AAAC00-1"], source={"input": "mapping"}))
    d = replay_script.describe(a)[-1]
    assert d["status"] == "not_replayable"
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert "cell_metadata.csv" in script


# --- API --------------------------------------------------------------------------------------

def _register(dm, a: ad.AnnData | None = None) -> str:
    d = Path(dm.data_dir) / "uploads" / "li"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "li", "name": "li", "status": "ready"}))
    (a if a is not None else _adata()).write_h5ad(d / "data.h5ad")
    return "li"


async def test_import_route_saves_and_reloads(client, dataset_manager):
    did = _register(dataset_manager)
    body = {"column": "author", "labels": {f"AAAC{i:02d}-1": "T" for i in range(12)}}
    r = await client.post(f"{API}/datasets/{did}/labels/import", json={**body, "dry_run": True})
    assert r.status_code == 200 and r.json()["dry_run"] is True
    r = await client.post(f"{API}/datasets/{did}/labels/import", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["n_matched"] == 12
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert (adaptor.adata.obs["author"] == "T").all()
    cols = (await client.get(f"{API}/datasets/{did}/labels/columns")).json()["columns"]
    assert "author" in cols


async def test_import_route_refusals(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/labels/import"
    r = await client.post(url, json={"column": "a", "labels": {"x": "T"}, "values": ["T"]})
    assert r.status_code == 400 and "Give one of" in r.json()["detail"]
    r = await client.post(url, json={"column": "a", "values": ["T"] * 5})
    assert r.status_code == 400 and "5 rows and the dataset 12 cells" in r.json()["detail"]
    r = await client.post(url, json={"column": "a", "labels": {"nope": "T"}})
    assert r.status_code == 400 and "Only 0 of 1" in r.json()["detail"]
    r = await client.post(url, json={"column": "a", "labels": {"AAAC00-1": "T"}, "type": "x"})
    assert r.status_code == 422  # an unknown key is refused, not ignored


async def test_import_file_route_takes_a_csv_upload(client, dataset_manager):
    did = _register(dataset_manager)
    data = _table([(f"AAAC{i:02d}-1", "s", "B cell") for i in range(12)], ("barcode", "sample", "cell_type"))
    r = await client.post(f"{API}/datasets/{did}/labels/import-file",
                          files={"file": ("geo_labels.csv", data, "text/csv")},
                          data={"column": "geo", "label_column": "cell_type"})
    assert r.status_code == 200, r.text
    assert r.json()["n_matched"] == 12
    adaptor = await dataset_manager.get_or_load_dataset(did)
    rec = provenance.read_provenance(adaptor.adata)["history"][-1]
    assert rec["step"] == "import_labels" and rec["params"]["file"] == "geo_labels.csv"
    r = await client.post(f"{API}/datasets/{did}/labels/import-file",
                          files={"file": ("x.csv", data, "text/csv")}, data={"column": "geo2"})
    assert r.status_code == 400 and "Name the label column" in r.json()["detail"]


async def test_import_file_route_checks_the_order_per_sample_and_counts_changes(client, dataset_manager):
    did = _register(dataset_manager, _two_samples())
    url = f"{API}/datasets/{did}/labels/import-file"
    swapped = _table([("B", "M")] * 6 + [("A", "T")] * 6, ("orig.ident", "cell_type"))
    r = await client.post(url, files={"file": ("o.csv", swapped, "text/csv")},
                          data={"column": "ct", "label_column": "cell_type", "match": "order"})
    assert r.status_code == 400 and "does not line up with the dataset's 'sample'" in r.json()["detail"]
    good = _table([("A", "T")] * 6 + [("B", "M")] * 6, ("orig.ident", "cell_type"))
    r = await client.post(url, files={"file": ("o.csv", good, "text/csv")},
                          data={"column": "ct", "label_column": "cell_type", "match": "order",
                                "sample_column": "orig.ident"})
    assert r.status_code == 200, r.text
    assert r.json()["order_check"]["lined_up"] is True and r.json()["n_changed"] == 12
    fix = _table([("AAAC00-1", "M"), ("AAAC06-1", "M")], ("barcode", "cell_type"))
    r = await client.post(url, files={"file": ("fix.csv", fix, "text/csv")},
                          data={"column": "ct", "label_column": "cell_type", "new_column": "false"})
    assert r.status_code == 200, r.text
    assert r.json()["created"] is False and r.json()["n_matched"] == 2 and r.json()["n_changed"] == 1
    r = await client.post(url, files={"file": ("fix.csv", fix, "text/csv")},
                          data={"column": "sample", "label_column": "cell_type", "new_column": "false"})
    assert r.status_code == 400 and "not a user annotation column" in r.json()["detail"]


# --- MCP --------------------------------------------------------------------------------------

async def _mcp(name: str, args: dict, error: bool = False):
    from mcp import Client

    from scview.mcp.server import SERVER

    async with Client(SERVER) as c:
        r = await c.call_tool(name, args)
    assert r.is_error == error, r.content
    text = "".join(getattr(b, "text", "") for b in r.content)
    return text if error else json.loads(text)


async def test_mcp_import_labels_inline_table_and_server_path(client, dataset_manager, tmp_path, monkeypatch):
    from scview.mcp import api

    did = _register(dataset_manager)
    r = await _mcp("import_labels", {"dataset_id": did, "column": "author",
                                     "labels": {f"AAAC{i:02d}-1": "T" for i in range(6)}})
    assert r["n_matched"] == 6 and r["n_unmatched_cells"] == 6 and "next" in r
    text = await _mcp("import_labels", {"dataset_id": did, "column": "x", "labels": {"a": "T"},
                                        "values": ["T"]}, error=True)
    assert "Give one of" in text
    text = await _mcp("import_labels", {"dataset_id": did, "column": "x",
                                        "table": "cell,lab\nzz,T\n"}, error=True)
    assert "Only 0 of 1" in text

    share = tmp_path / "share"
    share.mkdir()
    (share / "lab.tsv.gz").write_bytes(gzip.compress(_table([(f"AAAC{i:02d}-1", "B") for i in range(12)],
                                                            sep="\t")))
    text = await _mcp("import_labels", {"dataset_id": did, "column": "p", "path": str(share / "lab.tsv.gz")},
                      error=True)
    assert "MCP_IMPORT_DIRS" in text
    monkeypatch.setattr(api.settings(), "MCP_IMPORT_DIRS", str(share))
    r = await _mcp("import_labels", {"dataset_id": did, "column": "p", "path": str(share / "lab.tsv.gz")})
    assert r["n_matched"] == 12
    text = await _mcp("import_labels", {"dataset_id": did, "column": "q", "path": str(tmp_path / "x.csv")},
                      error=True)
    assert "does not exist" in text
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert provenance.read_provenance(adaptor.adata)["history"][-1]["params"]["file"] == "lab.tsv.gz"
