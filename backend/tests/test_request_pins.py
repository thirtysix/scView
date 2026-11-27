"""A dataset a request was handed stays readable until the request ends (core/request_pins.py).

The class: a route takes a BACKED dataset, hops to a worker thread (or waits), and reads the
matrix from the file. An edit that reads the dataset in, or the manager dropping it for
room, closed the file under the read (h5py: invalid identifier, a 500). v0.85 fixed two
routes that pinned after a wait; most routes pinned nothing. Each test below follows that
order of events through a real route: the request takes the on-disk matrix, the other event
happens, the request reads.
"""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import anndata_adaptor, assessment_gate, coexpression
from scview.core.anndata_adaptor import AnnDataAdaptor

API = "/api/v1"
SRC = Path(__file__).resolve().parents[1] / "src" / "scview"


def _register(dm, did: str, n: int = 300) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, 40)).astype(np.float32))
    obs = pd.DataFrame({"leiden": pd.Categorical(rng.choice(["0", "1", "2"], n))},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(40)]))
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return a


class _SlowRead:
    """Stands in for a route's worker: takes the on-disk matrix, waits, then reads it."""

    def __init__(self) -> None:
        self.started, self.go = threading.Event(), threading.Event()

    def read(self, adata) -> float:
        counts = adata.X  # the on-disk matrix, read a block at a time
        assert not isinstance(counts, (np.ndarray, sp.spmatrix)), "the test needs a backed dataset"
        self.started.set()
        assert self.go.wait(20)
        return float(counts[np.arange(5)].sum())


@pytest.fixture()
def backed(monkeypatch):
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)


def _coexpression(monkeypatch) -> _SlowRead:
    slow = _SlowRead()
    monkeypatch.setattr(coexpression, "coexpression",
                        lambda adata, gene, **_kw: {"gene": gene, "sum": slow.read(adata)})
    return slow


async def _started(slow: _SlowRead) -> None:
    assert await asyncio.to_thread(slow.started.wait, 20)


async def test_a_thread_read_survives_an_edit_reading_the_dataset_in(client, dataset_manager, backed, monkeypatch):
    # POST /expression/coexpression takes adaptor.adata on the loop and reads in a thread,
    # with no pin of its own. An edit's read-in (to_memory) meanwhile closed the file.
    a = _register(dataset_manager, "readin")
    slow = _coexpression(monkeypatch)
    req = asyncio.ensure_future(client.post(f"{API}/datasets/readin/expression/coexpression", json={"gene": "G1"}))
    await _started(slow)
    adaptor = dataset_manager.get_dataset("readin")
    await asyncio.to_thread(adaptor.to_memory)  # the edit, in its worker thread
    slow.go.set()
    r = await req
    assert r.status_code == 200, r.text
    assert r.json()["sum"] == pytest.approx(float(a.X[:5].sum()))
    assert adaptor._pins == 0 and not adaptor._retired  # closed once the request has ended


async def test_a_thread_read_survives_the_dataset_being_dropped_for_room(client, dataset_manager, backed, monkeypatch):
    # The same read while the manager drops the dataset (another dataset opened, or memory
    # needed): ``AnnDataAdaptor.close`` closed the backed file at once.
    a = _register(dataset_manager, "dropped")
    _register(dataset_manager, "other")
    slow = _coexpression(monkeypatch)
    dataset_manager.max_datasets = 1
    req = asyncio.ensure_future(client.post(f"{API}/datasets/dropped/expression/coexpression", json={"gene": "G1"}))
    await _started(slow)
    adaptor = dataset_manager.get_dataset("dropped")
    assert (await client.get(f"{API}/datasets/other")).status_code == 200  # opening it drops the first
    assert dataset_manager.get_dataset("dropped") is None
    slow.go.set()
    r = await req
    assert r.status_code == 200, r.text
    assert r.json()["sum"] == pytest.approx(float(a.X[:5].sum()))
    assert adaptor._pins == 0 and not adaptor._retired


async def test_a_violin_read_survives_a_read_in(client, dataset_manager, backed, monkeypatch):
    # GET /expression/violin hands the adaptor's own method to a thread: no adata is taken
    # on the loop at all, and still nothing held the file.
    a = _register(dataset_manager, "violin")
    slow = _SlowRead()
    monkeypatch.setattr(AnnDataAdaptor, "get_expression_for_violin",
                        lambda self, gene, groupby, layer=None: [{"name": "all", "sum": slow.read(self.adata)}])
    req = asyncio.ensure_future(client.get(f"{API}/datasets/violin/expression/violin?gene=G1&groupby=leiden"))
    await _started(slow)
    await asyncio.to_thread(dataset_manager.get_dataset("violin").to_memory)
    slow.go.set()
    r = await req
    assert r.status_code == 200, r.text
    assert r.json()["groups"][0]["sum"] == pytest.approx(float(a.X[:5].sum()))


async def test_shared_work_outlives_the_request_that_started_it(client, dataset_manager, backed, monkeypatch):
    # The assessment is one task that later requests share (assessment_gate.run, no budget).
    # Its first client leaves: the request ends, and with it the request's pin and the
    # route's ``reading``. The dataset is read in, and the task, which a second page is
    # waiting for, still reads the backed object it was started with.
    from scview.api.v1 import assessment

    _register(dataset_manager, "shared")
    slow = _SlowRead()
    real = assessment.assess_preprocessing

    def slowed(adata):
        slow.read(adata)
        return real(adata)

    monkeypatch.setattr(assessment, "assess_preprocessing", slowed)
    req = asyncio.ensure_future(client.get(f"{API}/datasets/shared/assessment"))
    await _started(slow)
    adaptor = dataset_manager.get_dataset("shared")
    (task,) = [t for g in assessment_gate._gates.values() for t in g.inflight.values()]
    req.cancel()  # the client went away
    with pytest.raises(asyncio.CancelledError):
        await req
    await asyncio.to_thread(adaptor.to_memory)
    slow.go.set()
    assert "species" in await task
    await asyncio.sleep(0)
    assert adaptor._pins == 0 and not adaptor._retired


async def test_a_request_that_reads_the_dataset_in_itself_does_not_keep_the_file(dataset_manager, backed):
    # An edit or a run: its own pin must not hold the old backed object (and the tables it
    # shares) until its stream ends. Another request's pin still does.
    from scview.core import request_pins

    _register(dataset_manager, "own")
    with request_pins.scope():
        adaptor = await dataset_manager.load_dataset("own")
        old = adaptor.adata
        assert adaptor._pins == 1
        await asyncio.to_thread(adaptor.to_memory)  # to_thread carries the request's scope
        assert adaptor._pins == 0 and not adaptor._retired and not old.file.is_open
    with request_pins.scope():
        adaptor = await dataset_manager.load_dataset("own")
        assert adaptor._pins == 1
    assert adaptor._pins == 0

    _register(dataset_manager, "run")
    with request_pins.scope():
        adaptor = await dataset_manager.load_dataset("run")
        old = adaptor.adata
        with request_pins.scope():  # another request reading it
            assert await dataset_manager.load_dataset("run") is adaptor
            dataset_manager_evict = dataset_manager.evict
            with request_pins.scope():  # a third, a run, drops it: only its own pin goes
                await dataset_manager.load_dataset("run")
                dataset_manager_evict("run")
            assert old.file.is_open and adaptor._pins == 2
        assert old.file.is_open
    assert not old.file.is_open and adaptor._pins == 0


async def test_nothing_is_pinned_outside_a_request_or_after_it(dataset_manager, backed):
    from scview.core import request_pins

    _register(dataset_manager, "out")
    adaptor = await dataset_manager.load_dataset("out")
    assert adaptor._pins == 0
    with request_pins.scope() as s:
        pass
    token = request_pins._current.set(s)  # a task that outlived its request
    try:
        assert await dataset_manager.load_dataset("out") is adaptor
        assert adaptor._pins == 0  # no pin nobody would release
        release = request_pins.keep()
        release()
    finally:
        request_pins._current.reset(token)


# -- static guards -------------------------------------------------------------------------

def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            yield node, (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))


def test_the_request_scope_is_the_outermost_middleware():
    from scview.core import request_pins
    from scview.main import app

    assert app.user_middleware[0].cls is request_pins.Middleware, (
        "request_pins.Middleware must be added last (outermost): a middleware outside it "
        "runs the routes in a context that has no request scope, and nothing is pinned.")


def test_the_manager_pins_every_adaptor_it_hands_out():
    tree = ast.parse((SRC / "core" / "dataset_manager.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DatasetManager")
    bad = []
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name in ("_load", "_handed"):
            continue
        if "AnnDataAdaptor" not in ast.unparse(fn.returns or ast.Constant(None)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                value = ast.unparse(node.value)
                if "_handed(" not in value and "self.load_dataset(" not in value:  # itself handed
                    bad.append(f"{fn.name}:{node.lineno}")
    assert not bad, (
        f"DatasetManager returns an adaptor without pinning it for the request ({bad}): "
        "return self._handed(adaptor), so a backed file stays open under the request's reads.")


# Work started in api/ or mcp/ that goes on after its request has ended: the request's pin
# is gone by then. Each entry says what holds the dataset instead.
OUTLIVES_REQUEST = {
    ("mcp/jobs.py", "create_task"): "a job calls routes in process: each call is a request of its own",
    ("api/v1/export_rds.py", "BackgroundTask"): "deletes the job's files after the download; reads no dataset",
}
STARTERS = {"create_task", "BackgroundTask", "BackgroundTasks", "add_task", "Thread", "run_coroutine_threadsafe"}


def test_api_code_gets_its_datasets_from_the_manager_and_starts_no_unpinned_background_work():
    bad = []
    for folder in ("api", "mcp"):
        for path in sorted((SRC / folder).rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text())
            for node, name in _calls(tree):
                if name == "AnnDataAdaptor":
                    bad.append(f"{rel}:{node.lineno} builds its own AnnDataAdaptor: ask the dataset manager "
                               "(get_or_load_dataset), whose adaptors are pinned until the request ends")
                elif name in STARTERS and (rel, name) not in OUTLIVES_REQUEST:
                    bad.append(f"{rel}:{node.lineno} starts work with {name} that outlives the request: its "
                               "datasets are unpinned when the response is sent. Await it in the request, or "
                               "take request_pins.keep() and release it when the work ends, then list it in "
                               "OUTLIVES_REQUEST with what holds the dataset")
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "_datasets":
                    bad.append(f"{rel}:{node.lineno} reads DatasetManager._datasets: use get_dataset / "
                               "get_or_load_dataset (pinned until the request ends)")
    assert not bad, "\n".join(bad)


# MCP tools that read a dataset themselves, not through a route (no HTTP request holds it).
MCP_OWN_READS_OK = {
    "_adaptor": "the helper itself",
    "_run_result": "a job's summary note: reads obs columns only, nothing from the file",
}


def test_an_mcp_tool_that_reads_a_dataset_itself_has_its_own_scope():
    bad = []
    for path in sorted((SRC / "mcp").glob("*.py")):
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name in MCP_OWN_READS_OK:
                continue
            loads = any(name in ("_adaptor", "get_or_load_dataset", "load_dataset", "get_dataset")
                        for _n, name in _calls(fn))
            if loads and "_own_reads" not in [ast.unparse(d) for d in fn.decorator_list]:
                bad.append(f"{path.name}:{fn.lineno} {fn.name}")
    assert not bad, (
        f"{bad}: an MCP tool that loads a dataset itself runs outside any HTTP request, so nothing pins "
        "the dataset under its reads. Call the route (api.get/post), or decorate the tool with "
        "@_own_reads below @tool (mcp/tools.py).")


async def test_the_plot_tool_pins_the_dataset_it_reads(dataset_manager, backed, monkeypatch):
    from scview.mcp import api, tools

    _register(dataset_manager, "tool")
    monkeypatch.setattr(api, "dataset_manager", lambda: dataset_manager)
    seen = []

    async def fake_get(*_a, **_k):
        return {}

    async def reads(dataset_id):
        adaptor = await tools._adaptor(dataset_id)
        seen.append(adaptor._pins)
        return adaptor

    monkeypatch.setattr(tools, "_get", fake_get)
    adaptor = await tools._own_reads(reads)("tool")
    assert seen == [1] and adaptor._pins == 0
    assert tools.plot.__wrapped__.__name__ == "plot"  # the tool itself is wrapped


# create_task in core/: tasks that are not awaited by the request that starts them.
CORE_TASKS = {
    "core/assessment_gate.py": "request_pins.keep() + _pin_now until the task is done",
    "core/conversion.py": "converts an upload: no open dataset",
    "core/rds_export.py": "the job loads the dataset itself, inside dataset_edit (locked: never dropped)",
    "core/ingestion/session.py": "converts staged files: no open dataset",
}


def test_core_tasks_that_outlive_a_request_say_what_holds_their_dataset():
    bad = []
    for path in sorted((SRC / "core").rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if any(name == "create_task" for _n, name in _calls(ast.parse(path.read_text()))) and rel not in CORE_TASKS:
            bad.append(rel)
    assert not bad, (
        f"{bad}: a task made with create_task is not awaited by its request, so the request's pin "
        "(core/request_pins.py) ends before it. Take request_pins.keep() when it starts and release it "
        "when it ends (as assessment_gate.run does), or say in CORE_TASKS why it needs none.")
    gate = (SRC / "core" / "assessment_gate.py").read_text()
    assert "request_pins.keep()" in gate
