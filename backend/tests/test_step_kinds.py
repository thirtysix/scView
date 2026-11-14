"""``core/step_kinds.RECORDED_STEPS`` is every step kind the source can record, and each
kind is sorted for staleness: a step that only adds a result (Moran's I, programs, a
label column) must not make earlier stored results stale."""

from __future__ import annotations

import ast
from pathlib import Path

import scview
from scview.core import results_store
from scview.core.pipeline import ALL_STEPS
from scview.core.replay_script import _STEP_CODE
from scview.core.step_kinds import RECORDED_STEPS

SRC = Path(scview.__file__).parent

# Calls whose step is a variable, by (module, enclosing function): what they can record.
DYNAMIC = {
    ("core/pipeline.py", "record_skip"): "a pipeline step + '_skipped'",
    ("core/pipeline.py", "_record_pipeline_step"): "a pipeline step (ALL_STEPS)",
    ("core/user_labels.py", "_record"): "its callers' literals (scanned below)",
    ("core/vdj.py", "_record"): "its callers' literals (scanned below)",
}
# Modules whose ``_record(adata, "<step>", ...)`` wraps record_step.
WRAPPERS = {"core/user_labels.py", "core/vdj.py"}

# Recorded steps that change the data earlier results were computed on: the cells, the
# matrix, the clustering or a label column a result may have grouped by. Everything else
# is in ADDITIVE_STEPS or LABEL_STEPS. A new step kind goes in one or the other.
DATA_CHANGING = {
    "ingest", "integrate", *ALL_STEPS, "run_errors",
    "subcluster", "merge_clusters", "keep_resolution", "keep_hvg_setting", "keep_layout",
    "doublet_threshold", "mito_refilter",
    "protein_dsb",
    "vdj_attach", "vdj_clonotypes", "vdj_remove", "crispr_assign", "mixscape",
}


def _scan() -> tuple[set[str], set[tuple[str, str]]]:
    """(step kinds named at record_step / wrapper calls, (module, function) of calls naming
    their step by a variable)."""
    found: set[str] = set()
    dynamic: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text())
        consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                  for t in n.targets if isinstance(t, ast.Name)}

        def visit(node: ast.AST, func: str) -> None:
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
                if isinstance(child, ast.Call):
                    f = child.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                    arg = None
                    if name == "record_step":
                        arg = next((k.value for k in child.keywords if k.arg == "step"), None)
                    elif name == "_record" and rel in WRAPPERS and len(child.args) >= 2:
                        arg = child.args[1]
                    if name == "record_step" or arg is not None:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            found.add(arg.value)
                        elif isinstance(arg, ast.Name) and arg.id in consts:
                            found.add(consts[arg.id])
                        else:
                            dynamic.add((rel, func))
                visit(child, inner)

        visit(tree, "<module>")
    return found, dynamic


def test_every_recorded_step_kind_is_listed():
    found, dynamic = _scan()
    assert len(found) > 40  # the scan reads the calls at all
    assert found - set(RECORDED_STEPS) == set(), "add these to core/step_kinds.RECORDED_STEPS"
    assert dynamic == set(DYNAMIC), "a call names its step by a variable: say what it records in DYNAMIC"
    assert set(ALL_STEPS) <= set(RECORDED_STEPS)
    assert set(_STEP_CODE) <= set(RECORDED_STEPS)  # the replay script knows no kind nobody records
    assert len(set(RECORDED_STEPS)) == len(RECORDED_STEPS)


def test_every_step_kind_is_sorted_for_staleness():
    additive, label = results_store.ADDITIVE_STEPS, results_store.LABEL_STEPS
    assert not additive & label and not (additive | label) & DATA_CHANGING
    assert (additive | label) <= set(RECORDED_STEPS)
    assert set(RECORDED_STEPS) - additive - label == DATA_CHANGING, (
        "a new step kind: ADDITIVE_STEPS when it only adds a result or a new column, else DATA_CHANGING")


def test_an_additive_step_leaves_stored_results_current():
    """Moran's I (and every other additive kind) adds a step but not to the count a
    stored result is compared by; a data-changing step does."""

    class Adaptor:
        def __init__(self, steps):
            import anndata as ad
            import numpy as np

            from scview.core import provenance

            self.adata = ad.AnnData(np.zeros((3, 2), dtype=np.float32))
            for s in steps:
                provenance.record_step(self.adata, step=s, tool="test")

        def cells_fingerprint(self):
            return "cells"

    base = ["ingest", "qc_metrics", "clustering"]
    stored = results_store.fingerprint(Adaptor(base))
    for kind in sorted(results_store.ADDITIVE_STEPS):
        assert not results_store._is_stale(stored, results_store.fingerprint(Adaptor([*base, kind]))), kind
    assert results_store._is_stale(stored, results_store.fingerprint(Adaptor([*base, "filtering"])))


def test_a_hashing_call_leaves_de_and_mixscape_results_current():
    """A demux writes the hash_* columns only (QA 2026-11-14: one marked every DE and Mixscape
    result stale); one that moved antibody features out of the protein modality changed
    that modality's data, so it counts."""
    import anndata as ad
    import numpy as np

    from scview.core import provenance

    class Adaptor:
        def __init__(self, extra=None):
            self.adata = ad.AnnData(np.zeros((3, 2), dtype=np.float32))
            for s in ("ingest", "qc_metrics", "clustering"):
                provenance.record_step(self.adata, step=s, tool="test")
            if extra:
                provenance.record_step(self.adata, step="hashing_demux", tool="test", params=extra)
                provenance.record_step(self.adata, step="hashing_samples", tool="test", params={"column": "s"})

        def cells_fingerprint(self):
            return "cells"

    stored = results_store.fingerprint(Adaptor())
    plain = {"method": "htodemux", "tags": ["A", "B"], "moved_from_protein": [], "moved_back_to_protein": []}
    for kind in ("mixscape", "de", "milo", None):
        assert not results_store._is_stale(stored, results_store.fingerprint(Adaptor(plain)), kind=kind), kind
    moved = Adaptor({**plain, "moved_from_protein": ["Hashtag1"]})
    now = results_store.fingerprint(moved)
    assert results_store._is_stale(stored, now, kind="mixscape")
    reason = results_store.stale_reason(stored, now, results_store._history_steps(moved), kind="mixscape")
    assert "hashtags moved out of the protein data" in reason
