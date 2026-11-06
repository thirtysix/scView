"""``core/step_kinds.RECORDED_STEPS`` is every step kind the source can record."""

from __future__ import annotations

import ast
from pathlib import Path

import scview
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
