"""NL co-pilot actions: command detection + allow-list validation.

The LLM mapping itself needs the network; these hermetic tests cover the cheap
command gate and the strict allow-listing of the model's JSON (the governance part).
"""

from __future__ import annotations

from scview.core.assistant import _coerce_actions, _looks_like_command


def test_command_heuristic_distinguishes_commands_from_questions():
    for q in ("color by cluster", "highlight the NK cluster", "go to marker genes",
              "show CD8A", "open history", "switch to 3D"):
        assert _looks_like_command(q), q
    for q in ("what cell type is the NK cluster?", "which genes mark B cells?",
              "how many cells are there?", "is this dataset normalized?"):
        assert not _looks_like_command(q), q


def test_coerce_actions_accepts_valid_allowlisted():
    cols = ["cluster", "stim"]
    raw = (
        '[{"type":"set_color_by","column":"cluster"},'
        '{"type":"highlight_cluster","column":"cluster","value":"NK"},'
        '{"type":"open_panel","panel":"markers"}]'
    )
    acts = _coerce_actions(raw, cols)
    assert [a.type for a in acts] == ["set_color_by", "highlight_cluster", "open_panel"]
    assert acts[0].column == "cluster"
    assert acts[1].value == "NK"
    assert acts[2].panel == "markers"
    assert all(a.label for a in acts)  # every action carries a confirmation label


def test_coerce_actions_rejects_anything_off_the_allowlist():
    cols = ["cluster"]
    assert _coerce_actions('[{"type":"set_color_by","column":"nope"}]', cols) == []   # bad column
    assert _coerce_actions('[{"type":"highlight_cluster","column":"cluster"}]', cols) == []  # no value
    assert _coerce_actions('[{"type":"open_panel","panel":"hack"}]', cols) == []       # bad panel
    assert _coerce_actions('[{"type":"run_rm_rf"}]', cols) == []                       # unknown type
    assert _coerce_actions("not json at all", cols) == []                             # unparseable


def test_coerce_actions_caps_count():
    cols = ["cluster"]
    raw = "[" + ",".join('{"type":"set_color_by","column":"cluster"}' for _ in range(10)) + "]"
    assert len(_coerce_actions(raw, cols)) <= 4
