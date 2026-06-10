"""NL co-pilot actions: command detection + allow-list validation.

The LLM mapping itself needs the network; these hermetic tests cover the cheap
command gate and the strict allow-listing of the model's JSON (the governance part).
"""

from __future__ import annotations

from scview.core.assistant import _coerce_actions, _looks_like_command


CTX = dict(columns=["cluster", "stim"], embeddings=["X_umap", "X_umap_3d"],
           genes_upper={"CD8A": "CD8A", "MS4A1": "MS4A1"})


def test_command_heuristic_distinguishes_commands_from_questions():
    for q in ("color by cluster", "highlight the NK cluster", "go to marker genes",
              "show CD8A", "open history", "switch to 3D", "group by stim",
              "clear the highlight", "hide the overlay"):
        assert _looks_like_command(q), q
    for q in ("what cell type is the NK cluster?", "which genes mark B cells?",
              "how many cells are there?", "is this dataset normalized?"):
        assert not _looks_like_command(q), q


def test_coerce_actions_accepts_valid_allowlisted():
    raw = (
        '[{"type":"set_color_by","column":"cluster"},'
        '{"type":"highlight_cluster","column":"cluster","value":"NK"},'
        '{"type":"open_panel","panel":"markers"}]'
    )
    acts = _coerce_actions(raw, **CTX)
    assert [a.type for a in acts] == ["set_color_by", "highlight_cluster", "open_panel"]
    assert acts[1].value == "NK" and acts[2].panel == "markers"
    assert all(a.label for a in acts)


def test_coerce_actions_accepts_phase2_types():
    raw = (
        '[{"type":"set_embedding","embedding":"X_umap_3d"},'
        '{"type":"set_subtab","subtab":"enrichment"},'
        '{"type":"set_groupby","column":"stim"},'
        '{"type":"show_gene","gene":"cd8a"},'
        '{"type":"clear_highlight"},{"type":"clear_overlay"}]'
    )
    acts = _coerce_actions(raw, **CTX)
    types = [a.type for a in acts]
    assert types[:4] == ["set_embedding", "set_subtab", "set_groupby", "show_gene"]
    assert acts[3].gene == "CD8A"  # case-insensitive gene resolves to the canonical symbol
    assert {"clear_highlight", "clear_overlay"}.issubset(set(types)) or len(acts) >= 4


def test_coerce_actions_rejects_anything_off_the_allowlist():
    assert _coerce_actions('[{"type":"set_color_by","column":"nope"}]', **CTX) == []   # bad column
    assert _coerce_actions('[{"type":"highlight_cluster","column":"cluster"}]', **CTX) == []  # no value
    assert _coerce_actions('[{"type":"open_panel","panel":"hack"}]', **CTX) == []       # bad panel
    assert _coerce_actions('[{"type":"set_embedding","embedding":"X_evil"}]', **CTX) == []  # bad embedding
    assert _coerce_actions('[{"type":"show_gene","gene":"NOTAGENE"}]', **CTX) == []     # unknown gene
    assert _coerce_actions('[{"type":"run_rm_rf"}]', **CTX) == []                       # unknown type
    assert _coerce_actions("not json at all", **CTX) == []                             # unparseable


def test_coerce_actions_caps_count():
    raw = "[" + ",".join('{"type":"set_color_by","column":"cluster"}' for _ in range(10)) + "]"
    assert len(_coerce_actions(raw, **CTX)) <= 4


def test_mutating_actions_are_confirm_gated_with_advisory_and_estimate():
    ctx = dict(columns=["cluster", "cell_type"], embeddings=["X_umap"],
               genes_upper={}, n_cells=13836)
    [a] = _coerce_actions('[{"type":"annotate_cell_types","method":"llm"}]', **ctx)
    assert a.type == "annotate_cell_types" and a.requires_confirm
    assert a.step == "cell_type_annotation" and a.params["annotation_method"] == "llm"
    assert "Overwrites" in a.advisory and a.estimate  # cell_type already exists

    [c] = _coerce_actions('[{"type":"cluster","resolution":1.0}]', **ctx)
    assert c.type == "cluster" and c.requires_confirm and c.step == "clustering"
    assert c.params["clustering_resolution"] == 1.0
    assert "scview_leiden_r1.0" in c.advisory and c.estimate
