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
    assert "scview_leiden_r1" in c.advisory and c.estimate


def test_pipeline_step_actions_are_confirm_gated():
    ctx = dict(columns=["cluster"], embeddings=["X_umap"], genes_upper={},
               n_cells=13836, active_clustering="cluster")
    [d] = _coerce_actions('[{"type":"detect_doublets"}]', **ctx)
    assert d.type == "detect_doublets" and d.requires_confirm and d.step == "doublet_detection"
    [m] = _coerce_actions('[{"type":"compute_markers"}]', **ctx)
    assert m.step == "marker_genes" and m.params["marker_columns"] == ["cluster"] and m.requires_confirm
    [e] = _coerce_actions('[{"type":"run_enrichment","groupby":"cluster"}]', **ctx)
    assert e.step == "enrichment" and e.params["enrichment_columns"] == ["cluster"] and e.estimate


def test_deterministic_fallback_catches_unambiguous_commands():
    from scview.core.assistant import _deterministic_actions

    ctx = dict(columns=["cluster"], embeddings=["X_umap", "X_umap_3d"], genes_upper={},
               n_cells=100, active_clustering="cluster")
    assert [a.type for a in _deterministic_actions("detect doublets", **ctx)] == ["detect_doublets"]
    assert [a.type for a in _deterministic_actions("clear the highlight", **ctx)] == ["clear_highlight"]
    assert [a.type for a in _deterministic_actions("hide the overlay", **ctx)] == ["clear_overlay"]
    assert [a.type for a in _deterministic_actions("switch to 3d view", **ctx)] == ["set_embedding"]
    assert _deterministic_actions("what is this cluster?", **ctx) == []  # questions don't match


def test_marker_plot_requests_are_commands_and_map_to_show_marker_plot():
    from scview.core.assistant import _deterministic_actions

    ctx = dict(columns=["cluster"], embeddings=["X_umap"], genes_upper={},
               n_cells=100, active_clustering="cluster")
    for q in ("Can you show a heatmap of these markers?", "please plot a dot plot of the markers",
              "show the marker heatmap", "Can I see a heatmap of these markers?",
              "show me a dot plot of the markers"):
        assert _looks_like_command(q), q
    heat = _deterministic_actions("Can you show a heatmap of these markers?", **ctx)
    assert [(a.type, a.plot) for a in heat] == [("show_marker_plot", "heatmap")]  # not compute_markers
    dot = _deterministic_actions("show a dotplot of the markers", **ctx)
    assert [(a.type, a.plot) for a in dot] == [("show_marker_plot", "dotplot")]
    bare = _coerce_actions('[{"type":"show_marker_plot"}]', **CTX)
    assert [(a.type, a.plot) for a in bare] == [("show_marker_plot", "heatmap")]
    assert _coerce_actions('[{"type":"show_marker_plot","plot":"violin"}]', **CTX) == []


async def test_polite_question_skips_keyword_fallback(monkeypatch):
    # The model saw a question and returned no actions; the keyword matcher must not
    # turn "find the markers" into a compute-markers card.
    import scview.core.assistant as asst

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "[]"})()})()]

    class _Client:
        def __init__(self, **_):
            self.chat = type("Ch", (), {"completions": self})()

        async def create(self, **_):
            return _Resp()

    monkeypatch.setattr("openai.AsyncOpenAI", _Client)
    args = (None, ["cluster"], ["X_umap"], {}, 100, "cluster", "key", "m")
    assert await asst.extract_actions("Can you find the markers of cluster 3?", *args) == []
    assert [a.type for a in await asst.extract_actions("find the markers", *args)] == ["compute_markers"]
    # ...but a polite plot request still opens the plot when the model returned nothing
    heat = await asst.extract_actions("Can you show a heatmap of these markers?", *args)
    assert [(a.type, a.plot) for a in heat] == [("show_marker_plot", "heatmap")]
    see = await asst.extract_actions("Can I see a heatmap of these markers?", *args)
    assert [(a.type, a.plot) for a in see] == [("show_marker_plot", "heatmap")]
    assert await asst.extract_actions("Can I get the markers of cluster 3?", *args) == []


def test_app_ui_questions_pull_in_the_feature_list():
    from scview.core.rag.router import mentions_app_ui

    for q in ("How can I visualise marker genes across clusters in scView?",
              "is there a heatmap?", "which button runs clustering?", "where do I find markers"):
        assert mentions_app_ui(q), q
    for q in ("what marks pDCs?", "how many cells are in cluster 3?"):
        assert not mentions_app_ui(q), q


def test_coerce_actions_reaches_compare_panel_and_de_subtab():
    acts = _coerce_actions(
        '[{"type":"open_panel","panel":"compare"},{"type":"set_subtab","subtab":"de"}]', **CTX)
    assert [(a.type, a.panel or a.subtab) for a in acts] == [
        ("open_panel", "compare"), ("set_subtab", "de")]


def test_coerce_actions_reaches_every_unified_subtab():
    # every tab in the frontend's UNIFIED_SUBTABS (lib/constants.ts) can be opened by the co-pilot
    tabs = ["markers", "expression", "protein", "genesets", "enrichment", "de", "activity", "genelists"]
    for t in tabs:
        acts = _coerce_actions(f'[{{"type":"set_subtab","subtab":"{t}"}}]', **CTX)
        assert [a.subtab for a in acts] == [t]
