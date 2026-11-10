"""Cell-type annotation step: groupby resolution + method dispatch.

The CellTypist path needs a model download (network), so it's exercised manually;
these hermetic tests cover the routing logic that guards it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.config import Settings
from scview.core.pipeline import (
    PipelineParams,
    _resolve_annotation_groupby,
    _run_cell_type_annotation,
)


@pytest.fixture(autouse=True)
def no_marker_db(monkeypatch, tmp_path):
    """marker_score here tests the built-in sets: no marker databases, no download."""
    from scview.core import marker_db

    monkeypatch.setattr(marker_db, "marker_db_dir", lambda: tmp_path / "marker_db")
    monkeypatch.setattr(marker_db, "auto_download_enabled", lambda: False)


def _clustered(n: int = 20) -> ad.AnnData:
    a = ad.AnnData(X=np.random.RandomState(0).poisson(1, (n, 5)).astype("float32"))
    a.obs["cluster"] = pd.Categorical(["a", "b"] * (n // 2))
    return a


def test_groupby_resolves_to_clustering_column():
    a = _clustered()
    assert _resolve_annotation_groupby(a, PipelineParams()) == "cluster"
    # explicit override is honored
    a.obs["my_clusters"] = a.obs["cluster"]
    assert _resolve_annotation_groupby(
        a, PipelineParams(annotation_groupby="my_clusters")
    ) == "my_clusters"


def test_groupby_empty_without_clustering():
    """No clustering -> "" so CellTypist over-clusters internally (not an error)."""
    a = ad.AnnData(X=np.zeros((6, 3), dtype="float32"))
    assert _resolve_annotation_groupby(a, PipelineParams()) == ""


def test_unknown_and_unimplemented_methods_raise():
    a = _clustered()
    with pytest.raises(ValueError):
        _run_cell_type_annotation(a, PipelineParams(annotation_method="bogus"))
    with pytest.raises(ValueError):  # marker_score: no marker-set genes in this toy data
        _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score"))


def test_marker_score_assigns_top_cell_type():
    """Offline marker_score: a T-marker cluster -> T cell, a B-marker cluster -> B cell."""
    from scview.core.pipeline import _annotate_marker_score

    genes = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A", "CD79B", "CD19"]
    rng = np.random.default_rng(0)
    X = (rng.random((20, 8)) * 0.1).astype("float32")
    X[:10, :4] += 3.0  # cluster 0 high in T-cell genes
    X[10:, 4:] += 3.0  # cluster 1 high in B-cell genes
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    _annotate_marker_score(a, PipelineParams(), "cluster", "cell_type")
    m = a.uns["cell_type_markerscore_mapping"]
    assert m["0"] == "T cell" and m["1"] == "B cell"
    assert "cell_type" in a.obs and a.obs["cell_type"].nunique() == 2


def test_llm_requires_clustering():
    """LLM-from-markers needs clusters to label; with none it errors before any LLM call."""
    a = ad.AnnData(X=np.zeros((6, 3), dtype="float32"))
    with pytest.raises(ValueError):
        _run_cell_type_annotation(a, PipelineParams(annotation_method="llm"))


def test_parse_cluster_labels():
    from scview.core.pipeline import _parse_cluster_labels

    raw = "B: B cells\nCD14 Mono: Monocytes\n- NK: NK cells"
    out = _parse_cluster_labels(raw, ["B", "CD14 Mono", "NK"])
    assert out == {"B": "B cells", "CD14 Mono": "Monocytes", "NK": "NK cells"}


def test_annotation_step_is_registered():
    from scview.core.pipeline import ALL_STEPS, _STEP_PROVENANCE, _STEP_RUNNERS

    assert "cell_type_annotation" in ALL_STEPS
    assert "cell_type_annotation" in _STEP_RUNNERS
    assert "cell_type_annotation" in _STEP_PROVENANCE


def test_celltype_cluster_map_for_copilot():
    """The co-pilot's cluster -> cell-type map: prefers the recorded uns mapping,
    else majority cell type per cluster; handles the _celltypeAnno naming."""
    from scview.core.assistant import _celltype_cluster_map

    a = ad.AnnData(X=np.zeros((6, 2), dtype="float32"))
    a.obs["cluster"] = pd.Categorical(["0", "0", "1", "1", "2", "2"])
    a.obs["cell_type"] = pd.Categorical(["B", "B", "NK", "NK", "Mono", "Mono"])

    a.uns["cell_type_llm_mapping"] = {"0": "B cell", "1": "NK cell", "2": "Monocyte"}
    assert _celltype_cluster_map(a, "cell_type") == {"0": "B cell", "1": "NK cell", "2": "Monocyte"}

    del a.uns["cell_type_llm_mapping"]
    a.uns["scview_active_clustering"] = "cluster"
    assert _celltype_cluster_map(a, "cell_type") == {"0": "B", "1": "NK", "2": "Mono"}

    a.obs["cluster_celltypeAnno"] = a.obs["cell_type"].values
    assert _celltype_cluster_map(a, "cluster_celltypeAnno") == {"0": "B", "1": "NK", "2": "Mono"}


def test_annotation_computes_markers_for_its_column():
    """The new cell-type column gets its own markers; the default stays on the clustering."""
    genes = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A", "CD79B", "CD19"]
    rng = np.random.default_rng(0)
    X = (rng.random((20, 8)) * 0.1).astype("float32")
    X[:10, :4] += 3.0
    X[10:, 4:] += 3.0
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    a.uns["rank_genes_groups"] = {"params": {"groupby": "cluster"}}  # the clustering's markers
    _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score"))
    rgg = a.uns["rank_genes_groups__cell_type"]
    assert set(rgg["names"].dtype.names) == {"T cell", "B cell"}
    assert a.uns["rank_genes_groups"]["params"]["groupby"] == "cluster"


def test_top_markers_for_a_new_column_leave_the_default_markers_alone():
    from scview.core.pipeline import _top_markers_per_group

    a = _clustered(40)
    a.var_names = [f"G{i}" for i in range(5)]
    a.uns["rank_genes_groups"] = {"params": {"groupby": "leiden"}}
    out = _top_markers_per_group(a, "cluster", n=3)
    assert set(out) == {"a", "b"} and all(len(v) <= 3 for v in out.values())
    assert a.uns["rank_genes_groups"]["params"]["groupby"] == "leiden"
    assert "rank_genes_groups__cluster" in a.uns


def test_new_markers_drop_that_columns_cached_enrichment():
    from scview.core.pipeline import _markers_for_column

    a = _clustered(40)
    a.var_names = [f"G{i}" for i in range(5)]
    a.uns["enrichment__cluster__a"] = "[]"
    a.uns["ora_meta__cluster__a"] = {"n_genes": 100}
    a.uns["enrichment__other__a"] = "[]"
    _markers_for_column(a, "cluster")
    assert "rank_genes_groups__cluster" in a.uns
    assert "enrichment__cluster__a" not in a.uns and "ora_meta__cluster__a" not in a.uns
    assert "enrichment__other__a" in a.uns


def test_tissue_line_and_label_case():
    from scview.core.pipeline import _label_case, _parse_cluster_labels, _parse_tissue_line

    raw = "tissue: peripheral blood.\n0: monocytes\n1: pDC\n2: CD8+ T cells"
    assert _parse_tissue_line(raw) == "peripheral blood"
    labels = _parse_cluster_labels(raw, ["0", "1", "2"])
    assert labels == {"0": "monocytes", "1": "pDC", "2": "CD8+ T cells"}
    assert [_label_case(v) for v in labels.values()] == ["Monocytes", "pDC", "CD8+ T cells"]
    # Positional fallback never takes the tissue line for a cluster.
    assert _parse_cluster_labels("tissue: blood\nB cells\nNK cells", ["a", "b"]) == {
        "a": "B cells", "b": "NK cells"}


def test_annotator_sends_informative_markers_and_records_the_tissue(monkeypatch):
    """Kang's main T-cell cluster: its top markers were all RPS/RPL, and the model
    called it "Ribosomal cells"; monocytes were "macrophages" without a tissue.
    Score order put FTL first in CD14 monocytes, ahead of their own genes."""
    import sys
    import types

    from scview.core import pipeline as P
    from scview.core.uns_keys import markers_key

    a = _clustered()
    groups = [str(g) for g in a.obs["cluster"].cat.categories]
    # The stored result, in score order: ribosomal/mito genes and FTL (broad, small
    # fold change) lead; RARE is in 5% of the cluster; CD3D and IL7R are the markers.
    genes = ["RPS14", "RPL32", "MT-CO1", "MRPL3", "FTL", "RARE", "IL7R", "CD3D"]
    lfc = [3.0, 3.0, 3.0, 3.0, 0.4, 6.0, 1.5, 2.5]
    pts = [1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.6, 0.7]
    a.uns[markers_key("cluster")] = {
        "params": {"groupby": "cluster"},
        "names": np.rec.fromarrays([np.array(genes, dtype=object)] * 2, names=groups),
        "logfoldchanges": np.rec.fromarrays([np.array(lfc)] * 2, names=groups),
        "pts": pd.DataFrame({g: pts for g in groups}, index=genes),
    }
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            reply = "tissue: blood\n" + "\n".join(f"{g}: t cells" for g in groups)
            msg = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(
        DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")

    listing = sent[0].split("Clusters and marker genes:")[1]
    assert listing.strip().splitlines() == ["a: CD3D, IL7R, FTL", "b: CD3D, IL7R, FTL"]
    assert "tissue" in sent[0] and "monocytes, not macrophages" in sent[0]
    assert set(a.obs["cell_type"].astype(str)) == {"T cells"}
    assert a.uns["cell_type_llm_tissue"] == "blood"


def test_parse_numeric_cluster_ids_by_id_not_line_order():
    """Numeric ids were stripped with the list marks, so a reply with one extra line
    shifted every label by one (positional fallback)."""
    from scview.core.pipeline import _parse_cluster_labels

    raw = ("Here are the annotations:\ntissue: blood\n2: NK cells | CL:0000623\n0: B cells\n"
           "- 1: Monocytes\n**10**: pDC\nCluster 13: Platelets\n3. 3: T cells")
    out = _parse_cluster_labels(raw, ["0", "1", "2", "3", "10", "13"])
    assert out == {"0": "B cells", "1": "Monocytes", "2": "NK cells | CL:0000623", "3": "T cells",
                   "10": "pDC", "13": "Platelets"}
    # A cluster the model left out stays out (Unknown), it does not take a neighbour's label.
    assert _parse_cluster_labels("0: B cells\n2: NK cells", ["0", "1", "2"]) == {"0": "B cells", "2": "NK cells"}


def test_annotation_max_tokens_grows_with_clusters():
    from scview.core.pipeline import _annotation_max_tokens

    assert _annotation_max_tokens(14) >= 14 * 25 + 100
    assert _annotation_max_tokens(74) >= 74 * 25 + 100  # GTEx v9: 74 groups
    assert _annotation_max_tokens(10_000) == 8192
    assert _annotation_max_tokens(14, 2000) == _annotation_max_tokens(14) + 2000
    assert _annotation_max_tokens(10_000, 2000) == 8192


def _numbered_with_markers(n_groups: int = 6) -> ad.AnnData:
    """Numeric cluster ids "0".."n-1", each with stored markers (no wilcoxon run)."""
    from scview.core.uns_keys import markers_key

    a = ad.AnnData(X=np.random.RandomState(0).poisson(1, (n_groups * 4, 5)).astype("float32"))
    a.obs["cluster"] = pd.Categorical([str(i % n_groups) for i in range(n_groups * 4)],
                                      categories=[str(i) for i in range(n_groups)])
    groups = list(a.obs["cluster"].cat.categories)
    genes = ["CD3D", "IL7R", "MS4A1"]
    a.uns[markers_key("cluster")] = {
        "params": {"groupby": "cluster"},
        "names": np.rec.fromarrays([np.array(genes, dtype=object)] * n_groups, names=groups),
        "logfoldchanges": np.rec.fromarrays([np.array([2.0, 1.5, 1.0])] * n_groups, names=groups),
        "pts": pd.DataFrame({g: [0.7, 0.6, 0.5] for g in groups}, index=genes),
    }
    return a


LABELS = {"0": "B cells", "1": "NK cells", "2": "Monocytes", "3": "pDC", "4": "Platelets", "5": "T cells"}


def _reply(order: list[str], cut_after: int | None = None) -> str:
    """'tissue' + one line per cluster in ``order`` (not input order); with ``cut_after``,
    that many cluster lines and then the first 5 characters of the next."""
    lines = ["tissue: blood"] + [f"{g}: {LABELS[g]} | CL:none" for g in order]
    if cut_after is None:
        return "\n".join(lines)
    return "\n".join(lines[: cut_after + 1]) + "\n" + lines[cut_after + 1][:5]


def _scripted_openai(monkeypatch, replies: list[tuple[str, str]]) -> list[int]:
    """openai.OpenAI answering with ``replies`` [(content, finish_reason)] in turn;
    returns the max_tokens of each call."""
    import sys
    import types

    asked: list[int] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, max_tokens, **_):
            content, finish = replies[len(asked)]
            asked.append(max_tokens)
            msg = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(
        DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="deepseek-ai/DeepSeek-V4.1-Flash"))
    return asked


def test_annotator_retries_a_cut_reply_with_more_room(monkeypatch):
    """A reply cut at max_tokens left the later clusters Unknown with only a log line.
    Replies name clusters out of input order, so a positional fallback cannot hide."""
    from scview.core import pipeline as P

    a = _numbered_with_markers()
    asked = _scripted_openai(monkeypatch, [
        (_reply(["4", "2", "0", "5", "1", "3"], cut_after=2), "length"),  # 4, 2 named; "0: B " cut
        (_reply(["5", "3", "1", "0", "4", "2"]), "stop"),
    ])
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")

    assert len(asked) == 2 and asked[1] > asked[0] == P._annotation_max_tokens(6)
    assert a.uns["cell_type_llm_mapping"] == LABELS
    assert "llm_retry" in fit and "llm_reply_cut" not in fit


def test_annotator_says_when_the_reply_is_still_cut(monkeypatch):
    from scview.core import pipeline as P

    a = _numbered_with_markers()
    _scripted_openai(monkeypatch, [
        (_reply(["4", "2", "0", "5", "1", "3"], cut_after=1), "length"),  # 4 named; "2: Mo" cut
        (_reply(["5", "3", "1", "0", "4", "2"], cut_after=3), "length"),  # 5, 3, 1 named; "0: B " cut
    ])
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")

    mapping = a.uns["cell_type_llm_mapping"]
    # The reply naming more clusters is kept; the cut-off line is not a label ("B").
    assert {g: mapping[g] for g in ("5", "3", "1")} == {g: LABELS[g] for g in ("5", "3", "1")}
    assert {mapping[g] for g in ("0", "2", "4")} == {"Unknown"}
    assert fit["llm_reply_cut"].startswith("reply cut at")
    assert "3 of 6 clusters left Unknown (0, 2, 4)" in fit["llm_reply_cut"]
    # Recorded with the step, so History shows it
    from scview.core.pipeline import _run_cell_type_annotation

    _scripted_openai(monkeypatch, [(_reply(["5"]), "length"), (_reply(["5", "3"]), "length")])
    assert "llm_reply_cut" in _run_cell_type_annotation(
        a, PipelineParams(annotation_method="llm", annotation_groupby="cluster"))


def test_annotator_retry_keeps_the_first_reply_when_the_second_names_fewer(monkeypatch):
    from scview.core import pipeline as P

    a = _numbered_with_markers()
    _scripted_openai(monkeypatch, [
        (_reply(["4", "2", "0", "5", "1", "3"], cut_after=4), "length"),  # 4, 2, 0, 5 named
        ("", "length"),  # all reasoning, no content
    ])
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")
    mapping = a.uns["cell_type_llm_mapping"]
    assert {g: mapping[g] for g in ("4", "2", "0", "5")} == {g: LABELS[g] for g in ("4", "2", "0", "5")}
    assert "2 of 6 clusters left Unknown (1, 3)" in fit["llm_reply_cut"]


def test_no_retry_for_a_whole_reply_and_a_reasoning_model_gets_room(monkeypatch):
    from scview.core import pipeline as P
    from scview.core.llm import reply_room

    a = _numbered_with_markers()
    asked = _scripted_openai(monkeypatch, [(_reply(["5", "3", "1", "0", "4", "2"]), "stop")])
    fit = P._annotate_llm(a, P.PipelineParams(annotation_llm_model="openai/gpt-oss-120b"), "cluster", "cell_type")
    assert asked == [P._annotation_max_tokens(6) + reply_room("openai/gpt-oss-120b")]
    assert not {"llm_retry", "llm_reply_cut"} & set(fit)


def test_complete_lines_drops_only_a_cut_last_line():
    from scview.core.pipeline import _complete_lines

    assert _complete_lines("tissue: x\n0: B cells\n1: NK c", True) == "tissue: x\n0: B cells\n"
    assert _complete_lines("tissue: x\n0: B cells\n", True) == "tissue: x\n0: B cells\n"
    assert _complete_lines("0: B ce", True) == ""
    assert _complete_lines("tissue: x\n0: B cells", False) == "tissue: x\n0: B cells"
