"""The co-pilot's feature guide (core/feature_guide.py): every feature with its
real path, a compact copy sent with dataset answers too (so the model names
real tabs instead of denying a feature or inventing a menu). The paths are
checked against the frontend's labels in frontend/src/lib/featureGuide.test.ts."""

from __future__ import annotations

from scview.core import feature_guide
from scview.core.assistant import SYSTEM_PROMPT, build_app_context, build_feature_context
from tests.test_selection_facts import _register

API = "/api/v1"

# Features the capture run (2026-10-26) found missing from the old guide.
NEWER = ("Activity > TFs (CollecTRI)", "Cytokine response", "Programs", "Layouts", "Protein",
         "Neighbourhoods (Milo)", "Compare conditions", "Audit this result", "GSVA", "DSB",
         "Ambient RNA", "Cell hashing", "Clonotypes", "Perturbation", "Label from a reference dataset",
         "Stored results")


def test_the_guide_lists_the_newer_features_and_says_it_is_complete():
    full = feature_guide.full_guide()
    compact = feature_guide.compact_guide()
    for name in NEWER:
        assert name in full and name in compact, name
    assert "Unified View > Communication > Compare conditions" in full
    for text in (full, compact):
        assert "These are all of scView's tabs and tools" in text
        assert "Compare > Cell-cell communication" not in text


def test_the_compact_guide_is_small_and_has_no_library():
    text, sources = build_feature_context()
    assert len(text) < 1600, len(text)  # ~350 tokens, on every dataset answer
    assert "Dataset library" not in text and "IMPORTING" not in text
    assert [s.ref for s in sources] == ["app:features"]
    # the app route keeps the full guide and the library
    ctx, app_sources = build_app_context([{"name": "demo", "n_cells": 10, "n_genes": 5}])
    assert "Dataset library" in ctx and "IMPORTING" in ctx
    assert [s.ref for s in app_sources] == ["app:features", "app:library"]


def test_the_prompt_asks_for_the_guide_s_paths():
    assert "never make up a menu path" in SYSTEM_PROMPT


async def test_a_dataset_answer_carries_the_feature_list_not_the_library(client, dataset_manager):
    did = _register(dataset_manager, "fg1")
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "Which clusters does this dataset have?"})
    assert r.status_code == 200, r.text
    body = r.json()
    refs = [s["ref"] for s in body["sources"]]
    assert "data" in body["route"] and "app" not in body["route"]
    assert "app:features" in refs and "app:library" not in refs
