"""POST /expression/by-group: the per-group summary plus a "Selection" row over a cell
set (C035, the marker dot plot / heatmap linked to the scView selection)."""

from __future__ import annotations

import numpy as np
import pytest

from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.large_fixture import counts_adata, make_large, small_twin
from tests.test_expression_by_group import API, _register


async def test_selection_row_alongside_the_groups(client, dataset_manager):
    did = _register(dataset_manager)
    # X = [[2, 1], [4, 0], [0, 3], [0, 0]]; cells 1 and 2 cross the two groups.
    r = await client.post(
        f"{API}/datasets/{did}/expression/by-group",
        json={"genes": ["G0", "G1"], "groupby": "cluster", "layer": "X", "cells": [2, 1, 2]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == ["B", "A"]  # the GET's payload is all there
    assert body["mean"] == [[0.0, 1.5], [3.0, 0.5]]
    sel = body["selection"]
    assert sel["n_cells"] == 2  # duplicates dropped
    assert sel["genes"] == ["G0", "G1"]
    assert sel["mean"] == [2.0, 1.5]
    assert sel["fraction"] == [0.5, 0.5]
    assert sel["mean_expressing"] == [4.0, 3.0]


async def test_selection_only(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(
        f"{API}/datasets/{did}/expression/by-group",
        json={"genes": ["G1", "NOPE"], "layer": "X", "cells": [0], "include_groups": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == [] and body["genes"] == ["G1"]
    assert body["selection"]["mean"] == [1.0] and body["selection"]["fraction"] == [1.0]


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"genes": ["G0"], "cells": [4]}, 400),  # past the last cell
        ({"genes": ["G0"], "cells": [-1]}, 400),
        ({"genes": ["G0"], "cells": []}, 422),
        ({"genes": ["NOPE"], "cells": [0], "include_groups": False}, 404),
        ({"genes": ["G0"], "groupby": "nope", "cells": [0]}, 404),
    ],
)
async def test_bad_requests(client, dataset_manager, payload, status):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/expression/by-group", json={"layer": "X", **payload})
    assert r.status_code == status, r.text


def test_large_dataset_reads_the_selection_from_the_gene_store(tmp_path):
    counts = counts_adata()
    path = tmp_path / "twin.h5ad"
    small_twin(counts).write_h5ad(path)
    twin = AnnDataAdaptor(str(path))
    large = AnnDataAdaptor(str(make_large(tmp_path / "ds", counts)))
    try:
        assert large.is_large
        cells = np.arange(0, large.adata.n_obs, 7)
        genes = ["G10", "G30", "MT-2", "absent"]
        got = large.expression_for_cells(genes, cells)
        want = twin.expression_for_cells(genes, cells, layer="X")
        assert got["genes"] == want["genes"] == ["G10", "G30", "MT-2"]
        assert got["n_cells"] == want["n_cells"] == len(cells)
        for key in ("mean", "mean_expressing", "fraction"):
            np.testing.assert_allclose(got[key], want[key], atol=1.5e-4)
    finally:
        large.close()
