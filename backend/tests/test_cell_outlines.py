import numpy as np
import pytest

from scview.core import cell_outlines as co


def _store(tmp_path):
    path = co.store_path(tmp_path, "d1", "cell")
    xy = np.array([[0, 0], [1, 0], [1, 1], [5, 5], [6, 5], [6, 6], [5, 6], [9, 9], [9, 8], [8, 8]], dtype=float)
    co.write(path, ["a", "b", "c"], np.array([0, 3, 7, 10]), xy, source="test")
    return path


def test_read_follows_obs_order_and_leaves_missing_cells_empty(tmp_path):
    offsets, xy = co.read_for(_store(tmp_path), ["c", "gone", "a"])
    assert offsets.tolist() == [0, 3, 3, 6]
    assert xy[:3].tolist() == [[9, 9], [9, 8], [8, 8]]
    assert xy[3:].tolist() == [[0, 0], [1, 0], [1, 1]]


def test_read_with_no_cell_in_the_store(tmp_path):
    offsets, xy = co.read_for(_store(tmp_path), ["x", "y"])
    assert offsets.tolist() == [0, 0, 0]
    assert xy.shape == (0, 2)


def test_wire_format(tmp_path):
    offsets, xy = co.read_for(_store(tmp_path), ["b"])
    raw = co.encode(offsets, xy)
    head = np.frombuffer(raw, "<i4", 2)
    assert head.tolist() == [1, 4]
    assert np.frombuffer(raw, "<i4", 2, offset=8).tolist() == [0, 4]
    assert np.frombuffer(raw, "<f4", offset=16).tolist() == [5, 5, 6, 5, 6, 6, 5, 6]


def test_kinds_and_remove(tmp_path):
    assert co.kinds_of(tmp_path, "d1") == []
    _store(tmp_path)
    assert co.kinds_of(tmp_path, "d1") == ["cell"]
    co.remove(tmp_path, "d1")
    assert co.kinds_of(tmp_path, "d1") == []


@pytest.mark.parametrize("offsets,xy", [
    ([0, 3], np.zeros((3, 2))),  # one offset short of two cells
    ([1, 2, 3], np.zeros((2, 2))),  # does not start at 0
    ([0, 2, 1], np.zeros((1, 2))),  # goes backwards
    ([0, 1, 2], np.zeros((3, 2))),  # more vertices than the offsets say
])
def test_write_refuses_what_does_not_fit_and_leaves_nothing(tmp_path, offsets, xy):
    path = tmp_path / "x.h5"
    with pytest.raises(co.OutlineError):
        co.write(path, ["a", "b"], np.array(offsets), xy)
    assert not path.exists()
    assert not path.with_name("x.h5.part").exists()


def test_unknown_kind(tmp_path):
    with pytest.raises(co.OutlineError):
        co.store_path(tmp_path, "d1", "membrane")
