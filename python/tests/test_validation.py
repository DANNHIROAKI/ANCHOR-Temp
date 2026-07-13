from __future__ import annotations

import numpy as np
import pytest

from anchor_exp.validation import PAIR_DTYPE, holm_rejections, materialize_join, validate_pairs
from anchor_exp.workload import read_workload, write_workload


def _tiny(tmp_path):
    path = tmp_path / "validation.bin"
    write_workload(
        path,
        r_ids=[1, 2, 3],
        r_lower=[[0], [1], [3]],
        r_upper=[[1], [3], [4]],
        s_ids=[10, 11, 12],
        s_lower=[[1], [2], [3]],
        s_upper=[[2], [3], [4]],
        endpoint_type="int64",
    )
    return read_workload(path)


def test_materialized_join_uses_strict_half_open_semantics(tmp_path):
    workload = _tiny(tmp_path)
    join = materialize_join(workload)
    assert {tuple(map(int, pair)) for pair in join.tolist()} == {
        (2, 10),
        (2, 11),
        (3, 12),
    }
    validate_pairs(workload, join)


def test_invalid_pair_is_rejected(tmp_path):
    workload = _tiny(tmp_path)
    pair = np.asarray([(1, 10)], dtype=PAIR_DTYPE)
    with pytest.raises(AssertionError, match="not a strict"):
        validate_pairs(workload, pair)


def test_holm_step_down():
    assert holm_rejections({"a": 0.001, "b": 0.004, "c": 0.2}, family_level=0.01) == {
        "a",
        "b",
    }
