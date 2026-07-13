from __future__ import annotations

import json

import numpy as np
import pytest

from anchor_exp.stable_hash import canonical_json_bytes, stable_hash128, stable_id64
from anchor_exp.workload import WorkloadFormatError, read_workload, write_workload


@pytest.mark.parametrize("endpoint_type", ["float64", "int64"])
def test_workload_round_trip(tmp_path, endpoint_type):
    dtype = np.float64 if endpoint_type == "float64" else np.int64
    rlo = np.asarray([[0, 1], [2, 3]], dtype=dtype)
    rhi = rlo + 2
    slo = np.asarray([[1, 2]], dtype=dtype)
    shi = slo + 3
    path = tmp_path / "tiny.anchor"
    manifest = write_workload(
        path,
        r_ids=np.asarray([11, 12], dtype=np.uint64),
        r_lower=rlo,
        r_upper=rhi,
        s_ids=np.asarray([21], dtype=np.uint64),
        s_lower=slo,
        s_upper=shi,
        endpoint_type=endpoint_type,
        metadata={"dataset_id": "tiny"},
    )
    loaded = read_workload(path, verify_payload=True, verify_file=True)
    assert loaded.dimension == 2
    assert loaded.endpoint_type == endpoint_type
    np.testing.assert_array_equal(loaded.r_lower, rlo)
    np.testing.assert_array_equal(loaded.s_upper, shi)
    assert loaded.file_sha256 == manifest["workload"]["sha256"]
    parsed = json.loads((tmp_path / "tiny.anchor.manifest.json").read_text())
    assert parsed["metadata"]["dataset_id"] == "tiny"


def test_empty_box_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_workload(
            tmp_path / "bad.anchor",
            r_ids=[1],
            r_lower=[[0.0]],
            r_upper=[[0.0]],
            s_ids=[2],
            s_lower=[[0.0]],
            s_upper=[[1.0]],
            endpoint_type="float64",
        )


def test_payload_corruption_is_detected(tmp_path):
    path = tmp_path / "corrupt.anchor"
    write_workload(
        path,
        r_ids=[1],
        r_lower=[[0]],
        r_upper=[[1]],
        s_ids=[2],
        s_lower=[[0]],
        s_upper=[[1]],
        endpoint_type="int64",
    )
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(WorkloadFormatError, match="SHA-256"):
        read_workload(path)


def test_canonical_hash_normalizes_mapping_order_and_negative_zero():
    assert canonical_json_bytes({"b": -0.0, "a": 1}) == b'{"a":1,"b":0.0}'
    assert stable_hash128("x", {"a": 1, "b": 2}) == stable_hash128(
        "x", {"b": 2, "a": 1}
    )
    assert 0 <= stable_id64("id", "object") < 2**64


def test_numpy_metadata_and_negative_ids():
    assert canonical_json_bytes({"x": np.asarray([1, 2], dtype=np.int64)}) == b'{"x":[1,2]}'
    with pytest.raises(ValueError, match="uint64"):
        from anchor_exp.workload import _as_ids

        _as_ids("ids", [-1])
