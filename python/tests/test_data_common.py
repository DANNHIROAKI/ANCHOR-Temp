from __future__ import annotations

import pathlib

import numpy as np
import pytest

from anchor_exp.hf_real import (
    IMPORTER_ID,
    DataPreparationError,
    load_source_lock,
    verify_collection,
    write_collection_manifest,
)
from anchor_exp.stable_hash import hash_file
from anchor_exp.workload import write_workload


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY / "data_sources.lock.json"


def _metadata() -> dict:
    source = load_source_lock(LOCK_PATH)["sources"]["cmab"]
    return {
        "dataset_id": "CMAB-1M",
        "real_dataset": "CMAB-1M",
        "source_identifier": f"{source['repo_id']}@{source['revision']}",
        "source": {
            "kind": "huggingface_dataset",
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "source_lock_sha256": hash_file(LOCK_PATH),
            "assets": source["assets"],
        },
        "importer_id": IMPORTER_ID,
        "importer_sha256": hash_file(
            REPOSITORY / "python" / "anchor_exp" / "hf_real.py"
        ),
        "preprocessing_config_sha256": "d" * 64,
    }


def _write_tiny_collection(
    directory: pathlib.Path,
    *,
    lock_digest: str | None = None,
    r_id: int = 1,
    name: str = "tiny.bin",
) -> None:
    write_workload(
        directory / name,
        r_ids=np.asarray([r_id], dtype=np.uint64),
        r_lower=np.asarray([[0.0, 0.0]]),
        r_upper=np.asarray([[1.0, 1.0]]),
        s_ids=np.asarray([2], dtype=np.uint64),
        s_lower=np.asarray([[0.5, 0.5]]),
        s_upper=np.asarray([[1.5, 1.5]]),
        endpoint_type="float64",
        metadata=_metadata(),
    )
    (directory / "sidecar.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    write_collection_manifest(
        directory,
        dataset_id="CMAB-1M",
        source_lock_sha256=lock_digest or hash_file(LOCK_PATH),
        workloads=[{"file_name": name, "dimension": 2, "N": 2}],
        extra_files=["sidecar.jsonl"],
    )


def test_collection_manifest_detects_artifact_tampering(tmp_path: pathlib.Path) -> None:
    _write_tiny_collection(tmp_path)
    manifest = verify_collection(
        tmp_path, dataset_id="CMAB-1M", source_lock_path=LOCK_PATH
    )
    assert manifest["workloads"][0]["sha256"]

    (tmp_path / "tiny.bin").write_bytes(b"tampered")
    with pytest.raises(DataPreparationError, match="checksum mismatch"):
        verify_collection(tmp_path, dataset_id="CMAB-1M", source_lock_path=LOCK_PATH)


def test_collection_manifest_rejects_a_different_source_lock(
    tmp_path: pathlib.Path,
) -> None:
    _write_tiny_collection(tmp_path, lock_digest="0" * 64)
    with pytest.raises(DataPreparationError, match="different source lock"):
        verify_collection(tmp_path, dataset_id="CMAB-1M", source_lock_path=LOCK_PATH)


def test_collection_manifest_detects_sidecar_tampering(tmp_path: pathlib.Path) -> None:
    _write_tiny_collection(tmp_path)
    (tmp_path / "sidecar.jsonl").write_text('{"id":2}\n', encoding="utf-8")
    with pytest.raises(DataPreparationError, match="checksum mismatch"):
        verify_collection(tmp_path, dataset_id="CMAB-1M", source_lock_path=LOCK_PATH)
