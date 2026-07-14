from __future__ import annotations

import copy
import json
import pathlib

import pytest

from anchor_exp.aggregate import (
    aggregate_records,
    derive_count_consistency,
    publication_preflight,
    resource_boundaries,
)
from anchor_exp.environment import capture_environment
from anchor_exp.protocol import (
    OFFICIAL_MEMORY_CAP_BYTES,
    OFFICIAL_MEMORY_CAP_GIB,
    OFFICIAL_TIMEOUT_SECONDS,
)
from anchor_exp.experiments import (
    ExperimentCase,
    _finalize_reconciled_jsonl,
    _validate_real_cross_workload_identities,
    _validate_synthetic_manifest,
    expand_runs,
)
from anchor_exp.stable_hash import (
    canonical_json_bytes,
    hash_file,
    stable_hash,
    stable_hash_hex,
)


ALGORITHMS = ("AC", "AS", "SweepRT", "LiftedRT")


def test_official_resource_limits_are_frozen():
    assert OFFICIAL_TIMEOUT_SECONDS == 1800
    assert OFFICIAL_MEMORY_CAP_GIB == 950
    assert OFFICIAL_MEMORY_CAP_BYTES == 950 * 1024 ** 3
    assert OFFICIAL_MEMORY_CAP_BYTES > 900_000_000_000


def test_default_run_order_is_fixed_ac_as_sweeprt_liftedrt():
    case = ExperimentCase(
        experiment_id="x",
        dataset_type="real",
        dataset_id="d",
        workload_id="w",
        workload_path=pathlib.Path("w.bin"),
        sweep="t-sweep",
        x_name="t",
        x_value=10,
        t=10,
        dimension=2,
        data_seed=None,
        level=None,
        parameters={"d": 2, "t": 10},
        experiment_config_sha256="e" * 64,
    )
    config = {
        "algorithms": list(ALGORITHMS),
        "process_repeat_ids": [0],
        "sample_seed_id": 0,
    }
    runs = expand_runs([case], config)
    assert [(run.task, run.measurement_mode) for run in runs[:8]] == (
        [("oneshot", "time")] * 4 + [("oneshot", "memory")] * 4
    )
    assert [run.algorithm for run in runs[:4]] == list(ALGORITHMS)
    assert [run.algorithm for run in runs[4:8]] == list(ALGORITHMS)
    assert {run.task for run in runs} == {"oneshot", "count-only", "prepared-query"}
    assert (
        runs[0].sample_seed_hex
        == stable_hash("sample-master-seed", "x", "d", "w", "AC", 0).hex()
    )


def test_synthetic_admission_rejects_swapped_configuration():
    configuration = {
        "n_R": 10,
        "n_S": 10,
        "dimension": 2,
        "alpha_target": 10.0,
        "shape_sigma": 0.0,
        "solver": {"certification_checkpoints": 15},
    }
    manifest = {
        "workload": {
            "endpoint_type": "float64",
            "n_R": 10,
            "n_S": 10,
        },
        "metadata": {
            "dataset_id": "Alacarte",
            "experiment_id": "exp",
            "workload_id": "wid",
            "data_seed": 3,
            "model_version": "alacarte-binary64-scale-v2",
            "numeric_certificate_protocol": {
                "conditional_probability": "directed-binary64"
            },
            "configuration": configuration,
            "configuration_sha256": stable_hash_hex("alacarte-config", configuration),
            "coverage_solver_config_sha256": stable_hash_hex(
                "alacarte-solver-config", configuration["solver"]
            ),
            "coverage_status": "CERTIFIED",
            "coverage": {"status": "CERTIFIED", "certified": True},
        },
    }
    parameters = {
        "N": 20,
        "d": 2,
        "alpha_target": 10.0,
        "shape_sigma": 0.0,
        "data_seed": 3,
    }
    _validate_synthetic_manifest(
        experiment_id="exp",
        workload_id="wid",
        parameters=parameters,
        manifest=manifest,
        path=pathlib.Path("synthetic.bin"),
    )
    manifest["metadata"]["configuration"]["shape_sigma"] = 0.8
    with pytest.raises(ValueError, match="configuration SHA-256"):
        _validate_synthetic_manifest(
            experiment_id="exp",
            workload_id="wid",
            parameters=parameters,
            manifest=manifest,
            path=pathlib.Path("synthetic.bin"),
        )


def test_real_cross_workload_importer_identity_is_checked(tmp_path):
    def manifest(importer_sha256: str, name: str) -> dict:
        return {
            "workload": {
                "R_ids_sha256": "r",
                "S_ids_sha256": "s",
                "file_name": name,
            },
            "metadata": {
                "dataset_id": "CMAB-1M",
                "source_identifier": "DannHiroaki/CMAB-Spatial-Join-0.08B@" + "a" * 40,
                "importer_id": "anchor-hf-real-import-v2",
                "importer_sha256": importer_sha256,
                "preprocessing_config_sha256": "config",
                "source": {
                    "kind": "huggingface_dataset",
                    "repo_id": "DannHiroaki/CMAB-Spatial-Join-0.08B",
                    "revision": "a" * 40,
                    "source_lock_sha256": "b" * 64,
                    "assets": [],
                },
            },
        }

    with pytest.raises(ValueError, match="importer SHA"):
        _validate_real_cross_workload_identities(
            {
                tmp_path / "level-1.bin": manifest("a" * 64, "level-1.bin"),
                tmp_path / "level-2.bin": manifest("c" * 64, "level-2.bin"),
            }
        )


def _synthetic_records() -> list[dict]:
    records: list[dict] = []
    for data_seed in (0,):
        for repeat in (0,):
            for algorithm in ALGORITHMS:
                group = f"g-{data_seed}-{repeat}-{algorithm}"
                for mode in ("time", "memory"):
                    record = {
                        "schema_version": "anchor-run-record-v1",
                        "machine_id": "machine-a",
                        "code_commit": "commit-a",
                        "build_sha256": "b" * 64,
                        "experiment_config_sha256": "e" * 64,
                        "algorithm_config_sha256": f"cfg-{algorithm}",
                        "workload_config_sha256": f"workload-cfg-{data_seed}",
                        "experiment_id": "Alacarte-G2-t",
                        "dataset_type": "synthetic",
                        "dataset_id": "Alacarte",
                        "workload_id": f"workload-{data_seed}",
                        "workload_sha256": f"{data_seed}" * 64,
                        "subset_id": None,
                        "sweep": "t-sweep",
                        "x_name": "t",
                        "x_value": 10,
                        "t": 10,
                        "level": None,
                        "data_seed": data_seed,
                        "sample_seed_id": 0,
                        "process_repeat_id": repeat,
                        "run_group_id": group,
                        "measurement_mode": mode,
                        "task": "oneshot",
                        "algorithm": algorithm,
                        "status": "OK",
                        "W": "17",
                        "output_length": 10,
                        "OneShotTime": 1.0 + repeat / 10,
                        "PeakMemoryTotal": 1000 + repeat,
                        "InputMemory": 100,
                        "BaselineMemory": 200,
                        "PeakMemoryIncremental": 900 + repeat,
                        "PeakMemoryAux": 800 + repeat,
                        "ProcessMaxRSS": 900,
                    }
                    records.append(record)
    return records


def test_publication_preflight_accepts_complete_protocol_records():
    records = _synthetic_records()
    report = publication_preflight(records)
    assert report["status"] == "OK"
    reconciled, mismatches = derive_count_consistency(records)
    assert mismatches == []
    assert all(record["count_consistency_checked"] for record in reconciled)


def test_staging_records_are_atomically_reconciled_before_publication(tmp_path):
    staging = tmp_path / "raw.jsonl.partial"
    final = tmp_path / "raw.jsonl"
    staging.write_bytes(
        b"".join(
            canonical_json_bytes(record) + b"\n" for record in _synthetic_records()
        )
    )
    _finalize_reconciled_jsonl(staging, final)
    assert final.is_file() and not staging.exists()
    published = [json.loads(line) for line in final.read_text().splitlines()]
    assert published
    assert all(record["count_consistency_checked"] for record in published)


@pytest.mark.parametrize("mutation", ["duplicate", "repeat", "seed", "identity"])
def test_publication_preflight_rejects_mixed_or_incomplete_records(mutation):
    records = _synthetic_records()
    if mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "repeat":
        records[0]["process_repeat_id"] = 1
    elif mutation == "seed":
        records[0]["data_seed"] = 1
    else:
        records[-1]["machine_id"] = "machine-b"
    with pytest.raises(ValueError, match="preflight failed"):
        publication_preflight(records)


def test_count_mismatch_is_derived_and_excluded_from_aggregate_metrics():
    records = _synthetic_records()
    records[0]["W"] = "18"
    aggregate = aggregate_records(records)
    assert aggregate["derived_count_mismatch_count"] == 8
    assert (
        aggregate["output_consistency_mismatches"][0]["scope"] == "fixed-workload-count"
    )
    assert aggregate["metrics"] == []


def test_algorithm_resource_failure_is_preserved_but_mode_pair_is_not_admitted():
    records = _synthetic_records()
    records[0]["status"] = "OOM"
    records[0].pop("W")
    records[0].pop("output_length")
    reconciled, mismatches = derive_count_consistency(records)
    assert reconciled[0]["status"] == "OOM"
    assert reconciled[0]["consistency_admitted"] is False
    assert reconciled[1]["status"] == "OK"
    assert reconciled[1]["consistency_admitted"] is False
    assert any(item["scope"] == "time-memory-pair" for item in mismatches)


def test_sampled_memory_cap_is_reported_as_a_resource_boundary():
    boundaries = resource_boundaries(
        [
            {
                "experiment_id": "CMAB-R1-level",
                "algorithm": "AC",
                "measurement_mode": "memory",
                "task": "oneshot",
                "x_value": 1,
                "status": "OK",
            },
            {
                "experiment_id": "CMAB-R1-level",
                "algorithm": "AC",
                "measurement_mode": "memory",
                "task": "oneshot",
                "x_value": 2,
                "status": "MEMORY-CAP-EXCEEDED",
            },
        ]
    )
    assert boundaries[0]["last_successful_x"] == 1.0
    assert boundaries[0]["first_resource_failure_x"] == 2.0


def test_real_t_sweep_must_reuse_one_fixed_workload():
    records = _synthetic_records()
    for record in records:
        record["dataset_type"] = "real"
        record["dataset_id"] = "CMAB-1M"
        record["experiment_id"] = "CMAB-R2-t"
        record["data_seed"] = None
        record["workload_id"] = "cmab-fixed"
        record["workload_sha256"] = "c" * 64
    publication_preflight(records)
    records[-1]["workload_id"] = "swapped-workload"
    with pytest.raises(ValueError, match="real workload is not fixed"):
        publication_preflight(records)


def test_machine_capture_requires_matching_ok_validation_report(tmp_path):
    benchmark = tmp_path / "anchor_bench"
    benchmark.write_bytes(b"frozen benchmark")
    report = tmp_path / "validation.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "anchor-validation-report-v1",
                "status": "OK",
                "benchmark_sha256": hash_file(benchmark),
            }
        )
    )
    manifest = capture_environment(
        benchmark_path=benchmark,
        validation_report_path=report,
        memory_cap_bytes=1024,
        cpu_core=0,
        numa_node=0,
        code_commit="commit",
        memory_configuration="2 channels, DDR5-4800",
        smt_sibling_idle_confirmed=True,
        build_flags=("-O3",),
        linker_id="ld-test",
        target_isa="x86_64-test",
    )
    assert manifest["validation_report_status"] == "OK"
    assert manifest["validation_report_sha256"] == hash_file(report)
    assert manifest["host"]["memory_total_bytes"] is not None
    assert manifest["timeout_seconds"] == OFFICIAL_TIMEOUT_SECONDS
    assert manifest["setup_timeout_seconds"] == OFFICIAL_TIMEOUT_SECONDS

    report.write_text(
        json.dumps(
            {
                "schema_version": "anchor-validation-report-v1",
                "status": "DEVELOPER-SMOKE-ONLY",
                "benchmark_sha256": hash_file(benchmark),
            }
        )
    )
    with pytest.raises(ValueError, match="status OK"):
        capture_environment(
            benchmark_path=benchmark,
            validation_report_path=report,
            memory_cap_bytes=1024,
            cpu_core=0,
            numa_node=0,
            code_commit="commit",
            memory_configuration="2 channels, DDR5-4800",
            smt_sibling_idle_confirmed=True,
        )
