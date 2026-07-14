"""Experiment-matrix expansion and isolated benchmark orchestration."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from anchor_exp.protocol import (
    OFFICIAL_MEMORY_CAP_BYTES,
    OFFICIAL_MEMORY_CAP_GIB,
    OFFICIAL_TIMEOUT_SECONDS,
)
from anchor_exp.stable_hash import (
    canonical_json_bytes,
    hash_file,
    stable_hash,
    stable_hash_hex,
)
from anchor_exp.workload import read_manifest


ALGORITHM_CLI = {
    "AC": "ac",
    "AS": "as",
    "SweepRT": "sweeprt",
    "LiftedRT": "liftedrt",
}
MEASUREMENT_MODES = ("time", "memory")
ALGORITHM_IMPLEMENTATION_ID = "anchor-cpp-exact-box-samplers-v1"

REAL_DATASET_RULES: dict[str, dict[str, Any]] = {
    "CMAB-1M": {
        "dimension": 2,
        "n_R": 471_643,
        "n_S": 471_642,
        "endpoint_type": "float64",
        "repo_id": "DannHiroaki/CMAB-Spatial-Join-0.08B",
        "revision": "41e3c90fa42fc8eede910404fe3db29ad3897b81",
        "crs_id": "CMAB-HF-Albers-Equal-Area-m",
        "levels": {1, 2, 3, 4},
    },
    "GeoLife-3D-1M": {
        "repo_id": "DannHiroaki/Geolife-Spatial-Join-0.15B",
        "revision": "a9b8439beb16de106f6ff3f54c73c6b6964d77af",
        "crs_id": "EPSG:3857-centimeter-plus-epoch-millisecond",
        "dimension": 3,
        "n_R": 500_000,
        "n_S": 500_000,
        "endpoint_type": "int64",
        "levels": {1, 2, 3},
    },
    "GeoLife-4D-1M": {
        "dimension": 4,
        "repo_id": "DannHiroaki/Geolife-Spatial-Join-0.15B",
        "revision": "a9b8439beb16de106f6ff3f54c73c6b6964d77af",
        "crs_id": "EPSG:3857-centimeter-plus-epoch-millisecond",
        "n_R": 500_000,
        "n_S": 500_000,
        "endpoint_type": "int64",
        "levels": {1, 2, 3},
    },
    "COCO-1M": {
        "dimension": 3,
        "n_R": 500_000,
        "repo_id": "DannHiroaki/COCO-Spatial-Join-1.23B",
        "revision": "2e5f2a1ba741ba1148f0b2f42209a9da4635a6cb",
        "crs_id": "COCO-original-image-pixels-plus-selected-image-index",
        "n_S": 500_000,
        "endpoint_type": "float64",
        "subset_id": "coco_hash_subset_0",
    },
}


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in text
    )


def _required_frozen_metadata_text(
    metadata: Mapping[str, Any], key: str, path: pathlib.Path
) -> str:
    value = str(metadata.get(key, "")).strip()
    if not value or "REPLACE" in value.upper():
        raise ValueError(f"real workload {path} lacks frozen metadata field {key!r}")
    return value


def _real_hf_source(
    metadata: Mapping[str, Any], path: pathlib.Path
) -> tuple[str, str, str]:
    source = metadata.get("source")
    if not isinstance(source, Mapping) or source.get("kind") != "huggingface_dataset":
        raise ValueError(f"real workload {path} lacks Hugging Face source provenance")
    repository = str(source.get("repo_id", ""))
    revision = str(source.get("revision", ""))
    lock_digest = str(source.get("source_lock_sha256", "")).lower()
    if repository.count("/") != 1 or len(revision) != 40 or not _is_sha256(lock_digest):
        raise ValueError(f"real workload {path} has incomplete pinned Hub provenance")
    if (
        metadata.get("source_repository") != repository
        or metadata.get("source_revision") != revision
    ):
        raise ValueError(f"real workload {path} has inconsistent Hub provenance")
    if metadata.get("source_identifier") != f"{repository}@{revision}":
        raise ValueError(f"real workload {path} has inconsistent source_identifier")
    assets = source.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"real workload {path} has no locked Hub assets")
    return repository, revision, lock_digest


def _validate_real_manifest(
    *,
    dataset_id: str,
    dimension: int,
    level: int | None,
    manifest: Mapping[str, Any],
    path: pathlib.Path,
) -> None:
    """Reject a correctly hashed but misidentified real-data workload."""

    if dataset_id not in REAL_DATASET_RULES:
        raise ValueError(f"no frozen real-data admission rule for {dataset_id!r}")
    rule = REAL_DATASET_RULES[dataset_id]
    workload = manifest.get("workload")
    metadata = manifest.get("metadata")
    if not isinstance(workload, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"incomplete real workload manifest for {path}")
    if metadata.get("dataset_id") != dataset_id:
        raise ValueError(f"dataset_id mismatch for {path}")
    if metadata.get("real_dataset") != dataset_id:
        raise ValueError(f"real_dataset mismatch for {path}")
    if _required_frozen_metadata_text(metadata, "crs_id", path) != rule["crs_id"]:
        raise ValueError(f"real workload CRS mismatch for {path}")
    expected_fields = {
        "dimension": dimension,
        "n_R": rule["n_R"],
        "n_S": rule["n_S"],
        "endpoint_type": rule["endpoint_type"],
    }
    if dimension != int(rule["dimension"]):
        raise ValueError(f"experiment dimension is invalid for {dataset_id}")
    for key, expected in expected_fields.items():
        actual = workload.get(key)
        if actual != expected:
            raise ValueError(
                f"real workload {path} has {key}={actual!r}, expected {expected!r}"
            )
    if int(workload.get("N_total", -1)) != int(rule["n_R"]) + int(rule["n_S"]):
        raise ValueError(f"real workload total input size mismatch for {path}")
    _required_frozen_metadata_text(metadata, "source_identifier", path)
    if metadata.get("importer_id") != "anchor-hf-real-import-v2":
        raise ValueError(f"real workload {path} has the wrong HF importer id")
    if not _is_sha256(metadata.get("importer_sha256")):
        raise ValueError(f"real workload {path} lacks importer_sha256")
    if not _is_sha256(metadata.get("preprocessing_config_sha256")):
        raise ValueError(f"real workload {path} lacks preprocessing_config_sha256")
    repository, revision, _ = _real_hf_source(metadata, path)
    if repository != rule["repo_id"] or revision != rule["revision"]:
        raise ValueError(
            f"real workload {path} uses the wrong pinned Hugging Face revision"
        )

    levels = rule.get("levels")
    if levels is not None:
        if level not in levels or metadata.get("level") != level:
            raise ValueError(f"real workload level mismatch for {path}")
    elif metadata.get("level") is not None:
        raise ValueError(f"non-level workload {path} unexpectedly declares a level")

    if dataset_id == "CMAB-1M":
        if metadata.get("split_method") != "cmab_hf_stratified_hash_tile_10km_v1":
            raise ValueError(f"CMAB split method mismatch for {path}")
        if not isinstance(metadata.get("boundary_touching_diagnostic"), Mapping):
            raise ValueError(f"CMAB boundary diagnostic is incomplete for {path}")
        if metadata.get("object_identity_fields") != ["source_file", "source_fid"]:
            raise ValueError(f"CMAB object identity fields mismatch for {path}")
        if metadata.get("object_id_method") != "uint64_lexicographic_rank_v1":
            raise ValueError(f"CMAB object id method mismatch for {path}")
        if not _is_sha256(metadata.get("source_identity_sha256")):
            raise ValueError(f"CMAB source identity digest is missing for {path}")
        expected_uid_diagnostics = {
            "published_building_uid_unique_count": 938_387,
            "published_building_uid_duplicate_groups": 4_898,
            "published_building_uid_excess_rows": 4_898,
            "published_building_uid_max_multiplicity": 2,
        }
        for key, expected in expected_uid_diagnostics.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"CMAB published building_uid diagnostic mismatch for {path}: {key}"
                )
    elif dataset_id.startswith("GeoLife-"):
        if metadata.get("split_method") != (
            "geolife_hf_user_hash_then_month_tile_largest_remainder_v1"
        ):
            raise ValueError(f"GeoLife split method mismatch for {path}")
        if metadata.get("dimension") != dimension:
            raise ValueError(f"GeoLife metadata dimension mismatch for {path}")
        for side in ("R", "S"):
            selected = metadata.get(f"selected_point_manifest_{side}")
            if (
                not isinstance(selected, Mapping)
                or selected.get("count") != 500_000
                or not _is_sha256(selected.get("sha256"))
            ):
                raise ValueError(
                    f"GeoLife selected-point manifest {side} is incomplete for {path}"
                )
        if not isinstance(metadata.get("spatial_temporal_coverage_summary"), Mapping):
            raise ValueError(f"GeoLife coverage diagnostic is incomplete for {path}")
    elif dataset_id == "COCO-1M":
        if metadata.get("image_subset_id") != rule["subset_id"]:
            raise ValueError(f"COCO subset mismatch for {path}")
        if metadata.get("proposal_stage") != "hf_published_rpn_top10000":
            raise ValueError(f"COCO proposal stage mismatch for {path}")
        selected_images = metadata.get("selected_images")
        if not isinstance(selected_images, list) or len(selected_images) != 100:
            raise ValueError(f"COCO selected-image manifest is incomplete for {path}")
        if int(metadata.get("eligible_image_count", 0)) != 123_287:
            raise ValueError(f"COCO eligible-image count is invalid for {path}")
        if not isinstance(metadata.get("rank_or_score_summary"), Mapping):
            raise ValueError(f"COCO rank/score diagnostic is incomplete for {path}")
        if metadata.get("proposal_split_method") != (
            "coco_hf_published_hash_v1_balanced_5000_5000"
        ):
            raise ValueError(f"COCO proposal split method mismatch for {path}")
        if metadata.get("proposal_identity_fields") != [
            "canonical_split",
            "coco_image_id",
            "rank",
            "rect_id",
        ]:
            raise ValueError(f"COCO proposal identity version mismatch for {path}")
        if metadata.get("coordinate_source_type") != "float32":
            raise ValueError(f"COCO published coordinate type is not frozen for {path}")
        if metadata.get("coordinate_conversion") != (
            "exact float32-to-float64 promotion"
        ):
            raise ValueError(f"COCO coordinate conversion mismatch for {path}")
        if metadata.get("model_config_id") != (
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        ):
            raise ValueError(f"COCO model config id mismatch for {path}")
        for digest_field in (
            "checkpoint_sha256",
            "upstream_builder_manifest_sha256",
            "image_subset_sha256",
        ):
            if not _is_sha256(metadata.get(digest_field)):
                raise ValueError(f"COCO {digest_field} is missing for {path}")
        sidecar = metadata.get("selected_images_sidecar")
        if (
            not isinstance(sidecar, Mapping)
            or not str(sidecar.get("file_name", "")).endswith(".json")
            or not _is_sha256(sidecar.get("sha256"))
        ):
            raise ValueError(f"COCO selected-image sidecar is incomplete for {path}")
        image_keys: set[tuple[str, int]] = set()
        for workload_z, image in enumerate(selected_images):
            if not isinstance(image, Mapping):
                raise ValueError(f"COCO selected-image entry is invalid for {path}")
            key = (str(image.get("split")), int(image.get("image_id", -1)))
            if (
                key[0] not in {"train2017", "val2017"}
                or key in image_keys
                or image.get("workload_z_idx") != workload_z
                or int(image.get("source_z_idx", -1)) < 0
                or int(image.get("width", 0)) <= 0
                or int(image.get("height", 0)) <= 0
            ):
                raise ValueError(f"COCO image identity/order is invalid for {path}")
            image_keys.add(key)
            if int(image.get("proposal_count", 0)) != 10_000 or not _is_sha256(
                image.get("proposal_rows_sha256")
            ):
                raise ValueError(
                    f"COCO published proposal provenance is incomplete for {path}"
                )
        source_shards = metadata.get("source_shards")
        if not isinstance(source_shards, list) or not source_shards:
            raise ValueError(f"COCO source-shard provenance is incomplete for {path}")
        for shard in source_shards:
            if (
                not isinstance(shard, Mapping)
                or not str(shard.get("path", "")).endswith(".parquet")
                or int(shard.get("size", 0)) <= 0
                or not _is_sha256(shard.get("linked_sha256"))
                or not isinstance(shard.get("row_groups"), list)
                or not shard.get("row_groups")
                or any(int(group) < 0 for group in shard.get("row_groups", []))
            ):
                raise ValueError(f"COCO source-shard provenance is invalid for {path}")


def _validate_real_cross_workload_identities(
    manifests: Mapping[pathlib.Path, Mapping[str, Any]],
) -> None:
    """Enforce the shared-object promises spanning real workload files."""

    def entries(dataset_predicate: Any) -> list[tuple[pathlib.Path, Mapping[str, Any]]]:
        return [
            (path, manifest)
            for path, manifest in manifests.items()
            if dataset_predicate(manifest.get("metadata", {}).get("dataset_id"))
        ]

    def require_same(
        records: Sequence[tuple[pathlib.Path, Mapping[str, Any]]],
        label: str,
        getter: Any,
    ) -> None:
        values = {canonical_json_bytes(getter(manifest)) for _, manifest in records}
        if len(values) > 1:
            paths = ", ".join(str(path) for path, _ in records)
            raise ValueError(f"cross-workload {label} mismatch across: {paths}")

    for family, records in (
        ("CMAB", entries(lambda value: value == "CMAB-1M")),
        (
            "GeoLife",
            entries(
                lambda value: isinstance(value, str) and value.startswith("GeoLife-")
            ),
        ),
    ):
        if len(records) < 2:
            continue
        require_same(
            records,
            f"{family} R object ids",
            lambda item: item["workload"]["R_ids_sha256"],
        )
        require_same(
            records,
            f"{family} S object ids",
            lambda item: item["workload"]["S_ids_sha256"],
        )
        require_same(
            records,
            f"{family} source identifier",
            lambda item: item["metadata"]["source_identifier"],
        )
        require_same(
            records,
            f"{family} importer id",
            lambda item: item["metadata"]["importer_id"],
        )
        require_same(
            records,
            f"{family} importer SHA",
            lambda item: item["metadata"]["importer_sha256"],
        )
        require_same(
            records,
            f"{family} preprocessing config",
            lambda item: item["metadata"]["preprocessing_config_sha256"],
        )
        require_same(
            records,
            f"{family} Hub source",
            lambda item: _real_hf_source(
                item["metadata"], pathlib.Path(item["workload"]["file_name"])
            ),
        )
        if family == "GeoLife":
            verified_sidecars: dict[pathlib.Path, str] = {}
            for side in ("R", "S"):
                require_same(
                    records,
                    f"GeoLife selected point keys {side}",
                    lambda item, side=side: item["metadata"][
                        f"selected_point_manifest_{side}"
                    ]["sha256"],
                )
                for workload_path, manifest in records:
                    point_manifest = manifest["metadata"][
                        f"selected_point_manifest_{side}"
                    ]
                    sidecar = workload_path.parent / str(point_manifest["file_name"])
                    expected = str(point_manifest["sha256"]).lower()
                    actual = verified_sidecars.get(sidecar)
                    if actual is None:
                        if not sidecar.is_file():
                            raise ValueError(
                                f"GeoLife selected-point sidecar is missing: {sidecar}"
                            )
                        actual = hash_file(sidecar)
                        verified_sidecars[sidecar] = actual
                    if actual != expected:
                        raise ValueError(
                            f"GeoLife selected-point sidecar SHA-256 mismatch: {sidecar}"
                        )


def _validate_synthetic_manifest(
    *,
    experiment_id: str,
    workload_id: str,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
    path: pathlib.Path,
) -> None:
    workload = manifest.get("workload")
    metadata = manifest.get("metadata")
    if not isinstance(workload, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"incomplete Alacarte manifest for {path}")
    configuration = metadata.get("configuration")
    coverage = metadata.get("coverage")
    if not isinstance(configuration, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError(
            f"Alacarte configuration/coverage metadata is missing for {path}"
        )
    if metadata.get("model_version") != "alacarte-binary64-scale-v2":
        raise ValueError(f"Alacarte model version mismatch for {path}")
    numeric_protocol = metadata.get("numeric_certificate_protocol")
    if not isinstance(numeric_protocol, Mapping) or not numeric_protocol.get(
        "conditional_probability"
    ):
        raise ValueError(f"Alacarte numeric certificate protocol is missing for {path}")
    if metadata.get("configuration_sha256") != stable_hash_hex(
        "alacarte-config", configuration
    ):
        raise ValueError(f"Alacarte configuration SHA-256 mismatch for {path}")
    solver = configuration.get("solver")
    if not isinstance(solver, Mapping) or metadata.get(
        "coverage_solver_config_sha256"
    ) != stable_hash_hex("alacarte-solver-config", solver):
        raise ValueError(f"Alacarte solver configuration SHA-256 mismatch for {path}")
    expected = {
        "dataset_id": "Alacarte",
        "experiment_id": experiment_id,
        "workload_id": workload_id,
        "data_seed": int(parameters["data_seed"]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Alacarte {key} mismatch for {path}")
    if workload.get("endpoint_type") != "float64":
        raise ValueError(f"Alacarte endpoint type must be float64 for {path}")
    if int(configuration.get("dimension", -1)) != int(parameters["d"]):
        raise ValueError(f"Alacarte dimension mismatch for {path}")
    if int(configuration.get("n_R", -1)) != int(workload.get("n_R", -2)) or int(
        configuration.get("n_S", -1)
    ) != int(workload.get("n_S", -2)):
        raise ValueError(f"Alacarte side sizes mismatch for {path}")
    if int(configuration.get("n_R", -1)) + int(configuration.get("n_S", -1)) != int(
        parameters["N"]
    ):
        raise ValueError(f"Alacarte total size mismatch for {path}")
    if float(configuration.get("alpha_target", float("nan"))) != float(
        parameters["alpha_target"]
    ):
        raise ValueError(f"Alacarte alpha target mismatch for {path}")
    if float(configuration.get("shape_sigma", float("nan"))) != float(
        parameters["shape_sigma"]
    ):
        raise ValueError(f"Alacarte shape sigma mismatch for {path}")
    if (
        metadata.get("coverage_status") != "CERTIFIED"
        or coverage.get("status") != "CERTIFIED"
        or coverage.get("certified") is not True
    ):
        raise ValueError(f"Alacarte workload is not coverage-certified: {path}")


@dataclasses.dataclass(frozen=True)
class ExperimentCase:
    experiment_id: str
    dataset_type: str
    dataset_id: str
    workload_id: str
    workload_path: pathlib.Path
    sweep: str
    x_name: str
    x_value: int | float | str
    t: int
    dimension: int
    data_seed: int | None
    level: int | None
    parameters: Mapping[str, Any]
    experiment_config_sha256: str


@dataclasses.dataclass(frozen=True)
class RunSpec:
    case: ExperimentCase
    algorithm: str
    sample_seed_id: int
    process_repeat_id: int
    measurement_mode: str
    task: str = "oneshot"

    @property
    def sample_seed_hex(self) -> str:
        return stable_hash(
            "sample-master-seed",
            self.case.experiment_id,
            self.case.dataset_id,
            self.case.workload_id,
            self.algorithm,
            self.sample_seed_id,
        ).hex()

    @property
    def run_group_id(self) -> str:
        return stable_hash(
            "run-group-id",
            self.case.experiment_id,
            self.case.dataset_id,
            self.case.workload_id,
            self.case.experiment_config_sha256,
            ALGORITHM_IMPLEMENTATION_ID,
            self.algorithm,
            self.case.t,
            self.sample_seed_id,
            self.process_repeat_id,
            self.task,
        )[:16].hex()


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != "anchor-experiments-v1":
        raise ValueError("unsupported experiment configuration schema")
    return value


def validate_machine(machine: Mapping[str, Any]) -> None:
    if machine.get("schema_version") != "anchor-machine-v1":
        raise ValueError("unsupported machine manifest schema")
    if int(machine.get("memory_cap_bytes", 0)) != OFFICIAL_MEMORY_CAP_BYTES:
        raise ValueError(
            "publication machine manifest requires "
            f"the {OFFICIAL_MEMORY_CAP_GIB}-GiB RSS cap"
        )
    if (
        int(machine.get("timeout_seconds", 0)) != OFFICIAL_TIMEOUT_SECONDS
        or int(machine.get("setup_timeout_seconds", 0)) != OFFICIAL_TIMEOUT_SECONDS
    ):
        raise ValueError("publication machine manifest requires 1800-second timeouts")
    if machine.get("memory_measurement_backend") != "procfs_vmrss_polling":
        raise ValueError("machine manifest requires procfs_vmrss_polling")
    if int(machine.get("memory_poll_interval_ms", 0)) != 5:
        raise ValueError(
            "publication machine manifest requires a 5 ms memory poll interval"
        )
    monitor_core = machine.get("monitor_cpu_core")
    if monitor_core is None or int(monitor_core) < 0:
        raise ValueError("machine manifest needs a non-negative monitor_cpu_core")
    monitor_runner = pathlib.Path(
        str(machine.get("memory_monitor_runner", "scripts/run_procfs.py"))
    ).resolve()
    if not monitor_runner.is_file():
        raise FileNotFoundError(monitor_runner)
    required_provenance = (
        "machine_id",
        "code_commit",
        "build_sha256",
        "compiler_id",
        "linker_id",
        "build_flags",
        "target_isa",
        "allocator_id",
        "prng_id",
        "kernel_id",
        "memory_configuration",
        "third_party_commits",
    )
    missing_provenance = [
        key
        for key in required_provenance
        if machine.get(key) is None or str(machine.get(key)).startswith("REPLACE")
    ]
    if missing_provenance:
        raise ValueError(
            f"machine manifest lacks frozen provenance: {missing_provenance}"
        )
    build_flags = machine.get("build_flags")
    if (
        not isinstance(build_flags, list)
        or not build_flags
        or any(not isinstance(flag, str) or not flag.strip() for flag in build_flags)
    ):
        raise ValueError(
            "publication machine manifest needs nonempty exact build_flags"
        )
    if not isinstance(machine.get("third_party_commits"), Mapping):
        raise ValueError("third_party_commits must be a component-to-commit mapping")
    host = machine.get("host")
    if not isinstance(host, Mapping):
        raise ValueError(
            "publication machine manifest must contain captured host facts"
        )
    identity = {
        key: value
        for key, value in machine.items()
        if key not in {"machine_id", "protocol_violations"}
    }
    expected_machine_id = stable_hash("machine-manifest-v1", identity)[:16].hex()
    if machine.get("machine_id") != expected_machine_id:
        raise ValueError("machine_id does not match the captured manifest contents")
    benchmark = pathlib.Path(str(machine["benchmark_path"])).resolve()
    if not benchmark.is_file():
        raise FileNotFoundError(benchmark)
    benchmark_sha256 = hash_file(benchmark)
    if machine.get("build_sha256") != benchmark_sha256:
        raise ValueError("machine build_sha256 does not match benchmark executable")
    validation_path_value = machine.get("validation_report_path")
    if not validation_path_value:
        raise ValueError("machine manifest needs validation_report_path")
    validation_path = pathlib.Path(str(validation_path_value)).resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    with validation_path.open("r", encoding="utf-8") as stream:
        validation = json.load(stream)
    if validation.get("schema_version") != "anchor-validation-report-v1":
        raise ValueError("unsupported validation report schema")
    if validation.get("status") != "OK":
        raise ValueError("publication runs require a validation report with status OK")
    if validation.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("validation report was produced for a different benchmark")
    validation_sha256 = hash_file(validation_path)
    if machine.get("validation_report_sha256") != validation_sha256:
        raise ValueError("machine validation_report_sha256 does not match its report")
    if not machine.get("benchmark_handles_affinity", False):
        numactl = machine.get("numactl_path")
        if not numactl or not pathlib.Path(str(numactl)).is_file():
            raise ValueError(
                "publication runs require numactl_path, unless the benchmark "
                "explicitly records benchmark_handles_affinity=true"
            )
    from anchor_exp.environment import environment_violations

    violations = environment_violations(dict(machine))
    if violations:
        raise ValueError("machine violates frozen protocol: " + "; ".join(violations))


def _workload_id(
    experiment_id: str, dataset_id: str, parameters: Mapping[str, Any]
) -> str:
    suffix = stable_hash("workload-id", experiment_id, dataset_id, parameters)[:8].hex()
    return f"{experiment_id}-{suffix}"


def _format_path(
    template: str, values: Mapping[str, Any], data_root: pathlib.Path
) -> pathlib.Path:
    fields = dict(values)
    fields["data_root"] = str(data_root)
    return pathlib.Path(template.format(**fields)).resolve()


def expand_experiments(
    config: Mapping[str, Any],
    *,
    data_root: str | pathlib.Path | None = None,
    require_workloads: bool = False,
) -> list[ExperimentCase]:
    """Expand the five synthetic and seven real sweep definitions."""

    root = pathlib.Path(data_root or config.get("data_root", ".")).resolve()
    config_sha256 = stable_hash("experiment-config-v1", config).hex()
    cases: list[ExperimentCase] = []
    verified: dict[pathlib.Path, Mapping[str, Any]] = {}
    for sweep in config["sweeps"]:
        experiment_id = str(sweep["experiment_id"])
        varying = str(sweep["varying"])
        values = list(sweep["values"])
        seeds: Sequence[int | None]
        seeds = (
            config["data_seeds"] if sweep["dataset_type"] == "synthetic" else (None,)
        )
        for seed in seeds:
            for value in values:
                parameters = dict(sweep.get("fixed", {}))
                parameters[varying] = value
                if seed is not None:
                    parameters["data_seed"] = int(seed)
                dimension = int(parameters.get("d", sweep.get("dimension", 0)))
                t = int(parameters.get("t", sweep.get("t", 0)))
                level_value = parameters.get("level")
                path = _format_path(sweep["workload_template"], parameters, root)
                workload_parameters = dict(parameters)
                # t changes a query, not a frozen workload identity.
                workload_parameters.pop("t", None)
                workload_id = _workload_id(
                    experiment_id, str(sweep["dataset_id"]), workload_parameters
                )
                if require_workloads:
                    manifest = verified.get(path)
                    if manifest is None:
                        manifest = read_manifest(path)
                        if hash_file(path) != manifest["workload"]["sha256"]:
                            raise ValueError(f"workload SHA-256 mismatch for {path}")
                        verified[path] = manifest
                    workload = manifest["workload"]
                    if int(workload["dimension"]) != dimension:
                        raise ValueError(f"dimension mismatch for {path}")
                    if "N" in parameters and int(workload["N_total"]) != int(
                        parameters["N"]
                    ):
                        raise ValueError(f"input-size mismatch for {path}")
                    if sweep["dataset_type"] == "synthetic":
                        _validate_synthetic_manifest(
                            experiment_id=experiment_id,
                            workload_id=workload_id,
                            parameters=parameters,
                            manifest=manifest,
                            path=path,
                        )
                    else:
                        _validate_real_manifest(
                            dataset_id=str(sweep["dataset_id"]),
                            dimension=dimension,
                            level=int(level_value) if level_value is not None else None,
                            manifest=manifest,
                            path=path,
                        )
                cases.append(
                    ExperimentCase(
                        experiment_id=experiment_id,
                        dataset_type=str(sweep["dataset_type"]),
                        dataset_id=str(sweep["dataset_id"]),
                        workload_id=workload_id,
                        workload_path=path,
                        sweep=str(sweep["sweep"]),
                        x_name=varying,
                        x_value=value,
                        t=t,
                        dimension=dimension,
                        data_seed=int(seed) if seed is not None else None,
                        level=int(level_value) if level_value is not None else None,
                        parameters=parameters,
                        experiment_config_sha256=config_sha256,
                    )
                )
    if require_workloads:
        _validate_real_cross_workload_identities(verified)
    expected = config.get("expected_sweep_count", 12)
    actual = len({case.experiment_id for case in cases})
    if actual != expected:
        raise ValueError(f"configuration expands {actual} sweeps, expected {expected}")
    return cases


def expand_runs(
    cases: Sequence[ExperimentCase],
    config: Mapping[str, Any],
    *,
    tasks: Sequence[str] = ("oneshot", "count-only", "prepared-query"),
) -> list[RunSpec]:
    algorithms = list(config["algorithms"])
    repeats = [int(item) for item in config["process_repeat_ids"]]
    sample_seed_id = int(config.get("sample_seed_id", 0))
    runs: list[RunSpec] = []
    for case in cases:
        for repeat in repeats:
            ordered = algorithms
            for task in tasks:
                modes = MEASUREMENT_MODES if task == "oneshot" else ("time",)
                for mode in modes:
                    for algorithm in ordered:
                        if task == "prepared-query" and algorithm == "AS":
                            continue
                        runs.append(
                            RunSpec(
                                case,
                                algorithm,
                                sample_seed_id,
                                repeat,
                                mode,
                                task,
                            )
                        )
    return runs


def _benchmark_command(run: RunSpec, machine: Mapping[str, Any]) -> list[str]:
    executable = pathlib.Path(str(machine["benchmark_path"])).resolve()
    command = [
        str(executable),
        "run",
        "--workload",
        str(run.case.workload_path),
        "--algorithm",
        ALGORITHM_CLI[run.algorithm],
        "--samples",
        str(run.case.t),
        "--seed-hex",
        run.sample_seed_hex,
        "--measurement-mode",
        run.measurement_mode,
        "--task",
        run.task,
        "--timeout-seconds",
        str(int(machine.get("timeout_seconds", OFFICIAL_TIMEOUT_SECONDS))),
        "--setup-timeout-seconds",
        str(int(machine.get("setup_timeout_seconds", OFFICIAL_TIMEOUT_SECONDS))),
    ]
    numactl = machine.get("numactl_path")
    if numactl:
        command = [
            str(pathlib.Path(str(numactl)).resolve()),
            f"--physcpubind={int(machine['cpu_core'])}",
            f"--membind={int(machine['numa_node'])}",
            *command,
        ]
    return command


def _parse_one_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("benchmark produced no JSON record")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark stdout is not exactly one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark JSON result must be an object")
    return value


def _algorithm_config_sha256(run: RunSpec) -> str:
    """Hash the frozen implementation and task-specific sampler options."""

    return stable_hash(
        "algorithm-config-v1",
        {
            "algorithm": run.algorithm,
            "implementation": ALGORITHM_IMPLEMENTATION_ID,
            "orientation": "R-to-S",
            "interval_semantics": "strict-half-open",
            "task": run.task,
            "count_only_builds_sampling_index": not (
                run.task == "count-only" and run.algorithm in {"LiftedRT", "SweepRT"}
            ),
        },
    ).hex()


def _identity(
    run: RunSpec, machine: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    workload = manifest["workload"]
    metadata = manifest.get("metadata", {})
    record = {
        "schema_version": "anchor-run-record-v1",
        "experiment_id": run.case.experiment_id,
        "dataset_type": run.case.dataset_type,
        "dataset_id": run.case.dataset_id,
        "workload_id": run.case.workload_id,
        "subset_id": metadata.get("subset_id") or metadata.get("image_subset_id"),
        "run_group_id": run.run_group_id,
        "measurement_mode": run.measurement_mode,
        "task": run.task,
        "data_seed": run.case.data_seed,
        "sample_seed_id": run.sample_seed_id,
        "process_repeat_id": run.process_repeat_id,
        "sample_seed_hex": run.sample_seed_hex,
        "machine_id": machine["machine_id"],
        "code_commit": machine.get("code_commit"),
        "build_sha256": machine.get("build_sha256"),
        "compiler_id": machine.get("compiler_id"),
        "allocator_id": machine.get("allocator_id"),
        "prng_id": machine.get("prng_id"),
        "kernel_id": machine.get("kernel_id"),
        "cpu_core": machine.get("cpu_core"),
        "numa_node": machine.get("numa_node"),
        "monitor_cpu_core": machine.get("monitor_cpu_core"),
        "memory_cap_bytes": int(machine["memory_cap_bytes"]),
        "memory_measurement_backend": machine.get("memory_measurement_backend"),
        "memory_poll_interval_ms": int(machine["memory_poll_interval_ms"]),
        "experiment_config_sha256": run.case.experiment_config_sha256,
        "algorithm_config_sha256": _algorithm_config_sha256(run),
        "N_total": int(workload["N_total"]),
        "n_R": int(workload["n_R"]),
        "n_S": int(workload["n_S"]),
        "d": int(workload["dimension"]),
        "endpoint_type": workload["endpoint_type"],
        "t": run.case.t,
        "workload_sha256": workload["sha256"],
        "sweep": run.case.sweep,
        "x_name": run.case.x_name,
        "x_value": run.case.x_value,
        "level": run.case.level,
        "algorithm": run.algorithm,
        "orientation": "R-to-S",
        "timeout_seconds": int(
            machine.get("timeout_seconds", OFFICIAL_TIMEOUT_SECONDS)
        ),
        "setup_timeout_seconds": int(
            machine.get("setup_timeout_seconds", OFFICIAL_TIMEOUT_SECONDS)
        ),
        **{
            key: value for key, value in run.case.parameters.items() if key not in {"t"}
        },
    }
    metadata_fields = (
        "alpha_target",
        "alpha_expected",
        "coverage_status",
        "coverage_interval",
        "coverage_solver_config_sha256",
        "aspect_ratio_quantiles",
        "saturation_fraction",
        "split_method",
        "cmab_crs",
        "horizontal_crs",
        "projection_unit",
        "stratification_id",
        "image_subset_id",
        "proposal_pipeline_id",
        "proposal_stage",
        "real_dataset",
        "crs_id",
        "source_repository",
        "source_revision",
        "importer_id",
        "importer_sha256",
        "source_identifier",
        "preprocessing_config_sha256",
        "boundary_touching_diagnostic",
        "spatial_temporal_coverage_summary",
        "rank_or_score_summary",
    )
    for key in metadata_fields:
        if key in metadata:
            record[key] = metadata[key]
    configuration = metadata.get("configuration")
    coverage = metadata.get("coverage")
    geometry = metadata.get("geometry_diagnostics")
    if isinstance(configuration, Mapping):
        record.setdefault("alpha_target", configuration.get("alpha_target"))
        record.setdefault("shape_sigma", configuration.get("shape_sigma"))
    if isinstance(coverage, Mapping):
        record.setdefault("alpha_expected", coverage.get("output_density_estimate"))
        record.setdefault("coverage_status", coverage.get("status"))
        record.setdefault("coverage_interval", coverage.get("output_density_interval"))
    record.setdefault(
        "coverage_solver_config_sha256",
        metadata.get("coverage_solver_config_sha256")
        or metadata.get("configuration_sha256"),
    )
    if isinstance(geometry, Mapping):
        record["geometry_diagnostics"] = dict(geometry)
        record.setdefault(
            "aspect_ratio_quantiles",
            geometry.get("normalized_aspect_ratio_quantiles"),
        )
        record.setdefault("saturation_fraction", geometry.get("saturation_fraction"))
    record["workload_config_sha256"] = metadata.get(
        "configuration_sha256"
    ) or metadata.get("preprocessing_config_sha256")
    source = metadata.get("source")
    if isinstance(source, Mapping):
        record["source_lock_sha256"] = source.get("source_lock_sha256")
        assets = source.get("assets")
        if isinstance(assets, Sequence) and not isinstance(assets, (str, bytes)):
            record["source_asset_count"] = len(assets)
    diagnostic_fields = (
        "function_class_counts_R",
        "function_class_counts_S",
        "tile_count_balance",
        "aabb_area_summary_R",
        "aabb_area_summary_S",
        "selected_summary_R",
        "selected_summary_S",
        "image_size_summary",
        "proposal_summary_R",
        "proposal_summary_S",
    )
    diagnostics = {key: metadata[key] for key in diagnostic_fields if key in metadata}
    if diagnostics:
        record["dataset_diagnostics"] = diagnostics
    return record


def _terminate_process_group(
    process: subprocess.Popen[str], *, grace_seconds: float = 0.25
) -> str | None:
    if process.poll() is not None:
        return None
    sent = "SIGTERM"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if process.poll() is None:
        sent = "SIGKILL"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sent


def _run_time_process(
    command: Sequence[str], *, timeout_seconds: float
) -> tuple[subprocess.CompletedProcess[str], bool, str | None]:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    termination_signal: str | None = None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination_signal = _terminate_process_group(process)
        stdout, stderr = process.communicate()
    return (
        subprocess.CompletedProcess(
            list(command),
            124 if timed_out else int(process.returncode or 0),
            stdout,
            stderr,
        ),
        timed_out,
        termination_signal,
    )


def run_one(run: RunSpec, machine: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one fresh process and merge its result with immutable identity."""

    manifest = read_manifest(run.case.workload_path)
    record = _identity(run, machine, manifest)
    command = _benchmark_command(run, machine)
    setup_timeout = int(machine.get("setup_timeout_seconds", OFFICIAL_TIMEOUT_SECONDS))
    timeout = (
        int(machine.get("timeout_seconds", OFFICIAL_TIMEOUT_SECONDS))
        + 2 * setup_timeout
    )
    wrapper_timeout = timeout + 30
    wrapper_report: dict[str, Any] = {}
    try:
        if run.measurement_mode == "memory":
            runner = pathlib.Path(
                str(machine.get("memory_monitor_runner", "scripts/run_procfs.py"))
            ).resolve()
            report_directory = (
                pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
                / f"anchor-procfs-report-{os.getpid()}-{time.time_ns()}"
            )
            report_directory.mkdir(parents=True, exist_ok=False)
            report_path = report_directory / "report.json"
            wrapped = [
                sys.executable,
                str(runner),
                "--memory-cap-bytes",
                str(int(machine["memory_cap_bytes"])),
                "--timeout-seconds",
                str(wrapper_timeout),
                "--poll-interval-ms",
                str(int(machine["memory_poll_interval_ms"])),
                "--monitor-cpu-core",
                str(int(machine["monitor_cpu_core"])),
                "--report",
                str(report_path),
                "--",
                *command,
            ]
            try:
                completed = subprocess.run(
                    wrapped,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if report_path.exists():
                    with report_path.open("r", encoding="utf-8") as stream:
                        wrapper_report = json.load(stream)
            finally:
                report_path.unlink(missing_ok=True)
                report_directory.rmdir()
        else:
            completed, timed_out, termination_signal = _run_time_process(
                command, timeout_seconds=wrapper_timeout
            )
            if timed_out:
                wrapper_report = {
                    "status": "TO",
                    "timeout_source": "time-process-safety-timeout",
                    "termination_signal": termination_signal,
                }

        try:
            benchmark = _parse_one_json(completed.stdout)
        except ValueError:
            if wrapper_report.get("status") in {
                "OOM",
                "TO",
                "MEMORY-CAP-EXCEEDED",
                "MEMORY-MEASUREMENT-UNAVAILABLE",
            }:
                benchmark = {
                    "status": wrapper_report["status"],
                    "error_message": wrapper_report.get("error_message"),
                }
            elif completed.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
                benchmark = {
                    "status": "OOM",
                    "failure_stage": "benchmark-process",
                    "error_message": "benchmark was killed without a JSON record",
                }
            else:
                benchmark = {
                    "status": "BENCHMARK-ERROR",
                    "error_message": completed.stderr.strip()[-4000:],
                    "exit_code": completed.returncode,
                }

        if "schema_version" in benchmark:
            record["benchmark_schema_version"] = benchmark.pop("schema_version")
        forced = wrapper_report.get("status")
        if forced in {
            "OOM",
            "TO",
            "MEMORY-CAP-EXCEEDED",
            "MEMORY-MEASUREMENT-UNAVAILABLE",
        }:
            benchmark["status"] = forced
            if forced == "TO":
                benchmark.setdefault("failure_stage", "process-wrapper")
            elif forced == "MEMORY-CAP-EXCEEDED":
                benchmark.setdefault("failure_stage", "memory-polling")
            elif forced == "MEMORY-MEASUREMENT-UNAVAILABLE":
                benchmark.setdefault("failure_stage", "memory-measurement")
        if completed.returncode in (-signal.SIGALRM, 128 + signal.SIGALRM):
            benchmark["status"] = "TO"
            benchmark.setdefault("failure_stage", "main-algorithm-interval")
        if completed.returncode != 0 and benchmark.get("status") == "OK":
            benchmark["status"] = "BENCHMARK-ERROR"
            benchmark[
                "error_message"
            ] = f"benchmark reported OK but process exited {completed.returncode}"

        if run.measurement_mode == "time":
            forbidden_memory_fields = (
                "InputMemory",
                "BaselineMemory",
                "PeakRSSPollBytes",
                "PeakMemoryTotal",
                "PeakMemoryIncremental",
                "PeakMemoryAux",
                "MemoryAfterPrepare",
                "rss_sample_count",
            )
            leaked = [key for key in forbidden_memory_fields if key in benchmark]
            if leaked:
                benchmark["status"] = "BENCHMARK-ERROR"
                benchmark[
                    "error_message"
                ] = f"time benchmark emitted memory measurements: {leaked}"

        immutable = dict(record)
        record.update(benchmark)
        for key, value in immutable.items():
            if key in benchmark and benchmark[key] != value:
                record["status"] = "BENCHMARK-ERROR"
                record[
                    "error_message"
                ] = f"benchmark attempted to overwrite immutable field {key!r}"
            record[key] = value

        if run.measurement_mode == "memory":
            for key in (
                "InputMemory",
                "BaselineMemory",
                "PeakRSSPollBytes",
                "PeakMemoryTotal",
                "PeakMemoryIncremental",
                "PeakMemoryAux",
                "MemoryAfterPrepare",
                "memory_measurement_backend",
                "memory_poll_interval_ms",
                "peak_is_sampled",
                "rss_sample_count",
                "rss_first_sample_monotonic_ns",
                "rss_last_sample_monotonic_ns",
                "last_observed_rss_bytes",
                "memory_cap_exceeded",
                "termination_signal",
                "timeout_source",
                "monitor_cpu_core",
                "monitor_cpu_affinity_applied",
                "monitor_cpu_affinity_error",
            ):
                if key in wrapper_report:
                    record[key] = wrapper_report[key]

        if record.get("W") is not None and record.get("alpha_realized") is None:
            record["alpha_realized"] = int(record["W"]) / int(record["N_total"])

        if (
            run.measurement_mode == "memory"
            and run.task == "oneshot"
            and record.get("status") == "OK"
        ):
            required_memory = (
                "InputMemory",
                "BaselineMemory",
                "PeakRSSPollBytes",
                "PeakMemoryTotal",
                "PeakMemoryIncremental",
                "PeakMemoryAux",
                "ProcessMaxRSS",
                "rss_sample_count",
            )
            missing = [key for key in required_memory if record.get(key) is None]
            if run.algorithm != "AS" and record.get("MemoryAfterPrepare") is None:
                missing.append("MemoryAfterPrepare")
            protocol_ok = record.get("memory_protocol_handshake") is True
            metadata_ok = (
                record.get("memory_measurement_backend") == "procfs_vmrss_polling"
                and record.get("peak_is_sampled") is True
                and int(record.get("memory_poll_interval_ms", 0)) == 5
                and int(record.get("rss_sample_count", 0)) > 0
                and record.get("monitor_cpu_affinity_applied") is True
            )
            if missing or not protocol_ok or not metadata_ok:
                record["status"] = "MEMORY-MEASUREMENT-UNAVAILABLE"
                record["error_message"] = "procfs memory protocol is incomplete" + (
                    f"; missing fields: {missing}" if missing else ""
                )
            else:
                input_memory = int(record["InputMemory"])
                baseline_memory = int(record["BaselineMemory"])
                total = int(record["PeakMemoryTotal"])
                polled = int(record["PeakRSSPollBytes"])
                if (
                    total != polled
                    or int(record["PeakMemoryIncremental"])
                    != max(0, total - input_memory)
                    or int(record["PeakMemoryAux"]) != max(0, total - baseline_memory)
                ):
                    record["status"] = "MEMORY-MEASUREMENT-UNAVAILABLE"
                    record["error_message"] = "inconsistent procfs peak-memory formulas"
    except subprocess.TimeoutExpired:
        record.update(
            {
                "status": "TO",
                "failure_stage": "process-wrapper",
                "error_message": (
                    f"external safety timeout after {wrapper_timeout} seconds"
                ),
            }
        )
    record.setdefault("status", "BENCHMARK-ERROR")
    record["harness_timestamp_ns"] = time.time_ns()
    return record


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while persisting JSONL")
        view = view[written:]


def append_jsonl(path: str | pathlib.Path, record: Mapping[str, Any]) -> None:
    destination = pathlib.Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(record) + b"\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def completed_run_keys(path: str | pathlib.Path) -> set[tuple[str, str]]:
    source = pathlib.Path(path)
    if not source.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                keys.add((str(record["run_group_id"]), str(record["measurement_mode"])))
    return keys


def _finalize_reconciled_jsonl(
    staging_path: pathlib.Path, destination: pathlib.Path
) -> None:
    from anchor_exp.aggregate import derive_count_consistency, publication_preflight

    records: list[dict[str, Any]] = []
    with staging_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{staging_path}:{line_number}: record is not an object"
                )
            records.append(value)
    publication_preflight(records)
    reconciled, mismatches = derive_count_consistency(records)
    if mismatches:
        raise ValueError(
            "suite output consistency gate failed; raw staging records retained at "
            f"{staging_path}; first mismatch: {mismatches[0]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for record in reconciled:
            _write_all(descriptor, canonical_json_bytes(record) + b"\n")
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, destination)
    staging_path.unlink()


def run_suite(
    runs: Iterable[RunSpec],
    machine: Mapping[str, Any],
    output_jsonl: str | pathlib.Path,
    *,
    resume: bool = True,
) -> int:
    validate_machine(machine)
    run_list = list(runs)
    destination = pathlib.Path(output_jsonl).resolve()
    staging = destination.with_name(destination.name + ".partial")
    if not resume:
        if staging.exists():
            staging.unlink()
    elif not staging.exists() and destination.exists():
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_bytes(destination.read_bytes())
    if resume and staging.exists():
        expected_configs = {run.case.experiment_config_sha256 for run in run_list}
        with staging.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                expected_identity = {
                    "machine_id": machine.get("machine_id"),
                    "code_commit": machine.get("code_commit"),
                    "build_sha256": machine.get("build_sha256"),
                }
                for field, expected in expected_identity.items():
                    if record.get(field) != expected:
                        raise ValueError(
                            f"cannot resume {staging}:{line_number}: {field} changed; "
                            "use a new output path or --no-resume"
                        )
                if record.get("experiment_config_sha256") not in expected_configs:
                    raise ValueError(
                        f"cannot resume {staging}:{line_number}: experiment config changed; "
                        "use a new output path or --no-resume"
                    )
    completed = completed_run_keys(staging) if resume else set()
    count = 0
    for run in run_list:
        key = (run.run_group_id, run.measurement_mode)
        if key in completed:
            continue
        append_jsonl(staging, run_one(run, machine))
        completed.add(key)
        count += 1
    _finalize_reconciled_jsonl(staging, destination)
    return count
