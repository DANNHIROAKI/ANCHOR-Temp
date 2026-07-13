"""Specification-aligned aggregation of immutable JSONL run records."""

from __future__ import annotations

import csv
import json
import math
import pathlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from anchor_exp.stable_hash import canonical_json_bytes


METRICS = (
    "OneShotTime",
    "PrepareTime",
    "QueryTime",
    "CountOnlyTime",
    "PreparedQueryTime",
    "InputMemory",
    "BaselineMemory",
    "PeakMemoryTotal",
    "PeakMemoryIncremental",
    "PeakMemoryAux",
    "ProcessMaxRSS",
    "MemoryAfterPrepare",
)

EXPECTED_REPEAT_IDS = {0}
EXPECTED_SYNTHETIC_SEEDS = {0}
GLOBAL_IDENTITY_FIELDS = (
    "machine_id",
    "code_commit",
    "build_sha256",
    "experiment_config_sha256",
)
EXPECTED_ALGORITHM_MODES = {
    "oneshot": {
        (algorithm, mode)
        for algorithm in ("AC", "AS", "SweepRT", "LiftedRT")
        for mode in ("time", "memory")
    },
    "count-only": {
        (algorithm, "time")
        for algorithm in ("AC", "AS", "SweepRT", "LiftedRT")
    },
    "prepared-query": {
        (algorithm, "time")
        for algorithm in ("AC", "SweepRT", "LiftedRT")
    },
}


def _preflight_error(errors: Sequence[str]) -> ValueError:
    detail = "; ".join(errors[:12])
    if len(errors) > 12:
        detail += f"; ... and {len(errors) - 12} more"
    return ValueError("publication aggregation preflight failed: " + detail)


def publication_preflight(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject mixed, duplicated, or incomplete publication run sets."""

    if not records:
        raise ValueError("publication aggregation requires at least one record")
    errors: list[str] = []
    for field in GLOBAL_IDENTITY_FIELDS:
        values = {record.get(field) for record in records}
        if None in values or "" in values:
            errors.append(f"missing {field}")
        if len(values) != 1:
            errors.append(f"mixed {field}: {sorted(map(str, values))}")

    algorithm_configs: dict[tuple[Any, Any], set[Any]] = defaultdict(set)
    workload_configs: dict[Any, set[Any]] = defaultdict(set)
    for record in records:
        algorithm_configs[
            (record.get("algorithm"), record.get("task", "oneshot"))
        ].add(record.get("algorithm_config_sha256"))
        workload_configs[record.get("workload_id")].add(
            record.get("workload_config_sha256")
        )
    for key, values in algorithm_configs.items():
        if None in values or "" in values or len(values) != 1:
            errors.append(f"mixed/missing algorithm config for {key}: {values}")
    for key, values in workload_configs.items():
        if key is None or None in values or "" in values or len(values) != 1:
            errors.append(f"mixed/missing workload config for {key}: {values}")

    run_keys: Counter[tuple[Any, Any]] = Counter(
        (record.get("run_group_id"), record.get("measurement_mode"))
        for record in records
    )
    duplicates = [key for key, count in run_keys.items() if count != 1]
    if duplicates:
        errors.append(f"duplicate run keys: {duplicates[:5]}")

    repeat_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    seed_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    coverage_groups: dict[tuple[Any, ...], set[tuple[str, str]]] = defaultdict(set)
    real_workloads: dict[tuple[Any, ...], set[tuple[Any, Any, Any]]] = defaultdict(set)
    for record in records:
        if record.get("sample_seed_id") != 0:
            errors.append(
                f"sample_seed_id must be 0, got {record.get('sample_seed_id')!r}"
            )
        repeat_key = (
            record.get("experiment_id"),
            record.get("dataset_id"),
            record.get("workload_id"),
            record.get("t"),
            record.get("data_seed"),
            record.get("algorithm"),
            record.get("measurement_mode"),
            record.get("task", "oneshot"),
            record.get("sample_seed_id"),
        )
        try:
            repeat_groups[repeat_key].add(int(record.get("process_repeat_id")))
        except (TypeError, ValueError):
            errors.append(f"invalid process_repeat_id in {repeat_key}")

        coverage_key = (
            record.get("experiment_id"),
            record.get("dataset_id"),
            record.get("workload_id"),
            record.get("t"),
            record.get("data_seed"),
            record.get("process_repeat_id"),
            record.get("task", "oneshot"),
            record.get("sample_seed_id"),
        )
        coverage_groups[coverage_key].add(
            (str(record.get("algorithm")), str(record.get("measurement_mode")))
        )

        if record.get("dataset_type") == "synthetic":
            seed_key = (
                record.get("experiment_id"),
                record.get("dataset_id"),
                record.get("sweep"),
                record.get("x_name"),
                record.get("x_value"),
                record.get("t"),
                record.get("level"),
                record.get("algorithm"),
                record.get("measurement_mode"),
                record.get("task", "oneshot"),
                record.get("sample_seed_id"),
            )
            try:
                seed_groups[seed_key].add(int(record.get("data_seed")))
            except (TypeError, ValueError):
                errors.append(f"invalid synthetic data_seed in {seed_key}")
        elif record.get("dataset_type") == "real":
            if record.get("data_seed") is not None:
                errors.append("real records must have data_seed=null")
            # A t-sweep is one fixed workload. A level-sweep has one frozen
            # workload per level, shared by every algorithm/mode/repeat/task.
            real_key = (
                record.get("experiment_id"),
                record.get("dataset_id"),
                record.get("x_name"),
                None if record.get("x_name") == "t" else record.get("x_value"),
            )
            real_workloads[real_key].add(
                (
                    record.get("workload_id"),
                    record.get("workload_sha256"),
                    record.get("subset_id"),
                )
            )
        else:
            errors.append(f"invalid dataset_type {record.get('dataset_type')!r}")

    for key, values in repeat_groups.items():
        if values != EXPECTED_REPEAT_IDS:
            errors.append(f"repeat ids for {key} are {sorted(values)}")
    for key, values in seed_groups.items():
        if values != EXPECTED_SYNTHETIC_SEEDS:
            errors.append(f"synthetic seeds for {key} are {sorted(values)}")
    for key, values in coverage_groups.items():
        task = str(key[6])
        expected = EXPECTED_ALGORITHM_MODES.get(task)
        if expected is None:
            errors.append(f"unknown task {task!r}")
        elif values != expected:
            errors.append(
                f"algorithm/mode coverage for {key} differs: "
                f"missing={sorted(expected - values)}, extra={sorted(values - expected)}"
            )
    for key, values in real_workloads.items():
        if len(values) != 1 or any(item[0] is None or item[1] is None for item in values):
            errors.append(f"real workload is not fixed for {key}: {values}")

    if errors:
        raise _preflight_error(errors)
    return {
        "status": "OK",
        "global_identity": {
            field: records[0].get(field) for field in GLOBAL_IDENTITY_FIELDS
        },
        "run_key_count": len(run_keys),
    }


def derive_count_consistency(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return reconciled copies; never rewrite immutable execution records."""

    derived = [dict(record) for record in records]
    mode_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    count_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        mode_groups[
            (
                record.get("experiment_id"),
                record.get("dataset_id"),
                record.get("workload_id"),
                record.get("t"),
                record.get("data_seed"),
                record.get("sample_seed_id"),
                record.get("process_repeat_id"),
                record.get("task", "oneshot"),
                record.get("algorithm"),
            )
        ].append(index)
        count_groups[
            (
                record.get("dataset_id"),
                record.get("workload_id"),
                record.get("workload_sha256"),
                record.get("data_seed"),
                record.get("sample_seed_id"),
            )
        ].append(index)

    count_affected: dict[int, set[str]] = defaultdict(set)
    mode_affected: dict[int, set[str]] = defaultdict(set)
    mismatch_rows: list[dict[str, Any]] = []
    for key, indices in mode_groups.items():
        # Only oneshot has both modes. Different algorithms may legitimately
        # reach different resource boundaries and are deliberately not paired.
        if len(indices) < 2:
            continue
        mismatched_fields: list[str] = []
        statuses = {str(records[index].get("status", "MISSING")) for index in indices}
        if len(statuses) != 1:
            mismatched_fields.append("status")
        lengths = {records[index].get("output_length") for index in indices}
        if len(lengths) != 1:
            mismatched_fields.append("output_length")
        if mismatched_fields:
            for index in indices:
                mode_affected[index].update(mismatched_fields)
            mismatch_rows.append(
                {
                    "scope": "time-memory-pair",
                    "key": list(key),
                    "mismatched_fields": mismatched_fields,
                }
            )

    for key, indices in count_groups.items():
        successful = [
            index
            for index in indices
            if records[index].get("status") in {"OK", "EMPTY-JOIN"}
        ]
        counts = {
            str(records[index].get("W"))
            for index in successful
            if records[index].get("W") is not None
        }
        missing = [index for index in successful if records[index].get("W") is None]
        if len(counts) > 1 or missing:
            for index in successful:
                count_affected[index].add("W")
            mismatch_rows.append(
                {
                    "scope": "fixed-workload-count",
                    "key": list(key),
                    "mismatched_fields": ["W"],
                    "observed_counts": sorted(counts),
                    "successful_records_missing_W": len(missing),
                }
            )

    for index, record in enumerate(derived):
        if index in count_affected:
            record["derived_original_status"] = record.get("status")
            record["status"] = "COUNT-MISMATCH"
            record["failure_stage"] = "suite-count-consistency"
            record["consistency_mismatched_fields"] = sorted(count_affected[index])
            record["consistency_admitted"] = False
            record["count_consistency_checked"] = False
        elif index in mode_affected:
            record["mode_consistency_mismatched_fields"] = sorted(
                mode_affected[index]
            )
            record["consistency_admitted"] = False
            record["count_consistency_checked"] = False
        else:
            record["consistency_admitted"] = True
            record["count_consistency_checked"] = (
                record.get("status") in {"OK", "EMPTY-JOIN"}
                and record.get("W") is not None
            )
    return derived, mismatch_rows


def read_jsonl(paths: Iterable[str | pathlib.Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_name in paths:
        source = pathlib.Path(source_name)
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number}: run record is not an object")
                records.append(value)
    return records


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _base_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("experiment_id"),
        record.get("dataset_type"),
        record.get("dataset_id"),
        record.get("sweep"),
        record.get("x_name"),
        record.get("x_value"),
        record.get("t"),
        record.get("level"),
        record.get("algorithm"),
    )


def _key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    names = (
        "experiment_id",
        "dataset_type",
        "dataset_id",
        "sweep",
        "x_name",
        "x_value",
        "t",
        "level",
        "algorithm",
    )
    return dict(zip(names, key, strict=True))


def _metric_allowed(record: Mapping[str, Any], metric: str) -> bool:
    if record.get("consistency_admitted") is False:
        return False
    mode = record.get("measurement_mode")
    task = record.get("task", "oneshot")
    if metric.startswith("PeakMemory") or metric in {
        "InputMemory",
        "BaselineMemory",
        "ProcessMaxRSS",
        "MemoryAfterPrepare",
    }:
        return mode == "memory" and task == "oneshot"
    if metric == "CountOnlyTime":
        return mode == "time" and task == "count-only"
    if metric == "PreparedQueryTime":
        return mode == "time" and task == "prepared-query"
    return mode == "time" and task == "oneshot"


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose the single frozen-run value for every successful configuration."""

    values: dict[tuple[Any, ...], tuple[Any, float]] = {}
    for record in records:
        if record.get("status") != "OK":
            continue
        for metric in METRICS:
            if not _metric_allowed(record, metric):
                continue
            value = _number(record.get(metric))
            if value is None:
                continue
            key = (*_base_key(record), metric)
            if key in values:
                raise ValueError(f"multiple successful values for frozen run {key}")
            values[key] = (record.get("data_seed"), value)

    return [
        {
            **_key_dict(key[:-1]),
            "metric": key[-1],
            "data_seed": data_seed,
            "value": value,
        }
        for key, (data_seed, value) in sorted(
            values.items(), key=lambda item: repr(item[0])
        )
    ]


def aggregate_speedups(
    records: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str] = ("LiftedRT", "SweepRT"),
) -> list[dict[str, Any]]:
    """Compute direct paired speedups from the frozen single time runs."""

    timings: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    identities: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in records:
        if (
            record.get("status") != "OK"
            or record.get("consistency_admitted") is False
            or record.get("measurement_mode") != "time"
            or record.get("task", "oneshot") != "oneshot"
        ):
            continue
        value = _number(record.get("OneShotTime"))
        if value is None or value <= 0:
            continue
        key = (
            record.get("experiment_id"),
            record.get("dataset_id"),
            record.get("workload_id"),
            record.get("x_name"),
            record.get("x_value"),
            record.get("t"),
            record.get("data_seed"),
            record.get("process_repeat_id"),
        )
        algorithm = str(record["algorithm"])
        if algorithm in timings[key]:
            raise ValueError(f"duplicate frozen timing for {key}, {algorithm}")
        timings[key][algorithm] = value
        identities[key] = record

    output: list[dict[str, Any]] = []
    algorithms = sorted({str(record.get("algorithm")) for record in records})
    for pair_key, values in sorted(timings.items(), key=lambda item: repr(item[0])):
        identity = identities[pair_key]
        for baseline in baselines:
            if baseline not in values:
                continue
            for algorithm in algorithms:
                if algorithm == baseline or algorithm not in values:
                    continue
                output.append(
                    {
                        "experiment_id": identity.get("experiment_id"),
                        "dataset_type": identity.get("dataset_type"),
                        "dataset_id": identity.get("dataset_id"),
                        "sweep": identity.get("sweep"),
                        "x_name": identity.get("x_name"),
                        "x_value": identity.get("x_value"),
                        "t": identity.get("t"),
                        "level": identity.get("level"),
                        "data_seed": identity.get("data_seed"),
                        "algorithm": algorithm,
                        "baseline": baseline,
                        "speedup": values[baseline] / values[algorithm],
                    }
                )
    return output


def status_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for record in records:
        key = (
            *_base_key(record),
            record.get("measurement_mode"),
            record.get("task", "oneshot"),
        )
        groups[key][str(record.get("status", "MISSING"))] += 1
    output: list[dict[str, Any]] = []
    for key, statuses in sorted(groups.items(), key=lambda item: repr(item[0])):
        output.append(
            {
                **_key_dict(key[:9]),
                "measurement_mode": key[9],
                "task": key[10],
                "statuses": dict(sorted(statuses.items())),
            }
        )
    return output


def count_consistency(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        if record.get("status") in {"OK", "EMPTY-JOIN"} and record.get("W") is not None:
            key = (record.get("dataset_id"), record.get("workload_id"))
            groups[key].add(str(record["W"]))
    return [
        {
            "dataset_id": key[0],
            "workload_id": key[1],
            "counts": sorted(values),
            "consistent": len(values) == 1,
        }
        for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))
    ]


def run_group_consistency(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("task", "oneshot") == "oneshot":
            groups[str(record.get("run_group_id"))][str(record.get("measurement_mode"))] = record
    output: list[dict[str, Any]] = []
    for group_id, modes in sorted(groups.items()):
        missing = sorted({"time", "memory"}.difference(modes))
        mismatches: list[str] = []
        if not missing:
            time_record = modes["time"]
            memory_record = modes["memory"]
            for field in ("W", "status", "output_length"):
                if time_record.get(field) != memory_record.get(field):
                    mismatches.append(field)
        output.append(
            {
                "run_group_id": group_id,
                "missing_modes": missing,
                "mismatched_fields": mismatches,
                "consistent": not missing and not mismatches,
            }
        )
    return output


def resource_boundaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[float, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        if record.get("task", "oneshot") != "oneshot":
            continue
        key = (
            str(record.get("experiment_id")),
            str(record.get("algorithm")),
            str(record.get("measurement_mode")),
        )
        x_value = _number(record.get("x_value"))
        if x_value is not None:
            groups[key][x_value].add(str(record.get("status")))
    output: list[dict[str, Any]] = []
    for key, by_x in sorted(groups.items()):
        successful = sorted(x for x, statuses in by_x.items() if statuses == {"OK"})
        failed = sorted(
            x
            for x, statuses in by_x.items()
            if statuses.intersection(
                {"OOM", "MEMORY-CAP-EXCEEDED", "TO"}
            )
        )
        output.append(
            {
                "experiment_id": key[0],
                "algorithm": key[1],
                "measurement_mode": key[2],
                "last_successful_x": successful[-1] if successful else None,
                "first_resource_failure_x": failed[0] if failed else None,
            }
        )
    return output


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    preflight = publication_preflight(records)
    reconciled, mismatches = derive_count_consistency(records)
    return {
        "schema_version": "anchor-aggregate-v1",
        "record_count": len(records),
        "preflight": preflight,
        "derived_count_mismatch_count": sum(
            record.get("status") == "COUNT-MISMATCH" for record in reconciled
        ),
        "output_consistency_mismatches": mismatches,
        "metrics": aggregate_metrics(reconciled),
        "speedups": aggregate_speedups(reconciled),
        "statuses": status_summary(reconciled),
        "count_consistency": count_consistency(records),
        "run_group_consistency": run_group_consistency(reconciled),
        "resource_boundaries": resource_boundaries(reconciled),
    }


def write_aggregate_json(path: str | pathlib.Path, aggregate: Mapping[str, Any]) -> None:
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(aggregate) + b"\n")


def write_metrics_csv(path: str | pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
