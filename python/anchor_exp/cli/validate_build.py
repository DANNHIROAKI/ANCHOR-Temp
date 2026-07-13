"""Run the frozen small-instance uniformity and independence gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
from typing import Any

import numpy as np

from anchor_exp.stable_hash import canonical_json_bytes, hash_file, stable_hash
from anchor_exp.validation import (
    PAIR_DTYPE,
    distribution_report,
    holm_rejections,
    materialize_join,
    validate_pairs,
)
from anchor_exp.workload import read_workload


ALGORITHMS = ("ac", "as", "sweeprt", "liftedrt")


def _parse_one_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("benchmark produced no validation record")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError("benchmark validation stdout is not one JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError("benchmark validation record is not an object")
    return value


def _sample_count(join_size: int, override: int | None) -> int:
    protocol = min(10_000_000, max(1_000_000, 1000 * join_size))
    if override is None:
        return protocol
    if override <= 0:
        raise ValueError("--sample-count must be positive")
    return override


def _seed(workload_sha256: str, algorithm: str) -> str:
    return stable_hash(
        "uniformity-validation-seed-v1", workload_sha256, algorithm
    ).hex()


def _run_algorithm(
    benchmark: pathlib.Path,
    workload_path: pathlib.Path,
    algorithm: str,
    samples: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="anchor-validation-") as directory:
        dump = pathlib.Path(directory) / "pairs.bin"
        completed = subprocess.run(
            [
                str(benchmark),
                "run",
                "--workload",
                str(workload_path),
                "--algorithm",
                algorithm,
                "--samples",
                str(samples),
                "--seed-hex",
                _seed(hash_file(workload_path), algorithm),
                "--measurement-mode",
                "time",
                "--task",
                "oneshot",
                "--timeout-seconds",
                str(timeout_seconds),
                "--dump-output",
                str(dump),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds + 300,
        )
        record = _parse_one_json(completed.stdout)
        if completed.returncode != 0 or record.get("status") != "OK":
            raise RuntimeError(
                f"{algorithm} validation sampling failed: exit={completed.returncode}, "
                f"record={record}, stderr={completed.stderr[-1000:]!r}"
            )
        expected_bytes = samples * PAIR_DTYPE.itemsize
        if not dump.is_file() or dump.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"{algorithm} output dump has {dump.stat().st_size if dump.exists() else 0} "
                f"bytes, expected {expected_bytes}"
            )
        pairs = np.fromfile(dump, dtype=PAIR_DTYPE)
        if hash_file(dump) != record.get("output_sha256"):
            raise RuntimeError(f"{algorithm} output dump SHA-256 mismatch")
        return record, pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workload",
        type=pathlib.Path,
        action="append",
        required=True,
        help="materializable canonical workload; repeat for a validation family",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-cross-product", type=int, default=100_000_000)
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--family-level", type=float, default=0.01)
    parser.add_argument(
        "--sample-count",
        type=int,
        help=(
            "developer-only override; omission enforces "
            "min(1e7,max(1e6,1000W))"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    benchmark = args.benchmark.resolve()
    if not benchmark.is_file():
        raise FileNotFoundError(benchmark)
    if not 0.0 < args.family_level < 1.0:
        raise ValueError("--family-level must lie in (0,1)")

    results: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}
    protocol_sample_counts = args.sample_count is None
    for source in args.workload:
        workload_path = source.resolve()
        workload = read_workload(
            workload_path, verify_payload=True, verify_file=True
        )
        join = materialize_join(
            workload, max_cross_product=args.max_cross_product
        )
        if not 2 <= len(join) <= 2000:
            raise ValueError(
                f"{workload_path}: validation requires 2 <= W <= 2000, got {len(join)}"
            )
        samples = _sample_count(len(join), args.sample_count)
        workload_sha = str(workload.file_sha256)
        for algorithm in ALGORITHMS:
            record, pairs = _run_algorithm(
                benchmark,
                workload_path,
                algorithm,
                samples,
                args.timeout_seconds,
            )
            if int(record.get("W", -1)) != len(join):
                raise RuntimeError(
                    f"{algorithm} returned W={record.get('W')}, oracle W={len(join)}"
                )
            if int(record.get("output_length", -1)) != samples:
                raise RuntimeError(f"{algorithm} returned an incomplete validation batch")
            validate_pairs(workload, pairs)
            report = distribution_report(join, pairs, buckets=args.buckets)
            label = f"{workload_sha}:{algorithm}"
            pvalues[f"{label}:pair-frequency"] = report.pearson_pvalue
            if report.transition_pvalue is None:
                raise RuntimeError(
                    f"{algorithm} transition test is undefined; choose a workload "
                    "whose stable pair buckets occupy at least two rows and columns"
                )
            pvalues[f"{label}:bucket-transition"] = report.transition_pvalue
            results.append(
                {
                    "workload": str(workload_path),
                    "workload_sha256": workload_sha,
                    "algorithm": algorithm,
                    "join_size": len(join),
                    "sample_count": samples,
                    "output_sha256": record["output_sha256"],
                    "pearson_statistic": report.pearson_statistic,
                    "pearson_pvalue": report.pearson_pvalue,
                    "transition_statistic": report.transition_statistic,
                    "transition_pvalue": report.transition_pvalue,
                    "lag1_bucket_autocorrelation": report.lag1_bucket_autocorrelation,
                    "buckets": report.buckets,
                }
            )

    rejected = sorted(holm_rejections(pvalues, family_level=args.family_level))
    status = (
        "OK"
        if not rejected and protocol_sample_counts
        else (
            "DEVELOPER-SMOKE-ONLY"
            if not protocol_sample_counts and not rejected
            else "UNIFORMITY-VALIDATION-FAILED"
        )
    )
    output = {
        "schema_version": "anchor-validation-report-v1",
        "status": status,
        "benchmark_path": str(benchmark),
        "benchmark_sha256": hash_file(benchmark),
        "protocol_sample_counts": protocol_sample_counts,
        "family_level": args.family_level,
        "pvalues": pvalues,
        "holm_rejections": rejected,
        "results": results,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(output) + b"\n")
    print(
        json.dumps(
            {"status": status, "output": str(destination), "tests": len(pvalues)},
            sort_keys=True,
        )
    )
    return 0 if status in {"OK", "DEVELOPER-SMOKE-ONLY"} else 5


if __name__ == "__main__":
    raise SystemExit(main())
