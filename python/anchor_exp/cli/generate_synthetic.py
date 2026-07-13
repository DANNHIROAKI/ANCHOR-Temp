"""CLI for certified Alacarte workload generation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

from anchor_exp.alacarte import (
    AlacarteConfig,
    CoverageStatus,
    NumericDegeneracyError,
    config_from_mapping,
    generate_at_coverage,
    solve_coverage,
)
from anchor_exp.workload import write_workload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve and certify the Alacarte coverage parameter, generate independent "
            "R/S box sets, and write a canonical binary workload."
        )
    )
    parser.add_argument("--output", required=True, type=pathlib.Path, help="output .bin workload")
    parser.add_argument("--manifest", type=pathlib.Path, help="manifest path (default: OUTPUT.manifest.json)")
    parser.add_argument("--config", type=pathlib.Path, help="alacarte-config-v1 JSON")
    parser.add_argument("--seed", type=int, help="override data_seed")
    cardinality = parser.add_mutually_exclusive_group()
    cardinality.add_argument("--n-total", type=int, help="balanced total N (R receives ceil(N/2))")
    cardinality.add_argument("--n-r", type=int, help="override R cardinality; pair with --n-s")
    parser.add_argument("--n-s", type=int, help="override S cardinality")
    parser.add_argument("--dimension", type=int, help="override d; resets default universe to [0,1)^d")
    parser.add_argument("--alpha", type=float, help="override target output density")
    parser.add_argument("--shape-sigma", type=float, help="override shape sigma")
    parser.add_argument(
        "--volume-family",
        choices=("fixed", "exponential", "lognormal", "normal"),
        help="override volume family",
    )
    parser.add_argument("--volume-cv", type=float, help="override volume CV")
    parser.add_argument("--epsilon-alpha", type=float, help="override absolute density tolerance")
    parser.add_argument("--delta", type=float, help="override certification failure budget")
    parser.add_argument(
        "--status-json",
        type=pathlib.Path,
        help="also write the CLI result/status as JSON, including failures",
    )
    return parser


def _load(path: pathlib.Path | None) -> tuple[AlacarteConfig, int, dict[str, Any]]:
    if path is None:
        return AlacarteConfig(), 0, {}
    with path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict):
        raise ValueError("Alacarte config root must be a JSON object")
    schema = mapping.get("schema_version", "alacarte-config-v1")
    if schema != "alacarte-config-v1":
        raise ValueError(f"unsupported Alacarte config schema: {schema!r}")
    return config_from_mapping(mapping), int(mapping.get("data_seed", 0)), mapping


def _overrides(config: AlacarteConfig, args: argparse.Namespace) -> AlacarteConfig:
    updates: dict[str, Any] = {}
    if args.n_total is not None:
        updates["n_r"] = (args.n_total + 1) // 2
        updates["n_s"] = args.n_total // 2
    elif args.n_r is not None:
        if args.n_s is None:
            raise ValueError("--n-r requires --n-s")
        updates["n_r"] = args.n_r
        updates["n_s"] = args.n_s
    elif args.n_s is not None:
        raise ValueError("--n-s requires --n-r")
    if args.dimension is not None:
        updates["dimension"] = args.dimension
        updates["universe_lower"] = None
        updates["universe_upper"] = None
    if args.alpha is not None:
        updates["alpha_target"] = args.alpha
    if args.shape_sigma is not None:
        updates["shape_sigma"] = args.shape_sigma
    if args.volume_family is not None:
        updates["volume_family"] = args.volume_family
        if args.volume_cv is None:
            updates["volume_cv"] = {
                "fixed": 0.0,
                "exponential": 1.0,
                "lognormal": config.volume_cv,
                "normal": min(config.volume_cv, 0.999999999),
            }[args.volume_family]
    if args.volume_cv is not None:
        updates["volume_cv"] = args.volume_cv
    if args.epsilon_alpha is not None:
        updates["epsilon_alpha"] = args.epsilon_alpha
    if args.delta is not None:
        updates["delta"] = args.delta
    return dataclasses.replace(config, **updates)


def _emit(value: dict[str, Any], destination: pathlib.Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    print(rendered)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, config_seed, original_mapping = _load(args.config)
        config = _overrides(config, args)
        config.validate()
        seed = config_seed if args.seed is None else args.seed
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        _emit(
            {"status": CoverageStatus.INVALID_INPUT.value, "error": str(error)},
            args.status_json,
        )
        return 2

    coverage = solve_coverage(config, seed)
    if not coverage.certified or coverage.theta is None:
        _emit(
            {
                "status": coverage.status.value,
                "coverage": coverage.to_dict(),
                "output": str(args.output),
            },
            args.status_json,
        )
        return 3

    try:
        dataset = generate_at_coverage(
            config,
            coverage.theta,
            seed,
            coverage=coverage,
        )
        dataset.metadata["data_seed"] = seed
        dataset.metadata["source_config_path"] = str(args.config) if args.config else None
        dataset.metadata["source_config_extras"] = {
            key: original_mapping[key]
            for key in ("experiment_id", "dataset_id", "workload_id", "sweep", "sweep_value")
            if key in original_mapping
        }
        manifest = write_workload(
            args.output,
            r_ids=dataset.r_ids,
            r_lower=dataset.r_lower,
            r_upper=dataset.r_upper,
            s_ids=dataset.s_ids,
            s_lower=dataset.s_lower,
            s_upper=dataset.s_upper,
            endpoint_type="float64",
            metadata=dataset.metadata,
            manifest_path=args.manifest,
        )
    except (OSError, ValueError, NumericDegeneracyError) as error:
        _emit(
            {
                "status": "NUMERIC-DEGENERACY" if isinstance(error, NumericDegeneracyError) else "WRITE-FAILED",
                "coverage": coverage.to_dict(),
                "error": str(error),
            },
            args.status_json,
        )
        return 4

    result = {
        "status": CoverageStatus.CERTIFIED.value,
        "output": str(args.output.resolve()),
        "workload_sha256": manifest["workload"]["sha256"],
        "payload_sha256": manifest["workload"]["payload_sha256"],
        "n_R": config.n_r,
        "n_S": config.n_s,
        "dimension": config.dimension,
        "alpha_target": config.alpha_target,
        "alpha_expected": coverage.output_density_estimate,
        "coverage_interval": coverage.output_density_interval,
        "coverage_theta": coverage.theta,
    }
    _emit(result, args.status_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
