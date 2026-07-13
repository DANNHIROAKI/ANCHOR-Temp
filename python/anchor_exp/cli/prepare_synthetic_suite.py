"""Materialize every unique synthetic workload referenced by a suite config."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

from anchor_exp.alacarte import (
    AlacarteConfig,
    CoverageFailure,
    config_from_mapping,
    generate_synthetic,
)
from anchor_exp.experiments import expand_experiments, load_config
from anchor_exp.stable_hash import hash_file, stable_hash_hex
from anchor_exp.workload import read_manifest, write_workload


def _epsilon(config: AlacarteConfig, policy: dict[str, Any]) -> float:
    if policy.get("type") != "relative_with_absolute_floor_and_cap":
        raise ValueError("unsupported epsilon_alpha policy")
    value = max(
        float(policy["absolute_floor"]),
        min(
            float(policy["absolute_cap"]),
            float(policy["relative_to_target"]) * config.alpha_target,
            float(policy["relative_to_upper_margin"])
            * (config.a - config.alpha_target),
        ),
    )
    if not 0.0 < value < min(config.alpha_target, config.a - config.alpha_target):
        raise ValueError("epsilon_alpha policy produced an invalid strict tolerance")
    return value


def _case_config(
    base: AlacarteConfig, parameters: dict[str, Any], generation: dict[str, Any]
) -> AlacarteConfig:
    total = int(parameters.get("N", base.n_r + base.n_s))
    n_r = total // 2
    n_s = total - n_r
    dimension = int(parameters.get("d", base.dimension))
    solver = dataclasses.replace(base.solver, **generation.get("solver_overrides", {}))
    config = dataclasses.replace(
        base,
        n_r=n_r,
        n_s=n_s,
        dimension=dimension,
        universe_lower=None,
        universe_upper=None,
        alpha_target=float(parameters.get("alpha_target", base.alpha_target)),
        shape_sigma=float(parameters.get("shape_sigma", base.shape_sigma)),
        solver=solver,
    )
    return dataclasses.replace(
        config,
        epsilon_alpha=_epsilon(config, generation["epsilon_alpha_policy"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all Alacarte suite workloads"
    )
    parser.add_argument("--experiments", default="configs/experiments.json")
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suite = load_config(args.experiments)
    generation = dict(suite["synthetic_generation"])
    base_path = pathlib.Path(generation["base_config"])
    if not base_path.is_absolute():
        base_path = pathlib.Path(args.experiments).resolve().parent.parent / base_path
    with base_path.open("r", encoding="utf-8") as stream:
        base_mapping = json.load(stream)
    base = config_from_mapping(base_mapping)
    cases = [
        case
        for case in expand_experiments(
            suite, data_root=args.data_root, require_workloads=False
        )
        if case.dataset_type == "synthetic"
    ]
    unique: dict[pathlib.Path, Any] = {}
    for case in cases:
        unique.setdefault(case.workload_path, case)
    expected = int(suite.get("expected_synthetic_workload_count", 25))
    if len(unique) != expected:
        raise RuntimeError(
            f"experiment suite expands to {len(unique)} unique synthetic workloads; "
            f"expected {expected}"
        )

    plan: list[dict[str, Any]] = []
    for path, case in unique.items():
        config = _case_config(base, dict(case.parameters), generation)
        item = {
            "output": str(path),
            "experiment_id": case.experiment_id,
            "data_seed": case.data_seed,
            "N": config.n_r + config.n_s,
            "d": config.dimension,
            "alpha_target": config.alpha_target,
            "epsilon_alpha": config.epsilon_alpha,
            "shape_sigma": config.shape_sigma,
        }
        plan.append(item)
        if args.dry_run:
            continue
        manifest_path = pathlib.Path(str(path) + ".manifest.json")
        if not args.overwrite and (path.exists() or manifest_path.exists()):
            if not path.is_file() or not manifest_path.is_file():
                raise RuntimeError(
                    f"partial existing workload at {path}; use --overwrite after "
                    "inspecting or removing the incomplete artifact"
                )
            manifest = read_manifest(manifest_path)
            expected_config_sha = stable_hash_hex("alacarte-config", config.to_dict())
            metadata = manifest.get("metadata", {})
            mismatches: list[str] = []
            if hash_file(path) != manifest.get("workload", {}).get("sha256"):
                mismatches.append("file SHA-256")
            if metadata.get("configuration_sha256") != expected_config_sha:
                mismatches.append("Alacarte configuration identity")
            if metadata.get("workload_id") != case.workload_id:
                mismatches.append("workload_id")
            if metadata.get("experiment_id") != case.experiment_id:
                mismatches.append("experiment_id")
            if metadata.get("data_seed") != int(case.data_seed or 0):
                mismatches.append("data_seed")
            if (
                metadata.get("epsilon_alpha_policy")
                != generation["epsilon_alpha_policy"]
            ):
                mismatches.append("epsilon_alpha_policy")
            if mismatches:
                raise RuntimeError(
                    f"existing workload {path} does not match this generation plan: "
                    + ", ".join(mismatches)
                    + "; use --overwrite to replace it explicitly"
                )
            continue
        try:
            dataset = generate_synthetic(config, int(case.data_seed or 0))
        except CoverageFailure as error:
            result = error.result
            raise RuntimeError(
                f"Alacarte certification failed for {path} "
                f"(experiment={case.experiment_id}, value={case.x_value}, "
                f"epsilon_alpha={config.epsilon_alpha}, "
                f"certification_samples={result.certification_samples}): {error}"
            ) from error
        dataset.metadata.update(
            {
                "experiment_id": case.experiment_id,
                "dataset_id": case.dataset_id,
                "workload_id": case.workload_id,
                "data_seed": int(case.data_seed or 0),
                "sweep": case.sweep,
                "sweep_parameter": case.x_name,
                "sweep_value": case.x_value,
                "epsilon_alpha_policy": generation["epsilon_alpha_policy"],
            }
        )
        write_workload(
            path,
            r_ids=dataset.r_ids,
            r_lower=dataset.r_lower,
            r_upper=dataset.r_upper,
            s_ids=dataset.s_ids,
            s_lower=dataset.s_lower,
            s_upper=dataset.s_upper,
            endpoint_type="float64",
            metadata=dataset.metadata,
        )
    print(
        json.dumps(
            {"status": "DRY-RUN" if args.dry_run else "OK", "workloads": plan},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
