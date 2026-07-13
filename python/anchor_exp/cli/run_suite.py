"""Expand and execute the twelve ANCHOR experiment sweeps."""

from __future__ import annotations

import argparse
import json
import pathlib

from anchor_exp.experiments import (
    expand_experiments,
    expand_runs,
    load_config,
    run_suite,
    validate_machine,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=pathlib.Path, default="configs/experiments.json")
    parser.add_argument("--machine", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--experiment-id", action="append", default=[])
    parser.add_argument(
        "--tasks",
        default="oneshot",
        help="comma-separated: oneshot,count-only,prepared-query (default: oneshot)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_machine(path: pathlib.Path, *, allow_placeholders: bool = False) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        machine = json.load(stream)
    if machine.get("schema_version") != "anchor-machine-v1":
        raise ValueError("unsupported machine manifest schema")
    placeholders = [
        key
        for key in ("machine_id", "code_commit", "build_sha256", "compiler_id", "allocator_id", "prng_id", "kernel_id")
        if str(machine.get(key, "")).startswith("REPLACE")
    ]
    if placeholders and not allow_placeholders:
        raise ValueError(f"machine manifest still contains placeholders: {placeholders}")
    return machine


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.experiments)
    machine = _load_machine(args.machine, allow_placeholders=args.dry_run)
    if not args.dry_run:
        validate_machine(machine)
    cases = expand_experiments(
        config,
        data_root=args.data_root,
        require_workloads=not args.dry_run,
    )
    if args.experiment_id:
        selected = set(args.experiment_id)
        cases = [case for case in cases if case.experiment_id in selected]
        missing = selected.difference(case.experiment_id for case in cases)
        if missing:
            raise ValueError(f"unknown experiment ids: {sorted(missing)}")
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    unknown = set(tasks).difference({"oneshot", "count-only", "prepared-query"})
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    runs = expand_runs(cases, config, tasks=tasks)
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "runs": len(runs)}, sort_keys=True))
        return 0
    completed = run_suite(runs, machine, args.output, resume=not args.no_resume)
    print(json.dumps({"status": "OK", "new_records": completed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
