from __future__ import annotations

import argparse
import json
import pathlib

from anchor_exp.environment import capture_environment, environment_violations
from anchor_exp.stable_hash import canonical_json_bytes


def _third_party_commits(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, commit = value.partition("=")
        if not separator or not name.strip() or not commit.strip():
            raise ValueError("--third-party-commit requires NAME=COMMIT")
        if name in result:
            raise ValueError(f"duplicate third-party component: {name}")
        result[name] = commit
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture an ANCHOR machine manifest")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--memory-cap-bytes", type=int, required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    parser.add_argument(
        "--monitor-cpu-core",
        type=int,
        required=True,
        help="logical CPU reserved for the 5 ms RSS polling monitor",
    )
    parser.add_argument("--memory-poll-interval-ms", type=int, default=5)
    parser.add_argument("--numa-node", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument(
        "--memory-configuration",
        required=True,
        help="frozen DIMM/frequency/channel descriptor from the experiment host",
    )
    parser.add_argument("--smt-sibling-idle-confirmed", action="store_true")
    parser.add_argument("--allocator-id", default="system-default")
    parser.add_argument("--prng-id", default="anchor-domain-rng-v1")
    parser.add_argument("--build-flags", action="append", default=[])
    parser.add_argument("--linker-id")
    parser.add_argument("--target-isa")
    parser.add_argument("--third-party-commit", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-noncompliant", action="store_true")
    args = parser.parse_args()
    manifest = capture_environment(
        benchmark_path=args.benchmark,
        memory_cap_bytes=args.memory_cap_bytes,
        cpu_core=args.cpu_core,
        monitor_cpu_core=args.monitor_cpu_core,
        memory_poll_interval_ms=args.memory_poll_interval_ms,
        numa_node=args.numa_node,
        code_commit=args.code_commit,
        validation_report_path=args.validation_report,
        memory_configuration=args.memory_configuration,
        smt_sibling_idle_confirmed=args.smt_sibling_idle_confirmed,
        allocator_id=args.allocator_id,
        prng_id=args.prng_id,
        build_flags=args.build_flags,
        linker_id=args.linker_id,
        target_isa=args.target_isa,
        third_party_commits=_third_party_commits(args.third_party_commit),
    )
    violations = environment_violations(manifest)
    manifest["protocol_violations"] = violations
    if violations and not args.allow_noncompliant:
        raise SystemExit("machine is not publication-compliant: " + "; ".join(violations))
    destination = pathlib.Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(json.dumps({"machine_id": manifest["machine_id"], "output": str(destination)}))


if __name__ == "__main__":
    main()
