#!/usr/bin/env bash
# Execute the mid profile for all twelve Setup.md sweep families.
# Every logical run has one data seed, one process repeat, one time run, and one
# memory run. Completed run keys are resumed instead of being executed again.

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

readonly TIMEOUT_SECONDS=7200
# 901 GiB is strictly greater than both 900 GiB and decimal 900 GB.  The
# effective cap is lowered to the current host/cgroup ceiling when necessary.
readonly REQUESTED_MEMORY_CAP_BYTES=$((901 * 1024 * 1024 * 1024))
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
script_path="$repo_root/$(basename -- "${BASH_SOURCE[0]}")"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
source_experiments="${SOURCE_EXPERIMENTS:-$repo_root/configs/experiments.json}"
machine="${MACHINE_MANIFEST:-$repo_root/configs/machine.json}"
data_root="${DATA_ROOT:-$repo_root/data}"
results_root="${RESULTS_ROOT:-${RESULTS_DIR:-$repo_root/results/mid}}"
tasks="${TASKS:-oneshot}"
plot_formats="${PLOT_FORMATS:-png,pdf}"
dry_run=false
reset=false
overwrite_synthetic=false
generate_plots=false

usage() {
  cat <<'EOF'
Usage: ./run_all_mid.sh [options]

Options:
  --data-root PATH          Dataset root (default: ./data)
  --results-root PATH       Results root (default: ./results/mid)
  --machine PATH            Validated anchor-machine-v1 manifest
  --experiments PATH        Base experiment configuration (mid overrides apply)
  --tasks LIST              Comma-separated tasks (default: oneshot)
  --plots                    Render plots after aggregation
  --plot-formats LIST        Comma-separated formats (default: png,pdf)
  --overwrite-synthetic      Regenerate frozen Alacarte workloads
  --reset                    Remove previous mid result artifacts
  --dry-run                  Validate and print the 12-family/55-case plan only
  -h, --help                 Show this help

The default runs only the main one-shot time and memory paths. Use
--tasks oneshot,count-only,prepared-query to request auxiliary experiments.
Missing real workloads are prepared with scripts/data/prepare_real_data.sh --all.
Every benchmark process has a hard 2-hour timeout; its main and setup timeout
fields are also 7200 seconds. The memory cap is 901 GiB, or the current
host/cgroup/allowed-NUMA ceiling when that is lower.
Sweep points with the same complete configuration execute once physically;
the measured record is explicitly aliased into every sweep that contains it.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) [[ $# -ge 2 ]] || { echo "--data-root requires a path" >&2; exit 2; }; data_root="$2"; shift 2 ;;
    --results-root) [[ $# -ge 2 ]] || { echo "--results-root requires a path" >&2; exit 2; }; results_root="$2"; shift 2 ;;
    --machine) [[ $# -ge 2 ]] || { echo "--machine requires a path" >&2; exit 2; }; machine="$2"; shift 2 ;;
    --experiments) [[ $# -ge 2 ]] || { echo "--experiments requires a path" >&2; exit 2; }; source_experiments="$2"; shift 2 ;;
    --tasks) [[ $# -ge 2 ]] || { echo "--tasks requires a value" >&2; exit 2; }; tasks="$2"; shift 2 ;;
    --plots) generate_plots=true; shift ;;
    --plot-formats) [[ $# -ge 2 ]] || { echo "--plot-formats requires a value" >&2; exit 2; }; plot_formats="$2"; shift 2 ;;
    --overwrite-synthetic) overwrite_synthetic=true; shift ;;
    --reset) reset=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fatal() { echo "ERROR: $*" >&2; exit 1; }
command -v "$python_bin" >/dev/null 2>&1 || fatal "Python interpreter not found: $python_bin"
python_bin="$(command -v "$python_bin")"
export PYTHON_BIN="$python_bin"

resolve_path() {
  "$python_bin" - "$1" <<'PY'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

source_experiments="$(resolve_path "$source_experiments")"
machine="$(resolve_path "$machine")"
data_root="$(resolve_path "$data_root")"
results_root="$(resolve_path "$results_root")"
[[ -f "$source_experiments" ]] || fatal "experiment config not found: $source_experiments"

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - <<'PY'
import importlib
import importlib.util

required = ("numpy", "scipy", "pyarrow", "requests")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "missing Python dependencies: " + ", ".join(missing)
        + "; activate the dhy environment or install requirements-preprocess.txt"
    )
for name in required:
    try:
        importlib.import_module(name)
    except Exception as error:
        raise SystemExit(f"Python dependency {name!r} cannot be imported: {error}") from error
import anchor_exp
PY

raw_dir="$results_root/raw"
log_dir="$results_root/logs"
manifest_dir="$results_root/manifests"
aggregate_dir="$results_root/aggregate"
figure_dir="$results_root/figures"
raw_jsonl="$raw_dir/results.jsonl"
raw_csv="$raw_dir/results.csv"
effective_experiments="$manifest_dir/experiments.mid.json"
effective_machine="$manifest_dir/machine.mid.json"
aggregate_json="$aggregate_dir/results.json"
metrics_csv="$aggregate_dir/metrics.csv"
run_manifest="$manifest_dir/run-manifest.json"
preview_experiments=""
preview_machine=""

cleanup_temporary_files() {
  [[ -z "$preview_experiments" ]] || rm -f -- "$preview_experiments" || :
  [[ -z "$preview_machine" ]] || rm -f -- "$preview_machine" || :
}

on_error() {
  local status=$?
  echo "ERROR: run_all_mid.sh failed at line ${BASH_LINENO[0]:-unknown} (exit=$status)." >&2
  echo "Partial benchmark records, when present, remain at $raw_jsonl.partial and are resumable." >&2
  exit "$status"
}
trap on_error ERR
trap cleanup_temporary_files EXIT

write_mid_experiments() {
  local destination="$1"
  "$python_bin" - "$source_experiments" "$destination" "$repo_root" \
    "$TIMEOUT_SECONDS" "$REQUESTED_MEMORY_CAP_BYTES" <<'PY'
import json, pathlib, sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
repo_root = pathlib.Path(sys.argv[3])
timeout = int(sys.argv[4])
requested_memory_cap = int(sys.argv[5])
config = json.loads(source.read_text(encoding="utf-8"))

if config.get("schema_version") != "anchor-experiments-v1":
    raise SystemExit("unsupported experiment configuration schema")

# A mid run is a single independent repetition, exactly like lite.
config["data_seeds"] = [0]
config["sample_seed_id"] = 0
config["process_repeat_ids"] = [0]

generation = config.get("synthetic_generation")
if not isinstance(generation, dict):
    raise SystemExit("synthetic_generation is missing")
base = pathlib.Path(str(generation.get("base_config", "")))
if not base.is_absolute():
    candidates = (
        source.parent.parent / base,
        repo_root / base,
        source.parent / base,
    )
    base = next((item.resolve() for item in candidates if item.is_file()), candidates[0].resolve())
if not base.is_file():
    raise SystemExit(f"Alacarte base configuration not found: {base}")
generation["base_config"] = str(base)

synthetic_ids = {
    "Alacarte-G1-N",
    "Alacarte-G2-t",
    "Alacarte-G3-alpha",
    "Alacarte-G4-shape",
    "Alacarte-G5-d",
}
sweeps = {
    str(item.get("experiment_id")): item
    for item in config.get("sweeps", [])
    if item.get("dataset_type") == "synthetic"
}
if set(sweeps) != synthetic_ids:
    raise SystemExit(f"unexpected synthetic sweep ids: {sorted(sweeps)}")

for experiment_id, sweep in sweeps.items():
    fixed = dict(sweep.get("fixed", {}))
    if experiment_id != "Alacarte-G5-d":
        fixed["d"] = 3
    sweep["fixed"] = fixed

    # d is absent from four lite path templates. Reusing those paths would
    # either reject d=3 or overwrite the frozen d=2 workloads, so every mid
    # synthetic workload lives in its own namespace.
    template = str(sweep.get("workload_template", ""))
    if "/workloads/alacarte_mid/" not in template:
        marker = "/workloads/alacarte/"
        if marker not in template:
            raise SystemExit(f"cannot isolate synthetic template for {experiment_id}: {template}")
        template = template.replace(marker, "/workloads/alacarte_mid/", 1)
    sweep["workload_template"] = template

config["mid_policy"] = {
    "schema_version": "anchor-mid-policy-v1",
    "data_seeds": [0],
    "process_repeat_ids": [0],
    "sample_seed_id": 0,
    "default_synthetic_dimension": 3,
    "timeout_seconds": timeout,
    "setup_timeout_seconds": timeout,
    "process_timeout_seconds": timeout,
    "requested_memory_cap_bytes": requested_memory_cap,
    "physical_execution_deduplication": "semantic-configuration-v1",
    "result_aliases_preserve_all_sweep_points": True,
    "sample_seed_scheme": "sample-master-seed-mid-semantic-v1",
    "note": "All sweep values are retained; each semantic configuration is physically executed once.",
}

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

write_effective_machine() {
  local destination="$1"
  "$python_bin" - "$machine" "$destination" "$TIMEOUT_SECONDS" \
    "$REQUESTED_MEMORY_CAP_BYTES" "$script_path" <<'PY'
import json, os, pathlib, re, sys
from anchor_exp.experiments import validate_machine
from anchor_exp.stable_hash import canonical_json_bytes, hash_file, stable_hash

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
timeout = int(sys.argv[3])
requested = int(sys.argv[4])
runner_script = pathlib.Path(sys.argv[5]).resolve()
if timeout != 2 * 60 * 60:
    raise SystemExit(f"mid timeout must be exactly 7200 seconds, got {timeout}")
if requested <= 900 * 1024**3:
    raise SystemExit("mid requested memory cap must be strictly greater than 900 GiB")

machine = json.loads(source.read_text(encoding="utf-8"))
limits = {}

def add_limit(name: str, value: object) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return
    if parsed > 0:
        limits[name] = parsed

host = machine.get("host", {})
if isinstance(host, dict):
    add_limit("manifest_host_total", host.get("memory_total_bytes"))

meminfo = pathlib.Path("/proc/meminfo")
if meminfo.is_file():
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            add_limit("proc_memtotal", int(line.split()[1]) * 1024)
            break

def parse_node_list(value: str) -> list[int]:
    nodes = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            nodes.update(range(start, end + 1))
        else:
            nodes.add(int(item))
    return sorted(nodes)

def node_total_bytes(nodes: list[int]) -> int:
    total = 0
    for node in nodes:
        node_meminfo = pathlib.Path(
            f"/sys/devices/system/node/node{node}/meminfo"
        )
        if not node_meminfo.is_file():
            return 0
        match = re.search(
            r"MemTotal:\s+(\d+)\s+kB",
            node_meminfo.read_text(encoding="utf-8"),
        )
        if not match:
            return 0
        total += int(match.group(1)) * 1024
    return total

# Honor the tightest finite memory.max from the current cgroup-v2 ancestry.
cgroup_file = pathlib.Path("/proc/self/cgroup")
cgroup_root = pathlib.Path("/sys/fs/cgroup")
cgroup_values = []
allowed_numa_nodes = []
if cgroup_file.is_file() and cgroup_root.is_dir():
    for line in cgroup_file.read_text(encoding="utf-8").splitlines():
        if "::" not in line:
            continue
        relative = line.split("::", 1)[1].strip().lstrip("/")
        current = (cgroup_root / relative).resolve()
        root = cgroup_root.resolve()
        if current != root and root not in current.parents:
            continue
        cpuset_path = current / "cpuset.mems.effective"
        if cpuset_path.is_file():
            allowed_numa_nodes = parse_node_list(
                cpuset_path.read_text(encoding="utf-8")
            )
        while True:
            limit_path = current / "memory.max"
            if limit_path.is_file():
                value = limit_path.read_text(encoding="utf-8").strip()
                if value != "max":
                    try:
                        parsed = int(value)
                    except ValueError:
                        pass
                    else:
                        if parsed > 0:
                            cgroup_values.append(parsed)
            if current == root:
                break
            current = current.parent
if cgroup_values:
    add_limit("cgroup_v2", min(cgroup_values))
if not allowed_numa_nodes:
    online_nodes = pathlib.Path("/sys/devices/system/node/online")
    if online_nodes.is_file():
        allowed_numa_nodes = parse_node_list(
            online_nodes.read_text(encoding="utf-8")
        )
allowed_numa_total = node_total_bytes(allowed_numa_nodes)
if allowed_numa_total:
    add_limit("allowed_numa_nodes_total", allowed_numa_total)

if not limits:
    raise SystemExit("cannot determine the current machine memory ceiling")
detected_limit = min(limits.values())
effective_cap = min(requested, detected_limit)

machine["timeout_seconds"] = timeout
machine["setup_timeout_seconds"] = timeout
machine["process_timeout_seconds"] = timeout
machine["memory_cap_bytes"] = effective_cap
machine["memory_cap_requested_bytes"] = requested
machine["memory_limit_detected_bytes"] = detected_limit
machine["memory_limit_sources"] = limits
machine["memory_numa_policy"] = "interleave-allowed-nodes"
machine["memory_numa_nodes"] = allowed_numa_nodes
machine["numa_node_semantics"] = (
    "cpu-local node only; memory placement is defined by memory_numa_policy"
)
machine["runner_script_path"] = str(runner_script)
machine["runner_script_sha256"] = hash_file(runner_script)
machine["execution_profile"] = "mid"
identity = {
    key: value
    for key, value in machine.items()
    if key not in {"machine_id", "protocol_violations"}
}
machine["machine_id"] = stable_hash("machine-manifest-v1", identity)[:16].hex()

# The repository's publication validator freezes 900-second timeouts. Validate
# every other invariant against a normalized copy; the mid runner below checks
# and uses the actual 7200-second values.
validation_copy = dict(machine)
validation_copy["timeout_seconds"] = 900
validation_copy["setup_timeout_seconds"] = 900
validation_identity = {
    key: value
    for key, value in validation_copy.items()
    if key not in {"machine_id", "protocol_violations"}
}
validation_copy["machine_id"] = stable_hash(
    "machine-manifest-v1", validation_identity
)[:16].hex()
validate_machine(validation_copy)

destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_bytes(canonical_json_bytes(machine) + b"\n")
os.chmod(temporary, 0o644)
temporary.replace(destination)
print(json.dumps({
    "machine_id": machine["machine_id"],
    "timeout_seconds": timeout,
    "setup_timeout_seconds": timeout,
    "process_timeout_seconds": timeout,
    "requested_memory_cap_bytes": requested,
    "detected_memory_limit_bytes": detected_limit,
    "effective_memory_cap_bytes": effective_cap,
    "memory_limit_sources": limits,
    "memory_numa_policy": machine["memory_numa_policy"],
    "memory_numa_nodes": machine["memory_numa_nodes"],
    "numa_node_semantics": machine["numa_node_semantics"],
}, sort_keys=True))
PY
}

preview_experiments="$(mktemp "${TMPDIR:-/tmp}/anchor-mid-experiments.XXXXXX.json")"
write_mid_experiments "$preview_experiments"

log "Validating the frozen mid plan"
"$python_bin" - "$preview_experiments" "$data_root" "$tasks" <<'PY'
import collections, sys
from anchor_exp.experiments import expand_experiments, expand_runs, load_config

config = load_config(sys.argv[1])
if config.get("algorithms") != ["AC", "AS", "SweepRT", "LiftedRT"]:
    raise SystemExit("algorithm order must be AC, AS, SweepRT, LiftedRT")
if (
    config.get("data_seeds") != [0]
    or config.get("process_repeat_ids") != [0]
    or config.get("sample_seed_id") != 0
):
    raise SystemExit("mid config must freeze data_seed, sample_seed_id, and process_repeat_id to 0")

expected = {
    "Alacarte-G1-N": {
        "sweep": "N-sweep", "varying": "N",
        "values": [20000, 100000, 200000, 1000000, 2000000],
        "fixed": {"d": 3, "t": 100000, "alpha_target": 10.0, "shape_sigma": 0.0},
    },
    "Alacarte-G2-t": {
        "sweep": "t-sweep", "varying": "t",
        "values": [1000, 10000, 100000, 1000000, 10000000],
        "fixed": {"N": 200000, "d": 3, "alpha_target": 10.0, "shape_sigma": 0.0},
    },
    "Alacarte-G3-alpha": {
        "sweep": "alpha-sweep", "varying": "alpha_target",
        "values": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        "fixed": {"N": 200000, "d": 3, "t": 100000, "shape_sigma": 0.0},
    },
    "Alacarte-G4-shape": {
        "sweep": "shape-sweep", "varying": "shape_sigma",
        "values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "fixed": {"N": 200000, "d": 3, "t": 100000, "alpha_target": 10.0},
    },
    "Alacarte-G5-d": {
        "sweep": "d-sweep", "varying": "d", "values": [2, 3, 4, 5],
        "fixed": {"N": 200000, "t": 100000, "alpha_target": 10.0, "shape_sigma": 0.0},
    },
}
synthetic = {
    str(item.get("experiment_id")): item
    for item in config.get("sweeps", [])
    if item.get("dataset_type") == "synthetic"
}
if set(synthetic) != set(expected):
    raise SystemExit(f"unexpected synthetic sweep ids: {sorted(synthetic)}")
for experiment_id, wanted in expected.items():
    actual = synthetic[experiment_id]
    for field, value in wanted.items():
        if actual.get(field) != value:
            raise SystemExit(
                f"{experiment_id}.{field} mismatch: expected {value!r}, got {actual.get(field)!r}"
            )
    if "/workloads/alacarte_mid/" not in str(actual.get("workload_template", "")):
        raise SystemExit(f"{experiment_id} does not use the isolated mid workload namespace")

task_items = tuple(item.strip() for item in sys.argv[3].split(",") if item.strip())
unknown = set(task_items) - {"oneshot", "count-only", "prepared-query"}
if unknown or not task_items:
    raise SystemExit(f"invalid task list: {sorted(unknown)}")
duplicates = [item for item, count in collections.Counter(task_items).items() if count > 1]
if duplicates:
    raise SystemExit(f"duplicate tasks would repeat a configuration: {duplicates}")

cases = expand_experiments(config, data_root=sys.argv[2], require_workloads=False)
runs = expand_runs(cases, config, tasks=task_items)
if len(cases) != 55 or len({case.experiment_id for case in cases}) != 12:
    raise SystemExit(f"expected 12 sweeps and 55 cases, got {len(cases)} cases")
by_dataset = collections.Counter(case.dataset_id for case in cases)
synthetic_paths = {
    case.workload_path for case in cases if case.dataset_type == "synthetic"
}
if len(synthetic_paths) != 25:
    raise SystemExit(f"expected 25 unique Alacarte workloads, got {len(synthetic_paths)}")
run_keys = [(run.run_group_id, run.measurement_mode) for run in runs]
if len(run_keys) != len(set(run_keys)):
    raise SystemExit("expanded plan contains duplicate run keys")
if any(run.process_repeat_id != 0 or run.sample_seed_id != 0 for run in runs):
    raise SystemExit("expanded plan contains an independent repetition")

def semantic_case_key(case):
    return (
        case.dataset_id,
        tuple(
            sorted(
                (str(key), repr(value))
                for key, value in case.parameters.items()
            )
        ),
    )

semantic_cases = {semantic_case_key(case) for case in cases}
if len(semantic_cases) != 48:
    raise SystemExit(
        f"expected 48 semantic configurations after cross-sweep reuse, got {len(semantic_cases)}"
    )
physical_keys = {
    (
        semantic_case_key(run.case),
        run.algorithm,
        run.task,
        run.measurement_mode,
        run.sample_seed_id,
        run.process_repeat_id,
    )
    for run in runs
}
per_case = sum(
    8 if task == "oneshot" else 4 if task == "count-only" else 3
    for task in task_items
)
expected_physical = len(semantic_cases) * per_case
if len(physical_keys) != expected_physical:
    raise SystemExit(
        f"expected {expected_physical} physical execution keys, got {len(physical_keys)}"
    )
print(
    f"sweeps=12 cases={len(cases)} synthetic_workloads=25 "
    f"logical_run_records={len(runs)} unique_run_keys={len(run_keys)} "
    f"semantic_configurations={len(semantic_cases)} "
    f"physical_benchmark_processes={len(physical_keys)} "
    f"tasks={','.join(task_items)}"
)
for dataset in ("Alacarte", "CMAB-1M", "GeoLife-3D-1M", "GeoLife-4D-1M", "COCO-1M"):
    print(f"  {dataset}: cases={by_dataset[dataset]}")
PY

[[ -r /proc/self/status ]] || fatal "/proc/self/status is unavailable"
grep -q '^VmRSS:' /proc/self/status || fatal "/proc/self/status has no VmRSS field"
[[ -f "$machine" ]] || fatal "validated machine manifest not found: $machine"

if [[ "$dry_run" == true ]]; then
  preview_machine="$(mktemp "${TMPDIR:-/tmp}/anchor-mid-machine.XXXXXX.json")"
  write_effective_machine "$preview_machine"
  log "Dry-run complete; no datasets, result directories, or benchmarks were touched"
  exit 0
fi

mkdir -p "$raw_dir" "$log_dir" "$manifest_dir" "$aggregate_dir"
command -v flock >/dev/null 2>&1 || fatal "flock is required to prevent duplicate concurrent runs"
exec 9>"$results_root/.run_all_mid.lock"
flock -n 9 || fatal "another run_all_mid.sh is already using $results_root"

if [[ "$overwrite_synthetic" == true && "$reset" != true ]] \
  && [[ -e "$raw_jsonl" || -e "$raw_jsonl.partial" ]]
then
  fatal "--overwrite-synthetic with existing results requires --reset"
fi

if [[ "$reset" == true ]]; then
  log "Removing previous mid result artifacts"
  rm -f -- "$raw_jsonl" "$raw_jsonl.partial" "$raw_csv" "$aggregate_json" \
    "$metrics_csv" "$run_manifest" "$effective_experiments" "$effective_machine"
  rm -rf -- "$figure_dir" "$log_dir"
  mkdir -p "$log_dir"
fi

"$python_bin" - "$preview_experiments" "$effective_experiments" <<'PY'
import os, pathlib, sys
source, destination = map(pathlib.Path, sys.argv[1:3])
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_bytes(source.read_bytes())
os.chmod(temporary, 0o644)
temporary.replace(destination)
PY
log "Creating the effective 2-hour, 901-GiB-or-host-limit machine manifest"
write_effective_machine "$effective_machine" | tee "$log_dir/00-effective-machine.log"

mkdir -p "$data_root"
exec 8>"$data_root/.run_all_mid.prepare.lock"
flock -n 8 || fatal "another run_all_mid.sh is preparing workloads in $data_root"

log "Checking final real-workload checksums"
if ! "$repo_root/scripts/data/prepare_real_data.sh" \
  --all --data-root "$data_root" --verify-only
then
  log "Real workloads are missing; importing the pinned Hugging Face datasets"
  "$repo_root/scripts/data/prepare_real_data.sh" --all --data-root "$data_root" | tee "$log_dir/01-prepare-real-data.log"
fi

log "Preparing all 25 isolated mid Alacarte workloads with data_seed=0"
synthetic_args=(-m anchor_exp.cli.prepare_synthetic_suite --experiments "$effective_experiments" --data-root "$data_root")
[[ "$overwrite_synthetic" == true ]] && synthetic_args+=(--overwrite)
"$python_bin" "${synthetic_args[@]}" | tee "$log_dir/02-prepare-alacarte.log"

log "Verifying every canonical workload and manifest"
"$python_bin" - "$effective_experiments" "$data_root" <<'PY'
import sys
from anchor_exp.experiments import expand_experiments, load_config
cases = expand_experiments(load_config(sys.argv[1]), data_root=sys.argv[2], require_workloads=True)
if len(cases) != 55:
    raise SystemExit(f"expected 55 verified cases, got {len(cases)}")
print(f"verified_cases={len(cases)}")
PY

log "Running all datasets serially in Setup order (AC, AS, SweepRT, LiftedRT)"
"$python_bin" - "$effective_experiments" "$effective_machine" "$data_root" \
  "$raw_jsonl" "$tasks" "$TIMEOUT_SECONDS" "$REQUESTED_MEMORY_CAP_BYTES" <<'PY' \
  | tee "$log_dir/03-benchmarks.log"
import collections, json, pathlib, sys, time
import anchor_exp.experiments as experiments_module
from anchor_exp.stable_hash import hash_file, stable_hash

experiments_path, machine_path, data_root, output, tasks = sys.argv[1:6]
expected_timeout = int(sys.argv[6])
expected_requested_cap = int(sys.argv[7])
publication_validate_machine = experiments_module.validate_machine
publication_benchmark_command = experiments_module._benchmark_command
publication_identity = experiments_module._identity
publication_subprocess_run = experiments_module.subprocess.run

def validate_mid_machine(machine):
    if (
        int(machine.get("timeout_seconds", 0)) != expected_timeout
        or int(machine.get("setup_timeout_seconds", 0)) != expected_timeout
        or int(machine.get("process_timeout_seconds", 0)) != expected_timeout
    ):
        raise ValueError("mid machine manifest requires 7200-second timeouts")
    if int(machine.get("memory_cap_requested_bytes", 0)) != expected_requested_cap:
        raise ValueError("mid machine manifest has an unexpected requested memory cap")
    effective_cap = int(machine.get("memory_cap_bytes", 0))
    detected_limit = int(machine.get("memory_limit_detected_bytes", 0))
    if effective_cap != min(expected_requested_cap, detected_limit):
        raise ValueError("mid effective memory cap is not min(requested, machine limit)")

    actual_identity = {
        key: value
        for key, value in machine.items()
        if key not in {"machine_id", "protocol_violations"}
    }
    actual_machine_id = stable_hash(
        "machine-manifest-v1", actual_identity
    )[:16].hex()
    if machine.get("machine_id") != actual_machine_id:
        raise ValueError("mid machine_id does not match the effective manifest")
    runner_script = pathlib.Path(str(machine.get("runner_script_path", ""))).resolve()
    if (
        not runner_script.is_file()
        or hash_file(runner_script) != machine.get("runner_script_sha256")
    ):
        raise ValueError("mid runner script changed after the machine manifest was written")

    # Reuse all frozen publication checks except its legacy exact-900-second
    # invariant. Execution continues with the original 7200-second dictionary.
    normalized = dict(machine)
    normalized["timeout_seconds"] = 900
    normalized["setup_timeout_seconds"] = 900
    identity = {
        key: value
        for key, value in normalized.items()
        if key not in {"machine_id", "protocol_violations"}
    }
    normalized["machine_id"] = stable_hash(
        "machine-manifest-v1", identity
    )[:16].hex()
    publication_validate_machine(normalized)

config = experiments_module.load_config(experiments_path)
machine = json.loads(pathlib.Path(machine_path).read_text(encoding="utf-8"))
validate_mid_machine(machine)

def parse_node_list(value):
    nodes = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            nodes.update(range(start, end + 1))
        else:
            nodes.add(int(item))
    return sorted(nodes)

def current_allowed_numa_nodes():
    cgroup_file = pathlib.Path("/proc/self/cgroup")
    root = pathlib.Path("/sys/fs/cgroup")
    if cgroup_file.is_file() and root.is_dir():
        for line in cgroup_file.read_text(encoding="utf-8").splitlines():
            if "::" not in line:
                continue
            relative = line.split("::", 1)[1].strip().lstrip("/")
            cpuset = root / relative / "cpuset.mems.effective"
            if cpuset.is_file():
                return parse_node_list(cpuset.read_text(encoding="utf-8"))
    online = pathlib.Path("/sys/devices/system/node/online")
    if online.is_file():
        return parse_node_list(online.read_text(encoding="utf-8"))
    return []

recorded_numa_nodes = list(machine.get("memory_numa_nodes", []))
live_numa_nodes = current_allowed_numa_nodes()
if not recorded_numa_nodes or live_numa_nodes != recorded_numa_nodes:
    raise RuntimeError(
        "allowed NUMA nodes changed after the effective machine manifest was written: "
        f"recorded={recorded_numa_nodes}, live={live_numa_nodes}"
    )

def mid_benchmark_command(run, machine_value):
    if machine_value.get("memory_numa_policy") != "interleave-allowed-nodes":
        raise ValueError("mid machine manifest requires interleave-allowed-nodes policy")
    numactl = pathlib.Path(str(machine_value.get("numactl_path", ""))).resolve()
    if not numactl.is_file():
        raise FileNotFoundError(numactl)
    unwrapped = dict(machine_value)
    unwrapped["numactl_path"] = None
    command = publication_benchmark_command(run, unwrapped)
    return [
        str(numactl),
        f"--physcpubind={int(machine_value['cpu_core'])}",
        "--interleave=" + ",".join(
            str(node) for node in machine_value["memory_numa_nodes"]
        ),
        *command,
    ]

def mid_identity(run, machine_value, manifest):
    record = publication_identity(run, machine_value, manifest)
    record["memory_numa_policy"] = machine_value["memory_numa_policy"]
    record["memory_numa_nodes"] = list(machine_value["memory_numa_nodes"])
    record["numa_node_semantics"] = machine_value["numa_node_semantics"]
    return record

def mid_run_time_process(command, *, timeout_seconds):
    del timeout_seconds
    original_affinity = experiments_module.os.sched_getaffinity(0)
    process = experiments_module.subprocess.Popen(
        list(command),
        stdin=experiments_module.subprocess.DEVNULL,
        stdout=experiments_module.subprocess.PIPE,
        stderr=experiments_module.subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        experiments_module.os.sched_setaffinity(
            0, {int(machine["monitor_cpu_core"])}
        )
    except (AttributeError, OSError) as error:
        experiments_module._terminate_process_group(process)
        process.communicate()
        raise RuntimeError(
            f"cannot pin the time-run monitor to CPU {machine.get('monitor_cpu_core')}: {error}"
        ) from error
    deadline = time.monotonic() + expected_timeout
    poll_seconds = max(
        0.001, int(machine["memory_poll_interval_ms"]) / 1000.0
    )
    timed_out = False
    memory_cap_exceeded = False
    termination_signal = None
    stdout = stderr = ""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                termination_signal = experiments_module._terminate_process_group(process)
                stdout, stderr = process.communicate()
                break
            try:
                stdout, stderr = process.communicate(
                    timeout=min(poll_seconds, remaining)
                )
                break
            except experiments_module.subprocess.TimeoutExpired:
                try:
                    status = pathlib.Path(
                        f"/proc/{process.pid}/status"
                    ).read_text(encoding="utf-8")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                rss_bytes = 0
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        rss_bytes = int(line.split()[1]) * 1024
                        break
                if rss_bytes > int(machine["memory_cap_bytes"]):
                    memory_cap_exceeded = True
                    termination_signal = "SIGKILL"
                    try:
                        experiments_module.os.killpg(
                            process.pid, experiments_module.signal.SIGKILL
                        )
                    except ProcessLookupError:
                        pass
                    _, stderr = process.communicate()
                    stdout = ""
                    break
    except BaseException:
        if process.poll() is None:
            experiments_module._terminate_process_group(process)
            process.communicate()
        raise
    finally:
        experiments_module.os.sched_setaffinity(0, original_affinity)
    return (
        experiments_module.subprocess.CompletedProcess(
            list(command),
            137
            if memory_cap_exceeded
            else 124
            if timed_out
            else int(process.returncode or 0),
            stdout,
            stderr,
        ),
        timed_out,
        termination_signal,
    )

def mid_subprocess_run(command, *args, **kwargs):
    if (
        isinstance(command, (list, tuple))
        and "--memory-cap-bytes" in command
        and "--timeout-seconds" in command
        and "--report" in command
    ):
        command = list(command)
        timeout_index = command.index("--timeout-seconds") + 1
        command[timeout_index] = str(expected_timeout)
    return publication_subprocess_run(command, *args, **kwargs)

experiments_module._benchmark_command = mid_benchmark_command
experiments_module._identity = mid_identity
experiments_module._run_time_process = mid_run_time_process
experiments_module.subprocess.run = mid_subprocess_run
cases = experiments_module.expand_experiments(
    config, data_root=data_root, require_workloads=True
)
task_items = tuple(item.strip() for item in tasks.split(",") if item.strip())
unknown = set(task_items) - {"oneshot", "count-only", "prepared-query"}
if unknown or not task_items or len(task_items) != len(set(task_items)):
    raise ValueError(f"invalid or duplicated task list: {task_items}")
runs = experiments_module.expand_runs(cases, config, tasks=task_items)

manifest_cache = {}

def manifest_for(run):
    path = pathlib.Path(run.case.workload_path)
    if path not in manifest_cache:
        manifest_cache[path] = experiments_module.read_manifest(path)
    return manifest_cache[path]

def workload_sha(run):
    return str(manifest_for(run)["workload"]["sha256"])

def semantic_sample_seed_hex(run):
    return stable_hash(
        "sample-master-seed-mid-semantic-v1",
        run.case.dataset_id,
        workload_sha(run),
        run.case.t,
        run.algorithm,
        run.sample_seed_id,
    ).hex()

# Cross-sweep aliases for the same complete configuration must have identical
# workload contents and sampling. Give them a seed from semantic input identity
# instead of experiment_id/workload_id, which are presentation identities.
experiments_module.RunSpec.sample_seed_hex = property(semantic_sample_seed_hex)

def execution_key(run):
    return (
        run.case.dataset_id,
        workload_sha(run),
        run.case.t,
        run.algorithm,
        run.task,
        run.measurement_mode,
        run.sample_seed_hex,
        run.process_repeat_id,
    )

def semantic_case_key(case):
    return (
        case.dataset_id,
        tuple(
            sorted(
                (str(key), repr(value))
                for key, value in case.parameters.items()
            )
        ),
    )

groups = collections.OrderedDict()
for run in runs:
    groups.setdefault(execution_key(run), []).append(run)
semantic_case_count = len({semantic_case_key(case) for case in cases})
per_case = sum(
    8 if task == "oneshot" else 4 if task == "count-only" else 3
    for task in task_items
)
expected_physical = semantic_case_count * per_case
if semantic_case_count != 48 or len(groups) != expected_physical:
    raise ValueError(
        "semantic execution plan mismatch: "
        f"cases={semantic_case_count}, physical={len(groups)}, expected={expected_physical}"
    )

experiment_config_sha256 = cases[0].experiment_config_sha256
if any(
    case.experiment_config_sha256 != experiment_config_sha256
    for case in cases
):
    raise ValueError("expanded cases contain mixed experiment configuration identities")

lineage_fields = {
    "physical_execution_id",
    "physical_run_group_id",
    "physical_execution_reused",
    "result_alias_of_run_group_id",
    "sample_seed_scheme",
}

def identity_for(run):
    return experiments_module._identity(run, machine, manifest_for(run))

def physical_execution_id_for(key):
    return stable_hash(
        "mid-physical-execution-v1",
        machine["machine_id"],
        machine["build_sha256"],
        experiment_config_sha256,
        key,
    )[:16].hex()

def payload_for(record, run):
    immutable = identity_for(run)
    return {
        field: value
        for field, value in record.items()
        if field not in immutable and field not in lineage_fields
    }

destination = pathlib.Path(output).resolve()
staging = destination.with_name(destination.name + ".partial")
if not staging.exists() and destination.exists():
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(destination.read_bytes())

runs_by_key = {
    (run.run_group_id, run.measurement_mode): run
    for run in runs
}
if len(runs_by_key) != len(runs):
    raise ValueError("logical run plan contains duplicate run keys")
existing = {}
if staging.exists():
    with staging.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                str(record.get("run_group_id")),
                str(record.get("measurement_mode")),
            )
            if key in existing:
                raise ValueError(f"duplicate resumed run key at {staging}:{line_number}: {key}")
            run = runs_by_key.get(key)
            if run is None:
                raise ValueError(f"unknown resumed run key at {staging}:{line_number}: {key}")
            expected_identity = identity_for(run)
            for field, expected in expected_identity.items():
                if record.get(field) != expected:
                    raise ValueError(
                        f"cannot resume {staging}:{line_number}: {field} changed; "
                        "use --reset"
                    )
            existing[key] = record

# Validate lineage and the shared physical payload even for a fully completed
# group, which would otherwise be skipped by the execution loop below.
for execution, members in groups.items():
    physical_id = physical_execution_id_for(execution)
    origin_run_group_id = members[0].run_group_id
    reference_payload = None
    for run in members:
        logical_key = (run.run_group_id, run.measurement_mode)
        record = existing.get(logical_key)
        if record is None:
            continue
        reused = run.run_group_id != origin_run_group_id
        expected_lineage = {
            "physical_execution_id": physical_id,
            "physical_run_group_id": origin_run_group_id,
            "physical_execution_reused": reused,
            "result_alias_of_run_group_id": origin_run_group_id if reused else None,
            "sample_seed_scheme": "sample-master-seed-mid-semantic-v1",
        }
        for field, expected in expected_lineage.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"cannot resume {staging}: invalid {field} for {logical_key}; "
                    "use --reset"
                )
        payload = payload_for(record, run)
        if reference_payload is None:
            reference_payload = payload
        elif payload != reference_payload:
            raise ValueError(
                f"cannot resume {staging}: physical aliases have different payloads; "
                "use --reset"
            )

new_records = 0
new_physical_processes = 0
for key, members in groups.items():
    pending = [
        run
        for run in members
        if (run.run_group_id, run.measurement_mode) not in existing
    ]
    if not pending:
        continue
    available = [
        run
        for run in members
        if (run.run_group_id, run.measurement_mode) in existing
    ]
    if available:
        source_run = available[0]
        source_record = existing[(source_run.run_group_id, source_run.measurement_mode)]
    else:
        source_run = members[0]
        current_workload_sha = hash_file(source_run.case.workload_path)
        if current_workload_sha != workload_sha(source_run):
            raise ValueError(
                f"workload changed after verification: {source_run.case.workload_path}"
            )
        source_record = experiments_module.run_one(source_run, machine)
        new_physical_processes += 1

    physical_execution_id = physical_execution_id_for(key)
    origin_run_group_id = members[0].run_group_id
    payload = payload_for(source_record, source_run)
    for run in pending:
        record = identity_for(run)
        record.update(payload)
        record["physical_execution_id"] = physical_execution_id
        record["physical_run_group_id"] = origin_run_group_id
        record["physical_execution_reused"] = (
            run.run_group_id != origin_run_group_id
        )
        record["result_alias_of_run_group_id"] = (
            origin_run_group_id if record["physical_execution_reused"] else None
        )
        record["sample_seed_scheme"] = "sample-master-seed-mid-semantic-v1"
        experiments_module.append_jsonl(staging, record)
        logical_key = (run.run_group_id, run.measurement_mode)
        existing[logical_key] = record
        new_records += 1

experiments_module._finalize_reconciled_jsonl(staging, destination)
print(json.dumps({
    "status": "OK",
    "logical_run_records": len(runs),
    "physical_benchmark_processes": len(groups),
    "new_records": new_records,
    "new_physical_processes": new_physical_processes,
}, sort_keys=True))
PY
flock -u 8
exec 8>&-

log "Writing direct-value aggregate and CSV artifacts"
"$python_bin" - "$raw_jsonl" "$raw_csv" "$aggregate_json" "$metrics_csv" <<'PY' | tee "$log_dir/04-aggregate.log"
import csv, json, pathlib, sys
from anchor_exp.aggregate import aggregate_records, read_jsonl, write_aggregate_json, write_metrics_csv
raw, raw_csv, aggregate_path, metrics_path = map(pathlib.Path, sys.argv[1:5])
records = read_jsonl([raw])
fields = sorted({key for record in records for key in record})
raw_csv.parent.mkdir(parents=True, exist_ok=True)
with raw_csv.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in record.items()})
aggregate = aggregate_records(records)
write_aggregate_json(aggregate_path, aggregate)
write_metrics_csv(metrics_path, aggregate["metrics"])
print(json.dumps({"status": "OK", "records": len(records), "metrics": len(aggregate["metrics"])}, sort_keys=True))
PY

if [[ "$generate_plots" == true ]]; then
  log "Rendering plots"
  "$python_bin" -m anchor_exp.cli.plot_results "$raw_jsonl" --output-dir "$figure_dir" --formats "$plot_formats" | tee "$log_dir/05-plots.log"
fi

log "Writing run manifest"
"$python_bin" - "$source_experiments" "$effective_experiments" "$effective_machine" \
  "$raw_jsonl" "$raw_csv" "$aggregate_json" "$metrics_csv" "$run_manifest" \
  "$script_path" "$tasks" <<'PY'
import datetime as dt, json, pathlib, sys
from anchor_exp.stable_hash import hash_file
source_experiments, experiments, machine, raw, raw_csv, aggregate, metrics, output = map(
    pathlib.Path, sys.argv[1:9]
)
runner_script = pathlib.Path(sys.argv[9])
machine_value = json.loads(machine.read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in raw.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
physical_execution_ids = {
    str(record["physical_execution_id"])
    for record in records
}
value = {
    "schema_version": "anchor-mid-run-manifest-v1",
    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "data_seed": 0,
    "sample_seed_id": 0,
    "process_repeat_id": 0,
    "timeout_seconds": int(machine_value["timeout_seconds"]),
    "setup_timeout_seconds": int(machine_value["setup_timeout_seconds"]),
    "process_timeout_seconds": int(machine_value["process_timeout_seconds"]),
    "memory_cap_bytes": int(machine_value["memory_cap_bytes"]),
    "memory_cap_requested_bytes": int(machine_value["memory_cap_requested_bytes"]),
    "memory_limit_detected_bytes": int(machine_value["memory_limit_detected_bytes"]),
    "logical_run_records": len(records),
    "physical_benchmark_processes": len(physical_execution_ids),
    "reused_logical_records": sum(
        bool(record.get("physical_execution_reused")) for record in records
    ),
    "tasks": [item.strip() for item in sys.argv[10].split(",") if item.strip()],
    "artifacts": {
        name: {"path": str(path.resolve()), "sha256": hash_file(path)}
        for name, path in (
            ("source_experiments", source_experiments),
            ("effective_experiments", experiments),
            ("runner_script", runner_script),
            ("machine", machine), ("results_jsonl", raw),
            ("results_csv", raw_csv), ("aggregate", aggregate), ("metrics_csv", metrics),
        )
    },
}
output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output.resolve())
PY

log "All twelve mid experiment families completed"
printf '  raw JSONL: %s\n  raw CSV:   %s\n  logs:      %s\n  manifests: %s\n' "$raw_jsonl" "$raw_csv" "$log_dir" "$manifest_dir"
