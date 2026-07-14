#!/usr/bin/env bash
# Execute all twelve Setup.md sweep families with one time and one memory run.

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
experiments="${SOURCE_EXPERIMENTS:-$repo_root/configs/experiments.json}"
machine="${MACHINE_MANIFEST:-$repo_root/configs/machine.json}"
data_root="${DATA_ROOT:-$repo_root/data}"
results_root="${RESULTS_ROOT:-${RESULTS_DIR:-$repo_root/results/lite}}"
tasks="${TASKS:-oneshot}"
plot_formats="${PLOT_FORMATS:-png,pdf}"
dry_run=false
reset=false
overwrite_synthetic=false
generate_plots=false

usage() {
  cat <<'EOF'
Usage: ./run_all_lite.sh [options]

Options:
  --data-root PATH          Dataset root (default: ./data)
  --results-root PATH       Results root (default: ./results/lite)
  --machine PATH            Validated anchor-machine-v1 manifest
  --experiments PATH        Experiment configuration
  --tasks LIST              Comma-separated tasks (default: oneshot)
  --plots                    Render plots after aggregation
  --plot-formats LIST        Comma-separated formats (default: png,pdf)
  --overwrite-synthetic      Regenerate frozen Alacarte workloads
  --reset                    Remove previous lite result artifacts
  --dry-run                  Validate and print the 12-family/55-case plan only
  -h, --help                 Show this help

The default runs only the main one-shot time and memory paths. Use
--tasks oneshot,count-only,prepared-query to request auxiliary experiments.
Missing real workloads are prepared with scripts/data/prepare_real_data.sh --all.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) [[ $# -ge 2 ]] || { echo "--data-root requires a path" >&2; exit 2; }; data_root="$2"; shift 2 ;;
    --results-root) [[ $# -ge 2 ]] || { echo "--results-root requires a path" >&2; exit 2; }; results_root="$2"; shift 2 ;;
    --machine) [[ $# -ge 2 ]] || { echo "--machine requires a path" >&2; exit 2; }; machine="$2"; shift 2 ;;
    --experiments) [[ $# -ge 2 ]] || { echo "--experiments requires a path" >&2; exit 2; }; experiments="$2"; shift 2 ;;
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

experiments="$(resolve_path "$experiments")"
machine="$(resolve_path "$machine")"
data_root="$(resolve_path "$data_root")"
results_root="$(resolve_path "$results_root")"
[[ -f "$experiments" ]] || fatal "experiment config not found: $experiments"

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
protocol_limits="$("$python_bin" - <<'PY'
from anchor_exp.protocol import OFFICIAL_MEMORY_CAP_BYTES, OFFICIAL_TIMEOUT_SECONDS
print(f"{OFFICIAL_TIMEOUT_SECONDS}:{OFFICIAL_MEMORY_CAP_BYTES}")
PY
)"
readonly TIMEOUT_SECONDS="${protocol_limits%%:*}"
readonly MEMORY_CAP_BYTES="${protocol_limits#*:}"

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
effective_machine="$manifest_dir/machine.lite.json"
aggregate_json="$aggregate_dir/results.json"
metrics_csv="$aggregate_dir/metrics.csv"
run_manifest="$manifest_dir/run-manifest.json"
on_error() {
  local status=$?
  echo "ERROR: run_all_lite.sh failed at line ${BASH_LINENO[0]:-unknown} (exit=$status)." >&2
  echo "Partial benchmark records, when present, remain at $raw_jsonl.partial and are resumable." >&2
  exit "$status"
}
trap on_error ERR

log "Validating the frozen lite plan"
"$python_bin" - "$experiments" "$data_root" "$tasks" <<'PY'
import collections, sys
from anchor_exp.experiments import expand_experiments, expand_runs, load_config
config = load_config(sys.argv[1])
if config.get("algorithms") != ["AC", "AS", "SweepRT", "LiftedRT"]:
    raise SystemExit("algorithm order must be AC, AS, SweepRT, LiftedRT")
if config.get("data_seeds") != [0] or config.get("process_repeat_ids") != [0]:
    raise SystemExit("lite config must freeze data_seed=0 and process_repeat_id=0")
tasks = tuple(item.strip() for item in sys.argv[3].split(",") if item.strip())
unknown = set(tasks) - {"oneshot", "count-only", "prepared-query"}
if unknown or not tasks:
    raise SystemExit(f"invalid task list: {sorted(unknown)}")
cases = expand_experiments(config, data_root=sys.argv[2], require_workloads=False)
runs = expand_runs(cases, config, tasks=tasks)
if len(cases) != 55 or len({case.experiment_id for case in cases}) != 12:
    raise SystemExit(f"expected 12 sweeps and 55 cases, got {len(cases)} cases")
by_dataset = collections.Counter(case.dataset_id for case in cases)
synthetic_paths = {
    case.workload_path for case in cases if case.dataset_type == "synthetic"
}
if len(synthetic_paths) != 25:
    raise SystemExit(f"expected 25 unique Alacarte workloads, got {len(synthetic_paths)}")
print(
    f"sweeps=12 cases={len(cases)} synthetic_workloads=25 benchmark_processes={len(runs)} tasks={','.join(tasks)}"
)
for dataset in ("Alacarte", "CMAB-1M", "GeoLife-3D-1M", "GeoLife-4D-1M", "COCO-1M"):
    print(f"  {dataset}: cases={by_dataset[dataset]}")
PY

if [[ "$dry_run" == true ]]; then
  log "Dry-run complete; no datasets, machine state, or benchmarks were touched"
  exit 0
fi

mkdir -p "$raw_dir" "$log_dir" "$manifest_dir" "$aggregate_dir"

[[ -r /proc/self/status ]] || fatal "/proc/self/status is unavailable"
grep -q '^VmRSS:' /proc/self/status || fatal "/proc/self/status has no VmRSS field"
[[ -f "$machine" ]] || fatal "validated machine manifest not found: $machine"

if [[ "$reset" == true ]]; then
  log "Removing previous lite result artifacts"
  rm -f -- "$raw_jsonl" "$raw_jsonl.partial" "$raw_csv" "$aggregate_json" "$metrics_csv" "$run_manifest" "$effective_machine"
  rm -rf -- "$figure_dir" "$log_dir"
  mkdir -p "$log_dir"
fi

log "Checking final real-workload checksums"
if ! "$repo_root/scripts/data/prepare_real_data.sh" \
  --all --data-root "$data_root" --verify-only
then
  log "Real workloads are missing; importing the pinned Hugging Face datasets"
  "$repo_root/scripts/data/prepare_real_data.sh" --all --data-root "$data_root" | tee "$log_dir/00-prepare-real-data.log"
fi

log "Preparing all 25 unique Alacarte workloads with data_seed=0"
synthetic_args=(-m anchor_exp.cli.prepare_synthetic_suite --experiments "$experiments" --data-root "$data_root")
[[ "$overwrite_synthetic" == true ]] && synthetic_args+=(--overwrite)
"$python_bin" "${synthetic_args[@]}" | tee "$log_dir/01-prepare-alacarte.log"

log "Verifying every canonical workload and manifest"
"$python_bin" - "$experiments" "$data_root" <<'PY'
import sys
from anchor_exp.experiments import expand_experiments, load_config
cases = expand_experiments(load_config(sys.argv[1]), data_root=sys.argv[2], require_workloads=True)
if len(cases) != 55:
    raise SystemExit(f"expected 55 verified cases, got {len(cases)}")
print(f"verified_cases={len(cases)}")
PY

log "Creating the effective 1800-second, 950-GiB procfs machine manifest"
"$python_bin" - "$machine" "$effective_machine" "$TIMEOUT_SECONDS" "$MEMORY_CAP_BYTES" <<'PY'
import json, pathlib, sys
from anchor_exp.experiments import validate_machine
from anchor_exp.stable_hash import canonical_json_bytes, stable_hash
source, destination = map(pathlib.Path, sys.argv[1:3])
timeout = int(sys.argv[3])
memory_cap = int(sys.argv[4])
machine = json.loads(source.read_text(encoding="utf-8"))
machine["timeout_seconds"] = timeout
machine["setup_timeout_seconds"] = timeout
machine["memory_cap_bytes"] = memory_cap
identity = {key: value for key, value in machine.items() if key not in {"machine_id", "protocol_violations"}}
machine["machine_id"] = stable_hash("machine-manifest-v1", identity)[:16].hex()
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(canonical_json_bytes(machine) + b"\n")
validate_machine(machine)
print(
    f"machine_id={machine['machine_id']} memory_cap_bytes={memory_cap} "
    f"memory_backend={machine['memory_measurement_backend']} "
    f"poll_ms={machine['memory_poll_interval_ms']}"
)
PY

log "Running all datasets serially in Setup order (AC, AS, SweepRT, LiftedRT)"
"$python_bin" -m anchor_exp.cli.run_suite \
  --experiments "$experiments" \
  --machine "$effective_machine" \
  --data-root "$data_root" \
  --output "$raw_jsonl" \
  --tasks "$tasks" | tee "$log_dir/02-benchmarks.log"

log "Writing direct-value aggregate and CSV artifacts"
"$python_bin" - "$raw_jsonl" "$raw_csv" "$aggregate_json" "$metrics_csv" <<'PY' | tee "$log_dir/03-aggregate.log"
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
  "$python_bin" -m anchor_exp.cli.plot_results "$raw_jsonl" --output-dir "$figure_dir" --formats "$plot_formats" | tee "$log_dir/04-plots.log"
fi

log "Writing run manifest"
"$python_bin" - "$experiments" "$effective_machine" "$raw_jsonl" "$raw_csv" "$aggregate_json" "$metrics_csv" "$run_manifest" "$tasks" "$TIMEOUT_SECONDS" "$MEMORY_CAP_BYTES" <<'PY'
import datetime as dt, json, pathlib, sys
from anchor_exp.stable_hash import hash_file
experiments, machine, raw, raw_csv, aggregate, metrics, output = map(pathlib.Path, sys.argv[1:8])
timeout = int(sys.argv[9])
memory_cap = int(sys.argv[10])
value = {
    "schema_version": "anchor-lite-run-manifest-v2",
    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "data_seed": 0,
    "process_repeat_id": 0,
    "memory_cap_bytes": memory_cap,
    "timeout_seconds": timeout,
    "setup_timeout_seconds": timeout,
    "tasks": [item.strip() for item in sys.argv[8].split(",") if item.strip()],
    "artifacts": {
        name: {"path": str(path.resolve()), "sha256": hash_file(path)}
        for name, path in (
            ("experiments", experiments), ("machine", machine), ("results_jsonl", raw),
            ("results_csv", raw_csv), ("aggregate", aggregate), ("metrics_csv", metrics),
        )
    },
}
output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output.resolve())
PY

log "All twelve lite experiment families completed"
printf '  raw JSONL: %s\n  raw CSV:   %s\n  logs:      %s\n  manifests: %s\n' "$raw_jsonl" "$raw_csv" "$log_dir" "$manifest_dir"
