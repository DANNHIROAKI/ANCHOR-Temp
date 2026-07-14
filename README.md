# ANCHOR exact intersection-join sampling experiments

This repository implements the complete experimental pipeline for exact,
with-replacement i.i.d. sampling from a bichromatic axis-aligned box
intersection join. It contains the two proposed ANCHOR algorithms—Compiled
(`AC`) and Streaming (`AS`)—and the `SweepRT` and `LiftedRT` range-tree
baselines, together with on-demand Alacarte generation, finished-data import,
process-isolated measurement, validation, aggregation, and plotting.

The mathematical sources are preserved in [`docs/spec`](docs/spec). All four
algorithms use half-open boxes and strict overlap; duplicate geometries retain
distinct identities. Counts and sampling weights are exact integers, and every
successful positive-length query returns an ordered i.i.d. uniform sample from
the full join without materializing it.

## Repository map

```text
cpp/                 C++20 algorithms, benchmark executable, and unit tests
python/anchor_exp/   generators, HF importers, suite runner, analysis tools
configs/             frozen Alacarte, real-data, experiment, and machine inputs
scripts/             procfs monitor and unified data-preparation entry points
docs/                formats, implementation decisions, and research specs
data/manifests/      versioned manifest location (raw/workload data are ignored)
results/             append-only raw records and derived figures (ignored)
```

## Build

Requirements for the measured executable are a C++20 compiler and CMake 3.20+
(GCC 12+ or Clang 15+ are recommended):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

A dependency-free GNU Make fallback is also provided:

```bash
make release
make test
```

Install the Python package and light test dependencies in an isolated
environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,plots]'
```

Real-data import uses only `pyarrow` and `requests` beyond the base package. It
does not require GIS software, Detectron2, PyTorch, images, or model weights.

## Quick correctness smoke test

Generate a small certified Alacarte workload, inspect it, then exercise all
algorithms:

```bash
anchor-generate-synthetic \
  --config configs/alacarte.default.json \
  --n-r 1000 --n-s 1000 --dimension 2 --alpha 2 \
  --seed 0 --output data/workloads/smoke.bin

anchor-inspect-workload data/workloads/smoke.bin --verify-file

for algorithm in ac as sweeprt liftedrt; do
  build/anchor_bench run \
    --workload data/workloads/smoke.bin \
    --algorithm "$algorithm" --samples 10000 \
    --seed-hex 0000000000000000000000000000000000000000000000000000000000000001 \
    --measurement-mode time --task oneshot --timeout-seconds 30 \
    --setup-timeout-seconds 30
done
```

`anchor_bench` writes exactly one JSON record to standard output. Input loading,
pre-touch, checksums, and post-run membership validation occur outside the
one-shot timing interval.

## Dataset preparation from an empty checkout

The repository intentionally ships without experiment datasets. Synthetic
workloads are generated on demand by the Alacarte implementation from the
parameters in `configs/experiments.json`; the 55 experiment cases collapse to
25 distinct synthetic workload files, so a fixed workload is reused across
algorithms and across each `t` sweep.

Real workloads are imported only from three finished Hugging Face datasets:

- `DannHiroaki/CMAB-Spatial-Join-0.08B`
- `DannHiroaki/Geolife-Spatial-Join-0.15B`
- `DannHiroaki/COCO-Spatial-Join-1.23B`

The published CMAB `building_uid` column has 4,898 collisions in Guangdong, so
the importer preserves every row and derives collision-free object ids from the
unique `(source_file, source_fid)` key; this mapping is frozen in each manifest.

`data_sources.lock.json` pins an immutable repository revision and the exact
published Parquet/JSON assets used from each dataset. No path in this repository
rebuilds CMAB from GIS files, GeoLife from `.plt` trajectories, or COCO by
running Detectron2. Install the light importer dependencies and run:

```bash
python -m pip install -r requirements-preprocess.txt
./scripts/data/prepare_real_data.sh --all --data-root ./data
```

The same preparation happens automatically at the start of
`run_all_lite.sh` when any real workload is absent or invalid. The importer
checks dependencies and free space first, uses resumable `*.partial` downloads,
validates the pinned Hub revision and every locked asset, and publishes a
collection only after rereading and validating all outputs. COCO proposal
shards are accessed by Parquet row group over HTTP ranges, so the importer does
not download the full 1.23B-row dataset.

Useful non-mutating modes are:

```bash
./scripts/data/prepare_real_data.sh --all --data-root ./data --check-only
./scripts/data/prepare_real_data.sh --all --data-root ./data --verify-only
```

The final layout is:

```text
data/
  sources/huggingface/<repo-slug>/<revision>/
  workloads/cmab_1m/
  workloads/geolife_3d_1m/
  workloads/geolife_4d_1m/
  workloads/coco_1m/
  manifests/
```

Each final collection contains canonical uncompressed workloads, adjacent
per-workload manifests, a collection `manifest.json`, and `checksums.sha256`.
The importer verifies cardinality, dimension, endpoint validity, object IDs,
cross-level identity, source provenance, payload hashes, and file hashes. The
format is documented in
[`docs/WORKLOAD_FORMAT.md`](docs/WORKLOAD_FORMAT.md).

The commands fail with `DATASET-CONSTRUCTION-FAILED` rather than substituting a
toy, approximate, stale, or partially downloaded artifact. This repository
records public acquisition metadata but does not redistribute source data.

## The twelve experiment families

`configs/experiments.json` encodes the full one-factor-at-a-time matrix:

| Dataset | Experiments |
|---|---|
| Alacarte | N, t, target-alpha, shape, and dimension sweeps |
| CMAB-1M | level and t sweeps |
| GeoLife-3D-1M | level and t sweeps |
| GeoLife-4D-1M | level and t sweeps |
| COCO-1M | t sweep |

Inspect expansion without running anything:

```bash
anchor-prepare-synthetic-suite \
  --experiments configs/experiments.json --dry-run

anchor-run-suite --experiments configs/experiments.json \
  --machine configs/machine.example.json --output results/raw.jsonl --dry-run
```

Before a publication run, validate the exact executable on one or more
materializable workloads and bind that successful report into the machine
manifest:

```bash
anchor-generate-synthetic --config configs/alacarte.default.json \
  --n-r 16 --n-s 16 --dimension 2 --alpha 1 \
  --epsilon-alpha 0.5 --delta 0.1 --seed 7 \
  --output data/workloads/validation.bin

anchor-validate --benchmark build/anchor_bench \
  --workload data/workloads/validation.bin \
  --output results/validation.json

anchor-capture-machine --benchmark build/anchor_bench \
  --validation-report results/validation.json \
  --memory-cap-bytes 1020054732800 --cpu-core 0 --monitor-cpu-core 2 \
  --memory-poll-interval-ms 5 --numa-node 0 \
  --code-commit COMMIT --memory-configuration 'FROZEN_DMI_DESCRIPTION' \
  --build-flags=-O3 --build-flags=-DNDEBUG \
  --smt-sibling-idle-confirmed --output configs/machine.json
```

The validation command enforces the Setup sample-count rule, invokes all four
C++ algorithms, checks their exact counts and every sampled pair, then applies
one family-wide Holm correction to pair-frequency and lag-1 transition tests.
Its `--sample-count` override produces a clearly labelled developer smoke
report that cannot unlock a publication run.

Each logical main run is two fresh processes: a time run and a memory run with
the same workload and sampling seed. The external memory monitor reads
`/proc/<pid>/status` `VmRSS` at fixed 5 ms absolute intervals and captures the
input, baseline, post-prepare, and sampled peak values through an acknowledged
event protocol. No delegated cgroup or privileged write is required. Both the
main and setup safety timeouts are 1800 seconds (30 minutes). The sampled RSS
cap is 950 GiB. The suite defaults to `oneshot`;
`count-only` and `prepared-query` run only when explicitly requested.

Run the complete single-run matrix (12 sweeps, 55 configurations, four
algorithms, time plus memory) with:

```bash
./run_all_lite.sh --data-root ./data --results-root ./results/lite
```

The script verifies or prepares real data, generates all frozen Alacarte inputs,
runs datasets serially in Setup order, and writes
`raw/results.jsonl`, `raw/results.csv`, logs, and manifests.

## Results

Execution records are appended to `OUTPUT.partial`. After all requested runs
are present, the harness checks identities and cross-run consistency, writes a
reconciled JSONL atomically to `OUTPUT`, and removes the staging file. Aggregate
and render that finalized raw file with:

```bash
anchor-aggregate results/raw.jsonl \
  --output results/aggregate.json --metrics-csv results/metrics.csv
anchor-plot results/raw.jsonl --output-dir results/figures
```

Aggregation preserves `OOM`, `MEMORY-CAP-EXCEEDED`, and `TO` boundaries and
exposes the one successful raw value for each frozen configuration. Speedups are
direct paired ratios on the same workload and `t`; no medians, min/max bands, or bootstrap intervals
are synthesized from a single run.

## Reproducibility and scalability

- A run seed excludes `process_repeat_id` and, for a fixed workload t-sweep,
  excludes `t`. Domain-tagged substreams isolate outer selection, quotas,
  offsets, and shuffles.
- Alacarte calibration uses common random numbers; every fixed candidate is
  evaluated with outward geometry intervals and exact binary64 mean reduction.
  Certification uses an independent Philox domain, a rigorous sample-variance
  upper bound, and an upward-rounded empirical-Bernstein radius before the
  final, disjoint R/S generation streams are opened.
- `LiftedRT` and high-dimensional `SweepRT` intentionally retain the direct
  node-based range-tree space costs stated in the paper. They may hit the fixed
  resource limit at million scale. Such failures are experimental outcomes—
  there is no hidden fallback to a scan or approximate index.
- Machine, workload, solver, algorithm, source, and build identities are all
  checksum-addressed. See
  [`docs/IMPLEMENTATION_DECISIONS.md`](docs/IMPLEMENTATION_DECISIONS.md).

## Validation

The test suite covers exact bounded integers and alias mass conservation,
Fenwick select, static/dynamic node-range-tree count and active-rank sampling,
strict range boundaries, SweepRT event ownership, ANCHOR local
ownership and terminal-array lifetime, endpoint-touching boxes, duplicate
geometry identities, empty/full/singleton joins, exhaustive small cross-checks,
and deterministic Alacarte/preprocessing behavior.

Full statistical goodness-of-fit validation is deliberately separate from
performance measurement. Run `anchor-validate` as shown above; a machine
manifest tied to the exact executable is accepted only when that report has
status `OK`. Unit tests remain fast and deterministic, while the validation
gate performs the required million-to-ten-million-sample statistical protocol.
