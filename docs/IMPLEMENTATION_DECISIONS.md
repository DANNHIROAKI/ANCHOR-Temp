# Frozen implementation decisions

The research specifications define the mathematics and measurement boundaries
but intentionally leave several software choices open. This file freezes those
choices so results from this repository are interpretable and reproducible.

1. **Language and execution.** All four measured algorithms are C++20 and are
   linked into one `anchor_bench` executable. Dataset import/generation, suite
   orchestration, validation, aggregation, and plotting are Python. Measured
   algorithm code is single-threaded.
2. **No baseline substitution.** `LiftedRT` and `SweepRT` use standard
   multi-level range trees with canonical terminal blocks. A high-dimensional
   structure that exceeds the configured memory or time limit is reported as
   `OOM` or `TO`; it is never replaced by a scan, kd-tree, approximate index, or
   lower-dimensional special case.
3. **Coordinates.** Frozen workloads contain IEEE-754 binary64 or signed int64
   endpoints. Non-finite floating values and empty boxes are rejected. Integer
   lifting uses ascending `U` coordinates plus strict suffix queries, avoiding
   `-INT64_MIN`.
4. **Wide integers.** Join counts, weights, prefix sums, quota totals, and alias
   scaled masses use checked unsigned 128-bit arithmetic. Values written to a
   narrower type are range-checked.
5. **Randomness.** A 256-bit run seed is deterministically split by explicit
   domain tags. Uniform bounded integers use rejection sampling, weighted
   choices use the integer Vose construction, and no modulo-biased choice is
   permitted. The executable records its PRNG and seed-to-state version.
6. **Input serialization.** Workloads use the uncompressed v1 format documented
   in `WORKLOAD_FORMAT.md`. Object IDs are uint64 values derived from full
   stable keys; preprocessors reject a truncation collision.
7. **Stable hashing.** `anchor-canonical-json-v1` is sorted, whitespace-free,
   NFC UTF-8 JSON with finite numbers and normalized negative zero. SHA-256 is
   used in full for provenance and in the prefixes prescribed by the setup.
8. **Result serialization.** Each process emits one JSON object. Suite output is
   append-only JSON Lines with schema/version fields; aggregation never edits
   raw records in place.
9. **External datasets.** Real workloads are derived only from the finished
   Parquet/JSON artifacts published in the three pinned Hugging Face dataset
   repositories. The implementation does not download raw GIS, `.plt`, COCO
   images, or model weights and does not rerun the upstream builders. Missing
   dependencies or incomplete frozen metadata cause an explicit construction
   failure; there is no toy or approximate fallback under a real-dataset
   identifier.
10. **Alacarte numeric protocol and budgets.** Values in
    `configs/alacarte.default.json` are repository defaults, not unstated claims
    of the paper. The implementation uses the concrete-distribution option in
    Alacarte section 11.6: `numpy.exp` first freezes every latent relative
    length as binary64. The P1D kernel and dimensional products then use
    outward `nextafter` interval operations under IEEE-754 binary64
    round-to-nearest; calibration and certification means use an exact
    binary64 superaccumulator; the sample-variance upper bound is the exact
    inequality `n/(n-1) * mean * (1-mean)` maximized over the mean interval;
    and the empirical-Bernstein radius is evaluated upward at 80 decimal
    digits. Thus no unverified libm error claim is made for the real-valued
    exponential transform. Every workload manifest records this numeric
    protocol, the entire solver configuration, and the certification result.
    The suite tolerance is 20% of target density with an absolute floor of
    `0.05`, capped at `5.0` and at 10% of the upper-domain margin. These values
    are explicit because the research setup does not freeze them; all 25 suite
    workloads must obtain a strict certificate within 15 checkpoints.
11. **Memory measurement.** An external process polls the benchmark process'
    `/proc/<pid>/status` `VmRSS` every 5 ms on absolute monotonic deadlines.
    A two-pipe acknowledged event protocol freezes `InputMemory`,
    `BaselineMemory`, `MemoryAfterPrepare`, and the final sample while their
    allocations are live. `PeakMemoryTotal` is the largest sampled RSS;
    incremental and auxiliary peaks are derived from the frozen baselines.
    The cap is enforced against sampled RSS for the whole process group. No
    cgroup delegation or privileged filesystem write is required.
12. **Auxiliary experiments.** `count-only` and `prepared-query` are separate
    process tasks. They never populate the main one-shot timing field and are
    skipped by the default lite/main invocation.
13. **Single-run aggregation.** Every configuration uses `data_seed=0` and
    `process_repeat_id=0`. Successful raw values and direct paired speedups are
    reported without medians, min/max summaries, or bootstrap intervals.
14. **Real-data acquisition.** `data_sources.lock.json` freezes each Hub repo,
    immutable revision, asset path, byte size, and SHA-256. Static downloads
    resume through partial files. COCO uses pinned Parquet metadata and HTTP
    range reads to fetch only row groups belonging to selected images. Verified
    sources and workload collections are published by atomic rename only after
    their full validation gates pass.
    CMAB preserves every published row and uses the lexicographic rank of the
    unique `(source_file, source_fid)` pair as its uint64 object id. The
    published `building_uid` has 4,898 two-row collisions in Guangdong and is
    retained only as provenance; its collision diagnostics are admission data.

Any change to one of these choices must update the relevant version identifier
and produces a new algorithm, workload, or machine configuration checksum.
