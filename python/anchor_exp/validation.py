"""Correctness oracles kept strictly outside performance measurement."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .stable_hash import stable_hash
from .workload import Workload


PAIR_DTYPE = np.dtype([("r_id", "<u8"), ("s_id", "<u8")])


def materialize_join(workload: Workload, *, max_cross_product: int = 100_000_000) -> np.ndarray:
    """Materialize the exact join for a validation-sized workload."""

    cross = workload.n_r * workload.n_s
    if cross > max_cross_product:
        raise ValueError(
            f"validation cross product {cross} exceeds cap {max_cross_product}; "
            "do not materialize a performance join"
        )
    chunks: list[np.ndarray] = []
    for r_index in range(workload.n_r):
        overlap = np.all(
            (workload.s_lower < workload.r_upper[r_index])
            & (workload.r_lower[r_index] < workload.s_upper),
            axis=1,
        )
        s_indices = np.flatnonzero(overlap)
        if s_indices.size:
            chunk = np.empty(s_indices.size, dtype=PAIR_DTYPE)
            chunk["r_id"] = workload.r_ids[r_index]
            chunk["s_id"] = workload.s_ids[s_indices]
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=PAIR_DTYPE)


def validate_pairs(workload: Workload, pairs: np.ndarray) -> None:
    """Check side identity and strict half-open membership for every pair."""

    values = np.asarray(pairs, dtype=PAIR_DTYPE)
    r_lookup = {int(identifier): index for index, identifier in enumerate(workload.r_ids)}
    s_lookup = {int(identifier): index for index, identifier in enumerate(workload.s_ids)}
    for output_index, pair in enumerate(values):
        rid = int(pair["r_id"])
        sid = int(pair["s_id"])
        if rid not in r_lookup or sid not in s_lookup:
            raise AssertionError(f"sample {output_index} uses an id from the wrong side")
        ri, si = r_lookup[rid], s_lookup[sid]
        if not np.all(
            (workload.r_lower[ri] < workload.s_upper[si])
            & (workload.s_lower[si] < workload.r_upper[ri])
        ):
            raise AssertionError(f"sample {output_index} is not a strict half-open intersection")


def _pair_key(pair: Any) -> tuple[int, int]:
    if isinstance(pair, np.void) and pair.dtype.fields:
        return int(pair["r_id"]), int(pair["s_id"])
    return int(pair[0]), int(pair[1])


@dataclasses.dataclass(frozen=True)
class DistributionReport:
    samples: int
    join_size: int
    pearson_statistic: float
    pearson_pvalue: float
    transition_statistic: float | None
    transition_pvalue: float | None
    lag1_bucket_autocorrelation: float | None
    buckets: int


def distribution_report(
    join: np.ndarray,
    samples: np.ndarray,
    *,
    buckets: int = 128,
) -> DistributionReport:
    """Run the specified pair-frequency and lag-1 bucket diagnostics."""

    if not 2 <= len(join) <= 2000:
        raise ValueError("uniformity validation requires 2 <= W <= 2000")
    if buckets < 2:
        raise ValueError("at least two buckets are required")
    scipy_stats = __import__("scipy.stats", fromlist=["stats"])
    labels = {_pair_key(pair): index for index, pair in enumerate(join)}
    counts = np.zeros(len(join), dtype=np.int64)
    bucket_sequence = np.empty(len(samples), dtype=np.int64)
    for index, pair in enumerate(samples):
        key = _pair_key(pair)
        if key not in labels:
            raise AssertionError(f"sample pair {key!r} is outside the materialized join")
        counts[labels[key]] += 1
        bucket_sequence[index] = int.from_bytes(
            stable_hash("validation-pair-bucket", key)[:8], "big"
        ) % buckets
    expected = np.full(len(join), len(samples) / len(join), dtype=np.float64)
    pearson = scipy_stats.chisquare(counts, expected)

    transition_statistic: float | None = None
    transition_pvalue: float | None = None
    autocorrelation: float | None = None
    if len(samples) >= 2:
        transitions = np.zeros((buckets, buckets), dtype=np.int64)
        np.add.at(transitions, (bucket_sequence[:-1], bucket_sequence[1:]), 1)
        nonempty_rows = transitions.sum(axis=1) > 0
        nonempty_cols = transitions.sum(axis=0) > 0
        reduced = transitions[np.ix_(nonempty_rows, nonempty_cols)]
        if reduced.shape[0] >= 2 and reduced.shape[1] >= 2:
            test = scipy_stats.chi2_contingency(reduced, correction=False)
            transition_statistic = float(test.statistic)
            transition_pvalue = float(test.pvalue)
        x = bucket_sequence[:-1].astype(np.float64)
        y = bucket_sequence[1:].astype(np.float64)
        if np.std(x) > 0 and np.std(y) > 0:
            autocorrelation = float(np.corrcoef(x, y)[0, 1])
    return DistributionReport(
        samples=len(samples),
        join_size=len(join),
        pearson_statistic=float(pearson.statistic),
        pearson_pvalue=float(pearson.pvalue),
        transition_statistic=transition_statistic,
        transition_pvalue=transition_pvalue,
        lag1_bucket_autocorrelation=autocorrelation,
        buckets=buckets,
    )


def holm_rejections(pvalues: Mapping[str, float], *, family_level: float = 0.01) -> set[str]:
    """Return hypotheses rejected by the Holm step-down procedure."""

    if not 0 < family_level < 1:
        raise ValueError("family_level must lie in (0,1)")
    ranked = sorted(pvalues.items(), key=lambda item: (float(item[1]), item[0]))
    rejected: set[str] = set()
    total = len(ranked)
    for rank, (label, pvalue) in enumerate(ranked):
        if not 0 <= float(pvalue) <= 1:
            raise ValueError(f"invalid p-value for {label!r}")
        if pvalue <= family_level / (total - rank):
            rejected.add(label)
        else:
            break
    return rejected

