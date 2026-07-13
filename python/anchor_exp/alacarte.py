"""Reproducible Alacarte synthetic box generation.

The implementation follows ``docs/spec/Alacarte.md``.  Object positions are
integrated out while solving for the nominal coverage parameter, so a solver
evaluation costs ``O(M d)`` rather than enumerating the Cartesian product.
The returned certificate concerns the expected output density of the random
generator; a realized workload is deliberately *not* rejection-sampled to an
exact join cardinality.  Numerically, the solver follows the concrete
binary64-induced distribution option in section 11.6: the scale transform is
frozen at its binary64 output, after which geometry, reductions, variance
bounds, and confidence radii are evaluated with deterministic enclosures.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from .stable_hash import stable_hash, stable_hash_hex


ALACARTE_MODEL_VERSION = "alacarte-binary64-scale-v2"
PHILOX_STREAM_VERSION = "alacarte-philox-domains-v1"
_LOG_TWO = math.log(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_FLOAT = np.finfo(np.float64)
_MIN_SUBNORMAL = float(np.nextafter(0.0, 1.0))
_MANTISSA_MASK = np.uint64((1 << 52) - 1)
_SIGNIFICAND_LOW_MASK = np.uint64((1 << 26) - 1)
_BINARY64_DENOMINATOR = 1 << 1074
VolumeFamily = Literal["fixed", "exponential", "lognormal", "normal"]
Side = Literal["R", "S"]


class CoverageStatus(str, enum.Enum):
    CERTIFIED = "CERTIFIED"
    INVALID_INPUT = "INVALID_INPUT"
    TARGET_BELOW_NUMERIC_RESOLUTION = "TARGET_BELOW_NUMERIC_RESOLUTION"
    SAMPLE_BUDGET_EXCEEDED = "SAMPLE_BUDGET_EXCEEDED"
    NONFINITE_EVALUATION = "NONFINITE_EVALUATION"
    NUMERICAL_RESOLUTION_EXHAUSTED = "NUMERICAL_RESOLUTION_EXHAUSTED"
    BRACKET_NOT_FOUND = "BRACKET_NOT_FOUND"
    EXPANSION_LIMIT_REACHED = "EXPANSION_LIMIT_REACHED"
    BISECTION_LIMIT_REACHED = "BISECTION_LIMIT_REACHED"
    CERTIFICATION_NOT_REACHED = "CERTIFICATION_NOT_REACHED"


class NumericDegeneracyError(RuntimeError):
    """A positive real-model box cannot be represented as binary64 endpoints."""


def _fraction_to_float_bounds(value: Fraction) -> tuple[float, float]:
    """Return adjacent binary64 bounds containing an exact rational value."""

    nearest = float(value)
    if not math.isfinite(nearest):
        raise NumericDegeneracyError("exact rational value is outside binary64 range")
    represented = Fraction.from_float(nearest)
    lower = nearest if represented <= value else math.nextafter(nearest, -math.inf)
    upper = nearest if represented >= value else math.nextafter(nearest, math.inf)
    return lower, upper


def _fraction_upper(value: Fraction) -> float:
    return _fraction_to_float_bounds(value)[1]


def _exact_scale(config: "AlacarteConfig") -> Fraction:
    return Fraction(config.n_r * config.n_s, config.n_r + config.n_s)


def _target_probability(config: "AlacarteConfig") -> Fraction:
    return Fraction.from_float(config.alpha_target) / _exact_scale(config)


def _probability_tolerance(config: "AlacarteConfig") -> Fraction:
    return Fraction.from_float(config.epsilon_alpha) / _exact_scale(config)


def _binary64_sum_units(values: np.ndarray) -> int:
    """Exactly sum nonnegative finite binary64 values as multiples of 2^-1074.

    The two 26-bit significand limbs are accumulated by exponent with
    ``bincount``.  Every bin stays below 2^53, so these integer-valued binary64
    additions are exact.  The final exponent shifts use Python integers.
    """

    array = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise NumericDegeneracyError("exact accumulator requires finite nonnegative values")
    # The high limb is below 2^27.  This bound makes every weighted bin sum an
    # exactly representable integer.  It is far above all configured batches.
    if array.size >= (1 << 26):
        raise NumericDegeneracyError("exact binary64 accumulator batch limit exceeded")
    if array.size == 0:
        return 0

    bits = array.view(np.uint64) & np.uint64((1 << 63) - 1)
    exponent_field = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    if np.any(exponent_field == 0x7FF):
        raise NumericDegeneracyError("exact accumulator received a nonfinite value")
    significand = bits & _MANTISSA_MASK
    normal = exponent_field != 0
    significand = significand.copy()
    significand[normal] |= np.uint64(1 << 52)
    shifts = np.maximum(exponent_field - 1, 0)

    low = (significand & _SIGNIFICAND_LOW_MASK).astype(np.float64)
    high = (significand >> np.uint64(26)).astype(np.float64)
    low_bins = np.bincount(shifts, weights=low, minlength=1023)
    high_bins = np.bincount(shifts, weights=high, minlength=1023)
    occupied = np.flatnonzero((low_bins != 0.0) | (high_bins != 0.0))

    total = 0
    for shift in occupied:
        limb_sum = int(low_bins[shift]) + (int(high_bins[shift]) << 26)
        total += limb_sum << int(shift)
    return total


def _directed_down(values: np.ndarray) -> np.ndarray:
    return np.nextafter(values, -math.inf)


def _directed_up(values: np.ndarray) -> np.ndarray:
    return np.nextafter(values, math.inf)


def _exact_sum_at_least_one(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classify x+y >= 1 using an error-free TwoSum residual at ties."""

    rounded = x + y
    other = rounded - x
    residual = (x - (rounded - other)) + (y - other)
    return (rounded > 1.0) | ((rounded == 1.0) & (residual >= 0.0))


def _pair_probability_enclosures(
    config: "AlacarteConfig", theta: float, samples: "BaseSamples"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outward bounds for each conditional pair probability.

    ``np.exp`` first freezes each relative length as a binary64 number; this is
    the repository's concrete scale-transform semantics.  From those exact
    binary64 lengths onward, every operation in the closed-form P1D kernel and
    every dimensional product is enclosed by one outward ``nextafter`` step.
    Basic binary64 +, -, *, and / are required to use IEEE round-to-nearest.
    """

    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    d = config.dimension
    tau_max = math.log1p(-config.eps_geom)
    eta_r = theta - math.log(config.n_r) + samples.log_z_r
    eta_s = theta - math.log(config.n_s) + samples.log_z_s
    tau_r = np.minimum(eta_r[:, None] / d + samples.shape_r, tau_max)
    tau_s = np.minimum(eta_s[:, None] / d + samples.shape_s, tau_max)
    rho_r = np.exp(tau_r)
    rho_s = np.exp(tau_s)
    if (
        np.any(~np.isfinite(rho_r))
        or np.any(~np.isfinite(rho_s))
        or np.any(rho_r < 0.0)
        or np.any(rho_s < 0.0)
        or np.any(rho_r >= 1.0)
        or np.any(rho_s >= 1.0)
    ):
        raise NumericDegeneracyError("relative-length transform left [0,1)")

    q_lower = np.ones(samples.count, dtype=np.float64)
    q_upper = np.ones(samples.count, dtype=np.float64)
    q_point = np.ones(samples.count, dtype=np.float64)
    for dimension in range(d):
        x = rho_r[:, dimension]
        y = rho_s[:, dimension]
        certain = (x > 0.0) & (y > 0.0) & _exact_sum_at_least_one(x, y)
        positive = (x > 0.0) & (y > 0.0)
        interior = positive & ~certain
        p_lower = np.zeros(samples.count, dtype=np.float64)
        p_upper = np.zeros(samples.count, dtype=np.float64)
        p_lower[certain] = 1.0
        p_upper[certain] = 1.0

        if np.any(interior):
            xi = x[interior]
            yi = y[interior]
            sum_lower = _directed_down(xi + yi)
            sum_upper = _directed_up(xi + yi)
            gap_lower = np.maximum(0.0, _directed_down(1.0 - sum_upper))
            gap_upper = np.maximum(0.0, _directed_up(1.0 - sum_lower))
            one_minus_x_lower = _directed_down(1.0 - xi)
            one_minus_x_upper = _directed_up(1.0 - xi)
            one_minus_y_lower = _directed_down(1.0 - yi)
            one_minus_y_upper = _directed_up(1.0 - yi)

            term_x_lower = _directed_down(xi * gap_lower)
            term_x_upper = _directed_up(xi * gap_upper)
            term_y_lower = _directed_down(yi * one_minus_y_lower)
            term_y_upper = _directed_up(yi * one_minus_y_upper)
            numerator_lower = np.maximum(
                0.0, _directed_down(term_x_lower + term_y_lower)
            )
            numerator_upper = _directed_up(term_x_upper + term_y_upper)
            denominator_lower = np.maximum(
                _MIN_SUBNORMAL,
                _directed_down(one_minus_x_lower * one_minus_y_lower),
            )
            denominator_upper = _directed_up(one_minus_x_upper * one_minus_y_upper)
            p_lower[interior] = np.maximum(
                0.0, _directed_down(numerator_lower / denominator_upper)
            )
            p_upper[interior] = np.minimum(
                1.0, _directed_up(numerator_upper / denominator_lower)
            )

        midpoint = p_lower + (p_upper - p_lower) / 2.0
        midpoint = np.minimum(p_upper, np.maximum(p_lower, midpoint))
        q_lower = np.maximum(0.0, _directed_down(q_lower * p_lower))
        q_upper = np.minimum(1.0, _directed_up(q_upper * p_upper))
        q_point = np.minimum(q_upper, np.maximum(q_lower, q_point * midpoint))

    if np.any(q_lower > q_upper) or np.any(~np.isfinite(q_point)):
        raise NumericDegeneracyError("pair-probability interval became invalid")
    return q_point, q_lower, q_upper


class CoverageFailure(RuntimeError):
    def __init__(self, result: "CoverageResult") -> None:
        super().__init__(f"coverage solver returned {result.status.value}: {result.message}")
        self.result = result


@dataclasses.dataclass(frozen=True)
class SolverBudget:
    """Finite, explicit work limits for calibration and certification."""

    calibration_rounds: int = 4
    certification_checkpoints: int = 10
    calibration_initial: int = 8192
    certification_initial: int = 8192
    certification_batch: int = 65536
    train_tolerance_initial: float = 1.0e-4
    theta_tolerance: float = 1.0e-10
    theta_min: float = -700.0
    theta_max: float = 700.0
    max_expand: int = 256
    max_bisect: int = 128

    def validate(self) -> None:
        integer_fields = {
            "calibration_rounds": self.calibration_rounds,
            "certification_checkpoints": self.certification_checkpoints,
            "calibration_initial": self.calibration_initial,
            "certification_initial": self.certification_initial,
            "certification_batch": self.certification_batch,
            "max_expand": self.max_expand,
            "max_bisect": self.max_bisect,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"solver.{name} must be a positive integer")
        if self.certification_initial < 2:
            raise ValueError("solver.certification_initial must be at least 2")
        if not math.isfinite(self.train_tolerance_initial) or self.train_tolerance_initial <= 0:
            raise ValueError("solver.train_tolerance_initial must be finite and positive")
        if not math.isfinite(self.theta_tolerance) or self.theta_tolerance <= 0:
            raise ValueError("solver.theta_tolerance must be finite and positive")
        if not (
            math.isfinite(self.theta_min)
            and math.isfinite(self.theta_max)
            and self.theta_min < self.theta_max
        ):
            raise ValueError("solver theta range must be finite and nonempty")


@dataclasses.dataclass(frozen=True)
class AlacarteConfig:
    n_r: int = 100_000
    n_s: int = 100_000
    dimension: int = 2
    alpha_target: float = 10.0
    universe_lower: tuple[float, ...] | None = None
    universe_upper: tuple[float, ...] | None = None
    volume_family: VolumeFamily = "fixed"
    volume_cv: float = 0.0
    shape_sigma: float = 0.0
    eps_geom: float = 0.01
    epsilon_alpha: float = 1.0
    delta: float = 0.01
    solver: SolverBudget = dataclasses.field(default_factory=SolverBudget)

    def universe(self) -> tuple[np.ndarray, np.ndarray]:
        lower = (
            np.zeros(self.dimension, dtype=np.float64)
            if self.universe_lower is None
            else np.asarray(self.universe_lower, dtype=np.float64)
        )
        upper = (
            np.ones(self.dimension, dtype=np.float64)
            if self.universe_upper is None
            else np.asarray(self.universe_upper, dtype=np.float64)
        )
        return lower, upper

    @property
    def log_a(self) -> float:
        return float(
            math.log(self.n_r)
            + math.log(self.n_s)
            - np.logaddexp(math.log(self.n_r), math.log(self.n_s))
        )

    @property
    def a(self) -> float:
        return float(_exact_scale(self))

    @property
    def p_target(self) -> float:
        return float(_target_probability(self))

    @property
    def epsilon_p(self) -> float:
        return float(_probability_tolerance(self))

    def validate(self) -> None:
        for name, value in (("n_r", self.n_r), ("n_s", self.n_s), ("dimension", self.dimension)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_r + self.n_s > np.iinfo(np.uint64).max:
            raise ValueError("n_r+n_s does not fit the global uint64 object-id space")
        lower, upper = self.universe()
        if lower.shape != (self.dimension,) or upper.shape != (self.dimension,):
            raise ValueError("universe bounds must each have exactly dimension entries")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("universe bounds must be finite")
        if not np.all(lower < upper):
            raise ValueError("each universe lower bound must be below its upper bound")
        scale = _exact_scale(self)
        alpha = Fraction.from_float(self.alpha_target) if math.isfinite(self.alpha_target) else None
        if alpha is None or not (0 < alpha < scale):
            raise ValueError("alpha_target must lie strictly between 0 and A")
        if self.volume_family not in {"fixed", "exponential", "lognormal", "normal"}:
            raise ValueError("volume_family must be fixed, exponential, lognormal, or normal")
        if not math.isfinite(self.volume_cv) or self.volume_cv < 0:
            raise ValueError("volume_cv must be finite and nonnegative")
        if self.volume_family == "fixed" and self.volume_cv != 0.0:
            raise ValueError("fixed volume requires volume_cv=0")
        if self.volume_family == "exponential" and self.volume_cv != 1.0:
            raise ValueError("exponential volume requires volume_cv=1")
        if self.volume_family == "normal" and not (0.0 <= self.volume_cv < 1.0):
            raise ValueError("zero-left-truncated normal volume requires 0<=volume_cv<1")
        if not math.isfinite(self.shape_sigma) or self.shape_sigma < 0:
            raise ValueError("shape_sigma must be finite and nonnegative")
        if not math.isfinite(self.eps_geom) or not (0.0 < self.eps_geom < 0.5):
            raise ValueError("eps_geom must lie strictly between 0 and 1/2")
        if 1.0 - self.eps_geom == 1.0:
            raise ValueError("eps_geom is below binary64 resolution at 1")
        epsilon = (
            Fraction.from_float(self.epsilon_alpha) if math.isfinite(self.epsilon_alpha) else None
        )
        if epsilon is None or not (0 < epsilon < min(alpha, scale - alpha)):
            raise ValueError("epsilon_alpha must stay inside both target-domain boundaries")
        if not math.isfinite(self.delta) or not (0.0 < self.delta < 1.0):
            raise ValueError("delta must lie strictly between 0 and 1")
        self.solver.validate()

    def to_dict(self) -> dict[str, Any]:
        lower, upper = self.universe()
        return {
            "schema_version": "alacarte-config-v1",
            "n_R": int(self.n_r),
            "n_S": int(self.n_s),
            "dimension": int(self.dimension),
            "alpha_target": float(self.alpha_target),
            "universe": {"lower": lower.tolist(), "upper": upper.tolist()},
            "volume": {"family": self.volume_family, "cv": float(self.volume_cv)},
            "shape_sigma": float(self.shape_sigma),
            "eps_geom": float(self.eps_geom),
            "epsilon_alpha": float(self.epsilon_alpha),
            "delta": float(self.delta),
            "solver": dataclasses.asdict(self.solver),
        }


def config_from_mapping(value: Mapping[str, Any]) -> AlacarteConfig:
    """Parse the frozen JSON config while tolerating flat programmatic fields."""

    universe = value.get("universe", {})
    volume = value.get("volume", {})
    solver_value = dict(value.get("solver", {}))
    solver = SolverBudget(**solver_value)
    return AlacarteConfig(
        n_r=int(value.get("n_R", value.get("n_r", 100_000))),
        n_s=int(value.get("n_S", value.get("n_s", 100_000))),
        dimension=int(value.get("dimension", value.get("d", 2))),
        alpha_target=float(value.get("alpha_target", 10.0)),
        universe_lower=(
            tuple(float(x) for x in universe["lower"])
            if "lower" in universe
            else value.get("universe_lower")
        ),
        universe_upper=(
            tuple(float(x) for x in universe["upper"])
            if "upper" in universe
            else value.get("universe_upper")
        ),
        volume_family=str(volume.get("family", value.get("volume_family", "fixed"))),  # type: ignore[arg-type]
        volume_cv=float(volume.get("cv", value.get("volume_cv", 0.0))),
        shape_sigma=float(value.get("shape_sigma", 0.0)),
        eps_geom=float(value.get("eps_geom", 0.01)),
        epsilon_alpha=float(value.get("epsilon_alpha", 1.0)),
        delta=float(value.get("delta", 0.01)),
        solver=solver,
    )


def _domain_words(base_seed: int | str, domain: Sequence[Any]) -> list[int]:
    digest = stable_hash(PHILOX_STREAM_VERSION, base_seed, list(domain))
    return [int.from_bytes(digest[i : i + 4], "big") for i in range(0, len(digest), 4)]


def philox_rng(base_seed: int | str, *domain: Any) -> np.random.Generator:
    """Create a deterministic, domain-separated NumPy Philox generator."""

    seed_sequence = np.random.SeedSequence(_domain_words(base_seed, domain))
    return np.random.Generator(np.random.Philox(seed_sequence))


def philox_stream_id(base_seed: int | str, *domain: Any) -> str:
    return stable_hash_hex(PHILOX_STREAM_VERSION, base_seed, list(domain))


def _open_uniform(rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
    values = rng.random(size, dtype=np.float64)
    # Generator.random is [0,1); replacing its possible zero realizes the open
    # interval required by log transforms without perturbing any other draw.
    return np.maximum(values, _MIN_SUBNORMAL)


def _lognormal_sigma_squared(cv: float) -> float:
    if cv <= 1.0:
        squared = cv * cv
        if cv > 0.0 and squared == 0.0:
            raise NumericDegeneracyError("nonzero lognormal CV squared underflowed")
        return math.log1p(squared)
    return 2.0 * math.log(cv) + math.log1p(cv ** -2)


def _truncated_standard_moments(kappa: float) -> tuple[float, float]:
    """Return mean of (kappa+G) and Var(G), G|G>-kappa.

    The far negative tail uses the inverse-Mills expansion directly, avoiding
    subtraction of two numbers of size ``-kappa``.  The remaining range uses
    log-CDF arithmetic.
    """

    if kappa < -12.0:
        x = -kappa
        y = 1.0 / x
        y2 = y * y
        mean = y * (1.0 + y2 * (-2.0 + y2 * (10.0 + y2 * (-74.0 + 706.0 * y2))))
        variance = y2 * (
            1.0 + y2 * (-6.0 + y2 * (50.0 + y2 * (-518.0 + 6354.0 * y2)))
        )
        if mean > 0.0 and variance > 0.0:
            return mean, variance
    log_lambda = -0.5 * kappa * kappa - _LOG_SQRT_2PI - float(log_ndtr(kappa))
    mills = math.exp(log_lambda)
    mean = kappa + mills
    variance = 1.0 - kappa * mills - mills * mills
    if not (math.isfinite(mean) and math.isfinite(variance) and mean > 0.0 and variance > 0.0):
        raise NumericDegeneracyError("truncated-normal moments lost numeric resolution")
    return mean, variance


@functools.lru_cache(maxsize=64)
def _normal_parameters(cv: float) -> tuple[float, float]:
    if cv == 0.0:
        return math.inf, 1.0
    if not (0.0 < cv < 1.0):
        raise ValueError("truncated-normal CV must lie strictly between zero and one")

    def coefficient(kappa: float) -> float:
        mean, variance = _truncated_standard_moments(kappa)
        return math.sqrt(variance) / mean

    lo, hi = -1.0, 1.0
    while coefficient(lo) < cv:
        lo *= 2.0
        if lo < -1.0e7:
            raise NumericDegeneracyError("normal CV is too close to one for binary64")
    while coefficient(hi) > cv:
        hi *= 2.0
        if hi > 1.0e7:
            raise NumericDegeneracyError("normal CV is too close to zero for binary64")
    for _ in range(160):
        mid = lo + (hi - lo) / 2.0
        if mid == lo or mid == hi:
            break
        if coefficient(mid) > cv:
            lo = mid
        else:
            hi = mid
    kappa = lo + (hi - lo) / 2.0
    mean, _ = _truncated_standard_moments(kappa)
    achieved = coefficient(kappa)
    # Around kappa=-10 the two truncated moments are individually stable but
    # their ratio has a few 1e-10 of binary64 jitter.  This tolerance is still
    # far below any generator-level statistical or geometric tolerance.
    if abs(achieved - cv) > max(2.0e-9, 2.0e-9 * cv):
        raise NumericDegeneracyError("normal CV parameter solve did not converge")
    return kappa, mean


def sample_log_volume_multipliers(
    family: VolumeFamily,
    cv: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample log(Z) for a positive unit-mean volume multiplier."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    if not math.isfinite(cv) or cv < 0.0:
        raise ValueError("volume CV must be finite and nonnegative")
    if family == "fixed" and cv != 0.0:
        raise ValueError("fixed volume requires CV=0")
    if family == "exponential" and cv != 1.0:
        raise ValueError("exponential volume requires CV=1")
    if family == "normal" and not (0.0 <= cv < 1.0):
        raise ValueError("truncated-normal volume requires 0<=CV<1")
    if family == "fixed" or (family in {"lognormal", "normal"} and cv == 0.0):
        return np.zeros(count, dtype=np.float64)
    if family == "exponential":
        u = _open_uniform(rng, count)
        return np.log(-np.log1p(-u))
    if family == "lognormal":
        sigma2 = _lognormal_sigma_squared(cv)
        return rng.normal(-0.5 * sigma2, math.sqrt(sigma2), size=count)
    if family == "normal":
        kappa, mean = _normal_parameters(cv)
        u = _open_uniform(rng, count)
        log_conditional_tail = float(log_ndtr(kappa)) + np.log(u)
        standard = -ndtri_exp(log_conditional_tail)
        excess = kappa + standard
        if not np.all(np.isfinite(excess)) or not np.all(excess > 0.0):
            raise NumericDegeneracyError("truncated-normal tail inversion lost positive excess")
        return np.log(excess) - math.log(mean)
    raise ValueError(f"unsupported volume family: {family}")


def sample_log_shapes(
    shape_sigma: float,
    count: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 0 or dimension <= 0:
        raise ValueError("shape count must be nonnegative and dimension positive")
    if not math.isfinite(shape_sigma) or shape_sigma < 0.0:
        raise ValueError("shape_sigma must be finite and nonnegative")
    if dimension == 1 or shape_sigma == 0.0:
        return np.zeros((count, dimension), dtype=np.float64)
    values = rng.normal(0.0, shape_sigma, size=(count, dimension))
    # NumPy uses pairwise reductions for the relevant dimensions.  A second
    # symmetric projection absorbs the first projection's residual without
    # privileging a coordinate.
    values -= np.mean(values, axis=1, keepdims=True)
    values -= np.mean(values, axis=1, keepdims=True)
    return values


@dataclasses.dataclass
class BaseSamples:
    log_z_r: np.ndarray
    shape_r: np.ndarray
    log_z_s: np.ndarray
    shape_s: np.ndarray

    @property
    def count(self) -> int:
        return int(self.log_z_r.size)

    def append(self, other: "BaseSamples") -> "BaseSamples":
        if self.count == 0:
            return other
        return BaseSamples(
            np.concatenate((self.log_z_r, other.log_z_r)),
            np.concatenate((self.shape_r, other.shape_r), axis=0),
            np.concatenate((self.log_z_s, other.log_z_s)),
            np.concatenate((self.shape_s, other.shape_s), axis=0),
        )


class _PairSampleStream:
    def __init__(self, config: AlacarteConfig, base_seed: int | str, domain: Sequence[Any]):
        self.config = config
        prefix = tuple(domain)
        self._r_volume = philox_rng(base_seed, *prefix, "R", "volume")
        self._r_shape = philox_rng(base_seed, *prefix, "R", "shape")
        self._s_volume = philox_rng(base_seed, *prefix, "S", "volume")
        self._s_shape = philox_rng(base_seed, *prefix, "S", "shape")

    def draw(self, count: int) -> BaseSamples:
        cfg = self.config
        return BaseSamples(
            sample_log_volume_multipliers(cfg.volume_family, cfg.volume_cv, count, self._r_volume),
            sample_log_shapes(cfg.shape_sigma, count, cfg.dimension, self._r_shape),
            sample_log_volume_multipliers(cfg.volume_family, cfg.volume_cv, count, self._s_volume),
            sample_log_shapes(cfg.shape_sigma, count, cfg.dimension, self._s_shape),
        )


def draw_base_samples(
    config: AlacarteConfig,
    base_seed: int | str,
    count: int,
    *domain: Any,
) -> BaseSamples:
    config.validate()
    return _PairSampleStream(config, base_seed, domain or ("public-base-samples",)).draw(count)


def p1d(a: Any, b: Any, width: float = 1.0) -> Any:
    """Closed-form intersection probability for two uniformly placed intervals."""

    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    x, y = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64) / width,
        np.asarray(b, dtype=np.float64) / width,
    )
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("interval lengths must be finite")
    if np.any((x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)):
        raise ValueError("interval lengths must lie in [0,width]")
    result = np.zeros_like(x)
    positive = (x > 0.0) & (y > 0.0)
    certain = positive & (x + y >= 1.0)
    interior = positive & ~certain
    result[certain] = 1.0
    if np.any(interior):
        xi, yi = x[interior], y[interior]
        numerator = xi * (1.0 - xi - yi) + yi * (1.0 - yi)
        result[interior] = numerator / ((1.0 - xi) * (1.0 - yi))
    result = np.clip(result, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def _log1mexp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    near = values > -_LOG_TWO
    result[near] = np.log(-np.expm1(values[near]))
    result[~near] = np.log1p(-np.exp(values[~near]))
    return result


def log_p1d_from_log_relative_lengths(log_x: Any, log_y: Any) -> Any:
    """Stable log P1D from log relative lengths (both must be <= 0)."""

    x, y = np.broadcast_arrays(
        np.asarray(log_x, dtype=np.float64), np.asarray(log_y, dtype=np.float64)
    )
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(x > 0.0) or np.any(y > 0.0):
        raise ValueError("log relative lengths must be non-NaN and no greater than zero")
    result = np.full_like(x, -np.inf)
    positive = np.isfinite(x) & np.isfinite(y)
    if np.any(positive):
        xp, yp = x[positive], y[positive]
        log_sum = np.logaddexp(xp, yp)
        local = np.zeros_like(log_sum)
        interior = log_sum < 0.0
        if np.any(interior):
            xi, yi, si = xp[interior], yp[interior], log_sum[interior]
            log_numerator = np.logaddexp(
                xi + _log1mexp(si),
                yi + _log1mexp(yi),
            )
            local[interior] = (
                log_numerator - _log1mexp(xi) - _log1mexp(yi)
            )
        result[positive] = np.minimum(local, 0.0)
    return float(result) if result.ndim == 0 else result


def log_pair_probabilities(
    config: AlacarteConfig, theta: float, samples: BaseSamples
) -> np.ndarray:
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    d = config.dimension
    tau_max = math.log1p(-config.eps_geom)
    eta_r = theta - math.log(config.n_r) + samples.log_z_r
    eta_s = theta - math.log(config.n_s) + samples.log_z_s
    tau_r = np.minimum(eta_r[:, None] / d + samples.shape_r, tau_max)
    tau_s = np.minimum(eta_s[:, None] / d + samples.shape_s, tau_max)
    log_dims = log_p1d_from_log_relative_lengths(tau_r, tau_s)
    result = np.sum(log_dims, axis=1)
    if np.any(np.isnan(result)):
        raise NumericDegeneracyError("nonfinite pair probability")
    # This function is a point-valued diagnostic API.  Certification instead
    # uses _pair_probability_enclosures and never relies on this clamp.
    return np.minimum(result, 0.0)


@dataclasses.dataclass(frozen=True)
class ProbabilityEstimate:
    value: float
    log_value: float
    lower: float
    upper: float
    sample_count: int


def estimate_pair_probability(
    config: AlacarteConfig, theta: float, samples: BaseSamples
) -> ProbabilityEstimate:
    if samples.count <= 0:
        raise ValueError("at least one base sample is required")
    q_point, q_lower, q_upper = _pair_probability_enclosures(config, theta, samples)
    denominator = samples.count * _BINARY64_DENOMINATOR
    lower_exact = Fraction(_binary64_sum_units(q_lower), denominator)
    upper_exact = Fraction(_binary64_sum_units(q_upper), denominator)
    point_exact = Fraction(_binary64_sum_units(q_point), denominator)
    lower = max(0.0, _fraction_to_float_bounds(lower_exact)[0])
    upper = min(1.0, _fraction_to_float_bounds(upper_exact)[1])
    value = min(upper, max(lower, float(point_exact)))
    log_mean = math.log(value) if value > 0.0 else -math.inf
    return ProbabilityEstimate(value, log_mean, lower, upper, samples.count)


@dataclasses.dataclass(frozen=True)
class CertificationCheckpoint:
    sample_count: int
    p_num: float
    variance_upper: float
    numeric_radius: float
    statistical_radius: float
    probability_interval: tuple[float, float]
    output_density_interval: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CoverageResult:
    status: CoverageStatus
    theta: float | None = None
    probability_estimate: float | None = None
    output_density_estimate: float | None = None
    output_density_interval: tuple[float, float] | None = None
    calibration_round: int = 0
    calibration_samples: int = 0
    calibration_evaluations: int = 0
    certification_samples: int = 0
    checkpoints: tuple[CertificationCheckpoint, ...] = ()
    message: str = ""

    @property
    def certified(self) -> bool:
        return self.status is CoverageStatus.CERTIFIED

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["status"] = self.status.value
        value["certified"] = self.certified
        return value


@dataclasses.dataclass(frozen=True)
class _RootResult:
    status: CoverageStatus
    theta: float | None
    estimate: ProbabilityEstimate | None
    evaluations: int
    message: str


def _comparison(estimate: ProbabilityEstimate, target: Fraction) -> int:
    if Fraction.from_float(estimate.upper) < target:
        return -1
    if Fraction.from_float(estimate.lower) > target:
        return 1
    return 0


def _residual_within(
    estimate: ProbabilityEstimate, target: Fraction, tolerance: float
) -> bool:
    radius = Fraction.from_float(tolerance)
    return (
        Fraction.from_float(estimate.lower) >= target - radius
        and Fraction.from_float(estimate.upper) <= target + radius
    )


def _point_distance(value: float, target: Fraction) -> Fraction:
    return abs(Fraction.from_float(value) - target)


def _small_object_initial(config: AlacarteConfig) -> float:
    d = config.dimension
    log_kappa = config.log_a + d * float(
        np.logaddexp(-math.log(config.n_r) / d, -math.log(config.n_s) / d)
    )
    return math.log(config.alpha_target) - log_kappa


def _find_empirical_root(
    config: AlacarteConfig,
    samples: BaseSamples,
    train_tolerance: float,
) -> _RootResult:
    target = _target_probability(config)
    budget = config.solver
    evaluations = 0

    def evaluate(theta: float) -> ProbabilityEstimate:
        nonlocal evaluations
        evaluations += 1
        return estimate_pair_probability(config, theta, samples)

    theta0 = min(budget.theta_max, max(budget.theta_min, _small_object_initial(config)))
    try:
        initial = evaluate(theta0)
    except (FloatingPointError, OverflowError, NumericDegeneracyError, ValueError) as error:
        return _RootResult(CoverageStatus.NONFINITE_EVALUATION, None, None, evaluations, str(error))
    if not math.isfinite(initial.log_value):
        return _RootResult(
            CoverageStatus.TARGET_BELOW_NUMERIC_RESOLUTION,
            None,
            initial,
            evaluations,
            "empirical probability is not representable",
        )
    initial_cmp = _comparison(initial, target)
    if initial_cmp == 0:
        if _residual_within(initial, target, train_tolerance):
            return _RootResult(CoverageStatus.CERTIFIED, theta0, initial, evaluations, "empirical root at initial point")
        return _RootResult(
            CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
            None,
            initial,
            evaluations,
            "numeric enclosure contains target but exceeds training tolerance",
        )

    lo: float | None = None
    hi: float | None = None
    lo_est: ProbabilityEstimate | None = None
    hi_est: ProbabilityEstimate | None = None
    if initial_cmp < 0:
        lo, lo_est = theta0, initial
        direction = 1.0
    else:
        hi, hi_est = theta0, initial
        direction = -1.0

    theta = theta0
    for _ in range(budget.max_expand):
        candidate = theta + direction * _LOG_TWO
        if candidate < budget.theta_min or candidate > budget.theta_max:
            return _RootResult(
                CoverageStatus.BRACKET_NOT_FOUND,
                None,
                None,
                evaluations,
                "target was not bracketed inside configured theta range",
            )
        if candidate == theta:
            return _RootResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                None,
                None,
                evaluations,
                "theta expansion made no floating-point progress",
            )
        theta = candidate
        try:
            current = evaluate(theta)
        except (FloatingPointError, OverflowError, NumericDegeneracyError, ValueError) as error:
            return _RootResult(CoverageStatus.NONFINITE_EVALUATION, None, None, evaluations, str(error))
        cmp = _comparison(current, target)
        if cmp == 0:
            if _residual_within(current, target, train_tolerance):
                return _RootResult(CoverageStatus.CERTIFIED, theta, current, evaluations, "empirical root during expansion")
            return _RootResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                None,
                current,
                evaluations,
                "numeric enclosure contains target but exceeds training tolerance",
            )
        if direction > 0:
            if cmp > 0:
                hi, hi_est = theta, current
                break
            lo, lo_est = theta, current
        else:
            if cmp < 0:
                lo, lo_est = theta, current
                break
            hi, hi_est = theta, current
    else:
        return _RootResult(
            CoverageStatus.EXPANSION_LIMIT_REACHED,
            None,
            None,
            evaluations,
            "configured expansion limit reached",
        )

    if lo is None or hi is None or lo_est is None or hi_est is None:
        return _RootResult(CoverageStatus.BRACKET_NOT_FOUND, None, None, evaluations, "internal bracket failure")

    best_theta = lo if _point_distance(lo_est.value, target) < _point_distance(hi_est.value, target) else hi
    best_est = lo_est if best_theta == lo else hi_est
    for _ in range(budget.max_bisect):
        mid = lo + (hi - lo) / 2.0
        if mid == lo or mid == hi:
            return _RootResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                None,
                best_est,
                evaluations,
                "theta bisection made no floating-point progress",
            )
        try:
            estimate = evaluate(mid)
        except (FloatingPointError, OverflowError, NumericDegeneracyError, ValueError) as error:
            return _RootResult(CoverageStatus.NONFINITE_EVALUATION, None, None, evaluations, str(error))
        if _point_distance(estimate.value, target) < _point_distance(best_est.value, target):
            best_theta, best_est = mid, estimate
        cmp = _comparison(estimate, target)
        residual_within = _residual_within(estimate, target, train_tolerance)
        if residual_within and ((hi - lo) <= budget.theta_tolerance or cmp == 0):
            return _RootResult(CoverageStatus.CERTIFIED, mid, estimate, evaluations, "empirical bisection converged")
        if cmp < 0:
            lo, lo_est = mid, estimate
        elif cmp > 0:
            hi, hi_est = mid, estimate
        elif residual_within:
            return _RootResult(CoverageStatus.CERTIFIED, mid, estimate, evaluations, "empirical residual converged")
        else:
            return _RootResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                None,
                estimate,
                evaluations,
                "numeric enclosure cannot determine a bisection direction",
            )

    return _RootResult(
        CoverageStatus.BISECTION_LIMIT_REACHED,
        None,
        best_est,
        evaluations,
        "configured bisection limit reached",
    )


@dataclasses.dataclass
class _OnlineMoments:
    count: int = 0
    lower_sum_units: int = 0
    upper_sum_units: int = 0

    def update(self, lower: np.ndarray, upper: np.ndarray) -> None:
        if lower.shape != upper.shape:
            raise ValueError("probability enclosure arrays must have matching shapes")
        n = int(lower.size)
        if n == 0:
            return
        if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(lower > upper):
            raise NumericDegeneracyError("invalid conditional-probability enclosure")
        self.lower_sum_units += _binary64_sum_units(lower)
        self.upper_sum_units += _binary64_sum_units(upper)
        self.count += n

    def bounds(self) -> tuple[float, float, float, float, float]:
        if self.count < 2:
            raise ValueError("at least two samples are needed")
        denominator = self.count * _BINARY64_DENOMINATOR
        exact_lower = Fraction(self.lower_sum_units, denominator)
        exact_upper = Fraction(self.upper_sum_units, denominator)
        if not (0 <= exact_lower <= exact_upper <= 1):
            raise NumericDegeneracyError("exact mean enclosure left [0,1]")
        lower = max(0.0, _fraction_to_float_bounds(exact_lower)[0])
        upper = min(1.0, _fraction_to_float_bounds(exact_upper)[1])

        # For q_i in [0,1], unbiased sample variance obeys
        # V <= n/(n-1) * mean * (1-mean).  Maximize this concave expression
        # over the exact mean enclosure.  This remains adaptive in the sparse
        # regime without a cancellation-prone floating second moment.
        candidates = [exact_lower * (1 - exact_lower), exact_upper * (1 - exact_upper)]
        if exact_lower <= Fraction(1, 2) <= exact_upper:
            candidates.append(Fraction(1, 4))
        variance_exact_upper = Fraction(self.count, self.count - 1) * max(candidates)
        variance_upper = min(1.0, _fraction_upper(variance_exact_upper))

        point_exact = (exact_lower + exact_upper) / 2
        p_num = min(upper, max(lower, float(point_exact)))
        numeric_radius_exact = max(
            abs(Fraction.from_float(p_num) - Fraction.from_float(lower)),
            abs(Fraction.from_float(upper) - Fraction.from_float(p_num)),
        )
        numeric_radius = _fraction_upper(numeric_radius_exact)
        return lower, upper, variance_upper, p_num, numeric_radius


def _empirical_bernstein_radius_upper(
    variance_upper: float,
    sample_count: int,
    delta: float,
    calibration_rounds: int,
    certification_checkpoints: int,
) -> float:
    """Upward-rounded empirical-Bernstein radius.

    Decimal's correctly rounded ``ln`` and ``sqrt`` are evaluated at 80 digits;
    an extra decimal successor encloses each transcendental result.  All other
    operations use ROUND_CEILING.  The final Decimal-to-binary64 conversion is
    also directed upward.
    """

    if sample_count < 2 or not (0.0 <= variance_upper <= 1.0):
        raise ValueError("invalid empirical-Bernstein inputs")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0,1)")
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        multiplicity = Decimal(4 * calibration_rounds * certification_checkpoints)
        argument = multiplicity / Decimal.from_float(delta)
        log_factor = argument.ln(context=context).next_plus(context=context)
        first_argument = (
            Decimal(2)
            * Decimal.from_float(variance_upper)
            * log_factor
            / Decimal(sample_count)
        )
        first = first_argument.sqrt(context=context).next_plus(context=context)
        second = (
            Decimal(7)
            * log_factor
            / (Decimal(3) * Decimal(sample_count - 1))
        )
        radius = first + second
        nearest = float(radius)
        if Decimal.from_float(nearest) < radius:
            nearest = math.nextafter(nearest, math.inf)
        return nearest


def _probability_confidence_interval(
    p_num: float, statistical_radius: float, numeric_radius: float
) -> tuple[float, float]:
    total_radius = Fraction.from_float(statistical_radius) + Fraction.from_float(
        numeric_radius
    )
    point = Fraction.from_float(p_num)
    exact_lower = max(Fraction(0), point - total_radius)
    exact_upper = min(Fraction(1), point + total_radius)
    return (
        max(0.0, _fraction_to_float_bounds(exact_lower)[0]),
        min(1.0, _fraction_to_float_bounds(exact_upper)[1]),
    )


def _scale_probability_interval(
    config: AlacarteConfig, interval: tuple[float, float]
) -> tuple[float, float]:
    scale = _exact_scale(config)
    exact_lower = Fraction.from_float(interval[0]) * scale
    exact_upper = Fraction.from_float(interval[1]) * scale
    return (
        max(0.0, _fraction_to_float_bounds(exact_lower)[0]),
        _fraction_to_float_bounds(exact_upper)[1],
    )


def _scale_probability_point(config: AlacarteConfig, probability: float) -> float:
    return float(Fraction.from_float(probability) * _exact_scale(config))


@dataclasses.dataclass(frozen=True)
class _CertificationResult:
    status: CoverageStatus
    checkpoint: CertificationCheckpoint | None
    checkpoints: tuple[CertificationCheckpoint, ...]
    disjoint: bool
    message: str


def _certify_candidate(
    config: AlacarteConfig,
    theta: float,
    base_seed: int | str,
    calibration_round: int,
) -> _CertificationResult:
    budget = config.solver
    stream = _PairSampleStream(config, base_seed, ("certification", calibration_round))
    moments = _OnlineMoments()
    checkpoints: list[CertificationCheckpoint] = []
    target = _target_probability(config)
    epsilon_p = _probability_tolerance(config)

    for index in range(budget.certification_checkpoints):
        desired = budget.certification_initial * (1 << index)
        while moments.count < desired:
            count = min(budget.certification_batch, desired - moments.count)
            try:
                samples = stream.draw(count)
                _, q_lower, q_upper = _pair_probability_enclosures(config, theta, samples)
                moments.update(q_lower, q_upper)
            except (FloatingPointError, OverflowError, NumericDegeneracyError, ValueError) as error:
                return _CertificationResult(
                    CoverageStatus.NONFINITE_EVALUATION, None, tuple(checkpoints), False, str(error)
                )

        try:
            _, _, variance_upper, p_num, numeric_radius = moments.bounds()
            statistical_radius = _empirical_bernstein_radius_upper(
                variance_upper,
                desired,
                config.delta,
                budget.calibration_rounds,
                budget.certification_checkpoints,
            )
            probability_interval = _probability_confidence_interval(
                p_num, statistical_radius, numeric_radius
            )
            output_interval = _scale_probability_interval(config, probability_interval)
        except (ArithmeticError, NumericDegeneracyError, ValueError) as error:
            return _CertificationResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                None,
                tuple(checkpoints),
                False,
                str(error),
            )
        checkpoint = CertificationCheckpoint(
            desired,
            p_num,
            variance_upper,
            numeric_radius,
            statistical_radius,
            probability_interval,
            output_interval,
        )
        checkpoints.append(checkpoint)

        if Fraction.from_float(numeric_radius) > epsilon_p / 8:
            return _CertificationResult(
                CoverageStatus.TARGET_BELOW_NUMERIC_RESOLUTION,
                checkpoint,
                tuple(checkpoints),
                False,
                "numeric probability radius exceeds epsilon_p/8",
            )
        target_band = (target - epsilon_p, target + epsilon_p)
        interval_exact = (
            Fraction.from_float(probability_interval[0]),
            Fraction.from_float(probability_interval[1]),
        )
        if interval_exact[0] >= target_band[0] and interval_exact[1] <= target_band[1]:
            return _CertificationResult(
                CoverageStatus.CERTIFIED,
                checkpoint,
                tuple(checkpoints),
                False,
                "independent empirical-Bernstein certificate succeeded",
            )
        disjoint = interval_exact[1] < target_band[0] or interval_exact[0] > target_band[1]
        if disjoint:
            return _CertificationResult(
                CoverageStatus.CERTIFICATION_NOT_REACHED,
                checkpoint,
                tuple(checkpoints),
                True,
                "independent interval rejects this calibration candidate",
            )

    return _CertificationResult(
        CoverageStatus.SAMPLE_BUDGET_EXCEEDED,
        checkpoints[-1] if checkpoints else None,
        tuple(checkpoints),
        False,
        "certification budget exhausted while interval still overlaps target band",
    )


def solve_coverage(config: AlacarteConfig, base_seed: int | str = 0) -> CoverageResult:
    """Calibrate theta=log(C), then independently certify expected density."""

    try:
        config.validate()
    except (ValueError, NumericDegeneracyError) as error:
        return CoverageResult(CoverageStatus.INVALID_INPUT, message=str(error))
    target = _target_probability(config)
    epsilon_p = _probability_tolerance(config)
    target_bounds = _fraction_to_float_bounds(target)
    epsilon_eighth_lower = _fraction_to_float_bounds(epsilon_p / 8)[0]
    if target_bounds[1] <= 0.0 or epsilon_eighth_lower <= 0.0:
        return CoverageResult(
            CoverageStatus.TARGET_BELOW_NUMERIC_RESOLUTION,
            message="target probability or epsilon_p/8 is below binary64 resolution",
        )

    calibration_stream = _PairSampleStream(config, base_seed, ("calibration",))
    samples = BaseSamples(
        np.empty(0), np.empty((0, config.dimension)), np.empty(0), np.empty((0, config.dimension))
    )
    total_evaluations = 0
    last_root: _RootResult | None = None
    last_certification: _CertificationResult | None = None

    for round_index in range(1, config.solver.calibration_rounds + 1):
        desired = config.solver.calibration_initial * (1 << (round_index - 1))
        try:
            samples = samples.append(calibration_stream.draw(desired - samples.count))
        except (FloatingPointError, OverflowError, NumericDegeneracyError, ValueError) as error:
            return CoverageResult(
                CoverageStatus.NUMERICAL_RESOLUTION_EXHAUSTED,
                calibration_round=round_index,
                calibration_samples=samples.count,
                calibration_evaluations=total_evaluations,
                message=str(error),
            )
        train_tolerance = min(
            epsilon_eighth_lower,
            config.solver.train_tolerance_initial / (1 << (round_index - 1)),
        )
        root = _find_empirical_root(config, samples, train_tolerance)
        last_root = root
        total_evaluations += root.evaluations
        if root.status is not CoverageStatus.CERTIFIED or root.theta is None or root.estimate is None:
            return CoverageResult(
                root.status,
                calibration_round=round_index,
                calibration_samples=samples.count,
                calibration_evaluations=total_evaluations,
                message=root.message,
            )

        certification = _certify_candidate(config, root.theta, base_seed, round_index)
        last_certification = certification
        checkpoint = certification.checkpoint
        if certification.status is CoverageStatus.CERTIFIED and checkpoint is not None:
            return CoverageResult(
                CoverageStatus.CERTIFIED,
                theta=root.theta,
                probability_estimate=checkpoint.p_num,
                output_density_estimate=_scale_probability_point(config, checkpoint.p_num),
                output_density_interval=checkpoint.output_density_interval,
                calibration_round=round_index,
                calibration_samples=samples.count,
                calibration_evaluations=total_evaluations,
                certification_samples=checkpoint.sample_count,
                checkpoints=certification.checkpoints,
                message=certification.message,
            )
        if certification.disjoint:
            continue
        return CoverageResult(
            certification.status,
            theta=root.theta,
            probability_estimate=checkpoint.p_num if checkpoint else None,
            output_density_estimate=(
                _scale_probability_point(config, checkpoint.p_num) if checkpoint else None
            ),
            output_density_interval=checkpoint.output_density_interval if checkpoint else None,
            calibration_round=round_index,
            calibration_samples=samples.count,
            calibration_evaluations=total_evaluations,
            certification_samples=checkpoint.sample_count if checkpoint else 0,
            checkpoints=certification.checkpoints,
            message=certification.message,
        )

    checkpoint = last_certification.checkpoint if last_certification else None
    return CoverageResult(
        CoverageStatus.CERTIFICATION_NOT_REACHED,
        theta=last_root.theta if last_root else None,
        probability_estimate=checkpoint.p_num if checkpoint else None,
        output_density_estimate=(
            _scale_probability_point(config, checkpoint.p_num) if checkpoint else None
        ),
        output_density_interval=checkpoint.output_density_interval if checkpoint else None,
        calibration_round=config.solver.calibration_rounds,
        calibration_samples=samples.count,
        calibration_evaluations=total_evaluations,
        certification_samples=checkpoint.sample_count if checkpoint else 0,
        checkpoints=last_certification.checkpoints if last_certification else (),
        message="all calibration candidates were independently rejected",
    )


@dataclasses.dataclass(frozen=True)
class GeneratedBoxes:
    ids: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    log_relative_lengths: np.ndarray
    saturated: np.ndarray


def generate_box_set(
    config: AlacarteConfig,
    theta: float,
    base_seed: int | str,
    side: Side,
) -> GeneratedBoxes:
    """Generate one side from streams independent of every solver stream."""

    config.validate()
    if side not in {"R", "S"}:
        raise ValueError("side must be R or S")
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    count = config.n_r if side == "R" else config.n_s
    n_side = count
    volume_rng = philox_rng(base_seed, "generation", side, "volume")
    shape_rng = philox_rng(base_seed, "generation", side, "shape")
    position_rng = philox_rng(base_seed, "generation", side, "position")
    log_z = sample_log_volume_multipliers(config.volume_family, config.volume_cv, count, volume_rng)
    shapes = sample_log_shapes(config.shape_sigma, count, config.dimension, shape_rng)
    eta = theta - math.log(n_side) + log_z
    raw_tau = eta[:, None] / config.dimension + shapes
    tau_max = math.log1p(-config.eps_geom)
    saturated = raw_tau >= tau_max
    tau = np.minimum(raw_tau, tau_max)
    rho = np.exp(tau)
    if not np.isfinite(rho).all() or not np.all(rho > 0.0):
        raise NumericDegeneracyError("positive relative length is not representable")
    slack = 1.0 - rho
    q = _open_uniform(position_rng, (count, config.dimension))
    lower_normalized = q * slack
    upper_normalized = lower_normalized + rho
    universe_lower, universe_upper = config.universe()
    widths = universe_upper - universe_lower
    lower = universe_lower + widths * lower_normalized
    upper = universe_lower + widths * upper_normalized
    if not (
        np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.all(lower >= universe_lower)
        and np.all(upper <= universe_upper)
        and np.all(lower < upper)
    ):
        raise NumericDegeneracyError("generated positive box collapsed or left its universe in binary64")
    start = 0 if side == "R" else config.n_r
    ids = np.arange(start, start + count, dtype=np.uint64)
    return GeneratedBoxes(ids, lower, upper, tau, saturated)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    keys = ("p50", "p90", "p95", "p99")
    result = np.quantile(values, (0.5, 0.9, 0.95, 0.99))
    return {key: float(item) for key, item in zip(keys, result, strict=True)}


def _geometry_diagnostics(config: AlacarteConfig, r: GeneratedBoxes, s: GeneratedBoxes) -> dict[str, Any]:
    tau = np.concatenate((r.log_relative_lengths, s.log_relative_lengths), axis=0)
    saturated = np.concatenate((r.saturated, s.saturated), axis=0)
    log_volume = np.sum(tau, axis=1)
    log_aspect = np.max(tau, axis=1) - np.min(tau, axis=1)
    aspect = np.exp(np.minimum(log_aspect, math.log(_FLOAT.max)))
    lower = np.concatenate((r.lower, s.lower), axis=0)
    upper = np.concatenate((r.upper, s.upper), axis=0)
    universe_lower, universe_upper = config.universe()
    duplicate_fraction: list[float] = []
    for dimension in range(config.dimension):
        endpoints = np.concatenate((lower[:, dimension], upper[:, dimension]))
        duplicate_fraction.append(float(1.0 - np.unique(endpoints).size / endpoints.size))
    return {
        "relative_length_quantiles": _quantiles(np.exp(tau.ravel())),
        "relative_volume_quantiles": _quantiles(np.exp(log_volume)),
        "log_relative_volume_quantiles": _quantiles(log_volume),
        "normalized_aspect_ratio_quantiles": _quantiles(aspect),
        "saturation_fraction": [float(x) for x in np.mean(saturated, axis=0)],
        "saturation_fraction_R": [float(x) for x in np.mean(r.saturated, axis=0)],
        "saturation_fraction_S": [float(x) for x in np.mean(s.saturated, axis=0)],
        "duplicate_endpoint_fraction": duplicate_fraction,
        "boundary_touching_fraction": {
            "lower": [float(x) for x in np.mean(lower == universe_lower, axis=0)],
            "upper": [float(x) for x in np.mean(upper == universe_upper, axis=0)],
        },
    }


@dataclasses.dataclass(frozen=True)
class SyntheticDataset:
    r_ids: np.ndarray
    r_lower: np.ndarray
    r_upper: np.ndarray
    s_ids: np.ndarray
    s_lower: np.ndarray
    s_upper: np.ndarray
    metadata: dict[str, Any]
    coverage: CoverageResult | None = None


def generate_at_coverage(
    config: AlacarteConfig,
    theta: float,
    base_seed: int | str = 0,
    *,
    coverage: CoverageResult | None = None,
) -> SyntheticDataset:
    """Generate R/S at a supplied theta; normally called with a certificate."""

    config.validate()
    if coverage is not None and (not coverage.certified or coverage.theta != theta):
        raise ValueError("coverage must be a matching CERTIFIED result")
    r = generate_box_set(config, theta, base_seed, "R")
    s = generate_box_set(config, theta, base_seed, "S")
    try:
        coverage_parameter: float | None = math.exp(theta)
    except OverflowError:
        coverage_parameter = None
    if coverage_parameter is not None and not math.isfinite(coverage_parameter):
        coverage_parameter = None
    domains = {}
    for phase, side, variable in (
        ("generation", "R", "volume"),
        ("generation", "R", "shape"),
        ("generation", "R", "position"),
        ("generation", "S", "volume"),
        ("generation", "S", "shape"),
        ("generation", "S", "position"),
    ):
        name = f"{phase}/{side}/{variable}"
        domains[name] = philox_stream_id(base_seed, phase, side, variable)
    solver_domains: dict[str, Any] = {
        "calibration": philox_stream_id(base_seed, "calibration"),
    }
    if coverage is not None:
        solver_domains["certification_candidates"] = [
            philox_stream_id(base_seed, "certification", round_index)
            for round_index in range(1, coverage.calibration_round + 1)
        ]
    metadata: dict[str, Any] = {
        "dataset_family": "Alacarte",
        "model_version": ALACARTE_MODEL_VERSION,
        "endpoint_semantics": "binary64-half-open-strict-overlap",
        "numeric_certificate_protocol": {
            "scale_transform": "numpy-exp-output-frozen-as-binary64",
            "conditional_probability": "p1d-directed-binary64-interval-v1",
            "mean_reduction": "exact-binary64-superaccumulator-v1",
            "variance_bound": "n-over-n-minus-1-times-mean-one-minus-mean",
            "statistical_radius": "decimal80-upward-empirical-bernstein-v1",
            "basic_arithmetic_assumption": "IEEE-754-binary64-round-to-nearest",
            "random_source_assumption": "Philox streams model independent iid draws",
        },
        "alpha_target": float(config.alpha_target),
        "alpha_expected": coverage.output_density_estimate if coverage is not None else None,
        "alpha_realized": None,
        "coverage_status": coverage.status.value if coverage is not None else "SUPPLIED_UNCERTIFIED_THETA",
        "coverage_interval": coverage.output_density_interval if coverage is not None else None,
        "coverage_solver_config_sha256": stable_hash_hex(
            "alacarte-solver-config", dataclasses.asdict(config.solver)
        ),
        "shape_sigma": float(config.shape_sigma),
        "volume_family": config.volume_family,
        "volume_cv": float(config.volume_cv),
        "configuration": config.to_dict(),
        "configuration_sha256": stable_hash_hex("alacarte-config", config.to_dict()),
        "coverage_theta": float(theta),
        "nominal_coverage_parameter": coverage_parameter,
        "coverage": coverage.to_dict() if coverage is not None else {
            "status": "SUPPLIED_UNCERTIFIED_THETA",
            "certified": False,
        },
        "random_source": {
            "algorithm": "numpy.random.Philox",
            "numpy_version": np.__version__,
            "domain_scheme": PHILOX_STREAM_VERSION,
            "base_seed": base_seed,
            "final_generation_streams": domains,
            "solver_domain_roots": solver_domains,
            "solver_streams_are_disjoint": True,
            "R_and_S_streams_are_disjoint": True,
        },
        "geometry_diagnostics": _geometry_diagnostics(config, r, s),
    }
    return SyntheticDataset(
        r.ids,
        r.lower,
        r.upper,
        s.ids,
        s.lower,
        s.upper,
        metadata,
        coverage,
    )


def generate_synthetic(
    config: AlacarteConfig, base_seed: int | str = 0
) -> SyntheticDataset:
    """Solve, certify, and generate; never emits data from a failed solver."""

    coverage = solve_coverage(config, base_seed)
    if not coverage.certified or coverage.theta is None:
        raise CoverageFailure(coverage)
    return generate_at_coverage(config, coverage.theta, base_seed, coverage=coverage)


__all__ = [
    "ALACARTE_MODEL_VERSION",
    "AlacarteConfig",
    "BaseSamples",
    "CertificationCheckpoint",
    "CoverageFailure",
    "CoverageResult",
    "CoverageStatus",
    "GeneratedBoxes",
    "NumericDegeneracyError",
    "ProbabilityEstimate",
    "SolverBudget",
    "SyntheticDataset",
    "config_from_mapping",
    "draw_base_samples",
    "estimate_pair_probability",
    "generate_at_coverage",
    "generate_box_set",
    "generate_synthetic",
    "log_p1d_from_log_relative_lengths",
    "p1d",
    "philox_rng",
    "philox_stream_id",
    "sample_log_shapes",
    "sample_log_volume_multipliers",
    "solve_coverage",
]
