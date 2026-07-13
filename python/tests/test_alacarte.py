from __future__ import annotations

import dataclasses
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from anchor_exp.alacarte import (
    _BINARY64_DENOMINATOR,
    _OnlineMoments,
    _binary64_sum_units,
    _empirical_bernstein_radius_upper,
    _pair_probability_enclosures,
    AlacarteConfig,
    BaseSamples,
    CoverageStatus,
    SolverBudget,
    draw_base_samples,
    estimate_pair_probability,
    generate_box_set,
    log_p1d_from_log_relative_lengths,
    p1d,
    philox_rng,
    philox_stream_id,
    sample_log_volume_multipliers,
    solve_coverage,
)


def _config(**changes: object) -> AlacarteConfig:
    base = AlacarteConfig(
        n_r=64,
        n_s=64,
        dimension=3,
        alpha_target=2.0,
        volume_family="lognormal",
        volume_cv=0.5,
        shape_sigma=0.4,
        eps_geom=0.02,
        epsilon_alpha=0.5,
        delta=0.05,
    )
    return dataclasses.replace(base, **changes)


def test_p1d_closed_form_scale_and_boundaries() -> None:
    expected = 1.0 - 0.7**2 / (0.9 * 0.8)
    assert p1d(0.1, 0.2) == pytest.approx(expected)
    assert p1d(1.0, 2.0, 10.0) == pytest.approx(expected)
    assert p1d(0.0, 0.2) == 0.0
    assert p1d(0.4, 0.6) == 1.0
    assert p1d(0.6, 0.6) == 1.0
    assert p1d(0.1, 0.2) == p1d(0.2, 0.1)


def test_log_p1d_matches_closed_form_over_extreme_scales() -> None:
    x = np.array([1e-250, 1e-100, 1e-12, 1e-4, 0.1, 0.49, 0.8])
    y = np.array([2e-250, 1e-90, 2e-12, 2e-4, 0.2, 0.50, 0.3])
    logged = log_p1d_from_log_relative_lengths(np.log(x), np.log(y))
    direct = p1d(x, y)
    assert np.all(np.isfinite(logged))
    assert np.exp(logged) == pytest.approx(direct, rel=1e-13, abs=0.0)
    assert log_p1d_from_log_relative_lengths(-math.inf, math.log(0.2)) == -math.inf


def test_outward_p1d_encloses_exact_rational_extremes() -> None:
    xs = np.array(
        [
            np.nextafter(0.0, 1.0),
            1.0e-250,
            np.nextafter(0.5, 0.0),
            0.5,
            np.nextafter(1.0, 0.0),
        ]
    )
    ys = np.array(
        [
            2.0e-250,
            1.0e-100,
            0.5,
            0.5,
            np.nextafter(0.0, 1.0),
        ]
    )
    config = AlacarteConfig(
        n_r=1,
        n_s=1,
        dimension=1,
        alpha_target=0.1,
        epsilon_alpha=0.05,
        eps_geom=np.spacing(1.0),
    )
    # Compensate for theta=0,n=1 so the frozen scale transform receives these
    # adversarial log inputs.  The exact check uses the binary64 exp outputs,
    # which are the documented implementation-induced length semantics.
    samples = BaseSamples(
        np.log(xs),
        np.zeros((xs.size, 1)),
        np.log(ys),
        np.zeros((ys.size, 1)),
    )
    _, lower, upper = _pair_probability_enclosures(config, 0.0, samples)
    tau_max = math.log1p(-config.eps_geom)
    frozen_x = np.exp(np.minimum(samples.log_z_r, tau_max))
    frozen_y = np.exp(np.minimum(samples.log_z_s, tau_max))
    for x, y, lo, hi in zip(frozen_x, frozen_y, lower, upper, strict=True):
        x_exact = Fraction.from_float(float(x))
        y_exact = Fraction.from_float(float(y))
        if x_exact == 0 or y_exact == 0:
            probability = Fraction(0)
        elif x_exact + y_exact >= 1:
            probability = Fraction(1)
        else:
            numerator = x_exact * (1 - x_exact - y_exact) + y_exact * (1 - y_exact)
            probability = numerator / ((1 - x_exact) * (1 - y_exact))
        assert Fraction.from_float(float(lo)) <= probability
        assert probability <= Fraction.from_float(float(hi))


def test_exact_superaccumulator_and_variance_bound_on_adversarial_values() -> None:
    values = np.array(
        [
            0.0,
            np.nextafter(0.0, 1.0),
            np.nextafter(0.5, 0.0),
            0.5,
            np.nextafter(0.5, 1.0),
            1.0,
        ],
        dtype=np.float64,
    )
    exact_sum = sum((Fraction.from_float(float(x)) for x in values), Fraction())
    assert Fraction(_binary64_sum_units(values), _BINARY64_DENOMINATOR) == exact_sum

    moments = _OnlineMoments()
    moments.update(values[:2], values[:2])
    moments.update(values[2:], values[2:])
    lower, upper, variance_upper, p_num, numeric_radius = moments.bounds()
    exact_mean = exact_sum / values.size
    exact_variance = sum(
        ((Fraction.from_float(float(x)) - exact_mean) ** 2 for x in values), Fraction()
    ) / (values.size - 1)
    assert Fraction.from_float(lower) <= exact_mean <= Fraction.from_float(upper)
    assert Fraction.from_float(variance_upper) >= exact_variance
    assert lower <= p_num <= upper
    assert Fraction.from_float(numeric_radius) >= max(
        Fraction.from_float(p_num) - Fraction.from_float(lower),
        Fraction.from_float(upper) - Fraction.from_float(p_num),
    )


def test_empirical_bernstein_radius_is_upward_rounded() -> None:
    variance_upper = np.nextafter(1.0e-300, math.inf)
    sample_count = 1_000_003
    delta = np.nextafter(0.01, 0.0)
    radius = _empirical_bernstein_radius_upper(
        variance_upper, sample_count, delta, 7, 13
    )
    with localcontext() as context:
        context.prec = 160
        log_factor = (
            Decimal(4 * 7 * 13) / Decimal.from_float(delta)
        ).ln(context=context)
        reference = (
            Decimal(2)
            * Decimal.from_float(variance_upper)
            * log_factor
            / Decimal(sample_count)
        ).sqrt(context=context) + (
            Decimal(7) * log_factor / (Decimal(3) * Decimal(sample_count - 1))
        )
    assert Decimal.from_float(radius) >= reference


@pytest.mark.parametrize(
    ("family", "cv"),
    [("fixed", 0.0), ("exponential", 1.0), ("lognormal", 0.7), ("normal", 0.7)],
)
def test_generated_box_invariants_for_all_volume_families(family: str, cv: float) -> None:
    config = _config(volume_family=family, volume_cv=cv)
    boxes = generate_box_set(config, theta=-2.0, base_seed=19, side="R")
    lower, upper = config.universe()
    relative = np.exp(boxes.log_relative_lengths)
    assert boxes.lower.shape == (config.n_r, config.dimension)
    assert np.all(np.isfinite(boxes.lower))
    assert np.all(np.isfinite(boxes.upper))
    assert np.all(boxes.lower >= lower)
    assert np.all(boxes.upper <= upper)
    assert np.all(boxes.lower < boxes.upper)
    assert np.all(relative > 0.0)
    assert np.all(relative <= 1.0 - config.eps_geom)
    assert np.unique(boxes.ids).size == config.n_r


def test_common_random_numbers_make_empirical_objective_strictly_monotone() -> None:
    config = _config()
    samples = draw_base_samples(config, 123, 4096, "monotonicity")
    estimates = [
        estimate_pair_probability(config, theta, samples).value
        for theta in (-5.0, -3.0, -1.0, 1.0)
    ]
    assert estimates == sorted(estimates)
    assert all(left < right for left, right in zip(estimates, estimates[1:]))


def test_reproducibility_and_domain_separation() -> None:
    config = _config(volume_family="fixed", volume_cv=0.0, shape_sigma=0.0)
    first = generate_box_set(config, -2.0, 77, "R")
    # Consuming an unrelated calibration stream cannot perturb final generation.
    draw_base_samples(config, 77, 1000, "calibration")
    replay = generate_box_set(config, -2.0, 77, "R")
    other_side = generate_box_set(config, -2.0, 77, "S")
    other_seed = generate_box_set(config, -2.0, 78, "R")
    np.testing.assert_array_equal(first.lower, replay.lower)
    np.testing.assert_array_equal(first.upper, replay.upper)
    assert not np.array_equal(first.lower, other_side.lower)
    assert not np.array_equal(first.lower, other_seed.lower)
    stream_ids = {
        philox_stream_id(77, "calibration", "R", "shape"),
        philox_stream_id(77, "certification", 1, "R", "shape"),
        philox_stream_id(77, "generation", "R", "shape"),
        philox_stream_id(77, "generation", "S", "shape"),
    }
    assert len(stream_ids) == 4


def test_tail_safe_truncated_normal_sampler() -> None:
    rng = philox_rng(5, "normal-tail-test")
    values = np.exp(sample_log_volume_multipliers("normal", 0.99, 50_000, rng))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert np.mean(values) == pytest.approx(1.0, abs=0.035)
    assert np.std(values) / np.mean(values) == pytest.approx(0.99, abs=0.05)


def test_solver_returns_independent_empirical_bernstein_certificate() -> None:
    budget = SolverBudget(
        calibration_rounds=1,
        certification_checkpoints=4,
        calibration_initial=256,
        certification_initial=256,
        certification_batch=128,
        train_tolerance_initial=0.01,
        theta_tolerance=1e-8,
        theta_min=-50.0,
        theta_max=50.0,
        max_expand=64,
        max_bisect=80,
    )
    config = AlacarteConfig(
        n_r=10,
        n_s=10,
        dimension=2,
        alpha_target=1.0,
        volume_family="fixed",
        volume_cv=0.0,
        shape_sigma=0.0,
        epsilon_alpha=0.5,
        delta=0.1,
        solver=budget,
    )
    result = solve_coverage(config, 11)
    assert result.status is CoverageStatus.CERTIFIED
    assert result.theta is not None
    assert result.output_density_interval is not None
    low, high = result.output_density_interval
    assert config.alpha_target - config.epsilon_alpha <= low <= config.alpha_target
    assert config.alpha_target <= high <= config.alpha_target + config.epsilon_alpha
    assert result.certification_samples >= 2
    assert result.checkpoints


def test_solver_reports_invalid_input_and_finite_budget_exhaustion() -> None:
    invalid = _config(alpha_target=32.0)
    assert solve_coverage(invalid, 0).status is CoverageStatus.INVALID_INPUT

    tiny_budget = SolverBudget(
        calibration_rounds=1,
        certification_checkpoints=1,
        calibration_initial=64,
        certification_initial=2,
        certification_batch=2,
        train_tolerance_initial=0.1,
        theta_tolerance=1e-6,
        theta_min=-50.0,
        theta_max=50.0,
        max_expand=64,
        max_bisect=80,
    )
    config = AlacarteConfig(
        n_r=10,
        n_s=10,
        dimension=2,
        alpha_target=1.0,
        epsilon_alpha=0.1,
        delta=0.1,
        solver=tiny_budget,
    )
    assert solve_coverage(config, 0).status is CoverageStatus.SAMPLE_BUDGET_EXCEEDED
