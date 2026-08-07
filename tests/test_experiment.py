"""Power, MDE and CUPED tests.

The power calculator is checked against its own inverse and against the
directions the formula must move in; CUPED is checked against the identity that
its variance reduction equals the squared correlation, which is the sharpest
available test that the implementation is right rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiment import (
    cuped_adjust,
    cuped_demo,
    design_validation_experiment,
    minimum_detectable_effect,
    required_sample_size,
)
from src.simulate import SyntheticData, simulate


def test_smaller_effects_need_more_customers() -> None:
    big = required_sample_size(0.05, absolute_mde=0.01)
    small = required_sample_size(0.05, absolute_mde=0.005)
    assert small.n_per_arm > big.n_per_arm


def test_sample_size_scales_roughly_with_the_inverse_square_of_the_effect() -> None:
    """Halving the detectable effect should roughly quadruple the sample."""
    big = required_sample_size(0.05, absolute_mde=0.01)
    small = required_sample_size(0.05, absolute_mde=0.005)
    assert 3.5 < small.n_per_arm / big.n_per_arm < 4.5


def test_more_power_needs_more_customers() -> None:
    low = required_sample_size(0.05, absolute_mde=0.01, power=0.80)
    high = required_sample_size(0.05, absolute_mde=0.01, power=0.95)
    assert high.n_per_arm > low.n_per_arm


def test_a_stricter_alpha_needs_more_customers() -> None:
    loose = required_sample_size(0.05, absolute_mde=0.01, alpha=0.10)
    strict = required_sample_size(0.05, absolute_mde=0.01, alpha=0.01)
    assert strict.n_per_arm > loose.n_per_arm


def test_rarer_outcomes_need_more_customers_for_the_same_relative_lift() -> None:
    common = required_sample_size(0.15, relative_mde=0.20)
    rare = required_sample_size(0.01, relative_mde=0.20)
    assert rare.n_per_arm > common.n_per_arm


def test_absolute_and_relative_effect_sizes_agree() -> None:
    absolute = required_sample_size(0.10, absolute_mde=0.02)
    relative = required_sample_size(0.10, relative_mde=0.20)
    assert absolute.n_per_arm == relative.n_per_arm


def test_mde_is_the_inverse_of_the_sample_size_calculation() -> None:
    """The two directions must be consistent, which is why one calls the other."""
    for baseline in (0.005, 0.05, 0.20):
        sizing = required_sample_size(baseline, relative_mde=0.25)
        recovered = minimum_detectable_effect(baseline, sizing.n_per_arm)
        assert recovered["absolute_mde"] == pytest.approx(sizing.absolute_mde, rel=0.02)


def test_more_customers_detect_smaller_effects() -> None:
    few = minimum_detectable_effect(0.05, 1_000)
    many = minimum_detectable_effect(0.05, 100_000)
    assert many["absolute_mde"] < few["absolute_mde"]


def test_allocation_ratio_is_respected() -> None:
    result = required_sample_size(0.05, absolute_mde=0.01, allocation_ratio=2.0)
    assert result.n_total == pytest.approx(3 * result.n_per_arm, rel=0.01)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"absolute_mde": 0.01, "relative_mde": 0.2}, "exactly one"),
        ({"absolute_mde": -0.01}, "must be positive"),
        ({"absolute_mde": 0.01, "alpha": 0.0}, "strictly between 0 and 1"),
        ({"absolute_mde": 0.01, "allocation_ratio": 0.0}, "must be positive"),
    ],
)
def test_power_inputs_are_validated(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        required_sample_size(0.05, **kwargs)


def test_invalid_baseline_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        required_sample_size(1.5, absolute_mde=0.01)


def test_mde_rejects_a_non_positive_sample() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        minimum_detectable_effect(0.05, 0)


# ---------------------------------------------------------------- CUPED


def test_cuped_variance_reduction_equals_squared_correlation(medium: SyntheticData) -> None:
    """The identity that proves the implementation, not just its plausibility."""
    _, result = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert result.variance_reduction == pytest.approx(result.correlation**2, abs=1e-9)


def test_cuped_does_not_move_the_effect_estimate(medium: SyntheticData) -> None:
    """CUPED must reduce variance without shifting the answer."""
    _, result = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert result.ate_adjusted == pytest.approx(result.ate_raw, abs=3 * result.stderr_raw)


def test_cuped_reduces_the_standard_error(medium: SyntheticData) -> None:
    _, result = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert result.stderr_adjusted < result.stderr_raw
    assert result.effective_sample_multiplier > 1.0


def test_cuped_helps_far_less_on_a_rare_binary_outcome(medium: SyntheticData) -> None:
    """The practical lesson: the technique is metric-dependent, not free."""
    _, binary = cuped_adjust(medium.outcome.astype(np.float64), medium.pre_period, medium.treatment)
    _, continuous = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert binary.variance_reduction < continuous.variance_reduction


def test_cuped_gains_nothing_from_an_unrelated_covariate(medium: SyntheticData) -> None:
    rng = np.random.default_rng(0)
    _, result = cuped_adjust(medium.engagement, rng.normal(size=medium.n), medium.treatment)
    assert result.variance_reduction < 0.01


def test_cuped_rejects_a_constant_covariate(medium: SyntheticData) -> None:
    with pytest.raises(ValueError, match="zero variance"):
        cuped_adjust(medium.engagement, np.ones(medium.n), medium.treatment)


def test_cuped_adjusted_values_have_the_right_shape(medium: SyntheticData) -> None:
    adjusted, _ = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert adjusted.shape == (medium.n,)
    assert adjusted.mean() == pytest.approx(medium.engagement.mean(), abs=1e-9)


def test_cuped_demo_reports_both_metrics() -> None:
    report = cuped_demo(n=8_000, seed=3)
    assert set(report) == {"conversion", "engagement"}
    for stats in report.values():
        assert stats["variance_reduction"] == pytest.approx(stats["correlation_squared"], abs=1e-9)


def test_cuped_on_a_post_treatment_covariate_would_bias_the_estimate() -> None:
    """Documents the failure mode rather than only warning about it in prose.

    Using a covariate the treatment influenced pulls the estimate toward zero,
    which is exactly what happens when a 'pre-period' window overlaps the campaign.
    """
    data = simulate(n=20_000, seed=21)
    contaminated = data.pre_period + 5.0 * data.treatment
    _, honest = cuped_adjust(data.engagement, data.pre_period, data.treatment)
    _, biased = cuped_adjust(data.engagement, contaminated, data.treatment)
    assert abs(biased.ate_adjusted - data.true_ate * 20.0) > abs(
        honest.ate_adjusted - data.true_ate * 20.0
    )


# ---------------------------------------------------------------- test design


def test_validation_design_needs_more_customers_for_a_smaller_edge() -> None:
    big = design_validation_experiment(
        0.01, expected_policy_uplift=0.02, expected_random_uplift=0.01
    )
    small = design_validation_experiment(
        0.01, expected_policy_uplift=0.012, expected_random_uplift=0.01
    )
    assert small["n_per_cell"] > big["n_per_cell"]


def test_validation_design_reports_infeasibility_honestly() -> None:
    """If the policy is not expected to beat the incumbent, say so, do not size a test."""
    result = design_validation_experiment(
        0.01, expected_policy_uplift=0.005, expected_random_uplift=0.01
    )
    assert not np.isfinite(result["n_per_cell"])


def test_cuped_shrinks_the_required_sample() -> None:
    without = design_validation_experiment(0.01, 0.02, 0.01, cuped_variance_reduction=0.0)
    with_cuped = design_validation_experiment(0.01, 0.02, 0.01, cuped_variance_reduction=0.5)
    assert with_cuped["n_per_cell"] == pytest.approx(without["n_per_cell"] * 0.5, rel=0.01)


def test_weekly_volume_converts_to_a_duration() -> None:
    result = design_validation_experiment(0.01, 0.02, 0.01, weekly_volume=10_000)
    assert result["weeks_required"] == np.ceil(result["n_total"] / 10_000)
