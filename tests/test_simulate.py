"""The simulator is the yardstick for everything else, so it is tested hardest."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulate import (
    SLEEPING_DOG_THRESHOLD,
    SLEEPING_DOG_UPLIFT,
    SyntheticData,
    describe,
    simulate,
    simulate_confounded,
)


def test_shapes_are_consistent(small: SyntheticData) -> None:
    n = small.n
    for array in (
        small.treatment,
        small.outcome,
        small.spend,
        small.true_uplift,
        small.baseline_prob,
        small.pre_period,
        small.engagement,
        small.propensity,
    ):
        assert array.shape == (n,)
    assert small.features.shape[1] == len(small.feature_names)


def test_treatment_is_independent_of_covariates(medium: SyntheticData) -> None:
    """Randomisation means no feature predicts assignment beyond chance."""
    treated = medium.treatment == 1
    for j in range(medium.features.shape[1]):
        gap = medium.features[treated, j].mean() - medium.features[~treated, j].mean()
        standardised = abs(gap) / medium.features[:, j].std()
        assert standardised < 0.10, f"feature x{j} is imbalanced across arms"


def test_naive_difference_recovers_the_true_ate(medium: SyntheticData) -> None:
    """Under randomisation the naive gap is unbiased. This is the premise of the memo."""
    gap = (
        medium.outcome[medium.treatment == 1].mean() - medium.outcome[medium.treatment == 0].mean()
    )
    assert gap == pytest.approx(medium.true_ate, abs=0.01)


def test_confounding_breaks_that_guarantee(confounded: SyntheticData) -> None:
    """Same individual effects, non-random assignment, badly wrong naive answer."""
    gap = (
        confounded.outcome[confounded.treatment == 1].mean()
        - confounded.outcome[confounded.treatment == 0].mean()
    )
    assert gap > 3.0 * confounded.true_ate


def test_confounded_variant_preserves_the_effects(medium: SyntheticData) -> None:
    other = simulate_confounded(n=medium.n, seed=12)
    np.testing.assert_allclose(other.true_uplift, medium.true_uplift)
    np.testing.assert_allclose(other.features, medium.features)


def test_sleeping_dogs_exist_and_sit_where_designed(medium: SyntheticData) -> None:
    dogs = medium.sleeping_dog_mask
    assert dogs.any(), "the DGP must contain a genuinely harmed subgroup"
    designed = medium.features[:, 2] < SLEEPING_DOG_THRESHOLD
    # Every member of the designed subgroup is harmed, without exception: the
    # sleeping-dog effect overrides the responsive one rather than offsetting it.
    assert medium.true_uplift[designed].max() < 0.0
    assert medium.true_uplift[designed].mean() == pytest.approx(SLEEPING_DOG_UPLIFT, abs=0.01)
    # And the effect is positive on average for everyone else.
    assert medium.true_uplift[~designed].mean() > 0.0
    # The subgroup is a meaningful slice of the file, not a rounding error.
    assert 0.10 < designed.mean() < 0.25


def test_noise_features_have_no_effect_on_anything(medium: SyntheticData) -> None:
    """Columns x4 onward must be inert, or a model could 'find' signal in noise."""
    for j in range(4, medium.features.shape[1]):
        assert abs(np.corrcoef(medium.features[:, j], medium.true_uplift)[0, 1]) < 0.05
        assert abs(np.corrcoef(medium.features[:, j], medium.baseline_prob)[0, 1]) < 0.05


def test_counterfactual_outcomes_are_coupled(medium: SyntheticData) -> None:
    """A customer helped by treatment can never convert only in the control world."""
    helped = medium.true_uplift > 0
    impossible = helped & (medium.outcome_if_control == 1) & (medium.outcome_if_treated == 0)
    assert not impossible.any()


def test_true_uplift_equals_the_realised_probability_gap(medium: SyntheticData) -> None:
    """`true_uplift` must be the effect the sampler actually applied, post-clipping."""
    treated_prob = medium.baseline_prob + medium.true_uplift
    assert treated_prob.min() >= 0.0
    assert treated_prob.max() <= 1.0
    realised = medium.outcome_if_treated.mean() - medium.outcome_if_control.mean()
    assert realised == pytest.approx(medium.true_ate, abs=0.01)


def test_pre_period_covariate_is_unaffected_by_treatment(medium: SyntheticData) -> None:
    """The condition CUPED depends on: the covariate predates assignment."""
    treated = medium.treatment == 1
    gap = medium.pre_period[treated].mean() - medium.pre_period[~treated].mean()
    assert abs(gap) < 0.10 * medium.pre_period.std()


def test_engagement_responds_to_treatment(medium: SyntheticData) -> None:
    treated = medium.treatment == 1
    gap = medium.engagement[treated].mean() - medium.engagement[~treated].mean()
    assert gap > 0.0


def test_treatment_share_is_respected(imbalanced: SyntheticData) -> None:
    assert imbalanced.treatment.mean() == pytest.approx(0.15, abs=0.02)


def test_seeds_are_reproducible() -> None:
    a, b = simulate(n=1_000, seed=7), simulate(n=1_000, seed=7)
    np.testing.assert_array_equal(a.outcome, b.outcome)
    np.testing.assert_array_equal(a.treatment, b.treatment)


def test_different_seeds_differ() -> None:
    a, b = simulate(n=1_000, seed=7), simulate(n=1_000, seed=8)
    assert not np.array_equal(a.outcome, b.outcome)


def test_describe_reports_the_known_truth(small: SyntheticData) -> None:
    summary = describe(small)
    assert summary["n"] == small.n
    assert summary["true_ate"] == pytest.approx(small.true_ate)
    assert 0.0 < summary["sleeping_dog_share"] < 1.0
    assert summary["sleeping_dog_mean_uplift"] < 0.0
    assert summary["best_decile_mean_uplift"] > summary["true_ate"]


def test_to_frame_round_trips(small: SyntheticData) -> None:
    frame = small.to_frame()
    assert len(frame) == small.n
    assert {"treatment", "outcome", "spend", "true_uplift", "pre_period"} <= set(frame.columns)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_features": 3}, "at least 4 features"),
        ({"treatment_share": 0.0}, "strictly between 0 and 1"),
        ({"treatment_share": 1.0}, "strictly between 0 and 1"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate(n=100, **kwargs)  # type: ignore[arg-type]
