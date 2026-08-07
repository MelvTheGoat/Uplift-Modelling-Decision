"""Evaluation metrics, checked against cases whose answer is known in advance.

The important tests here are the degenerate ones: a random score must produce a
Qini coefficient near zero, and an oracle score must beat it. A metric that
rewards a random ranking is worse than no metric, because it will be believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators import SLearner, TLearner
from src.evaluation import (
    auuc_score,
    bootstrap_qini,
    cross_val_uplift,
    evaluate,
    qini_curve,
    recovery_error,
    transformed_outcome,
    transformed_outcome_loss,
    uplift_by_decile,
)
from src.simulate import SyntheticData


def test_qini_curve_starts_at_zero_and_is_monotone_in_fraction(medium: SyntheticData) -> None:
    rng = np.random.default_rng(0)
    result = qini_curve(medium.outcome, medium.treatment, rng.normal(size=medium.n))
    assert result.qini[0] == 0.0
    assert result.fraction[0] == 0.0
    assert result.fraction[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.fraction) > 0)


def test_qini_endpoint_equals_the_overall_incremental_effect(medium: SyntheticData) -> None:
    """At 100% targeting the curve must land on the trial's total incremental effect."""
    rng = np.random.default_rng(1)
    result = qini_curve(medium.outcome, medium.treatment, rng.normal(size=medium.n))
    treated, control = medium.treatment == 1, medium.treatment == 0
    expected = medium.outcome[treated].sum() - medium.outcome[control].sum() * (
        treated.sum() / control.sum()
    )
    assert result.total_incremental == pytest.approx(expected, rel=1e-9)


def test_random_scores_give_a_qini_coefficient_near_zero(medium: SyntheticData) -> None:
    """The central sanity check: noise must not look like signal."""
    rng = np.random.default_rng(2)
    coefficients = [
        qini_curve(medium.outcome, medium.treatment, rng.normal(size=medium.n)).qini_coefficient
        for _ in range(20)
    ]
    spread = np.std(coefficients)
    assert abs(np.mean(coefficients)) < spread


def test_oracle_scores_beat_random_scores(medium: SyntheticData) -> None:
    rng = np.random.default_rng(3)
    oracle = qini_curve(medium.outcome, medium.treatment, medium.true_uplift).qini_coefficient
    random = qini_curve(
        medium.outcome, medium.treatment, rng.normal(size=medium.n)
    ).qini_coefficient
    assert oracle > random


def test_reversing_the_score_reverses_the_sign(medium: SyntheticData) -> None:
    forward = qini_curve(medium.outcome, medium.treatment, medium.true_uplift).qini_coefficient
    backward = qini_curve(medium.outcome, medium.treatment, -medium.true_uplift).qini_coefficient
    assert forward > 0 > backward


def test_constant_scores_do_not_inherit_signal_from_row_order(medium: SyntheticData) -> None:
    """Ties are broken at random, so a flat prediction must score like a coin toss.

    Without random tie-breaking a degenerate model would be ranked by row order,
    and in a sorted file that can carry real signal.
    """
    flat = np.zeros(medium.n)
    coefficients = [
        qini_curve(medium.outcome, medium.treatment, flat, seed=s).qini_coefficient
        for s in range(15)
    ]
    assert abs(np.mean(coefficients)) < 2.0 * np.std(coefficients)


def test_qini_rejects_a_single_armed_dataset(medium: SyntheticData) -> None:
    with pytest.raises(ValueError, match="both arms"):
        qini_curve(medium.outcome, np.ones_like(medium.treatment), medium.true_uplift)


def test_qini_rejects_mismatched_shapes(medium: SyntheticData) -> None:
    with pytest.raises(ValueError, match="same shape"):
        qini_curve(medium.outcome, medium.treatment, medium.true_uplift[:-1])


def test_auuc_agrees_with_qini_on_ordering(medium: SyntheticData) -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(size=medium.n)
    assert auuc_score(medium.outcome, medium.treatment, medium.true_uplift) > auuc_score(
        medium.outcome, medium.treatment, noise
    )


def test_transformed_outcome_is_unbiased_for_the_true_effect(medium: SyntheticData) -> None:
    """E[Z] = ATE. This identity is what makes held-out validation possible."""
    z = transformed_outcome(medium.outcome, medium.treatment, 0.5)
    assert z.mean() == pytest.approx(medium.true_ate, abs=0.02)


def test_transformed_outcome_rejects_degenerate_propensity(medium: SyntheticData) -> None:
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            transformed_outcome(medium.outcome, medium.treatment, bad)


def test_transformed_outcome_loss_prefers_the_oracle(medium: SyntheticData) -> None:
    rng = np.random.default_rng(5)
    oracle = transformed_outcome_loss(medium.outcome, medium.treatment, medium.true_uplift, 0.5)
    noisy = transformed_outcome_loss(
        medium.outcome, medium.treatment, rng.normal(0, 0.2, size=medium.n), 0.5
    )
    assert oracle < noisy


def test_decile_table_has_the_expected_shape(medium: SyntheticData) -> None:
    table = uplift_by_decile(medium.outcome, medium.treatment, medium.true_uplift)
    assert len(table) == 10
    assert table["n"].sum() == medium.n
    assert (table["n_treated"] + table["n_control"] == table["n"]).all()


def test_decile_table_descends_for_an_oracle_score(medium: SyntheticData) -> None:
    """Predicted uplift must fall monotonically; measured uplift must fall on average."""
    table = uplift_by_decile(medium.outcome, medium.treatment, medium.true_uplift)
    assert np.all(np.diff(table["predicted_uplift"].to_numpy()) < 0)
    assert table["observed_uplift"].iloc[0] > table["observed_uplift"].iloc[-1]
    assert table["observed_uplift"].iloc[-1] < 0.0, "the bottom decile is genuinely harmed"


def test_bootstrap_interval_brackets_the_point_estimate(medium: SyntheticData) -> None:
    result = bootstrap_qini(medium.outcome, medium.treatment, medium.true_uplift, n_boot=60)
    assert result["ci_low"] <= result["qini_coefficient"] <= result["ci_high"]
    assert 0.0 <= result["share_beating_random"] <= 1.0


def test_beats_random_test_fires_for_an_oracle(medium: SyntheticData) -> None:
    result = bootstrap_qini(medium.outcome, medium.treatment, medium.true_uplift, n_boot=100)
    assert result["beats_random"]


def test_beats_random_test_does_not_fire_for_noise(medium: SyntheticData) -> None:
    """The check that matters: a useless model must be reported as useless."""
    rng = np.random.default_rng(6)
    failures = sum(
        not bootstrap_qini(
            medium.outcome, medium.treatment, rng.normal(size=medium.n), n_boot=60, seed=s
        )["beats_random"]
        for s in range(8)
    )
    assert failures >= 6, "a random score passed the beats-random test too often"


def test_cross_val_uplift_covers_every_row(small: SyntheticData) -> None:
    predictions = cross_val_uplift(
        SLearner(), small.features, small.treatment, small.outcome, n_folds=3
    )
    assert predictions.shape == (small.n,)
    assert np.isfinite(predictions).all()


def test_cross_val_predictions_are_worse_than_in_sample(medium: SyntheticData) -> None:
    """Out-of-fold scoring must actually cost something, or the split is leaking."""
    model = TLearner()
    in_sample = model.fit_predict(medium.features, medium.treatment, medium.outcome)
    out_of_fold = cross_val_uplift(
        TLearner(), medium.features, medium.treatment, medium.outcome, n_folds=3
    )
    assert (
        recovery_error(in_sample, medium.true_uplift)["pehe"]
        < recovery_error(out_of_fold, medium.true_uplift)["pehe"]
    )


def test_evaluate_assembles_everything(medium: SyntheticData) -> None:
    result = evaluate(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, propensity=0.5
    )
    assert result.model == "oracle"
    assert result.qini.qini_coefficient > 0
    assert result.top_decile_uplift > result.bottom_decile_uplift
    assert len(result.deciles) == 10


def test_recovery_error_is_exact_for_a_perfect_prediction(small: SyntheticData) -> None:
    result = recovery_error(small.true_uplift, small.true_uplift)
    assert result["pehe"] == pytest.approx(0.0)
    assert result["ate_bias"] == pytest.approx(0.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["sd_ratio"] == pytest.approx(1.0)
    assert result["sleeping_dogs_detected"] == pytest.approx(1.0)


def test_recovery_error_penalises_a_reversed_prediction(small: SyntheticData) -> None:
    result = recovery_error(-small.true_uplift, small.true_uplift)
    assert result["kendall_tau"] == pytest.approx(-1.0)
    assert result["sleeping_dogs_detected"] == pytest.approx(0.0)
