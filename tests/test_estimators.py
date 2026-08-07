"""Estimator tests, all against known synthetic truth.

The behavioural claims made in the module docstring of ``src.estimators`` are
asserted here rather than merely stated — in particular that S-learner shrinks
the spread of estimated effects toward zero and that T-learner inflates it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators import (
    CausalForestLearner,
    LearnerFactory,
    SLearner,
    TLearner,
    UpliftEstimator,
    XLearner,
    build_estimators,
)
from src.evaluation import cross_val_uplift, recovery_error
from src.simulate import SyntheticData

META_LEARNERS = [SLearner, TLearner, XLearner]


@pytest.mark.parametrize("cls", META_LEARNERS)
def test_predict_before_fit_raises(cls: type[UpliftEstimator]) -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        cls().predict_uplift(np.zeros((5, 4)))


@pytest.mark.parametrize("cls", META_LEARNERS)
def test_fit_returns_self_and_predicts_the_right_shape(
    cls: type[UpliftEstimator], small: SyntheticData, fast_factory: LearnerFactory
) -> None:
    model = cls(fast_factory)  # type: ignore[call-arg]
    returned = model.fit(small.features, small.treatment, small.outcome)
    assert returned is model
    predictions = model.predict_uplift(small.features)
    assert predictions.shape == (small.n,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("cls", META_LEARNERS)
def test_estimated_ate_is_close_to_the_truth(
    cls: type[UpliftEstimator], medium: SyntheticData, fast_factory: LearnerFactory
) -> None:
    """The average effect is the easy part; every estimator must get it roughly right."""
    model = cls(fast_factory)  # type: ignore[call-arg]
    predicted = model.fit_predict(medium.features, medium.treatment, medium.outcome)
    assert predicted.mean() == pytest.approx(medium.true_ate, abs=0.015)


@pytest.mark.parametrize("cls", META_LEARNERS)
def test_predictions_correlate_with_the_true_effect(
    cls: type[UpliftEstimator], medium: SyntheticData, fast_factory: LearnerFactory
) -> None:
    model = cls(fast_factory)  # type: ignore[call-arg]
    predicted = model.fit_predict(medium.features, medium.treatment, medium.outcome)
    assert recovery_error(predicted, medium.true_uplift)["kendall_tau"] > 0.15


@pytest.mark.parametrize("cls", META_LEARNERS)
def test_sleeping_dogs_are_ranked_below_everyone_else(
    cls: type[UpliftEstimator], medium: SyntheticData, fast_factory: LearnerFactory
) -> None:
    """The core capability: a harmed subgroup must score lower than the rest."""
    model = cls(fast_factory)  # type: ignore[call-arg]
    predicted = model.fit_predict(medium.features, medium.treatment, medium.outcome)
    dogs = medium.sleeping_dog_mask
    assert predicted[dogs].mean() < predicted[~dogs].mean()


def test_s_learner_shrinks_and_t_learner_spreads(medium: SyntheticData) -> None:
    """The two documented failure modes, asserted rather than assumed.

    Both are properties of *out-of-sample* predictions, which is why this test
    cross-validates instead of fitting in sample: in sample, both learners can
    reproduce the training arms closely enough to hide the effect.

    S-learner offers the base model one binary treatment column competing against
    strong outcome predictors; a regularised booster uses it sparingly, so the
    spread of predicted effects comes out narrower than the truth. T-learner
    differences two independently estimated surfaces, so their error terms add and
    the spread comes out wider. The ordering between them is the stable claim and
    it is the one the memo relies on.
    """
    args = (medium.features, medium.treatment, medium.outcome)
    s_ratio = cross_val_uplift(SLearner(), *args, n_folds=3).std() / medium.true_uplift.std()
    t_ratio = cross_val_uplift(TLearner(), *args, n_folds=3).std() / medium.true_uplift.std()

    assert s_ratio < 1.0, "S-learner should shrink the spread of effects toward zero"
    assert t_ratio > 1.0, "T-learner should inflate the spread of effects"
    assert s_ratio < t_ratio


def test_x_learner_beats_t_learner_under_imbalance(
    imbalanced: SyntheticData, fast_factory: LearnerFactory
) -> None:
    """X-learner's stated advantage, tested at a 15/85 split.

    Both learners start from the same two outcome models. X-learner's second stage
    models the imputed effect directly and leans on the arm estimated from more
    data, and should therefore recover the individual effects more accurately.
    """
    args = (imbalanced.features, imbalanced.treatment, imbalanced.outcome)
    t_pehe = recovery_error(TLearner(fast_factory).fit_predict(*args), imbalanced.true_uplift)[
        "pehe"
    ]
    x_pehe = recovery_error(
        XLearner(fast_factory, propensity=0.15).fit_predict(*args), imbalanced.true_uplift
    )["pehe"]
    assert x_pehe < t_pehe


def test_x_learner_accepts_a_known_propensity_without_fitting_one(
    small: SyntheticData, fast_factory: LearnerFactory
) -> None:
    model = XLearner(fast_factory, propensity=0.5).fit(
        small.features, small.treatment, small.outcome
    )
    assert model.propensity_model_ is None


def test_x_learner_estimates_propensity_when_not_given(
    small: SyntheticData, fast_factory: LearnerFactory
) -> None:
    model = XLearner(fast_factory, propensity=None).fit(
        small.features, small.treatment, small.outcome
    )
    assert model.propensity_model_ is not None


@pytest.mark.parametrize("kind", ["lightgbm", "hgb", "forest", "linear"])
def test_every_base_learner_family_works(kind: str, small: SyntheticData) -> None:
    factory = LearnerFactory(kind=kind, random_state=0)
    predicted = TLearner(factory).fit_predict(small.features, small.treatment, small.outcome)
    assert predicted.shape == (small.n,)
    assert np.isfinite(predicted).all()


def test_unknown_base_learner_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        LearnerFactory(kind="magic")


def test_estimators_are_deterministic_given_a_seed(
    small: SyntheticData, fast_factory: LearnerFactory
) -> None:
    args = (small.features, small.treatment, small.outcome)
    first = TLearner(fast_factory).fit_predict(*args)
    second = TLearner(LearnerFactory("lightgbm", 0, n_estimators=40, num_leaves=7)).fit_predict(
        *args
    )
    np.testing.assert_allclose(first, second)


def test_regression_mode_handles_a_continuous_outcome(
    small: SyntheticData, fast_factory: LearnerFactory
) -> None:
    """Uplift on revenue, not just conversion."""
    model = TLearner(fast_factory, classifier=False)
    predicted = model.fit_predict(small.features, small.treatment, small.spend)
    assert predicted.shape == (small.n,)
    assert np.isfinite(predicted).all()


def test_build_estimators_returns_the_expected_slate() -> None:
    slate = build_estimators(include_causal_forest=False)
    assert set(slate) == {"s-learner", "t-learner", "x-learner"}
    assert set(build_estimators(include_causal_forest=True)) == {
        "s-learner",
        "t-learner",
        "x-learner",
        "causal-forest",
    }


@pytest.mark.slow
def test_causal_forest_recovers_the_effect(small: SyntheticData) -> None:
    model = CausalForestLearner(n_estimators=100, min_samples_leaf=40, random_state=0)
    predicted = model.fit_predict(small.features, small.treatment, small.outcome)
    assert predicted.shape == (small.n,)
    assert predicted.mean() == pytest.approx(small.true_ate, abs=0.03)
    assert recovery_error(predicted, small.true_uplift)["kendall_tau"] > 0.10
