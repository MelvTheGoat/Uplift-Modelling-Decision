"""Robustness-check tests.

These verify that the checks detect what they are supposed to detect. A placebo
test that never fails is worse than no placebo test, so the important cases here
are the ones where a check must return a negative verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators import SLearner, TLearner
from src.robustness import (
    balance_summary,
    cost_sensitivity,
    cost_sensitivity_summary,
    covariate_balance,
    placebo_test,
    run_all_checks,
    seed_stability,
)
from src.simulate import SyntheticData


def test_balance_passes_on_a_randomised_trial(medium: SyntheticData) -> None:
    balance = covariate_balance(medium.features, medium.treatment, medium.feature_names)
    summary = balance_summary(balance)
    assert len(balance) == len(medium.feature_names)
    assert summary["max_abs_smd"] < 0.10
    assert summary["passes"] == 1.0


def test_balance_fails_when_assignment_depends_on_covariates(
    confounded: SyntheticData,
) -> None:
    """The check must actually catch a broken design, not just bless a good one."""
    balance = covariate_balance(confounded.features, confounded.treatment, confounded.feature_names)
    summary = balance_summary(balance)
    assert summary["max_abs_smd"] > 0.25
    assert summary["passes"] == 0.0
    # And it must finger the right column: x3 is what drives assignment.
    assert balance.iloc[0]["feature"] == "x3"


def test_balance_table_is_sorted_by_severity(confounded: SyntheticData) -> None:
    balance = covariate_balance(confounded.features, confounded.treatment, confounded.feature_names)
    values = balance["abs_std_mean_diff"].to_numpy()
    assert np.all(np.diff(values) <= 0)


def test_placebo_null_is_centred_on_zero(medium: SyntheticData) -> None:
    """With treatment permuted there is no effect, so the null must sit at zero."""
    result = placebo_test(
        SLearner(), medium.features, medium.treatment, medium.outcome, n_replicates=5, n_folds=3
    )
    assert abs(result.mean_qini) < 2.0 * result.sd_qini
    assert result.n_replicates == 5
    assert result.n_folds == 3


def test_placebo_passes_when_there_is_real_signal(medium: SyntheticData) -> None:
    result = placebo_test(
        TLearner(), medium.features, medium.treatment, medium.outcome, n_replicates=6, n_folds=3
    )
    assert result.real_qini > result.mean_qini
    assert result.passes


def test_placebo_fails_when_the_outcome_carries_no_treatment_effect(
    medium: SyntheticData,
) -> None:
    """Feed the check a dataset with a genuinely null effect; it must say so.

    The outcome is replaced with the control-world potential outcome for everyone,
    so treatment does nothing by construction while the covariates still predict
    the outcome strongly. A model can still fit the outcome; it has no effect to find.
    """
    result = placebo_test(
        TLearner(),
        medium.features,
        medium.treatment,
        medium.outcome_if_control,
        n_replicates=6,
        n_folds=3,
    )
    assert not result.passes


def test_cost_sensitivity_sweeps_the_grid(medium: SyntheticData) -> None:
    sweep = cost_sensitivity(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        costs=np.array([0.05, 0.50]),
        margins=np.array([0.30]),
    )
    assert len(sweep) == 2
    assert set(sweep.columns) >= {"cost_per_contact", "margin_rate", "optimal_fraction"}


def test_a_higher_cost_narrows_targeting_across_the_sweep(medium: SyntheticData) -> None:
    sweep = cost_sensitivity(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        costs=np.array([0.01, 1.00]),
        margins=np.array([0.30]),
    )
    cheap = sweep[sweep["cost_per_contact"] == 0.01]["optimal_fraction"].iloc[0]
    dear = sweep[sweep["cost_per_contact"] == 1.00]["optimal_fraction"].iloc[0]
    assert dear <= cheap


def test_cost_sensitivity_summary_reports_the_range(medium: SyntheticData) -> None:
    sweep = cost_sensitivity(medium.outcome, medium.treatment, medium.true_uplift, medium.spend)
    summary = cost_sensitivity_summary(sweep)
    assert summary["min_optimal_fraction"] <= summary["median_optimal_fraction"]
    assert summary["median_optimal_fraction"] <= summary["max_optimal_fraction"]
    assert summary["fraction_range"] == pytest.approx(
        summary["max_optimal_fraction"] - summary["min_optimal_fraction"]
    )


def test_seed_stability_reports_spread_and_overlap(small: SyntheticData) -> None:
    result = seed_stability(
        TLearner(),
        small.features,
        small.treatment,
        small.outcome,
        small.spend,
        seeds=(0, 1, 2),
        n_folds=3,
    )
    assert result["n_seeds"] == 3.0
    assert result["qini_min"] <= result["qini_mean"] <= result["qini_max"]
    assert 0.0 <= result["mean_selection_overlap"] <= 1.0
    assert result["min_selection_overlap"] <= result["mean_selection_overlap"]


def test_seed_stability_overlap_is_one_when_the_score_cannot_change(
    small: SyntheticData,
) -> None:
    """A single seed repeated must select an identical list, or the metric is broken."""
    result = seed_stability(
        TLearner(),
        small.features,
        small.treatment,
        small.outcome,
        small.spend,
        seeds=(4, 4),
        n_folds=3,
    )
    assert result["mean_selection_overlap"] == pytest.approx(1.0)
    assert result["qini_sd"] == pytest.approx(0.0, abs=1e-9)


def test_run_all_checks_assembles_every_verdict(small: SyntheticData) -> None:
    checks = run_all_checks(
        TLearner(),
        small.features,
        small.treatment,
        small.outcome,
        small.spend,
        small.feature_names,
        small.true_uplift,
        n_placebo=3,
        seeds=(0, 1),
        n_folds=3,
        n_boot=40,
    )
    assert set(checks) >= {
        "placebo",
        "balance",
        "balance_table",
        "cost_sensitivity",
        "cost_sensitivity_table",
        "seed_stability",
        "bootstrap_qini",
        "overall_pass",
    }
    assert isinstance(checks["overall_pass"], bool)
