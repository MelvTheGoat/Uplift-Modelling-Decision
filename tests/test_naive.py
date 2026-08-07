"""Tests for the naive analysis and the two demonstrations built on it."""

from __future__ import annotations

import numpy as np
import pytest

from src.naive import (
    compare_randomised_and_confounded,
    naive_comparison,
    response_versus_uplift_ranking,
    run_naive_analysis,
)
from src.simulate import SyntheticData


def test_naive_comparison_recovers_the_ate_under_randomisation(medium: SyntheticData) -> None:
    """This is the one thing the naive calculation gets right, and the memo says so."""
    result = naive_comparison(medium.outcome, medium.treatment, medium.spend)
    assert result.absolute_difference == pytest.approx(medium.true_ate, abs=0.015)
    assert result.ci_low < medium.true_ate < result.ci_high


def test_naive_comparison_arithmetic_is_internally_consistent(medium: SyntheticData) -> None:
    result = naive_comparison(medium.outcome, medium.treatment, medium.spend)
    assert result.absolute_difference == pytest.approx(result.rate_treated - result.rate_control)
    assert result.relative_lift == pytest.approx(result.absolute_difference / result.rate_control)
    assert result.n_treated + result.n_control == medium.n
    assert result.revenue_difference == pytest.approx(
        result.revenue_treated - result.revenue_control
    )


def test_naive_comparison_detects_a_real_effect(medium: SyntheticData) -> None:
    result = naive_comparison(medium.outcome, medium.treatment)
    assert result.p_value < 0.01


def test_naive_comparison_requires_both_arms(medium: SyntheticData) -> None:
    with pytest.raises(ValueError, match="both arms"):
        naive_comparison(medium.outcome, np.ones_like(medium.treatment))


def test_confounding_inflates_the_naive_gap() -> None:
    """The headline number of the memo's second section."""
    result = compare_randomised_and_confounded(n=20_000, seed=1)
    assert abs(result["bias_randomised"]) < 0.01
    assert result["bias_confounded"] > 0.05
    assert result["overstatement_factor"] > 3.0


def test_no_confounding_means_no_inflation() -> None:
    result = compare_randomised_and_confounded(n=20_000, seed=1, confounding_strength=0.0)
    assert result["naive_gap_confounded"] == pytest.approx(result["true_ate"], abs=0.015)


def test_uplift_ranking_beats_response_ranking_on_true_incremental_effect() -> None:
    """The core claim: ranking by likelihood to convert is the wrong ranking."""
    result = response_versus_uplift_ranking(n=20_000, seed=1)
    assert (
        result["true_uplift_captured_by_uplift_model"]
        > result["true_uplift_captured_by_response_model"]
    )
    assert (
        result["true_uplift_captured_by_response_model"] > result["true_uplift_captured_by_random"]
    )
    assert (
        result["true_uplift_captured_by_oracle"] >= result["true_uplift_captured_by_uplift_model"]
    )


def test_response_ranking_contacts_more_sleeping_dogs() -> None:
    """A response model cannot see harm, so it mails the harmed group anyway."""
    result = response_versus_uplift_ranking(n=20_000, seed=1)
    assert (
        result["sleeping_dogs_contacted_by_response_model"]
        > result["sleeping_dogs_contacted_by_uplift_model"]
    )


def test_uplift_ranking_tracks_the_truth_more_closely() -> None:
    result = response_versus_uplift_ranking(n=20_000, seed=1)
    assert result["rank_correlation_uplift_vs_truth"] > result["rank_correlation_response_vs_truth"]


def test_run_naive_analysis_assembles_both_sections(small: SyntheticData) -> None:
    result = run_naive_analysis(small, seed=1)
    assert set(result) == {"confounding", "ranking"}
