"""Targeting-policy tests.

The economics are checked by hand-computable cases wherever possible, because a
profit number that is wrong in a way nobody notices is the most expensive kind of
bug in this project.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policy import (
    bootstrap_policy,
    choose_policy,
    profit_frontier,
    random_targeting_profit,
    sleeping_dog_report,
)
from src.simulate import SyntheticData


def test_frontier_covers_the_whole_range(medium: SyntheticData) -> None:
    frontier = profit_frontier(
        medium.outcome, medium.treatment, medium.true_uplift, medium.spend, n_points=20
    )
    assert len(frontier) == 20
    assert frontier["fraction"].iloc[-1] == pytest.approx(1.0)
    assert frontier["n_targeted"].iloc[-1] == medium.n
    assert (np.diff(frontier["n_targeted"].to_numpy()) > 0).all()


def test_frontier_costs_are_exactly_contacts_times_price(medium: SyntheticData) -> None:
    frontier = profit_frontier(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        cost_per_contact=0.25,
        n_points=10,
    )
    np.testing.assert_allclose(
        frontier["contact_cost"].to_numpy(), frontier["n_targeted"].to_numpy() * 0.25
    )


def test_profit_is_revenue_times_margin_less_cost(medium: SyntheticData) -> None:
    frontier = profit_frontier(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        cost_per_contact=0.10,
        margin_rate=0.30,
        n_points=10,
    )
    expected = frontier["incremental_revenue"] * 0.30 - frontier["contact_cost"]
    np.testing.assert_allclose(frontier["incremental_profit"].to_numpy(), expected.to_numpy())


def test_an_oracle_ranking_beats_random_at_the_same_depth(medium: SyntheticData) -> None:
    """The whole premise: ordering by true uplift must be worth money."""
    result = choose_policy(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, medium.spend
    )
    assert result.profit_vs_random > 0


def test_a_useless_ranking_does_not_beat_random_by_much(medium: SyntheticData) -> None:
    rng = np.random.default_rng(0)
    gains = [
        choose_policy(
            "noise",
            medium.outcome,
            medium.treatment,
            rng.normal(size=medium.n),
            medium.spend,
        ).profit_vs_random
        for _ in range(8)
    ]
    oracle = choose_policy(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, medium.spend
    ).profit_vs_random
    assert np.mean(gains) < oracle


def test_targeting_the_whole_file_equals_random_targeting_the_whole_file(
    medium: SyntheticData,
) -> None:
    """At 100% depth the ranking is irrelevant, so the two must coincide exactly."""
    frontier = profit_frontier(
        medium.outcome, medium.treatment, medium.true_uplift, medium.spend, n_points=10
    )
    treat_all = frontier["incremental_profit"].iloc[-1]
    random = random_targeting_profit(medium.outcome, medium.treatment, medium.spend, 1.0)
    assert treat_all == pytest.approx(random, rel=1e-9)


def test_a_higher_cost_shrinks_the_optimal_fraction(medium: SyntheticData) -> None:
    cheap = choose_policy(
        "oracle",
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        cost_per_contact=0.01,
    )
    dear = choose_policy(
        "oracle",
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        cost_per_contact=2.00,
    )
    assert dear.optimal_fraction <= cheap.optimal_fraction


def test_the_budget_caps_the_recommendation(medium: SyntheticData) -> None:
    budget = 0.10 * medium.n * 0.10  # enough to contact a tenth of the file
    result = choose_policy(
        "oracle",
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        budget=budget,
        cost_per_contact=0.10,
    )
    assert result.budget_cap_fraction == pytest.approx(0.10, abs=0.01)
    assert result.recommended_fraction <= result.budget_cap_fraction + 1e-9
    assert result.recommended_fraction <= result.optimal_fraction + 1e-9


def test_an_unlimited_budget_does_not_bind(medium: SyntheticData) -> None:
    result = choose_policy(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, medium.spend, budget=None
    )
    assert result.budget_cap_fraction == 1.0
    assert result.recommended_fraction == pytest.approx(result.optimal_fraction)


def test_a_zero_budget_recommends_contacting_nobody(medium: SyntheticData) -> None:
    """And the profit of contacting nobody must be zero, not the first frontier row.

    The frontier's smallest row is 1/n_points, so a nearest-row lookup on a zero
    recommendation silently returns the profit of a campaign we decided not to run.
    Asserting only the fraction leaves that wrong number unguarded.
    """
    result = choose_policy(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, medium.spend, budget=0.0
    )
    assert result.recommended_fraction == 0.0
    assert result.recommended_profit == 0.0
    assert result.profit_vs_random == 0.0


def test_a_budget_too_small_for_one_frontier_step_earns_nothing(
    medium: SyntheticData,
) -> None:
    """A budget below the frontier's resolution is the same decision as no budget."""
    result = choose_policy(
        "oracle",
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        budget=1.0,
        cost_per_contact=0.10,
    )
    assert result.budget_cap_fraction < 1.0 / 40.0
    assert result.recommended_fraction == 0.0
    assert result.recommended_profit == 0.0


def test_sleeping_dogs_are_found_and_measured(medium: SyntheticData) -> None:
    """Given a perfect ranking, the flagged group must be measurably harmed."""
    report = sleeping_dog_report(medium.outcome, medium.treatment, medium.true_uplift, medium.spend)
    assert report["n_flagged"] > 0
    assert report["mean_predicted_uplift"] < 0
    assert report["measured_uplift"] < 0
    assert report["cost_of_treating_them"] > 0, "excluding a harmed group must save money"


def test_sleeping_dog_report_handles_an_all_positive_score(medium: SyntheticData) -> None:
    report = sleeping_dog_report(medium.outcome, medium.treatment, np.ones(medium.n), medium.spend)
    assert report["n_flagged"] == 0.0
    assert report["cost_of_treating_them"] == 0.0
    assert np.isnan(report["measured_uplift"])


def test_sleeping_dog_report_keys_match_across_branches(medium: SyntheticData) -> None:
    """Both branches must return the same keys or downstream reporting breaks."""
    populated = sleeping_dog_report(
        medium.outcome, medium.treatment, medium.true_uplift, medium.spend
    )
    empty = sleeping_dog_report(medium.outcome, medium.treatment, np.ones(medium.n), medium.spend)
    assert set(populated) == set(empty)


def test_bootstrap_policy_brackets_the_point_estimate(medium: SyntheticData) -> None:
    result = bootstrap_policy(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        fraction=0.3,
        n_boot=60,
    )
    assert result["ci_low"] <= result["profit"] <= result["ci_high"]
    assert 0.0 <= result["prob_loses_money"] <= 1.0


def test_bootstrap_policy_matches_the_frontier_at_the_same_depth(
    medium: SyntheticData,
) -> None:
    """The interval must be centred on the same number the frontier reports."""
    frontier = profit_frontier(
        medium.outcome, medium.treatment, medium.true_uplift, medium.spend, n_points=10
    )
    row = frontier.iloc[int((frontier["fraction"] - 0.3).abs().idxmin())]
    result = bootstrap_policy(
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        fraction=float(row["fraction"]),
        n_boot=20,
    )
    assert result["profit"] == pytest.approx(float(row["incremental_profit"]), rel=1e-6)
