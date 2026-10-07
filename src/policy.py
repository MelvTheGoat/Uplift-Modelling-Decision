"""Turning an uplift score into a decision: whom to contact, given a budget.

A ranking is not a policy. The model produces an ordering; the decision is where
to cut it. That cut depends on three things the model knows nothing about: what a
contact costs, what an incremental sale is worth, and how much money there is.

Everything in this module measures incremental effects on *held-out experimental
data* rather than reading them off the model's own predictions. A model that
predicts a large uplift for a group is not evidence that the group has a large
uplift; the randomised comparison within that group is. This distinction is the
difference between a forecast and a measurement, and it is where uplift analyses
most often quietly cheat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.arrays import Array
from src.config import COST_PER_CONTACT, MARGIN_RATE, N_BOOTSTRAP
from src.evaluation import _order_by_score


@dataclass(frozen=True)
class FrontierPoint:
    """One row of the incremental-revenue-versus-spend frontier.

    Attributes:
        fraction: Share of the file contacted.
        n_targeted: Number of customers contacted.
        incremental_conversions: Extra conversions versus not contacting them.
        incremental_revenue: Extra revenue versus not contacting them.
        contact_cost: Total cost of the contacts.
        incremental_profit: ``incremental_revenue * margin - contact_cost``.
        profit_per_contact: Profit divided by contacts, the marginal-efficiency view.
        observed_uplift: Measured treated-minus-control response rate in the selection.
    """

    fraction: float
    n_targeted: int
    incremental_conversions: float
    incremental_revenue: float
    contact_cost: float
    incremental_profit: float
    profit_per_contact: float
    observed_uplift: float


@dataclass(frozen=True)
class PolicyResult:
    """The recommendation and the evidence behind it.

    Attributes:
        model: Which uplift model produced the ranking.
        frontier: The full frontier as a frame, one row per candidate fraction.
        optimal_fraction: Profit-maximising share of the file, ignoring any budget.
        optimal_profit: Profit at that fraction.
        budget_cap_fraction: The largest share the budget can pay to contact.
        budget_fraction: The most profitable share *within* that cap — which may be
            strictly smaller than the cap when profit turns down before the money
            runs out.
        budget_profit: Profit at ``budget_fraction``.
        recommended_fraction: The smaller of the two — what to actually do.
        recommended_profit: Profit at the recommended fraction.
        treat_all_profit: Profit from contacting the entire file.
        random_profit: Profit from contacting the same number of customers at random.
        profit_vs_random: Recommended profit minus random-targeting profit.
        profit_vs_treat_all: Recommended profit minus treat-everyone profit.
        cost_per_contact: The assumed contact cost.
        margin_rate: The assumed gross margin on incremental revenue.
    """

    model: str
    frontier: pd.DataFrame
    optimal_fraction: float
    optimal_profit: float
    budget_cap_fraction: float
    budget_fraction: float
    budget_profit: float
    recommended_fraction: float
    recommended_profit: float
    treat_all_profit: float
    random_profit: float
    profit_vs_random: float
    profit_vs_treat_all: float
    cost_per_contact: float
    margin_rate: float


def _incremental_at(
    selected: Array,
    outcome: Array,
    treatment: Array,
    spend: Array,
) -> tuple[float, float, float]:
    """Measured incremental effect within a selected set, scaled to the whole set.

    Inside the selection, the treated and control customers are still randomly
    assigned — that is the property the trial buys us, and it survives any
    selection rule that depends only on covariates. So the difference in mean
    response between the two arms *within the selection* estimates the effect of
    contacting everyone in it.

    Args:
        selected: Boolean mask of chosen customers.
        outcome: Binary outcome.
        treatment: Binary assignment.
        spend: Revenue per customer.

    Returns:
        A tuple of ``(incremental_conversions, incremental_revenue, observed_uplift)``
        where the first two are totals for the whole selected set.
    """
    n_selected = int(selected.sum())
    if n_selected == 0:
        return 0.0, 0.0, 0.0

    treated = selected & (treatment == 1)
    control = selected & (treatment == 0)
    if not treated.any() or not control.any():
        return 0.0, 0.0, 0.0

    uplift_rate = float(outcome[treated].mean() - outcome[control].mean())
    revenue_rate = float(spend[treated].mean() - spend[control].mean())
    return uplift_rate * n_selected, revenue_rate * n_selected, uplift_rate


def profit_frontier(
    outcome: Array,
    treatment: Array,
    score: Array,
    spend: Array,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
    n_points: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the incremental-revenue-versus-spend frontier.

    Customers are ranked by predicted uplift. For each candidate cut-off, the
    incremental revenue actually observed in the randomised data among the
    selected customers is compared with what it would cost to contact them.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift used for ranking.
        spend: Revenue per customer over the outcome window.
        cost_per_contact: Marginal cost of one contact.
        margin_rate: Gross margin retained on incremental revenue.
        n_points: Number of cut-offs to evaluate between 0 and 1.
        seed: Tie-breaking seed for the ranking.

    Returns:
        A frame of :class:`FrontierPoint` rows, ascending in ``fraction``.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64)
    spend = np.asarray(spend, dtype=np.float64)
    n = outcome.shape[0]

    order = _order_by_score(np.asarray(score, dtype=np.float64), seed=seed)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    fractions = np.linspace(1.0 / n_points, 1.0, n_points)
    rows: list[dict[str, float]] = []
    for fraction in fractions:
        cutoff = int(round(fraction * n))
        selected = rank < cutoff
        conversions, revenue, uplift_rate = _incremental_at(selected, outcome, treatment, spend)
        cost = cutoff * cost_per_contact
        profit = revenue * margin_rate - cost
        rows.append(
            {
                "fraction": float(fraction),
                "n_targeted": float(cutoff),
                "incremental_conversions": conversions,
                "incremental_revenue": revenue,
                "contact_cost": cost,
                "incremental_profit": profit,
                "profit_per_contact": profit / cutoff if cutoff else 0.0,
                "observed_uplift": uplift_rate,
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def random_targeting_profit(
    outcome: Array,
    treatment: Array,
    spend: Array,
    fraction: float,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
) -> float:
    """Profit from contacting a random ``fraction`` of the file.

    The honest baseline. Random selection captures the average effect on the
    fraction contacted, so its profit is linear in the fraction. An uplift model
    that cannot beat this line has not earned its keep, whatever its Qini
    coefficient says.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        spend: Revenue per customer.
        fraction: Share of the file contacted.
        cost_per_contact: Marginal cost of one contact.
        margin_rate: Gross margin on incremental revenue.

    Returns:
        Expected incremental profit.
    """
    n_targeted = int(round(fraction * outcome.shape[0]))
    treated = treatment == 1
    revenue_rate = float(spend[treated].mean() - spend[~treated].mean())
    return revenue_rate * n_targeted * margin_rate - n_targeted * cost_per_contact


def choose_policy(
    model: str,
    outcome: Array,
    treatment: Array,
    score: Array,
    spend: Array,
    budget: float | None = None,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
    n_points: int = 40,
    seed: int = 0,
) -> PolicyResult:
    """Pick a targeting fraction and report what it is worth.

    Two constraints can bind. The first is economic: past some depth into the
    ranked list, the customers being added contribute less incremental margin
    than they cost, and profit turns down. The second is the budget. The
    recommendation is the tighter of the two, which means that when the budget
    binds the correct answer is "spend it all on the top of the list", and when
    economics bind first the correct answer is "do not spend the whole budget".

    Args:
        model: Name of the model that produced the ranking.
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        spend: Revenue per customer.
        budget: Total money available for contacts. ``None`` means unconstrained.
        cost_per_contact: Marginal cost of one contact.
        margin_rate: Gross margin on incremental revenue.
        n_points: Frontier resolution.
        seed: Tie-breaking seed.

    Returns:
        A :class:`PolicyResult`.
    """
    frontier = profit_frontier(
        outcome, treatment, score, spend, cost_per_contact, margin_rate, n_points, seed
    )
    n = outcome.shape[0]

    # `argmax`/`argmin` on the underlying array rather than `idxmax`/`idxmin` on
    # the frame. The pandas versions return an index *label*, which only
    # coincides with a row position while the frame carries a default
    # RangeIndex — true here, but `affordable` below is a filtered view, so one
    # careless `.iloc` on a label would silently read the wrong row.
    best = int(frontier["incremental_profit"].to_numpy().argmax())
    optimal_fraction = float(frontier["fraction"].iloc[best])
    optimal_profit = float(frontier["incremental_profit"].iloc[best])

    budget_cap_fraction = 1.0 if budget is None else min(1.0, budget / (cost_per_contact * n))
    affordable = frontier[frontier["fraction"] <= budget_cap_fraction + 1e-9]
    if affordable.empty:
        budget_profit = 0.0
        budget_fraction = 0.0
    else:
        best_affordable = int(affordable["incremental_profit"].to_numpy().argmax())
        budget_fraction = float(affordable["fraction"].iloc[best_affordable])
        budget_profit = float(affordable["incremental_profit"].iloc[best_affordable])

    recommended_fraction = min(optimal_fraction, budget_fraction)
    if recommended_fraction <= 0.0:
        # Contacting nobody earns nothing. This needs saying explicitly because the
        # frontier's smallest row is 1/n_points, not 0 — so the nearest-row lookup
        # below would snap a zero recommendation onto the first row and report its
        # profit, i.e. money earned by a campaign we just decided not to run.
        recommended_profit = 0.0
    else:
        # Both `optimal_fraction` and `budget_fraction` are read off the frontier,
        # so their minimum is always exactly a row value and this lookup is a
        # lookup rather than an approximation.
        nearest = int((frontier["fraction"] - recommended_fraction).abs().to_numpy().argmin())
        recommended_profit = float(frontier["incremental_profit"].iloc[nearest])

    treat_all_profit = float(frontier.iloc[-1]["incremental_profit"])
    random_profit = random_targeting_profit(
        outcome, treatment, spend, recommended_fraction, cost_per_contact, margin_rate
    )

    return PolicyResult(
        model=model,
        frontier=frontier,
        optimal_fraction=optimal_fraction,
        optimal_profit=optimal_profit,
        budget_cap_fraction=budget_cap_fraction,
        budget_fraction=budget_fraction,
        budget_profit=budget_profit,
        recommended_fraction=recommended_fraction,
        recommended_profit=recommended_profit,
        treat_all_profit=treat_all_profit,
        random_profit=random_profit,
        profit_vs_random=recommended_profit - random_profit,
        profit_vs_treat_all=recommended_profit - treat_all_profit,
        cost_per_contact=cost_per_contact,
        margin_rate=margin_rate,
    )


def sleeping_dog_report(
    outcome: Array,
    treatment: Array,
    score: Array,
    spend: Array,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Quantify the customers the model says are harmed by being contacted.

    "Sleeping dogs" are customers whose response *falls* when they are contacted —
    the unsubscribe, the annoyance, the reminder to cancel. They are invisible to a
    response model, which can only rank people by how likely they are to buy, and
    they are the single clearest argument for uplift modelling: they are the one
    group where the correct action is the opposite of the intuitive one.

    Two numbers matter and they are different. The *predicted* group is whom the
    model would exclude. The *measured* uplift inside that group is whether the
    model was right. A model can flag 20% of the file as sleeping dogs and be
    wrong about all of them; the confidence interval below is what settles it.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        spend: Revenue per customer.
        cost_per_contact: Marginal cost of one contact.
        margin_rate: Gross margin on incremental revenue.
        threshold: Score below which a customer is called a sleeping dog.

    Returns:
        Mapping describing the flagged group, the effect measured inside it, and
        the profit consequence of contacting them anyway.
    """
    score = np.asarray(score, dtype=np.float64)
    dogs = score < threshold
    n_dogs = int(dogs.sum())
    if n_dogs == 0:
        # The model found nobody it expects to be harmed. That is a finding, not an
        # error: it means the estimator predicts a non-negative effect everywhere.
        return {
            "n_flagged": 0.0,
            "share_flagged": 0.0,
            "mean_predicted_uplift": float("nan"),
            "measured_uplift": float("nan"),
            "measured_uplift_stderr": float("nan"),
            "measured_uplift_ci_low": float("nan"),
            "measured_uplift_ci_high": float("nan"),
            "measured_uplift_is_negative": 0.0,
            "incremental_conversions_if_treated": 0.0,
            "incremental_revenue_if_treated": 0.0,
            "cost_if_treated": 0.0,
            "profit_if_treated": 0.0,
            "cost_of_treating_them": 0.0,
        }

    treated = dogs & (treatment == 1)
    control = dogs & (treatment == 0)
    rate_t = float(outcome[treated].mean()) if treated.any() else float("nan")
    rate_c = float(outcome[control].mean()) if control.any() else float("nan")
    measured = rate_t - rate_c
    stderr = float(
        np.sqrt(
            rate_t * (1 - rate_t) / max(int(treated.sum()), 1)
            + rate_c * (1 - rate_c) / max(int(control.sum()), 1)
        )
    )

    conversions, revenue, _ = _incremental_at(dogs, outcome, treatment, spend)
    cost = n_dogs * cost_per_contact
    profit = revenue * margin_rate - cost

    return {
        "n_flagged": float(n_dogs),
        "share_flagged": float(dogs.mean()),
        "mean_predicted_uplift": float(score[dogs].mean()),
        "measured_uplift": measured,
        "measured_uplift_stderr": stderr,
        "measured_uplift_ci_low": measured - 1.96 * stderr,
        "measured_uplift_ci_high": measured + 1.96 * stderr,
        # 1.0 when the confidence interval sits entirely below zero, i.e. the harm
        # is measurable rather than merely predicted.
        "measured_uplift_is_negative": float((measured + 1.96 * stderr) < 0.0),
        "incremental_conversions_if_treated": conversions,
        "incremental_revenue_if_treated": revenue,
        "cost_if_treated": cost,
        "profit_if_treated": profit,
        # Positive means excluding them makes money.
        "cost_of_treating_them": -profit,
    }


def bootstrap_policy(
    outcome: Array,
    treatment: Array,
    score: Array,
    spend: Array,
    fraction: float,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Confidence interval for the profit of a fixed targeting fraction.

    The fraction is held fixed rather than re-optimised inside each replicate.
    Re-optimising would answer a different and less useful question — the spread
    of the best achievable profit under hindsight — and would understate the
    uncertainty in the number actually being proposed.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        spend: Revenue per customer.
        fraction: The targeting fraction being evaluated.
        cost_per_contact: Marginal cost of one contact.
        margin_rate: Gross margin on incremental revenue.
        n_boot: Bootstrap replicates.
        seed: Seed.
        alpha: Two-sided level.

    Returns:
        Mapping with the point estimate, interval, and the probability that the
        campaign at this fraction loses money.
    """
    rng = np.random.default_rng(seed)
    n = outcome.shape[0]
    treated_idx = np.flatnonzero(treatment == 1)
    control_idx = np.flatnonzero(treatment == 0)

    order = _order_by_score(np.asarray(score, dtype=np.float64), seed=seed)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    cutoff = int(round(fraction * n))
    selected_mask = rank < cutoff

    def _profit(idx: Array) -> float:
        chosen = selected_mask[idx]
        if not chosen.any():
            return 0.0
        sub_t = chosen & (treatment[idx] == 1)
        sub_c = chosen & (treatment[idx] == 0)
        if not sub_t.any() or not sub_c.any():
            return float("nan")
        revenue_rate = float(spend[idx][sub_t].mean() - spend[idx][sub_c].mean())
        n_selected = int(chosen.sum())
        return revenue_rate * n_selected * margin_rate - n_selected * cost_per_contact

    point = _profit(np.arange(n))
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = np.concatenate(
            [
                rng.choice(treated_idx, size=treated_idx.size, replace=True),
                rng.choice(control_idx, size=control_idx.size, replace=True),
            ]
        )
        draws[b] = _profit(pick)

    finite = draws[np.isfinite(draws)]
    return {
        "fraction": float(fraction),
        "profit": float(point),
        "ci_low": float(np.quantile(finite, alpha / 2)) if finite.size else float("nan"),
        "ci_high": float(np.quantile(finite, 1 - alpha / 2)) if finite.size else float("nan"),
        "prob_loses_money": float((finite < 0).mean()) if finite.size else float("nan"),
        "n_bootstrap": float(finite.size),
    }
