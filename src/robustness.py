"""Checks designed to make the model fail.

Each of these has a known right answer that does not depend on the model being
any good. If a check comes back wrong, the finding is wrong, and the memo says so
rather than quietly dropping the check.

* **Placebo.** Assign a fake treatment at random, throwing away the real one, and
  refit everything. There is now no effect to find, by construction. A model that
  still produces a healthy Qini coefficient is measuring its own overfitting.
  This is the single most informative check here and the one most often omitted.
* **Covariate balance.** Confirm the arms actually look alike. On Hillstrom they
  should, because the trial was randomised — so this is really a check on the data
  handling, and it is what would catch a botched filter or a leaked column.
* **Cost sensitivity.** The recommendation depends on an assumed cost per contact
  and margin rate, neither of which is a measurement. Sweep both and see whether
  the answer survives. A recommendation that flips between 20% and 100% targeting
  over a plausible cost range is not a recommendation.
* **Seed stability.** Refit under different random seeds. If the customers
  selected change substantially, the model is fitting noise, and the specific list
  it produces should not be trusted even if the average performance is real.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.arrays import Array
from src.config import COST_PER_CONTACT, MARGIN_RATE
from src.estimators import UpliftEstimator
from src.evaluation import _order_by_score, bootstrap_qini, cross_val_uplift, qini_curve
from src.policy import choose_policy


@dataclass(frozen=True)
class PlaceboResult:
    """Outcome of refitting on a randomly reassigned treatment.

    Attributes:
        model: Estimator name.
        n_replicates: How many fake assignments were tried.
        n_folds: Cross-validation folds used for both the real and the fake fits.
        mean_qini: Average Qini coefficient across replicates. Should be near zero.
        sd_qini: Spread across replicates, which is the natural null scale.
        real_qini: The Qini coefficient on the real treatment, refit under exactly
            the same protocol as the placebos. It will not match the headline
            leaderboard figure if that used a different number of folds, and the
            matched number is the one this comparison must use.
        z_score: How many placebo standard deviations the real result sits above
            the placebo mean. Below about 2 means the real result is inside the
            range the model produces from pure noise.
        share_placebos_exceeding_real: Empirical p-value — the fraction of fake
            assignments that scored at least as well as the real one.
        passes: True when the placebo distribution is centred near zero *and* the
            real result stands clear of it.
    """

    model: str
    n_replicates: int
    n_folds: int
    mean_qini: float
    sd_qini: float
    real_qini: float
    z_score: float
    share_placebos_exceeding_real: float
    passes: bool


def placebo_test(
    estimator: UpliftEstimator,
    features: Array,
    treatment: Array,
    outcome: Array,
    n_replicates: int = 10,
    n_folds: int = 3,
    seed: int = 0,
) -> PlaceboResult:
    """Refit the model on randomly reassigned treatment labels.

    The permutation preserves the number of treated customers but destroys any
    link between assignment and outcome, so the true uplift is exactly zero
    everywhere. Whatever Qini coefficient the model still manages to produce is
    the amount it can manufacture from noise, and it sets the bar the real result
    has to clear.

    Args:
        estimator: An unfitted estimator, refit for every replicate.
        features: Covariate matrix.
        treatment: The real assignment.
        outcome: Observed outcome.
        n_replicates: Number of fake assignments.
        n_folds: Cross-validation folds inside each replicate.
        seed: Seed.

    Returns:
        A :class:`PlaceboResult`.
    """
    rng = np.random.default_rng(seed)

    real_oof = cross_val_uplift(
        copy.deepcopy(estimator), features, treatment, outcome, n_folds=n_folds, seed=seed
    )
    real_qini = qini_curve(outcome, treatment, real_oof).qini_coefficient

    placebo_scores = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        fake = rng.permutation(treatment)
        oof = cross_val_uplift(
            copy.deepcopy(estimator), features, fake, outcome, n_folds=n_folds, seed=seed + r
        )
        placebo_scores[r] = qini_curve(outcome, fake, oof).qini_coefficient

    mean, sd = float(placebo_scores.mean()), float(placebo_scores.std(ddof=1))
    z = (real_qini - mean) / sd if sd > 0 else float("nan")
    exceed = float((placebo_scores >= real_qini).mean())

    return PlaceboResult(
        model=estimator.name,
        n_replicates=n_replicates,
        n_folds=n_folds,
        mean_qini=mean,
        sd_qini=sd,
        real_qini=float(real_qini),
        z_score=float(z),
        share_placebos_exceeding_real=exceed,
        # Two conditions, both required: the null must sit at zero (otherwise the
        # metric itself is biased) and the real result must stand outside it.
        passes=bool(abs(mean) < 2.0 * sd and exceed < 0.10),
    )


def covariate_balance(
    features: Array,
    treatment: Array,
    feature_names: list[str],
) -> pd.DataFrame:
    """Standardised mean differences for every covariate across the two arms.

    The standardised mean difference is the gap in means divided by the pooled
    standard deviation, so it is comparable across variables measured on different
    scales. The convention in the causal-inference literature is that anything
    under 0.10 in absolute value is balance and anything over 0.25 is a problem.

    A p-value is reported alongside, but with a caveat that matters: with 40,000
    customers, a trivially small imbalance will be statistically significant. The
    standardised difference is the number to read; the p-value is there because
    people ask for it.

    Args:
        features: Covariate matrix.
        treatment: Binary assignment.
        feature_names: Names matching the columns.

    Returns:
        A frame with one row per covariate, sorted by absolute imbalance.
    """
    from scipy import stats

    treated, control = treatment == 1, treatment == 0
    rows: list[dict[str, float | str]] = []
    for j, name in enumerate(feature_names):
        a, b = features[treated, j], features[control, j]
        mean_a, mean_b = float(a.mean()), float(b.mean())
        pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
        smd = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "feature": name,
                "mean_treated": mean_a,
                "mean_control": mean_b,
                "std_mean_diff": smd,
                "abs_std_mean_diff": abs(smd),
                "p_value": float(p_value),
                "balanced": float(abs(smd) < 0.10),
            }
        )
    frame: pd.DataFrame = pd.DataFrame(rows).sort_values("abs_std_mean_diff", ascending=False)
    ordered: pd.DataFrame = frame.reset_index(drop=True)
    return ordered


def balance_summary(balance: pd.DataFrame) -> dict[str, float]:
    """Reduce a balance table to the numbers the memo quotes.

    Args:
        balance: Output of :func:`covariate_balance`.

    Returns:
        Mapping with the worst imbalance, how many covariates exceed the
        thresholds, and an overall pass flag.
    """
    worst = float(balance["abs_std_mean_diff"].max())
    return {
        "n_features": float(len(balance)),
        "max_abs_smd": worst,
        "n_above_0.10": float((balance["abs_std_mean_diff"] >= 0.10).sum()),
        "n_above_0.25": float((balance["abs_std_mean_diff"] >= 0.25).sum()),
        "n_significant_at_5pct": float((balance["p_value"] < 0.05).sum()),
        "passes": float(worst < 0.10),
    }


def cost_sensitivity(
    outcome: Array,
    treatment: Array,
    score: Array,
    spend: Array,
    costs: Array | None = None,
    margins: Array | None = None,
    model: str = "model",
) -> pd.DataFrame:
    """How the recommendation moves as the cost and margin assumptions move.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        spend: Revenue per customer.
        costs: Grid of per-contact costs. Defaults to a wide sweep around the
            assumed value, from effectively free to ten times the assumption.
        margins: Grid of margin rates.
        model: Label for the output.

    Returns:
        A frame with one row per (cost, margin) pair giving the profit-maximising
        fraction and the profit there.
    """
    if costs is None:
        costs = np.array([0.01, 0.05, COST_PER_CONTACT, 0.25, 0.50, 1.00])
    if margins is None:
        margins = np.array([0.15, MARGIN_RATE, 0.50])

    rows: list[dict[str, float | str]] = []
    for cost in costs:
        for margin in margins:
            result = choose_policy(
                model,
                outcome,
                treatment,
                score,
                spend,
                budget=None,
                cost_per_contact=float(cost),
                margin_rate=float(margin),
            )
            rows.append(
                {
                    "cost_per_contact": float(cost),
                    "margin_rate": float(margin),
                    "optimal_fraction": result.optimal_fraction,
                    "optimal_profit": result.optimal_profit,
                    "treat_all_profit": result.treat_all_profit,
                    "profit_vs_random": result.profit_vs_random,
                    "beats_treat_all": float(result.optimal_profit > result.treat_all_profit),
                }
            )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def cost_sensitivity_summary(sensitivity: pd.DataFrame) -> dict[str, float]:
    """Reduce a cost-sensitivity sweep to a statement about stability.

    Args:
        sensitivity: Output of :func:`cost_sensitivity`.

    Returns:
        Mapping with the range of optimal fractions across the sweep and how often
        selective targeting beats contacting everyone.
    """
    fractions = sensitivity["optimal_fraction"]
    return {
        "min_optimal_fraction": float(fractions.min()),
        "max_optimal_fraction": float(fractions.max()),
        "median_optimal_fraction": float(fractions.median()),
        "fraction_range": float(fractions.max() - fractions.min()),
        "share_beating_treat_all": float(sensitivity["beats_treat_all"].mean()),
        "share_profitable": float((sensitivity["optimal_profit"] > 0).mean()),
    }


def seed_stability(
    estimator: UpliftEstimator,
    features: Array,
    treatment: Array,
    outcome: Array,
    spend: Array,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_folds: int = 3,
    top_fraction: float = 0.3,
    cost_per_contact: float = COST_PER_CONTACT,
    margin_rate: float = MARGIN_RATE,
) -> dict[str, float]:
    """Refit under different seeds and measure how much the answer moves.

    Two kinds of stability are reported and they can disagree. Aggregate
    performance — the Qini coefficient — can be stable while the *specific
    customers selected* churn heavily, because many customers sit near the
    cut-off and small perturbations reorder them. The overlap statistic is
    therefore the more demanding test, and it is the one that matters
    operationally: it is the answer to "if we rerun this next month, will we mail
    the same people?"

    Args:
        estimator: An unfitted estimator.
        features: Covariate matrix.
        treatment: Binary assignment.
        outcome: Observed outcome.
        spend: Revenue per customer.
        seeds: Seeds to try.
        n_folds: Cross-validation folds per seed.
        top_fraction: The selection depth whose stability is measured.
        cost_per_contact: Contact cost for the profit figures.
        margin_rate: Margin rate for the profit figures.

    Returns:
        Mapping with the spread of the Qini coefficient, of the recommended
        fraction, and the average pairwise overlap of the selected customer sets.
    """
    qinis: list[float] = []
    fractions: list[float] = []
    profits: list[float] = []
    selections: list[set[int]] = []

    n = features.shape[0]
    k = int(round(top_fraction * n))

    for seed in seeds:
        model = copy.deepcopy(estimator)
        oof = cross_val_uplift(model, features, treatment, outcome, n_folds=n_folds, seed=seed)
        qinis.append(qini_curve(outcome, treatment, oof).qini_coefficient)
        policy = choose_policy(
            estimator.name,
            outcome,
            treatment,
            oof,
            spend,
            cost_per_contact=cost_per_contact,
            margin_rate=margin_rate,
        )
        fractions.append(policy.optimal_fraction)
        profits.append(policy.optimal_profit)
        selections.append(set(_order_by_score(oof, seed=0)[:k].tolist()))

    overlaps = [
        len(a & b) / len(a | b) for i, a in enumerate(selections) for b in selections[i + 1 :]
    ]

    return {
        "n_seeds": float(len(seeds)),
        "qini_mean": float(np.mean(qinis)),
        "qini_sd": float(np.std(qinis, ddof=1)) if len(qinis) > 1 else 0.0,
        "qini_min": float(np.min(qinis)),
        "qini_max": float(np.max(qinis)),
        "optimal_fraction_mean": float(np.mean(fractions)),
        "optimal_fraction_sd": float(np.std(fractions, ddof=1)) if len(fractions) > 1 else 0.0,
        "optimal_fraction_min": float(np.min(fractions)),
        "optimal_fraction_max": float(np.max(fractions)),
        "profit_mean": float(np.mean(profits)),
        "profit_sd": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
        "mean_selection_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "min_selection_overlap": float(np.min(overlaps)) if overlaps else float("nan"),
        "top_fraction": top_fraction,
        # All seeds agreeing on the sign of the Qini coefficient is the weakest
        # useful form of stability; failing this means the direction is not established.
        "sign_consistent": float(all(q > 0 for q in qinis) or all(q < 0 for q in qinis)),
    }


def run_all_checks(
    estimator: UpliftEstimator,
    features: Array,
    treatment: Array,
    outcome: Array,
    spend: Array,
    feature_names: list[str],
    score: Array,
    n_placebo: int = 10,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_folds: int = 3,
    seed: int = 0,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Run every robustness check and collect the verdicts.

    Args:
        estimator: An unfitted estimator.
        features: Covariate matrix.
        treatment: Binary assignment.
        outcome: Observed outcome.
        spend: Revenue per customer.
        feature_names: Covariate names.
        score: Out-of-fold predicted uplift from the main run.
        n_placebo: Placebo replicates.
        seeds: Seeds for the stability check.
        n_folds: Cross-validation folds.
        seed: Seed for the placebo test, so its reference fit matches the main run.
        n_boot: Bootstrap replicates for the Qini interval.

    Returns:
        Mapping of check name to its results, plus an ``overall_pass`` flag that is
        True only when every individual check passes.
    """
    placebo = placebo_test(
        estimator,
        features,
        treatment,
        outcome,
        n_replicates=n_placebo,
        n_folds=n_folds,
        seed=seed,
    )
    balance = covariate_balance(features, treatment, feature_names)
    balance_stats = balance_summary(balance)
    sensitivity = cost_sensitivity(outcome, treatment, score, spend, model=estimator.name)
    sensitivity_stats = cost_sensitivity_summary(sensitivity)
    stability = seed_stability(
        estimator, features, treatment, outcome, spend, seeds=seeds, n_folds=n_folds
    )
    bootstrap = bootstrap_qini(outcome, treatment, score, n_boot=n_boot, seed=seed)

    return {
        "placebo": placebo,
        "balance_table": balance,
        "balance": balance_stats,
        "cost_sensitivity_table": sensitivity,
        "cost_sensitivity": sensitivity_stats,
        "seed_stability": stability,
        "bootstrap_qini": bootstrap,
        "overall_pass": bool(
            placebo.passes
            and balance_stats["passes"] > 0.5
            and stability["sign_consistent"] > 0.5
            and bootstrap["beats_random"]
        ),
    }
