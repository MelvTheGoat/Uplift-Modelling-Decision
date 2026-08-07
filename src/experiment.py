"""Designing the experiment that would confirm or kill the recommendation.

A model fitted to a 2008 mailing is a hypothesis, not a result. This module sizes
the test that would settle it, and shows how to make that test cheaper using a
pre-period covariate.

Two things are computed here:

* **Power and minimum detectable effect.** Given a baseline conversion rate and
  the smallest effect worth acting on, how many customers per arm are needed —
  and, run backwards, given the customers available, what is the smallest effect
  the test could actually detect. The second direction is the one that stops
  underpowered tests from being run, because it converts "we have 40,000
  customers" into "so we can only detect a lift of X", which is often visibly
  larger than any lift anyone expects.

* **CUPED.** Controlled-experiment Using Pre-Experiment Data: subtract off the
  part of each customer's outcome that was predictable before the experiment
  started. The treatment effect is untouched; the variance falls. Same test, same
  answer, fewer customers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.arrays import Array
from src.simulate import simulate


@dataclass(frozen=True)
class PowerResult:
    """Sample size required for a two-proportion test.

    Attributes:
        baseline_rate: Assumed control conversion rate.
        treatment_rate: Implied treated rate under the alternative.
        absolute_mde: Effect size in percentage points.
        relative_mde: Effect size as a fraction of the baseline.
        alpha: Two-sided significance level.
        power: Desired power.
        n_per_arm: Customers needed in each arm.
        n_total: Total customers needed.
        expected_conversions_per_arm: Converters expected per arm, a sanity check —
            a design needing fewer than a few dozen is fragile whatever the formula says.
    """

    baseline_rate: float
    treatment_rate: float
    absolute_mde: float
    relative_mde: float
    alpha: float
    power: float
    n_per_arm: int
    n_total: int
    expected_conversions_per_arm: float


def required_sample_size(
    baseline_rate: float,
    absolute_mde: float | None = None,
    relative_mde: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
) -> PowerResult:
    """Customers needed per arm to detect a given lift in a conversion rate.

    Uses the standard normal approximation for the difference of two proportions::

        n = (z_{1-a/2} * sqrt(2 * p_bar * (1 - p_bar))
             + z_{power} * sqrt(p0(1-p0) + p1(1-p1)))^2 / delta^2

    The approximation is reliable when the expected number of conversions per arm
    is comfortably into double figures, which the returned
    ``expected_conversions_per_arm`` lets the reader check.

    Args:
        baseline_rate: Control conversion rate, in (0, 1).
        absolute_mde: Smallest effect worth detecting, in percentage points
            (0.005 = half a point). Mutually exclusive with ``relative_mde``.
        relative_mde: Smallest effect as a fraction of baseline (0.20 = a 20% lift).
        alpha: Two-sided significance level.
        power: Desired power.
        allocation_ratio: Treated customers per control customer. 1.0 is equal
            arms and is the most efficient design for a fixed total.

    Returns:
        A :class:`PowerResult`.

    Raises:
        ValueError: If neither or both effect sizes are given, or inputs are out of range.
    """
    if (absolute_mde is None) == (relative_mde is None):
        raise ValueError("supply exactly one of absolute_mde or relative_mde")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must lie strictly between 0 and 1")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie strictly between 0 and 1")
    if allocation_ratio <= 0.0:
        raise ValueError("allocation_ratio must be positive")

    if absolute_mde is not None:
        delta = absolute_mde
    elif relative_mde is not None:
        delta = baseline_rate * relative_mde
    else:  # pragma: no cover - excluded by the check above
        raise ValueError("supply exactly one of absolute_mde or relative_mde")
    if delta <= 0.0:
        raise ValueError("the effect size must be positive")

    p0 = baseline_rate
    p1 = min(p0 + delta, 1.0 - 1e-9)
    k = allocation_ratio
    p_bar = (p0 + k * p1) / (1.0 + k)

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))

    numerator = (
        z_alpha * np.sqrt((1.0 + 1.0 / k) * p_bar * (1.0 - p_bar))
        + z_power * np.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1) / k)
    ) ** 2
    n_control = int(np.ceil(numerator / delta**2))
    n_treated = int(np.ceil(k * n_control))

    return PowerResult(
        baseline_rate=p0,
        treatment_rate=p1,
        absolute_mde=delta,
        relative_mde=delta / p0,
        alpha=alpha,
        power=power,
        n_per_arm=n_control,
        n_total=n_control + n_treated,
        expected_conversions_per_arm=n_control * p0,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    """The inverse: given the customers available, what can the test actually detect.

    Solved numerically by bisection on :func:`required_sample_size`, which keeps
    the two directions guaranteed consistent rather than relying on a separately
    derived closed form that might drift.

    Args:
        baseline_rate: Control conversion rate.
        n_per_arm: Customers available in each arm.
        alpha: Two-sided significance level.
        power: Desired power.

    Returns:
        Mapping with the absolute and relative minimum detectable effect and the
        treated rate it corresponds to.

    Raises:
        ValueError: If ``n_per_arm`` is not positive.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")

    low, high = 1e-6, min(0.5, 1.0 - baseline_rate - 1e-6)
    for _ in range(200):
        mid = 0.5 * (low + high)
        needed = required_sample_size(baseline_rate, absolute_mde=mid, alpha=alpha, power=power)
        if needed.n_per_arm > n_per_arm:
            low = mid  # too small an effect to detect: need a bigger one
        else:
            high = mid
    absolute = high
    return {
        "baseline_rate": baseline_rate,
        "n_per_arm": float(n_per_arm),
        "absolute_mde": absolute,
        "relative_mde": absolute / baseline_rate,
        "detectable_treatment_rate": baseline_rate + absolute,
        "alpha": alpha,
        "power": power,
    }


@dataclass(frozen=True)
class CupedResult:
    """Variance reduction achieved by adjusting for a pre-period covariate.

    Attributes:
        theta: The regression coefficient subtracted off.
        correlation: Correlation between the outcome and the pre-period covariate.
        variance_raw: Variance of the unadjusted outcome.
        variance_adjusted: Variance after adjustment.
        variance_reduction: Fractional reduction, which equals the squared
            correlation in the population.
        effective_sample_multiplier: How many times more customers the unadjusted
            test would need for the same precision.
        ate_raw: Treatment effect estimated without adjustment.
        ate_adjusted: Treatment effect estimated with adjustment. Should match
            ``ate_raw`` closely — CUPED must not move the estimate, only its error.
        stderr_raw: Standard error without adjustment.
        stderr_adjusted: Standard error with adjustment.
    """

    theta: float
    correlation: float
    variance_raw: float
    variance_adjusted: float
    variance_reduction: float
    effective_sample_multiplier: float
    ate_raw: float
    ate_adjusted: float
    stderr_raw: float
    stderr_adjusted: float


def cuped_adjust(
    outcome: Array,
    pre_period: Array,
    treatment: Array,
) -> tuple[Array, CupedResult]:
    """Apply CUPED and report the variance reduction.

    The adjustment is::

        Y_adjusted = Y - theta * (X_pre - mean(X_pre)),  theta = Cov(Y, X_pre) / Var(X_pre)

    Assumptions, all of which must hold or the method is either useless or wrong:

    1. **The covariate is measured strictly before assignment.** This is the load-
       bearing one. If the treatment can influence the covariate, subtracting it
       removes part of the effect being measured and biases the estimate toward
       zero. A "pre-period" that overlaps the campaign window is the usual way
       this goes wrong in practice.
    2. **The covariate correlates with the outcome.** The variance reduction is
       exactly the squared correlation, so a covariate correlated 0.3 with the
       outcome buys 9% — not worth the complexity. It is worth doing at 0.5 and
       transformative at 0.8.
    3. **Theta is estimated on pooled data.** Under randomisation this is
       harmless; the estimate is consistent and the finite-sample bias is of order
       1/n. Estimating theta separately per arm re-introduces the difference the
       test is trying to measure.
    4. **CUPED reduces variance, never bias.** It cannot rescue a broken
       randomisation, and it does nothing about confounding.

    Args:
        outcome: Observed outcome.
        pre_period: Pre-treatment covariate.
        treatment: Binary assignment.

    Returns:
        A tuple of the adjusted outcome and a :class:`CupedResult`.

    Raises:
        ValueError: If the pre-period covariate has no variance.
    """
    y = np.asarray(outcome, dtype=np.float64)
    x = np.asarray(pre_period, dtype=np.float64)
    w = np.asarray(treatment, dtype=np.int64)

    var_x = float(x.var())
    if var_x <= 0.0:
        raise ValueError("pre_period covariate has zero variance; nothing to adjust for")

    theta = float(np.cov(y, x, bias=True)[0, 1] / var_x)
    adjusted = y - theta * (x - x.mean())

    treated, control = w == 1, w == 0

    def _effect(values: Array) -> tuple[float, float]:
        effect = float(values[treated].mean() - values[control].mean())
        stderr = float(
            np.sqrt(
                values[treated].var(ddof=1) / treated.sum()
                + values[control].var(ddof=1) / control.sum()
            )
        )
        return effect, stderr

    ate_raw, se_raw = _effect(y)
    ate_adjusted, se_adjusted = _effect(adjusted)

    var_raw, var_adjusted = float(y.var()), float(adjusted.var())
    reduction = 1.0 - var_adjusted / var_raw if var_raw > 0 else 0.0

    return adjusted, CupedResult(
        theta=theta,
        correlation=float(np.corrcoef(y, x)[0, 1]),
        variance_raw=var_raw,
        variance_adjusted=var_adjusted,
        variance_reduction=reduction,
        effective_sample_multiplier=1.0 / (1.0 - reduction) if reduction < 1.0 else float("inf"),
        ate_raw=ate_raw,
        ate_adjusted=ate_adjusted,
        stderr_raw=se_raw,
        stderr_adjusted=se_adjusted,
    )


def cuped_demo(n: int = 40_000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Demonstrate CUPED on synthetic data where the true effect is known.

    Run on two metrics, because the answer is very different and the difference
    is the practical lesson:

    * **conversion** — a rare binary outcome. A binary variable's correlation with
      any continuous covariate is mechanically limited, so the variance reduction
      is small no matter how good the covariate is. This is the case that matters
      for the campaign decision, and the honest answer is that CUPED barely helps.
    * **engagement** — a continuous in-window metric with a genuine pre-period
      analogue. Here CUPED does what the literature advertises.

    Both are checked against the identity ``variance reduction = correlation²``,
    which is what confirms the implementation is right rather than merely plausible.

    Args:
        n: Sample size.
        seed: Random seed.

    Returns:
        Mapping from metric name to its CUPED diagnostics, including the known
        true effect so the reader can confirm the adjustment did not move it.
    """
    data = simulate(n=n, seed=seed)
    metrics = {
        "conversion": (data.outcome.astype(np.float64), data.true_ate),
        "engagement": (data.engagement, float((20.0 * data.true_uplift).mean())),
    }

    report: dict[str, dict[str, float]] = {}
    for name, (values, true_effect) in metrics.items():
        _, result = cuped_adjust(values, data.pre_period, data.treatment)
        report[name] = {
            "true_ate": true_effect,
            "ate_raw": result.ate_raw,
            "ate_adjusted": result.ate_adjusted,
            "ate_shift": result.ate_adjusted - result.ate_raw,
            "correlation": result.correlation,
            "correlation_squared": result.correlation**2,
            "theta": result.theta,
            "variance_reduction": result.variance_reduction,
            "effective_sample_multiplier": result.effective_sample_multiplier,
            "stderr_raw": result.stderr_raw,
            "stderr_adjusted": result.stderr_adjusted,
            "stderr_reduction": 1.0 - result.stderr_adjusted / result.stderr_raw,
            "equivalent_extra_customers": n * (result.effective_sample_multiplier - 1.0),
        }
    return report


def design_validation_experiment(
    baseline_rate: float,
    expected_policy_uplift: float,
    expected_random_uplift: float,
    alpha: float = 0.05,
    power: float = 0.80,
    weekly_volume: int | None = None,
    cuped_variance_reduction: float = 0.0,
) -> dict[str, float]:
    """Size the head-to-head test of model targeting against business as usual.

    The design being sized is a four-cell test. The eligible file is split at
    random into a model-targeted cell and a control-targeted cell; within each,
    a holdback is left uncontacted so that the incremental effect of each policy
    is measured rather than assumed. The quantity under test is the difference
    between the two policies' incremental conversion rates.

    Args:
        baseline_rate: Conversion rate among uncontacted customers.
        expected_policy_uplift: Incremental conversion rate expected under model
            targeting, in percentage points.
        expected_random_uplift: Incremental conversion rate expected under the
            incumbent policy.
        alpha: Two-sided significance level.
        power: Desired power.
        weekly_volume: Contactable customers per week, used to convert the sample
            size into a duration.
        cuped_variance_reduction: Fractional variance reduction expected from
            CUPED, which divides the required sample size by ``1 / (1 - r)``.

    Returns:
        Mapping with the effect under test, the required cell sizes, the total
        file needed, and the run length in weeks if ``weekly_volume`` is given.
    """
    effect = expected_policy_uplift - expected_random_uplift
    if effect <= 0.0:
        return {
            "difference_under_test": effect,
            "n_per_cell": float("inf"),
            "n_total": float("inf"),
            "weeks_required": float("inf"),
            "note": 1.0,  # the model is not expected to beat the incumbent
        }

    sizing = required_sample_size(
        baseline_rate=baseline_rate + expected_random_uplift,
        absolute_mde=effect,
        alpha=alpha,
        power=power,
    )
    n_per_cell = sizing.n_per_arm * (1.0 - cuped_variance_reduction)
    n_total = 2.0 * n_per_cell

    return {
        "difference_under_test": effect,
        "baseline_for_test": baseline_rate + expected_random_uplift,
        "n_per_cell": float(np.ceil(n_per_cell)),
        "n_total": float(np.ceil(n_total)),
        "expected_conversions_per_cell": n_per_cell * (baseline_rate + expected_random_uplift),
        "weeks_required": (
            float(np.ceil(n_total / weekly_volume)) if weekly_volume else float("nan")
        ),
        "alpha": alpha,
        "power": power,
        "cuped_variance_reduction": cuped_variance_reduction,
    }
