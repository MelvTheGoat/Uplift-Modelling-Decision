"""Synthetic data with a KNOWN individual treatment effect.

This module exists so that every estimator and every evaluation metric can be
checked against an answer we already know before any of them is pointed at real
data. On Hillstrom the true individual uplift is unobservable — each customer is
either mailed or not, never both — so a model that looks good there might simply
be reproducing our own mistakes. Here it cannot hide.

The generating process deliberately contains four things:

1. **Confounder-free randomisation.** Treatment is assigned by a coin flip that
   ignores the covariates entirely, so the only reason treated and control
   customers differ is the treatment and sampling noise.
2. **Heterogeneous effects.** The treatment helps some customers much more than
   others, driven by two of the features.
3. **Sleeping dogs.** A well-defined subgroup whose response is genuinely
   *reduced* by being contacted. Any honest uplift method must find them; a
   response model structurally cannot.
4. **Noise features.** Columns with exactly zero effect on anything, to catch
   estimators that manufacture signal from nothing.

A confounded variant (`simulate_confounded`) is provided for the naive-analysis
demonstration: same effects, but treatment now depends on the covariates, which
is what an observational marketing log actually looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.arrays import Array

# ---------------------------------------------------------------- effect structure
#
# Named constants so the tests can assert against the truth without duplicating
# magic numbers, and so the docstring and the code cannot drift apart.

#: Uplift every customer receives regardless of features (percentage points, absolute).
BASE_UPLIFT: float = 0.02
#: Slope of uplift in `x0` — the "engaged customer" axis.
UPLIFT_SLOPE_X0: float = 0.05
#: Extra uplift for customers on the positive side of `x1`.
UPLIFT_BONUS_X1: float = 0.03
#: Uplift for the sleeping-dog subgroup. It *replaces* the usual effect rather than
#: being subtracted from it, so that every member of the subgroup is genuinely harmed
#: and the group has an unambiguous right answer for the tests to check against.
SLEEPING_DOG_UPLIFT: float = -0.06
#: A customer is a sleeping dog when `x2` falls below this threshold (~16% of the file).
SLEEPING_DOG_THRESHOLD: float = -1.0
#: Revenue booked per conversion, before margin.
REVENUE_PER_CONVERSION: float = 100.0


@dataclass(frozen=True)
class SyntheticData:
    """A synthetic randomised experiment where the individual effect is known.

    Attributes:
        features: Covariate matrix, shape ``(n, n_features)``. Columns ``x0..x{k-1}``
            drive the outcome or the effect; the remainder are pure noise.
        treatment: Binary assignment, 1 = contacted.
        outcome: Realised binary conversion under the assigned arm.
        spend: Realised revenue (0 for non-converters).
        true_uplift: The individual treatment effect on conversion probability,
            ``P(Y=1 | do(W=1), X) - P(Y=1 | do(W=0), X)``. Never available in real life.
        baseline_prob: ``P(Y=1 | do(W=0), X)``, the untreated conversion probability.
        outcome_if_treated: Counterfactual outcome had the customer been treated.
        outcome_if_control: Counterfactual outcome had the customer been left alone.
        pre_period: A pre-treatment covariate, measured before assignment, correlated
            with both outcomes. The input CUPED needs.
        engagement: A continuous post-treatment metric (think site sessions in the
            outcome window). Included because variance-reduction techniques behave
            very differently on a continuous metric than on a 1%-rate binary one,
            and the difference is worth showing rather than asserting.
        propensity: The true probability of treatment for each row.
        feature_names: Column names matching ``features``.
        informative_features: Names of the columns that actually do something.
    """

    features: Array
    treatment: Array
    outcome: Array
    spend: Array
    true_uplift: Array
    baseline_prob: Array
    outcome_if_treated: Array
    outcome_if_control: Array
    pre_period: Array
    engagement: Array
    propensity: Array
    feature_names: list[str] = field(default_factory=list)
    informative_features: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of rows."""
        return int(self.features.shape[0])

    @property
    def true_ate(self) -> float:
        """The population average treatment effect, known exactly."""
        return float(self.true_uplift.mean())

    @property
    def sleeping_dog_mask(self) -> Array:
        """Boolean mask of customers whose true uplift is negative."""
        return self.true_uplift < 0.0

    def to_frame(self) -> pd.DataFrame:
        """Return an analysis-ready frame (truth columns included, prefixed ``true_``)."""
        frame: pd.DataFrame = pd.DataFrame(self.features, columns=self.feature_names)
        frame["pre_period"] = self.pre_period
        frame["engagement"] = self.engagement
        frame["treatment"] = self.treatment
        frame["outcome"] = self.outcome
        frame["spend"] = self.spend
        frame["true_uplift"] = self.true_uplift
        frame["true_baseline"] = self.baseline_prob
        return frame


def _uplift_function(features: Array) -> Array:
    """The known individual treatment effect, in absolute conversion probability.

    Deliberately not a single monotone function, so that a linear estimator cannot
    get it exactly right by accident:

    * a constant lift everyone gets,
    * a smooth slope in ``x0`` plus a step in ``x1`` (heterogeneity),
    * an override for the sleeping-dog subgroup defined by ``x2``, who are harmed
      no matter how promising they look on every other dimension.

    The override matters. If the penalty were merely subtracted, a customer deep in
    the subgroup but high on ``x0`` would come out positive, and the subgroup would
    no longer have a clean right answer to test against.

    Args:
        features: Covariate matrix.

    Returns:
        Individual treatment effects, shape ``(n,)``.
    """
    x0, x1, x2 = features[:, 0], features[:, 1], features[:, 2]
    responsive = BASE_UPLIFT + UPLIFT_SLOPE_X0 * x0 + UPLIFT_BONUS_X1 * (x1 > 0.0)
    uplift = np.where(x2 < SLEEPING_DOG_THRESHOLD, SLEEPING_DOG_UPLIFT, responsive)
    return uplift.astype(np.float64)


def _baseline_function(features: Array) -> Array:
    """Untreated conversion probability ``P(Y=1 | do(W=0), X)``.

    Note that ``x3`` is a strong driver of the *baseline* but has no effect on
    *uplift* at all. This is the trap the memo is about: customers high on ``x3``
    look like the best targets to a response model and are worth nothing extra.

    Args:
        features: Covariate matrix.

    Returns:
        Baseline probabilities in (0, 1), shape ``(n,)``.
    """
    x0, x3 = features[:, 0], features[:, 3]
    logit = -2.0 + 0.35 * x0 + 1.20 * x3
    return 1.0 / (1.0 + np.exp(-logit))


def simulate(
    n: int = 20_000,
    n_features: int = 10,
    treatment_share: float = 0.5,
    seed: int = 0,
    noise_sd: float = 1.0,
) -> SyntheticData:
    """Generate a randomised experiment with a known individual treatment effect.

    Args:
        n: Number of customers.
        n_features: Total covariates. The first four are informative; the rest are noise.
        treatment_share: Probability of assignment to treatment. Set below 0.5 to
            create the treatment imbalance that X-learner is designed for.
        seed: Random seed.
        noise_sd: Standard deviation of the covariates.

    Returns:
        A :class:`SyntheticData` bundle including the counterfactual outcomes.

    Raises:
        ValueError: If fewer than four features are requested or the share is not a
            probability strictly between 0 and 1.
    """
    if n_features < 4:
        raise ValueError("need at least 4 features: x0..x3 are the informative ones")
    if not 0.0 < treatment_share < 1.0:
        raise ValueError("treatment_share must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, noise_sd, size=(n, n_features))

    baseline = _baseline_function(features)
    uplift = _uplift_function(features)

    # Clip the treated probability into a valid range, then recompute the uplift
    # from the clipped value so that `true_uplift` is exactly the effect realised
    # by the sampler rather than an idealised version of it.
    treated_prob = np.clip(baseline + uplift, 1e-4, 1.0 - 1e-4)
    realised_uplift = treated_prob - baseline

    # Randomisation: a coin flip that never looks at `features`. This is what makes
    # the naive difference in means an unbiased estimate of the average effect.
    propensity = np.full(n, treatment_share, dtype=np.float64)
    treatment = rng.binomial(1, propensity).astype(np.int64)

    # Draw both potential outcomes with a shared uniform so that the counterfactual
    # pair is coupled: the same customer does not flip from converter to
    # non-converter purely because of independent sampling noise.
    draw = rng.uniform(size=n)
    outcome_if_control = (draw < baseline).astype(np.int64)
    outcome_if_treated = (draw < treated_prob).astype(np.int64)
    outcome = np.where(treatment == 1, outcome_if_treated, outcome_if_control)

    spend = outcome * rng.gamma(shape=2.0, scale=REVENUE_PER_CONVERSION / 2.0, size=n)

    # A persistent customer trait that shows up both before and during the
    # experiment. This is what makes a pre-period covariate useful at all: without
    # something stable about the customer, last month tells you nothing about this one.
    loyalty = rng.normal(0.0, 1.0, size=n)

    # Measured *before* assignment, so the treatment cannot have touched it — the
    # condition CUPED depends on and the one most often violated in practice.
    pre_period = 4.0 + 2.5 * loyalty + baseline * 6.0 + rng.normal(0.0, 1.2, size=n)

    # A continuous in-window metric, affected by treatment proportionally to that
    # customer's uplift so the effect is heterogeneous here too.
    engagement = (
        4.0 + 2.5 * loyalty + 20.0 * realised_uplift * treatment + rng.normal(0.0, 1.5, size=n)
    )

    names = [f"x{i}" for i in range(n_features)]
    return SyntheticData(
        features=features,
        treatment=treatment,
        outcome=outcome,
        spend=spend,
        true_uplift=realised_uplift,
        baseline_prob=baseline,
        outcome_if_treated=outcome_if_treated,
        outcome_if_control=outcome_if_control,
        pre_period=pre_period,
        engagement=engagement,
        propensity=propensity,
        feature_names=names,
        informative_features=["x0", "x1", "x2", "x3"],
    )


def simulate_confounded(
    n: int = 20_000,
    n_features: int = 10,
    seed: int = 0,
    confounding_strength: float = 1.5,
) -> SyntheticData:
    """Same effects, but treatment is assigned the way a real marketing team assigns it.

    Customers with a high baseline propensity to convert (driven by ``x3``) are more
    likely to be mailed — because they are the engaged segment, they are in the
    loyalty programme, they opened the last campaign. Nobody randomised anything.

    The individual treatment effects are identical to :func:`simulate`. Only the
    assignment mechanism changes. This isolates one thing: how much of the naive
    treated-versus-untreated gap is real effect and how much is selection.

    Args:
        n: Number of customers.
        n_features: Total covariates.
        seed: Random seed.
        confounding_strength: Coefficient on ``x3`` in the assignment logit. Zero
            recovers randomisation.

    Returns:
        A :class:`SyntheticData` bundle whose ``propensity`` varies by customer.
    """
    randomised = simulate(n=n, n_features=n_features, seed=seed)

    rng = np.random.default_rng(seed + 9_999)
    logit = confounding_strength * randomised.features[:, 3]
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity).astype(np.int64)

    outcome = np.where(treatment == 1, randomised.outcome_if_treated, randomised.outcome_if_control)
    spend = outcome * rng.gamma(shape=2.0, scale=REVENUE_PER_CONVERSION / 2.0, size=n)

    # Swap the randomised treatment's contribution to engagement for the
    # confounded one, holding the customer's own noise draw fixed.
    effect_size = 20.0 * randomised.true_uplift
    engagement = (
        randomised.engagement - effect_size * randomised.treatment + effect_size * treatment
    )

    return SyntheticData(
        features=randomised.features,
        treatment=treatment,
        outcome=outcome,
        spend=spend,
        true_uplift=randomised.true_uplift,
        baseline_prob=randomised.baseline_prob,
        outcome_if_treated=randomised.outcome_if_treated,
        outcome_if_control=randomised.outcome_if_control,
        pre_period=randomised.pre_period,
        engagement=engagement,
        propensity=propensity,
        feature_names=randomised.feature_names,
        informative_features=randomised.informative_features,
    )


def describe(data: SyntheticData) -> dict[str, float]:
    """Summarise the known truth of a synthetic sample.

    Args:
        data: A simulated dataset.

    Returns:
        Mapping of summary statistic name to value.
    """
    dogs = data.sleeping_dog_mask
    return {
        "n": float(data.n),
        "treated_share": float(data.treatment.mean()),
        "true_ate": data.true_ate,
        "baseline_conversion": float(data.baseline_prob.mean()),
        "sleeping_dog_share": float(dogs.mean()),
        "sleeping_dog_mean_uplift": float(data.true_uplift[dogs].mean()) if dogs.any() else 0.0,
        "best_decile_mean_uplift": float(
            data.true_uplift[data.true_uplift >= np.quantile(data.true_uplift, 0.9)].mean()
        ),
        "uplift_sd": float(data.true_uplift.std()),
    }
