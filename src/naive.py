"""The wrong answer, computed carefully, so the memo can explain why it is wrong.

Almost every campaign readout in existence is the calculation in this file:
contacted customers converted at X%, uncontacted at Y%, X is bigger than Y,
therefore the campaign works, therefore we should send more e-mail.

There are two separate problems with that sentence and they are usually confused
with each other.

**Problem one: is the gap real?** In a randomised trial, it is. The gap is an
unbiased estimate of the average treatment effect and nothing more needs to be
said. In an observational setting — which is what every "we mailed our engaged
segment and they bought more" readout actually is — the gap is contaminated by
whatever made those customers get mailed in the first place. The engaged segment
would have bought more anyway. :func:`compare_randomised_and_confounded`
quantifies exactly how much of the gap that accounts for, using synthetic data
where the true answer is known.

**Problem two: even when the gap is real, it does not tell you whom to mail.**
This one survives randomisation, and it is the more important of the two because
it is the one people do not expect. The overall gap is an *average*. It is
consistent with every customer having the same small effect, and it is equally
consistent with a third of customers having a large positive effect and another
third being actively annoyed. Those two worlds call for opposite campaigns and
the headline number cannot distinguish them.

The corollary is the expensive one. Once a campaign "works", the natural next
move is to find the customers most likely to convert and send them more. That is
a response model, and it ranks customers by ``P(convert | contacted)``, which is
dominated by who was going to buy anyway. :func:`response_versus_uplift_ranking`
shows on synthetic data that this ranking can be almost uncorrelated with the
ranking by actual incremental effect — and that the customers it puts at the very
top can include the ones the campaign harms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.arrays import Array
from src.estimators import LearnerFactory, TLearner, _positive_probability
from src.simulate import SyntheticData, simulate, simulate_confounded


@dataclass(frozen=True)
class NaiveComparison:
    """The headline treated-versus-untreated comparison.

    Attributes:
        rate_treated: Response rate among contacted customers.
        rate_control: Response rate among uncontacted customers.
        absolute_difference: ``rate_treated - rate_control``, in percentage points.
        relative_lift: The difference as a percentage of the control rate — the
            number that ends up in the slide title.
        stderr: Standard error of the difference.
        ci_low: Lower bound of the 95% interval.
        ci_high: Upper bound of the 95% interval.
        p_value: Two-sided p-value from a two-proportion z-test.
        n_treated: Contacted customers.
        n_control: Uncontacted customers.
        revenue_treated: Mean revenue per contacted customer.
        revenue_control: Mean revenue per uncontacted customer.
        revenue_difference: Incremental revenue per customer contacted.
    """

    rate_treated: float
    rate_control: float
    absolute_difference: float
    relative_lift: float
    stderr: float
    ci_low: float
    ci_high: float
    p_value: float
    n_treated: int
    n_control: int
    revenue_treated: float
    revenue_control: float
    revenue_difference: float


def naive_comparison(
    outcome: Array,
    treatment: Array,
    spend: Array | None = None,
) -> NaiveComparison:
    """Compute the standard campaign readout.

    Nothing here is a mistake in itself. Under randomisation this is the correct
    estimate of the average treatment effect, and the memo says so. The mistake is
    what gets concluded from it.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        spend: Optional revenue per customer.

    Returns:
        A :class:`NaiveComparison`.

    Raises:
        ValueError: If either arm is empty.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64)
    treated, control = treatment == 1, treatment == 0
    if not treated.any() or not control.any():
        raise ValueError("both arms must be non-empty")

    n_t, n_c = int(treated.sum()), int(control.sum())
    rate_t, rate_c = float(outcome[treated].mean()), float(outcome[control].mean())
    difference = rate_t - rate_c
    stderr = float(np.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c))

    # Pooled-variance two-proportion z-test, the textbook version.
    pooled = float(outcome.mean())
    pooled_se = float(np.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c)))
    z = difference / pooled_se if pooled_se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    if spend is None:
        revenue_t = revenue_c = 0.0
    else:
        spend = np.asarray(spend, dtype=np.float64)
        revenue_t, revenue_c = float(spend[treated].mean()), float(spend[control].mean())

    return NaiveComparison(
        rate_treated=rate_t,
        rate_control=rate_c,
        absolute_difference=difference,
        relative_lift=difference / rate_c if rate_c > 0 else float("nan"),
        stderr=stderr,
        ci_low=difference - 1.96 * stderr,
        ci_high=difference + 1.96 * stderr,
        p_value=p_value,
        n_treated=n_t,
        n_control=n_c,
        revenue_treated=revenue_t,
        revenue_control=revenue_c,
        revenue_difference=revenue_t - revenue_c,
    )


def compare_randomised_and_confounded(
    n: int = 40_000,
    seed: int = 0,
    confounding_strength: float = 1.5,
) -> dict[str, float]:
    """Show what the naive gap measures when nobody randomised anything.

    Two synthetic populations with *identical* individual treatment effects. In
    the first, treatment is a coin flip. In the second, customers likely to buy
    anyway are the ones who get mailed — the way real campaign lists are actually
    built. The true average effect is the same in both. The naive gap is not.

    Args:
        n: Sample size for each population.
        seed: Random seed.
        confounding_strength: How strongly assignment depends on the covariate
            that also drives the baseline conversion rate.

    Returns:
        Mapping with the true effect, the naive gap under each design, and the
        multiple by which confounding inflates the gap.
    """
    randomised = simulate(n=n, seed=seed)
    confounded = simulate_confounded(n=n, seed=seed, confounding_strength=confounding_strength)

    naive_rand = naive_comparison(randomised.outcome, randomised.treatment, randomised.spend)
    naive_conf = naive_comparison(confounded.outcome, confounded.treatment, confounded.spend)
    truth = randomised.true_ate

    return {
        "true_ate": truth,
        "naive_gap_randomised": naive_rand.absolute_difference,
        "naive_gap_confounded": naive_conf.absolute_difference,
        "bias_randomised": naive_rand.absolute_difference - truth,
        "bias_confounded": naive_conf.absolute_difference - truth,
        "overstatement_factor": (
            naive_conf.absolute_difference / truth if abs(truth) > 1e-9 else float("nan")
        ),
        "relative_lift_randomised": naive_rand.relative_lift,
        "relative_lift_confounded": naive_conf.relative_lift,
    }


def response_versus_uplift_ranking(
    n: int = 40_000,
    seed: int = 0,
    top_fraction: float = 0.2,
) -> dict[str, float]:
    """Compare targeting by likelihood-to-convert against targeting by uplift.

    Fits two models on the *same randomised* synthetic data:

    * a **response model** — ordinary ``P(convert | contacted)``, the thing a
      marketing analytics team builds by default;
    * an **uplift model** — a T-learner estimating the incremental effect.

    Then asks what each would capture if it selected the top slice of the file,
    measured against the known individual effects. Randomisation is not in
    question here; both models see clean experimental data. The gap between them
    is entirely about what question each one was asked.

    Args:
        n: Sample size.
        seed: Random seed.
        top_fraction: Share of the file each strategy is allowed to contact.

    Returns:
        Mapping with the true uplift captured by each ranking, their correlation,
        and how many sleeping dogs the response model would contact.
    """
    data = simulate(n=n, seed=seed)
    features, treatment, outcome = data.features, data.treatment, data.outcome

    # Response model: trained on treated customers only, which is exactly how a
    # "who responds to our e-mail" model gets built from campaign history.
    factory = LearnerFactory(kind="lightgbm", random_state=seed)
    response_model = factory.classifier()
    treated = treatment == 1
    response_model.fit(features[treated], outcome[treated])
    response_score = _positive_probability(response_model, features)

    uplift_model = TLearner(factory).fit(features, treatment, outcome)
    uplift_score = uplift_model.predict_uplift(features)

    k = int(round(top_fraction * n))
    top_response = np.argsort(-response_score)[:k]
    top_uplift = np.argsort(-uplift_score)[:k]
    top_oracle = np.argsort(-data.true_uplift)[:k]

    dogs = data.sleeping_dog_mask
    return {
        "top_fraction": top_fraction,
        "true_uplift_captured_by_response_model": float(data.true_uplift[top_response].sum()),
        "true_uplift_captured_by_uplift_model": float(data.true_uplift[top_uplift].sum()),
        "true_uplift_captured_by_oracle": float(data.true_uplift[top_oracle].sum()),
        "true_uplift_captured_by_random": float(data.true_uplift.mean() * k),
        "mean_true_uplift_response_model": float(data.true_uplift[top_response].mean()),
        "mean_true_uplift_uplift_model": float(data.true_uplift[top_uplift].mean()),
        "mean_true_uplift_random": float(data.true_uplift.mean()),
        "rank_correlation_response_vs_uplift": float(
            stats.spearmanr(response_score, uplift_score).statistic
        ),
        "rank_correlation_response_vs_truth": float(
            stats.spearmanr(response_score, data.true_uplift).statistic
        ),
        "rank_correlation_uplift_vs_truth": float(
            stats.spearmanr(uplift_score, data.true_uplift).statistic
        ),
        "sleeping_dogs_contacted_by_response_model": float(dogs[top_response].sum()),
        "sleeping_dogs_contacted_by_uplift_model": float(dogs[top_uplift].sum()),
        "sleeping_dog_share_of_file": float(dogs.mean()),
    }


def run_naive_analysis(data: SyntheticData | None = None, seed: int = 0) -> dict[str, object]:
    """Assemble every naive-analysis result used in the memo.

    Args:
        data: Optional pre-generated synthetic data; ignored except for its size.
        seed: Random seed.

    Returns:
        Mapping of section name to results.
    """
    n = data.n if data is not None else 40_000
    return {
        "confounding": compare_randomised_and_confounded(n=n, seed=seed),
        "ranking": response_versus_uplift_ranking(n=n, seed=seed),
    }
