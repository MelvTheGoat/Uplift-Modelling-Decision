"""Evaluating uplift models. Deliberately contains no AUC.

Why not AUC
-----------
AUC measures how well a score ranks customers by *whether they responded*. An
uplift model is not trying to do that. The best possible uplift model would give
a low score to a customer who was always going to buy, because contacting them
changes nothing, and a response-AUC would punish it for exactly that. A model can
have an excellent AUC and be worthless for targeting, and vice versa. So it is not
computed anywhere in this study.

What is used instead
--------------------
Every metric here compares the *treated* and *control* outcomes among the
customers a score would have selected, which is the only comparison that reflects
the decision being made.

* **Qini curve** — walk down the customer list in descending predicted uplift.
  At each point ask: among the customers selected so far, how many more
  conversions did the treated ones produce than a like-sized control group would
  have? Plotted against the fraction selected. A useful model bows above the
  straight line that random selection would trace.
* **Qini coefficient** — the area between the model's curve and the random line.
  Reported in incremental conversions, so it has units a business reader can
  check, alongside a unitless version normalised by the random baseline.
* **AUUC** — the same idea computed on the uplift curve, which scales the observed
  rate difference by the number of customers selected.
* **Uplift by decile** — the flat-footed version: split the file into ten groups by
  predicted uplift and show the measured treated-minus-control difference in each.
  If the top decile does not beat the bottom decile, nothing else matters.
* **Transformed outcome** — a per-customer target whose expectation equals the true
  individual effect, which turns effect estimation into an ordinary regression
  problem and gives a validation loss that can be computed on held-out data.
* **Beats-random test** — an explicit hypothesis test that the model's ordering is
  better than shuffling the list. Many published uplift models do not pass it and
  the failure is usually not reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.arrays import Array
from src.config import N_BOOTSTRAP, N_FOLDS
from src.estimators import UpliftEstimator


@dataclass(frozen=True)
class QiniResult:
    """A Qini curve and the summary numbers taken from it.

    Attributes:
        fraction: Fraction of the file targeted, ascending from 0 to 1.
        qini: Incremental conversions captured by targeting that fraction.
        random: The same quantity under random targeting (a straight line).
        qini_auc: Area under the model curve, in incremental conversions.
        random_auc: Area under the random line.
        qini_coefficient: ``qini_auc - random_auc``. Positive means the ordering helps.
        normalized_qini: ``qini_coefficient / |random_auc|``, unitless. Comparable
            across datasets of different sizes; undefined when the overall
            treatment effect is zero.
        total_incremental: Incremental conversions from treating the whole file.
    """

    fraction: Array
    qini: Array
    random: Array
    qini_auc: float
    random_auc: float
    qini_coefficient: float
    normalized_qini: float
    total_incremental: float


@dataclass(frozen=True)
class EvaluationResult:
    """Everything computed for one model on one dataset.

    Attributes:
        model: Estimator name.
        qini: The Qini result.
        auuc: Area under the uplift curve.
        transformed_outcome_mse: Validation loss against the transformed outcome.
        deciles: Uplift-by-decile table.
        top_decile_uplift: Measured uplift in the highest-scoring tenth.
        bottom_decile_uplift: Measured uplift in the lowest-scoring tenth.
        predicted_uplift: The out-of-fold scores the metrics were computed on.
        extra: Any additional named numbers.
    """

    model: str
    qini: QiniResult
    auuc: float
    transformed_outcome_mse: float
    deciles: pd.DataFrame
    top_decile_uplift: float
    bottom_decile_uplift: float
    predicted_uplift: Array
    extra: dict[str, float] = field(default_factory=dict)


def _order_by_score(score: Array, seed: int = 0) -> Array:
    """Indices sorting ``score`` descending, breaking ties at random.

    Ties matter more than they look. A model that predicts a single constant —
    which is what a badly shrunk S-learner does — would otherwise be ranked by
    row order, and row order in a sorted file can carry real signal, producing a
    Qini curve that flatters a model with no information in it at all.

    Args:
        score: Predicted uplift.
        seed: Seed for the tie-breaking jitter.

    Returns:
        Index array, highest score first.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(size=score.shape[0])
    return np.asarray(np.lexsort((jitter, -score)), dtype=np.int64)


def qini_curve(
    outcome: Array,
    treatment: Array,
    score: Array,
    seed: int = 0,
) -> QiniResult:
    """Compute the Qini curve for a set of predicted uplift scores.

    At each point down the ranked list, the curve is::

        Q(k) = responders_treated(k) - responders_control(k) * n_treated(k) / n_control(k)

    The second term rescales the control group to the size of the treated group,
    which is what makes the difference an estimate of incremental conversions
    rather than a comparison of two differently sized counts.

    Args:
        outcome: Binary outcome, shape ``(n,)``.
        treatment: Binary assignment, shape ``(n,)``.
        score: Predicted uplift used to rank, shape ``(n,)``.
        seed: Tie-breaking seed.

    Returns:
        A :class:`QiniResult`.

    Raises:
        ValueError: If the inputs disagree in length or an arm is empty.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if not (outcome.shape == treatment.shape == score.shape):
        raise ValueError("outcome, treatment and score must have the same shape")
    if treatment.sum() == 0 or (1 - treatment).sum() == 0:
        raise ValueError("need customers in both arms to compute a Qini curve")

    order = _order_by_score(score, seed=seed)
    y, w = outcome[order], treatment[order]

    n_treated = np.cumsum(w)
    n_control = np.cumsum(1 - w)
    y_treated = np.cumsum(y * w)
    y_control = np.cumsum(y * (1 - w))

    # Where no control customer has appeared yet the rescaling is undefined; the
    # incremental estimate is 0 there by convention.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_control > 0, n_treated / np.maximum(n_control, 1), 0.0)
    qini = y_treated - y_control * ratio

    n = outcome.shape[0]
    fraction = np.arange(1, n + 1, dtype=np.float64) / n
    fraction = np.concatenate([[0.0], fraction])
    qini = np.concatenate([[0.0], qini])

    total_incremental = float(qini[-1])
    random_line = fraction * total_incremental

    qini_auc = float(np.trapezoid(qini, fraction))
    random_auc = float(np.trapezoid(random_line, fraction))
    coefficient = qini_auc - random_auc
    normalized = coefficient / abs(random_auc) if abs(random_auc) > 1e-12 else float("nan")

    return QiniResult(
        fraction=fraction,
        qini=qini,
        random=random_line,
        qini_auc=qini_auc,
        random_auc=random_auc,
        qini_coefficient=coefficient,
        normalized_qini=normalized,
        total_incremental=total_incremental,
    )


def auuc_score(
    outcome: Array,
    treatment: Array,
    score: Array,
    seed: int = 0,
) -> float:
    """Area under the uplift curve.

    The uplift curve differs from the Qini curve in how it handles group sizes:
    it takes the difference of *rates* and multiplies by the number of customers
    selected, rather than rescaling counts. The two agree when the arms are
    balanced and diverge when they are not.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        seed: Tie-breaking seed.

    Returns:
        Area under the uplift curve, in incremental conversions.
    """
    order = _order_by_score(np.asarray(score, dtype=np.float64), seed=seed)
    y = np.asarray(outcome, dtype=np.float64)[order]
    w = np.asarray(treatment, dtype=np.int64)[order]

    n_treated = np.cumsum(w)
    n_control = np.cumsum(1 - w)
    y_treated = np.cumsum(y * w)
    y_control = np.cumsum(y * (1 - w))

    valid = (n_treated > 0) & (n_control > 0)
    rate_treated = np.divide(y_treated, np.maximum(n_treated, 1), dtype=np.float64)
    rate_control = np.divide(y_control, np.maximum(n_control, 1), dtype=np.float64)
    uplift = np.where(valid, (rate_treated - rate_control) * (n_treated + n_control), 0.0)

    n = y.shape[0]
    fraction = np.concatenate([[0.0], np.arange(1, n + 1, dtype=np.float64) / n])
    uplift = np.concatenate([[0.0], uplift])
    return float(np.trapezoid(uplift, fraction))


def transformed_outcome(
    outcome: Array,
    treatment: Array,
    propensity: float | Array,
) -> Array:
    """Per-customer target whose expectation is the individual treatment effect.

    Defined as::

        Z = Y * (W - p) / (p * (1 - p))

    Under randomisation, ``E[Z | X] = tau(X)``. That is a strong and useful fact:
    it converts an unobservable quantity into an ordinary regression target, so
    an uplift model can be scored on held-out data with a plain squared error.

    The catch is variance. Individual values of ``Z`` are enormous — for a
    50/50 split a single converting treated customer contributes ``+2`` — so the
    loss is unbiased but noisy, and it is only trustworthy averaged over many
    rows. It is used here as a model-selection criterion, never as a headline.

    Args:
        outcome: Observed outcome.
        treatment: Binary assignment.
        propensity: Treatment probability, scalar or per-row.

    Returns:
        Transformed outcomes, shape ``(n,)``.

    Raises:
        ValueError: If any propensity is outside (0, 1).
    """
    p = np.asarray(propensity, dtype=np.float64)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("propensity must lie strictly between 0 and 1")
    w = np.asarray(treatment, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    return y * (w - p) / (p * (1.0 - p))


def transformed_outcome_loss(
    outcome: Array,
    treatment: Array,
    score: Array,
    propensity: float | Array,
) -> float:
    """Mean squared error of predicted uplift against the transformed outcome.

    Args:
        outcome: Observed outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        propensity: Treatment probability.

    Returns:
        The loss. Lower is better; the absolute value is not interpretable because
        it includes an irreducible variance term identical across models.
    """
    z = transformed_outcome(outcome, treatment, propensity)
    return float(np.mean((z - np.asarray(score, dtype=np.float64)) ** 2))


def uplift_by_decile(
    outcome: Array,
    treatment: Array,
    score: Array,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Measured uplift within bins of predicted uplift.

    This is the table to show a sceptic. Nothing is modelled: within each bin the
    treated response rate is simply compared with the control response rate. If
    the model orders customers by true uplift, the numbers descend down the table.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        n_bins: Number of equal-sized bins (10 for deciles).

    Returns:
        A frame with one row per bin, ordered best-predicted first, containing
        counts, response rates, the measured uplift and its standard error.
    """
    order = _order_by_score(np.asarray(score, dtype=np.float64))
    y = np.asarray(outcome, dtype=np.float64)[order]
    w = np.asarray(treatment, dtype=np.int64)[order]
    s = np.asarray(score, dtype=np.float64)[order]

    bins = np.array_split(np.arange(y.shape[0]), n_bins)
    rows: list[dict[str, float]] = []
    for i, idx in enumerate(bins, start=1):
        yt, yc = y[idx][w[idx] == 1], y[idx][w[idx] == 0]
        rate_t = float(yt.mean()) if yt.size else float("nan")
        rate_c = float(yc.mean()) if yc.size else float("nan")
        # Standard error of a difference of two independent proportions.
        var_t = rate_t * (1 - rate_t) / yt.size if yt.size else float("nan")
        var_c = rate_c * (1 - rate_c) / yc.size if yc.size else float("nan")
        observed = rate_t - rate_c
        stderr = float(np.sqrt(var_t + var_c))
        rows.append(
            {
                "decile": float(i),
                "n": float(idx.size),
                "n_treated": float(yt.size),
                "n_control": float(yc.size),
                "predicted_uplift": float(s[idx].mean()),
                "rate_treated": rate_t,
                "rate_control": rate_c,
                "observed_uplift": observed,
                "stderr": stderr,
                "ci_low": observed - 1.96 * stderr,
                "ci_high": observed + 1.96 * stderr,
                "incremental_conversions": observed * float(idx.size),
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def bootstrap_qini(
    outcome: Array,
    treatment: Array,
    score: Array,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence interval for the Qini coefficient, and a beats-random test.

    Rows are resampled with replacement, stratified by arm so that the treated and
    control counts stay fixed and the resampling reflects sampling noise in the
    outcomes rather than in the design.

    The beats-random test is one-sided: the null is that the model's ordering is
    no better than random, so the p-value is the share of bootstrap replicates in
    which the Qini coefficient came out at or below zero. This is the check that
    published uplift work most often skips.

    Args:
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift.
        n_boot: Number of bootstrap replicates.
        seed: Seed.
        alpha: Two-sided level for the interval.

    Returns:
        Mapping with the point estimate, interval bounds, the one-sided p-value and
        the share of replicates that beat random.
    """
    rng = np.random.default_rng(seed)
    treated_idx = np.flatnonzero(treatment == 1)
    control_idx = np.flatnonzero(treatment == 0)

    point = qini_curve(outcome, treatment, score).qini_coefficient
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = np.concatenate(
            [
                rng.choice(treated_idx, size=treated_idx.size, replace=True),
                rng.choice(control_idx, size=control_idx.size, replace=True),
            ]
        )
        try:
            draws[b] = qini_curve(
                outcome[pick], treatment[pick], score[pick], seed=b
            ).qini_coefficient
        except ValueError:  # pragma: no cover - needs a degenerate resample
            draws[b] = np.nan

    finite = draws[np.isfinite(draws)]
    share_positive = float((finite > 0).mean()) if finite.size else float("nan")
    return {
        "qini_coefficient": float(point),
        "ci_low": float(np.quantile(finite, alpha / 2)) if finite.size else float("nan"),
        "ci_high": float(np.quantile(finite, 1 - alpha / 2)) if finite.size else float("nan"),
        "share_beating_random": share_positive,
        "p_value_one_sided": float(1.0 - share_positive) if finite.size else float("nan"),
        "beats_random": bool(finite.size and (1.0 - share_positive) < 0.05),
        "n_bootstrap": float(finite.size),
    }


def cross_val_uplift(
    estimator: UpliftEstimator,
    features: Array,
    treatment: Array,
    outcome: Array,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> Array:
    """Out-of-fold uplift predictions.

    Uplift scores must be evaluated out of sample. In sample, every estimator
    looks like it found heterogeneity, because it fit the noise in each arm.
    Folds are stratified on the treatment-by-outcome cross so that every fold
    contains both arms and some converters even at a 0.9% base rate.

    Args:
        estimator: An unfitted estimator; it is refit inside each fold.
        features: Covariate matrix.
        treatment: Binary assignment.
        outcome: Outcome.
        n_folds: Number of folds.
        seed: Seed for the fold split.

    Returns:
        Out-of-fold predicted uplift, aligned with the input rows.
    """
    strata = treatment.astype(np.int64) * 2 + (np.asarray(outcome) > 0).astype(np.int64)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    predictions = np.zeros(features.shape[0], dtype=np.float64)
    for train_idx, test_idx in splitter.split(features, strata):
        import copy

        fold_model = copy.deepcopy(estimator)
        fold_model.fit(features[train_idx], treatment[train_idx], outcome[train_idx])
        predictions[test_idx] = fold_model.predict_uplift(features[test_idx])
    return predictions


def evaluate(
    model_name: str,
    outcome: Array,
    treatment: Array,
    score: Array,
    propensity: float | None = None,
    n_bins: int = 10,
) -> EvaluationResult:
    """Run the full metric suite for one set of predictions.

    Args:
        model_name: Label for reporting.
        outcome: Binary outcome.
        treatment: Binary assignment.
        score: Predicted uplift (out-of-fold).
        propensity: Known treatment probability; estimated from the data if omitted.
        n_bins: Number of bins for the decile table.

    Returns:
        An :class:`EvaluationResult`.
    """
    p = float(np.mean(treatment)) if propensity is None else propensity
    qini = qini_curve(outcome, treatment, score)
    deciles = uplift_by_decile(outcome, treatment, score, n_bins=n_bins)
    return EvaluationResult(
        model=model_name,
        qini=qini,
        auuc=auuc_score(outcome, treatment, score),
        transformed_outcome_mse=transformed_outcome_loss(outcome, treatment, score, p),
        deciles=deciles,
        top_decile_uplift=float(deciles["observed_uplift"].iloc[0]),
        bottom_decile_uplift=float(deciles["observed_uplift"].iloc[-1]),
        predicted_uplift=np.asarray(score, dtype=np.float64),
    )


def recovery_error(predicted: Array, truth: Array) -> dict[str, float]:
    """How close an estimator got to a known individual treatment effect.

    Only computable on synthetic data. This is the gate every estimator passes
    through before it is allowed near the real file.

    Args:
        predicted: Estimated individual effects.
        truth: True individual effects.

    Returns:
        Mapping with PEHE (root mean squared error on the individual effect), the
        bias of the average effect, the correlation and rank correlation with
        truth, and the share of sleeping dogs correctly identified as negative.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pehe = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    # Undefined rather than zero when the truth contains no harmed customers,
    # so that "nothing to find" is never reported as "found nothing".
    negatives = truth < 0
    detected = float((predicted[negatives] < 0).mean()) if negatives.any() else float("nan")

    from scipy.stats import kendalltau, spearmanr

    return {
        "pehe": pehe,
        "ate_bias": float(predicted.mean() - truth.mean()),
        "ate_estimated": float(predicted.mean()),
        "ate_true": float(truth.mean()),
        "pearson_r": float(np.corrcoef(predicted, truth)[0, 1]),
        "spearman_r": float(spearmanr(predicted, truth).statistic),
        "kendall_tau": float(kendalltau(predicted, truth).statistic),
        "sleeping_dogs_detected": detected,
        "predicted_sd": float(predicted.std()),
        "true_sd": float(truth.std()),
        "sd_ratio": float(predicted.std() / truth.std()) if truth.std() > 0 else float("nan"),
    }
