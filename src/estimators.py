"""Uplift estimators with a single fit/predict interface.

Four meta-learners, three of them written out by hand over an ordinary sklearn or
LightGBM base model so the mechanics are visible rather than hidden behind a
library call, plus a causal forest from EconML.

What each one assumes, and where each one breaks
------------------------------------------------

**S-learner** ("single model"). Fits one model on the pooled data with treatment
as just another feature, then differences the prediction with the flag on and
off. Assumes the base learner will *choose* to split on the treatment flag. That
is the failure mode: with one binary column competing against a dozen strong
predictors of the outcome, a regularised tree ensemble often barely uses it, and
the predicted uplift collapses toward a constant. S-learner is biased toward
finding no heterogeneity. It is the most stable of the four and the most likely
to be uselessly flat.

**T-learner** ("two models"). Fits one model on the treated and another on the
control, and differences them. Makes no assumption that the two response
surfaces share a shape, which is its advantage. Its weakness is that it models
the *outcome* twice and never models the *difference*: both models spend their
capacity fitting the parts of the response that are common to both arms and
irrelevant to the decision. The two independent error terms add, so with a rare
outcome the difference is mostly noise.

**X-learner**. A two-stage repair for T-learner's weakness under imbalance. Fit
the two outcome models; impute each customer's counterfactual from the *other*
arm's model; that gives every row a pseudo-effect; then regress the pseudo-effect
on the covariates — so the second stage is modelling the difference directly.
Combine the treated-side and control-side effect models weighted by propensity,
which puts weight on whichever arm is larger and therefore better estimated.
This is the reason to prefer it when one arm is much smaller than the other: at a
90/10 split the control model is fit on scarce data, and X-learner leans on the
treated-side estimate where the control model is weakest. It inherits the base
learner's bias and adds a second layer of it.

**Causal forest** (EconML ``CausalForestDML``). Trees grown with splits chosen to
maximise heterogeneity in the *effect* rather than in the outcome, with honest
sample splitting so the leaf that estimates an effect is not the leaf that chose
the split. Gives valid confidence intervals under its assumptions. Assumes
unconfoundedness (satisfied here by randomisation) and overlap. Costs far more
compute and, on a rare binary outcome, honesty halves the data available for
estimation, which can leave it noisier than the cruder learners.

All four assume the same three things, which randomisation gives us for free on
Hillstrom and which an observational marketing log does not: unconfoundedness,
overlap, and no interference between customers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.arrays import Array

__all__ = [
    "CausalForestLearner",
    "LearnerFactory",
    "SLearner",
    "TLearner",
    "UpliftEstimator",
    "XLearner",
    "build_estimators",
]


class LearnerFactory:
    """Builds fresh base classifiers and regressors of a chosen family.

    The meta-learners need both: classifiers for modelling a binary outcome and
    regressors for modelling continuous pseudo-effects. Keeping the choice in one
    place means an estimator can be swapped from gradient boosting to logistic
    regression in one argument, which is how the robustness checks test whether a
    conclusion depends on the model family.

    Args:
        kind: One of ``"lightgbm"``, ``"hgb"`` (sklearn histogram boosting),
            ``"forest"``, or ``"linear"``.
        random_state: Seed passed to every base model.
        **params: Overrides merged into the default hyper-parameters.

    Raises:
        ValueError: On an unknown ``kind``.
    """

    KINDS = ("lightgbm", "hgb", "forest", "linear")

    def __init__(self, kind: str = "lightgbm", random_state: int = 0, **params: Any) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind = kind
        self.random_state = random_state
        self.params = params

    def _lightgbm_defaults(self) -> dict[str, Any]:
        """Conservative LightGBM settings suited to a rare binary outcome."""
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
            "random_state": self.random_state,
        }

    def classifier(self) -> BaseEstimator:
        """Return an unfitted classifier exposing ``predict_proba``."""
        if self.kind == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(**{**self._lightgbm_defaults(), **self.params})
        if self.kind == "hgb":
            defaults = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15}
            return HistGradientBoostingClassifier(
                **{**defaults, **self.params}, random_state=self.random_state
            )
        if self.kind == "forest":
            defaults = {"n_estimators": 300, "min_samples_leaf": 50, "n_jobs": -1}
            return RandomForestClassifier(
                **{**defaults, **self.params}, random_state=self.random_state
            )
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.0, **self.params)),
            ]
        )

    def regressor(self) -> BaseEstimator:
        """Return an unfitted regressor."""
        if self.kind == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**{**self._lightgbm_defaults(), **self.params})
        if self.kind == "hgb":
            defaults = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15}
            return HistGradientBoostingRegressor(
                **{**defaults, **self.params}, random_state=self.random_state
            )
        if self.kind == "forest":
            defaults = {"n_estimators": 300, "min_samples_leaf": 50, "n_jobs": -1}
            return RandomForestRegressor(
                **{**defaults, **self.params}, random_state=self.random_state
            )
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0, **self.params))])


def _positive_probability(model: Any, features: Array) -> Array:
    """Predicted probability of the positive class, robust to degenerate arms.

    If an arm happens to contain only one outcome class — which occurs in small
    bootstrap resamples of a 0.9% conversion rate — sklearn fits a single-class
    model and ``predict_proba`` returns one column. Treat that as a constant.

    Args:
        model: A fitted classifier.
        features: Covariates to score.

    Returns:
        Probabilities in [0, 1], shape ``(n,)``.
    """
    proba = np.asarray(model.predict_proba(features), dtype=np.float64)
    if proba.shape[1] == 1:
        only_class = int(np.asarray(model.classes_)[0])
        return np.full(features.shape[0], float(only_class))
    return proba[:, 1]


class UpliftEstimator(ABC):
    """Common interface: ``fit(features, treatment, outcome)`` then ``predict_uplift``.

    Predicted uplift is always on the same scale as the outcome — for a binary
    outcome, an absolute change in probability. A prediction of 0.03 means "three
    more conversions per hundred customers like this one".
    """

    name: str = "uplift-estimator"

    @abstractmethod
    def fit(self, features: Array, treatment: Array, outcome: Array) -> UpliftEstimator:
        """Fit the estimator.

        Args:
            features: Covariate matrix, shape ``(n, d)``.
            treatment: Binary assignment, shape ``(n,)``.
            outcome: Outcome, shape ``(n,)``.

        Returns:
            ``self``, for chaining.
        """

    @abstractmethod
    def predict_uplift(self, features: Array) -> Array:
        """Predict the individual treatment effect.

        Args:
            features: Covariate matrix, shape ``(n, d)``.

        Returns:
            Estimated effects, shape ``(n,)``.
        """

    def fit_predict(self, features: Array, treatment: Array, outcome: Array) -> Array:
        """Fit and score the same data. In-sample; use only for diagnostics.

        Args:
            features: Covariate matrix.
            treatment: Binary assignment.
            outcome: Outcome.

        Returns:
            In-sample estimated effects.
        """
        return self.fit(features, treatment, outcome).predict_uplift(features)


class SLearner(UpliftEstimator):
    """One model on the pooled data with treatment appended as a feature.

    Args:
        factory: Supplies the base learner.
        classifier: Whether the outcome is binary.
    """

    name = "s-learner"

    def __init__(self, factory: LearnerFactory | None = None, classifier: bool = True) -> None:
        self.factory = factory or LearnerFactory()
        self.classifier = classifier
        self.model_: BaseEstimator | None = None

    def fit(self, features: Array, treatment: Array, outcome: Array) -> SLearner:
        """Fit the pooled model. See :meth:`UpliftEstimator.fit`."""
        design = np.column_stack([features, treatment.astype(np.float64)])
        model = self.factory.classifier() if self.classifier else self.factory.regressor()
        model.fit(design, outcome)
        self.model_ = model
        return self

    def predict_uplift(self, features: Array) -> Array:
        """Difference the pooled model with the treatment flag on and off."""
        if self.model_ is None:
            raise RuntimeError("call fit() before predict_uplift()")
        ones = np.column_stack([features, np.ones(features.shape[0])])
        zeros = np.column_stack([features, np.zeros(features.shape[0])])
        if self.classifier:
            return _positive_probability(self.model_, ones) - _positive_probability(
                self.model_, zeros
            )
        treated = np.asarray(self.model_.predict(ones), dtype=np.float64)
        control = np.asarray(self.model_.predict(zeros), dtype=np.float64)
        return treated - control


class TLearner(UpliftEstimator):
    """Separate outcome models for the treated and control arms, then differenced.

    Args:
        factory: Supplies the base learners.
        classifier: Whether the outcome is binary.
    """

    name = "t-learner"

    def __init__(self, factory: LearnerFactory | None = None, classifier: bool = True) -> None:
        self.factory = factory or LearnerFactory()
        self.classifier = classifier
        self.model_treated_: BaseEstimator | None = None
        self.model_control_: BaseEstimator | None = None

    def _new(self) -> BaseEstimator:
        return self.factory.classifier() if self.classifier else self.factory.regressor()

    def fit(self, features: Array, treatment: Array, outcome: Array) -> TLearner:
        """Fit one model per arm. See :meth:`UpliftEstimator.fit`."""
        treated = treatment == 1
        self.model_treated_ = self._new().fit(features[treated], outcome[treated])
        self.model_control_ = self._new().fit(features[~treated], outcome[~treated])
        return self

    def _score(self, model: BaseEstimator, features: Array) -> Array:
        if self.classifier:
            return _positive_probability(model, features)
        return np.asarray(model.predict(features), dtype=np.float64)

    def predict_uplift(self, features: Array) -> Array:
        """Treated-arm prediction minus control-arm prediction."""
        if self.model_treated_ is None or self.model_control_ is None:
            raise RuntimeError("call fit() before predict_uplift()")
        return self._score(self.model_treated_, features) - self._score(
            self.model_control_, features
        )


class XLearner(UpliftEstimator):
    """Two-stage learner that models the imputed effect directly.

    Stage one fits an outcome model per arm, exactly as T-learner does. Stage two
    imputes each customer's unobserved counterfactual from the opposite arm's
    model, producing a pseudo-effect per row, and fits a *regression* of that
    pseudo-effect on the covariates. The two resulting effect models are blended
    by the propensity score so that the arm with more data carries more weight
    where it is more reliable.

    Args:
        factory: Supplies the base learners.
        classifier: Whether the outcome is binary. Stage two is always a regression.
        propensity: Known assignment probability. Supply it when the design is a
            randomised trial with a known split — it is the correct value and
            estimating it only adds noise. ``None`` estimates it from the data.
    """

    name = "x-learner"

    def __init__(
        self,
        factory: LearnerFactory | None = None,
        classifier: bool = True,
        propensity: float | None = None,
    ) -> None:
        self.factory = factory or LearnerFactory()
        self.classifier = classifier
        self.propensity = propensity
        self.model_treated_: BaseEstimator | None = None
        self.model_control_: BaseEstimator | None = None
        self.effect_treated_: BaseEstimator | None = None
        self.effect_control_: BaseEstimator | None = None
        self.propensity_model_: BaseEstimator | None = None

    def _new_outcome(self) -> BaseEstimator:
        return self.factory.classifier() if self.classifier else self.factory.regressor()

    def _score(self, model: BaseEstimator, features: Array) -> Array:
        if self.classifier:
            return _positive_probability(model, features)
        return np.asarray(model.predict(features), dtype=np.float64)

    def fit(self, features: Array, treatment: Array, outcome: Array) -> XLearner:
        """Fit both stages. See :meth:`UpliftEstimator.fit`."""
        treated = treatment == 1
        x_treated, x_control = features[treated], features[~treated]
        y_treated, y_control = outcome[treated], outcome[~treated]

        # Stage 1: outcome models, one per arm.
        self.model_treated_ = self._new_outcome().fit(x_treated, y_treated)
        self.model_control_ = self._new_outcome().fit(x_control, y_control)

        # Stage 2: impute the missing potential outcome from the *other* arm's model.
        # A treated customer's effect is what they did minus what the control model
        # says they would have done; a control customer's is the mirror image.
        imputed_treated = y_treated.astype(np.float64) - self._score(self.model_control_, x_treated)
        imputed_control = self._score(self.model_treated_, x_control) - y_control.astype(np.float64)

        self.effect_treated_ = self.factory.regressor().fit(x_treated, imputed_treated)
        self.effect_control_ = self.factory.regressor().fit(x_control, imputed_control)

        if self.propensity is None:
            self.propensity_model_ = clone(self.factory.classifier()).fit(features, treatment)
        return self

    def _propensity_scores(self, features: Array) -> Array:
        if self.propensity is not None:
            return np.full(features.shape[0], self.propensity, dtype=np.float64)
        if self.propensity_model_ is None:
            raise RuntimeError("call fit() before predict_uplift()")
        return np.clip(_positive_probability(self.propensity_model_, features), 0.01, 0.99)

    def predict_uplift(self, features: Array) -> Array:
        """Propensity-weighted blend of the two effect models.

        The weights are the standard ones: ``g(x) * tau_control + (1 - g(x)) * tau_treated``.
        Where treatment is rare, ``g`` is small and the treated-side effect model —
        the one fit on the scarce arm — is down-weighted in favour of the
        control-side model fit on plentiful data.
        """
        if self.effect_treated_ is None or self.effect_control_ is None:
            raise RuntimeError("call fit() before predict_uplift()")
        weight = self._propensity_scores(features)
        tau_treated = np.asarray(self.effect_treated_.predict(features), dtype=np.float64)
        tau_control = np.asarray(self.effect_control_.predict(features), dtype=np.float64)
        return weight * tau_control + (1.0 - weight) * tau_treated


class CausalForestLearner(UpliftEstimator):
    """EconML's ``CausalForestDML``: honest causal forest with orthogonalisation.

    Args:
        n_estimators: Number of trees.
        min_samples_leaf: Minimum leaf size. Larger values trade resolution for
            stability, which a rare outcome needs.
        random_state: Seed.
        max_depth: Optional depth cap.
        discrete_outcome: Whether the outcome is binary.
    """

    name = "causal-forest"

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 50,
        random_state: int = 0,
        max_depth: int | None = None,
        discrete_outcome: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_depth = max_depth
        self.discrete_outcome = discrete_outcome
        self.model_: Any | None = None

    def fit(self, features: Array, treatment: Array, outcome: Array) -> CausalForestLearner:
        """Fit the forest. See :meth:`UpliftEstimator.fit`."""
        from econml.dml import CausalForestDML

        nuisance_regressor = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=self.random_state
        )
        nuisance_classifier = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=self.random_state
        )
        model = CausalForestDML(
            model_y=nuisance_classifier if self.discrete_outcome else nuisance_regressor,
            model_t=nuisance_classifier,
            discrete_outcome=self.discrete_outcome,
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            cv=2,
            random_state=self.random_state,
        )
        model.fit(outcome.astype(np.float64), treatment, X=features)
        self.model_ = model
        return self

    def predict_uplift(self, features: Array) -> Array:
        """Constant marginal effect of treatment at each covariate value."""
        if self.model_ is None:
            raise RuntimeError("call fit() before predict_uplift()")
        return np.asarray(self.model_.effect(features), dtype=np.float64).ravel()


def build_estimators(
    kind: str = "lightgbm",
    random_state: int = 0,
    propensity: float | None = None,
    include_causal_forest: bool = True,
    classifier: bool = True,
) -> dict[str, UpliftEstimator]:
    """Construct the standard slate of estimators used throughout the study.

    Args:
        kind: Base learner family, see :class:`LearnerFactory`.
        random_state: Seed for every component.
        propensity: Known treatment probability, passed to X-learner.
        include_causal_forest: Set False to skip the slow one.
        classifier: Whether the outcome is binary.

    Returns:
        Mapping of estimator name to an unfitted estimator.
    """
    factory = LearnerFactory(kind=kind, random_state=random_state)
    estimators: dict[str, UpliftEstimator] = {
        "s-learner": SLearner(factory, classifier=classifier),
        "t-learner": TLearner(factory, classifier=classifier),
        "x-learner": XLearner(factory, classifier=classifier, propensity=propensity),
    }
    if include_causal_forest:
        estimators["causal-forest"] = CausalForestLearner(
            random_state=random_state, discrete_outcome=classifier
        )
    return estimators
