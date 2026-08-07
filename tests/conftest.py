"""Shared fixtures. Every test in this suite runs on synthetic data only.

Nothing here touches the network or the Hillstrom cache. That is deliberate: the
test suite has to pass in CI without a dataset download, and a test whose
expected value comes from real data would be asserting that the past has not
changed rather than that the code is correct.
"""

from __future__ import annotations

import pytest

from src.estimators import LearnerFactory
from src.simulate import SyntheticData, simulate, simulate_confounded


@pytest.fixture(scope="session")
def small() -> SyntheticData:
    """A small randomised sample, sized so the whole suite stays quick."""
    return simulate(n=4_000, seed=11)


@pytest.fixture(scope="session")
def medium() -> SyntheticData:
    """A larger randomised sample for assertions that need estimation precision."""
    return simulate(n=20_000, seed=12)


@pytest.fixture(scope="session")
def imbalanced() -> SyntheticData:
    """A sample with only 15% treated, where X-learner is meant to shine."""
    return simulate(n=20_000, seed=13, treatment_share=0.15)


@pytest.fixture(scope="session")
def confounded() -> SyntheticData:
    """Same effects, non-random assignment."""
    return simulate_confounded(n=20_000, seed=12)


@pytest.fixture
def fast_factory() -> LearnerFactory:
    """A deliberately small LightGBM so tests fit models in well under a second."""
    return LearnerFactory(kind="lightgbm", random_state=0, n_estimators=40, num_leaves=7)
