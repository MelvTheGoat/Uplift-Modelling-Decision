"""The JavaScript sample-size calculator must agree with the Python one.

``web/js/stats.js`` re-implements three functions from ``src/experiment.py`` so
that the hosted site can respond to numbers a visitor types instead of serving
a lookup table. Two implementations of one formula is a bug waiting to happen:
the obvious failure is someone fixing a formula on one side only, and the
result is a site that quietly contradicts the study it is presenting.

So this test runs both over a grid and fails if they disagree.

It needs Node to be installed, and skips if it is not — Node is not a
dependency of the study itself, and CI should not fail on a machine that only
has Python. CI does install it, so the check does run there.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.config import ROOT
from src.experiment import (
    design_validation_experiment,
    minimum_detectable_effect,
    required_sample_size,
)

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: Baseline rates spanning the range the site allows, including the 0.57% rate
#: the real campaign actually ran at, where the normal approximation is at its
#: least comfortable and the two ports are most likely to disagree.
BASELINE_RATES = (0.0057, 0.01, 0.05, 0.12, 0.30)

#: Effects as a fraction of baseline.
RELATIVE_EFFECTS = (0.05, 0.10, 0.20, 0.50, 1.00)

#: Arm sizes for the inverse direction.
ARM_SIZES = (500, 5_000, 21_306, 150_000)


def _run_node(script: str) -> object:
    """Execute a snippet against ``web/js/stats.js`` and parse what it prints.

    Args:
        script: JavaScript body. It must print one JSON document on stdout.

    Returns:
        The decoded result.
    """
    module = (ROOT / "web" / "js" / "stats.js").as_uri()
    source = f"import * as stats from {json.dumps(module)};\n{script}\n"
    path = Path(ROOT / "web" / ".parity.mjs")
    path.write_text(source, encoding="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(NODE), str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)
    return json.loads(completed.stdout)


def test_norm_inv_matches_scipy() -> None:
    """The inverse normal CDF underpins every other number here."""
    from scipy import stats as scipy_stats

    probabilities = [0.001, 0.025, 0.05, 0.2, 0.5, 0.8, 0.9, 0.95, 0.975, 0.999]
    got = _run_node(f"console.log(JSON.stringify({json.dumps(probabilities)}.map(stats.normInv)));")
    assert isinstance(got, list)
    for probability, value in zip(probabilities, got, strict=True):
        expected = float(scipy_stats.norm.ppf(probability))
        assert value == pytest.approx(expected, abs=1e-9), f"at p={probability}"


def test_required_sample_size_matches_python() -> None:
    """Forward direction: effect in, customers out."""
    cases = [
        {"baselineRate": rate, "relativeMde": effect}
        for rate in BASELINE_RATES
        for effect in RELATIVE_EFFECTS
    ]
    got = _run_node(
        f"console.log(JSON.stringify({json.dumps(cases)}.map("
        "(c) => stats.requiredSampleSize(c).nPerArm)));"
    )
    assert isinstance(got, list)
    for case, value in zip(cases, got, strict=True):
        expected = required_sample_size(
            baseline_rate=case["baselineRate"], relative_mde=case["relativeMde"]
        ).n_per_arm
        assert value == expected, f"at {case}: js {value} vs python {expected}"


def test_minimum_detectable_effect_matches_python() -> None:
    """Reverse direction: customers in, smallest detectable effect out."""
    cases = [
        {"baselineRate": rate, "nPerArm": size} for rate in BASELINE_RATES for size in ARM_SIZES
    ]
    got = _run_node(
        f"console.log(JSON.stringify({json.dumps(cases)}.map("
        "(c) => stats.minimumDetectableEffect(c).absoluteMde)));"
    )
    assert isinstance(got, list)
    for case, value in zip(cases, got, strict=True):
        expected = minimum_detectable_effect(
            baseline_rate=case["baselineRate"], n_per_arm=int(case["nPerArm"])
        )["absolute_mde"]
        # Both sides bisect 200 times from the same bracket, so they should
        # agree to far better than this; the tolerance is here to tolerate
        # float64 ordering differences, not genuine disagreement.
        assert value == pytest.approx(expected, rel=1e-9), f"at {case}"


def test_design_validation_matches_python() -> None:
    """The head-to-head design, which is the number the site leads with."""
    cases = [
        {
            "baselineRate": 0.005726086548390125,
            "expectedPolicyUplift": policy,
            "expectedRandomUplift": 0.0068050065196156,
            "weeklyVolume": 20_000,
            "cupedVarianceReduction": reduction,
        }
        for policy in (0.009, 0.0091, 0.012, 0.02)
        for reduction in (0.0, 0.2)
    ]
    got = _run_node(
        f"console.log(JSON.stringify({json.dumps(cases)}.map(stats.designValidation)));"
    )
    assert isinstance(got, list)
    for case, value in zip(cases, got, strict=True):
        assert isinstance(value, dict)
        expected = design_validation_experiment(
            baseline_rate=case["baselineRate"],
            expected_policy_uplift=case["expectedPolicyUplift"],
            expected_random_uplift=case["expectedRandomUplift"],
            weekly_volume=int(case["weeklyVolume"]),
            cuped_variance_reduction=case["cupedVarianceReduction"],
        )
        assert value["nPerCell"] == expected["n_per_cell"], f"at {case}"
        assert value["nTotal"] == expected["n_total"], f"at {case}"
        assert value["weeksRequired"] == expected["weeks_required"], f"at {case}"


def test_design_validation_reports_an_impossible_test() -> None:
    """A policy expected to do no better than the incumbent cannot be sized.

    This is the branch that matters for the site: a visitor who types a policy
    uplift below the incumbent's should get "this test cannot be run", not a
    confident sample size computed from a negative effect.
    """
    got = _run_node(
        "console.log(JSON.stringify(stats.designValidation("
        "{baselineRate: 0.0057, expectedPolicyUplift: 0.001, expectedRandomUplift: 0.002})));"
    )
    assert isinstance(got, dict)
    assert got["impossible"] is True
    assert got["differenceUnderTest"] == pytest.approx(-0.001)
    # JSON has no Infinity, so `JSON.stringify` turns it into null. The site
    # reads the `impossible` flag rather than the sizes, which is why that flag
    # exists at all.
    assert got["nPerCell"] is None
    assert got["nTotal"] is None
