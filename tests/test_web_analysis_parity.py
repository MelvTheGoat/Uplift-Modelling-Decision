"""The browser analysis engine must agree with the Python one.

``web/js/analysis.js`` re-implements the core of ``src/evaluation.py``,
``src/policy.py`` and ``src/naive.py`` so that a visitor can run the study's
analysis on their own campaign without installing anything. That is the whole
value of the tool, and it is worthless if the two drift apart — a site that
quietly disagrees with the study it is presenting is worse than no site.

So this generates a campaign, runs both implementations over it, and compares
every number.

**Why the scores are forced to be distinct.** Both implementations break ties
in the ranking at random, deliberately: row order in a sorted export carries
real signal and would flatter a model that has none. They cannot use the *same*
random numbers, because reproducing NumPy's PCG64 bit stream in JavaScript is
not worth the trouble. With tied scores the two can therefore order tied rows
differently and diverge in the last decimal place. With distinct scores — which
any real-valued model produces — tie-breaking never fires and the two agree
exactly. That is the contract being tested here, and ``analysis.js`` documents
the limitation rather than hiding it.

Needs Node, and skips without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.config import ROOT
from src.evaluation import qini_curve, uplift_by_decile
from src.naive import naive_comparison
from src.policy import profit_frontier, random_targeting_profit, sleeping_dog_report
from src.simulate import simulate

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

COST = 0.12
MARGIN = 0.25


@pytest.fixture(scope="module")
def campaign() -> dict[str, object]:
    """A synthetic campaign with strictly distinct uplift scores.

    The score is the true uplift plus a tiny deterministic nudge per row, which
    guarantees no two customers tie while leaving the ranking essentially the
    true one.
    """
    data = simulate(n=4000, seed=404)
    nudge = np.linspace(0.0, 1e-9, data.n)
    score = data.true_uplift + nudge
    assert len(np.unique(score)) == data.n, "scores must be distinct for an exact comparison"
    return {
        "outcome": data.outcome.astype(np.float64),
        "treatment": data.treatment.astype(np.int64),
        "spend": data.spend.astype(np.float64),
        "score": score,
    }


def _run_js(campaign: dict[str, object], script: str) -> object:
    """Run a snippet against ``web/js/analysis.js`` with the campaign in scope.

    Args:
        campaign: The arrays, which are injected as ``data``.
        script: JavaScript that prints one JSON document.

    Returns:
        The decoded result.
    """
    module = (ROOT / "web" / "js" / "analysis.js").as_uri()
    payload = json.dumps({key: np.asarray(value).tolist() for key, value in campaign.items()})
    source = (
        f"import * as a from {json.dumps(module)};\n"
        f"const data = {payload};\n"
        f"const outcome = data.outcome, treatment = data.treatment,"
        f" spend = data.spend, score = data.score;\n"
        f"{script}\n"
    )
    path = ROOT / "web" / ".analysis-parity.mjs"
    path.write_text(source, encoding="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(NODE), str(path)], capture_output=True, text=True, timeout=180, check=True
        )
    finally:
        path.unlink(missing_ok=True)
    return json.loads(completed.stdout)


def test_naive_comparison_matches(campaign: dict[str, object]) -> None:
    """The headline 'did it work?' numbers."""
    got = _run_js(campaign, "console.log(JSON.stringify(a.naiveComparison(outcome, treatment)));")
    assert isinstance(got, dict)
    expected = naive_comparison(
        campaign["outcome"],  # type: ignore[arg-type]
        campaign["treatment"],  # type: ignore[arg-type]
    )
    assert got["rateTreated"] == pytest.approx(expected.rate_treated)
    assert got["rateControl"] == pytest.approx(expected.rate_control)
    assert got["absoluteDifference"] == pytest.approx(expected.absolute_difference)
    assert got["stderr"] == pytest.approx(expected.stderr)
    assert got["ciLow"] == pytest.approx(expected.ci_low)
    assert got["ciHigh"] == pytest.approx(expected.ci_high)
    # The p-value goes through an error-function approximation on the JS side,
    # so it gets a looser tolerance than the arithmetic above.
    assert got["pValue"] == pytest.approx(expected.p_value, abs=1e-9)


def test_qini_coefficient_matches(campaign: dict[str, object]) -> None:
    """The single number the whole targeting argument rests on."""
    got = _run_js(
        campaign,
        "const q = a.qiniCurve(outcome, treatment, score);"
        "console.log(JSON.stringify({c: q.qiniCoefficient, t: q.totalIncremental,"
        " auc: q.qiniAuc, rand: q.randomAuc}));",
    )
    assert isinstance(got, dict)
    expected = qini_curve(
        campaign["outcome"],  # type: ignore[arg-type]
        campaign["treatment"],  # type: ignore[arg-type]
        campaign["score"],  # type: ignore[arg-type]
    )
    assert got["t"] == pytest.approx(expected.total_incremental, rel=1e-12)
    assert got["auc"] == pytest.approx(expected.qini_auc, rel=1e-12)
    assert got["rand"] == pytest.approx(expected.random_auc, rel=1e-12)
    assert got["c"] == pytest.approx(expected.qini_coefficient, rel=1e-12)


def test_decile_table_matches(campaign: dict[str, object]) -> None:
    """Every row of the table that is meant to convince a sceptic.

    This also pins ``arraySplit``: NumPy puts the remainder in the earliest
    groups, and getting that backwards shifts every decile boundary by a row in
    a way nothing else would catch.
    """
    got = _run_js(
        campaign,
        "console.log(JSON.stringify(a.upliftByDecile(outcome, treatment, score)));",
    )
    assert isinstance(got, list)
    expected = uplift_by_decile(
        campaign["outcome"],  # type: ignore[arg-type]
        campaign["treatment"],  # type: ignore[arg-type]
        campaign["score"],  # type: ignore[arg-type]
    )
    assert len(got) == len(expected)
    for row, (_, want) in zip(got, expected.iterrows(), strict=True):
        assert isinstance(row, dict)
        where = f"decile {want['decile']:.0f}"
        assert row["n"] == want["n"], where
        assert row["nTreated"] == want["n_treated"], where
        assert row["nControl"] == want["n_control"], where
        assert row["predictedUplift"] == pytest.approx(want["predicted_uplift"]), where
        assert row["observedUplift"] == pytest.approx(want["observed_uplift"]), where
        assert row["stderr"] == pytest.approx(want["stderr"]), where
        assert row["ciLow"] == pytest.approx(want["ci_low"]), where
        assert row["ciHigh"] == pytest.approx(want["ci_high"]), where


def test_profit_frontier_matches(campaign: dict[str, object]) -> None:
    """Every depth of the frontier, including how the cut-off is rounded.

    The rounding is the subtle one. Python's ``round`` is half-to-even and
    JavaScript's ``Math.round`` is half-up; with 40 cut-offs over 4,000
    customers the boundary is hit repeatedly, and the wrong one silently shifts
    a cut-off by one customer.
    """
    got = _run_js(
        campaign,
        f"console.log(JSON.stringify(a.profitFrontier(outcome, treatment, score, spend,"
        f" {{costPerContact: {COST}, marginRate: {MARGIN}}})));",
    )
    assert isinstance(got, list)
    expected = profit_frontier(
        campaign["outcome"],  # type: ignore[arg-type]
        campaign["treatment"],  # type: ignore[arg-type]
        campaign["score"],  # type: ignore[arg-type]
        campaign["spend"],  # type: ignore[arg-type]
        cost_per_contact=COST,
        margin_rate=MARGIN,
    )
    assert len(got) == len(expected)
    for row, (_, want) in zip(got, expected.iterrows(), strict=True):
        assert isinstance(row, dict)
        where = f"fraction {want['fraction']:.3f}"
        assert row["nTargeted"] == want["n_targeted"], where
        assert row["incrementalConversions"] == pytest.approx(want["incremental_conversions"]), (
            where
        )
        assert row["incrementalRevenue"] == pytest.approx(want["incremental_revenue"]), where
        assert row["incrementalProfit"] == pytest.approx(want["incremental_profit"]), where
        assert row["observedUplift"] == pytest.approx(want["observed_uplift"]), where


def test_random_targeting_profit_matches(campaign: dict[str, object]) -> None:
    """The straight line every targeting claim is measured against."""
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    got = _run_js(
        campaign,
        f"console.log(JSON.stringify({json.dumps(fractions)}.map((f) =>"
        f" a.randomTargetingProfit(outcome, treatment, spend, f,"
        f" {{costPerContact: {COST}, marginRate: {MARGIN}}}))));",
    )
    assert isinstance(got, list)
    for fraction, value in zip(fractions, got, strict=True):
        expected = random_targeting_profit(
            campaign["outcome"],  # type: ignore[arg-type]
            campaign["treatment"],  # type: ignore[arg-type]
            campaign["spend"],  # type: ignore[arg-type]
            fraction,
            cost_per_contact=COST,
            margin_rate=MARGIN,
        )
        assert value == pytest.approx(expected), f"at fraction {fraction}"


def test_sleeping_dog_report_matches(campaign: dict[str, object]) -> None:
    """The flagged group, and what measuring them actually shows."""
    got = _run_js(
        campaign,
        f"console.log(JSON.stringify(a.sleepingDogReport(outcome, treatment, score, spend,"
        f" {{costPerContact: {COST}, marginRate: {MARGIN}}})));",
    )
    assert isinstance(got, dict)
    expected = sleeping_dog_report(
        campaign["outcome"],  # type: ignore[arg-type]
        campaign["treatment"],  # type: ignore[arg-type]
        campaign["score"],  # type: ignore[arg-type]
        campaign["spend"],  # type: ignore[arg-type]
        cost_per_contact=COST,
        margin_rate=MARGIN,
    )
    assert got["nFlagged"] == expected["n_flagged"]
    assert got["shareFlagged"] == pytest.approx(expected["share_flagged"])
    assert got["meanPredictedUplift"] == pytest.approx(expected["mean_predicted_uplift"])
    assert got["measuredUplift"] == pytest.approx(expected["measured_uplift"])
    assert got["profitIfTreated"] == pytest.approx(expected["profit_if_treated"])


def test_round_half_to_even_matches_python() -> None:
    """The rounding helper, pinned directly rather than only through the frontier."""
    values = [0.5, 1.5, 2.5, 3.5, -0.5, 2.4999, 2.5001, 1000.5, 1001.5]
    got = _run_js({}, f"console.log(JSON.stringify({json.dumps(values)}.map(a.roundHalfToEven)));")
    assert isinstance(got, list)
    for value, result in zip(values, got, strict=True):
        assert result == round(value), f"at {value}: js {result} vs python {round(value)}"


def test_placebo_test_reports_a_weak_signal_as_weak(campaign: dict[str, object]) -> None:
    """A ranking made of pure noise must not pass the shuffle test.

    This is a behavioural check rather than a parity one: the two
    implementations use different random streams, so their placebo numbers
    cannot match exactly. What must hold is the verdict — scoring noise should
    not look like a finding.
    """
    got = _run_js(
        campaign,
        "const rng = a.makeRng(99);"
        "const noise = Array.from({length: outcome.length}, () => rng());"
        "console.log(JSON.stringify(a.placeboTest(outcome, treatment, noise,"
        " {nReplicates: 15})));",
    )
    assert isinstance(got, dict)
    assert got["passes"] is False, "a random ranking must not pass the shuffle test"


def test_placebo_test_accepts_a_genuine_signal(campaign: dict[str, object]) -> None:
    """And the true uplift, which is as strong a signal as exists, must pass."""
    got = _run_js(
        campaign,
        "console.log(JSON.stringify(a.placeboTest(outcome, treatment, score, {nReplicates: 15})));",
    )
    assert isinstance(got, dict)
    assert got["passes"] is True, "ranking by the true effect must survive the shuffle test"
