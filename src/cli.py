"""Run the study end to end and write the artefacts the memo quotes.

Three stages, in a deliberate order:

1. **validate** — every estimator against known synthetic truth. If an estimator
   cannot recover an effect we planted ourselves, its output on real data means
   nothing, so this runs first and its results are reported whether or not they
   are flattering.
2. **naive** — the wrong analysis, computed properly, plus the two demonstrations
   of why it misleads.
3. **hillstrom** — the real study: cross-validated uplift models, evaluation,
   targeting policy, robustness checks and experiment sizing.

Everything lands in ``results/`` as JSON and CSV so the memo's numbers can be
traced back to a file rather than to a paragraph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.arrays import Array
from src.config import (
    COST_PER_CONTACT,
    DEFAULT_BUDGET_FRACTION,
    MARGIN_RATE,
    N_FOLDS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data import UpliftDataset, dataset_summary, load_hillstrom
from src.estimators import build_estimators
from src.evaluation import (
    bootstrap_qini,
    cross_val_uplift,
    evaluate,
    recovery_error,
)
from src.experiment import (
    cuped_demo,
    design_validation_experiment,
    minimum_detectable_effect,
    required_sample_size,
)
from src.naive import (
    compare_randomised_and_confounded,
    naive_comparison,
    response_versus_uplift_ranking,
)
from src.policy import bootstrap_policy, choose_policy, sleeping_dog_report
from src.robustness import run_all_checks
from src.simulate import describe, simulate


def _jsonable(value: Any) -> Any:
    """Convert numpy, pandas and dataclass objects into JSON-serialisable form.

    Args:
        value: Any value appearing in a results payload.

    Returns:
        A structure containing only JSON-native types.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a results payload to disk.

    Args:
        payload: Results to serialise.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)}")


def stage_validate(
    n: int = 20_000,
    seed: int = RANDOM_SEED,
    results_dir: Path = RESULTS_DIR,
    quick: bool = False,
) -> dict[str, Any]:
    """Validate every estimator against a known individual treatment effect.

    Runs at two treatment shares. The balanced 50/50 case is the easy one. The
    imbalanced 15/85 case is where the learners are supposed to separate: it is
    the setting X-learner was designed for, and the comparison is the evidence for
    or against that claim rather than a citation of it.

    Args:
        n: Synthetic sample size.
        seed: Random seed.
        results_dir: Where to write artefacts.
        quick: Skip the causal forest and use fewer folds.

    Returns:
        The results payload.
    """
    print("\n=== STAGE 1: synthetic validation (known ground truth) ===")
    payload: dict[str, Any] = {}

    for label, share in (("balanced_50_50", 0.5), ("imbalanced_15_85", 0.15)):
        data = simulate(n=n, seed=seed, treatment_share=share)
        truth = describe(data)
        print(
            f"\n{label}: true ATE={truth['true_ate']:.4f}, "
            f"sleeping dogs={truth['sleeping_dog_share']:.1%}, "
            f"treated share={truth['treated_share']:.2f}"
        )

        estimators = build_estimators(
            propensity=share, include_causal_forest=not quick, random_state=seed
        )
        rows: list[dict[str, Any]] = []
        for name, estimator in estimators.items():
            started = time.time()
            oof = cross_val_uplift(
                estimator,
                data.features,
                data.treatment,
                data.outcome,
                n_folds=3 if quick else N_FOLDS,
                seed=seed,
            )
            recovery = recovery_error(oof, data.true_uplift)
            result = evaluate(name, data.outcome, data.treatment, oof, propensity=share)
            rows.append(
                {
                    "model": name,
                    **recovery,
                    "qini_coefficient": result.qini.qini_coefficient,
                    "normalized_qini": result.qini.normalized_qini,
                    "auuc": result.auuc,
                    "transformed_outcome_mse": result.transformed_outcome_mse,
                    "top_decile_uplift": result.top_decile_uplift,
                    "bottom_decile_uplift": result.bottom_decile_uplift,
                    "seconds": time.time() - started,
                }
            )
            print(
                f"  {name:15s} PEHE={recovery['pehe']:.4f} "
                f"kendall={recovery['kendall_tau']:+.3f} "
                f"ATE bias={recovery['ate_bias']:+.4f} "
                f"sd ratio={recovery['sd_ratio']:.2f} "
                f"dogs found={recovery['sleeping_dogs_detected']:.0%}"
            )

        table = pd.DataFrame(rows)
        table.to_csv(results_dir / f"validation_{label}.csv", index=False)
        payload[label] = {"truth": truth, "estimators": table}

    _write_json(payload, results_dir / "validation.json")
    return payload


def stage_naive(
    n: int = 40_000,
    seed: int = RANDOM_SEED,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Compute the naive analysis on Hillstrom and the two synthetic demonstrations.

    Args:
        n: Synthetic sample size for the demonstrations.
        seed: Random seed.
        results_dir: Where to write artefacts.

    Returns:
        The results payload.
    """
    print("\n=== STAGE 2: the naive analysis ===")
    payload: dict[str, Any] = {}

    for arm in ("mens", "womens"):
        for outcome in ("conversion", "visit"):
            dataset = load_hillstrom(arm, outcome)
            comparison = naive_comparison(dataset.outcome, dataset.treatment, dataset.spend)
            payload[f"hillstrom_{arm}_{outcome}"] = comparison
            print(
                f"  {arm:7s} {outcome:11s} treated={comparison.rate_treated:.4f} "
                f"control={comparison.rate_control:.4f} "
                f"lift={comparison.relative_lift:+.1%} p={comparison.p_value:.2e}"
            )

    confounding = compare_randomised_and_confounded(n=n, seed=seed)
    print(
        f"\n  synthetic: true ATE={confounding['true_ate']:.4f}, "
        f"randomised gap={confounding['naive_gap_randomised']:.4f}, "
        f"confounded gap={confounding['naive_gap_confounded']:.4f} "
        f"({confounding['overstatement_factor']:.1f}x overstated)"
    )

    ranking = response_versus_uplift_ranking(n=n, seed=seed)
    print(
        f"  response model captures {ranking['true_uplift_captured_by_response_model']:.0f} "
        f"true incremental conversions vs "
        f"{ranking['true_uplift_captured_by_uplift_model']:.0f} for uplift "
        f"(oracle {ranking['true_uplift_captured_by_oracle']:.0f}, "
        f"random {ranking['true_uplift_captured_by_random']:.0f})"
    )
    print(
        f"  sleeping dogs contacted: response model "
        f"{ranking['sleeping_dogs_contacted_by_response_model']:.0f}, "
        f"uplift model {ranking['sleeping_dogs_contacted_by_uplift_model']:.0f}"
    )

    payload["confounding"] = confounding
    payload["ranking"] = ranking
    _write_json(payload, results_dir / "naive.json")
    return payload


def _run_one_dataset(
    dataset: UpliftDataset,
    label: str,
    seed: int,
    quick: bool,
    budget_fraction: float,
    results_dir: Path,
) -> dict[str, Any]:
    """Fit, evaluate and turn into a policy, for one arm/outcome combination.

    Args:
        dataset: The loaded dataset.
        label: Short name used in filenames and printouts.
        seed: Random seed.
        quick: Skip the causal forest and reduce fold/bootstrap counts.
        budget_fraction: Share of the file the budget can pay for.
        results_dir: Where to write artefacts.

    Returns:
        The results payload for this dataset.
    """
    print(f"\n--- {label}: {dataset.treatment_name}, outcome = {dataset.outcome_name} ---")
    summary = dataset_summary(dataset)
    print(
        f"  n={summary['n']:.0f} treated={summary['n_treated']:.0f} "
        f"rate treated={summary['outcome_rate_treated']:.4f} "
        f"control={summary['outcome_rate_control']:.4f}"
    )

    propensity = dataset.treated_share
    estimators = build_estimators(
        propensity=propensity, include_causal_forest=not quick, random_state=seed
    )
    n_folds = 3 if quick else N_FOLDS
    n_boot = 200 if quick else 500

    scores: dict[str, Array] = {}
    rows: list[dict[str, Any]] = []
    for name, estimator in estimators.items():
        started = time.time()
        oof = cross_val_uplift(
            estimator, dataset.features, dataset.treatment, dataset.outcome, n_folds, seed
        )
        scores[name] = oof
        result = evaluate(name, dataset.outcome, dataset.treatment, oof, propensity=propensity)
        boot = bootstrap_qini(dataset.outcome, dataset.treatment, oof, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "model": name,
                "qini_coefficient": result.qini.qini_coefficient,
                "qini_ci_low": boot["ci_low"],
                "qini_ci_high": boot["ci_high"],
                "normalized_qini": result.qini.normalized_qini,
                "beats_random": boot["beats_random"],
                "p_value_one_sided": boot["p_value_one_sided"],
                "auuc": result.auuc,
                "transformed_outcome_mse": result.transformed_outcome_mse,
                "top_decile_uplift": result.top_decile_uplift,
                "bottom_decile_uplift": result.bottom_decile_uplift,
                "total_incremental": result.qini.total_incremental,
                "predicted_uplift_sd": float(oof.std()),
                "share_predicted_negative": float((oof < 0).mean()),
                "seconds": time.time() - started,
            }
        )
        print(
            f"  {name:15s} qini={result.qini.qini_coefficient:+8.2f} "
            f"[{boot['ci_low']:+.1f}, {boot['ci_high']:+.1f}] "
            f"beats random={str(boot['beats_random']):5s} (p={boot['p_value_one_sided']:.3f}) "
            f"top decile={result.top_decile_uplift:+.4f} bottom={result.bottom_decile_uplift:+.4f}"
        )
        result.deciles.to_csv(results_dir / f"deciles_{label}_{name}.csv", index=False)

    leaderboard = pd.DataFrame(rows).sort_values("qini_coefficient", ascending=False)
    leaderboard.to_csv(results_dir / f"models_{label}.csv", index=False)

    best_model = str(leaderboard.iloc[0]["model"])
    best_score = scores[best_model]
    print(f"  best by Qini: {best_model}")

    budget = budget_fraction * dataset.n * COST_PER_CONTACT
    policy = choose_policy(
        best_model,
        dataset.outcome,
        dataset.treatment,
        best_score,
        dataset.spend,
        budget=budget,
        cost_per_contact=COST_PER_CONTACT,
        margin_rate=MARGIN_RATE,
    )
    policy.frontier.to_csv(results_dir / f"frontier_{label}.csv", index=False)
    profit_ci = bootstrap_policy(
        dataset.outcome,
        dataset.treatment,
        best_score,
        dataset.spend,
        fraction=policy.recommended_fraction,
        n_boot=n_boot,
        seed=seed,
    )
    dogs = sleeping_dog_report(dataset.outcome, dataset.treatment, best_score, dataset.spend)
    print(
        f"  policy: budget caps at {policy.budget_cap_fraction:.0%}; "
        f"target {policy.recommended_fraction:.0%} "
        f"profit=${policy.recommended_profit:,.0f} "
        f"[{profit_ci['ci_low']:,.0f}, {profit_ci['ci_high']:,.0f}] "
        f"vs random-at-same-depth ${policy.random_profit:,.0f}; "
        f"unconstrained optimum {policy.optimal_fraction:.0%} "
        f"(${policy.optimal_profit:,.0f}), treat-all ${policy.treat_all_profit:,.0f}"
    )
    if dogs["n_flagged"] == 0:
        print("  sleeping dogs: none flagged - the model predicts no customer is harmed")
    else:
        print(
            f"  sleeping dogs: {dogs['n_flagged']:,.0f} flagged ({dogs['share_flagged']:.1%}), "
            f"measured uplift={dogs['measured_uplift']:+.4f} "
            f"[{dogs['measured_uplift_ci_low']:+.4f}, {dogs['measured_uplift_ci_high']:+.4f}], "
            f"harm confirmed={bool(dogs['measured_uplift_is_negative'])}"
        )

    checks = run_all_checks(
        estimators[best_model],
        dataset.features,
        dataset.treatment,
        dataset.outcome,
        dataset.spend,
        dataset.feature_names,
        best_score,
        n_placebo=5 if quick else 10,
        seeds=(0, 1, 2) if quick else (0, 1, 2, 3, 4),
        n_folds=3,
        seed=seed,
        n_boot=n_boot,
    )
    balance_table = checks.pop("balance_table")
    sensitivity_table = checks.pop("cost_sensitivity_table")
    if isinstance(balance_table, pd.DataFrame):
        balance_table.to_csv(results_dir / f"balance_{label}.csv", index=False)
    if isinstance(sensitivity_table, pd.DataFrame):
        sensitivity_table.to_csv(results_dir / f"cost_sensitivity_{label}.csv", index=False)

    placebo = checks["placebo"]
    stability = checks["seed_stability"]
    print(
        f"  placebo: mean qini={placebo.mean_qini:+.2f} (sd {placebo.sd_qini:.2f}), "
        f"real={placebo.real_qini:+.2f}, z={placebo.z_score:+.2f}, passes={placebo.passes}"
    )
    print(
        f"  stability: qini {stability['qini_min']:+.1f}..{stability['qini_max']:+.1f}, "
        f"selection overlap={stability['mean_selection_overlap']:.2f}"
    )
    print(f"  ROBUSTNESS OVERALL PASS: {checks['overall_pass']}")

    return {
        "summary": summary,
        "leaderboard": leaderboard,
        "best_model": best_model,
        "policy": {k: v for k, v in policy.__dict__.items() if not isinstance(v, pd.DataFrame)},
        "frontier": policy.frontier,
        "profit_ci": profit_ci,
        "sleeping_dogs": dogs,
        "robustness": checks,
    }


def stage_hillstrom(
    seed: int = RANDOM_SEED,
    quick: bool = False,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run the full study on the real data, for both arms and both outcomes.

    Args:
        seed: Random seed.
        quick: Reduce the compute budget.
        budget_fraction: Share of the file the marketing budget can pay for.
        results_dir: Where to write artefacts.

    Returns:
        The results payload.
    """
    print("\n=== STAGE 3: Hillstrom ===")
    combinations = [("mens", "conversion"), ("mens", "visit"), ("womens", "conversion")]
    payload: dict[str, Any] = {}
    for arm, outcome in combinations:
        label = f"{arm}_{outcome}"
        dataset = load_hillstrom(arm, outcome)
        payload[label] = _run_one_dataset(dataset, label, seed, quick, budget_fraction, results_dir)
    _write_json(payload, results_dir / "hillstrom.json")
    return payload


def stage_experiment(
    hillstrom: dict[str, Any] | None = None,
    seed: int = RANDOM_SEED,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Size the validation experiment and demonstrate CUPED.

    Args:
        hillstrom: Optional results from :func:`stage_hillstrom`, used to take the
            expected effect sizes from what was actually measured rather than
            from an invented assumption.
        seed: Random seed.
        results_dir: Where to write artefacts.

    Returns:
        The results payload.
    """
    print("\n=== STAGE 4: experiment design ===")
    cuped = cuped_demo(n=40_000, seed=seed)
    for metric, stats in cuped.items():
        print(
            f"  CUPED on {metric:11s}: correlation={stats['correlation']:.2f}, "
            f"variance reduction={stats['variance_reduction']:.1%}, "
            f"effective sample x{stats['effective_sample_multiplier']:.2f}"
        )

    dataset = load_hillstrom("mens", "conversion")
    baseline = float(dataset.outcome[dataset.treatment == 0].mean())

    sizing_examples: dict[str, Any] = {
        "detect_20pct_relative_lift": required_sample_size(baseline, relative_mde=0.20),
        "detect_50pct_relative_lift": required_sample_size(baseline, relative_mde=0.50),
        "mde_at_hillstrom_arm_size": minimum_detectable_effect(baseline, 21_306),
    }
    for name, value in sizing_examples.items():
        if isinstance(value, dict):
            print(
                f"  {name}: MDE={value['absolute_mde']:.4f} ({value['relative_mde']:.0%} relative)"
            )
        else:
            print(f"  {name}: n per arm={value.n_per_arm:,} (total {value.n_total:,})")

    # Size the head-to-head test using the uplift actually measured in the top
    # and average slices, rather than a hoped-for number.
    policy_uplift, random_uplift = 0.0, 0.0
    if hillstrom is not None and "mens_conversion" in hillstrom:
        entry = hillstrom["mens_conversion"]
        frontier = entry["frontier"]
        fraction = entry["policy"]["recommended_fraction"]
        if isinstance(frontier, pd.DataFrame):
            nearest = frontier.iloc[(frontier["fraction"] - fraction).abs().idxmin()]
            policy_uplift = float(nearest["observed_uplift"])
        random_uplift = float(
            entry["summary"]["outcome_rate_treated"] - entry["summary"]["outcome_rate_control"]
        )

    design = design_validation_experiment(
        baseline_rate=baseline,
        expected_policy_uplift=policy_uplift,
        expected_random_uplift=random_uplift,
        weekly_volume=20_000,
    )
    print(
        f"  head-to-head test: policy uplift={policy_uplift:.4f} vs "
        f"incumbent={random_uplift:.4f}, difference={design['difference_under_test']:+.4f}"
    )
    if np.isfinite(design["n_per_cell"]):
        print(
            f"  needs {design['n_per_cell']:,.0f} per cell "
            f"({design['n_total']:,.0f} total, {design['weeks_required']:.0f} weeks)"
        )
    else:
        print(
            "  the measured policy uplift does not exceed the incumbent: "
            "no sample size can make this test succeed as specified"
        )

    payload = {
        "cuped": cuped,
        "sizing": sizing_examples,
        "validation_design": design,
        "baseline_rate": baseline,
    }
    _write_json(payload, results_dir / "experiment.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="uplift",
        description="Uplift modelling study: whom to contact under a fixed budget.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "validate", "naive", "hillstrom", "experiment"],
        help="Which stage to run (default: all).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the causal forest and reduce folds and bootstrap replicates.",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=DEFAULT_BUDGET_FRACTION,
        help="Share of the file the marketing budget can pay to contact.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="Where to write artefacts."
    )
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    hillstrom: dict[str, Any] | None = None

    if args.stage in ("all", "validate"):
        stage_validate(seed=args.seed, results_dir=args.results_dir, quick=args.quick)
    if args.stage in ("all", "naive"):
        stage_naive(seed=args.seed, results_dir=args.results_dir)
    if args.stage in ("all", "hillstrom"):
        hillstrom = stage_hillstrom(
            seed=args.seed,
            quick=args.quick,
            budget_fraction=args.budget_fraction,
            results_dir=args.results_dir,
        )
    if args.stage in ("all", "experiment"):
        stage_experiment(hillstrom=hillstrom, seed=args.seed, results_dir=args.results_dir)

    print(f"\nDone in {time.time() - started:.0f}s. Artefacts in {args.results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
