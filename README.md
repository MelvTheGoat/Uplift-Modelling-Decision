# Who converts *because* of the campaign?

An uplift-modelling / causal-inference study that answers one question under a
budget constraint: **given a fixed marketing spend, which customers should receive
the campaign?** Not who is likely to convert — who converts *because* they were
contacted. Those are different people, and the difference is worth money.

The deliverable is **[`MEMO.md`](MEMO.md)** — a two-page decision memo for a
marketing director. Everything else in this repository exists to make the numbers
in it checkable.

## The short version

On the Hillstrom randomised e-mail trial (64,000 customers, 2008):

- The campaign works. Contacted customers converted at 1.25% against 0.57%.
- That fact tells you almost nothing about whom to contact, and this study shows
  precisely why, twice: once by simulating a realistic non-randomised campaign
  list (the headline gap overstates the truth by **7.5×**), and once by comparing
  a conventional response model against an uplift model on data where the true
  individual effect is known (the response model mails **five times as many**
  customers the campaign harms).
- Uplift targeting is worth about **$1,390** on this file, against about **$3,170**
  for simply raising the budget. The budget is the binding constraint, not the
  ranking.
- **The targeting gain failed the placebo test on the primary campaign.** The
  model's top decile is real; the ordering beneath it is not distinguishable from
  noise. This is reported in the memo rather than buried, because a negative
  result honestly reported is the point of the exercise.

## What's here

| File | What it does |
|---|---|
| `MEMO.md` | The deliverable. Two pages, plain language, for a marketing director. |
| `METHODS.md` | Estimator details, metric definitions, assumptions and failure modes. |
| `src/simulate.py` | Synthetic data with a **known** individual treatment effect: randomised, heterogeneous, with a genuinely harmed subgroup and inert noise features. Everything is validated here first. |
| `src/naive.py` | The wrong analysis, computed carefully, plus the two demonstrations of why it misleads. |
| `src/data.py` | Hillstrom loader (with a mirror fallback); optional Criteo-UPLIFT. |
| `src/estimators.py` | S-, T- and X-learners hand-implemented over a swappable sklearn/LightGBM base, plus an EconML causal forest. |
| `src/evaluation.py` | Qini curves and coefficients, AUUC, decile tables, transformed-outcome validation, bootstrap intervals, and an explicit beats-random test. **No AUC.** |
| `src/policy.py` | Budget frontier, optimal targeting depth, profit with uncertainty, sleeping-dog accounting. |
| `src/experiment.py` | Power/MDE calculator in both directions, and a CUPED variance-reduction demo. |
| `src/robustness.py` | Placebo test, covariate balance, cost sensitivity, seed stability. |
| `src/cli.py` | Runs the whole study and writes `results/`. |
| `tests/` | 135 tests, synthetic data only — no network, no dataset download. |

## Reproducing it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m src.cli all          # full study, ~10 minutes
```

Individual stages, in the order they are meant to be read:

```bash
python -m src.cli validate     # estimators vs known synthetic truth  (~30s)
python -m src.cli naive        # the wrong answer, and why            (~5s)
python -m src.cli hillstrom    # the real study                       (~9m)
python -m src.cli experiment   # power, MDE and CUPED                 (~10s)

python -m src.cli all --quick  # skip the causal forest, fewer folds  (~2m)
```

Artefacts land in `results/` as JSON and CSV: model leaderboards, decile tables,
the profit frontier, balance and cost-sensitivity tables. Every number in the memo
traces back to one of them.

The Hillstrom CSV downloads on first use and caches in `data/` (gitignored). The
canonical `minethatdata.com` URL is plain HTTP and blocked by many networks, so
`src/data.py` falls back to a byte-identical GitHub mirror automatically.

## Development

```bash
pytest -q            # 135 tests, all on synthetic data
ruff check . && ruff format --check .
mypy                 # strict
```

All three are clean, and CI (`.github/workflows/ci.yml`) runs them on Python
3.10–3.12. **The test suite never touches the real dataset** — it runs entirely on
data whose true individual treatment effect is known, so a test can assert that an
estimator is *correct* rather than that last year's output has not changed.

## How the study is organised, and why in that order

**1. Synthetic ground truth first.** On real data the individual treatment effect
is unobservable — each customer is either mailed or not, never both — so a model
that looks good there might just be reproducing our own mistakes. `src/simulate.py`
generates data with a known effect function, and no estimator or metric is trusted
on Hillstrom until it has recovered that known answer. The recovery errors are
reported in `results/validation_*.csv` whether or not they are flattering. They
are also how the documented failure modes get *measured* rather than asserted:
S-learner's predicted effects come out at 0.65× the true spread (it shrinks
heterogeneity toward zero), T-learner's at 1.69× (it inflates noise), and
X-learner beats T-learner under treatment imbalance exactly as claimed.

**2. The naive analysis second**, because it frames everything else. See
`MEMO.md` §2.

**3. Estimators, evaluation, policy, robustness.** The ordering matters: a policy
recommendation is only as good as the evaluation behind it, and the evaluation is
only as good as the robustness checks that try to break it.

Two methodological choices worth flagging:

- **No AUC anywhere.** AUC measures how well a score separates buyers from
  non-buyers. An uplift model is not trying to do that, and the *best* uplift model
  would score badly on it — a customer who was always going to buy has zero uplift
  and should rank low. High AUC with worthless targeting is the normal case.
- **Incremental effects are always measured from the randomised holdout inside the
  selected group, never read off the model's own predictions.** Using predictions
  as values is circular, and it is the most common way an uplift analysis
  overstates itself.

## Assumptions you can change

Everything the business might argue with lives in `src/config.py`: cost per
contact ($0.10), gross margin on incremental revenue (30%), budget as a share of
the file (30%). Edit and re-run. `src/robustness.py` sweeps the first two
automatically — and the answer is genuinely sensitive to them, which the memo says
out loud.

## Data

**Hillstrom MineThatData E-Mail Analytics Challenge** (2008) — 64,000 customers,
randomised three ways: men's e-mail, women's e-mail, no e-mail. Outcomes: visit,
conversion, spend. Public, no authentication.

**Criteo-UPLIFT v2** is supported behind `load_criteo()` and optional throughout;
nothing blocks on it.

## Licence

MIT. All dependencies are free and run locally — nothing here calls a paid API.
