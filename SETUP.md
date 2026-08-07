# Fresh start — no prior history

These are the project files with **no git history attached**. Nothing to inherit,
nothing to rewrite: you `git init`, commit, and every commit is yours.

## Publish it

Create an empty repo on GitHub (no README, no .gitignore, no licence), then:

```powershell
cd uplift-targeting
git init
git add .
git commit -m "Uplift modelling and causal targeting study"
git branch -M main
git remote add origin https://github.com/MelvTheGoat/<new-repo>.git
git push -u origin main
```

If you'd rather have the work land as several commits than one, a sensible split:

```powershell
git add pyproject.toml requirements.txt .gitignore src/config.py src/arrays.py src/__init__.py
git commit -m "Project scaffolding: packaging, lint and type config, shared constants"

git add src/simulate.py
git commit -m "Synthetic data generator with a known individual treatment effect"

git add src/data.py src/estimators.py
git commit -m "Hillstrom loader and the S-, T-, X-learner and causal forest estimators"

git add src/evaluation.py src/policy.py
git commit -m "Qini, AUUC and decile evaluation, plus the budget-constrained policy"

git add src/naive.py src/experiment.py src/robustness.py src/cli.py
git commit -m "Naive analysis, experiment design, robustness checks and the CLI"

git add tests .github
git commit -m "Test suite (136 tests, synthetic data only) and CI workflow"

git add results
git commit -m "Generated results artefacts"

git add MEMO.md METHODS.md README.md CODE_WALKTHROUGH.md SETUP.md
git commit -m "Decision memo, methods, README and code walkthrough"
```

## About the existing repo

`Uplift-Modelling-Decision-Memo` on GitHub already has the old history, so it will
keep showing a second contributor until you either delete that repo or force-push
rewritten history over it. Starting fresh here sidesteps it entirely — just delete
the old one once this is up.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python -m src.cli all      # ~10 min, or --quick for ~2 min
```

The Hillstrom CSV (4 MB) downloads on first run into `data/` and is gitignored, so
it is not in this archive. `results/` is included, so the memo's numbers are
readable without re-running anything.

## Checks

```powershell
pytest -q          # 136 passed
ruff check . ; ruff format --check .
mypy
```

All three are clean. Start reading at `MEMO.md`; `CODE_WALKTHROUGH.md` explains how
every piece was built and why.
