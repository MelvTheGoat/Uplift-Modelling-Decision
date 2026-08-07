# Building this from an empty folder

Right. Blank editor, empty directory, and a question from a marketing director:
*given a fixed budget, who should get the campaign?* Let's talk through what we're
about to build before we build any of it, because the order matters more than
usual here and the wrong order will waste a day.

I'm going to walk this the way it actually got built — which is not alphabetical
and is not "models first." Along the way I'll stop at the lines where the obvious
version is wrong, because in this project the obvious version usually produces a
number that looks *better*, not worse, and that's the dangerous kind of wrong.

---

## The thing that dictates everything: we can't see the answer

Here's the problem in one sentence. For each customer we want
`τ(x) = E[Y(1) − Y(0) | X = x]` — how much more likely they are to buy if we mail
them than if we don't. We will never observe both terms for anybody. We mail them
or we don't. The quantity we're estimating is, per customer, permanently
unobservable.

That fact drives the entire architecture. If we can't check our answer on real
data, then a model that produces a beautiful Qini curve on Hillstrom might be
producing a beautiful Qini curve *because our Qini function is wrong*. Both
mistakes look identical from the outside.

So: **we build a fake world where we know the answer, first.** Before we touch the
real dataset, before we write a single estimator. Every metric and every learner
has to recover a truth we planted ourselves. Only then are they allowed near
Hillstrom.

That gives us the build order:

1. Constants (everything else imports them)
2. The simulator (the yardstick)
3. The estimator interface (the shape everything downstream depends on)
4. Evaluation (needs the interface, validated against the simulator)
5. Policy (needs evaluation's ranking helper)
6. The real data loader
7. The naive analysis (needs the simulator and one estimator)
8. Experiment design (needs the simulator)
9. Robustness (needs everything)
10. The CLI that runs it
11. Tests
12. The memo

If you build the estimators first you'll write an evaluation function to check
them, and you'll tune the evaluation function until the estimators look good.
That's how you end up with a study that proves its own assumptions.

---

## 1. `src/config.py` — the numbers someone will argue with

**Why this file exists:** three numbers in this project are not measurements.
They're assumptions: what a contact costs, what margin we keep, how big the
budget is. Every profit figure in the memo is a function of them. If they're
scattered through function bodies as literals, then "what if a contact actually
costs 25 cents?" becomes a grep-and-pray exercise instead of a one-line edit.

**Why not fold it into the modules that use them?** Because `policy.py`,
`robustness.py` and `cli.py` all need the same values, and if each one carries
its own default they'll drift. One will get updated, two won't, and the memo will
quote three mutually inconsistent profit numbers with no error anywhere.

The paths block is unremarkable:

```python
# src/config.py:15-17
ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT / "data"
RESULTS_DIR: Final[Path] = ROOT / "results"
```

`.resolve()` matters — without it, `parent.parent` on a relative invocation path
gives you something that works when you run from the repo root and breaks when
you run from anywhere else.

Two URLs for the same file:

```python
# src/config.py:22-30
HILLSTROM_URL: Final[str] = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
#: Byte-identical copy of the same file, used when the canonical host is blocked.
HILLSTROM_MIRROR_URL: Final[str] = (
    "https://raw.githubusercontent.com/bscan/uplift/master/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
```

The canonical URL is plain HTTP. Corporate proxies, CI runners and locked-down
sandboxes block it routinely. If you only ship the canonical URL, your project
"doesn't work" for a large fraction of people and the failure mode is a confusing
network error rather than "your firewall ate it." The mirror is byte-identical —
I checked the md5 — so the fallback isn't a different dataset, it's the same
bytes from a host nobody blocks.

Now the interesting constants:

```python
# src/config.py:40-42
RANDOM_SEED: Final[int] = 20080320  # the date on the Hillstrom file, for luck
N_BOOTSTRAP: Final[int] = 500
N_FOLDS: Final[int] = 5
```

`N_BOOTSTRAP = 500` is a resolution decision, not a taste decision. The
beats-random test reports an empirical p-value as *the fraction of replicates
below zero*. With 500 replicates the smallest non-zero p-value you can express is
1/500 = 0.002. With 50 replicates it'd be 0.02, which is uncomfortably close to
the 0.05 threshold you're testing against — you'd have p-values that jump in
steps almost as large as the decision boundary. 500 is the point where the
discretisation stops mattering and the runtime is still tolerable.

`N_FOLDS = 5` on a 0.9% conversion rate: 42,613 rows × 0.9% ≈ 380 converters,
split five ways is ~76 converters per fold. That's thin but workable. Ten folds
would give ~38 per fold and the per-fold models start being noise. Three folds
throws away a third of the training data. Five is the compromise.

And the economics:

```python
# src/config.py:51-57
COST_PER_CONTACT: Final[float] = 0.10

#: Gross margin retained per dollar of incremental revenue.
MARGIN_RATE: Final[float] = 0.30

#: Share of the file we can afford to contact in one campaign, if a budget is imposed.
DEFAULT_BUDGET_FRACTION: Final[float] = 0.30
```

**What changes if you move these.** `COST_PER_CONTACT` is the one that actually
swings the conclusion. At $0.10 the profit-maximising campaign mails 85% of the
file. At $0.25 it mails 35%. At $0.50, mailing everyone *loses* $11,500 and the
optimum collapses to 2.5%. The whole shape of the recommendation is a function of
this number, which is why `robustness.cost_sensitivity` sweeps it and why the
memo says out loud that confirming the true fully-loaded cost matters more than
the model does. `MARGIN_RATE` scales revenue linearly, so it moves profit but
barely moves the *optimal fraction* — the cut-off is where marginal margin equals
marginal cost, and margin scales both sides of a comparison against a fixed cost,
so it does move the crossing point, just far less sharply than cost does.

`DEFAULT_BUDGET_FRACTION` is not a real budget — it's a stand-in that lets the
study demonstrate what a binding constraint does. Worth being honest about: in a
real engagement, someone hands you a dollar figure and this constant disappears.

---

## 2. `src/arrays.py` — a confession about build order

I'm putting this second because that's where it belongs *logically*, but it was
written near the end, under duress, when I turned mypy on in strict mode and got
141 errors, roughly 120 of which were the same one: `Missing type arguments for
generic type "ndarray"`.

```python
# src/arrays.py:22
Array: TypeAlias = npt.NDArray[Any]
```

Under `disallow_any_generics`, a bare `np.ndarray` annotation is an error. The
fix is to say what's in the array. The problem is that in this project, it
genuinely varies — `outcome` is `int64` from Hillstrom, `float64` in half the
tests, and continuous revenue when you run the estimators in regression mode.
Every function converts explicitly at the top of its body:

```python
# src/evaluation.py:154-156
outcome = np.asarray(outcome, dtype=np.float64)
treatment = np.asarray(treatment, dtype=np.int64)
score = np.asarray(score, dtype=np.float64)
```

So pinning `NDArray[np.float64]` in the signature would document a restriction the
code doesn't impose, and would force a pile of `cast()` calls at every call site
to say something untrue.

**Would I write it differently now?** Yes, slightly. The honest version is a
`Protocol` or a pair of aliases (`FloatArray`, `IntArray`) with the conversions
happening at module boundaries rather than inside every function. What's here is
the pragmatic version: it satisfies the checker, it's explicit rather than
implicit, and the docstring admits what it's doing. But it *is* a type alias whose
job is to make a linter stop talking, and you should know that when you read it.

---

## 3. `src/simulate.py` — the yardstick

**Why this file exists:** it's the only place in the project where `τ(x)` is
knowable. Everything downstream is validated here or it isn't validated at all.

**Why it's separate from the estimators:** because the moment ground truth and the
thing being measured live in the same module, someone will import the truth into
the estimator "just for a diagnostic" and you'll have leakage you can't see.
Physical separation is cheap insurance.

We need four properties, and they're not negotiable:

- **randomisation** with no confounding, so the naive difference is provably
  unbiased and we have a clean baseline;
- **heterogeneity**, so there's something for an uplift model to find;
- **a genuinely harmed subgroup**, because sleeping dogs are the whole argument
  for uplift modelling and we need to know whether our methods find them;
- **inert noise columns**, to catch estimators that manufacture signal.

### The effect function, and the trap I actually fell into

```python
# src/simulate.py:146-149
x0, x1, x2 = features[:, 0], features[:, 1], features[:, 2]
responsive = BASE_UPLIFT + UPLIFT_SLOPE_X0 * x0 + UPLIFT_BONUS_X1 * (x1 > 0.0)
uplift = np.where(x2 < SLEEPING_DOG_THRESHOLD, SLEEPING_DOG_UPLIFT, responsive)
return uplift.astype(np.float64)
```

Line 148 is `np.where` — an **override**. The first version I wrote was the
natural one:

```python
# THE VERSION I ACTUALLY WROTE FIRST — it's wrong, and here's the tell
uplift = (
    BASE_UPLIFT
    + UPLIFT_SLOPE_X0 * x0
    + UPLIFT_BONUS_X1 * (x1 > 0.0)
    - SLEEPING_DOG_PENALTY * (x2 < SLEEPING_DOG_THRESHOLD)
)
```

Subtract a penalty. Reads fine. Runs fine. And then this test failed:

```python
# tests/test_simulate.py:72
assert medium.true_uplift[designed].max() < 0.0
```

Because with a penalty of 0.12, a customer at `x2 = −1.5` (deep in the subgroup)
but `x0 = 3` (very responsive) comes out at `0.02 + 0.15 + 0.03 − 0.12 = +0.08`.
Positive. So "the sleeping-dog subgroup" contained customers who weren't sleeping
dogs, and the group had no clean right answer to test against.

Why that's insidious rather than merely annoying: the *aggregate* statistics
still looked perfect. Mean uplift in the subgroup was clearly negative, the
detection rates were plausible, `describe()` reported exactly what you'd expect.
Only a max-over-the-subgroup assertion catches it. If I'd written a softer test —
`assert mean < 0` — the ambiguity would have survived into every downstream claim
about sleeping-dog detection, and "the model found 74% of sleeping dogs" would
have been measured against a group that was 15% not-sleeping-dogs.

The override version has an unambiguous right answer: every member of the
subgroup has uplift exactly `SLEEPING_DOG_UPLIFT`, regardless of how promising
they look on every other axis.

### The baseline function and the trap it sets deliberately

```python
# src/simulate.py:165-167
x0, x3 = features[:, 0], features[:, 3]
logit = -2.0 + 0.35 * x0 + 1.20 * x3
return 1.0 / (1.0 + np.exp(-logit))
```

`x3` has a coefficient of **1.20 on the baseline and 0.00 on the uplift**. That
asymmetry is the entire point of the memo, encoded in one line. Customers high on
`x3` are the most likely to convert and are worth *nothing extra* from being
mailed. A response model will rank them at the top. An uplift model shouldn't.
`x0` appears in both (0.35 baseline, 0.05 uplift) so the two rankings are
correlated but not identical — which is realistic. If `x3` were the only baseline
driver and appeared nowhere in the uplift, response and uplift rankings would be
*orthogonal* and the demonstration would be a strawman. The `x0` overlap is what
makes the response model partially informative, which is the honest picture.

The intercept `−2.0` puts the mean baseline conversion around 17%. That's much
higher than Hillstrom's 0.9%, deliberately: the synthetic data is for validating
*mechanics*, and at 0.9% every estimator is so noise-dominated that you can't
tell a correct implementation from a broken one. Validate at a rate where signal
exists, then go find out what happens at 0.9% on real data.

### The four lines in `simulate()` that carry real judgement

```python
# src/simulate.py:205-209
# Clip the treated probability into a valid range, then recompute the uplift
# from the clipped value so that `true_uplift` is exactly the effect realised
# by the sampler rather than an idealised version of it.
treated_prob = np.clip(baseline + uplift, 1e-4, 1.0 - 1e-4)
realised_uplift = treated_prob - baseline
```

**The trap.** The natural thing is to clip the probability for sampling and then
store `uplift` as the truth:

```python
treated_prob = np.clip(baseline + uplift, 1e-4, 1 - 1e-4)
# ... later ...
true_uplift = (uplift,)  # <-- WRONG
```

Consider a customer with `baseline = 0.02` in the sleeping-dog group, so
`uplift = −0.06`. Then `baseline + uplift = −0.04`, clipped to `0.0001`. The
sampler will realise an effect of `0.0001 − 0.02 = −0.0199`. But you stored
`−0.06` as the truth. You've just told every estimator it should have predicted
three times the effect that actually exists in the data.

The consequence is nastily one-directional: **PEHE goes up, so every estimator
looks worse than it is**, and the error is concentrated in exactly the subgroup
you care most about. You'd conclude your estimators can't find sleeping dogs when
in fact your yardstick was lying. Recomputing from the clipped value costs one
line and makes `true_uplift` mean "the effect the sampler actually applied."

```python
# src/simulate.py:211-214
# Randomisation: a coin flip that never looks at `features`. This is what makes
# the naive difference in means an unbiased estimate of the average effect.
propensity = np.full(n, treatment_share, dtype=np.float64)
treatment = rng.binomial(1, propensity).astype(np.int64)
```

`np.full` — a constant vector, not a function of `features`. That's the definition
of a randomised trial and it's what `simulate_confounded` will break on purpose.
Storing it as a full vector rather than a scalar means both functions return the
same shape and downstream code doesn't branch.

Now the one that took me longest to get right:

```python
# src/simulate.py:216-222
# Draw both potential outcomes with a shared uniform so that the counterfactual
# pair is coupled: the same customer does not flip from converter to
# non-converter purely because of independent sampling noise.
draw = rng.uniform(size=n)
outcome_if_control = (draw < baseline).astype(np.int64)
outcome_if_treated = (draw < treated_prob).astype(np.int64)
outcome = np.where(treatment == 1, outcome_if_treated, outcome_if_control)
```

**The trap.** The obvious version draws twice:

```python
outcome_if_control = rng.binomial(1, baseline)  # tempting
outcome_if_treated = rng.binomial(1, treated_prob)  # also tempting
```

Both are marginally correct — the right proportion of converters in each
potential-outcome world. The `outcome` column would be statistically identical.
Every ATE estimate would be identical. You could run the entire study on it and
get the same headline numbers.

What breaks is *monotonicity*. With independent draws, a customer with
`baseline = 0.20`, `treated_prob = 0.30` can come out `outcome_if_control = 1,
outcome_if_treated = 0` — treatment turned a buyer into a non-buyer, in a
customer whose true effect is *positive*. Roughly 14% of that customer's draws do
this. The shared-uniform version makes it impossible: if `draw < baseline` and
`baseline < treated_prob`, then `draw < treated_prob` necessarily.

Why it matters: this test would silently fail without it —

```python
# tests/test_simulate.py:87-91
def test_counterfactual_outcomes_are_coupled(medium: SyntheticData) -> None:
    """A customer helped by treatment can never convert only in the control world."""
    helped = medium.true_uplift > 0
    impossible = helped & (medium.outcome_if_control == 1) & (medium.outcome_if_treated == 0)
    assert not impossible.any()
```

— and more practically, `outcome_if_control` is used as a *genuinely null
outcome* in the robustness tests (`tests/test_robustness.py:83`). If the
counterfactuals weren't coupled, that null wouldn't be clean, and the test that
checks the placebo test can fail would itself be unreliable. A broken check on a
check. Hard to find.

### The CUPED inputs, added later

```python
# src/simulate.py:226-239
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
```

These three lines were **not in the first version**. The original `pre_period` was
just `baseline * 10 + noise`, and when I ran the CUPED demo it produced a 13%
variance reduction — technically correct, useless as a demonstration, and I
couldn't tell whether the implementation was right or the covariate was weak.

The fix was to introduce `loyalty`: a persistent customer trait that shows up in
both the pre-period *and* the in-window metric. That's what makes a pre-period
covariate informative at all in reality. Without something stable about the
customer, last month tells you nothing about this month.

The critical line is `pre_period`. Look at what's **not** in it: `treatment`. The
covariate is a function of `loyalty` and `baseline` and noise, all of which exist
before assignment. That's the one CUPED assumption that, if violated, doesn't
degrade the method — it *biases* it. And `engagement` deliberately does contain
`treatment`, so we have one clean covariate and one contaminated outcome to
contrast.

The `20.0 *` multiplier on `realised_uplift` in `engagement` is arbitrary — it
just puts the treatment effect on a scale comparable to the noise so the effect is
detectable. It shows up again in `experiment.py:333` where the true effect for
the engagement metric is computed as `20.0 * data.true_uplift`. **That's a magic
number duplicated across two files**, which is exactly the kind of thing that
drifts. It should be a named constant in `simulate.py`. Flagging it now rather
than at the end.

### `simulate_confounded`

```python
# src/simulate.py:285-290
randomised = simulate(n=n, n_features=n_features, seed=seed)

rng = np.random.default_rng(seed + 9_999)
logit = confounding_strength * randomised.features[:, 3]
propensity = 1.0 / (1.0 + np.exp(-logit))
treatment = rng.binomial(1, propensity).astype(np.int64)
```

The design decision worth defending: this function **calls** `simulate` and reuses
its features, its potential outcomes and its true effects. Only the assignment
mechanism changes.

That's what makes the comparison in `naive.py` an actual controlled comparison.
If I'd written a separate generator with its own random draws, the difference
between the randomised and confounded gaps would blend "effect of confounding"
with "effect of different random data," and the 7.5× overstatement figure in the
memo would be uninterpretable. Same customers, same effects, one thing changed.

Note the deliberate choice of `x3` in the assignment logit — the column that
drives the *baseline* and not the uplift. That's what makes it a confounder: it
causes both treatment assignment and the outcome. Using `x0` would confound too
but less cleanly, since `x0` also affects the effect itself.

The engagement swap:

```python
# src/simulate.py:295-300
# Swap the randomised treatment's contribution to engagement for the
# confounded one, holding the customer's own noise draw fixed.
effect_size = 20.0 * randomised.true_uplift
engagement = (
    randomised.engagement - effect_size * randomised.treatment + effect_size * treatment
)
```

This exists because `engagement` depends on `treatment`, and the treatment just
changed. I can't regenerate it from scratch — I don't have `loyalty` any more,
it's a local in `simulate()`. So I algebraically remove the old treatment's
contribution and add the new one. It's exact, and it holds the customer's noise
draw fixed, which is what we want.

**Honestly: this is a smell.** The clean fix is to store `loyalty` on
`SyntheticData` and recompute properly. What's here works and is exact, but it's
the kind of code where a future edit to the `engagement` formula in `simulate()`
silently breaks the confounded variant, because the `20.0 *` reconstruction is a
copy of a formula that lives 60 lines away. Second instance of that magic number.

---

## 4. `src/estimators.py` — the interface that constrains everything downstream

**Why this file exists and why it's separate:** every downstream module —
evaluation, policy, robustness, the CLI — takes "an estimator" as an argument and
refits it. The moment that contract is defined, it's frozen. Everything else is
written against it.

### The signature is the important decision, not the algorithms

```python
# src/estimators.py:197-219
@abstractmethod
def fit(self, features: Array, treatment: Array, outcome: Array) -> UpliftEstimator: ...


@abstractmethod
def predict_uplift(self, features: Array) -> Array: ...
```

Three things are being decided here and each one propagates everywhere:

**One.** `fit` takes `(features, treatment, outcome)` as three separate arrays,
not a DataFrame and not a bundled dataclass. This is why `cross_val_uplift` can
index them independently with fold indices, and why `placebo_test` can pass a
*permuted* treatment while keeping the real outcome. If the signature were
`fit(dataset)`, the placebo test would need a way to construct a modified
dataset, and that constructor would become a place where bugs hide.

**Two.** `predict_uplift` takes only `features` — no treatment. That's what forces
the estimators to be genuine counterfactual predictors rather than things that
peek at the assignment. It also means the same fitted object can score customers
who haven't been assigned to anything, which is what a deployed model does.

**Three.** The return is on the outcome's own scale — absolute probability
difference for a binary outcome. A prediction of `0.03` means three more
conversions per hundred. Not a log-odds ratio, not a percentile, not a score. This
is what lets `policy.py` multiply predictions by counts and get conversions, and
it's what makes `sleeping_dog_report`'s `score < 0` threshold meaningful. If the
scale were a rank or a z-score, `< 0` would be arbitrary.

### `LearnerFactory`

```python
# src/estimators.py:102-107
def __init__(self, kind: str = "lightgbm", random_state: int = 0, **params: Any) -> None:
    if kind not in self.KINDS:
        raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
    self.kind = kind
    self.random_state = random_state
    self.params = params
```

**Why a factory rather than passing an estimator instance?** Because the
meta-learners need *fresh, unfitted* models — T-learner needs two, X-learner needs
four plus maybe a propensity model. If you pass an instance, you have to
`clone()` it everywhere and remember to, and forgetting once means two arms share
a fitted model and the uplift comes out as exactly zero. A factory makes "fresh
model" the only thing you can get.

It also needs to produce both classifiers *and* regressors, because X-learner's
second stage regresses on continuous pseudo-effects even when the outcome is
binary. A single "estimator" parameter couldn't express that.

The LightGBM defaults:

```python
# src/estimators.py:111-122
return {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    ...
}
```

`min_child_samples: 100` is the one doing real work. At a 0.9% conversion rate, a
leaf of 20 rows (the LightGBM default) contains 0.18 expected converters. Leaves
like that are pure noise, and the model will happily grow thousands of them and
report enormous heterogeneity. 100 gives ~0.9 expected converters per leaf, which
is still thin but at least the splits are being made on something. `num_leaves:
15` against LightGBM's default of 31 is the same instinct.

**What changes if you move them:** loosening `min_child_samples` inflates
`predicted_uplift_sd` and makes the Qini coefficient noisier in both directions.
It will occasionally produce a *better-looking* result on any single seed, which
is precisely why the seed-stability check exists.

### `_positive_probability` — a two-line guard against a real crash

```python
# src/estimators.py:180-184
proba = np.asarray(model.predict_proba(features), dtype=np.float64)
if proba.shape[1] == 1:
    only_class = int(np.asarray(model.classes_)[0])
    return np.full(features.shape[0], float(only_class))
return proba[:, 1]
```

The natural version is just `return model.predict_proba(features)[:, 1]`, and it
works right up until it doesn't. Inside `bootstrap_qini`, we resample rows with
replacement. At a 0.9% conversion rate, a bootstrap resample of a *subgroup* can
easily contain zero converters. sklearn then fits a single-class model and
`predict_proba` returns a `(n, 1)` array. `[:, 1]` → `IndexError`, 300 bootstrap
replicates in, with no useful traceback about why.

The guard returns the constant class, which is the right answer: if every training
row was class 0, the predicted probability of class 1 is 0.

**This is a "swallow" decision and it's worth being explicit about which side of
the line it sits on.** We're silently handling a degenerate case rather than
crashing. That's defensible here because the degenerate case is *expected* under
bootstrap resampling and the correct behaviour is well-defined. Compare it with
`qini_curve`'s handling of an empty arm, which raises — because there, a missing
arm means the caller passed something structurally wrong, and continuing would
produce a number.

### S-learner, and why the shrinkage is real

```python
# src/estimators.py:250-256
def fit(self, features: Array, treatment: Array, outcome: Array) -> SLearner:
    """Fit the pooled model. See :meth:`UpliftEstimator.fit`."""
    design = np.column_stack([features, treatment.astype(np.float64)])
    model = self.factory.classifier() if self.classifier else self.factory.regressor()
    model.fit(design, outcome)
    self.model_ = model
    return self
```

Treatment appended as the last column. Then:

```python
# src/estimators.py:262-263
ones = np.column_stack([features, np.ones(features.shape[0])])
zeros = np.column_stack([features, np.zeros(features.shape[0])])
```

Same column position, forced to 1 and 0. Straightforward — but note that the
column *order must match* between fit and predict, and nothing enforces it. If
someone later prepends the treatment column in `fit` and forgets to change
`predict_uplift`, the model would be differencing on `x0` instead of treatment and
returning garbage that still has the right shape and plausible magnitude. A named
`_design_matrix` helper used by both would remove the hazard. It's two functions
apart; I'd fix it.

The documented failure mode is asserted, not asserted-at:

```python
# tests/test_estimators.py:91-96
s_ratio = cross_val_uplift(SLearner(), *args, n_folds=3).std() / medium.true_uplift.std()
t_ratio = cross_val_uplift(TLearner(), *args, n_folds=3).std() / medium.true_uplift.std()

assert s_ratio < 1.0, "S-learner should shrink the spread of effects toward zero"
assert t_ratio > 1.0, "T-learner should inflate the spread of effects"
assert s_ratio < t_ratio
```

This test failed the first time I wrote it, using `fit_predict` instead of
`cross_val_uplift` — in-sample, T-learner's ratio came out at 0.41, not above 1.
That's not a bug in T-learner; it's that in-sample both models reproduce their
training arms closely enough to hide the effect. The claim is about
*out-of-sample* behaviour and the test has to cross-validate to see it. Worth
knowing if you rebuild: if this assertion fails for you, check whether you're
scoring in sample before you go debugging the learner.

### X-learner — the part everyone gets backwards

Stage one is T-learner. Stage two is the interesting bit:

```python
# src/estimators.py:366-370
# Stage 2: impute the missing potential outcome from the *other* arm's model.
# A treated customer's effect is what they did minus what the control model
# says they would have done; a control customer's is the mirror image.
imputed_treated = y_treated.astype(np.float64) - self._score(self.model_control_, x_treated)
imputed_control = self._score(self.model_treated_, x_control) - y_control.astype(np.float64)
```

Read those two lines carefully, because they are not symmetric and the asymmetry
is the whole method.

For a **treated** customer we have their *observed* outcome (`y_treated`) and we
need their counterfactual — what they'd have done untreated. We get that from the
**control** model. Observed minus predicted-counterfactual = effect.

For a **control** customer we have their observed outcome and we need what they'd
have done *treated*. That comes from the **treated** model. Predicted minus
observed — note the order flips, because the treated-world quantity is the one
being predicted and it goes first in `Y(1) − Y(0)`.

**The trap:** writing `imputed_control = y_control - self._score(self.model_treated_, x_control)`.
Sign-flipped. And here's why it's invisible: the *magnitudes* are all correct, the
array shapes are correct, the second-stage regression fits fine, and the final
blended prediction is a smooth function of the covariates that looks entirely
plausible. What you get is an uplift model whose predictions are anti-correlated
with the truth in half the population. On synthetic data `recovery_error`'s
`kendall_tau` would catch it. On real data — where you have no truth — you'd get a
Qini coefficient near zero and conclude "uplift modelling doesn't work on this
dataset," which is a conclusion that sounds sophisticated and humble and is
entirely an artefact of a minus sign.

The other trap: using the **same** arm's model. `y_treated − model_treated(x_treated)`
is a residual, not an effect. It'd be centred near zero with no signal, and it
would look like "no heterogeneity found."

Then the blend:

```python
# src/estimators.py:396-399
weight = self._propensity_scores(features)
tau_treated = np.asarray(self.effect_treated_.predict(features), dtype=np.float64)
tau_control = np.asarray(self.effect_control_.predict(features), dtype=np.float64)
return weight * tau_control + (1.0 - weight) * tau_treated
```

**This looks backwards and it isn't.** `weight` is `g(x) = P(treated)`. When
treatment is rare — say 15% — `weight` is small, so most of the mass goes on
`tau_treated`, the effect model fitted on the *scarce* treated arm. Surely you'd
want to lean on the arm with more data?

No, and here's the reasoning. `tau_treated` was fitted on
`y_treated − model_control(x_treated)`: an *observed* outcome minus a prediction
from the model fitted on the plentiful control arm. Its nuisance component is
well-estimated. `tau_control` was fitted on `model_treated(x_control) − y_control`:
it leans on the model fitted on the scarce arm, extrapolated onto the plentiful
one. Its nuisance component is the shaky one.

So the weighting isn't "trust the bigger arm," it's "trust the pseudo-effect
whose *imputed* half came from the better-estimated model." Get the weights
backwards and X-learner becomes strictly worse than T-learner under imbalance —
which is the exact opposite of the property that justifies using it, and the
justification you'd have written in your docstring.

This is the claim the test defends:

```python
# tests/test_estimators.py:108-115
args = (imbalanced.features, imbalanced.treatment, imbalanced.outcome)
t_pehe = recovery_error(TLearner(fast_factory).fit_predict(*args), imbalanced.true_uplift)[
    "pehe"
]
x_pehe = recovery_error(
    XLearner(fast_factory, propensity=0.15).fit_predict(*args), imbalanced.true_uplift
)["pehe"]
assert x_pehe < t_pehe
```

Flip the weights in `predict_uplift` and this test goes red. It's the only thing
standing between you and a plausible-looking sign error. (It uses `fit_predict`,
i.e. in-sample — weaker than it should be. Noted in the list at the end.)

One more:

```python
# src/estimators.py:379-384
def _propensity_scores(self, features: Array) -> Array:
    if self.propensity is not None:
        return np.full(features.shape[0], self.propensity, dtype=np.float64)
    ...
    return np.clip(_positive_probability(self.propensity_model_, features), 0.01, 0.99)
```

Passing a known propensity short-circuits the model. On a randomised trial with a
known 50/50 split, the true propensity is 0.5 for everyone and fitting a model to
estimate it adds variance with no bias reduction — you'd get 0.497 here and 0.503
there, purely from noise, and those wobbles propagate into every prediction. The
`clip(0.01, 0.99)` on the estimated path is standard overlap protection: without
it, a propensity of 0.001 makes the blend weight explode.

### The causal forest, and the argument I got wrong first

```python
# src/estimators.py:441-452
model = CausalForestDML(
    model_y=nuisance_classifier if self.discrete_outcome else nuisance_regressor,
    model_t=nuisance_classifier,
    discrete_outcome=self.discrete_outcome,
    discrete_treatment=True,
    ...
)
```

Line 444, `discrete_outcome`, was not in my first version, and the failure was
instructive: EconML raised `AttributeError: Cannot use a classifier as a first
stage model when the target is continuous!` — because without that flag, EconML
assumes a continuous Y and rejects the classifier I'd passed as `model_y`. The
flag tells the DML machinery the outcome is binary so the classifier is the right
nuisance model. Good failure: loud, immediate, unambiguous.

`cv=2` is a compute concession. EconML's default cross-fitting is 5-fold and on
42,613 rows the fit went from tolerable to ~4 minutes. With `cv=2` the whole
Hillstrom stage runs in ten minutes. It costs some efficiency in the nuisance
estimates, and I'd raise it if this were going into production.

---

## 5. `src/evaluation.py` — where a subtly wrong line looks like success

This is the file where I'd concentrate your attention if you only had an hour.
Nearly every mistake available here produces a *better* number.

**Why it's separate from estimators:** because the metrics have to be usable on
scores that didn't come from our estimators at all — the oracle `true_uplift`, a
random vector, a permuted-treatment fit. Tests pass raw numpy arrays constantly.
If evaluation lived on the estimator class you couldn't do that.

### The no-AUC decision

Stated at the top of the file, and it's a design constraint rather than a
preference:

> AUC measures how well a score ranks customers by *whether they responded*. An
> uplift model is not trying to do that. The best possible uplift model would give
> a low score to a customer who was always going to buy, because contacting them
> changes nothing, and a response-AUC would punish it for exactly that.

If you rebuild this and reach for `roc_auc_score` because you want "a number that
goes up when the model is good," you will get a number that goes up when the model
is bad. It's not a weaker metric, it's a metric for a different question.

### `_order_by_score` — four lines, one of which is a genuine landmine

```python
# src/evaluation.py:121-123
rng = np.random.default_rng(seed)
jitter = rng.uniform(size=score.shape[0])
return np.asarray(np.lexsort((jitter, -score)), dtype=np.int64)
```

**`np.lexsort` sorts by the LAST key primarily.** So `(jitter, -score)` means:
primary key `-score` (ascending negative = descending score), tie-break on
`jitter`. That's what we want.

**Trap one:** `np.lexsort((-score, jitter))`. Reads more naturally left-to-right —
"sort by score, then jitter" — and does the exact opposite: primary key is
`jitter`, so you get a **completely random ordering**. Every model scores like
noise. You'd conclude uplift modelling doesn't work. I want to stress how quiet
this is: no error, correct shape, plausible-looking Qini curves hovering around
zero, and a perfectly coherent story about how the signal is just too weak.

**Trap two, subtler and more dangerous:** `np.argsort(-score)`. Correct ordering,
no jitter. Works perfectly for any model with continuous predictions. Now feed it
a model that predicts a near-constant — which is precisely what a badly shrunk
S-learner does, and precisely the case you most need to detect. All ties. `argsort`
is stable, so ties resolve to **original row order**. And in a real customer file,
row order is *not* random: files are exported sorted by signup date, customer ID,
last purchase. If row order correlates with anything that correlates with uplift,
your do-nothing model gets a positive Qini coefficient and you ship it.

The test:

```python
# tests/test_evaluation.py:74-85
def test_constant_scores_do_not_inherit_signal_from_row_order(medium: SyntheticData) -> None:
    """Ties are broken at random, so a flat prediction must score like a coin toss.

    Without random tie-breaking a degenerate model would be ranked by row order,
    and in a sorted file that can carry real signal.
    """
    flat = np.zeros(medium.n)
    coefficients = [
        qini_curve(medium.outcome, medium.treatment, flat, seed=s).qini_coefficient
        for s in range(15)
    ]
    assert abs(np.mean(coefficients)) < 2.0 * np.std(coefficients)
```

Fifteen different tie-break seeds; the mean must sit inside the noise. With
`argsort` this returns the same number fifteen times, `np.std` is 0, and the
assertion fails on `< 0.0`. Good test — it catches the bug by construction rather
than by threshold.

### The Qini curve

```python
# src/evaluation.py:165-174
n_treated = np.cumsum(w)
n_control = np.cumsum(1 - w)
y_treated = np.cumsum(y * w)
y_control = np.cumsum(y * (1 - w))

# Where no control customer has appeared yet the rescaling is undefined; the
# incremental estimate is 0 there by convention.
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(n_control > 0, n_treated / np.maximum(n_control, 1), 0.0)
qini = y_treated - y_control * ratio
```

Line 174 is the definition and the `* ratio` is what makes it a causal quantity.

**The trap:** `qini = y_treated - y_control`. Two cumulative counts, differenced.
Looks like exactly what you want — "extra conversions among the treated." It's
wrong whenever the arms are unbalanced within the selected slice, which is
*always*, because slices are selected by a model score and the treated/control
split within any given slice fluctuates. If the top 10% happens to contain 5,200
treated and 4,900 control, you're comparing 5,200 customers' conversions against
4,900 customers' conversions and calling the difference an effect. The
`n_treated / n_control` rescaling puts the control group on the same footing.

The `np.maximum(n_control, 1)` inside the division prevents a divide-by-zero at
the very top of the list before any control customer has appeared; the
`np.where(n_control > 0, ..., 0.0)` then discards that region entirely. Belt and
braces — you need the `maximum` to stop numpy warning, and the `where` to get the
right value.

The endpoint has a property worth internalising:

```python
# src/evaluation.py:181-182
total_incremental = float(qini[-1])
random_line = fraction * total_incremental
```

At 100% selection, the ranking is irrelevant — you've picked everybody — so
`qini[-1]` is a property of the *trial*, not the model. Every model's curve ends
at the same point. The random line is the straight chord from origin to that
shared endpoint. That's what makes curves comparable across models, and it's what
this test pins down:

```python
# tests/test_evaluation.py:37-45
def test_qini_endpoint_equals_the_overall_incremental_effect(medium: SyntheticData) -> None:
    ...
    expected = medium.outcome[treated].sum() - medium.outcome[control].sum() * (
        treated.sum() / control.sum()
    )
    assert result.total_incremental == pytest.approx(expected, rel=1e-9)
```

`rel=1e-9` — this is an algebraic identity, not a statistical approximation, so it
should hold to floating point. If you slacken this to `rel=0.01` you stop testing
the rescaling.

Then the coefficient:

```python
# src/evaluation.py:184-187
qini_auc = float(np.trapezoid(qini, fraction))
random_auc = float(np.trapezoid(random_line, fraction))
coefficient = qini_auc - random_auc
normalized = coefficient / abs(random_auc) if abs(random_auc) > 1e-12 else float("nan")
```

The coefficient is **area between the curves, in incremental conversions**. Not
normalised by a "perfect model" curve, which is how several published
implementations define it. That normalisation needs the counterfactuals to
construct the perfect ordering — which don't exist outside `simulate.py`. Some
libraries approximate the perfect curve from observed data; that approximation is
optimistic in a way that's hard to characterise, so I skipped it. The cost is that
the absolute number isn't comparable across datasets of different sizes, which is
what `normalized_qini` is for.

`abs(random_auc)` — absolute value, so that when the campaign is net *harmful*
(negative total effect) the normalisation doesn't flip the sign of the
coefficient. A model that correctly ranks people in a harmful campaign still has a
positive Qini coefficient. Without the `abs`, it'd report negative and you'd think
the model was backwards.

The `1e-12` guard returns `nan`, not `0.0`. Deliberate: when the overall treatment
effect is zero the normalised quantity is genuinely undefined, and `0.0` would
read as "the model adds nothing," which is a different and much stronger claim.

### The transformed outcome

```python
# src/evaluation.py:274-279
p = np.asarray(propensity, dtype=np.float64)
if np.any(p <= 0.0) or np.any(p >= 1.0):
    raise ValueError("propensity must lie strictly between 0 and 1")
w = np.asarray(treatment, dtype=np.float64)
y = np.asarray(outcome, dtype=np.float64)
return y * (w - p) / (p * (1.0 - p))
```

`Z = Y(W − p) / (p(1 − p))`. Under randomisation `E[Z|X] = τ(X)`, which is the
identity that turns an unobservable into a regression target. Verified directly:

```python
# tests/test_evaluation.py:106-109
def test_transformed_outcome_is_unbiased_for_the_true_effect(medium: SyntheticData) -> None:
    """E[Z] = ATE. This identity is what makes held-out validation possible."""
    z = transformed_outcome(medium.outcome, medium.treatment, 0.5)
    assert z.mean() == pytest.approx(medium.true_ate, abs=0.02)
```

**The trap** is the simplified form you'll see written down: `Z = Y·W/p − Y·(1−W)/(1−p)`.
Expand it. For a treated customer: `Y/p`. The version here gives
`Y(1−p)/(p(1−p)) = Y/p`. Same. For control: simplified gives `−Y/(1−p)`; here,
`Y(0−p)/(p(1−p)) = −Y/(1−p)`. Also same. They're algebraically identical, so this
one is safe — but it's worth checking rather than assuming, because a
*near*-miss like `Y(W − p)/(p(1 − p))` with `p` swapped for `1 − p` gives you a
biased target that still has the right sign and roughly the right magnitude.

The strict propensity check raises rather than clamps. Right call: a propensity of
exactly 0 or 1 means somebody had no chance of being in one arm, which breaks
overlap and makes the whole estimand undefined. Silently clamping to 0.001 would
produce a number, and the number would be nonsense multiplied by 1000.

### `uplift_by_decile` — the table you show the sceptic

```python
# src/evaluation.py:331-341
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
```

`np.array_split` rather than `np.split` — handles `n` not divisible by `n_bins`
without raising. `enumerate(..., start=1)` so deciles read 1–10 not 0–9, because
this table goes in front of a marketing director.

Nothing is modelled inside a bin. `rate_t − rate_c` is a raw comparison of two
observed rates among customers who were randomly assigned. That's the point: it's
the version of the result that survives someone not trusting your model.

The standard error is the textbook difference-of-independent-proportions:
`sqrt(p₁(1−p₁)/n₁ + p₀(1−p₀)/n₀)`. **Unpooled**, because we're constructing a
confidence interval around an estimate, not testing a null of equality. (Compare
`naive.py`, which uses the pooled version for its p-value — I'll come back to why
they differ.)

This is where you read the real story on Hillstrom, by the way. `results/deciles_mens_conversion_t-learner.csv`
has predicted uplift descending 3.70% → −1.51% across the ten rows, and observed
uplift going 1.36, 0.46, 0.90, 0.46, 0.89, 0.50, 0.64, 0.58, 0.37, **0.63**. The
model's confident bottom decile measured *positive*. No aggregate metric tells you
that as bluntly.

### `cross_val_uplift` — the stratification line

```python
# src/evaluation.py:452-453
strata = treatment.astype(np.int64) * 2 + (np.asarray(outcome) > 0).astype(np.int64)
splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
```

Line 452 builds a four-level stratum: `0 = control/no-convert`, `1 =
control/convert`, `2 = treated/no-convert`, `3 = treated/convert`. Stratifying on
the *cross* guarantees every fold has both arms and some converters from each.

**Trap one:** `KFold`. Plain random splits. At 0.9% conversion with 5 folds, you
will eventually draw a fold whose control arm has zero converters. T-learner's
control model becomes single-class, `predict_proba` returns one column, and
without the `_positive_probability` guard you crash — with it, the fold silently
predicts a constant zero and your out-of-fold scores are quietly garbage for 20%
of rows.

**Trap two, the one that looks fine:** `StratifiedKFold(...).split(features, outcome)`.
Stratify on the outcome only. Every fold gets converters, so nothing crashes.
But the treated/control *balance* within folds now drifts freely, and a fold with
55/45 rather than 50/50 gives you an outcome model fitted on a skewed arm. Never
errors. Just adds variance you can't see and can't attribute.

And one wart, right there:

```python
# src/evaluation.py:455-458
for train_idx, test_idx in splitter.split(features, strata):
    import copy

    fold_model = copy.deepcopy(estimator)
```

`import copy` **inside the loop.** Python caches it so the cost is a dict lookup,
but it's sloppy and it should be at module scope. It's there because I added the
deepcopy as an afterthought when I realised a refit estimator was leaking state
between folds — which is itself the important bit. Without `deepcopy`, fold 2
fits an estimator that already has `model_treated_` set from fold 1. sklearn's
`fit` overwrites, so you'd usually get away with it — but `XLearner` holds five
sub-models and if any fold path skips setting one, you inherit the previous
fold's. That's leakage that improves your out-of-fold scores.

### `bootstrap_qini` — stratified resampling

```python
# src/evaluation.py:399-409
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
```

Resample **within each arm**, holding the arm sizes fixed.

**The trap:** `rng.choice(n, size=n, replace=True)` over all rows. Standard
bootstrap, one line shorter, and it resamples the *experimental design* along with
the data — some replicates come out 51/49, some 49/51. That extra variance is
real variance in a hypothetical "what if we'd run a different trial" sense, but
it's not the question. The trial was fixed at 21,307/21,306 by design. The
uncertainty we want to report is uncertainty in the *outcomes*, conditional on the
design. Resampling the design inflates the CI and makes `beats_random` too
conservative — you'd under-claim, which is the safer direction, but it's still the
wrong number.

`seed=b` inside the loop (line 408) is passed to `qini_curve` as the tie-break
seed. Fixed at 0, every replicate would break ties identically and correlate the
draws.

The verdict line:

```python
# src/evaluation.py:421
"beats_random": bool(finite.size and (1.0 - share_positive) < 0.05),
```

One-sided at 5%. The `finite.size and` guard makes an all-`nan` draw set report
`False` rather than `True` — failing closed, which is the right default for a
"did it pass?" flag.

**But be honest about what this test can and can't see.** It holds the *fitted
model* fixed and resamples only the evaluation rows. It cannot see variance from
model fitting — the risk that a different training sample would have produced a
different ranking entirely. The placebo test can. On Hillstrom the two disagree,
and where they disagree the placebo is right. If you rebuild this and only
implement one of them, implement the placebo.

---

## 6. `src/policy.py` — turning a ranking into a decision

**Why separate from evaluation:** evaluation answers "is this ranking any good?"
Policy answers "given a cost, a margin and a budget, where do I cut it?" Different
inputs, different consumers, and the economics change far more often than the
metrics do.

### The function that keeps the study honest

```python
# src/policy.py:117-128
n_selected = int(selected.sum())
if n_selected == 0:
    return 0.0, 0.0, 0.0

treated = selected & (treatment == 1)
control = selected & (treatment == 0)
if not treated.any() or not control.any():
    return 0.0, 0.0, 0.0

uplift_rate = float(outcome[treated].mean() - outcome[control].mean())
revenue_rate = float(spend[treated].mean() - spend[control].mean())
return uplift_rate * n_selected, revenue_rate * n_selected, uplift_rate
```

Lines 126–128 are the load-bearing lines of the whole project.

**The trap, and it is by far the most common way an uplift analysis lies:**

```python
# THE TEMPTING VERSION
incremental_conversions = score[selected].sum()  # WRONG
```

The model predicted an uplift of 0.02 for each of these 12,784 customers, so the
selection is worth 256 incremental conversions, right?

No. That's the model's *forecast* of its own value. You've selected customers
*because* the model said their uplift was high, and then valued them *using* the
same claim. Any model that predicts confidently — correctly or not — scores well.
It's circular, and the circularity is invisible because the resulting number is
reasonable, the units are right, and it correlates with everything you'd expect.

What lines 126–128 do instead: inside the selection, treated and control customers
are *still randomly assigned*, because the selection rule looked only at
covariates and randomisation is preserved under covariate-based conditioning. So
the difference in observed means within the selection is an unbiased *measurement*
of what contacting that group does. The model is used only to order. It never
gets to value its own work.

That distinction is why the memo can report "$3,066 vs $1,674 for random" as a
measurement rather than a forecast, and it's why the same memo can report that the
model's sleeping-dog group actually *gained* 0.44 points. A prediction-based
valuation could never have surfaced that — it would have dutifully reported the
harm the model predicted.

Now the swallowing. Both early returns hand back `(0.0, 0.0, 0.0)`. The
empty-selection case is fine — zero customers, zero effect. The second one is
murkier: if the selection contains treated customers but no controls, we can't
measure anything, and returning `0.0` records "no incremental effect" for a
situation that actually means "unmeasurable." At the top of the frontier
(`fraction = 0.025`, ~1,065 customers, both arms well represented) this never
fires on Hillstrom. On a small dataset or a very fine `n_points` grid it would,
and the frontier would show a flat zero region that looks like a real economic
finding. Should return `nan` and let the caller decide. Flagged.

### The frontier

```python
# src/policy.py:165-176
order = _order_by_score(np.asarray(score, dtype=np.float64), seed=seed)
rank = np.empty(n, dtype=np.int64)
rank[order] = np.arange(n)

fractions = np.linspace(1.0 / n_points, 1.0, n_points)
rows: list[dict[str, float]] = []
for fraction in fractions:
    cutoff = int(round(fraction * n))
    selected = rank < cutoff
    conversions, revenue, uplift_rate = _incremental_at(selected, outcome, treatment, spend)
    cost = cutoff * cost_per_contact
    profit = revenue * margin_rate - cost
```

Lines 166–167 are the rank-inversion idiom and worth having in your fingers:
`order` maps position → row, `rank` maps row → position. `rank[order] = arange(n)`
inverts the permutation in one line. Then `rank < cutoff` is a boolean mask over
the original row order, which is what every downstream indexing operation needs.

**The trap:** `selected = order[:cutoff]` — an index array, not a mask. It works
for `outcome[selected]`, and then silently breaks the moment you combine it with
another mask: `selected & (treatment == 1)` on an int array is a *bitwise and on
the indices*. No error. Nonsense selection.

`np.linspace(1.0 / n_points, 1.0, n_points)` starts at 1/40, not 0. Deliberate: a
0% row would be all zeros and would just be noise in the `idxmax`. The consequence
— and this is where the bug I'll list at the end comes from — is that **there is
no row at fraction 0**, so "contact nobody" isn't representable on the frontier.

### `choose_policy` and the two constraints

```python
# src/policy.py:266-292
best = frontier.loc[frontier["incremental_profit"].idxmax()]
optimal_fraction = float(best["fraction"])
optimal_profit = float(best["incremental_profit"])

budget_cap_fraction = 1.0 if budget is None else min(1.0, budget / (cost_per_contact * n))
affordable = frontier[frontier["fraction"] <= budget_cap_fraction + 1e-9]
if affordable.empty:
    budget_profit = 0.0
    budget_fraction = 0.0
else:
    budget_row = affordable.loc[affordable["incremental_profit"].idxmax()]
    budget_fraction = float(budget_row["fraction"])
    budget_profit = float(budget_row["incremental_profit"])

recommended_fraction = min(optimal_fraction, budget_fraction)
if recommended_fraction <= 0.0:
    # Contacting nobody earns nothing. This needs saying explicitly because the
    # frontier's smallest row is 1/n_points, not 0 — so the nearest-row lookup
    # below would snap a zero recommendation onto the first row and report its
    # profit, i.e. money earned by a campaign we just decided not to run.
    recommended_profit = 0.0
else:
    # Both `optimal_fraction` and `budget_fraction` are read off the frontier,
    # so their minimum is always exactly a row value and this lookup is a
    # lookup rather than an approximation.
    nearest = int((frontier["fraction"] - recommended_fraction).abs().idxmin())
    recommended_profit = float(frontier.iloc[nearest]["incremental_profit"])
```

Two constraints, and the distinction between `budget_cap_fraction` and
`budget_fraction` is a **later fix**, not the original design. The first version
had one field called `budget_fraction` that meant the affordability cap on entry
and got *reassigned* to the profit-maximising fraction within that cap. Same name,
two meanings, twelve lines apart. Not a bug — the arithmetic was right — but
unreadable, and the CLI printed it under a label that was wrong half the time.
Now `budget_cap_fraction` is "what you can afford" and `budget_fraction` is "the
best thing you can afford," which can be strictly smaller when profit turns down
before the money runs out.

The `+ 1e-9` on line 271 is float-comparison hygiene. `budget / (cost * n)` for a
budget of exactly 30% gives `0.30000000000000004`, and the frontier's
`np.linspace` row is `0.30000000000000004` or `0.2999999999999999` depending on
arithmetic order. Without the epsilon you drop the row you meant to include. The
resulting error is one grid step — small, plausible, and permanent.

**The `if recommended_fraction <= 0.0` guard is a bug fix, and it's worth knowing
what it's fixing.** Before it existed, this was just the `.abs().idxmin()` lookup.
That's correct for every fraction *except* zero — because both `optimal_fraction`
and `budget_fraction` are read off the frontier, so their minimum is always a real
row value. Except when the budget can't afford a single frontier step, in which
case `budget_fraction` is `0.0`, the frontier's smallest row is `1/n_points =
0.025`, and the nearest-row lookup snaps to it:

```
# before the fix
budget=0 -> cap: 0.0  budget_frac: 0.0  recommended_frac: 0.0
            recommended_profit: 233.9   budget_profit: 0.0
```

Contact nobody, earn $234. And the test that should have caught it only checked
the fraction:

```
# before — passes green while the profit field is wrong
def test_a_zero_budget_recommends_contacting_nobody(medium):
    result = choose_policy(..., budget=0.0)
    assert result.recommended_fraction == 0.0
```

That is exactly the shape of bug this whole project is about: an internally
consistent number that nobody asserted on. The test now pins the profit and the
`profit_vs_random` too, and there's a second one for a budget below the frontier's
resolution:

```python
# tests/test_policy.py:143-176
def test_a_zero_budget_recommends_contacting_nobody(medium: SyntheticData) -> None:
    """And the profit of contacting nobody must be zero, not the first frontier row.

    The frontier's smallest row is 1/n_points, so a nearest-row lookup on a zero
    recommendation silently returns the profit of a campaign we decided not to run.
    Asserting only the fraction leaves that wrong number unguarded.
    """
    result = choose_policy(
        "oracle", medium.outcome, medium.treatment, medium.true_uplift, medium.spend, budget=0.0
    )
    assert result.recommended_fraction == 0.0
    assert result.recommended_profit == 0.0
    assert result.profit_vs_random == 0.0


def test_a_budget_too_small_for_one_frontier_step_earns_nothing(
    medium: SyntheticData,
) -> None:
    """A budget below the frontier's resolution is the same decision as no budget."""
    result = choose_policy(
        "oracle",
        medium.outcome,
        medium.treatment,
        medium.true_uplift,
        medium.spend,
        budget=1.0,
        cost_per_contact=0.10,
    )
    assert result.budget_cap_fraction < 1.0 / 40.0
    assert result.recommended_fraction == 0.0
    assert result.recommended_profit == 0.0


def test_sleeping_dogs_are_found_and_measured(medium: SyntheticData) -> None:
```

The general lesson, if you're rebuilding: **whenever a lookup interpolates or
snaps onto a grid, check what happens at the boundaries the grid doesn't contain.**
This frontier deliberately starts at `1/n_points` rather than `0`, and everything
downstream inherited an assumption that the recommendation is always on the grid.

### `sleeping_dog_report` — the two-number discipline

```python
# src/policy.py:399-401
# 1.0 when the confidence interval sits entirely below zero, i.e. the harm
# is measurable rather than merely predicted.
"measured_uplift_is_negative": float((measured + 1.96 * stderr) < 0.0),
```

**The trap:** `float(measured < 0.0)`. The point estimate is negative, so the group
is harmed, so report harm. That's how you write a memo claiming a confirmed
sleeping-dog segment when the interval is `[−0.53, +0.28]` and crosses zero
comfortably. Requiring the *upper* bound to be below zero is what turns "the model
predicts harm" into "the data shows harm," and on Hillstrom the difference between
those two statements is the difference between a suppression list and a mistake.

The whole function is built around keeping two things apart: `n_flagged` and
`mean_predicted_uplift` are properties of the *model*; `measured_uplift` and its
interval are properties of the *world*. On the men's campaign the model flags
18.6% with a predicted mean of −0.87 points and the world says **+0.44 points, CI
[+0.02, +0.87]**. Positive, interval excluding zero. The model is not slightly
uncertain about these customers, it's wrong about them. Only a report that keeps
prediction and measurement in separate fields can say that.

`cost_of_treating_them` is `-profit`, sign-flipped so "positive means excluding
them makes money." Small thing; it stops the memo from having to explain a double
negative.

### `bootstrap_policy` — the fixed fraction

```python
# src/policy.py:451-455
order = _order_by_score(np.asarray(score, dtype=np.float64), seed=seed)
rank = np.empty(n, dtype=np.int64)
rank[order] = np.arange(n)
cutoff = int(round(fraction * n))
selected_mask = rank < cutoff
```

`selected_mask` is computed **once, outside the bootstrap loop**, from the full
data. Each replicate then indexes into it: `chosen = selected_mask[idx]`.

**The trap:** re-running `choose_policy` inside each replicate to find that
replicate's optimal fraction. It's what you'd naturally write — bootstrap the
whole procedure! — and it answers a genuinely different question: *the sampling
distribution of the best achievable profit under hindsight*. That distribution is
narrower than the one you want, because every replicate gets to pick its own
optimum after seeing its own data. You'd report a tight interval around an
optimistic number, and the interval would be tight *because* of the optimism.

What we want is the uncertainty in the number actually being proposed: "we will
mail the top 30%, and here's the spread of what that's worth." Fraction fixed,
data resampled.

---

## 7. `src/data.py` — the real file

Built after the estimators in wall-clock time, but it only depends on `config`,
so you can put it anywhere in the sequence. It comes here because from now on
things need it.

### The download with a fallback

```python
# src/data.py:96-100
destination.parent.mkdir(parents=True, exist_ok=True)
partial = destination.with_suffix(destination.suffix + ".partial")
with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
    partial.write_bytes(response.read())
partial.replace(destination)
```

Write to `.partial`, then `replace`. `Path.replace` is atomic on POSIX, so you
never end up with a half-downloaded `hillstrom.csv` that reads as a valid CSV with
40,000 rows. **The trap** is writing directly to `destination`: interrupt the
download at 60%, and the cache check `if destination.exists()` on the next run
happily returns a truncated file. Pandas parses it. You analyse two thirds of the
data and never find out.

```python
# src/data.py:124-131
errors: list[str] = []
for url in (HILLSTROM_URL, HILLSTROM_MIRROR_URL):
    try:
        _download(url, destination)
    except (urllib.error.URLError, OSError, ValueError) as exc:  # noqa: PERF203
        errors.append(f"{url}: {exc}")
    else:
        return destination
```

Collect errors, try both, and only then raise with everything you learned:

```python
# src/data.py:134-138
raise RuntimeError(
    "Could not download the Hillstrom dataset from either source:\n  "
    f"{joined}\n"
    f"Download it manually and place it at {destination}."
)
```

**Which failures we swallow and which we don't.** Network failures on the primary
URL are swallowed — expected, recoverable, there's a plan B. Failure of *both* is
fatal and raises, with both underlying errors and a manual workaround in the
message. That's the split throughout this codebase: swallow what you have a
correct fallback for, raise for anything where continuing would produce a number.

### Feature encoding, and the column that gets dropped

```python
# src/data.py:166-171
numeric = frame[NUMERIC_FEATURES].astype(np.float64)
dummies = pd.get_dummies(
    frame[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, drop_first=False
).astype(np.float64)
encoded = pd.concat([numeric, dummies], axis=1)
return encoded.to_numpy(dtype=np.float64), list(encoded.columns)
```

`drop_first=False` — keep all three zip codes and all three channels. For a linear
model you'd drop one to avoid perfect collinearity with the intercept. Tree
ensembles have no intercept and don't care, and keeping all levels makes the
feature names readable in the balance table. Since `LearnerFactory` can produce
logistic regression, there's mild collinearity in that path — sklearn's L2
penalty absorbs it, but it's a real if minor inconsistency.

`history_segment` is **not** in `NUMERIC_FEATURES` or `CATEGORICAL_FEATURES`.
It's a coarse binning of `history` (`"2) $100 - $200"`), and `history` is already
present at full resolution. Including both adds a perfectly collinear coarse copy
of a column you already have. Not fatal for trees, just noise in the split search
and three extra rows in the balance table.

### Two arms, not three

```python
# src/data.py:209-213
elif treatment_arm in arms:
    label = arms[treatment_arm]
    frame = raw[raw["segment"].isin([label, CONTROL_LABEL])].copy()
    treated_labels = {label}
    treatment_name = label
```

Filter to *one* e-mail arm plus control, drop the third entirely.

**The trap** is `treatment = (segment != "No E-Mail")` — pool both e-mails. It
doubles your treated sample, which feels like a free win at a 0.9% conversion
rate. But the two creatives have measurably different effects: men's e-mail lifts
conversion by 0.68 points, women's by 0.31. Pooling gives you the average effect
of a *randomly chosen creative*, which is not a campaign anyone would run. You'd
be estimating the effect of a coin flip between two campaigns. `treatment_arm="any"`
supports it for completeness with the caveat in the docstring, but it's not the
default and it's not what the memo uses.

`.copy()` after the boolean filter, then `.reset_index(drop=True)` at line 217.
Without the reset, the frame carries original indices with gaps, and
`frame[outcome].to_numpy()` is fine but any later positional/label indexing
mismatch bites. Cheap insurance.

### `load_criteo` — unfinished, and I'm saying so here

```python
# src/data.py:233-236
def load_criteo(
    data_dir: Path = DATA_DIR,
    sample_rows: int | None = 500_000,
) -> UpliftDataset:
```

This function has **never been executed**. Not once. The S3 host isn't reachable
from the sandbox this was built in, there's no test for it, and no CLI path calls
it. It's written from the documented schema and it is plausible code, but "reads
as if it works" and "works" are different claims and only one of them is
supported. If you rebuild this, either exercise it against the real file or delete
it. Shipping untested code that *looks* like a feature is worse than not having
the feature.

---

## 8. `src/naive.py` — computing the wrong answer properly

**Why it's a separate module rather than a section of the memo:** because the
memo's most quotable numbers — "7.5× overstated," "five times as many sleeping
dogs" — have to be *computed*, reproducibly, from data where the truth is known.
A memo that asserts them is an opinion. A memo backed by `results/naive.json` is a
result.

### The two standard errors, which look inconsistent and aren't

```python
# src/naive.py:114-121
difference = rate_t - rate_c
stderr = float(np.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c))

# Pooled-variance two-proportion z-test, the textbook version.
pooled = float(outcome.mean())
pooled_se = float(np.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c)))
z = difference / pooled_se if pooled_se > 0 else 0.0
p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
```

Two different standard errors, four lines apart, and this is correct rather than
sloppy.

The **unpooled** SE (line 115) is used for the confidence interval. We're
estimating the size of a difference we believe is real, so each arm contributes
its own variance.

The **pooled** SE (line 119) is used for the p-value. A hypothesis test asks "if
the null were true, how surprising is this?" — and under the null, both arms have
the *same* rate, so the best estimate of that shared rate uses all the data. Using
the unpooled SE for the test is a genuinely common error; it doesn't move the
answer much at these sample sizes, but it's the wrong quantity, and at small n
with rare outcomes it's anti-conservative.

If you rebuild this: computing both and labelling them is the honest move. Picking
one and using it for everything is the shortcut that makes a reviewer squint.

### The response model, trained the way real ones are

```python
# src/naive.py:219-225
# Response model: trained on treated customers only, which is exactly how a
# "who responds to our e-mail" model gets built from campaign history.
factory = LearnerFactory(kind="lightgbm", random_state=seed)
response_model = factory.classifier()
treated = treatment == 1
response_model.fit(features[treated], outcome[treated])
response_score = _positive_probability(response_model, features)
```

`features[treated], outcome[treated]` — **fitted on the treated arm only**.

This is the honest reconstruction of a response model, and getting it right
matters for the fairness of the comparison. A marketing team building "who
responds to our e-mail" trains on people who got the e-mail; the control group
isn't in the training data because from their perspective it isn't relevant. So
the model learns `P(convert | contacted, X)`.

**The trap** is fitting on the pooled data. You'd get `P(convert | X)` marginalised
over assignment, which is a slightly different and slightly *better* ranking — it
partially absorbs the treatment effect. Your response model would look better than
the ones real teams build, and the comparison would understate the problem you're
trying to demonstrate. Building a strawman is bad; accidentally building an
*anti*-strawman that weakens your own case is also bad.

Note this whole function scores **in sample** — `TLearner(factory).fit(features, treatment, outcome)`
then predicts on the same rows. For an evaluation that would be disqualifying. Here
it's defensible: both models get the same treatment, and the comparison is against
`data.true_uplift`, which neither model saw. In-sample optimism inflates both
sides roughly equally. Defensible, not ideal — an out-of-fold version would be
strictly better and I'd write it that way now.

The payoff:

```python
# src/naive.py:254-255
"sleeping_dogs_contacted_by_response_model": float(dogs[top_response].sum()),
"sleeping_dogs_contacted_by_uplift_model": float(dogs[top_uplift].sum()),
```

1,345 versus 254. That's the single most persuasive number in the memo and it's
two lines of counting.

---

## 9. `src/experiment.py` — power, MDE and CUPED

**Why separate:** it depends only on `scipy.stats` and the simulator. Nothing else
in the project imports it. It could be its own package, and keeping it isolated
means the power calculator is reusable for any binary-outcome test.

### The sample-size formula

```python
# src/experiment.py:123-128
numerator = (
    z_alpha * np.sqrt((1.0 + 1.0 / k) * p_bar * (1.0 - p_bar))
    + z_power * np.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1) / k)
) ** 2
n_control = int(np.ceil(numerator / delta**2))
n_treated = int(np.ceil(k * n_control))
```

Two variance terms and **they are different on purpose**, for exactly the reason
the two SEs in `naive.py` differ. The `z_alpha` term uses the *pooled* variance
`p_bar(1−p_bar)` — that's the variance under the null, which is what determines
the critical value. The `z_power` term uses the *separate* variances `p0(1−p0)` and
`p1(1−p1)` — the variance under the alternative, which is what determines the
distribution of the test statistic when the effect is real.

**The trap** is using pooled for both. Simpler, symmetric, and it *underestimates
the required sample size* — you'd design an underpowered test and not know it. The
direction of the error is the bad one: your test comes back non-significant and
you conclude the model doesn't work, when actually you didn't run it long enough.

`z_power = norm.ppf(power)`, **not** `norm.ppf(1 − power/2)`. Power is one-sided by
construction — you care about detecting an effect in the direction you expect.
Using the two-sided form here inflates n by roughly 25% at 80% power. Not a
disaster, just wrong, and it's a natural slip when the line above it does use a
two-sided `alpha/2`.

`expected_conversions_per_arm` is returned as a **sanity check on the formula
itself**. The normal approximation degrades when the expected event count is
small. Returning it lets the reader check whether the answer is trustworthy
instead of taking a number from a formula whose assumptions weren't verified.

### The inverse, by bisection

```python
# src/experiment.py:171-179
low, high = 1e-6, min(0.5, 1.0 - baseline_rate - 1e-6)
for _ in range(200):
    mid = 0.5 * (low + high)
    needed = required_sample_size(baseline_rate, absolute_mde=mid, alpha=alpha, power=power)
    if needed.n_per_arm > n_per_arm:
        low = mid  # too small an effect to detect: need a bigger one
    else:
        high = mid
absolute = high
```

**Why bisection rather than algebra?** Because inverting that formula by hand
gives you a second expression that has to stay consistent with the first one
forever. Change the allocation-ratio handling in `required_sample_size` and the
closed-form inverse silently disagrees. Bisection *calls* the function, so they
cannot drift. It costs 200 iterations of a cheap function — microseconds — and
buys a guarantee.

The direction on lines 175–178 is the thing to slow down on. `needed.n_per_arm > n_per_arm`
means "this candidate effect is too small to detect with the sample we have" —
so we need a **bigger** effect, so we raise the lower bound. Flip the branches and
you converge to the smallest effect in the search range, roughly `1e-6`, and
report that you can detect a rounding error. Which, to be fair, is obvious enough
in the output that you'd catch it. The *subtle* version is returning `low` instead
of `high` on line 179: off by one bisection step, always in the optimistic
direction, and utterly invisible.

The test that pins it:

```python
# tests/test_experiment.py:61-66
def test_mde_is_the_inverse_of_the_sample_size_calculation() -> None:
    """The two directions must be consistent, which is why one calls the other."""
    for baseline in (0.005, 0.05, 0.20):
        sizing = required_sample_size(baseline, relative_mde=0.25)
        recovered = minimum_detectable_effect(baseline, sizing.n_per_arm)
        assert recovered["absolute_mde"] == pytest.approx(sizing.absolute_mde, rel=0.02)
```

Round-trip at three baselines including a 0.5% one. `rel=0.02` accommodates the
integer `ceil` on sample size — you can't recover the effect exactly when the
forward direction rounds up.

### CUPED

```python
# src/experiment.py:271-272
theta = float(np.cov(y, x, bias=True)[0, 1] / var_x)
adjusted = y - theta * (x - x.mean())
```

`theta` on **pooled** data, both arms together.

**The trap:** computing theta per arm. It feels more careful — each arm gets its
own regression coefficient! — and it quietly re-introduces the very difference
you're measuring. The treated arm's `Cov(Y, X_pre)` includes the treatment effect;
subtracting a treatment-informed adjustment from the treated arm and a different
one from control biases the estimate. Under randomisation the pooled theta is
consistent and the finite-sample bias is O(1/n).

`bias=True` gives the population covariance (divide by n) matching `x.var()`'s
default (also n). Mixing `bias=False` (n−1) with `.var()` (n) gives you a theta off
by `n/(n−1)`. At n = 40,000 that's 0.0025% and completely invisible. It's the kind
of thing that's correct here and wrong in a unit test with n = 10.

Centring by `x.mean()` on line 272 keeps `adjusted.mean() == y.mean()`, which is
what this asserts:

```python
# tests/test_experiment.py:144-147
def test_cuped_adjusted_values_have_the_right_shape(medium: SyntheticData) -> None:
    adjusted, _ = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert adjusted.shape == (medium.n,)
    assert adjusted.mean() == pytest.approx(medium.engagement.mean(), abs=1e-9)
```

Drop the centring and the adjusted metric shifts by `theta * mean(x)` — the
*difference* between arms is unchanged, so every treatment-effect number stays
right, but the metric's level is now meaningless. Anyone reading "average
engagement: −1.4 sessions" would rightly stop trusting the analysis.

And the sharpest test in the repo:

```python
# tests/test_experiment.py:108-111
def test_cuped_variance_reduction_equals_squared_correlation(medium: SyntheticData) -> None:
    """The identity that proves the implementation, not just its plausibility."""
    _, result = cuped_adjust(medium.engagement, medium.pre_period, medium.treatment)
    assert result.variance_reduction == pytest.approx(result.correlation**2, abs=1e-9)
```

`abs=1e-9`. This isn't a statistical approximation, it's an algebraic identity:
regressing out a covariate removes exactly `ρ²` of the variance. Any error in
theta, in the centring, or in the variance computation breaks it at the ninth
decimal. Measured: 0.5075899852254115 versus 0.5075899852254109. That's a real
proof, not a smoke test, and it's the pattern to reach for whenever your
computation has a closed-form invariant.

The failure mode gets its own test:

```python
# tests/test_experiment.py:157-169
def test_cuped_on_a_post_treatment_covariate_would_bias_the_estimate() -> None:
    """Documents the failure mode rather than only warning about it in prose.

    Using a covariate the treatment influenced pulls the estimate toward zero,
    which is exactly what happens when a 'pre-period' window overlaps the campaign.
    """
    data = simulate(n=20_000, seed=21)
    contaminated = data.pre_period + 5.0 * data.treatment
    _, honest = cuped_adjust(data.engagement, data.pre_period, data.treatment)
    _, biased = cuped_adjust(data.engagement, contaminated, data.treatment)
    assert abs(biased.ate_adjusted - data.true_ate * 20.0) > abs(
        honest.ate_adjusted - data.true_ate * 20.0
    )
```

This encodes the number-one way CUPED goes wrong in practice — a "pre-period"
window that overlaps the campaign — as an executable assertion rather than a
warning nobody reads.

---

## 10. `src/robustness.py` — the checks that try to break it

**Why last among the analysis modules:** it imports from estimators, evaluation
*and* policy. It's the integration layer of the analysis.

### The placebo test — the most valuable 30 lines here

```python
# src/robustness.py:109-115
placebo_scores = np.empty(n_replicates, dtype=np.float64)
for r in range(n_replicates):
    fake = rng.permutation(treatment)
    oof = cross_val_uplift(
        copy.deepcopy(estimator), features, fake, outcome, n_folds=n_folds, seed=seed + r
    )
    placebo_scores[r] = qini_curve(outcome, fake, oof).qini_coefficient
```

Permute **treatment**. Keep **outcome** exactly as it is.

**The trap:** `fake_outcome = rng.permutation(outcome)`. Also destroys the
treatment–outcome relationship, so it superficially does the same job — but it
also destroys the relationship between *covariates* and outcome. The base learners
now have nothing at all to fit, so they predict constants, and your placebo
distribution is much tighter than it should be. The real result then clears it
easily, the check passes, and you've built a test that can't fail.

Permuting treatment keeps `X → Y` intact and destroys only `W → Y`. The model can
still fit the outcome beautifully; it just has no effect to find. That's the right
null, and it's why the placebo distribution on `mens_visit` has an SD of **25.0** —
that's the magnitude of Qini coefficient this pipeline manufactures from pure
noise, and it's the bar the real result has to clear.

Note that `fake` is passed to *both* `cross_val_uplift` and `qini_curve` (line
115). It has to be. Fit on the fake assignment, evaluate against the fake
assignment. Evaluating fake-fitted scores against the real treatment would be a
mismatch — and would produce a systematically negative placebo distribution that
makes the real result look wonderful.

The verdict:

```python
# src/robustness.py:130-132
# Two conditions, both required: the null must sit at zero (otherwise the
# metric itself is biased) and the real result must stand outside it.
passes=bool(abs(mean) < 2.0 * sd and exceed < 0.10),
```

Both conditions, and the first is the one people skip. If the placebo mean isn't
near zero, your *metric* has a bias — it systematically rewards or punishes
something unrelated to the effect — and no comparison against it means anything.
Only checking `exceed < 0.10` would let a metric with a large positive bias pass
whenever the real result happened to be even larger.

`exceed < 0.10` rather than `< 0.05`: with 10 replicates the empirical p-value can
only take values 0.0, 0.1, 0.2, … A threshold of 0.05 collapses to "zero
replicates may exceed," which is a much harsher test than intended. 0.10 means "at
most one of ten," which is what the resolution actually supports. If you raise
`n_replicates` you should tighten this in step — it's a coupled pair and there's
nothing in the code enforcing that. Worth a comment it doesn't have.

### The `real_qini` subtlety

```python
# src/robustness.py:104-107
real_oof = cross_val_uplift(
    copy.deepcopy(estimator), features, treatment, outcome, n_folds=n_folds, seed=seed
)
real_qini = qini_curve(outcome, treatment, real_oof).qini_coefficient
```

The reference is **refit inside this function** with the same `n_folds` as the
placebos, rather than taking the headline score from the main run. That's
deliberate: the main run uses 5 folds, the placebos use 3 (for speed), and
comparing a 5-fold real result against a 3-fold null is comparing two different
estimators. More training data per fold means better predictions means a higher
Qini, so the mismatch would flatter the real result. Refitting under the matched
protocol costs one extra fit and removes the confound.

This was **also a later fix**. The first version didn't take a `seed` at all and
defaulted to 0, while the main run used `RANDOM_SEED = 20080320`, so the placebo's
"real" figure was from a different fold split than the leaderboard's. The numbers
disagreed in the output and it looked like a bug in the Qini function. The
`PlaceboResult.n_folds` field exists so the output is self-documenting about which
protocol produced the comparison.

### Covariate balance

```python
# src/robustness.py:166-169
a, b = features[treated, j], features[control, j]
mean_a, mean_b = float(a.mean()), float(b.mean())
pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
smd = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0
```

Standardised mean difference: gap in means over pooled SD, so it's comparable
across variables on different scales. The p-value alongside it is nearly useless
at 40,000 rows — an SMD of 0.02 will be "significant" — which is why
`balance_summary` keys `passes` off the SMD:

```python
# src/robustness.py:204
"passes": float(worst < 0.10),
```

0.10 is the conventional threshold from the matching literature. The check exists
mostly as a smoke alarm on our own data handling — Hillstrom *was* randomised, so
a failure here indicts our filtering, not the trial. The test proves the check can
actually fail:

```python
# tests/test_robustness.py:34-43
def test_balance_fails_when_assignment_depends_on_covariates(
    confounded: SyntheticData,
) -> None:
    """The check must actually catch a broken design, not just bless a good one."""
    balance = covariate_balance(confounded.features, confounded.treatment, confounded.feature_names)
    summary = balance_summary(balance)
    assert summary["max_abs_smd"] > 0.25
    assert summary["passes"] == 0.0
    # And it must finger the right column: x3 is what drives assignment.
    assert balance.iloc[0]["feature"] == "x3"
```

That last assertion is the good one. It's not enough for the check to say "something's
wrong" — a sorted-by-severity table has to put the *actual* confounder on top,
because that's what someone debugging a real imbalance would act on.

Line 170 unpacks `t_stat, p_value` and never uses `t_stat`. Should be `_`. Ruff
doesn't flag unused tuple-unpack targets by default. Trivial, but it's the sort of
thing a reviewer notices and wonders what else got left behind.

### Seed stability, and the metric that actually matters

```python
# src/robustness.py:348
selections.append(set(_order_by_score(oof, seed=0)[:k].tolist()))
```

`seed=0` — **the tie-break seed is fixed** while the CV seed varies across the
loop.

That's the whole design of the check. We want to measure how much the *selected
customer list* changes when the model is refit on different fold splits. If the
tie-break seed varied too, we'd be measuring model instability plus tie-breaking
noise, and for a near-constant predictor the second term would dominate and swamp
the first. Fixing it isolates the variable we care about.

```python
# src/robustness.py:350-352
overlaps = [
    len(a & b) / len(a | b) for i, a in enumerate(selections) for b in selections[i + 1 :]
]
```

Jaccard over all unordered pairs. `selections[i + 1:]` gives each pair once —
`selections[i:]` would include each set with itself and inject a spurious 1.0 per
seed, pulling the mean up. With 5 seeds that's 5 spurious perfect overlaps among
15, dragging a true 0.44 up to about 0.58. Wrong in the reassuring direction.

Why report overlap at all when you already have the Qini spread? Because they
disagree, and the disagreement is the finding. On Hillstrom the Qini coefficient
is reasonably stable across seeds while the selected lists share only **44%** of
their members. Aggregate performance is reproducible; the specific list is not.
Operationally that's the difference between "this model works" and "we can mail
the list it produced."

### The aggregate verdict

```python
# src/robustness.py:435-440
"overall_pass": bool(
    placebo.passes
    and balance_stats["passes"] > 0.5
    and stability["sign_consistent"] > 0.5
    and bootstrap["beats_random"]
),
```

AND, not OR, not a weighted score. One failure fails the whole thing.

`> 0.5` on floats that only ever hold 0.0 or 1.0 is ugly — the dicts are typed
`dict[str, float]` so booleans got stored as floats, and now every read needs a
threshold comparison. It works, but `dict[str, float | bool]` or a small dataclass
would be honest. Small design debt from choosing a uniform dict type too early.

---

## 11. `src/cli.py` — the thing that runs it

**Why a separate module rather than `if __name__ == "__main__"` blocks in each
file:** because the stages share state — `stage_experiment` takes
`stage_hillstrom`'s output so it can size the validation test using effects we
actually measured rather than numbers we made up. And because ordering is
semantic: validation *must* run before Hillstrom for the study's logic to hold,
and a single entry point makes that ordering explicit rather than a convention
someone can violate.

### `_jsonable`

```python
# src/cli.py:73-87
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
```

**Order matters here and it isn't arbitrary.** `dict` before everything so nested
payloads recurse. `np.generic` (line 81) catches numpy scalars — `np.float64` is
*not* a Python `float` and `json.dumps` chokes on it with an unhelpful
`TypeError: Object of type float64 is not JSON serializable`, which is a
genuinely annoying thing to debug at the end of a ten-minute run.

`is_dataclass(value) and not isinstance(value, type)` — without the second clause,
a dataclass *class* (not instance) passes `is_dataclass` and `asdict` raises. Only
matters if a class object ends up in a payload, which it shouldn't, but the check
is one clause.

Line 85: non-finite floats become `null`. JSON has no NaN or Infinity. Python's
`json.dumps` emits bare `NaN` by default, which is invalid JSON that many parsers
reject. Mapping to `null` is lossy — you can't tell NaN from Infinity from a
genuine null in `results/hillstrom.json` — but it's valid, and the alternative is
files that only Python can read.

### The budget, defined circularly

```python
# src/cli.py:318
budget = budget_fraction * dataset.n * COST_PER_CONTACT
```

Worth flagging as a smell rather than a bug: the budget is *defined* as "enough to
contact 30% of the file," then handed to `choose_policy`, which divides by
`cost_per_contact * n` to recover... 30%. It round-trips.

That's fine for a study where the point is to show what a binding constraint does,
and `choose_policy` correctly takes a real dollar figure so it works when someone
hands you one. But if you read this line expecting a marketing budget, you'll be
confused about where the number came from. In a real engagement this line is
`budget = 15000.0` and the constant disappears.

### Picking the best model

```python
# src/cli.py:311-315
leaderboard = pd.DataFrame(rows).sort_values("qini_coefficient", ascending=False)
leaderboard.to_csv(results_dir / f"models_{label}.csv", index=False)

best_model = str(leaderboard.iloc[0]["model"])
best_score = scores[best_model]
```

Selecting the best model **on the same out-of-fold predictions used to evaluate
it**. That's a selection effect: with four models, the max of four noisy estimates
is biased upward, and the reported Qini for the winner is optimistic by an amount
nobody computes.

I'm not going to pretend this is fine. The clean version is nested
cross-validation — select on an inner loop, report on an outer one. It roughly
quadruples the runtime and I didn't do it.

What partly rescues the study is that the *conclusion* doesn't lean on the
leaderboard number. The placebo test refits the selected model from scratch, and
on the men's arm it fails, which is the memo's headline. The selection bias makes
the leaderboard figure optimistic; the placebo test doesn't care. If the memo's
claim were "the Qini coefficient is 10.05," this would be a serious problem. The
claim is "even the winner can't clear its own noise floor," which survives.

Also, line 314 is the only place `best_model` is decided, and `_run_one_dataset`
then uses it for the policy, the sleeping-dog report *and* the robustness checks.
Change the selection criterion here and four downstream sections change with it.
Nothing marks that.

### One inconsistency worth noting

```python
# src/cli.py:482
nearest = frontier.iloc[(frontier["fraction"] - fraction).abs().idxmin()]
```

Compare `src/policy.py:281`, which does the same lookup with an explicit
`int(...)` cast. It's here without one. Both work — `idxmin()` returns an integer
label on a default RangeIndex — but mypy only accepted the CLI version because
`frontier` arrives as `Any` out of a dict, so the type checker never looked. Two
spellings of the same idiom, one of them type-checked and one not.

---

## 12. The tests — what each is actually defending

Everything runs on synthetic data. No network, no dataset, deterministic.

```python
# tests/conftest.py:17-26
@pytest.fixture(scope="session")
def small() -> SyntheticData:
    """A small randomised sample, sized so the whole suite stays quick."""
    return simulate(n=4_000, seed=11)


@pytest.fixture(scope="session")
def medium() -> SyntheticData:
    """A larger randomised sample for assertions that need estimation precision."""
    return simulate(n=20_000, seed=12)
```

`scope="session"` — generated once for the whole run. Safe only because
`SyntheticData` is `frozen=True` and nothing mutates the arrays. If a test ever
did `data.outcome[0] = 1`, every subsequent test would silently see it. The frozen
dataclass protects attribute rebinding but *not* the array contents — numpy arrays
inside a frozen dataclass are still mutable. Worth knowing.

Different seeds per fixture (11, 12, 13) so a coincidence in one sample can't
prop up assertions in another.

**Why no test uses Hillstrom.** Two reasons. CI would need a 4 MB download on
every run, adding a network dependency to a test suite. And more importantly, a
test asserting `qini_coefficient > 5` on Hillstrom isn't testing that the code is
*correct* — it's testing that the 2008 data hasn't changed. It would pass with a
sign error in the Qini function if that error happened to produce a number above
5.

The tests that earn their place, grouped by the bug they catch:

**Metric integrity.** `test_random_scores_give_a_qini_coefficient_near_zero`
(test_evaluation.py:48) — noise must not look like signal. Fails if you get the
rescaling wrong.
`test_reversing_the_score_reverses_the_sign` (68) — catches sign errors and
absolute-value mistakes.
`test_constant_scores_do_not_inherit_signal_from_row_order` (74) — the `argsort`
trap, specifically.
`test_beats_random_test_does_not_fire_for_noise` (153) — the most important one in
the file. A useless model must be *reported* as useless.

**Estimator behaviour.** `test_s_learner_shrinks_and_t_learner_spreads`
(test_estimators.py:76) — turns two docstring claims into assertions.
`test_x_learner_beats_t_learner_under_imbalance` (99) — the only defence against a
flipped blend weight.
`test_sleeping_dogs_are_ranked_below_everyone_else` (66) — the core capability, for
all three meta-learners.
`test_cross_val_predictions_are_worse_than_in_sample` (test_evaluation.py:173) —
this one catches *leakage*. If out-of-fold PEHE ever comes out better than
in-sample, your fold split is leaking and this is how you find out.

**Economics.** `test_profit_is_revenue_times_margin_less_cost`
(test_policy.py:47) — arithmetic identity, `assert_allclose`.
`test_targeting_the_whole_file_equals_random_targeting_the_whole_file` (87) —
elegant: at 100% depth the ranking is irrelevant, so two independently-coded paths
must agree to `rel=1e-9`. That's a cross-check between `profit_frontier` and
`random_targeting_profit`.
`test_bootstrap_policy_matches_the_frontier_at_the_same_depth` (188) — same idea,
two paths, one answer.
`test_sleeping_dog_report_keys_match_across_branches` (166) — the empty branch and
the populated branch must return identical key sets. Boring, and it caught a real
bug: the empty branch was originally missing `mean_predicted_uplift` and
`incremental_conversions_if_treated`, so any consumer touching those keys would
`KeyError` only when a model happened to flag nobody.

**Checks that can fail.** `test_placebo_fails_when_the_outcome_carries_no_treatment_effect`
(test_robustness.py:70) is the cleverest test here. It feeds `outcome_if_control`
as the outcome — the control-world potential outcome for *everyone*, so treatment
does nothing by construction while covariates still predict the outcome strongly.
A model can fit; there's no effect to find. The placebo test must say so. This is a
test *of a test*, and it's the only thing standing between you and a robustness
check that always passes.

`test_seed_stability_overlap_is_one_when_the_score_cannot_change` (143) passes
`seeds=(4, 4)` — the same seed twice. The overlap must be exactly 1.0 and the Qini
SD exactly 0. If the metric is broken (self-pairs included, wrong set arithmetic)
this catches it with an unambiguous expected value.

---

## What I'd fix before showing this to anyone

Ordered by how much I'd care. The first three are **done** — the walkthrough above
describes the fixed code — and the rest are still open.

**1. ~~`choose_policy` reports a non-zero profit for a zero-fraction
recommendation.~~ FIXED.** `src/policy.py:280-292`. When `recommended_fraction`
was 0.0 — zero budget, or a budget too small for one frontier step —
`.abs().idxmin()` snapped to the smallest frontier row (fraction 0.025) and
returned its profit: `recommended_fraction=0.0, recommended_profit=233.90`. The
existing test asserted only the fraction, so it stayed green. Now guarded
explicitly, with `recommended_profit` and `profit_vs_random` both asserted and a
second test for a sub-resolution budget.

**2. ~~`np.trapezoid` needs NumPy ≥ 2.0; the declared floor is 1.24.~~ FIXED.**
`src/evaluation.py:184-185, 240` against `pyproject.toml:14`. `np.trapezoid`
arrived in NumPy 2.0 (1.x has only `np.trapz`), so anyone resolving to 1.26 — easy
via a transitive pin — got an `AttributeError` inside `qini_curve`. CI never saw it
because pip resolves to 2.2.x. Floor raised to `numpy>=2.0,<2.3` in both
`pyproject.toml` and `requirements.txt`, with a comment saying why.

**3. ~~Stale lint config.~~ FIXED.** `pyproject.toml:41` and `:78` both excluded
`legacy_jumia`, a directory that stopped existing when the history was replayed
onto an empty `main`. Removed from the ruff and mypy excludes.

**4. `load_criteo` has never been run.** `src/data.py:233`. No test, no CLI path,
host unreachable from where this was built. It reads as a working feature and
isn't verified as one. Delete it or exercise it.

**5. `_incremental_at` returns zeros for an unmeasurable selection.**
`src/policy.py:123-124`. A selection with treated customers but no controls yields
`(0.0, 0.0, 0.0)` — recorded as "no effect" rather than "cannot measure." Doesn't
fire on Hillstrom at the current grid; would fire on a small dataset or a finer
grid, producing a flat zero region on the frontier that looks like an economic
finding. Return `nan`.

**6. Best-model selection and evaluation use the same out-of-fold scores.**
`src/cli.py:311-315`. The winner's reported Qini is optimistic by an uncomputed
amount. Nested CV is the fix. The memo's conclusion survives because it rests on
the placebo test, not the leaderboard — but the leaderboard numbers in
`results/models_*.csv` should carry a caveat and don't.

**7. The `20.0` engagement multiplier is duplicated across files.**
`src/simulate.py:238` and `src/simulate.py:297` and `src/experiment.py:333`. Three
copies of one magic number, one of them in a different module. Edit the formula in
`simulate()` and the confounded variant and the CUPED demo both go quietly wrong.
Make it a named constant next to `REVENUE_PER_CONVERSION`.

**8. `simulate_confounded` reconstructs `engagement` algebraically.**
`src/simulate.py:295-300`. Correct and exact, but it depends on a formula defined
60 lines away and re-derived by hand. Storing `loyalty` on `SyntheticData` and
recomputing properly would remove the coupling.

**9. `test_x_learner_beats_t_learner_under_imbalance` scores in sample.**
`tests/test_estimators.py:108-115`. It uses `fit_predict`, so it's checking an
in-sample property. The claim it defends is about out-of-sample recovery. Should
use `cross_val_uplift`.

**10. `run_all_checks`' return dict is mutated by its caller.**
`src/cli.py:373-374` calls `.pop()` on the returned dict, then stores the
remainder in the payload. Works, but the function's contract now depends on what
the caller does to it afterwards. Return the tables separately.

**11. `import copy` inside the CV loop.** `src/evaluation.py:456`. Cosmetic. Same
for the function-local `scipy` imports at `src/evaluation.py:523` and
`src/robustness.py:161` — those were deliberate (scipy import is slow and only two
functions need it) but they're inconsistent with `naive.py`, which imports scipy
at module level.

**12. `random_targeting_profit` takes `outcome` only to read its length.**
`src/policy.py:193-222`. Misleading signature — it suggests the outcome affects the
result, and it doesn't; only `spend` and `treatment` do. Take `n: int` instead.

**13. `describe()` and `recovery_error` disagree about empty groups.**
`src/simulate.py:335` returns `0.0` for `sleeping_dog_mean_uplift` when there are
no dogs; `src/evaluation.py:521` returns `nan` in the analogous case, with a
comment explaining that "nothing to find" must not read as "found nothing." The
`nan` choice is right and `describe` should match it.

**14. The placebo threshold and replicate count are coupled but unlinked.**
`src/robustness.py:132` uses `exceed < 0.10`, which is calibrated to
`n_replicates=10`. Raise the replicates without tightening the threshold and the
test gets weaker than intended. Nothing in the code says so.

**15. `results/` is committed but `data/hillstrom.csv` is gitignored.** Someone
cloning the repo gets the outputs but has to re-download the input to reproduce
them. That's the right call for repo size, but the README should say the committed
results were produced by a specific run and re-running regenerates them.

---

## Rebuild checklist

From an empty directory, in this order:

1. `git init`; `.gitignore` (Python + `.venv/` + `data/*.csv`)
2. `pyproject.toml` — deps, ruff (strict select), mypy (strict), pytest config. Set `numpy>=2.0`.
3. `src/__init__.py` — empty package marker
4. `src/config.py` — paths, both dataset URLs, seed, `N_BOOTSTRAP`, `N_FOLDS`, cost, margin, budget fraction
5. `src/arrays.py` — the `Array` alias (write it now, not after mypy shouts)
6. `src/simulate.py` — `SyntheticData`, `_uplift_function` (override, not subtract), `_baseline_function` (x3 drives baseline only), `simulate` (clip-then-recompute uplift, shared uniform draw, loyalty → pre_period + engagement), `simulate_confounded`, `describe`
7. `tests/conftest.py` + `tests/test_simulate.py` — lock the yardstick before building on it. Assert `max` over the dog subgroup, not `mean`.
8. `src/estimators.py` — `LearnerFactory`, `_positive_probability`, `UpliftEstimator` ABC, S/T/X learners, `CausalForestLearner` (remember `discrete_outcome=True`), `build_estimators`
9. `tests/test_estimators.py` — shrink/spread out-of-fold, X-beats-T under imbalance, dogs ranked low
10. `src/evaluation.py` — `_order_by_score` (`lexsort((jitter, -score))`), `qini_curve` (rescale by `n_t/n_c`), `auuc_score`, `transformed_outcome`, `uplift_by_decile`, `bootstrap_qini` (stratified), `cross_val_uplift` (stratify on the 2×2 cross), `evaluate`, `recovery_error`
11. `tests/test_evaluation.py` — random→zero, oracle→positive, reversal→negative, constant→no row-order signal, beats-random→fires only for real signal
12. `src/policy.py` — `_incremental_at` (measure from the holdout), `profit_frontier`, `random_targeting_profit`, `choose_policy` (cap vs best-affordable as separate fields), `sleeping_dog_report` (upper CI < 0), `bootstrap_policy` (fixed fraction)
13. `tests/test_policy.py` — profit identity, treat-all == random-at-100%, budget caps, sleeping-dog key parity
14. `src/data.py` — atomic download with mirror fallback, `encode_features` (drop `history_segment`), `load_hillstrom` (two arms, never pooled), `dataset_summary`
15. `src/naive.py` — `naive_comparison` (pooled SE for p, unpooled for CI), `compare_randomised_and_confounded`, `response_versus_uplift_ranking` (response model on treated rows only)
16. `tests/test_naive.py` — unbiased under randomisation, inflated under confounding, uplift beats response on true captured effect
17. `src/experiment.py` — `required_sample_size` (pooled for α, unpooled for power), `minimum_detectable_effect` (bisect the forward function), `cuped_adjust` (pooled theta, centre the covariate), `cuped_demo`, `design_validation_experiment`
18. `tests/test_experiment.py` — round-trip MDE, variance reduction == ρ² to 1e-9, post-treatment covariate biases
19. `src/robustness.py` — `placebo_test` (permute treatment, not outcome; refit the reference under the same protocol), `covariate_balance`, `cost_sensitivity`, `seed_stability` (fixed tie-break seed, pairs via `[i+1:]`), `run_all_checks`
20. `tests/test_robustness.py` — balance catches confounding and names the right column, placebo fails on a null outcome, self-pair overlap == 1.0
21. `src/cli.py` — `_jsonable`, `stage_validate`, `stage_naive`, `_run_one_dataset`, `stage_hillstrom`, `stage_experiment`, `main`
22. Run `python -m src.cli all`; read `results/` before writing a word of the memo
23. `ruff check --fix`, `ruff format`, `mypy` — fix properly, don't loosen the config.
    Exclude `*.md` from ruff first: it formats fenced Python blocks inside Markdown,
    and it will silently rewrite any source you quoted verbatim — including turning a
    quoted keyword argument into a one-element tuple, which is a different program.
24. `.github/workflows/ci.yml` — lint, types, tests, plus a synthetic-only smoke run
25. `MEMO.md` from the actual numbers, then `METHODS.md`, then `README.md`

The one rule: **nothing at step N is allowed to depend on a result from step N+1.**
The moment you tune the simulator so an estimator looks better, or soften a metric
so a model passes, the study stops being able to tell you anything you didn't
already believe.
