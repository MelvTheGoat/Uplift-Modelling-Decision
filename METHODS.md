# Methods

Technical companion to `MEMO.md`. This document states what each estimator
assumes, how each metric is defined, and where each one fails. It is written for
someone who has to defend or replace this analysis.

---

## 1. The quantity being estimated

For customer *i* with covariates *Xᵢ*, write *Yᵢ(1)* for the outcome if they are
contacted and *Yᵢ(0)* for the outcome if they are not. The **individual treatment
effect** is

    τ(x) = E[Y(1) − Y(0) | X = x]

This is the number the campaign decision needs, and it is never observed: each
customer supplies exactly one of the two potential outcomes. Everything below is
a strategy for estimating a difference of two things only one of which was ever
seen — the "fundamental problem of causal inference".

The **average treatment effect** is `ATE = E[τ(X)]`. It is easy to estimate from a
randomised trial and it is not what a targeting decision needs, which is the
distinction the memo's second section is about.

### Assumptions

All four estimators below require:

1. **Unconfoundedness**: `(Y(1), Y(0)) ⫫ W | X`. Given the covariates, treatment
   assignment carries no information about the potential outcomes. Hillstrom
   satisfies this by design — assignment was randomised, so it holds even
   unconditionally. In an observational marketing log it does not hold at all
   and no amount of modelling repairs it.
2. **Overlap**: `0 < P(W = 1 | X) < 1` for all *x*. Every kind of customer must
   have had some chance of being in either arm. Hillstrom is a 1/3–1/3–1/3 split,
   so overlap is exact and known.
3. **SUTVA / no interference**: one customer's treatment does not change another
   customer's outcome. Plausible for direct mail; less plausible for anything with
   a social or inventory-mediated channel.

Randomisation buys us (1) and (2). It does not buy us (3), and it does not make
the *individual* effect observable — only the average.

---

## 2. Estimators

All four share the interface `fit(features, treatment, outcome)` →
`predict_uplift(features)`, and all return effects on the outcome's own scale (for
a binary outcome, an absolute change in probability).

### S-learner (single model)

Fit one model `μ(x, w)` on the pooled data with treatment as an extra feature.
Predict with the flag forced on and forced off, and difference:

    τ̂(x) = μ̂(x, 1) − μ̂(x, 0)

**Assumes** the base learner will actually split on the treatment column.

**Fails** when it does not. One binary column competes against a dozen strong
predictors of the outcome; a regularised gradient booster spends its splits where
the outcome variance is, which is not on the treatment flag. The predicted effects
then collapse toward a constant. This is a bias *toward finding no
heterogeneity*, and it is measured directly in this project: on synthetic data the
standard deviation of S-learner's predicted effects is **0.65×** the truth
(`results/validation_balanced_50_50.csv`, `sd_ratio`).

**Use it** when the treatment effect is large relative to the outcome signal, or
as the stable baseline that other learners must beat.

### T-learner (two models)

Fit `μ₁` on the treated arm and `μ₀` on the control arm separately:

    τ̂(x) = μ̂₁(x) − μ̂₀(x)

**Assumes** nothing about the two response surfaces sharing a shape, which is its
advantage over S-learner.

**Fails** because it models the *outcome* twice and the *difference* never. Both
models spend their capacity on the parts of the response common to both arms —
which cancel in the subtraction and are irrelevant to the decision — and their two
independent error terms add. Measured here: T-learner's predicted-effect spread is
**1.7×** the truth at a balanced split and **2.1×** under imbalance. It is
inflating noise, not finding heterogeneity.

**Use it** when both arms are large and the outcome is not rare.

### X-learner

Two stages.

*Stage 1*: fit `μ₁` and `μ₀` as in T-learner.

*Stage 2*: impute each customer's missing counterfactual from the **other** arm's
model, producing a pseudo-effect per row:

    D₁ᵢ = Yᵢ − μ̂₀(Xᵢ)   for treated customers
    D₀ᵢ = μ̂₁(Xᵢ) − Yᵢ   for control customers

Regress `D₁` on `X` among the treated to get `τ̂₁`, and `D₀` on `X` among the
controls to get `τ̂₀`. Combine by the propensity score `g(x) = P(W = 1 | X = x)`:

    τ̂(x) = g(x)·τ̂₀(x) + (1 − g(x))·τ̂₁(x)

**Why the weights go that way**, since this is the part that looks backwards: when
treatment is *rare*, `g(x)` is small, so the formula puts most weight on `τ̂₁` —
the effect model fitted on the small treated arm. That is deliberate. `τ̂₁` is
built from *observed* treated outcomes minus a control-model prediction, and the
control model is the well-estimated one. The alternative, `τ̂₀`, leans on `μ̂₁`,
fitted on the scarce arm, to predict counterfactuals for the plentiful arm. The
weighting picks whichever pseudo-effect rests on the better-estimated nuisance
model.

**Assumes**, additionally, that the second-stage regression can fit the effect
surface — it inherits the base learner's bias and adds another layer of it.

**Fails** when both arms are small, where the two-stage structure just compounds
error. Measured here: at a 15/85 split, X-learner's PEHE is **0.088** against
T-learner's **0.106** — the predicted advantage, and a modest one.

`propensity` should be passed explicitly when the design is a known randomised
split. Estimating a constant from the data only adds variance.

### Causal forest (EconML `CausalForestDML`)

Random forest whose splits are chosen to maximise heterogeneity in the *effect*
rather than in the outcome, with **honest** estimation: the sample used to choose
a split is disjoint from the sample used to estimate the effect in the resulting
leaf. Outcome and treatment are first orthogonalised against the covariates
(double machine learning), so the forest models the residual-on-residual
relationship.

**Assumes** unconfoundedness and overlap as above, plus enough data that honest
splitting is affordable.

**Fails** on rare outcomes, because honesty halves the data available for
estimation. With a 0.9% conversion rate, leaves that satisfy `min_samples_leaf`
still contain very few converters.

**Gives** valid pointwise confidence intervals, which the meta-learners do not.

| Estimator | Balanced PEHE | Imbalanced PEHE | sd(τ̂)/sd(τ) balanced | Cost |
|---|---|---|---|---|
| S-learner | 0.037 | 0.043 | 0.65 (shrinks) | fastest |
| T-learner | 0.086 | 0.106 | 1.69 (inflates) | fast |
| X-learner | 0.063 | 0.088 | 1.34 | fast |
| Causal forest | 0.035 | 0.040 | 0.66 | ~50× slower |

Source: `results/validation_*.csv`, n = 20,000, 5-fold cross-validated.

---

## 3. Evaluation

### Why there is no AUC anywhere

AUC measures how well a score separates responders from non-responders. An uplift
model is not trying to do that, and the best possible uplift model would score
*badly* on it: a customer who was always going to buy has zero uplift and should
be ranked low, which AUC penalises. High AUC with worthless targeting is the
normal case, not an edge case. The word does not appear in `src/evaluation.py`.

### Qini curve

Rank customers by predicted uplift, descending. Walking down that list, at
position *k*:

    Q(k) = Y_T(k) − Y_C(k) · N_T(k) / N_C(k)

where `Y_T(k)`, `N_T(k)` are responders and customers from the treated arm among
the first *k*, and likewise for control. The second term rescales the control
group to the treated group's size, which is what turns a comparison of two
differently sized counts into an estimate of *incremental conversions*.

`Q(N)` is the total incremental effect of treating everyone — independent of the
model. The straight line from `(0, 0)` to `(1, Q(N))` is what random targeting
traces. A model with information bows above it.

### Qini coefficient

    qini_coefficient = ∫Q(φ)dφ − ∫random(φ)dφ

Reported in **incremental conversions**, so it can be sanity-checked against the
business. `normalized_qini` divides by `|∫random|` to give a unitless version
comparable across datasets; it is undefined when the overall effect is zero.

Note this is *not* normalised by a "perfect model" curve, as some implementations
are. Constructing that curve requires the counterfactuals, which do not exist
outside `src/simulate.py`. The absolute version is honest about what is
measurable.

### AUUC

Area under the uplift curve, where `u(φ) = (Y_T/N_T − Y_C/N_C)·(N_T + N_C)`. Takes
the difference of *rates* and scales by the count, rather than rescaling counts.
Agrees with Qini under balanced arms and diverges when they are not.

### Uplift by decile

No modelling: split the ranked file into ten equal bins, and within each bin
report the treated response rate minus the control response rate, with the
standard error of a difference of two proportions. If the top decile does not
beat the bottom decile, nothing else in the analysis matters. This is the table to
put in front of a sceptic.

### Transformed outcome

    Z = Y · (W − p) / (p(1 − p))

Under randomisation, `E[Z | X] = τ(X)`. This converts effect estimation into an
ordinary regression problem and gives a validation loss computable on held-out
data: `mean((Z − τ̂)²)`.

The catch is variance. At a 50/50 split a single converting treated customer
contributes `Z = +2` against a true effect of order 0.01. The loss is unbiased and
extremely noisy, dominated by an irreducible term identical across models, so
absolute values are meaningless and only differences between models on identical
data carry information. Used here for model selection, never as a headline.

### Cross-validation

All reported metrics use out-of-fold predictions (5 folds, stratified on the
treatment × outcome cross so that every fold contains both arms and some
converters at a 0.9% base rate). In-sample, every estimator appears to have found
heterogeneity, because it fitted the noise in each arm separately.

### Bootstrap intervals

Rows are resampled with replacement **stratified by arm**, holding the treated and
control counts fixed, so the resampling reflects outcome noise rather than noise
in the experimental design. 500 replicates; percentile intervals.

### Beats-random test

One-sided: the null is that the model's ordering is no better than a shuffle. The
p-value is the share of bootstrap replicates with a Qini coefficient at or below
zero.

**This test is weaker than it looks**, and the difference matters. It holds the
*fitted model* fixed and resamples only the evaluation rows. It therefore cannot
see variance from model fitting — the risk that a different training sample would
have produced a different ranking. The placebo test does see that, and on this
dataset the two disagree: models that pass the bootstrap test fail the placebo
test. Where they disagree, believe the placebo.

### Recovery error (synthetic only)

* **PEHE**: `sqrt(mean((τ̂ − τ)²))`, the root mean squared error on the individual
  effect. The primary accuracy measure.
* **ATE bias**: `mean(τ̂) − mean(τ)`.
* **Kendall's τ / Spearman**: rank agreement with the truth. For a targeting
  decision the ranking is what matters, so these carry more weight than PEHE.
* **`sd_ratio`**: `sd(τ̂)/sd(τ)`. Below 1 means shrinkage, above 1 means noise
  inflation. This is the diagnostic that exposes S-learner and T-learner.
* **`sleeping_dogs_detected`**: share of genuinely harmed customers given a
  negative predicted effect.

---

## 4. Targeting policy

### Measuring, not forecasting

Given a ranking, for each candidate depth φ the incremental effect of treating the
top φ is measured **from the randomised holdout inside that selection**:

    incremental rate = mean(Y | selected, treated) − mean(Y | selected, control)
    incremental total = incremental rate × (number selected)

Selection depends only on covariates, so randomisation survives it and the
within-selection comparison remains unbiased. The model's own predicted uplift is
used *only* to order customers, never to value them. Using predictions as values
is circular and is the most common way an uplift analysis overstates itself.

### Profit

    profit(φ) = incremental_revenue(φ) × margin_rate − n_targeted(φ) × cost_per_contact

with `cost_per_contact = $0.10` and `margin_rate = 0.30` (see `src/config.py`).
Both are assumptions, not measurements, and `robustness.cost_sensitivity` sweeps
them.

### Choosing the depth

Two constraints can bind:

* **Economic**: past some depth, added customers contribute less incremental
  margin than they cost, and profit turns down.
* **Budget**: `budget / cost_per_contact` customers is all that can be afforded.

The recommendation is the tighter one. When economics bind first, the correct
answer is *not to spend the whole budget*.

### Uncertainty

`bootstrap_policy` holds the targeting fraction **fixed** and bootstraps profit at
that depth. Re-optimising the fraction inside each replicate would report the
spread of the best achievable profit under hindsight, which is a different and
smaller number than the uncertainty in the decision actually being proposed.

### Sleeping dogs

Customers with predicted uplift below zero. Two distinct numbers, and conflating
them is the standard error:

* **flagged**: how many the model would exclude — a property of the model;
* **measured uplift within the flagged group**, with a confidence interval — whether
  the model was right. `measured_uplift_is_negative` is set only when the interval
  lies entirely below zero.

A model can flag 20% of the file and be wrong about every one of them.

---

## 5. Experiment design

### Sample size

Normal approximation for the difference of two proportions:

    n = (z_{1−α/2}·sqrt((1 + 1/k)·p̄(1 − p̄)) + z_power·sqrt(p₀(1 − p₀) + p₁(1 − p₁)/k))² / δ²

where `k` is the allocation ratio and `p̄` the weighted pooled rate. Reliable when
the expected converters per arm reach at least a few dozen —
`expected_conversions_per_arm` is returned so this can be checked rather than
assumed.

The **minimum detectable effect** is obtained by bisection on the same function,
which guarantees the two directions stay consistent instead of relying on a
separately derived closed form that could drift.

### CUPED

    Y_adjusted = Y − θ·(X_pre − mean(X_pre)),   θ = Cov(Y, X_pre)/Var(X_pre)

Variance falls by exactly `ρ²`, where `ρ = corr(Y, X_pre)`; the effective sample
size multiplies by `1/(1 − ρ²)`. Verified numerically to 15 decimal places in
`tests/test_experiment.py`.

**Assumptions**:

1. **The covariate is measured strictly before assignment.** The load-bearing one.
   If the treatment can influence the covariate, subtracting it removes part of
   the effect and biases the estimate toward zero. A "pre-period" window that
   overlaps the campaign is how this goes wrong in practice — demonstrated in
   `test_cuped_on_a_post_treatment_covariate_would_bias_the_estimate`.
2. **The covariate correlates with the outcome.** Gain is `ρ²`, so ρ = 0.3 buys 9%
   and is not worth the complexity; ρ = 0.7 buys 50% and is.
3. **θ is estimated on pooled data.** Harmless under randomisation. Estimating it
   per arm re-introduces the very difference being measured.
4. **CUPED reduces variance, never bias.** It cannot rescue a broken randomisation
   and does nothing about confounding.

Measured on synthetic data: **50.1%** variance reduction on a continuous
engagement metric (effective sample ×2.00), **1.6%** on a rare binary conversion
(×1.02). The technique is metric-dependent, and quoting a number from a
continuous-metric case study to justify it on a conversion test is a mistake.

---

## 6. Robustness

| Check | What it does | What a failure means |
|---|---|---|
| **Placebo** | Permute treatment labels and refit end to end. True effect is zero by construction. | The Qini coefficient the model manufactures from noise. If the real result sits inside that distribution, the ranking is not established. |
| **Covariate balance** | Standardised mean differences across arms. <0.10 is balance, >0.25 is a problem. | On a randomised trial, a failure indicts the data handling, not the trial. |
| **Cost sensitivity** | Sweep cost per contact and margin. | A recommendation that swings wildly over a plausible cost range is not a recommendation. |
| **Seed stability** | Refit under different seeds; report the spread of the Qini coefficient *and* the Jaccard overlap of the selected customer lists. | Stable aggregate performance with churning customer lists still means the specific list is untrustworthy. The overlap is the more demanding test and the operationally relevant one. |

The placebo test is the most informative check here and the one most often
omitted from published uplift work. It is the only check that captures variance
from model fitting rather than only from evaluation sampling.

---

## 7. Data

**Hillstrom MineThatData** (2008). 64,000 customers who purchased in the previous
twelve months, randomised three ways two weeks before a two-week outcome window:
`Mens E-Mail` (21,307), `Womens E-Mail` (21,387), `No E-Mail` (21,306).

Outcomes: `visit` (binary, came to the site), `conversion` (binary, purchased),
`spend` (continuous revenue).

Features used: `recency`, `history`, `mens`, `womens`, `newbie`, plus one-hot
`zip_code` (Rural / Surburban / Urban) and `channel` (Phone / Web / Multichannel).
`history_segment` is **dropped**: it is a coarse binning of `history`, which is
already present at full resolution, so keeping both adds collinearity without
information.

The two e-mail arms are analysed **separately**, not pooled. They are different
creatives with different measured effects; averaging them would answer a question
nobody asked. `treatment_arm="any"` pools them for completeness, with the caveat
that the resulting "treatment" is a mixture.

The canonical URL (`minethatdata.com`, plain HTTP) is blocked by many egress
policies; `src/data.py` falls back to a byte-identical GitHub mirror.

**Criteo-UPLIFT v2** is supported behind `load_criteo()` and is optional
throughout — nothing in the study blocks on it.

---

## 8. Reproducibility

Seed `20080320` (`src/config.py`) throughout. Every estimator takes an explicit
`random_state`. Fold assignment, bootstrap resampling and Qini tie-breaking all
take explicit seeds.

Two sources of run-to-run variation remain: LightGBM's threading is not bit-exact
across machines with different core counts, and `results/` regenerates on every
run. Conclusions are stable across seeds; the specific customer list at a given
depth is not, which is itself one of the reported findings.
