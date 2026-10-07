/**
 * Sample-size arithmetic, ported from `src/experiment.py`.
 *
 * This is the one place where the browser computes a number rather than
 * reading one. It is here because a sample-size calculator is only useful if
 * it responds to your own numbers, and shipping a lookup table of every
 * baseline rate someone might type is not a plan.
 *
 * The port is deliberately line-for-line with the Python rather than
 * "equivalent". Two implementations of the same formula that drift apart is a
 * genuinely nasty bug class — the site would quietly disagree with the study
 * it is presenting — so `tests/test_web_parity.py` checks a grid of inputs
 * against this file and fails if they diverge by more than a rounding step.
 *
 * If you change anything in here, change it in `src/experiment.py` too, and
 * let that test tell you whether you got it right.
 */

/**
 * Inverse of the standard normal cumulative distribution (`scipy.stats.norm.ppf`).
 *
 * Acklam's rational approximation, with one Halley refinement step. Accurate
 * to about 1e-15 after refinement, which is far beyond anything that matters
 * for a sample size but costs nothing.
 */
export function normInv(p) {
  if (p <= 0 || p >= 1) return p <= 0 ? -Infinity : Infinity;

  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

  const lower = 0.02425;
  const upper = 1 - lower;
  let x;

  if (p < lower) {
    const q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= upper) {
    const q = p - 0.5;
    const r = q * q;
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  // One Halley step against the true CDF, which removes the approximation's
  // last few digits of error.
  const e = 0.5 * erfc(-x / Math.SQRT2) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

/** Complementary error function, via a Chebyshev-fitted rational approximation. */
export function erfc(x) {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;

  const coefficients = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
    -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
    6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10, 9.6467911e-11,
    2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15,
  ];

  let d = 0;
  let dd = 0;
  for (let j = coefficients.length - 1; j > 0; j -= 1) {
    const tmp = d;
    d = ty * d - dd + coefficients[j];
    dd = tmp;
  }
  const result = t * Math.exp(-z * z + 0.5 * (coefficients[0] + ty * d) - dd);
  return x >= 0 ? result : 2 - result;
}

/** Standard normal cumulative distribution. */
export function normCdf(x) {
  return 0.5 * erfc(-x / Math.SQRT2);
}

/**
 * Customers needed per arm to detect a given lift in a conversion rate.
 *
 * Port of `required_sample_size`. The formula is the standard normal
 * approximation for the difference of two proportions:
 *
 *   n = (z_alpha * sqrt((1 + 1/k) * p_bar * (1 - p_bar))
 *        + z_power * sqrt(p0(1-p0) + p1(1-p1)/k))^2 / delta^2
 *
 * The detail that is easy to get wrong, and that changes the answer by a
 * noticeable margin: the alpha term uses the POOLED rate `p_bar`, because
 * under the null hypothesis both arms share one rate; the power term uses the
 * two arms' SEPARATE rates, because under the alternative they do not. Using
 * pooled variance for both — the version that shows up in a lot of online
 * calculators — understates the sample size needed.
 *
 * @param {object} spec
 * @param {number} spec.baselineRate Control conversion rate, strictly in (0, 1).
 * @param {number} [spec.absoluteMde] Effect in absolute terms. Supply this or relativeMde.
 * @param {number} [spec.relativeMde] Effect as a fraction of baseline.
 * @param {number} [spec.alpha] Two-sided significance level.
 * @param {number} [spec.power] Desired power.
 * @param {number} [spec.allocationRatio] Treated per control. 1 is equal arms.
 */
export function requiredSampleSize({
  baselineRate,
  absoluteMde = null,
  relativeMde = null,
  alpha = 0.05,
  power = 0.8,
  allocationRatio = 1,
}) {
  if ((absoluteMde === null) === (relativeMde === null)) {
    throw new Error("supply exactly one of absoluteMde or relativeMde");
  }
  if (!(baselineRate > 0 && baselineRate < 1)) {
    throw new Error("baselineRate must lie strictly between 0 and 1");
  }

  const delta = absoluteMde !== null ? absoluteMde : baselineRate * relativeMde;
  if (delta <= 0) throw new Error("the effect size must be positive");

  const p0 = baselineRate;
  const p1 = Math.min(p0 + delta, 1 - 1e-9);
  const k = allocationRatio;
  const pBar = (p0 + k * p1) / (1 + k);

  const zAlpha = normInv(1 - alpha / 2);
  const zPower = normInv(power);

  const numerator =
    (zAlpha * Math.sqrt((1 + 1 / k) * pBar * (1 - pBar)) +
      zPower * Math.sqrt(p0 * (1 - p0) + (p1 * (1 - p1)) / k)) **
    2;

  const nControl = Math.ceil(numerator / delta ** 2);
  const nTreated = Math.ceil(k * nControl);

  return {
    baselineRate: p0,
    treatmentRate: p1,
    absoluteMde: delta,
    relativeMde: delta / p0,
    alpha,
    power,
    nPerArm: nControl,
    nTotal: nControl + nTreated,
    expectedConversionsPerArm: nControl * p0,
  };
}

/**
 * The inverse: given the customers available, what can the test actually detect.
 *
 * Port of `minimum_detectable_effect`. Solved by bisection on
 * `requiredSampleSize` rather than by a separately derived closed form, which
 * keeps the two directions guaranteed consistent — a closed form would be
 * faster and would eventually drift out of agreement with the forward
 * calculation after someone edits one of them.
 */
export function minimumDetectableEffect({ baselineRate, nPerArm, alpha = 0.05, power = 0.8 }) {
  if (nPerArm <= 0) throw new Error("nPerArm must be positive");

  let low = 1e-6;
  let high = Math.min(0.5, 1 - baselineRate - 1e-6);
  for (let i = 0; i < 200; i += 1) {
    const mid = 0.5 * (low + high);
    const needed = requiredSampleSize({ baselineRate, absoluteMde: mid, alpha, power });
    if (needed.nPerArm > nPerArm) low = mid;
    else high = mid;
  }

  return {
    baselineRate,
    nPerArm,
    absoluteMde: high,
    relativeMde: high / baselineRate,
    detectableTreatmentRate: baselineRate + high,
    alpha,
    power,
  };
}

/**
 * Size a head-to-head test of model targeting against the incumbent policy.
 *
 * Port of `design_validation_experiment`. The thing under test is the
 * DIFFERENCE between two policies' incremental rates, not either policy's
 * effect on its own — which is why this needs so many more customers than
 * proving the email works at all. Beating a baseline is easy; beating a
 * baseline by a measurable margin over another method is not.
 */
export function designValidation({
  baselineRate,
  expectedPolicyUplift,
  expectedRandomUplift,
  alpha = 0.05,
  power = 0.8,
  weeklyVolume = null,
  cupedVarianceReduction = 0,
}) {
  const effect = expectedPolicyUplift - expectedRandomUplift;
  if (effect <= 0) {
    return {
      differenceUnderTest: effect,
      nPerCell: Infinity,
      nTotal: Infinity,
      weeksRequired: Infinity,
      impossible: true,
    };
  }

  const baselineForTest = baselineRate + expectedRandomUplift;
  const sizing = requiredSampleSize({
    baselineRate: baselineForTest,
    absoluteMde: effect,
    alpha,
    power,
  });
  const nPerCell = sizing.nPerArm * (1 - cupedVarianceReduction);
  const nTotal = 2 * nPerCell;

  return {
    differenceUnderTest: effect,
    baselineForTest,
    nPerCell: Math.ceil(nPerCell),
    nTotal: Math.ceil(nTotal),
    expectedConversionsPerCell: nPerCell * baselineForTest,
    weeksRequired: weeklyVolume ? Math.ceil(nTotal / weeklyVolume) : null,
    alpha,
    power,
    cupedVarianceReduction,
    impossible: false,
  };
}
