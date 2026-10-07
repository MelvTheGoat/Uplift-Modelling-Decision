/**
 * The analysis engine, ported from `src/`.
 *
 * This module is what makes the site a tool rather than a report: it runs the
 * same analysis on a visitor's own campaign, in their browser, with the file
 * never leaving their machine.
 *
 * It is a straight port of the Python, function for function, and
 * `tests/test_web_analysis_parity.py` generates a campaign, runs both, and
 * fails if they disagree. If you change anything here, change `src/` too and
 * let that test tell you whether you got it right.
 *
 * Deliberately free of any DOM reference, so Node can import it for that test.
 *
 * ## One honest difference from the Python
 *
 * Ties in the ranking are broken at random rather than by row order, because
 * row order in a sorted export carries real signal and would flatter a model
 * that has none. Both implementations do this; they cannot use the *same*
 * random numbers, because reproducing NumPy's PCG64 stream in JavaScript is not
 * worth the trouble. So on data with tied scores the two can order tied rows
 * differently and the fourth decimal place of a Qini coefficient may differ.
 * On data with distinct scores they agree exactly, which is what the parity
 * test uses.
 */

import { normCdf, requiredSampleSize, minimumDetectableEffect } from "./stats.js";

/**
 * A small seeded random number generator (mulberry32).
 *
 * Seeded rather than `Math.random` so that re-running an analysis on the same
 * file gives the same answer. A tool whose numbers shift slightly every time
 * you press the button is a tool nobody trusts.
 */
export function makeRng(seed = 1) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Indices that sort `score` descending, breaking ties at random.
 *
 * Ties matter more than they look. A model that predicts one constant value —
 * which is what a badly shrunk single-model learner does — would otherwise be
 * ranked by row order, producing a curve that flatters a model carrying no
 * information at all.
 */
export function orderByScore(score, seed = 0) {
  const rng = makeRng(seed + 1);
  const jitter = new Float64Array(score.length);
  for (let i = 0; i < score.length; i += 1) jitter[i] = rng();

  const order = Array.from({ length: score.length }, (_, i) => i);
  order.sort((a, b) => {
    if (score[b] !== score[a]) return score[b] - score[a];
    return jitter[a] - jitter[b];
  });
  return order;
}

/**
 * Round half to even, matching Python's `round()`.
 *
 * `Math.round` rounds half *up*, Python rounds half to *even*, and the two
 * disagree on exactly the values a frontier hits: with 40 cut-offs over an
 * even number of customers, `fraction * n` lands on a .5 boundary regularly.
 * Using the wrong one shifts a cut-off by one customer, which is both
 * invisible and enough to make the two implementations disagree.
 */
export function roundHalfToEven(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Trapezoidal integration, matching `numpy.trapezoid`. */
export function trapezoid(y, x) {
  let total = 0;
  for (let i = 1; i < y.length; i += 1) {
    total += ((x[i] - x[i - 1]) * (y[i] + y[i - 1])) / 2;
  }
  return total;
}

/** Mean of an array, or NaN when empty. */
function mean(values) {
  if (!values.length) return NaN;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

// ------------------------------------------------------------------ naive

/**
 * The simple before-and-after comparison, with its uncertainty.
 *
 * A two-proportion z-test. Nothing clever, and that is the point: it answers
 * "did this work at all?", which is a far easier question than "who did it
 * work on?" and should always be answered first.
 */
export function naiveComparison(outcome, treatment) {
  const treated = [];
  const control = [];
  for (let i = 0; i < outcome.length; i += 1) {
    (treatment[i] === 1 ? treated : control).push(outcome[i]);
  }
  const rateTreated = mean(treated);
  const rateControl = mean(control);
  const difference = rateTreated - rateControl;

  const varTreated = (rateTreated * (1 - rateTreated)) / treated.length;
  const varControl = (rateControl * (1 - rateControl)) / control.length;
  const stderr = Math.sqrt(varTreated + varControl);

  // Pooled rate for the test statistic: under the null the two arms share one
  // rate, so the standard error is computed from the combined data rather than
  // from the two arms separately.
  const pooled = (rateTreated * treated.length + rateControl * control.length) /
    (treated.length + control.length);
  const pooledStderr = Math.sqrt(
    pooled * (1 - pooled) * (1 / treated.length + 1 / control.length),
  );
  const z = pooledStderr > 0 ? difference / pooledStderr : 0;
  const pValue = 2 * (1 - normCdf(Math.abs(z)));

  return {
    nTreated: treated.length,
    nControl: control.length,
    rateTreated,
    rateControl,
    absoluteDifference: difference,
    relativeLift: rateControl > 0 ? difference / rateControl : NaN,
    stderr,
    ciLow: difference - 1.96 * stderr,
    ciHigh: difference + 1.96 * stderr,
    z,
    pValue,
  };
}

// ------------------------------------------------------------------ balance

/**
 * Standardised differences between the arms, for every characteristic.
 *
 * In a textbook this check is unnecessary — random assignment guarantees
 * balance. In practice it is the check that catches a send that failed
 * silently for one region, a suppression list applied to one arm, or an export
 * that dropped rows unevenly. None of those announce themselves.
 */
export function covariateBalance(columns, treatment) {
  const treatedIndices = [];
  const controlIndices = [];
  for (let i = 0; i < treatment.length; i += 1) {
    (treatment[i] === 1 ? treatedIndices : controlIndices).push(i);
  }

  return columns.map(({ name, values }) => {
    const treated = treatedIndices.map((i) => values[i]);
    const control = controlIndices.map((i) => values[i]);
    const meanTreated = mean(treated);
    const meanControl = mean(control);

    const variance = (sample, centre) =>
      sample.length > 1
        ? sample.reduce((total, value) => total + (value - centre) ** 2, 0) / (sample.length - 1)
        : 0;
    const varTreated = variance(treated, meanTreated);
    const varControl = variance(control, meanControl);

    // Pooled standard deviation: the gap is expressed in units of how much
    // customers vary anyway, which is what makes it comparable across
    // characteristics measured on completely different scales.
    const pooledSd = Math.sqrt((varTreated + varControl) / 2);
    const smd = pooledSd > 0 ? (meanTreated - meanControl) / pooledSd : 0;

    const stderr = Math.sqrt(varTreated / treated.length + varControl / control.length);
    const z = stderr > 0 ? (meanTreated - meanControl) / stderr : 0;
    const pValue = 2 * (1 - normCdf(Math.abs(z)));

    return {
      feature: name,
      meanTreated,
      meanControl,
      stdMeanDiff: smd,
      absStdMeanDiff: Math.abs(smd),
      pValue,
      balanced: Math.abs(smd) < 0.1,
    };
  });
}

// ------------------------------------------------------------------ qini

/**
 * The Qini curve: extra sales earned at each depth down the ranked list.
 *
 * At each point the curve is
 *
 *     Q(k) = responders_treated(k) - responders_control(k) * n_treated(k) / n_control(k)
 *
 * The rescaling in the second term is load-bearing and the most common thing
 * to get wrong. Without it you are comparing two differently sized counts and
 * the curve measures group size as much as it measures effect.
 */
export function qiniCurve(outcome, treatment, score, seed = 0) {
  const n = outcome.length;
  const order = orderByScore(score, seed);

  const fraction = new Float64Array(n + 1);
  const qini = new Float64Array(n + 1);

  let nTreated = 0;
  let nControl = 0;
  let yTreated = 0;
  let yControl = 0;

  for (let k = 0; k < n; k += 1) {
    const i = order[k];
    const w = treatment[i];
    const y = outcome[i];
    if (w === 1) {
      nTreated += 1;
      yTreated += y;
    } else {
      nControl += 1;
      yControl += y;
    }
    // Before any control customer has appeared the rescaling is undefined; the
    // incremental estimate is zero there by convention.
    const ratio = nControl > 0 ? nTreated / nControl : 0;
    fraction[k + 1] = (k + 1) / n;
    qini[k + 1] = yTreated - yControl * ratio;
  }

  const totalIncremental = qini[n];
  const randomLine = new Float64Array(n + 1);
  for (let k = 0; k <= n; k += 1) randomLine[k] = fraction[k] * totalIncremental;

  const qiniAuc = trapezoid(qini, fraction);
  const randomAuc = trapezoid(randomLine, fraction);
  const coefficient = qiniAuc - randomAuc;

  return {
    fraction,
    qini,
    random: randomLine,
    qiniAuc,
    randomAuc,
    qiniCoefficient: coefficient,
    normalizedQini: Math.abs(randomAuc) > 1e-12 ? coefficient / Math.abs(randomAuc) : NaN,
    totalIncremental,
  };
}

/**
 * Split `n` indices into `parts` groups, matching `numpy.array_split`.
 *
 * NumPy puts the remainder in the EARLIEST groups, one extra each. Getting
 * this backwards shifts every decile boundary by one row, which is invisible
 * in the output and wrong in a way nobody would ever notice.
 */
export function arraySplit(n, parts) {
  const base = Math.floor(n / parts);
  const remainder = n % parts;
  const groups = [];
  let start = 0;
  for (let i = 0; i < parts; i += 1) {
    const size = base + (i < remainder ? 1 : 0);
    groups.push([start, start + size]);
    start += size;
  }
  return groups;
}

/**
 * Measured uplift within bins of predicted uplift.
 *
 * This is the table to show a sceptic. Nothing is modelled: within each bin
 * the treated response rate is simply compared with the control response rate.
 * If the ranking works, the numbers descend down the table.
 */
export function upliftByDecile(outcome, treatment, score, nBins = 10) {
  const order = orderByScore(score, 0);
  const groups = arraySplit(order.length, nBins);

  return groups.map(([start, end], index) => {
    const treated = [];
    const control = [];
    let scoreTotal = 0;
    for (let k = start; k < end; k += 1) {
      const i = order[k];
      scoreTotal += score[i];
      (treatment[i] === 1 ? treated : control).push(outcome[i]);
    }
    const size = end - start;
    const rateTreated = mean(treated);
    const rateControl = mean(control);
    const observed = rateTreated - rateControl;
    const stderr = Math.sqrt(
      (rateTreated * (1 - rateTreated)) / treated.length +
        (rateControl * (1 - rateControl)) / control.length,
    );
    return {
      decile: index + 1,
      n: size,
      nTreated: treated.length,
      nControl: control.length,
      predictedUplift: scoreTotal / size,
      rateTreated,
      rateControl,
      observedUplift: observed,
      stderr,
      ciLow: observed - 1.96 * stderr,
      ciHigh: observed + 1.96 * stderr,
      incrementalConversions: observed * size,
    };
  });
}

// ------------------------------------------------------------------ policy

/**
 * Measured incremental effect within a selected set, scaled to the whole set.
 *
 * Inside the selection the treated and control customers are still randomly
 * assigned — that is the property the trial buys you, and it survives any
 * selection rule that depends only on customer characteristics. So the
 * difference in response between the two arms *within the selection* estimates
 * the effect of contacting everyone in it.
 *
 * Note what this does NOT do: it never reads the model's predicted effect
 * sizes. Those are systematically too large, and a profit figure built on them
 * is fiction.
 */
function incrementalAt(indices, outcome, treatment, spend) {
  if (!indices.length) return { conversions: 0, revenue: 0, upliftRate: 0 };
  const treatedOutcome = [];
  const controlOutcome = [];
  const treatedSpend = [];
  const controlSpend = [];
  for (const i of indices) {
    if (treatment[i] === 1) {
      treatedOutcome.push(outcome[i]);
      treatedSpend.push(spend[i]);
    } else {
      controlOutcome.push(outcome[i]);
      controlSpend.push(spend[i]);
    }
  }
  if (!treatedOutcome.length || !controlOutcome.length) {
    return { conversions: 0, revenue: 0, upliftRate: 0 };
  }
  const upliftRate = mean(treatedOutcome) - mean(controlOutcome);
  const revenueRate = mean(treatedSpend) - mean(controlSpend);
  return {
    conversions: upliftRate * indices.length,
    revenue: revenueRate * indices.length,
    upliftRate,
  };
}

/**
 * Profit at every targeting depth.
 *
 * Walk down the ranked list; at each candidate cut-off, compare what the
 * emailed customers in the selection actually did against what the not-emailed
 * ones did, then subtract the cost of the sends.
 */
export function profitFrontier(
  outcome,
  treatment,
  score,
  spend,
  { costPerContact = 0.1, marginRate = 0.3, nPoints = 40, seed = 0 } = {},
) {
  const n = outcome.length;
  const order = orderByScore(score, seed);
  const rows = [];

  for (let point = 1; point <= nPoints; point += 1) {
    const fraction = point / nPoints;
    const take = roundHalfToEven(fraction * n);
    const indices = order.slice(0, take);
    const { conversions, revenue, upliftRate } = incrementalAt(
      indices,
      outcome,
      treatment,
      spend,
    );
    const cost = take * costPerContact;
    rows.push({
      fraction,
      nTargeted: take,
      incrementalConversions: conversions,
      incrementalRevenue: revenue,
      contactCost: cost,
      incrementalProfit: revenue * marginRate - cost,
      profitPerContact: take ? (revenue * marginRate - cost) / take : 0,
      observedUplift: upliftRate,
    });
  }
  return rows;
}

/**
 * Profit from contacting the same number of people chosen at random.
 *
 * The right comparison, and the one most write-ups omit. Picking at random has
 * no ranking to exploit, so the hundredth person is on average as responsive
 * as the first and profit grows in a straight line. The area between that line
 * and the model's curve is the entire value of knowing who to email.
 */
export function randomTargetingProfit(
  outcome,
  treatment,
  spend,
  fraction,
  { costPerContact = 0.1, marginRate = 0.3 } = {},
) {
  const all = Array.from({ length: outcome.length }, (_, i) => i);
  const { revenue } = incrementalAt(all, outcome, treatment, spend);
  const revenueRate = revenue / outcome.length;
  const nTargeted = roundHalfToEven(fraction * outcome.length);
  return revenueRate * nTargeted * marginRate - nTargeted * costPerContact;
}

// ------------------------------------------------------------------ checks

/**
 * Resample the campaign to get an honest range on the Qini coefficient.
 *
 * Resampling happens WITHIN each arm, keeping both arm sizes exactly as the
 * campaign had them. Resample the whole file instead and the split between the
 * arms drifts — 51% emailed in one draw, 49% in the next — and that wobble has
 * nothing to do with what is being measured but widens every interval built
 * on it.
 */
export function bootstrapQini(outcome, treatment, score, { nBoot = 200, seed = 0 } = {}) {
  const rng = makeRng(seed + 7);
  const treatedIdx = [];
  const controlIdx = [];
  for (let i = 0; i < treatment.length; i += 1) {
    (treatment[i] === 1 ? treatedIdx : controlIdx).push(i);
  }

  const point = qiniCurve(outcome, treatment, score, seed).qiniCoefficient;
  const draws = [];

  for (let b = 0; b < nBoot; b += 1) {
    const picked = new Int32Array(outcome.length);
    let cursor = 0;
    for (const pool of [treatedIdx, controlIdx]) {
      for (let k = 0; k < pool.length; k += 1) {
        picked[cursor] = pool[Math.floor(rng() * pool.length)];
        cursor += 1;
      }
    }
    const y = new Float64Array(picked.length);
    const w = new Int32Array(picked.length);
    const s = new Float64Array(picked.length);
    for (let i = 0; i < picked.length; i += 1) {
      y[i] = outcome[picked[i]];
      w[i] = treatment[picked[i]];
      s[i] = score[picked[i]];
    }
    draws.push(qiniCurve(y, w, s, b).qiniCoefficient);
  }

  draws.sort((a, b) => a - b);
  const quantile = (q) => draws[Math.min(draws.length - 1, Math.floor(q * draws.length))];
  const shareBeating = draws.filter((value) => value > 0).length / draws.length;

  return {
    qiniCoefficient: point,
    ciLow: quantile(0.025),
    ciHigh: quantile(0.975),
    shareBeatingRandom: shareBeating,
    pValueOneSided: 1 - shareBeating,
    beatsRandom: quantile(0.025) > 0,
    nBootstrap: nBoot,
  };
}

/**
 * The shuffle test: score the ranking against campaigns where nothing is there.
 *
 * Keep every customer and every purchase, then permute who got the email. The
 * link between treatment and outcome has now been severed by hand, so a
 * trustworthy pipeline should score near zero. It never scores exactly zero,
 * because with enough rows and columns *some* split always looks good by luck
 * — and that luck is precisely the bar the real result has to clear.
 *
 * This is the check most analyses skip, and it costs one permutation.
 */
export function placeboTest(outcome, treatment, score, { nReplicates = 20, seed = 0 } = {}) {
  const rng = makeRng(seed + 13);
  const real = qiniCurve(outcome, treatment, score, seed).qiniCoefficient;
  const shuffled = [];

  for (let r = 0; r < nReplicates; r += 1) {
    const permuted = Int32Array.from(treatment);
    // Fisher-Yates. Permuting preserves the number treated exactly, which
    // matters: re-randomising each row independently would also change the
    // arm sizes and confound the comparison.
    for (let i = permuted.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = permuted[i];
      permuted[i] = permuted[j];
      permuted[j] = tmp;
    }
    shuffled.push(qiniCurve(outcome, permuted, score, r).qiniCoefficient);
  }

  const meanQini = mean(shuffled);
  const sd = Math.sqrt(
    shuffled.reduce((total, value) => total + (value - meanQini) ** 2, 0) /
      Math.max(1, shuffled.length - 1),
  );
  const exceeding = shuffled.filter((value) => value >= real).length / shuffled.length;

  return {
    nReplicates,
    realQini: real,
    meanQini,
    sdQini: sd,
    zScore: sd > 0 ? (real - meanQini) / sd : 0,
    sharePlacebosExceedingReal: exceeding,
    passes: exceeding <= 0.1 && (sd > 0 ? (real - meanQini) / sd : 0) >= 1.64,
  };
}

/**
 * What the model says to leave alone, and what the data says about them.
 *
 * The second half is the part that matters. A negative prediction is a
 * hypothesis, not a finding: take the flagged customers, go back to the
 * randomised data, and measure them as a group. That costs nothing and it is
 * the only thing that settles the question.
 */
export function sleepingDogReport(
  outcome,
  treatment,
  score,
  spend,
  { costPerContact = 0.1, marginRate = 0.3 } = {},
) {
  const flagged = [];
  for (let i = 0; i < score.length; i += 1) if (score[i] < 0) flagged.push(i);

  if (!flagged.length) {
    return { nFlagged: 0, shareFlagged: 0 };
  }

  const { conversions, revenue, upliftRate } = incrementalAt(flagged, outcome, treatment, spend);
  const treated = flagged.filter((i) => treatment[i] === 1);
  const control = flagged.filter((i) => treatment[i] === 0);

  const rateT = mean(treated.map((i) => outcome[i]));
  const rateC = mean(control.map((i) => outcome[i]));
  const stderr = Math.sqrt(
    (rateT * (1 - rateT)) / treated.length + (rateC * (1 - rateC)) / control.length,
  );

  const cost = flagged.length * costPerContact;
  return {
    nFlagged: flagged.length,
    shareFlagged: flagged.length / score.length,
    meanPredictedUplift: mean(flagged.map((i) => score[i])),
    measuredUplift: upliftRate,
    measuredUpliftStderr: stderr,
    measuredUpliftCiLow: upliftRate - 1.96 * stderr,
    measuredUpliftCiHigh: upliftRate + 1.96 * stderr,
    incrementalConversionsIfTreated: conversions,
    incrementalRevenueIfTreated: revenue,
    costIfTreated: cost,
    profitIfTreated: revenue * marginRate - cost,
  };
}

/**
 * What a campaign of this size could and could not have detected.
 *
 * Run on every upload, whether or not the result was significant, because the
 * most common misreading of a null result is "it does not work" when the
 * honest statement is "this test could never have seen an effect that small".
 */
export function designDiagnostics(naive) {
  const perArm = Math.min(naive.nTreated, naive.nControl);
  const baseline = naive.rateControl;
  if (!(baseline > 0 && baseline < 1) || perArm < 2) return null;

  const mde = minimumDetectableEffect({ baselineRate: baseline, nPerArm: perArm });
  const detected = Math.abs(naive.absoluteDifference) >= mde.absoluteMde;
  const forTwenty = requiredSampleSize({ baselineRate: baseline, relativeMde: 0.2 });

  return {
    perArm,
    baseline,
    absoluteMde: mde.absoluteMde,
    relativeMde: mde.relativeMde,
    observedEffectIsAboveMde: detected,
    nPerArmForTwentyPercent: forTwenty.nPerArm,
    expectedConversionsPerArm: perArm * baseline,
  };
}

// ------------------------------------------------------------------ driver

/**
 * Run everything that the supplied columns allow, and say what was skipped.
 *
 * Structured so a campaign with no model scores still gets a real answer —
 * "did it work", balance, and what the design could detect — rather than an
 * error. Most people arriving with a CSV will not have an uplift score column,
 * and telling them to go away is not a useful tool.
 */
export function analyse({
  outcome,
  treatment,
  spend,
  score,
  covariates = [],
  costPerContact = 0.1,
  marginRate = 0.3,
  nBoot = 200,
  nPlacebo = 20,
}) {
  const result = { n: outcome.length, skipped: [] };

  result.naive = naiveComparison(outcome, treatment);
  result.design = designDiagnostics(result.naive);

  if (covariates.length) {
    result.balance = covariateBalance(covariates, treatment);
    result.maxAbsSmd = Math.max(...result.balance.map((row) => row.absStdMeanDiff));
  } else {
    result.skipped.push(
      "Covariate balance — no customer characteristics were mapped, so there was nothing to check the randomisation against.",
    );
  }

  if (!score) {
    result.skipped.push(
      "Everything about targeting — no uplift score column was mapped. Score your customers with a model elsewhere, add the scores as a column, and re-upload to get the ranking evaluated.",
    );
    return result;
  }

  const money = spend ?? outcome;
  result.usedOutcomeAsSpend = !spend;

  result.qini = qiniCurve(outcome, treatment, score);
  result.deciles = upliftByDecile(outcome, treatment, score);
  result.frontier = profitFrontier(outcome, treatment, score, money, {
    costPerContact,
    marginRate,
  });
  result.bootstrap = bootstrapQini(outcome, treatment, score, { nBoot });
  result.placebo = placeboTest(outcome, treatment, score, { nReplicates: nPlacebo });
  result.sleepingDogs = sleepingDogReport(outcome, treatment, score, money, {
    costPerContact,
    marginRate,
  });

  return result;
}
