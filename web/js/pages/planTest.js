/**
 * A working sample-size calculator.
 *
 * This is the one page where the browser computes rather than reads. It is
 * here because it is the most portable thing on the site: the finding about
 * this particular campaign is interesting, but "how many customers do I need?"
 * is a question a visitor can take straight back to their own work.
 *
 * The maths comes from `stats.js`, which is a line-for-line port of
 * `src/experiment.py` and is checked against it by `tests/test_web_parity.py`.
 */

import {
  el,
  prose,
  note,
  details,
  table,
  metrics,
  numberField,
  segmented,
  replaceChildren,
} from "../dom.js";
import {
  requiredSampleSize,
  minimumDetectableEffect,
  designValidation,
} from "../stats.js";
import { count, money, num, per100, percent } from "../format.js";

export function render(data) {
  const experiment = data.experiment;
  const main = data.hillstrom.mens_conversion;
  const baselineDefault = experiment.baseline_rate * 100;

  // Form state, in the units the fields show: rates as a percentage, effects
  // as a percentage of baseline. Converted at the edge, so the maths only ever
  // sees proportions.
  const state = {
    baselinePct: Number(baselineDefault.toFixed(3)),
    relativeLiftPct: 20,
    available: 20000,
    weekly: 20000,
  };

  const forward = el("div");
  const reverse = el("div");

  const redraw = () => {
    replaceChildren(forward, forwardResult(state));
    replaceChildren(reverse, reverseResult(state));
  };

  const fields = el("div", { class: "field-row" }, [
    numberField({
      label: "Your conversion rate (%)",
      min: 0.01,
      max: 90,
      step: 0.01,
      value: state.baselinePct,
      help: "What share of untouched customers buy anyway. This campaign's was 0.573%.",
      onInput: (value) => {
        state.baselinePct = value;
        redraw();
      },
    }),
    numberField({
      label: "Lift you want to detect (%)",
      min: 1,
      max: 500,
      step: 1,
      value: state.relativeLiftPct,
      help: "As a share of your rate. 20 means “a 20% improvement on what we get now”.",
      onInput: (value) => {
        state.relativeLiftPct = value;
        redraw();
      },
    }),
    numberField({
      label: "Customers you can test on, per group",
      min: 100,
      max: 5000000,
      step: 100,
      value: state.available,
      help: "Used for the second calculation: what this many people can actually detect.",
      onInput: (value) => {
        state.available = value;
        redraw();
      },
    }),
  ]);

  redraw();

  return [
    el("h1", { text: "Plan your own test" }),
    el("p", {
      class: "lede",
      text: "A real calculator, not a screenshot. Put your own numbers in.",
    }),

    prose(`
The most useful lesson from this study is not about this campaign. It is that
**measuring whether something works and measuring who it works best on are
different-sized problems**, and the second needs far more data than people
expect.

This page lets you check that for your own situation before you run anything.
    `),

    fields,

    el("h2", { text: "1. How many customers do you need?" }),
    prose(`
The question: you want to prove an effect of a given size is real. How many
people must be in each group?

Two conventions are baked in, and both are worth knowing rather than inheriting:

- **Fluke tolerance of 5%.** If there is really no effect, you accept a 1-in-20
  chance of wrongly announcing one.
- **Power of 80%.** If the effect really is there at the size you specified, you
  accept a 1-in-5 chance of missing it anyway.

That second one surprises people. A perfectly standard test design fails to
spot a real effect one time in five.
    `),
    forward,

    el("h2", { text: "2. What can the customers you have actually detect?" }),
    prose(`
The more useful direction, and the one almost nobody runs first. You usually
cannot choose your sample size — you have the list you have. So the honest
question is what that list can see.
    `),
    reverse,

    el("h2", { text: "The number that decides whether to buy a model" }),
    validationSection(experiment, main),

    el("h2", { text: "Getting a sharper answer for free" }),
    cupedSection(experiment),

    details(
      "Why does the formula use two different variances?",
      `
This is the detail that separates a correct power calculation from the version
in a lot of online calculators, and getting it wrong makes your test look
cheaper than it is.

The formula has two terms. The first asks “how extreme must the result be before
I call it real?” — and that question is asked **under the assumption that there
is no effect**, where both groups share one single rate. So it uses the
**pooled** rate across both arms.

The second asks “how likely am I to reach that bar if the effect is real?” — and
that question is asked **under the assumption that the effect exists**, where
the two groups have genuinely different rates. So it uses the two **separate**
rates.

Use the pooled variance for both and you understate the sample size needed.
Your test comes out looking affordable, you run it, and it quietly lacks the
power to find what you are looking for.
      `,
    ),

    details(
      "Why is the reverse direction solved by trial and error?",
      `
Because the forward formula cannot be cleanly rearranged, and the alternative is
worse than it looks.

You could derive a separate closed-form expression for the reverse direction.
Two separately derived formulas then sit in the codebase, and the first time
someone corrects one of them, they silently disagree — a bug nobody notices,
because both answers look plausible.

Instead the reverse direction repeatedly guesses an effect size, asks the
forward function how many customers that would need, and narrows the bracket.
Two hundred halvings pins it down past any precision that matters, and the two
directions are guaranteed consistent because there is only one formula.

Slower, and worth it.
      `,
    ),
  ];
}

function forwardResult(state) {
  const baseline = state.baselinePct / 100;
  const relative = state.relativeLiftPct / 100;

  if (!(baseline > 0 && baseline < 1) || !(relative > 0)) {
    return [note("warn", "Enter a conversion rate between 0 and 100%, and a positive lift.")];
  }

  const result = requiredSampleSize({ baselineRate: baseline, relativeMde: relative });
  const thin = result.expectedConversionsPerArm < 30;

  return [
    metrics([
      {
        label: "Customers needed per group",
        value: count(result.nPerArm),
      },
      {
        label: "Total customers needed",
        value: count(result.nTotal),
        note: "both groups together",
      },
      {
        label: "Purchases expected per group",
        value: count(result.expectedConversionsPerArm),
        tone: thin ? "bad" : null,
        note: thin ? "too few — the maths is unreliable here" : "healthy",
      },
    ]),
    prose(`
To detect a lift from **${percent(baseline, 3)} to
${percent(result.treatmentRate, 3)}** — a ${percent(relative)} improvement —
you need **${count(result.nPerArm)} customers in each group**,
${count(result.nTotal)} in total.
    `),
    thin
      ? note(
          "warn",
          `
**Careful with this one.** The design expects only
${count(result.expectedConversionsPerArm)} purchases per group, and the formula
behind it assumes enough purchases for a bell curve to be a fair approximation.
Below roughly thirty it starts to lie, usually in the optimistic direction.

The sample size shown is a lower bound at best. Either plan for a bigger test,
or measure something that happens more often — site visits rather than
purchases, say — and accept that you are then answering an easier question.
          `,
        )
      : null,
  ];
}

function reverseResult(state) {
  const baseline = state.baselinePct / 100;
  const available = Math.round(state.available);

  if (!(baseline > 0 && baseline < 1) || !(available > 0)) {
    return [note("warn", "Enter a positive number of customers per group.")];
  }

  const result = minimumDetectableEffect({ baselineRate: baseline, nPerArm: available });

  return [
    metrics([
      {
        label: "Smallest effect you could spot",
        value: `${per100(result.absoluteMde, 3, false)} per 100`,
      },
      {
        label: "As a share of your rate",
        value: percent(result.relativeMde),
        tone: result.relativeMde > 0.5 ? "bad" : result.relativeMde > 0.2 ? "warn" : "good",
      },
      {
        label: "Rate you would need to see",
        value: percent(result.detectableTreatmentRate, 3),
      },
    ]),
    prose(`
With ${count(available)} customers per group and a base rate of
${percent(baseline, 3)}, the smallest effect you could reliably detect is
**${percent(result.relativeMde)}** — anything smaller than that will come back
as “no significant difference” whether or not it is real.
    `),
    result.relativeMde > 0.3
      ? note(
          "warn",
          `
**That is a big effect to require.** A test that can only see improvements of
${percent(result.relativeMde)} or more will report “nothing happened” for any
genuine improvement below that — and most real marketing improvements are below
it.

If a test like that comes back negative, the honest conclusion is “we could not
tell”, not “it does not work”. Those get conflated constantly, and the second
one kills initiatives that were working.
          `,
        )
      : note(
          "good",
          `That is a reasonably sensitive test. Effects of
${percent(result.relativeMde)} or more should show up clearly.`,
        ),
  ];
}

function validationSection(experiment, main) {
  const design = experiment.validation_design;
  const policy = main.policy;

  // Recompute from the shipped numbers so the page and the study cannot drift.
  const live = designValidation({
    baselineRate: experiment.baseline_rate,
    expectedPolicyUplift: design.difference_under_test + design.baseline_for_test - experiment.baseline_rate,
    expectedRandomUplift: design.baseline_for_test - experiment.baseline_rate,
    weeklyVolume: 20000,
  });

  return [
    prose(`
Here is where the two questions pull apart hardest.

Proving the email works at all took tens of thousands of customers and came out
overwhelmingly clear. Proving that **model targeting beats business as usual**
is a different measurement: the thing under test is now the *difference between
two methods*, which is much smaller than either method's effect.

The study sized that test. To detect the edge the model claims over the
incumbent policy:
    `),
    metrics([
      {
        label: "Customers needed per cell",
        value: count(design.n_per_cell),
      },
      {
        label: "Total customers needed",
        value: count(design.n_total),
      },
      {
        label: "Weeks at 20,000 contactable per week",
        value: num(design.weeks_required, 0),
      },
    ]),
    prose(`
That is **${count(design.n_total)} customers** to answer "is the model worth
it?", against the ${count(main.summary.n)} that answered "does the email work?".
Roughly twice as many, to measure something far less dramatic.

The difference under test is only ${per100(design.difference_under_test)}
purchases per 100 — the model's claimed edge. Small differences need big tests.
That is not a flaw in the method; it is arithmetic.
    `),
    live.nTotal === design.n_total
      ? null
      : note(
          "warn",
          `Note: recomputing this design in the browser gives
${count(live.nTotal)} rather than the stored ${count(design.n_total)}. That is a
discrepancy worth investigating rather than ignoring.`,
        ),
    note(
      "info",
      `
**Which is the real recommendation of this whole study.** Before spending on
targeting infrastructure, run the test above. It costs
${count(design.n_total)} customers and ${num(design.weeks_required, 0)} weeks,
and it replaces an argument with a number.

Compare that against what the targeting is estimated to be worth —
${money(policy.profit_vs_random)} — and against what raising the budget cap is
worth: ${money(policy.optimal_profit - policy.recommended_profit)}.
      `,
    ),
  ];
}

function cupedSection(experiment) {
  const cuped = experiment.cuped;
  const rows = Object.entries(cuped);

  return [
    prose(`
One technique worth knowing because it costs nothing and is almost always
available: if you already know something about each customer from **before** the
test started — what they spent last quarter, how often they visited — you can
subtract out that known variation and get a sharper answer from the same number
of people.

No extra customers, no longer run time, no change to the estimate itself. Only
its precision improves. The industry calls it CUPED.

The size of the win is entirely determined by one thing: how strongly your
before-data correlates with the outcome. Specifically, the variance reduction
equals the correlation squared — exactly, not approximately.
    `),
    table({
      columns: [
        "What we measured",
        "Correlation with before-data",
        "Variance removed",
        "Correlation squared",
        "Equivalent to extra customers",
      ],
      numeric: [1, 2, 3, 4],
      rows: rows.map(([name, entry]) => [
        name === "conversion" ? "Purchases (a 0.6% event)" : "Engagement (a continuous metric)",
        num(entry.correlation, 3),
        percent(entry.variance_reduction, 1),
        num(entry.correlation_squared, 3),
        `×${num(entry.effective_sample_multiplier, 2)}`,
      ]),
    }),
    prose(`
Look at the third and fourth columns: identical. That is the theory holding
exactly, which is a good sign the implementation is right.

Now look at the two rows. On the continuous engagement metric the technique
removes ${percent(cuped.engagement.variance_reduction, 0)} of the variance —
worth as much as **${num(cuped.engagement.effective_sample_multiplier, 2)} times
the sample size**, free. On purchases it removes
${percent(cuped.conversion.variance_reduction, 1)}, which is worth essentially
nothing.
    `),
    note(
      "warn",
      `
**So the honest answer on whether to use it is “it depends, and you can check
in advance”.** Rare binary events like purchases are mostly unpredictable noise
at the individual level, so there is little for before-data to explain. Continuous
engagement metrics are much more stable per person, so there is a lot.

Compute the correlation on historical data before building anything. One line of
code tells you whether the technique will pay for itself, and the answer here
ranges from “doubles your test” to “do not bother” depending on which metric you
point it at.
      `,
    ),
    details(
      "What breaks CUPED?",
      `
One thing, and it breaks it badly: **the before-data must be measured strictly
before assignment.**

If the treatment can influence the covariate, subtracting it removes part of the
effect you are trying to measure and biases the estimate towards zero. A
"pre-period" that overlaps the campaign window is the usual way this happens in
practice, and it is easy to do by accident when someone builds the feature table
from a date range rather than from the assignment timestamp.

The symptom is a technique that appears to be working — your error bars shrink —
while quietly shrinking the effect as well. There is a check: CUPED must not move
the estimate, only its precision. If your adjusted and unadjusted effects differ
by much, the covariate is contaminated.
      `,
    ),
  ];
}
