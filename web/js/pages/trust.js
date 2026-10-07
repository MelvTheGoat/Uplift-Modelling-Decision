/** The checks that try to break the finding, including the ones that succeeded. */

import {
  el,
  prose,
  note,
  details,
  table,
  figure,
  metrics,
  segmented,
  pill,
  replaceChildren,
} from "../dom.js";
import { distributionStrip, chartColours } from "../charts.js";
import { count, money, modelLabel, num, percent, per100, studyLabel, STUDY_BLURBS } from "../format.js";

export function render(data) {
  const studies = data.meta.studies;
  let study = studies.includes("mens_conversion") ? "mens_conversion" : studies[0];

  const body = el("div");
  const pickerHost = el("div");

  const drawPicker = () =>
    replaceChildren(pickerHost, [
      segmented({
        label: "Which campaign?",
        options: studies.map((key) => ({ value: key, label: studyLabel(key) })),
        value: study,
        onChange: (next) => {
          study = next;
          drawPicker();
          replaceChildren(body, checks(data, study));
        },
      }),
      el("p", { class: "control-help", text: STUDY_BLURBS[study] ?? "" }),
    ]);

  drawPicker();
  replaceChildren(body, checks(data, study));

  return [
    el("h1", { text: "Can we trust it?" }),
    el("p", {
      class: "lede",
      text: "Four attempts to break the finding. Not all of them failed.",
    }),
    prose(`
A model that has been scored once, on data it was tuned against, with no
attempt made to break it, is a model with no evidence behind it. This page is
the attempt to break it.

Three of the four checks below are standard. The first is the one people skip,
and it does the most work.
    `),
    pickerHost,
    body,
    el("h2", { text: "So is the pipeline broken?" }),
    comparison(data),
  ];
}

function checks(data, study) {
  const result = data.hillstrom[study];
  const robust = result.robustness;
  const placebo = robust.placebo;
  const balance = robust.balance;
  const seeds = robust.seed_stability;
  const boot = robust.bootstrap_qini;
  const balanceRows = data.balance[study] ?? [];
  const colours = chartColours();

  return [
    robust.overall_pass
      ? note(
          "good",
          `**${studyLabel(study)}: every check holds up.** The finding on this
campaign survived everything thrown at it.`,
        )
      : note(
          "warn",
          `**${studyLabel(study)}: at least one check does not hold up.** The
details are below. This does not mean the code is wrong — it means the evidence
will not carry the weight of a confident recommendation.`,
        ),

    // ---------------------------------------------------------- placebo
    el("h2", { text: "1. The shuffle test" }),
    prose(`
**What it does.** Keep every customer and every purchase exactly as they are,
then shuffle the “who got an email” column at random. Nothing in the data links
the email to anything now, because we just severed that link ourselves. Run the
entire pipeline on it and see what score comes out.

A score near zero means the pipeline reports nothing when there is nothing to
report. A high score means it manufactures findings.

**What happened.** ${count(placebo.n_replicates)} shuffles, using
${modelLabel(placebo.model)}:
    `),
    figure({
      legend: [
        { label: "Where the shuffled runs landed", colour: colours.inkFaint },
        { label: "The real campaign", colour: colours.s1, shape: "line" },
        { label: "Zero", colour: colours.ink, shape: "dash" },
      ],
      chart: distributionStrip({
        band: {
          low: placebo.mean_qini - placebo.sd_qini,
          high: placebo.mean_qini + placebo.sd_qini,
          centre: placebo.mean_qini,
        },
        reference: placebo.real_qini,
        labels: { band: "Shuffled campaigns", reference: "The real campaign" },
        formatX: (v) => num(v),
      }),
      caption:
        "The grey band is the range the shuffled campaigns landed in. You want " +
        "the blue line well clear of it.",
    }),
    metrics([
      { label: "Real campaign", value: num(placebo.real_qini) },
      { label: "Average shuffle", value: num(placebo.mean_qini) },
      {
        label: "Shuffles that beat the real run",
        value: percent(placebo.share_placebos_exceeding_real),
        tone: placebo.share_placebos_exceeding_real > 0.1 ? "bad" : "good",
        note: "should be close to none",
      },
    ]),
    placebo.passes
      ? note(
          "good",
          `**Holds up.** The real campaign scored ${num(placebo.z_score)} standard
deviations above the shuffles, and only
${percent(placebo.share_placebos_exceeding_real)} of shuffles beat it. The score
is measuring something that is actually there.`,
        )
      : note(
          "bad",
          `**Does not hold up.** The real campaign scored only
${num(placebo.z_score)} standard deviations above the shuffles, and
${percent(placebo.share_placebos_exceeding_real)} of the shuffles beat it
outright. Shuffled data contains nothing by construction, so a real score this
easy to match by accident cannot be relied on.`,
        ),
    details(
      "Why does shuffled data score above zero at all?",
      `
Because the pipeline is allowed to pick the best-looking split it can find, and
in any finite dataset there is always *some* split that looks good by luck. With
tens of thousands of customers and eleven columns to slice them by, the number
of candidate patterns is enormous — and the best of those will always look like
a finding.

On shuffled data that luck is all there is, which is why the average shuffle
scored ${num(placebo.mean_qini)} rather than 0. The useful question is never “is
the score above zero?” but “is the score above what luck alone produces?”

This is the check that would have caught the mistake in this study, and the one
most write-ups never run. It costs one extra line of code: permute the treatment
column, keep everything else, re-run.
      `,
    ),

    // ---------------------------------------------------------- balance
    el("h2", { text: "2. Was the coin flip actually fair?" }),
    prose(`
**What it does.** Everything here rests on the two groups being alike apart
from the email. That was supposed to be guaranteed by random assignment, but
“supposed to be” is not a measurement. So we check each of the
${count(balance.n_features)} customer characteristics on file and ask whether
the emailed and not-emailed groups actually look the same on it.

The yardstick is a **standardised difference**: the gap between the two groups,
measured in units of how much customers vary anyway. Below 0.10 is conventionally
negligible; 0.25 and above is a problem.
    `),
    metrics([
      {
        label: "Largest difference found",
        value: num(balance.max_abs_smd, 4),
        note: "against a 0.10 threshold",
      },
      {
        label: "Characteristics out of line",
        value: `${Math.round(balance["n_above_0.10"])} of ${Math.round(balance.n_features)}`,
      },
    ]),
    balance.passes
      ? note(
          "good",
          `**Holds up.** The worst imbalance across all
${count(balance.n_features)} characteristics was ${num(balance.max_abs_smd, 4)}
— about ${num(0.1 / balance.max_abs_smd, 0)} times smaller than the threshold
where anyone would start worrying. The randomisation did its job, so differences
in outcomes can fairly be put down to the email.`,
        )
      : note(
          "bad",
          `**Does not hold up.** At least one characteristic differs between the
groups by more than chance comfortably explains. Treat every effect on this site
as potentially contaminated.`,
        ),
    balanceRows.length
      ? details(
          "Show every characteristic",
          table({
            columns: [
              "Characteristic",
              "Average, emailed",
              "Average, not emailed",
              "Standardised difference",
              "Fluke probability",
              "Verdict",
            ],
            numeric: [1, 2, 3, 4],
            rows: balanceRows.map((row) => [
              row.feature,
              num(row.mean_treated, 4),
              num(row.mean_control, 4),
              num(row.std_mean_diff, 4),
              num(row.p_value, 3),
              pill(Boolean(row.balanced), ["balanced", "out of line"]),
            ]),
          }),
        )
      : null,
    details(
      "If assignment was random, isn’t this check pointless?",
      `
In a textbook, yes. In practice it is the check that catches what actually goes
wrong: a send that failed silently for one region, a suppression list applied to
one arm only, a data export that dropped rows unevenly, someone helpfully
“fixing” an under-filled cell. None of those announce themselves, and all of
them are invisible in the final numbers.

It costs almost nothing to run and it is the difference between believing the
randomisation and checking it. Run it on every experiment, including the ones
you set up yourself.
      `,
    ),

    // ---------------------------------------------------------- seeds
    el("h2", { text: "3. Would a different random seed change the answer?" }),
    prose(`
**What it does.** These models involve randomness — which rows land in which
fold, which columns each tree considers. None of that is part of your business
problem. If changing the seed changes the recommendation, then the
recommendation is partly a property of the seed, and nobody should be making
decisions on that.

**What happened.** ${count(seeds.n_seeds)} seeds, everything else identical:
    `),
    metrics([
      {
        label: "Score range across seeds",
        value: `${num(seeds.qini_min)} to ${num(seeds.qini_max)}`,
        note: "same data, same model, only the seed changed",
      },
      {
        label: "Overlap in who gets emailed",
        value: percent(seeds.mean_selection_overlap),
        tone: seeds.mean_selection_overlap >= 0.7 ? "good" : "bad",
        note: `at the top ${percent(seeds.top_fraction)} of the list`,
      },
      {
        label: "Profit range",
        value: money(seeds.profit_mean),
        note: `± ${money(seeds.profit_sd)}`,
      },
    ]),
    seeds.mean_selection_overlap >= 0.7 && seeds.sign_consistent
      ? note(
          "good",
          `**Holds up.** The seeds agree on
${percent(seeds.mean_selection_overlap)} of who to email and never disagree on
the direction of the effect.`,
        )
      : note(
          "bad",
          `
**Does not hold up.** Two runs that differ only in their random seed agree on
just **${percent(seeds.mean_selection_overlap)}** of the people they pick. More
than half the mailing list is decided by the seed.

Worse, the score itself ranges from ${num(seeds.qini_min)} to
${num(seeds.qini_max)} — it crosses zero. One seed says the ranking is mildly
useful and another says it is worthless, on identical data.

The profit figure is much steadier (${money(seeds.profit_mean)} ±
${money(seeds.profit_sd)}), and that is not the consolation it looks like: it
means profit at this depth is driven by **how many** people you email, not
**which** ones. Another way of saying the ranking is not doing much.
          `,
        ),
    details(
      "Could you not just average over seeds and be done with it?",
      `
You can, and for a production system you should — fit several models with
different seeds and average their rankings. It genuinely reduces this noise.

But it does not rescue the finding here, for a reason worth being precise about.
Averaging gives you a more **stable** ranking. It does not give you more
**evidence** that the ranking is right. If the underlying signal is too small
for this much data to pin down, a stable estimate of it is a stable estimate of
almost nothing — and it now looks more trustworthy than it is, which is worse
than the noisy version that advertised its own weakness.

The fix for a signal this small is more data, or a bigger effect to find. Not
better smoothing.
      `,
    ),

    // ---------------------------------------------------------- bootstrap
    el("h2", { text: "4. Is the score distinguishable from zero?" }),
    prose(`
**What it does.** Re-draw the campaign ${count(boot.n_bootstrap)} times by
sampling customers with replacement — separately within the emailed and
not-emailed groups, so the design is preserved — and score the ranking in each.
The spread of those scores is how much the one real score could have differed
purely by which customers happened to be in the file.
    `),
    metrics([
      { label: "Score", value: num(boot.qini_coefficient) },
      { label: "Honest range", value: `${num(boot.ci_low)} to ${num(boot.ci_high)}` },
      {
        label: "Fluke probability",
        value: num(boot.p_value_one_sided, 3),
        tone: boot.beats_random ? "good" : "bad",
      },
    ]),
    boot.beats_random
      ? note(
          "good",
          `**Holds up.** The range runs from ${num(boot.ci_low)} to
${num(boot.ci_high)} and stays above zero throughout.
${percent(boot.share_beating_random, 1)} of the resampled campaigns beat a
random shuffle.`,
        )
      : note(
          "bad",
          `**Does not hold up.** The range runs from ${num(boot.ci_low)} to
${num(boot.ci_high)}, which includes zero.
${percent(boot.share_beating_random, 1)} of resampled campaigns beat a random
shuffle — short of the 95% conventionally required. The ranking might be worth
something; this data cannot show it.`,
        ),
    details(
      "Why resample inside each group rather than across the whole file?",
      `
Because resampling the whole file lets the split between the two groups drift.
One draw comes out 51% emailed, the next 49%, and that wobble has nothing to do
with what is being measured — but it adds spread to the result, and every honest
range built on it comes out too wide.

The campaign's split was fixed by design, so the resampling should treat it as
fixed too: draw within the emailed group and within the not-emailed group
separately, keeping both sizes exactly as they were. It is a two-line change,
and it is the difference between a range that means what it says and one that is
quietly too pessimistic.
      `,
    ),
  ];
}

function comparison(data) {
  const rows = data.meta.studies.map((key) => {
    const entry = data.hillstrom[key];
    const best = entry.leaderboard.find((row) => row.model === entry.best_model);
    return {
      key,
      label: studyLabel(key),
      model: modelLabel(entry.best_model),
      best,
      robust: entry.robustness,
    };
  });

  const passing = rows.filter((row) => row.robust.overall_pass).map((row) => row.label);

  return [
    table({
      columns: [
        "Campaign",
        "Best model",
        "Score",
        "Honest range",
        "Beats random?",
        "Survives the shuffle test?",
        "Coin flip fair?",
        "Everything holds up?",
      ],
      numeric: [2, 3],
      rows: rows.map((row) => [
        row.label,
        row.model,
        num(row.best.qini_coefficient),
        `${num(row.best.qini_ci_low)} to ${num(row.best.qini_ci_high)}`,
        row.best.beats_random ? "yes" : "not proven",
        pill(Boolean(row.robust.placebo.passes)),
        pill(Boolean(row.robust.balance.passes)),
        pill(Boolean(row.robust.overall_pass)),
      ]),
    }),
    prose(`
No — and this table is the evidence. The same code, the same checks, run on
three campaigns: ${passing.length ? passing.join(" and ") : "none"} came through
clean.

That is what separates “this data cannot answer the question” from “this code is
broken”. A broken pipeline fails everywhere. This one produces a clean,
positive, shuffle-surviving result on one campaign and an honest shrug on
another, which is exactly what working machinery looks like when the signal
differs between datasets.

Had every campaign failed, the right conclusion would have been to stop and
audit the code.
    `),
  ];
}
