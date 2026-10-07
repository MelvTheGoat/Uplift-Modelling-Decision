/** Four models, scored honestly, and the difference between the two questions. */

import { el, prose, note, details, table, figure, segmented, replaceChildren } from "../dom.js";
import { barsWithIntervals, barsWithPredictions, chartColours } from "../charts.js";
import { NO_ACCURACY, WHY_OVERSTATED } from "../copy.js";
import {
  MODEL_BLURBS,
  count,
  modelLabel,
  num,
  pValue,
  per100,
  percent,
  studyLabel,
  STUDY_BLURBS,
} from "../format.js";

export function render(data) {
  const ranking = data.naive.ranking;
  const studies = data.meta.studies;

  // Page-local state. Kept in closures rather than the URL: these are "show me
  // a different slice" controls, not addressable locations, and putting them in
  // the hash would mean every slider nudge adds a browser-history entry.
  let study = studies.includes("mens_conversion") ? "mens_conversion" : studies[0];
  let model = null;

  const scoreboard = el("div");
  const deciles = el("div");

  const drawScoreboard = () => replaceChildren(scoreboard, scoreboardSection(data, study));
  const drawDeciles = () => replaceChildren(deciles, decileSection(data, study, model));

  const picker = () =>
    segmented({
      label: "Which campaign?",
      options: studies.map((key) => ({ value: key, label: studyLabel(key) })),
      value: study,
      onChange: (next) => {
        study = next;
        model = null; // the previous pick may not exist for this campaign
        replaceChildren(pickerHost, picker());
        drawScoreboard();
        drawDeciles();
      },
    });
  const pickerHost = el("div", {}, [picker()]);

  drawScoreboard();
  drawDeciles();

  const responseShare =
    ranking.true_uplift_captured_by_response_model / ranking.true_uplift_captured_by_oracle;
  const upliftShare =
    ranking.true_uplift_captured_by_uplift_model / ranking.true_uplift_captured_by_oracle;
  const dogRatio =
    ranking.sleeping_dogs_contacted_by_response_model /
    ranking.sleeping_dogs_contacted_by_uplift_model;

  return [
    el("h1", { text: "Can we pick who to email?" }),
    el("p", { class: "lede", text: "Four ways to try, and an honest score for each." }),

    el("h2", { text: "First: the two questions are not the same" }),
    prose(`
Worth settling before looking at any model, because getting it wrong costs real
money and feels like success while it happens.

**Question A: who is likely to buy?** An ordinary prediction problem. Plenty of
tools answer it well.

**Question B: who buys *because of the email*?** A different question with a
different answer, and the one that decides where the budget goes.

On invented customers — where we can check, because we know the truth — the
study ranks everybody both ways and emails the top
${percent(ranking.top_fraction)}. Here is what each approach collects:
    `),

    table({
      columns: ["How we picked", "Extra sales collected", "Sleeping dogs emailed"],
      numeric: [1, 2],
      rows: [
        ["Perfect knowledge (impossible)", num(ranking.true_uplift_captured_by_oracle, 0), "—"],
        [
          "**Question B** — who the email moves",
          num(ranking.true_uplift_captured_by_uplift_model, 0),
          count(ranking.sleeping_dogs_contacted_by_uplift_model),
        ],
        [
          "**Question A** — who is likely to buy",
          num(ranking.true_uplift_captured_by_response_model, 0),
          count(ranking.sleeping_dogs_contacted_by_response_model),
        ],
        ["Names out of a hat", num(ranking.true_uplift_captured_by_random, 0), "—"],
      ],
    }),

    prose(`
Asking question A gets you ${percent(responseShare)} of what perfect knowledge
would. Asking question B gets you ${percent(upliftShare)}. Both beat a hat, so
both will look like a win on any dashboard.

The second column is the part that does not show up on a dashboard. The
question-A model emails **${num(dogRatio, 0)} times more sleeping dogs** —
people the email actively puts off. It emails them *because* they look likely to
buy, which they are, right up until the email arrives.
    `),

    details("Hold on — how can emailing someone make them less likely to buy?", HOW_HARM),

    el("h2", { text: "Now: the real campaign" }),
    pickerHost,
    scoreboard,

    el("h2", { text: "The check a sceptic should ask for" }),
    prose(`
A score is one number and easy to distrust. This is the same thing in a form
you can argue with.

Take the model's ranking and cut it into ten equal groups, best tenth first. In
each group, compare what the emailed people actually did against what the
not-emailed people actually did. **No model is involved in that comparison.**
It is counting.

If the ranking is any good, the bars slope downwards from left to right.
    `),
    deciles,
  ];
}

const HOW_HARM = `
Several ordinary ways, none of them exotic:

- They were going to buy at full price next week. The email reminds them a
  discount exists, so they wait for the next one.
- They had forgotten they were on the list. The email reminds them, and they
  unsubscribe.
- They are an infrequent but loyal buyer who finds being marketed at
  irritating, and the email is a small nudge towards a competitor.

None of this is unusual. What is unusual is **measuring** it, because you only
see it by holding back a control group and comparing.
`;

function scoreboardSection(data, study) {
  const result = data.hillstrom[study];
  const leaderboard = result.leaderboard;
  const colours = chartColours();

  const chart = barsWithIntervals(
    leaderboard
      .slice()
      .sort((a, b) => b.qini_coefficient - a.qini_coefficient)
      .map((row) => ({
        label: modelLabel(row.model),
        value: row.qini_coefficient,
        low: row.qini_ci_low,
        high: row.qini_ci_high,
        extra: [
          ["Fluke probability", num(row.p_value_one_sided, 3)],
          ["Beats random?", row.beats_random ? "yes" : "not proven"],
        ],
      })),
    { xTitle: "Extra sales won by this ranking, versus picking names at random" },
  );

  const beating = leaderboard.filter((row) => row.beats_random);
  const best = leaderboard.find((row) => row.model === result.best_model);

  return [
    el("p", { class: "control-help", text: STUDY_BLURBS[study] ?? "" }),
    figure({
      legend: [
        { label: "Score", colour: colours.s1 },
        { label: "Honest range", colour: colours.inkSoft, shape: "line" },
        { label: "Zero — no better than a shuffle", colour: colours.ink, shape: "dash" },
      ],
      chart,
      caption:
        "The bar is the score. The line through it is the range the true score " +
        "is probably in. A bar whose line crosses the dashed zero mark has not " +
        "shown that it beats a random shuffle.",
    }),

    beating.length === 0
      ? note(
          "bad",
          `
**None of the four models proved it beats a random shuffle on this campaign.**
The closest was ${modelLabel(result.best_model)}, at a fluke probability of
${pValue(best.p_value_one_sided)}. Close is not the same as there.
          `,
        )
      : note(
          "good",
          `
**${beating.length} of ${leaderboard.length} models beat a random shuffle on
this campaign** — their honest range sits entirely above zero. The best is
${modelLabel(result.best_model)}, at ${pValue(best.p_value_one_sided)}.
          `,
        ),

    details(
      "Show the full scoreboard",
      table({
        columns: [
          "Model",
          "Score",
          "Honest range",
          "Beats random?",
          "Fluke probability",
          "Best tenth, per 100",
          "Worst tenth, per 100",
          "Flagged as sleeping dogs",
          "Seconds to fit",
        ],
        numeric: [1, 2, 4, 5, 6, 7, 8],
        rows: leaderboard.map((row) => [
          modelLabel(row.model),
          num(row.qini_coefficient),
          `${num(row.qini_ci_low)} to ${num(row.qini_ci_high)}`,
          row.beats_random ? "yes" : "not proven",
          num(row.p_value_one_sided, 3),
          per100(row.top_decile_uplift),
          per100(row.bottom_decile_uplift),
          percent(row.share_predicted_negative, 1),
          num(row.seconds),
        ]),
      }),
    ),

    details(
      "What the four models actually do",
      Object.entries(MODEL_BLURBS)
        .map(([name, blurb]) => `**${modelLabel(name)}** — ${blurb}`)
        .join("\n\n"),
    ),

    details("Where is the accuracy score?", NO_ACCURACY),
  ];
}

function decileSection(data, study, requestedModel) {
  const result = data.hillstrom[study];
  const available = data.deciles[study] ?? {};
  const models = result.leaderboard.map((row) => row.model).filter((name) => available[name]);

  if (models.length === 0) {
    return [note("plain", "No decile breakdown was generated for this campaign.")];
  }

  let current = models.includes(requestedModel)
    ? requestedModel
    : models.includes(result.best_model)
      ? result.best_model
      : models[0];

  // The control and the body are separate hosts so that changing the model
  // replaces only the body. Rebuilding the whole section from inside its own
  // callback would nest a fresh copy inside the old one on every click.
  const controlHost = el("div");
  const body = el("div");

  const drawControl = () =>
    replaceChildren(controlHost, [
      segmented({
        label: "Which model's ranking?",
        options: models.map((name) => ({ value: name, label: modelLabel(name) })),
        value: current,
        onChange: (next) => {
          current = next;
          drawControl();
          replaceChildren(body, decileBody(available[current], current));
        },
      }),
    ]);

  drawControl();
  replaceChildren(body, decileBody(available[current], current));
  return [controlHost, body];
}

function decileBody(rows, model) {
  const colours = chartColours();

  const plotRows = rows.map((row) => ({
    label: String(Math.round(row.decile)),
    value: row.observed_uplift * 100,
    low: row.ci_low * 100,
    high: row.ci_high * 100,
    predicted: row.predicted_uplift * 100,
    people: row.n,
    tooltipTitle: `Tenth ${Math.round(row.decile)}`,
    extra: [["People in this tenth", count(row.n)]],
  }));

  const top = plotRows[0];
  const bottom = plotRows[plotRows.length - 1];
  const overshoot = top.predicted / (Math.abs(top.value) < 1e-9 ? 1e-9 : top.value);

  return [
    figure({
      legend: [
        { label: "Email helped this group", colour: colours.s1 },
        { label: "Email hurt this group", colour: colours.danger },
        { label: "What the model predicted", colour: colours.s2 },
        { label: "Honest range", colour: colours.inkSoft, shape: "line" },
      ],
      chart: barsWithPredictions(plotRows, {
        yTitle: "Extra purchases per 100 people",
        xTitle: "Model's ranking, best tenth first",
      }),
      caption:
        "Each bar is one tenth of the list. Its height is what actually " +
        "happened; the orange dot is what the model predicted would happen.",
    }),

    prose(`
For **${modelLabel(model)}** on this campaign:

- Its best tenth (${count(top.people)} people) actually gained
  **${top.value >= 0 ? "+" : ""}${num(top.value, 2)} purchases per 100**,
  somewhere between ${num(top.low, 2)} and ${num(top.high, 2)}.
- Its worst tenth gained **${bottom.value >= 0 ? "+" : ""}${num(bottom.value, 2)}
  per 100**.
- The model predicted **${num(top.predicted, 2)}** for that best tenth —
  ${num(overshoot)} times what the data actually shows.
    `),

    details(
      "Show the ten groups as a table",
      table({
        columns: [
          "Tenth",
          "People",
          "Model predicted, per 100",
          "Actually measured, per 100",
          "As low as",
          "As high as",
        ],
        numeric: [0, 1, 2, 3, 4, 5],
        rows: rows.map((row) => [
          Math.round(row.decile),
          count(row.n),
          per100(row.predicted_uplift),
          per100(row.observed_uplift),
          per100(row.ci_low),
          per100(row.ci_high),
        ]),
      }),
    ),

    details(
      "The model predicted far more than happened. Is it broken?",
      `
Not broken, but not to be taken at face value either. This gap is the normal
behaviour of these methods, and understanding it is the difference between a
model being *useful* and being *right*.

${WHY_OVERSTATED}

Which is why the decile table exists. Here, taking the model's own predictions
as a forecast would have overstated the best group by ${num(overshoot)} times.

Everything on the [What is it worth?](#/money) page follows that rule: the money
there comes from counting what happened in the randomised data, never from the
model's predictions.
      `,
    ),
  ];
}
