/**
 * The profit page: move the assumptions and watch the recommendation move.
 *
 * The recalculation here is EXACT, not an approximation, and that is worth
 * knowing because it is unusual. The stored frontier records the incremental
 * revenue actually measured at each depth, plus how many people were contacted
 * to get it. Profit at any cost and margin is therefore just
 *
 *     revenue * margin - people * cost
 *
 * which is arithmetic on numbers that were already counted. Nothing is being
 * re-estimated as the slider moves, and the figures reproduce the stored
 * policy numbers to the last decimal when the sliders sit at their defaults.
 */

import {
  el,
  prose,
  note,
  details,
  table,
  figure,
  metrics,
  segmented,
  slider,
  replaceChildren,
} from "../dom.js";
import { lines, heatmap, chartColours } from "../charts.js";
import { count, money, modelLabel, num, percent, studyLabel, STUDY_BLURBS } from "../format.js";

export function render(data) {
  const studies = data.meta.studies;
  let study = studies.includes("mens_conversion") ? "mens_conversion" : studies[0];

  const defaults = data.hillstrom[study].policy;
  let cost = defaults.cost_per_contact;
  let margin = defaults.margin_rate;
  let cap = defaults.budget_cap_fraction;

  const output = el("div");
  const controlsHost = el("div");
  const pickerHost = el("div");

  const redraw = () => replaceChildren(output, results(data, study, { cost, margin, cap }));

  const drawControls = () =>
    replaceChildren(controlsHost, [
      el("div", { class: "controls" }, [
        slider({
          label: "Cost of sending one email",
          min: 0.01,
          max: 0.5,
          step: 0.01,
          value: cost,
          format: (v) => money(v, 2),
          help:
            "Everything one more email costs: the sending platform's fee, creative " +
            "spread over the send, and the slow damage of list fatigue and spam " +
            "complaints. The study's $0.10 is deliberately pessimistic for pure " +
            "email, where the true marginal cost is often under a cent — " +
            "pessimistic because the fatigue cost is real and nobody can put a " +
            "clean number on it. Push it up to see how much of the answer rests " +
            "on that guess.",
          onInput: (value) => {
            cost = value;
            redraw();
          },
        }),
        slider({
          label: "Profit kept per $1 of sales",
          min: 0.05,
          max: 0.8,
          step: 0.05,
          value: margin,
          format: (v) => percent(v),
          help:
            "Of each extra dollar of revenue the campaign causes, the share left " +
            "after cost of goods, shipping and returns. 0.30 is a middling " +
            "general-merchandise retailer; fashion runs higher, electronics much " +
            "lower. At 0.10 the campaign has to work three times harder to break " +
            "even.",
          onInput: (value) => {
            margin = value;
            redraw();
          },
        }),
        slider({
          label: "Budget cap: most of the list you may email",
          min: 0.05,
          max: 1,
          step: 0.05,
          value: cap,
          format: (v) => percent(v),
          help:
            "A limit from outside the model — the budget the campaign was given. " +
            "Set it to 100% to see what the data says with no cap at all. The gap " +
            "between those two answers is the single most useful number here.",
          onInput: (value) => {
            cap = value;
            redraw();
          },
        }),
      ]),
    ]);

  const drawPicker = () =>
    replaceChildren(pickerHost, [
      segmented({
        label: "Which campaign?",
        options: studies.map((key) => ({ value: key, label: studyLabel(key) })),
        value: study,
        onChange: (next) => {
          study = next;
          const policy = data.hillstrom[study].policy;
          cost = policy.cost_per_contact;
          margin = policy.margin_rate;
          cap = policy.budget_cap_fraction;
          drawPicker();
          drawControls();
          redraw();
        },
      }),
      el("p", { class: "control-help", text: STUDY_BLURBS[study] ?? "" }),
    ]);

  drawPicker();
  drawControls();
  redraw();

  return [
    el("h1", { text: "What is it worth?" }),
    el("p", {
      class: "lede",
      text: "Where the profit peaks, and what the budget cap is costing.",
    }),
    pickerHost,
    prose(`
Everything below rests on counting, not forecasting. We take the model's
ranking, walk down it, and at each depth ask: among the people selected so far,
what did the emailed ones actually do compared with the not-emailed ones? That
difference is real revenue that really happened. Then we subtract what the
emails cost.

The model is only used to decide the **order**. It never supplies a number that
ends up in the profit figure — which matters, because the previous page showed
these models overstate the effects they predict several times over. A profit
figure built on those predictions would be fiction.
    `),

    el("h2", { text: "Your assumptions" }),
    el("p", {
      text:
        "These two are business inputs, not statistics. Move them and everything " +
        "below recalculates exactly, because the underlying revenue was already " +
        "measured at every depth.",
    }),
    controlsHost,
    output,
  ];
}

/**
 * Recompute everything at the given assumptions and build the output section.
 *
 * @param {object} data The study bundle.
 * @param {string} study Which campaign.
 * @param {{cost: number, margin: number, cap: number}} assumptions
 */
function results(data, study, { cost, margin, cap }) {
  const result = data.hillstrom[study];
  const policy = result.policy;
  const frontier = result.frontier;
  const colours = chartColours();

  // Average incremental revenue per customer across the whole file. This is
  // what "email this many people, chosen by coin flip" earns, and it is why the
  // random line is straight.
  const full = frontier[frontier.length - 1];
  const revenuePerCustomer = full.incremental_revenue / full.n_targeted;

  const rows = frontier.map((row) => ({
    fraction: row.fraction,
    people: row.n_targeted,
    conversions: row.incremental_conversions,
    revenue: row.incremental_revenue,
    modelProfit: row.incremental_revenue * margin - row.n_targeted * cost,
    randomProfit: revenuePerCustomer * row.n_targeted * margin - row.n_targeted * cost,
  }));

  const peak = rows.reduce((best, row) => (row.modelProfit > best.modelProfit ? row : best));
  const allowed = rows.filter((row) => row.fraction <= cap + 1e-9);
  const chosen = allowed.length
    ? allowed.reduce((best, row) => (row.modelProfit > best.modelProfit ? row : best))
    : null;

  const recommendedFraction = chosen ? chosen.fraction : 0;
  const recommendedProfit = chosen ? chosen.modelProfit : 0;
  const randomAtDepth = chosen ? chosen.randomProfit : 0;
  const treatAllProfit = rows[rows.length - 1].modelProfit;

  const capCost = peak.modelProfit - recommendedProfit;
  const targetingValue = recommendedProfit - randomAtDepth;

  const series = [
    {
      label: "Email the model's top picks",
      colour: colours.s1,
      points: rows.map((row) => ({ x: row.fraction * 100, y: row.modelProfit })),
    },
    {
      label: "Email the same number at random",
      colour: colours.s2,
      points: rows.map((row) => ({ x: row.fraction * 100, y: row.randomProfit })),
    },
  ];

  const grid = (data.cost_sensitivity[study] ?? []).map((row) => ({
    ...row,
    optimal_pct: row.optimal_fraction * 100,
  }));
  const sensitivity = result.robustness.cost_sensitivity;
  const profitCi = result.profit_ci;

  return [
    el("h3", { text: "What that gives you" }),
    metrics([
      {
        label: "Email this share of the list",
        value: percent(recommendedFraction),
        note: "the most profitable depth the cap allows",
      },
      {
        label: "Profit",
        value: money(recommendedProfit),
        tone: recommendedProfit >= 0 ? "good" : "bad",
        note: "measured revenue × margin, minus the sends",
      },
      {
        label: "Worth of the targeting",
        value: money(targetingValue),
        note: "versus the same number picked at random",
      },
      {
        label: "Cost of the cap",
        value: money(capCost),
        tone: capCost > 0 ? "bad" : null,
        note: `profit peaks at ${percent(peak.fraction)}`,
      },
    ]),

    capCost > targetingValue && capCost > 0
      ? note(
          "warn",
          `
**At these assumptions the cap costs
${num(capCost / Math.max(targetingValue, 1e-9))} times more than the targeting
earns** — ${money(capCost)} against ${money(targetingValue)}. Raising the cap
is the bigger lever, and it needs no model at all.
          `,
        )
      : targetingValue > 0
        ? note(
            "good",
            `
**At these assumptions the targeting is the bigger lever** —
${money(targetingValue)} against ${money(capCost)} for the cap. Here the model
is earning its keep.
            `,
          )
        : note(
            "plain",
            "At these assumptions neither lever is worth much: the campaign is " +
              "barely profitable at any depth. Try a lower email cost or a higher margin.",
          ),

    figure({
      legend: [
        { label: "Email the model's top picks", colour: colours.s1, shape: "line" },
        { label: "Email the same number at random", colour: colours.s2, shape: "line" },
        { label: "What we recommend", colour: colours.danger, shape: "dash" },
        { label: "Where profit peaks", colour: colours.inkFaint, shape: "dash" },
      ],
      chart: lines(series, {
        xTitle: "Share of the customer list emailed (%)",
        yTitle: "Profit, after paying for the emails ($)",
        formatY: (v) => money(v, 0),
        formatX: (v) => `${v.toFixed(0)}%`,
        markers: [
          { at: recommendedFraction * 100, label: "recommend", colour: colours.danger },
          { at: peak.fraction * 100, label: "peak", colour: colours.inkFaint },
        ],
      }),
      caption:
        "Left to right: emailing more of the list. The gap between the two lines " +
        "is what the ranking is worth.",
    }),

    table({
      caption: "Four strategies at your current assumptions",
      columns: ["Strategy", "Profit"],
      numeric: [1],
      rows: [
        [
          `Email **${percent(recommendedFraction)}** — the model's picks, under your cap`,
          money(recommendedProfit),
        ],
        [`Email **${percent(recommendedFraction)}** — picked at random`, money(randomAtDepth)],
        ["Email **everybody**", money(treatAllProfit)],
        [
          `Email **${percent(peak.fraction)}** — the model's picks, cap lifted`,
          money(peak.modelProfit),
        ],
      ],
    }),

    treatAllProfit > recommendedProfit
      ? note(
          "info",
          `
Note the third row. **Emailing everybody beats the capped, targeted campaign**
by ${money(treatAllProfit - recommendedProfit)} here.

Targeting only pays once you are allowed to go deep enough that the ranking has
room to matter, or once the email gets expensive enough that leaving people out
is the point. Try pushing the email cost up and watch the rows swap.
          `,
        )
      : null,

    details(
      "Show every depth as a table",
      table({
        columns: [
          "Share emailed",
          "People",
          "Extra purchases caused",
          "Extra revenue",
          "Email cost",
          "Profit",
          "Profit if picked at random",
        ],
        numeric: [0, 1, 2, 3, 4, 5, 6],
        rows: rows.map((row) => [
          percent(row.fraction, 1),
          count(row.people),
          num(row.conversions),
          money(row.revenue),
          money(row.people * cost),
          money(row.modelProfit),
          money(row.randomProfit),
        ]),
      }),
    ),

    el("h2", { text: "How sure are we?" }),
    prose(`
The study re-ran the whole selection ${count(profitCi.n_bootstrap)} times on
resampled versions of the campaign, at the original assumptions
(${money(policy.cost_per_contact, 2)} per email,
${percent(policy.margin_rate)} margin, ${percent(policy.budget_cap_fraction)}
cap). Profit came out at **${money(profitCi.profit)}**, and across those runs it
ranged from **${money(profitCi.ci_low)} to ${money(profitCi.ci_high)}**.
${percent(profitCi.prob_loses_money, 1)} of the runs lost money.
    `),
    profitCi.ci_low <= 0
      ? note(
          "warn",
          "That range includes zero, so losing money on this campaign is not " +
            "ruled out. The central estimate is still positive and still the best " +
            "guess — but it is a best guess, not a floor.",
        )
      : note(
          "good",
          "The whole range sits above zero: every resampled version of this " +
            "campaign made money. That is about as reassuring as this kind of " +
            "evidence gets.",
        ),
    el("p", {
      class: "control-help",
      text:
        "Those bounds are fixed at the study's original assumptions — the " +
        "resampling is far too slow to redo in a browser while you move a slider. " +
        "Every figure above it does follow your sliders.",
    }),

    el("h2", { text: "Does the answer survive a different business?" }),
    prose(`
Sliders show one scenario at a time. This grid shows all of them at once: every
combination of email cost and margin the study tested, coloured and labelled by
how deep you should email. Dark means “email nearly everybody”; pale means
“email hardly anybody”.
    `),
    grid.length
      ? figure({
          chart: heatmap({
            rows: grid,
            xKey: "cost_per_contact",
            yKey: "margin_rate",
            valueKey: "optimal_pct",
            xFormat: (v) => money(v, 2),
            yFormat: (v) => percent(v),
            valueFormat: (v) => `${v.toFixed(0)}%`,
            tooltip: (row) => [
              ["Email cost", money(row.cost_per_contact, 2)],
              ["Margin kept", percent(row.margin_rate)],
              ["Email this share", percent(row.optimal_fraction, 1)],
              ["Profit", money(row.optimal_profit)],
              ["Beats emailing everyone?", row.beats_treat_all ? "yes" : "no"],
            ],
          }),
          caption:
            "Rows are the margin you keep; columns are what one email costs. " +
            "Each cell is the share of the list worth emailing in that world.",
        })
      : note("plain", "No cost-sensitivity grid was generated for this campaign."),

    prose(`
Across those ${grid.length} scenarios the right depth runs from
**${percent(sensitivity.min_optimal_fraction)} to
${percent(sensitivity.max_optimal_fraction)}** of the list, with a typical
answer of ${percent(sensitivity.median_optimal_fraction)}.
${percent(sensitivity.share_profitable)} of the scenarios are profitable at all.

That spread is the honest headline. “Email 30% of the list” is not a fact about
your customers — it is a fact about your email cost and your margin, and it
moves across almost the whole range when those move. Anyone quoting a single
targeting depth without quoting the cost and margin behind it has skipped the
step that matters.
    `),

    details(
      "Why does the profit line turn down at the right-hand end?",
      `
Because you run out of worthwhile people. Going down the ranked list, each
additional person is less affected by the email than the last. Early on the
extra sales comfortably cover the sends. Eventually you reach people the email
barely moves, and after that people it moves the wrong way — and every one of
them still costs you a send.

The peak is where the next email stops paying for itself. Everything to the
right of it is spending money to make less money.
      `,
    ),

    details(
      "Why is the random line straight?",
      `
Because picking at random has no ranking to exploit: the hundredth person you
pick at random is, on average, exactly as responsive as the first. Each one adds
the same average profit, so the total grows in a straight line.

That is what makes it the right comparison. The model's line curves — steep at
the start, flattening, eventually falling — and the area between the two curves
is the entire value of knowing who to email.

On this campaign the average customer brings in
${money(revenuePerCustomer, 3)} of extra revenue, which is where the slope of
the straight line comes from.
      `,
    ),

    details(
      `Whose ranking is this, exactly?`,
      `
Yes — the ranking driving every number on this page comes from
**${modelLabel(policy.model)}**, which scored best on the
[previous page](#/targeting). On the men's campaign that model did not prove it
beats a random shuffle, so treat the gap between the two lines as the best
available estimate rather than a demonstrated fact.

The [Can we trust it?](#/trust) page is where that gets taken apart properly.
      `,
    ),
  ];
}
