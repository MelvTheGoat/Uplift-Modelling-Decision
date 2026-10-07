/** Customers a model says to leave alone — and whether the data agrees. */

import { el, prose, note, details, table, metrics, segmented, replaceChildren } from "../dom.js";
import { DOGS_ARE_REAL, HOW_HARM_HAPPENS, WHY_OVERSTATED } from "../copy.js";
import { count, missing, money, modelLabel, num, per100, percent, studyLabel } from "../format.js";

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
          replaceChildren(body, section(data, study));
        },
      }),
    ]);

  drawPicker();
  replaceChildren(body, section(data, study));

  return [
    el("h1", { text: "Who should we leave alone?" }),
    el("p", {
      class: "lede",
      text:
        "The group an ordinary model cannot see, and what happened when we " +
        "actually checked it.",
    }),
    prose(`
Most marketing analytics sorts customers into “worth contacting” and “not worth
contacting”. Uplift modelling adds a third box that is easy to miss and
expensive to ignore: **customers the contact makes worse**.

They are called sleeping dogs, from “let sleeping dogs lie”. Emailing one costs
you twice — the price of the send, and a sale you talked them out of.

Why they are invisible to ordinary models is worth sitting with: sleeping dogs
**do buy**. A model predicting who will buy puts them near the top of the list,
correctly. It is only when you compare them against a held-back control group
that you discover they would have bought slightly more had you left them alone.
    `),
    details("How can emailing someone make them less likely to buy?", HOW_HARM_HAPPENS),
    pickerHost,
    body,
    details("Are sleeping dogs real at all, or just a modelling artefact?", DOGS_ARE_REAL),
  ];
}

function section(data, study) {
  const result = data.hillstrom[study];
  const dogs = result.sleeping_dogs;
  const policy = result.policy;

  if (!dogs.n_flagged) {
    return [
      note(
        "info",
        `
**${modelLabel(policy.model)} flagged nobody as a sleeping dog on this
campaign.** Every customer's predicted effect came out positive or zero.

That is a reasonable outcome and worth a moment's thought: this campaign
measured site visits rather than purchases, and almost nothing about receiving
an email makes a person *less* likely to visit a website. Sleeping dogs are far
more plausible on a purchase outcome, where waiting for the next discount is a
real behaviour.
        `,
      ),
    ];
  }

  const measured = dogs.measured_uplift;
  const low = dogs.measured_uplift_ci_low;
  const high = dogs.measured_uplift_ci_high;
  const profitIfTreated = dogs.profit_if_treated;
  const marginShare = dogs.incremental_revenue_if_treated * policy.margin_rate;

  return [
    prose(`
**${modelLabel(policy.model)} flagged ${count(dogs.n_flagged)} customers** —
${percent(dogs.share_flagged, 1)} of the list — as people the email would push
the wrong way. It predicted they would buy
**${per100(dogs.mean_predicted_uplift)} purchases per 100** as a result of being
emailed: a negative number, meaning fewer sales, not more.

So the next step is the only one that matters. These customers were split by
coin flip like everyone else, so some of them got the email and some did not.
We can simply look.
    `),

    metrics([
      {
        label: "Flagged as sleeping dogs",
        value: count(dogs.n_flagged),
        note: `${percent(dogs.share_flagged, 1)} of the list`,
      },
      {
        label: "Model predicted, per 100",
        value: per100(dogs.mean_predicted_uplift),
        tone: "bad",
        note: "negative = the email costs sales",
      },
      {
        label: "Actually measured, per 100",
        value: per100(measured),
        tone: missing(measured) ? null : measured > 0 ? "good" : "bad",
        note: "counted from the randomised data",
      },
    ]),

    verdict(measured, low, high),

    el("h3", { text: "What emailing them actually costs" }),
    table({
      columns: ["", "Amount"],
      numeric: [1],
      rows: [
        [
          `Extra purchases these ${count(dogs.n_flagged)} people make if emailed`,
          num(dogs.incremental_conversions_if_treated),
        ],
        ["Extra revenue that brings in", money(dogs.incremental_revenue_if_treated)],
        [`Your share of it, at ${percent(policy.margin_rate)} margin`, money(marginShare)],
        [
          `Cost of emailing them, at ${money(policy.cost_per_contact, 2)} each`,
          money(-dogs.cost_if_treated),
        ],
        ["**Net**", `**${money(profitIfTreated)}**`],
      ],
    }),

    profitIfTreated < 0
      ? note(
          "info",
          `
**So: unprofitable, not harmful.** Emailing this group drains
${money(Math.abs(profitIfTreated))} — but not because the email damages them.
It is because the sales it causes do not cover the sends.

That distinction changes what you do about it. “Harmful” means suppress them on
principle and keep suppressing them. “Unprofitable at
${money(policy.cost_per_contact, 2)} an email” means the answer flips the moment
your costs or margins move. Push the email cost down on the
[What is it worth?](#/money) page and this group becomes worth contacting.
          `,
        )
      : note(
          "good",
          `**Emailing this group is profitable anyway** — ${money(profitIfTreated)}
net. The flag is not a reason to suppress them.`,
        ),

    details(
      "Why would a model predict harm where the data shows the opposite?",
      `
${WHY_OVERSTATED}

You can see the size of it in the numbers above: predicted
${per100(dogs.mean_predicted_uplift)} per 100, measured ${per100(measured)}. Not
merely the wrong size — the wrong sign.

The rule that follows is worth carrying to any project of this shape: **a
negative prediction is a hypothesis, not a finding.** Take the people a model
flags, go back to the randomised data, and measure them as a group. That
measurement is cheap, needs no modelling, and is the only thing that settles the
question.

Suppressing ${percent(dogs.share_flagged, 1)} of a mailing list on an unchecked
model output is how this goes wrong in production.
      `,
    ),
  ];
}

/** The three possible outcomes of checking a flagged group against the data. */
function verdict(measured, low, high) {
  if (missing(measured)) {
    return note("plain", "Not enough customers in this group to measure the effect.");
  }
  if (low > 0) {
    return note(
      "bad",
      `
**The model got the direction wrong.** It predicted these customers would be
*harmed* by the email. Measured in the randomised data, they gained
**${per100(measured)} purchases per 100**, with an honest range of
${per100(low)} to ${per100(high)} — entirely above zero.

These are not sleeping dogs. The email helped them. It helped them *less* than
it helped other people, which is a different claim with a different consequence.
      `,
    );
  }
  if (high < 0) {
    return note(
      "good",
      `
**The model was right, and this is a real finding.** These customers lost
**${per100(measured)} purchases per 100** when emailed, with a range of
${per100(low)} to ${per100(high)} — entirely below zero. Suppressing them is a
clear and immediate saving.
      `,
    );
  }
  return note(
    "warn",
    `
**Cannot tell.** Measured effect ${per100(measured)} per 100, range
${per100(low)} to ${per100(high)}. The range straddles zero, so the data cannot
distinguish “the email hurt them” from “the email did nothing” from “the email
helped a little”.
    `,
  );
}
