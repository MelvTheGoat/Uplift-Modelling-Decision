/** The simple comparison: did the people who got the email buy more? */

import { el, prose, note, details, table } from "../dom.js";
import { count, per100, tinyP, num } from "../format.js";

const ROW_LABELS = {
  hillstrom_mens_conversion: "Men's email → purchases",
  hillstrom_womens_conversion: "Women's email → purchases",
  hillstrom_mens_visit: "Men's email → site visits",
  hillstrom_womens_visit: "Women's email → site visits",
};

export function render(data) {
  const naive = data.naive;
  const mens = naive.hillstrom_mens_conversion;
  const womens = naive.hillstrom_womens_conversion;
  const visits = naive.hillstrom_mens_visit;
  const confounding = naive.confounding;

  // Only the campaign entries have these columns; `naive.json` also carries two
  // teaching demos run on invented data.
  const campaigns = Object.entries(naive).filter(([key]) => key.startsWith("hillstrom_"));

  return [
    el("h1", { text: "Did the email work?" }),
    el("p", {
      class: "lede",
      text: "The easy question, answered first, before anything clever happens.",
    }),

    prose(`
This page does one thing: compare the group that got an email with the group
that did not. No models, no machine learning. Just two numbers and the gap
between them.

That gap is trustworthy for one reason, and it is worth being clear about it:
**people were split into the two groups by coin flip.** Nobody chose who got an
email. So the two groups are alike in every respect — age, past spending, how
recently they shopped, things nobody even recorded — apart from the email
itself. Any difference that shows up has only one candidate explanation left.
    `),

    details(
      "What would go wrong without the coin flip?",
      `
Suppose the marketing team had sent the email to its most loyal customers
instead. They buy more. The email would look like a triumph, and you would have
learned nothing — you cannot separate “the email worked” from “we picked people
who were going to buy anyway”.

We can put a number on how bad that gets, because the study runs exactly this
experiment on invented customers where the true answer is known. With the coin
flip, the simple before-and-after gap lands at
${per100(confounding.naive_gap_randomised)} per 100 against a true effect of
${per100(confounding.true_ate)} — near enough exact. Let the marketing team
pick who gets the email instead, change nothing else, and the same calculation
reports ${per100(confounding.naive_gap_confounded)} per 100.

That is the true effect **overstated by a factor of
${num(confounding.overstatement_factor)}**, and nothing in the output looks
wrong. It is the single most common way campaign measurement goes wrong, no
amount of statistical sophistication afterwards can repair it, and the coin
flip has to happen before the campaign rather than after.
      `,
    ),

    el("h2", { text: "Every outcome, per 100 people" }),
    table({
      columns: [
        "Campaign",
        "Got the email",
        "Got nothing",
        "Difference",
        "As low as",
        "As high as",
        "Relative lift",
        "Fluke probability",
        "People emailed",
      ],
      numeric: [1, 2, 3, 4, 5, 6, 7, 8],
      rows: campaigns.map(([key, result]) => [
        ROW_LABELS[key] ?? key.replace("hillstrom_", "").replace(/_/g, " "),
        per100(result.rate_treated, 2, false),
        per100(result.rate_control, 2, false),
        per100(result.absolute_difference),
        per100(result.ci_low),
        per100(result.ci_high),
        `+${Math.round(result.relative_lift * 100)}%`,
        tinyP(result.p_value),
        count(result.n_treated),
      ]),
    }),
    el("p", {
      class: "control-help",
      text:
        "All rates are per 100 people, so 1.25 means 1.25 purchases for every " +
        "100 customers. The fluke column is the probability of seeing a gap this " +
        "large if the email did nothing at all.",
    }),

    el("h2", { text: "How to read that" }),
    prose(`
**The men’s email roughly doubled purchases** — from
${per100(mens.rate_control, 2, false)} per 100 to
${per100(mens.rate_treated, 2, false)} per 100, a lift of
${Math.round(mens.relative_lift * 100)}%. The fluke probability is
${tinyP(mens.p_value)}, a number small enough that writing it out in full would
need thirteen zeros. This one is real.

**The women’s email also worked, but less** —
${num(mens.absolute_difference / womens.absolute_difference)} times smaller an
effect, and its honest range is wide enough that you would not want to plan
around the exact size.

**Site visits moved much more than purchases.** The men’s email lifted visits
by ${per100(visits.absolute_difference)} per 100, against
${per100(mens.absolute_difference)} per 100 for purchases — about
${num(visits.absolute_difference / mens.absolute_difference, 0)} times easier to
move.
    `),

    note(
      "warn",
      `
**And that last point is a trap.** Visits are easier to measure, easier to
move, and go up and to the right faster — which makes them the tempting thing
to optimise. But a visit does not pay for the email. Tune a campaign to
maximise visits and you will reliably find the people who click and never buy,
and your dashboard will look excellent.

Every headline figure on this site is purchases, for that reason. The visit
numbers are here only so the gap between the two is visible.
      `,
    ),

    details(
      "Why does the relative lift look so big when the absolute numbers are tiny?",
      `
Because the starting point is tiny. Going from
${per100(mens.rate_control, 2, false)} purchases per 100 to
${per100(mens.rate_treated, 2, false)} per 100 is a
${Math.round(mens.relative_lift * 100)}% increase, which sounds enormous, and
in business terms it is. But in absolute terms you have moved
${per100(mens.absolute_difference)} purchases per 100 people, and that is the
number that decides whether the campaign pays.

Both framings are honest. Quoting only the relative one is how a small effect
gets sold as a transformation, so this site shows both side by side everywhere.
      `,
    ),

    details(
      "What is the honest range actually telling me?",
      `
It is the range the true effect is probably in, given that we measured a sample
rather than the whole world. For the men’s email it runs from
${per100(mens.ci_low)} to ${per100(mens.ci_high)} purchases per 100.

The important habit is checking whether it includes zero. This one does not —
every value in the range is positive, so “the email does nothing” is not a
story the data supports.

Later pages have ranges that **do** include zero, and there the honest answer
changes to “we cannot tell”. Watch for that: it is the difference between a
finding and a hunch, and it is only one column apart on the page.
      `,
    ),
  ];
}
