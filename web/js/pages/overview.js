/** Landing page: the problem, the answer, and the one number that decides it. */

import { el, prose, note, details, metrics } from "../dom.js";
import { STRAPLINE, THE_PROBLEM } from "../copy.js";
import { count, money, per100, percent, moneyDelta, tinyP, num } from "../format.js";

export function render(data) {
  const main = data.hillstrom.mens_conversion;
  const summary = main.summary;
  const policy = main.policy;
  const best = main.leaderboard.find((row) => row.model === main.best_model);
  const naive = data.naive.hillstrom_mens_conversion;
  const placebo = main.robustness.placebo;

  const capCost = policy.optimal_profit - policy.recommended_profit;
  const targetingValue = policy.profit_vs_random;

  // Revenue per purchase, backed out of the campaign rather than assumed.
  const revenuePerPurchase = summary.spend_treated / summary.outcome_rate_treated;
  const breakeven = policy.cost_per_contact / (revenuePerPurchase * policy.margin_rate);

  return [
    el("h1", { text: "Who should get the email?" }),
    el("p", { class: "lede", text: STRAPLINE }),
    prose(THE_PROBLEM),

    el("h2", { text: "The short answer" }),
    prose(`
Three findings, in the order they should change what you do.

**1. The email works.** People who got it bought at
${per100(summary.outcome_rate_treated, 2, false)} per 100, against
${per100(summary.outcome_rate_control, 2, false)} per 100 for people who got
nothing. That is roughly double, and it is not a fluke — on a test of
${count(summary.n)} customers, a gap that size turns up by chance about once in
a trillion.

**2. Picking who to email is not proven to help.** The best of four models
earned ${num(best.qini_coefficient)} extra sales over emailing the same number
of people at random. But the honest range on that figure runs from
${num(best.qini_ci_low)} to ${num(best.qini_ci_high)} — it includes zero. We
cannot rule out that the ranking is worth nothing.

**3. The budget cap costs more than the targeting earns.** This is the finding
worth acting on, and nobody asked for it.
    `),

    el("h3", { text: "The thing nobody asked about" }),
    metrics([
      {
        label: "Best targeting can earn you",
        value: money(targetingValue),
        note: "versus emailing the same number at random",
      },
      {
        label: "The budget cap costs you",
        value: money(capCost),
        tone: "bad",
        note: `${num(capCost / Math.max(targetingValue, 1e-9))}× larger`,
      },
      {
        label: "So the first move is",
        value: "Raise the cap",
        note: "not “buy a model”",
      },
    ]),

    note(
      "info",
      `
Read those two numbers together. Getting the targeting right is worth about
${money(targetingValue)}. Being allowed to email
${percent(policy.optimal_fraction)} of the list instead of
${percent(policy.budget_cap_fraction)} is worth about ${money(capCost)}.

The campaign stays profitable well past the point where the budget stops it. A
sophisticated model that helps you spend a cap set too low is solving the
second-biggest problem.
      `,
    ),

    details(
      "Why is a cap costing money? Shouldn’t spending less save money?",
      `
Only if the spending were wasteful, and here it is not. Each email costs
${money(policy.cost_per_contact, 2)} and the company keeps
${percent(policy.margin_rate)} of any sales it causes. The average purchase in
this campaign was worth ${money(revenuePerPurchase, 2)}, so one email pays for
itself as long as it causes about ${per100(breakeven, 2, false)} extra
purchases per 100 people — a very low bar, which most of the list clears.

So profit keeps climbing as you email more people, right up to
${percent(policy.optimal_fraction)} of the list. Stopping at
${percent(policy.budget_cap_fraction)} leaves profitable customers
uncontacted. The cap is not frugality; it is a self-imposed ceiling on profit.
      `,
    ),

    el("h2", { text: "The part most write-ups leave out" }),
    prose(`
We ran the whole pipeline again on deliberately broken data: same customers,
same purchases, but with “who got the email” **shuffled at random**. There is
nothing to find in that data, by construction. A trustworthy pipeline should
score close to zero.

It scored ${num(placebo.mean_qini)} on average, and
${percent(placebo.share_placebos_exceeding_real)} of the shuffled runs scored
*higher* than the real campaign did (${num(placebo.real_qini)}).

That does not mean the models are broken. It means the real signal here is
small enough that the noise in our measurement is the same size as the thing we
are trying to measure. Reporting “this model wins” from this data would be
reporting a coin flip.
    `),

    note(
      "warn",
      `
**The recommendation is therefore: raise the budget cap, email more of the
list, and run a proper test before buying into any targeting model.** The
[Plan your own test](#/plan-a-test) page sizes that test for you.
      `,
    ),

    details(
      "Is this a failure? It sounds like a lot of work for “we don’t know”.",
      `
It is a result, and a useful one. The alternative — picking the best-looking
model, quoting its score, and shipping it — would have produced a confident
recommendation off a measurement no better than a shuffle.

Two things support that this is the data’s limit rather than a broken pipeline:

- The **women’s email** campaign, run through exactly the same code, passes
  every check: a clear positive score, an honest range that excludes zero, and
  a shuffle test it comfortably survives. The machinery works.
- The simple before-and-after comparison on the men’s email is overwhelmingly
  significant (fluke probability ${tinyP(naive.p_value)}). Measuring *whether*
  the email works is easy with ${count(summary.n)} customers. Measuring *who it
  works best on* is a far harder question, and this many customers is not
  enough for it.

Knowing which questions your data can answer is worth more than an answer it
cannot support.
      `,
    ),

    el("h2", { text: "Where to go next" }),
    el("ul", { class: "next-links" }, [
      link("#/did-it-work", "Did the email work?", "The simple before-and-after comparison, for every campaign."),
      link("#/targeting", "Can we pick who to email?", "Four models, scored honestly, and what each one is doing."),
      link("#/money", "What is it worth?", "Move the email cost and the margin, and watch the recommendation change."),
      link("#/trust", "Can we trust it?", "The checks that broke, and the one campaign that passed everything."),
      link("#/sleeping-dogs", "Who should we leave alone?", "Customers a model says the email puts off — and whether it is right."),
      link("#/plan-a-test", "Plan your own test", "A working calculator: how many customers you need, and how to need fewer."),
      link("#/about", "How this was built", "The code, the honest limitations, and how to run it yourself."),
    ]),

    prose(`
The headline campaign throughout is the men’s email. It reached
${count(summary.n)} customers, split about evenly between email and no email,
and comes out ${moneyDelta(policy.profit_vs_treat_all)} than simply emailing
everyone — which is the comparison the budget cap forces, and the one the cap
loses.
    `),
  ];
}

function link(href, title, blurb) {
  return el("li", {}, [
    el("a", { href }, [el("strong", { text: title }), el("span", { text: blurb })]),
  ]);
}
