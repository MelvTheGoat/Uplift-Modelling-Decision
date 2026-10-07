/**
 * The words.
 *
 * Everything a visitor reads that is longer than a sentence lives here rather
 * than inline in the pages. Two reasons: the wording is the part most likely
 * to change after someone non-technical reads it, and keeping it together
 * makes it obvious when a page has started leaning on jargon it never
 * explained.
 *
 * House style:
 *
 * - No term appears before the sentence that defines it.
 * - Numbers get a unit and a comparison. "$3,066" means nothing; "$3,066,
 *   against $1,674 for picking names at random" means something.
 * - Where the honest answer is "we don't know", say that, in those words.
 */

/** Sidebar definitions. Short, and each one earns its place on some page. */
export const GLOSSARY = {
  Uplift:
    "The change the email causes in one person. Not “did they buy?” but “did they buy <em>because of</em> the email?”. Someone who would have bought anyway has an uplift of zero, however much they spend.",
  "Treatment and control":
    "Treatment is the group that got the email. Control is the group that deliberately did not. They were split by coin flip, which is what lets us blame any difference between them on the email.",
  "Targeting score":
    "One number for “is this ranking any good?”. Walk down the model’s list, best first, counting the extra sales earned at each step against emailing the same number of people at random. Bigger is better; zero means the ranking is worth no more than a shuffle.",
  "Honest range":
    "The range the true answer is probably in. If it includes zero, the data cannot rule out “this does nothing”. Statisticians call it a confidence interval.",
  "Fluke probability":
    "The chance of seeing a result this good if the thing being measured did not exist. Small is good. Below 0.05 is the usual bar, which is a convention rather than a law of nature. Statisticians call it a p-value.",
  "Sleeping dogs":
    "People the email puts <em>off</em> — they would have bought if left alone. Emailing them costs twice: the send, and the sale you talked them out of. An ordinary “who will buy?” model cannot see them, because they do buy.",
  "Out of sample":
    "Tested on data the model never saw while it was learning. Scores measured on a model’s own training data flatter it badly.",
  "Shuffle test":
    "Scramble who got the email, keep everything else, and re-run. There is nothing left to find, so a good score here means the pipeline is reporting patterns that are not there. Statisticians call it a placebo test.",
};

/** Under the title on the landing page. */
export const STRAPLINE =
  "Who should get the campaign email — and is picking them worth the trouble?";

/** The problem, for someone who has never heard of uplift modelling. */
export const THE_PROBLEM = `
A retailer sends a promotional email to two thirds of its customer list and
leaves the rest alone. The emailed group buys more. So far, so ordinary.

The tempting next step is to build a model that predicts who will buy, and
email those people. That is the wrong model, and it is wrong in a way that
looks like success. It finds your best customers — who were going to buy
regardless. You spend the budget taking credit.

The question worth asking is different: **who buys *because* of the email?**
Those are two different groups of people, and only the second is worth paying
for.

This site works through that question on a real randomised campaign, and is
honest about where the answer runs out.
`;

/** Why there is no accuracy or AUC number anywhere on the site. */
export const NO_ACCURACY = `
You will not find an accuracy or AUC figure anywhere here, and that is on
purpose rather than an oversight.

Those measure how well a model predicts **who buys**. A model that perfectly
ranks people by **how much the email changed them** will score badly on them,
because the people the email moves most are not the people most likely to buy.
A high accuracy score would be evidence that something had gone wrong.

What gets measured instead is the ranking: take the people the model puts at
the top, and check in the real randomised data whether the email actually did
more for them. That check cannot be gamed by predicting purchases well.
`;

/** Why models predict far bigger effects than the data shows. */
export const WHY_OVERSTATED = `
Every model here works by subtracting one prediction from another. Each of
those predictions carries its own error, and when you subtract them the errors
do not cancel — they add.

So the spread of predicted effects comes out wider than the spread of real
ones, and the people at the top of the list get assigned effects far larger
than anything the data can support. The tail of that too-wide spread also
crosses below zero, which is where the phantom sleeping dogs come from.

The practical rule, worth carrying to any project of this shape: use the
ranking to decide **who goes first**, and use counted outcomes to decide **how
much that is worth**. Never quote a model's own predicted effect sizes as a
forecast.
`;

/** What the three sleeping-dog outcomes mean, used on that page. */
export const DOGS_ARE_REAL = `
Real, and well documented — but usually a smaller group than models claim, and
usually in specific situations rather than scattered through a list.

The clearest real cases are retention campaigns. Phoning a quiet customer to
remind them about a subscription they had forgotten they were paying for
reliably produces cancellations that would not otherwise have happened. There
the effect is large, repeatable, and nobody needs a model to find it.

What this study shows is subtler: on this campaign, at this size, the models
flagged nearly a fifth of the list and were wrong about the direction. Both
things are true — the phenomenon is real, and this particular detection of it
was not.
`;

/** Ways a customer can be put off by a marketing email. */
export const HOW_HARM_HAPPENS = `
Several ordinary ways, none of them exotic:

- They were going to buy at full price next week. The email reminds them a
  discount exists, so they wait for the next one.
- They had forgotten they were on the list. The email reminds them, and they
  unsubscribe.
- They are an infrequent but loyal buyer who finds being marketed at
  irritating, and the email is a small nudge towards a competitor.

None of this is unusual. What is unusual is **measuring** it, because you only
see it by holding back a control group and comparing. If you email everyone,
the sleeping dogs are invisible — they just look like people who did not buy
this time.
`;

/** The data provenance line, shown in the sidebar and on the about page. */
export const DATA_SOURCE = `
The figures come from the MineThatData E-Mail Analytics And Data Mining
Challenge dataset, published by Kevin Hillstrom in 2008. 64,000 customers who
had bought something in the previous twelve months were split at random three
ways: a third got an email featuring men's merchandise, a third got one
featuring women's merchandise, and a third got nothing. Visits, purchases and
spend were recorded over the following two weeks.

It is a real randomised experiment, it is freely available, and it has been
analysed enough times that a wrong answer here would be noticed. That last
property is the reason to use it rather than something more exciting.
`;
